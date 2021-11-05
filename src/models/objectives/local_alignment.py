import logging
import math
from dataclasses import dataclass
from typing import Union, Tuple, Optional, Collection

import torch
from torch import nn
import torch.nn.functional as F

from common.config_utils import prepare_config
from models.components.fc import SequenceMLP
from models.components.utils import AttentionMask

log = logging.getLogger(__name__)

@dataclass
class LocalAlignmentObjectiveConfig:
    objective_type: str


@dataclass
class LocalMseLossConfig(LocalAlignmentObjectiveConfig):
    objective_type: str = 'local_mse'


class LocalMseLoss(nn.Module):
    def __init__(self, config: LocalMseLossConfig):
        super(LocalMseLoss, self).__init__()

    def forward(self,
                zl_a, zl_b,
                mask: AttentionMask = None, weights=None, structure_size: Union[None, int, Tuple[int, int]] = None):
        """

        :param zl_a:
        :param zl_b:
        :param mask:
        :param structure_size:
        :param weights: (B x N) weights sum over N is 1 for each sample in B
        :return:
        """
        assert zl_a.size() == zl_b.size()
        B, N, d = zl_a.size()

        if weights is None:
            weights = zl_a.new_ones(B, N)    # (B x N)
        if mask is not None:
            weights = mask.binary_mask * weights    # (B x N)
        weights = weights / weights.sum(dim=1, keepdim=True)  # (B x N)

        pdists = F.mse_loss(zl_a, zl_b, reduction='none').mean(-1)  # (B x N)
        pdists = weights * pdists  # (B x N)
        loss = pdists.sum(dim=1).mean()
        return loss, loss  # mse loss is symmetric


@dataclass
class LocalPredictorLossConfig(LocalAlignmentObjectiveConfig):
    objective_type: str = 'local_predictor'

    d_hidden: int = 2048


class LocalPredictorLoss(nn.Module):
    def __init__(self, config: LocalPredictorLossConfig, d_z: int):
        super(LocalPredictorLoss, self).__init__()
        self.predictor_a = SequenceMLP(d_z, d_z, d_hidden=config.d_hidden, norm='batch')
        self.predictor_b = SequenceMLP(d_z, d_z, d_hidden=config.d_hidden, norm='batch')

    def forward(self,
                zl_a, zl_b,
                mask: AttentionMask = None, weights=None, structure_size: Union[None, int, Tuple[int, int]] = None):
        """

        :param zl_a:
        :param zl_b:
        :param mask:
        :param structure_size:
        :param weights: (B x N) weights sum over N is 1 for each sample in B
        :return:
        """
        assert zl_a.size() == zl_b.size()
        B, N, d = zl_a.size()

        if weights is None:
            weights = zl_a.new_ones(B, N)  # (B x N)
        if mask is not None:
            weights = mask.binary_mask * weights    # (B x N)
        weights = weights / weights.sum(dim=1, keepdim=True)  # (B x N)

        hl_a = self.predictor_a(zl_a)  # (B x N x d)
        loss_1to2 = 2 - 2 * F.cosine_similarity(hl_a, zl_b, dim=-1)  # (B x N)
        loss_1to2 = weights * loss_1to2  # (B x N)
        loss_1to2 = loss_1to2.sum(dim=1).mean()

        hl_b = self.predictor_b(zl_b)  # (B x N x d)
        loss_2to1 = 2 - 2 * F.cosine_similarity(hl_b, zl_a, dim=-1)  # (B x N)
        loss_2to1 = weights * loss_2to1  # (B x N)
        loss_2to1 = loss_2to1.sum(dim=1).mean()

        return loss_1to2, loss_2to1


@dataclass
class LocalIntraSampleContrastiveLossConfig(LocalAlignmentObjectiveConfig):
    objective_type: str = 'local_intra_sample_contrastive'
    distance_threshold: float = 0.3
    similarity_temperature: float = 0.3  # pixel-contrast: 0.3
    temperature_trainable: bool = False
    threshold_type: str = 'relative'
    negatives_from_same_modality: bool = False
    normalize_by_num_negatives: bool = True

    smooth_weights: bool = False
    smooth_lambda: float = 1.

    cross_sample_negatives: bool = False

    accuracy_topk: Collection[int] = (1, 5)  # deprecated!


class LocalIntraSampleContrastiveLoss(nn.Module):
    def __init__(self,
                 config: LocalIntraSampleContrastiveLossConfig,
                 max_structure_size: Union[int, Tuple[int, int]]):
        super(LocalIntraSampleContrastiveLoss, self).__init__()

        assert config.objective_type == 'local_intra_sample_contrastive'
        config = prepare_config(config, LocalIntraSampleContrastiveLossConfig, log)
        self.config = config
        assert config.threshold_type in ('absolute', 'relative')
        self.temperature_trainable = config.temperature_trainable
        if self.temperature_trainable:
            self._temperature = nn.Parameter(torch.log(torch.tensor(config.similarity_temperature)))
        else:
            self._temperature = config.similarity_temperature
        self.negatives_from_same_modality = config.negatives_from_same_modality
        self.normalize_by_num_negatives = config.normalize_by_num_negatives
        self.cross_sample_negatives = config.cross_sample_negatives

        if isinstance(max_structure_size, int):
            N = max_structure_size
            coordinates = torch.arange(N)  # (N)
            if config.threshold_type == 'relative':
                coordinates = coordinates.float() / N

            distances = (coordinates[:, None] - coordinates).abs()  # (N x N)
            self.max_N = N
        else:
            H, W = max_structure_size
            coordinates = torch.meshgrid(torch.arange(H), torch.arange(W))
            coordinates = torch.stack(coordinates).float()  # (2 x H x W)
            coordinates = coordinates.view(2, -1).T  # ((H*W) x 2)
            if config.threshold_type == 'relative':
                # normalize by diagnoal, i.e. max distance. Can also be done on coords before computing distances
                coordinates /= math.sqrt(H ** 2 + W ** 2)

            distances = torch.cdist(coordinates, coordinates)  # ((H*W) x (H*W))
            distances = distances.view(H, W, H, W)  # ((H*W) x (H*W))
            self.max_N = H * W
        pos_mask, pos_weights = self._compute_pos_weights_from_distances(distances)
        negatives_mask = self.compute_negatives_mask(pos_mask, pos_weights)
        self.register_buffer('pos_weights', pos_weights)
        self.register_buffer('negatives_mask', negatives_mask)

    def _compute_pos_weights_from_distances(self, distances):
        pos_mask = distances <= self.config.distance_threshold
        if self.config.smooth_weights:
            # 1 - cumulative dist of exponential distribution
            distance_weights = torch.exp(-self.config.smooth_lambda * distances)
            pos_weights = pos_mask * distance_weights
        else:
            pos_weights = pos_mask.float()
        return pos_mask, pos_weights

    def compute_negatives_mask(self, pos_mask, pos_weights):
        negatives_mask = torch.zeros_like(pos_weights)
        negatives_mask.masked_fill_(pos_mask, float('-inf'))
        return negatives_mask

    def compute_pos_weights(self, spatial_positions, spatial_size):
        if self.config.threshold_type == 'relative':
            H, W = spatial_size
            # normalize by diagnoal, i.e. max distance. Can also be done on coords before computing distances
            spatial_positions = spatial_positions / math.sqrt(H ** 2 + W ** 2)

        distances = torch.cdist(spatial_positions, spatial_positions)  # (B x N x N)
        return self._compute_pos_weights_from_distances(distances)

    @property
    def similarity_temperature(self):
        if self.temperature_trainable:
            return torch.exp(self._temperature)
        else:
            return self._temperature

    def forward(self,
                zl_a, zl_b,
                mask: AttentionMask = None, weights=None, structure_size: Union[None, int, Tuple[int, int]] = None):
        has_weights = weights is not None
        assert zl_a.size() == zl_b.size()
        B, N, d = zl_a.size()

        if weights is None:
            weights = zl_a.new_ones(B, N)    # (B x N)
        if mask is not None:
            weights = mask.binary_mask * weights    # (B x N)
            additive_mask = mask.additive_mask
            N_negatives = mask.binary_mask.to(dtype=zl_a.dtype).sum(-1, keepdim=True)  # (B x 1)
        else:
            additive_mask = zl_a.new_zeros((B, N))
            N_negatives = torch.tensor(N, dtype=zl_a.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True)  # (B x N)

        pos_weights = self.pos_weights
        if isinstance(structure_size, int):
            assert N == structure_size
            assert pos_weights.ndim == 2
            pos_weights = pos_weights[:N, :N]
            if self.negatives_from_same_modality:
                negatives_mask = self.negatives_mask[:N, :N]
        else:
            H, W = structure_size
            assert N == H * W
            assert pos_weights.ndim == 4
            pos_weights = pos_weights[:H, :W, :H, :W]
            if self.negatives_from_same_modality:
                negatives_mask = self.negatives_mask[:H, :W, :H, :W].view(N, N)
        pos_weights = pos_weights.view(1, N, N).expand(B, N, N)  # (B x N x N)

        # mask out pad columns => 0 => should not be included in positive sums
        if mask is not None:
            pos_weights = pos_weights * mask.binary_mask[:, None, :]  # (B x N x N)
        # normalize weights row-wise
        pos_weights = pos_weights / pos_weights.sum(-1, keepdim=True).clamp_min(1e-12)  # (B x N x N)

        zl_a = F.normalize(zl_a, dim=-1, p=2)  # (B x N x d)
        zl_b = F.normalize(zl_b, dim=-1, p=2)  # (B x N x d)

        # within sample similarities of a<->b => in each batch, each local a with each local b
        all_scores_ab = torch.bmm(zl_a, zl_b.transpose(-1, -2)) / self.similarity_temperature  # (B x N x N)
        all_scores_ba = all_scores_ab.transpose(-1, -2)
        pos_scores_ab = pos_weights * all_scores_ab  # (B x N x N)
        pos_scores_ba = pos_weights * all_scores_ba  # (B x N x N)

        # mask out pad columns => -inf => should not be included in denominator of softmax
        all_scores_ab = all_scores_ab + additive_mask[:, None, :]  # (B x N x N)
        all_scores_ba = all_scores_ba + additive_mask[:, None, :]  # (B x N x N)

        if self.negatives_from_same_modality:
            all_scores_aa = torch.bmm(zl_a, zl_a.transpose(-1, -2)) / self.similarity_temperature  # (B x N x N)
            all_scores_bb = torch.bmm(zl_b, zl_b.transpose(-1, -2)) / self.similarity_temperature  # (B x N x N)

            all_scores_aa = all_scores_aa + negatives_mask  # (B x N x N)
            all_scores_bb = all_scores_bb + negatives_mask  # (B x N x N)

            # mask out pad columns => -inf => should not be included in denominator of softmax
            all_scores_aa = all_scores_aa + additive_mask[:, None, :]
            all_scores_bb = all_scores_bb + additive_mask[:, None, :]

            all_scores_ab = torch.cat([all_scores_ab, all_scores_aa], dim=-1)  # (B x N x 2N)
            all_scores_ba = torch.cat([all_scores_ba, all_scores_bb], dim=-1)  # (B x N x 2N)

            N_negatives *= 2

        if self.cross_sample_negatives:
            zl_a = zl_a.reshape(B*N, d)  # ((B * N) x d_z)
            zl_b = zl_b.reshape(B*N, d)  # ((B * N) x d_z)
            cross_scores_a2b = torch.mm(zl_a, zl_b.transpose(-1, -2)) / self.similarity_temperature  # ((B*N) x (B*N))
            cross_scores_a2b = cross_scores_a2b.view(B, N, B, N)
            # B with B is ignored, i.e. we do not use any comparison of local elements from the same sample as negatives
            cross_scores_a2b.diagonal(dim1=0, dim2=2).fill_(float('-inf'))
            cross_scores_b2a = cross_scores_a2b.permute(2, 3, 0, 1)  # transpose modalities (B x N x B x N)
            cross_scores_a2b = cross_scores_a2b + additive_mask[None, None, :, :]
            cross_scores_b2a = cross_scores_b2a + additive_mask[None, None, :, :]
            if self.bg_as_negative:
                cross_scores_a2b[:, :, :, 0] = float('-inf')
                cross_scores_b2a[:, :, :, 0] = float('-inf')
            cross_scores_a2b = cross_scores_a2b.view(B, N, B * N)
            cross_scores_b2a = cross_scores_b2a.view(B, N, B * N)
            all_scores_ab = torch.cat([all_scores_ab, cross_scores_a2b], dim=-1)
            all_scores_ba = torch.cat([all_scores_ba, cross_scores_b2a], dim=-1)
            if self.negatives_from_same_modality:
                raise NotImplementedError
            N_negatives *= B

        loss_ab = - pos_scores_ab.sum(-1) + torch.logsumexp(all_scores_ab, dim=-1)  # (B x N)
        loss_ba = - pos_scores_ba.sum(-1) + torch.logsumexp(all_scores_ba, dim=-1)  # (B x N)
        loss_ab = weights * loss_ab
        loss_ba = weights * loss_ba

        if self.normalize_by_num_negatives:
            N_negatives = N_negatives / (2 * self.max_N)  # negatives can come from same or other modality
            # normalize by number of negatives
            loss_ab = loss_ab - N_negatives.log()
            loss_ba = loss_ba - N_negatives.log()

        loss_ab = loss_ab.sum(-1).mean()
        loss_ba = loss_ba.sum(-1).mean()

        return loss_ab, loss_ba  # (1)


def get_local_alignment_objective(config: LocalAlignmentObjectiveConfig, max_structure_size: Union[int, Tuple[int, int]], d_z: int):
    if config.objective_type == 'local_mse':
        return LocalMseLoss(config)
    elif config.objective_type == 'local_predictor':
        return LocalPredictorLoss(config, d_z=d_z)
    elif config.objective_type == 'local_intra_sample_contrastive':
        return LocalIntraSampleContrastiveLoss(config, max_structure_size=max_structure_size)
    else:
        raise ValueError(f'Unknown local alignment objective "{config.objective_type}"')
