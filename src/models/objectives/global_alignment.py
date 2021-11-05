import logging
from dataclasses import dataclass
from typing import Collection, Optional

import torch
from torch import nn
from torch.nn import functional as F

from common.config_utils import prepare_config
from models.components.fc import MLP
from metrics.classification_metrics import TopKAccuracy

log = logging.getLogger(__name__)

@dataclass
class GlobalAlignmentObjectiveConfig:
    objective_type: str


@dataclass
class GlobalMseLossConfig(GlobalAlignmentObjectiveConfig):
    objective_type: str = 'global_mse'


class GlobalMseLoss(nn.Module):
    def __init__(self, config: GlobalMseLossConfig):
        super(GlobalMseLoss, self).__init__()
        assert config.objective_type == 'global_mse'

    def forward(self, zg_1, zg_2, compute_1to2=True, compute_2to1=True):
        """

        :param zg_1: B x d_z
        :param zg_2: B x d_z
        :return: loss (1)
        """
        loss = F.mse_loss(zg_1, zg_2)

        return loss if compute_1to2 else None, loss if compute_2to1 else None, {}


@dataclass
class GlobalPredictorLossConfig(GlobalAlignmentObjectiveConfig):
    objective_type: str = 'global_predictor'
    d_hidden: int = 2048


class GlobalPredictorLoss(nn.Module):
    """
    See BYOL https://github.com/PyTorchLightning/PyTorch-Lightning-Bolts/blob/master/pl_bolts/models/self_supervised/byol/byol_module.py#L17-L170
    Note: should only be used with BYOL training

    Note: BYOL uses d_hidden = 4096, pixel-contrastive uses d_hidden 2048
    """
    def __init__(self, config: GlobalPredictorLossConfig, d_z: int):
        super(GlobalPredictorLoss, self).__init__()
        assert config.objective_type == 'global_predictor'
        self.predictor_1 = MLP(d_z, d_z, d_hidden=config.d_hidden, norm='batch')
        self.predictor_2 = MLP(d_z, d_z, d_hidden=config.d_hidden, norm='batch')

    def forward(self, zg_1, zg_2, compute_1to2=True, compute_2to1=True):
        """

        :param zg_1: B x d_z
        :param zg_2: B x d_z
        :return: loss (1)
        """
        if compute_1to2:
            h_1 = self.predictor_1(zg_1)
            loss_1to2 = 2 - 2 * F.cosine_similarity(h_1, zg_2).mean()
        else:
            loss_1to2 = None

        if compute_2to1:
            h_2 = self.predictor_2(zg_2)
            loss_2to1 = 2 - 2 * F.cosine_similarity(h_2, zg_1).mean()
        else:
            loss_2to1 = None

        return loss_1to2, loss_2to1, {}


@dataclass
class GlobalNceLossConfig(GlobalAlignmentObjectiveConfig):
    objective_type: str = 'global_nce'
    similarity_temperature: float = 0.1  # ConVIRT: 0.1, SimCLR: 0.1
    temperature_trainable: bool = False

    negatives_from_same_modality: bool = True

    accuracy_topk: Collection[int] = (1, 5)


class GlobalNceLoss(nn.Module):
    """
    NT Xent Loss
    See https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py
    """
    def __init__(self, config: GlobalNceLossConfig):
        super(GlobalNceLoss, self).__init__()
        assert config.objective_type == 'global_nce'
        config = prepare_config(config, GlobalNceLossConfig, log)
        self.temperature_trainable = config.temperature_trainable
        if self.temperature_trainable:
            self._temperature = nn.Parameter(torch.log(torch.tensor(config.similarity_temperature)))
        else:
            self._temperature = config.similarity_temperature
        self.negatives_from_same_modality = config.negatives_from_same_modality

        self.topk_acc_metric = TopKAccuracy(topk=config.accuracy_topk)

    @property
    def similarity_temperature(self):
        if self.temperature_trainable:
            return torch.exp(self._temperature)
        else:
            return self._temperature

    def forward(self, zg_1, zg_2, compute_1to2=True, compute_2to1=True):
        """

        :param zg_1: (B x d_z)
        :param zg_2: (B x d_z)
        :return:
        """
        device = zg_1.device
        B, d_z = zg_1.size()
        zg_1 = F.normalize(zg_1, dim=-1, p=2)  # (B x d_z)
        zg_2 = F.normalize(zg_2, dim=-1, p=2)  # (B x d_z)

        zg_both = torch.cat([zg_1, zg_2], dim=0)  # (2B x d_z)

        all_scores = torch.mm(zg_both, zg_both.t()) / self.similarity_temperature  # (2B x 2B)
        pos_scores = all_scores[:B, B:]  # (B x B), the sector zg_1 x zg_2 (which is equal to zg_2 x zg_1)
        pos_scores = pos_scores.diagonal()  # (B)

        labels = torch.zeros(B, dtype=torch.long, device=device)

        metrics = {}
        if compute_1to2:
            if self.negatives_from_same_modality:
                # build the logits tensor, first column for positives, the other column for the negatives
                logits_1to2 = torch.empty(B, 1 + (2 * B), device=device, dtype=all_scores.dtype)  # (B x (1 + 2B))
                logits_1to2[:, 0] = pos_scores  # 1 positive (each 1 with related 2)
                logits_1to2[:, 1:] = all_scores[:B, :]  # negatives of 1->2 (for each 1 all of 1 and 2)
                # mask out the negatives in the diagonals
                logits_1to2[:, 1:B + 1].diagonal().fill_(float('-inf'))
                logits_1to2[:, B + 1:].diagonal().fill_(float('-inf'))
            else:
                # build the logits tensor, first column for positives, the other column for the negatives
                logits_1to2 = torch.empty(B, 1 + B, device=device, dtype=all_scores.dtype)  # (B x (1 + 2B))
                logits_1to2[:, 0] = pos_scores  # 1 positive (each 1 with related 2)
                logits_1to2[:, 1:] = all_scores[:B, B:]  # negatives of 1->2 (for each 1 all of 2)
                # mask out the negatives in the diagonals
                logits_1to2[:, 1:].diagonal().fill_(float('-inf'))
            loss_1to2 = F.cross_entropy(logits_1to2, labels)
            metrics.update({'a2b_' + name: acc.detach()
                            for name, acc in self.topk_acc_metric(logits_1to2, labels).items()})
            del logits_1to2
        else:
            loss_1to2 = None

        if compute_2to1:
            if self.negatives_from_same_modality:
                # build the logits tensor, first column for positives, the other column for the negatives
                logits_2to1 = torch.empty(B, 1 + (2 * B), device=device, dtype=all_scores.dtype)  # (B x (1 + 2B))
                logits_2to1[:, 0] = pos_scores  # 1 positive (each 2 with related 1)
                logits_2to1[:, 1:] = all_scores[B:, :]  # negatives of 2->1 (for each 2 all of 1 and 2)
                # mask out the negatives in the diagonals
                logits_2to1[:, 1:B + 1].diagonal().fill_(float('-inf'))
                logits_2to1[:, B + 1:].diagonal().fill_(float('-inf'))
            else:
                # build the logits tensor, first column for positives, the other column for the negatives
                logits_2to1 = torch.empty(B, 1 + B, device=device, dtype=all_scores.dtype)  # (B x (1 + 2B))
                logits_2to1[:, 0] = pos_scores  # 1 positive (each 2 with related 1)
                logits_2to1[:, 1:] = all_scores[B:, :B]  # negatives of 2->1 (for each 2 all of 1)
                # mask out the negatives in the diagonals
                logits_2to1[:, 1:].diagonal().fill_(float('-inf'))
            loss_2to1 = F.cross_entropy(logits_2to1, labels)
            metrics.update({'b2a_' + name: acc.detach()
                            for name, acc in self.topk_acc_metric(logits_2to1, labels).items()})
            del logits_2to1
        else:
            loss_2to1 = None
        return loss_1to2, loss_2to1, metrics


def get_global_alignment_objective(config: GlobalAlignmentObjectiveConfig, d_zg: int = 256, d_hidden: int = 2048):
    if config is None:
        return None

    if config.objective_type == 'global_mse':
        return GlobalMseLoss(config)
    elif config.objective_type == 'global_predictor':
        return GlobalPredictorLoss(config, d_z=d_zg)
    elif config.objective_type == 'global_nce':
        return GlobalNceLoss(config)
    else:
        raise ValueError(f'Unknown global alignment objective "{config.objective_type}"')
