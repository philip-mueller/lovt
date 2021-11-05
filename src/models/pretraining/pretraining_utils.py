import logging
import os
from dataclasses import dataclass, field
from glob import glob
from typing import Union, Optional, Collection, Any, Dict

import torch
from omegaconf import MISSING
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataloader import default_collate

from models.components.attention import CrossAttentionOutput
from models.components.utils import EncoderConfig, AttentionMask
from models.image.scan_encoder import ScanEncoderModel, ScanEncoderConfig, ScanInputInfo
from models.text.report_encoder import ReportEncoderModel, ReportEncoderConfig, ReportInputInfo
from common.config_utils import prepare_config

from common.dataclass_utils import TensorDataclassMixin

VAL_INPUT_PREFIX = 'val_'

log = logging.getLogger(__name__)


@dataclass
class BiModalModelConfig:
    load_checkpoint_path: Optional[str] = None
    component_loaded_from_checkpoint: Collection[str] = ('model_a', 'model_b')

    # encoders
    encoder_a: Any = MISSING
    encoder_b: Any = MISSING

    # projections
    d_z: int = 512
    d_zg: Optional[int] = None  # ConVIRT: 512, BYOL: 256?, SimCLR: 128?, pixel-contrastive: 256
    d_hidden: int = 2048  # BYOL: 4096, pixel-contrastive: 2048, SimCLR: 2048 (most of the time, same as d_y)
    projection_norm: Optional[str] = 'batch'  # ConVIRT: null
    projection_dropout_prob: float = 0.

    # attention
    attention_sim: str = 'scaled_dot_product'  # or cosine
    attention_prob_dropout_prob: float = 0.
    attended_dropout_prob: float = 0.
    attention_norm: Optional[str] = 'batch'
    symmetric_attention: bool = True
    project_attention_keys: bool = True
    project_attention_values: bool = True
    project_attention_outputs: bool = True
    attention_temperature: float = 1.0
    attention_temperature_trainable: bool = False

    l_weights_a: Optional[str] = None
    l_weights_b: Optional[str] = None
    l_weights_stop_grad: bool = False

    ll_alignments_a: Dict[str, Any] = field(default_factory=dict)
    ll_alignments_b: Dict[str, Any] = field(default_factory=dict)
    g_alignment: Optional[Any] = None

    loss_weights: Dict[str, float] = field(default_factory=dict)

    # training
    batch_size: int = 32  # ConVIRT: 32, BYOL (pt-l): 256, SimCLR(pt-l): 128, pixel-contr: 1024
    learning_rate: float = 1e-4  # ConVIRT: 1e-4, SimCLR: 1e-3, BYOL: 1e-3, pixel-contr: bs/256
    weight_decay: float = 1e-6  # ConVIRT: 1e-6, SimCLR: 1e-6, BYOL: 1.5e-6, pixel-contr: 1e-5
    optimizer: str = 'Adam'  # Adam, AdamW
    # linear_warmup_cosine_annealing, reduce_on_plateau, cosine_annealing_per_epoch
    lr_scheduler: Collection[str] = ('cosine_annealing_per_epoch',)
    max_lr: Optional[float] = None
    warmup_steps: int = 10  # BYOL: 10, SimCLR: 10
    lr_reduce_patience: int = 12  # ConVIRT: about 12
    lr_reduce_factor: float = 0.5  # ConVIRT: 0.5
    max_epochs: int = 1000

    augment_on_validation: bool = False

    compute_retrieval_metrics: bool = True
    compute_embedding_stats: bool = True
    compute_attention_stats: bool = True
    compute_metrics_for_train: bool = False


def load_encoder(config: Union[EncoderConfig, nn.Module], dataset_stats: Optional[dict]):
    if isinstance(config, nn.Module):
        # a module was directly passed, this will be used as the encoder
        return config
    assert config._encoder_cls_ in ('ReportEncoderModel', 'ScanEncoderModel'), config._encoder_cls_
    if config._encoder_cls_ == 'ReportEncoderModel':
        encoder_model = ReportEncoderModel.load(prepare_config(config, ReportEncoderConfig, log), dataset_stats=dataset_stats)
    elif config._encoder_cls_ == 'ScanEncoderModel':
        encoder_model = ScanEncoderModel.load(prepare_config(config, ScanEncoderConfig, log), dataset_stats=dataset_stats)
    else:
        raise ValueError(config._encoder_cls_)
    return encoder_model


class BiModalBatchCollator:
    def __init__(self, input_name_a, input_name_b, collator_a, collator_b, val=False, augment_on_validation=False):
        self.input_name_a = input_name_a
        self.input_name_b = input_name_b
        self.collator_a = collator_a
        self.collator_b = collator_b
        self.val = val
        self.augment_on_validation = augment_on_validation

    def __call__(self, batch: list):
        elem = batch[0]
        result = {
            key: default_collate([d[key] for d in batch])
            for key in elem
            if key not in (self.input_name_a, self.input_name_b,
                           VAL_INPUT_PREFIX + self.input_name_a, VAL_INPUT_PREFIX + self.input_name_b)
        }

        result[self.input_name_a] = self.collator_a([d[self.input_name_a] for d in batch])
        result[self.input_name_b] = self.collator_b([d[self.input_name_b] for d in batch])
        if self.val:
            if self.augment_on_validation:
                result[VAL_INPUT_PREFIX + self.input_name_a] = self.collator_a([d[VAL_INPUT_PREFIX + self.input_name_a]
                                                                                for d in batch])
                result[VAL_INPUT_PREFIX + self.input_name_b] = self.collator_b([d[VAL_INPUT_PREFIX + self.input_name_b]
                                                                                for d in batch])
            else:
                # no-prefix and val-prefix data is the same, we do not need to collate again
                result[VAL_INPUT_PREFIX + self.input_name_a] = result[self.input_name_a]
                result[VAL_INPUT_PREFIX + self.input_name_b] = result[self.input_name_b]

        return result


class BiModalTransform:
    def __init__(self, input_name_a, input_name_b, transform_a, transform_b,
                 val_transform_a=None, val_transform_b=None, val=False, augment_on_validation=False):
        """

        :param input_name_a:
        :param input_name_b:
        :param transform_a:
        :param transform_b:
        :param val_transform_a:
        :param val_transform_b:
        :param val: Whether this transform is used for validation/test or training
        :param augment_on_validation: only relevant if val = True
            If true then validation transforms are used for input data and val-prefixed data
            If false then training transforms are used for input data but validation transforms are used for val-prefixed data
                    => for representation validation train transforms are used but for online eval val transforms are used
        """
        self.val = val
        self.augment_on_validation = augment_on_validation
        self.input_name_a = input_name_a
        self.input_name_b = input_name_b
        if not val or augment_on_validation:
            self.transform_a = transform_a
            self.transform_b = transform_b
        if val:
            assert val_transform_a is not None and val_transform_b is not None
            self.val_transform_a = val_transform_a
            self.val_transform_b = val_transform_b

    def __call__(self, sample):
        transformed_sample = {key: value for key, value in sample.items() if key not in (self.input_name_a, self.input_name_b)}

        if not self.val or self.augment_on_validation:
            transformed_sample[self.input_name_a] = self.transform_a(sample[self.input_name_a])
            transformed_sample[self.input_name_b] = self.transform_b(sample[self.input_name_b])

        if self.val:
            transformed_sample[VAL_INPUT_PREFIX + self.input_name_a] = self.val_transform_a(sample[self.input_name_a])
            transformed_sample[VAL_INPUT_PREFIX + self.input_name_b] = self.val_transform_b(sample[self.input_name_b])

            if not self.augment_on_validation:
                transformed_sample[self.input_name_a] = transformed_sample[VAL_INPUT_PREFIX + self.input_name_a]
                transformed_sample[self.input_name_b] = transformed_sample[VAL_INPUT_PREFIX + self.input_name_b]

        return transformed_sample


@dataclass
class ModelInputData(TensorDataclassMixin):
    patient_id: str
    study_id: str
    x_a: Union[ScanInputInfo, ReportInputInfo]
    x_b: Union[ScanInputInfo, ReportInputInfo]
    chexpert_bin_labels: Optional[dict] = None

    def save(self, folder: str, batch_idx: int):
        os.makedirs(folder, exist_ok=True)
        torch.save(self.to_dict(), os.path.join(folder, f'inputs_{batch_idx}.pt'))

    @staticmethod
    def load(folder: str, batch_idx: int, device='cpu'):
        path = os.path.join(folder, f'inputs_{batch_idx}.pt')
        if os.path.exists(path):
            data = torch.load(path, map_location=device)
            data['x_a'] = ScanInputInfo.from_dict(data['x_a']) if ScanInputInfo.is_scan_input(data['x_a']) \
                else ReportInputInfo.from_dict(data['x_a'])
            data['x_b'] = ScanInputInfo.from_dict(data['x_b']) if ScanInputInfo.is_scan_input(data['x_b']) \
                else ReportInputInfo.from_dict(data['x_b'])
            return ModelInputData(**data)
        else:
            raise IndexError

    @staticmethod
    def num_batches(folder: str):
        return len(glob(f'{folder}/inputs_*.pt'))


@dataclass
class AttentionData(TensorDataclassMixin):
    attention_probs_a2b: Optional[torch.Tensor] = None
    attention_probs_b2a: Optional[torch.Tensor] = None
    attention_scores_a2b: Optional[torch.Tensor] = None
    attention_scores_b2a: Optional[torch.Tensor] = None

    def save(self, folder: str, batch_idx: int):
        os.makedirs(folder, exist_ok=True)
        torch.save(self.to_dict(), os.path.join(folder, f'attentions_{batch_idx}.pt'))

    @staticmethod
    def load(folder: str, batch_idx: int, device='cpu'):
        path = os.path.join(folder, f'attentions_{batch_idx}.pt')
        if os.path.exists(path):
            return AttentionData(**torch.load(path, map_location=device))
        else:
            raise IndexError

    @staticmethod
    def num_batches(folder: str):
        return len(glob(f'{folder}/attentions*.pt'))

    @staticmethod
    def from_attention_output(attention_output: CrossAttentionOutput):
        if attention_output is None:
            return AttentionData()
        return AttentionData(
            attention_probs_a2b=attention_output.attention_probs_a2b,
            attention_probs_b2a=attention_output.attention_probs_b2a,
            attention_scores_a2b=attention_output.attention_scores_a2b,
            attention_scores_b2a=attention_output.attention_scores_b2a,
        )


@dataclass
class ModalityEmbeddingsData(TensorDataclassMixin):
    yg: Optional[torch.Tensor] = None
    yl: Optional[torch.Tensor] = None
    zg: Optional[torch.Tensor] = None
    zl: Optional[torch.Tensor] = None
    mask: Optional[AttentionMask] = None
    weights: Optional[torch.Tensor] = None
    local_size: Optional[Any] = None


@dataclass
class EmbeddingsData(TensorDataclassMixin):
    yg_a: Optional[torch.Tensor] = None
    yg_b: Optional[torch.Tensor] = None
    yl_a: Optional[torch.Tensor] = None
    yl_b: Optional[torch.Tensor] = None
    zg_a: Optional[torch.Tensor] = None
    zg_b: Optional[torch.Tensor] = None
    zl_a: Optional[torch.Tensor] = None
    zl_b: Optional[torch.Tensor] = None
    zl_b2a: Optional[torch.Tensor] = None
    zl_a2b: Optional[torch.Tensor] = None
    mask_a: Optional[AttentionMask] = None
    mask_b: Optional[AttentionMask] = None
    weights_a: Optional[torch.Tensor] = None
    weights_b: Optional[torch.Tensor] = None
    local_size_a: Optional[Any] = None
    local_size_b: Optional[Any] = None

    def normalize(self):
        return self.apply(lambda x: F.normalize(x, dim=-1, p=2), ignore=['mask_a', 'mask_b', 'weights_a', 'weights_b'])

    def save(self, folder: str, batch_idx: int):
        os.makedirs(folder, exist_ok=True)
        torch.save(self.to_dict(), os.path.join(folder, f'embeddings_{batch_idx}.pt'))

    @staticmethod
    def from_modalities(modality_a: ModalityEmbeddingsData, modality_b: ModalityEmbeddingsData):
        return EmbeddingsData(
            yg_a=modality_a.yg, yg_b=modality_b.yg,
            yl_a=modality_a.yl, yl_b=modality_b.yl,
            zg_a=modality_a.zg, zg_b=modality_b.zg,
            zl_a=modality_a.zl, zl_b=modality_b.zl,
            mask_a=modality_a.mask, mask_b=modality_b.mask,
            weights_a=modality_a.weights, weights_b=modality_b.weights,
            local_size_a=modality_a.local_size, local_size_b=modality_b.local_size
        )

    @staticmethod
    def load(folder: str, batch_idx: int, device='cpu'):
        path = os.path.join(folder, f'embeddings_{batch_idx}.pt')
        if os.path.exists(path):
            data_dict = torch.load(path, map_location=device)
            mask_a = data_dict.get('mask_a')
            if mask_a is not None:
                data_dict['mask_a'] = AttentionMask(**mask_a)
            mask_b = data_dict.get('mask_b')
            if mask_b is not None:
                data_dict['mask_b'] = AttentionMask(**mask_b)

            return EmbeddingsData(**data_dict)
        else:
            raise IndexError

    @staticmethod
    def num_batches(folder: str):
        return len(glob(f'{folder}/embeddings*.pt'))