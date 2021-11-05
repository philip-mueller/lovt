from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Optional, Union, Tuple, Any

import torch
from omegaconf import MISSING
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from common.dataclass_utils import TensorDataclassMixin


def get_norm_layer(norm: Optional[str], d):
    if norm is None:
        return lambda x: x
    elif norm == 'layer':
        return nn.LayerNorm(d)
    elif norm == 'batch':
        return nn.BatchNorm1d(d)
    elif norm == 'l2':
        return partial(F.normalize, dim=-1, p=2)
    else:
        raise NotImplementedError


@dataclass
class EncoderConfig:
    _encoder_cls_: str = MISSING
    modality: str = MISSING


class EncoderInterface(ABC):
    def __init__(self):
        super(EncoderInterface, self).__init__()
        self.transform = lambda x: x
        self.val_transform = lambda x: x
        self.batch_collator = default_collate

    @property
    def max_region_size(self):
        return None

    @abstractmethod
    def update_data_augmentation(self, data_augmentation_config: Optional[Any] = None):
        ...


@dataclass
class AttentionMask(TensorDataclassMixin):
    binary_mask: torch.Tensor
    inverted_binary_mask: torch.Tensor
    additive_mask: torch.Tensor

    @staticmethod
    def from_binary_mask(binary_mask: torch.Tensor, dtype):
        if binary_mask is not None:
            binary_mask = binary_mask.bool()
        additive_mask = AttentionMask._compute_additive_attention_mask(binary_mask, dtype)
        return AttentionMask(binary_mask, ~binary_mask, additive_mask)

    @staticmethod
    def from_binary_mask_or_attention_mask(mask: Optional[Union['AttentionMask', torch.Tensor]], dtype):
        if mask is None or isinstance(mask, AttentionMask):
            return mask
        else:
            assert isinstance(mask, torch.Tensor) and (mask.dtype in (torch.bool, torch.uint8, torch.int64)), \
                (type(mask), mask.dtype)
            return AttentionMask.from_binary_mask(mask, dtype)

    @staticmethod
    def _compute_additive_attention_mask(binary_attention_mask: torch.Tensor, dtype):
        if binary_attention_mask is None:
            return None
        additive_attention_mask = torch.zeros_like(binary_attention_mask, dtype=dtype)
        additive_attention_mask.masked_fill_(~binary_attention_mask, float('-inf'))
        return additive_attention_mask

    @staticmethod
    def get_additive_mask(mask: Optional[Union['AttentionMask', torch.Tensor]], dtype):
        if mask is None:
            return None
        if isinstance(mask, AttentionMask):
            return mask.additive_mask
        elif mask.dtype == torch.bool or mask.dtype == torch.uint8:
            return AttentionMask._compute_additive_attention_mask(mask, dtype)
        else:
            return mask

    @staticmethod
    def get_additive_cross_attention_mask(mask_a: Optional['AttentionMask'] = None,
                                          mask_b: Optional['AttentionMask'] = None,
                                          mask_ab: Optional['AttentionMask'] = None):
        """

        :param mask_a: (B x N_a)
        :param mask_b: (B x N_b)
        :param mask_ab: (B x N_a x N_b)
        :return:
        """
        if mask_a is None and mask_b is None and mask_ab is None:
            return None
        else:
            mask = 0.
            if mask_ab is not None:
                mask = mask + mask_ab.additive_mask
            if mask_a is not None:
                mask = mask + mask_a.additive_mask[:, :, None]
            if mask_b is not None:
                mask = mask + mask_b.additive_mask[:, None, :]
            return mask


@dataclass
class EncoderOutput(TensorDataclassMixin):
    local_features: Optional[torch.Tensor]
    global_features: Optional[torch.Tensor]

    local_structure_size: Union[int, Tuple[int, int]]
    local_mask: Optional[AttentionMask] = None
    local_weights: Optional[torch.Tensor] = None
