from functools import partial
from typing import Optional

import torch
from torch import nn
from torch.nn import MultiheadAttention

from models.components.utils import AttentionMask


class GlobalMaxPool(nn.Module):
    def __init__(self, dim: int = 1, d=None):
        super(GlobalMaxPool, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, mask: Optional[AttentionMask] = None):
        """

        :param x: (B x N x d)
        :param mask: (B x N)
        :return:
        """
        if mask is not None:
            x = x + mask.additive_mask[:, :, None]
        return torch.max(x, dim=self.dim)[0]


class GlobalAvgPool(nn.Module):
    def __init__(self, dim: int = 1, d=None):
        super(GlobalAvgPool, self).__init__()
        self.dim = dim

    def forward(self, x, mask: Optional[AttentionMask] = None):
        if mask is not None:
            x = torch.masked_fill(x, mask.inverted_binary_mask[:, :, None], 0.)
        return torch.mean(x, dim=self.dim)


class GlobalAvgAttentionAggregator(nn.Module):
    def __init__(self, d, num_heads=8):
        super(GlobalAvgAttentionAggregator, self).__init__()

        self.avg_pool = GlobalAvgPool(dim=1)
        self.attention = MultiheadAttention(d, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[AttentionMask] = None, return_weights=False):
        """

        :param x: (B x N x d)
        :param mask:
        :return:
        """
        query = self.avg_pool(x, mask=mask)  # (B x d)
        bool_mask = mask.inverted_binary_mask if mask is not None else None
        # (B x 1 x d), (B x 1 x N)
        attn_output, attn_output_weights = self.attention(query[:, None, :], key=x, value=x, key_padding_mask=bool_mask)
        attn_output = attn_output.squeeze(dim=1)  # (B x d)
        attn_output_weights = attn_output_weights.squeeze(1)  # (B x N)

        if return_weights:
            return attn_output, attn_output_weights
        else:
            return attn_output


class GlobalTokenAggregator(nn.Module):
    def __init__(self, global_index=0, d=None):
        super(GlobalTokenAggregator, self).__init__()
        self.global_index = global_index

    def forward(self, x: torch.Tensor, mask: Optional[AttentionMask] = None):
        return x[:, self.global_index, :]


AGGREGATOR_DICT = {
    'max': GlobalMaxPool,
    'avg': GlobalAvgPool,
    'avgpool_attention': GlobalAvgAttentionAggregator,
    'token_0': partial(GlobalTokenAggregator, global_index=0),
}


def get_aggregator(aggregator, d: int, **kwargs):
    """

    :param aggregator:
    :param d:
    :param dim:
    :param kwargs:
    :return: Aggegrator: (B x N x d) -> (B x d)
    """
    if isinstance(aggregator, str):
        return AGGREGATOR_DICT[aggregator](d=d, **kwargs)
    else:
        return aggregator
