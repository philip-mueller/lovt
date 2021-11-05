
# ----- Similarities used in Attention -----
import math
from dataclasses import dataclass
from functools import partial
from typing import Union, Collection, NamedTuple, Optional

import torch
from torch import nn
import torch.nn.functional as F

# ---------- General Components and Similarities ----------
from models.components.utils import get_norm_layer, AttentionMask
from common.dataclass_utils import TensorDataclassMixin


class ScaledDotProductSimilarity(nn.Module):
    """
    See
     - A. Vaswani et al. "Attention is all you need" (2017), i.e. the original transformer
     - transformers.models.bert.modeling_bert.BertSelfAttention

     Note: This is equal to bilinear similarity if queries or keys are linearly projected first.
    """
    def __init__(self, d: int):
        super(ScaledDotProductSimilarity, self).__init__()
        self.d = d

    def forward(self, queries, keys):
        """

        Note: ... may be no dimension, a single dimension (e.g. batch) or multiple (e.g. batch, multi-head)
        :param queries: ([B x] [N_h x] [N_q x] d_k)
        :param keys: ([B x] [N_h x] x N_k x d_k)
        :return: ([B x] [N_h x] [N_q x] x N_k)
        """
        attention_scores = queries @ keys.transpose(-1, -2)  # ([B x] [N_h x] [N_q x] N_k)
        return attention_scores / math.sqrt(self.d)  # ([B x] [N_h x] [N_q x] N_k)


class CosineQueryKeySimilarity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CosineQueryKeySimilarity, self).__init__()

    def forward(self, queries, keys):
        """

        Note: ... may be no dimension, a single dimension (e.g. batch) or multiple (e.g. batch, multi-head)
        :param queries: ([B x] [N_h x] [N_q x] d_k)
        :param keys: ([B x] [N_h x] x N_k x d_k)
        :return: ([B x] [N_h x] [N_q x] x N_k)
        """
        queries = F.normalize(queries, dim=-1)
        keys = F.normalize(keys, dim=-1)
        return queries @ keys.transpose(-1, -2)  # ([B x] [N_h x] [N_q x] N_k)


SIMILARITIES_MAP = {
    'scaled_dot_product': ScaledDotProductSimilarity,
    'cosine': CosineQueryKeySimilarity
}


def get_similarity(similarity, d: int):
    if isinstance(similarity, str):
        similarity = SIMILARITIES_MAP[similarity](d)
    return similarity


@dataclass
class CrossAttentionOutput(TensorDataclassMixin):
    attended_b2a: Optional[torch.Tensor] = None
    attended_a2b: Optional[torch.Tensor] = None
    attention_probs_b2a: Optional[torch.Tensor] = None
    attention_probs_a2b: Optional[torch.Tensor] = None
    attention_scores_b2a: Optional[torch.Tensor] = None
    attention_scores_a2b: Optional[torch.Tensor] = None


class CrossAttention(nn.Module):
    def __init__(self,
                 d: int, d_k: int = None, d_v: int = None,
                 symmetric=True,
                 project_keys=True,
                 project_values=True,
                 project_output=True,
                 similarity='scaled_dot_product',
                 dropout_prob=0., attention_probs_dropout_prob=0., output_norm='layer',
                 temperature=1.,
                 temperature_trainable=False):
        super(CrossAttention, self).__init__()
        if d_k is None:
            d_k = d
        else:
            assert project_keys

        if d_v is None:
            d_v = d
        else:
            assert project_values and project_output

        self.similarity = get_similarity(similarity, d=d_k)
        self.attention_dropout = nn.Dropout(attention_probs_dropout_prob)
        self.has_attention_dropout = attention_probs_dropout_prob > 0.

        self.symmetric = symmetric
        self.temperature_trainable = temperature_trainable
        if temperature_trainable:
            self._temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self._temperature = temperature

        self.d = d
        self.d_k = d_k
        self.d_v = d_v

        if symmetric:
            self.key_projection = nn.Linear(d, d_k) if project_keys else None
            self.value_projection = nn.Linear(d, d_v) if project_values else None
            self.output_projection = nn.Linear(d_v, d) if project_output else None
        else:
            assert project_keys
            self.query_projection_a = nn.Linear(d, d_k)
            self.query_projection_b = nn.Linear(d, d_k)
            self.key_projection_a = nn.Linear(d, d_k)
            self.key_projection_b = nn.Linear(d, d_k)
            self.value_projection_a = nn.Linear(d, d_v) if project_values else None
            self.value_projection_b = nn.Linear(d, d_v) if project_values else None
            self.output_projection_a = nn.Linear(d_v, d) if project_output else None
            self.output_projection_b = nn.Linear(d_v, d) if project_output else None

        self.norm = get_norm_layer(output_norm, d)
        self.dropout = nn.Dropout(dropout_prob)

        if symmetric:
            if project_keys:
                nn.init.kaiming_normal_(self.key_projection.weight, mode='fan_out', nonlinearity='linear')
                nn.init.zeros_(self.key_projection.bias)
            if project_values:
                nn.init.kaiming_normal_(self.value_projection.weight, mode='fan_out', nonlinearity='linear')
                nn.init.zeros_(self.value_projection.bias)
            if project_output:
                nn.init.kaiming_normal_(self.output_projection.weight, mode='fan_out', nonlinearity='linear')
                nn.init.zeros_(self.output_projection.bias)
        else:
            nn.init.kaiming_normal_(self.query_projection_a.weight, mode='fan_out', nonlinearity='linear')
            nn.init.zeros_(self.query_projection_a.bias)
            nn.init.kaiming_normal_(self.query_projection_b.weight, mode='fan_out', nonlinearity='linear')
            nn.init.zeros_(self.query_projection_b.bias)
            nn.init.kaiming_normal_(self.key_projection_a.weight, mode='fan_out', nonlinearity='linear')
            nn.init.zeros_(self.key_projection_a.bias)
            nn.init.kaiming_normal_(self.key_projection_b.weight, mode='fan_out', nonlinearity='linear')
            nn.init.zeros_(self.key_projection_b.bias)
            if project_values:
                nn.init.kaiming_normal_(self.value_projection_a.weight, mode='fan_out', nonlinearity='linear')
                nn.init.zeros_(self.value_projection_a.bias)
                nn.init.kaiming_normal_(self.value_projection_b.weight, mode='fan_out', nonlinearity='linear')
                nn.init.zeros_(self.value_projection_b.bias)
            if project_output:
                nn.init.kaiming_normal_(self.output_projection_a.weight, mode='fan_out', nonlinearity='linear')
                nn.init.zeros_(self.output_projection_a.bias)
                nn.init.kaiming_normal_(self.output_projection_b.weight, mode='fan_out', nonlinearity='linear')
                nn.init.zeros_(self.output_projection_b.bias)

    @property
    def temperature(self):
        if self.temperature_trainable:
            return torch.exp(self._temperature)
        else:
            return self._temperature

    def forward(self,
                hidden_a, hidden_b,
                mask_a: Optional[AttentionMask] = None, mask_b: Optional[AttentionMask] = None,
                a2b=True, b2a=True):
        assert a2b or b2a

        # --- compute scores ---
        if self.symmetric:
            if self.key_projection:  # => bilinear similarity with a positive semi-definite matrix
                keys_a = self.key_projection(hidden_a)
                keys_b = self.key_projection(hidden_b)
            else:
                keys_a = hidden_a
                keys_b = hidden_b
            if self.value_projection:
                values_a = self.value_projection(hidden_a) if a2b else None
                values_b = self.value_projection(hidden_b) if b2a else None
            else:
                values_a = hidden_a
                values_b = hidden_b

            attention_scores_b2a = self.similarity(keys_a, keys_b)  # (B x N_a x N_b)
            attention_scores_a2b = attention_scores_b2a.transpose(-1, -2).clone()  # (B x N_b x N_a)
        else:
            queries_a = self.query_projection_a(hidden_a)
            queries_b = self.query_projection_b(hidden_b)
            keys_a = self.key_projection_a(hidden_a)
            keys_b = self.key_projection_b(hidden_b)

            if self.value_projection_a and self.value_projection_b:
                values_a = self.value_projection_a(hidden_a) if a2b else None
                values_b = self.value_projection_b(hidden_b) if b2a else None
            else:
                values_a = hidden_a
                values_b = hidden_b
            attention_scores_b2a = self.similarity(queries_a, keys_b)  # (B x N_a x N_b)
            attention_scores_a2b = self.similarity(queries_b, keys_a)  # (B x N_b x N_a)

        # --- compute probs and attend
        if b2a:
            attended_b2a, attention_probs_b2a = self._apply_attention(attention_scores_b2a, values_b,
                                                                      mask=mask_a,
                                                                      other_mask=mask_b,
                                                                      a2b=False)
        else:
            attended_b2a, attention_probs_b2a = None, None
        if a2b:
            attended_a2b, attention_probs_a2b = self._apply_attention(attention_scores_a2b, values_a,
                                                                      mask=mask_b,
                                                                      other_mask=mask_a,
                                                                      a2b=True)
        else:
            attended_a2b, attention_probs_a2b = None, None

        return CrossAttentionOutput(
            attended_b2a=attended_b2a,
            attended_a2b=attended_a2b,
            attention_probs_b2a=attention_probs_b2a,
            attention_probs_a2b=attention_probs_a2b,
            attention_scores_b2a=attention_scores_b2a,
            attention_scores_a2b=attention_scores_a2b
        )

    def _apply_attention(self, attention_scores, other_values, mask, other_mask, a2b):
        """

        :param attention_scores: (B x N_1 x N_2)
        :param other_values:  (B x N_2 x d)
        :return: (B x N_1 x d)
        """
        if self.temperature != 1.:
            attention_scores = attention_scores / self.temperature

        if other_mask is not None:
            attention_scores = attention_scores + other_mask.additive_mask[:, None, :]  # (B x N_1 x N_2)
        attention_probs = torch.softmax(attention_scores, dim=-1)  # (B x N_1 x N_2)
        if mask is not None:
            attention_probs = attention_probs * mask.binary_mask[:, :, None]

        attention_probs = self.attention_dropout(attention_probs)

        if self.has_attention_dropout:  # renormalize probs
            attention_probs = attention_probs / attention_probs.sum(-1, keepdim=True)
        attended = attention_probs @ other_values  # (B x N_1 x d)

        if self.symmetric:
            if self.output_projection:
                attended = self.output_projection(attended)  # (B x N_1 x d)
        else:
            if self.output_projection_a and self.output_projection_b:
                if a2b:
                    attended = self.output_projection_b(attended)  # (B x N_b x d)
                else:  # b2a
                    attended = self.output_projection_a(attended)  # (B x N_a x d)
        attended = self.dropout(attended)  # (B x N_1 x d)
        attended = self._apply_norm(attended)  # (B x N_1 x d)

        return attended, attention_probs

    def _apply_norm(self, x):
        if isinstance(self.norm, nn.BatchNorm1d):
            # BN work on the second last dim
            return self.norm(x.transpose(-1, -2)).transpose(-1, -2)
        else:
            return self.norm(x)
