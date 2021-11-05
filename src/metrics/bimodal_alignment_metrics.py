import gc

import torch
from torch import nn

from metrics.attention_metrics import CrossAttentionMetrics
from metrics.embedding_metrics import EmbeddingMetrics
from metrics.retrieval_metrics import LocalRetrievalMetrics
from models.pretraining.pretraining_utils import AttentionData, EmbeddingsData


class BiModalAlignmentMetrics(nn.Module):
    def __init__(self,
                 compute_retrieval_metrics=True, compute_embedding_stats=True, compute_attention_stats=True):
        super(BiModalAlignmentMetrics, self).__init__()

        self.compute_retrieval_metrics = compute_retrieval_metrics
        self.compute_embedding_stats = compute_embedding_stats
        self.compute_attention_stats = compute_attention_stats

        if compute_retrieval_metrics:
            self.l_retrieval_metrics_a_l2att = LocalRetrievalMetrics(
                acc_name='a_l2att_top_{k}_acc',
                weighted_acc_name='a_l2att_weighted_top_{k}_acc',
                retrieval_index_name='a_l2att_avg_retrieval_index',
                weighted_retrieval_index_name='a_l2att_weighted_avg_retrieval_index',
                avg_local_size_name='a_avg_local_size')
            self.l_retrieval_metrics_a_att2l = LocalRetrievalMetrics(
                acc_name='a_att2l_top_{k}_acc',
                weighted_acc_name='a_att2l_weighted_top_{k}_acc',
                retrieval_index_name='a_att2l_avg_retrieval_index',
                weighted_retrieval_index_name='a_att2l_weighted_avg_retrieval_index',
                avg_local_size_name=None)
            self.l_retrieval_metrics_b_l2att = LocalRetrievalMetrics(
                acc_name='b_l2att_top_{k}_acc',
                weighted_acc_name='b_l2att_weighted_top_{k}_acc',
                retrieval_index_name='b_l2att_avg_retrieval_index',
                weighted_retrieval_index_name='b_l2att_weighted_avg_retrieval_index',
                avg_local_size_name='b_avg_local_size')
            self.l_retrieval_metrics_b_att2l = LocalRetrievalMetrics(
                acc_name='b_att2l_top_{k}_acc',
                weighted_acc_name='b_att2l_weighted_top_{k}_acc',
                retrieval_index_name='b_att2l_avg_retrieval_index',
                weighted_retrieval_index_name='b_att2l_weighted_avg_retrieval_index',
                avg_local_size_name=None)
        if compute_embedding_stats:
            self.embedding_stats_metrics = EmbeddingMetrics()
        if compute_attention_stats:
            self.attention_stats_metrics = CrossAttentionMetrics()

    def forward(self, embeddings: EmbeddingsData, attention_probs: AttentionData):
        with torch.no_grad():
            if self.compute_retrieval_metrics:
                if embeddings.zl_a is not None and embeddings.zl_b2a is not None:
                    # (B x N_a x N_a)
                    distances_a = LocalRetrievalMetrics.compute_similarities(embeddings.zl_a, embeddings.zl_b2a)
                    self.l_retrieval_metrics_a_l2att(
                        distances_a,
                        mask=embeddings.mask_a,
                        weights=embeddings.weights_a
                    )
                    self.l_retrieval_metrics_a_att2l(
                        distances_a.transpose(-1, -2),
                        mask=embeddings.mask_a,
                        weights=embeddings.weights_a
                    )
                if embeddings.zl_b is not None and embeddings.zl_a2b is not None:
                    # (B x N_b x N_b)
                    distances_b = LocalRetrievalMetrics.compute_similarities(embeddings.zl_b, embeddings.zl_a2b)
                    self.l_retrieval_metrics_b_l2att(
                        distances_b,
                        mask=embeddings.mask_b,
                        weights=embeddings.weights_b
                    )
                    self.l_retrieval_metrics_b_att2l(
                        distances_b.transpose(-1, -2),
                        mask=embeddings.mask_b,
                        weights=embeddings.weights_b
                    )
            if self.compute_embedding_stats:
                self.embedding_stats_metrics(embeddings)
            if self.compute_attention_stats:
                self.attention_stats_metrics(
                    attention_probs.attention_probs_b2a, attention_probs.attention_probs_a2b,
                    embeddings.mask_a, embeddings.mask_b
                )

    def compute(self) -> dict:
        with torch.no_grad():
            metrics = {}
            if self.compute_retrieval_metrics:
                metrics.update(self.l_retrieval_metrics_a_l2att.compute())
                metrics.update(self.l_retrieval_metrics_a_att2l.compute())
                metrics.update(self.l_retrieval_metrics_b_l2att.compute())
                metrics.update(self.l_retrieval_metrics_b_att2l.compute())
            if self.compute_embedding_stats:
                metrics.update({'emb/' + key: value for key, value in self.embedding_stats_metrics.compute().items()})
            if self.compute_attention_stats:
                metrics.update({'att/' + key: value for key, value in self.attention_stats_metrics.compute().items()})
            return metrics

    def reset(self):
        if self.compute_retrieval_metrics:
            self.l_retrieval_metrics_a_l2att.reset()
            self.l_retrieval_metrics_a_att2l.reset()
            self.l_retrieval_metrics_b_l2att.reset()
            self.l_retrieval_metrics_b_att2l.reset()
        if self.compute_embedding_stats:
            self.embedding_stats_metrics = EmbeddingMetrics()
        if self.compute_attention_stats:
            self.attention_stats_metrics = CrossAttentionMetrics()
        gc.collect()