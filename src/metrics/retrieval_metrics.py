import torch
from torch.nn import functional as F
from torchmetrics import Metric

from models.components.utils import AttentionMask


class LocalRetrievalMetrics(Metric):
    def __init__(self, topk=(1, 5),
                 acc_name='top_{k}_acc',
                 weighted_acc_name='weighted_top_{k}_acc',
                 retrieval_index_name='avg_retrieval_index',
                 weighted_retrieval_index_name='weighted_avg_retrieval_index',
                 avg_local_size_name='avg_local_size',
                 dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)

        self.acc_names = [acc_name.format(k=k) for k in topk]
        self.weighted_acc_names = [weighted_acc_name.format(k=k) for k in topk]
        self.retrieval_index_name = retrieval_index_name
        self.weighted_retrieval_index_name = weighted_retrieval_index_name
        self.avg_local_size_name = avg_local_size_name

        self.register_buffer("topk", torch.tensor(topk, dtype=int))
        self.add_state("sample_acc_sum", default=torch.zeros(len(topk), dtype=float), dist_reduce_fx="sum")
        self.add_state("sample_weighted_acc_sum", default=torch.zeros(len(topk), dtype=float), dist_reduce_fx="sum")
        self.add_state("sample_avg_index_sum", default=torch.tensor(0., dtype=float), dist_reduce_fx="sum")
        self.add_state("sample_weighted_avg_index_sum", default=torch.tensor(0., dtype=float), dist_reduce_fx="sum")
        self.add_state("sample_count", default=torch.tensor(0., dtype=float), dist_reduce_fx="sum")
        self.add_state("weighted_sample_count", default=torch.tensor(0., dtype=float), dist_reduce_fx="sum")
        self.add_state("total_local", default=torch.tensor(0., dtype=float), dist_reduce_fx="sum")

    @staticmethod
    def compute_similarities(local_1: torch.Tensor, local_2: torch.Tensor):
        local_1 = F.normalize(local_1, dim=-1, p=2)  # (B x N x d)
        local_2 = F.normalize(local_2, dim=-1, p=2)  # (B x N x d)
        return torch.bmm(local_1, local_2.transpose(-1, -2))  # (B x N x N)

    def update(self, similarities: torch.Tensor, mask: AttentionMask=None, weights=None):
        """

        :param similarities: (B x N x N)
        :param mask: (B x N)
        :param weights: (B x N)
        """
        B, N, _ = similarities.size()

        if mask is not None:
            # set masked columns to -inf such that they are smaller than any other similarity
            similarities = similarities + mask.additive_mask[:, None, :]  # (B x N x N)

        true_similarities = similarities.diagonal(dim1=1, dim2=2)  # (B x N)
        retrieval_indices = (similarities > true_similarities[:, :, None]).sum(-1)  # (B x N)

        correct = retrieval_indices[None, :, :] < self.topk[:, None, None]  # (numk x B x N)
        correct = correct.float()
        retrieval_indices = retrieval_indices.float()

        if mask is not None:
            retrieval_indices = mask.binary_mask * retrieval_indices
            correct = mask.binary_mask[None, :, :] * correct

            num_local = mask.binary_mask.sum(-1)  # (B)
            accuracies = correct.sum(-1) / num_local[None, :]  # (numk x B)
            avg_index = retrieval_indices.sum(-1) / num_local  # (B)
            self.total_local += num_local.sum()
        else:
            accuracies = correct.mean(dim=-1)  # (numk x B)
            avg_index = retrieval_indices.mean(-1)  # (B)
            self.total_local += B * N
        self.sample_acc_sum += accuracies.sum(-1)  # (numk)
        self.sample_avg_index_sum += avg_index.sum()

        if weights is not None:
            self.sample_weighted_acc_sum += (weights[None, :, :] * correct).sum((1, 2))  # (numk)
            self.sample_weighted_avg_index_sum += (weights * retrieval_indices).sum()
            self.weighted_sample_count += B

        self.sample_count += B

    def compute(self):
        metrics = {}

        if self.sample_count > 0:
            if self.avg_local_size_name is not None:
                metrics[self.avg_local_size_name] = self.total_local / self.sample_count
            metrics[self.retrieval_index_name] = self.sample_avg_index_sum / self.sample_count

            topk_accuracies = self.sample_acc_sum / self.sample_count  # (numk)
            metrics.update({name: acc for name, acc in zip(self.acc_names, topk_accuracies)})

            if self.weighted_sample_count > 0:
                metrics[self.weighted_retrieval_index_name] = self.sample_weighted_avg_index_sum / self.weighted_sample_count

                weighted_topk_accuracies = self.sample_weighted_acc_sum / self.weighted_sample_count  # (numk)
                metrics.update({name: acc for name, acc in zip(self.weighted_acc_names, weighted_topk_accuracies)})

        return metrics