from collections import defaultdict

import torch
from torch.nn import functional as F
from torchmetrics import Metric


class CrossAttentionMetrics(Metric):
    def __init__(self):
        super(CrossAttentionMetrics, self).__init__(compute_on_step=False)
        self.batch_means = defaultdict(list)
        self.batch_vars = defaultdict(list)

    def update(self, probs_b2a, probs_a2b, mask_a=None, mask_b=None):
        """
        - mean entropy b2a and a2b => how equally spread is the attention on average
        - mean of per-sample std of entropy b2a and a2b => how differently is the entropy of the attention between different local regions
        - mean of per-sample mse between probs total probs => how differently are attentions distributed between different regions
        - mean and std of per sample entropy a and b
        :param probs_b2a: (B x N_a x N_b)
        :param probs_a2b: (B x N_b x N_a)
        :return:
        """
        if probs_b2a is not None:
            eps = torch.finfo(probs_b2a.dtype).eps
            device = probs_b2a.device
            B, N_a, N_b = probs_b2a.size()
            probs_b2a = probs_b2a.detach()
        elif probs_a2b is not None:
            eps = torch.finfo(probs_a2b.dtype).eps
            device = probs_a2b.device
            B, N_b, N_a = probs_a2b.size()
            probs_a2b = probs_a2b.detach()
        else:
            return

        mask_a = self._prepare_mask(mask_a, device, B, N_a)
        avg_correction_factor_a = mask_a.float().sum(dim=1) / N_a  # (B)
        mask_b = self._prepare_mask(mask_b, device, B, N_b)
        avg_correction_factor_b = mask_b.float().sum(dim=1) / N_b  # (B)

        batch_means = {}
        batch_vars = {}
        if probs_b2a is not None:
            mean_entropy_b2a, std_entropy_b2a, prob_b2a_mse_to_mean, entropy_b, entropy_b_var = \
                self._compute_individual_stats(probs_b2a, mask_a, mask_b, avg_correction_factor_a,
                                               avg_correction_factor_b, eps)
            batch_means.update({
                'b2a/mean_entropy': mean_entropy_b2a,   # b2a_att_mean_entropy
                'b2a/std_entropy': std_entropy_b2a,  # b2a_att_std_entropy
                'b2a/dist_to_mean': prob_b2a_mse_to_mean,  # b2a_att_mse_to_mean
                'b/entropy': entropy_b  # b_att_entropy
            })
            batch_vars['b/entropy'] = entropy_b_var
        if probs_a2b is not None:
            mean_entropy_a2b, std_entropy_a2b, prob_a2b_mse_to_mean, entropy_a, entropy_a_var = \
                self._compute_individual_stats(probs_a2b, mask_b, mask_a, avg_correction_factor_b, avg_correction_factor_a, eps)
            batch_means.update({
                'a2b/mean_entropy': mean_entropy_a2b,  # a2b_att_mean_entropy
                'a2b/std_entropy': std_entropy_a2b,  # a2b_att_std_entropy
                'a2b/dist_to_mean': prob_a2b_mse_to_mean,  # a2b_att_mse_to_mean
                'a/entropy': entropy_a,  # a_att_entropy
            })
            batch_vars['a/entropy'] = entropy_a_var

        self._add_batch_results(batch_means, batch_vars)

    def compute(self):
        metrics = {}
        with torch.no_grad():
            for key, means in self.batch_means.items():
                means = torch.stack(means, dim=0)
                metrics[key] = means.mean()

                if key in self.batch_vars:
                    vars = torch.stack(self.batch_vars[key], dim=0)
                    std = torch.sqrt(vars.mean() + means.var()) if len(vars) > 1 else torch.sqrt(vars.mean())
                    metrics[key + '__std'] = std

        return metrics

    def _compute_individual_stats(self, probs_b2a, mask_a, mask_b, avg_correction_factor_a, avg_correction_factor_b, eps):
        if probs_b2a is None:
            return None, None, None, None, None

        B, N_a, N_b = probs_b2a.size()
        probs_b2a = probs_b2a * mask_a[:, :, None] * mask_b[:, None, :]

        entropies_b2a = -((probs_b2a * probs_b2a.clamp_min(eps).log())).sum(dim=-1)  # (B x N_a)
        mean_entropy_b2a = entropies_b2a.mean(dim=-1) / avg_correction_factor_a  # (B)
        std_entropy_b2a = torch.sqrt(entropies_b2a.var(dim=1) / avg_correction_factor_a).mean()  # (B)

        probs_b = ((probs_b2a).mean(dim=1) / avg_correction_factor_a[:, None])  # (B x N_b)
        entropy_b = -(probs_b * probs_b.clamp_min(eps).log()).sum(dim=-1)  # (B)

        mean_probs = probs_b.unsqueeze(1).expand(B, N_a, N_b)  # (B x N_a x N_b)
        # (B x N_a)
        prob_pdists = F.pairwise_distance(probs_b2a.reshape(B*N_a, N_b), mean_probs.reshape(B*N_a, N_b), p=2).view(B, N_a)
        prob_pdists = prob_pdists * mask_a
        prob_b2a_mse_to_mean = prob_pdists.mean(dim=1) / avg_correction_factor_a  # (B)

        return mean_entropy_b2a.mean(), std_entropy_b2a.mean(), prob_b2a_mse_to_mean.mean(), entropy_b.mean(), entropy_b.var()

    def _add_batch_results(self, batch_means: dict, batch_vars: dict):
        for key, mean in batch_means.items():
            if mean is None:
                continue
            self.batch_means[key].append(mean)

            if key in batch_vars:
                self.batch_vars[key].append(batch_vars[key])

    def _prepare_mask(self, mask, device, B, N):
        if mask is None:
            mask = torch.ones(B, N, dtype=bool, device=device)
        else:
            mask = mask.binary_mask
        return mask

    def reset(self):
        self.batch_means = defaultdict(list)
        self.batch_vars = defaultdict(list)
