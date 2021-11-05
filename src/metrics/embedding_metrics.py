import torch
from pytorch_lightning.metrics import Metric
from torch.nn import functional as F

from analysis.downstream_embeddings import LocalScanDownstreamEmbeddings
from models.pretraining.pretraining_utils import EmbeddingsData


def compute_std_from_sample_vars(sample_means, sample_vars):
    return torch.sqrt(sample_vars.mean() + sample_means.var()) \
        if sample_means.shape[0] > 1 else torch.sqrt(sample_vars.mean())


def compute_avg_dist_to_mean(values):
    """

    :param values: (B x d)
    :return:
    """
    mean = values.mean(dim=0, keepdim=True)
    return F.pairwise_distance(values, mean, p=2).mean()


class GlobalEmbeddingSpaceMetrics:
    """
    -- across sample diversity --
     -> ch_std
     -> dist_emb2emb (dist_emb2emb__batch_estimate)
    """
    def __init__(self, prefix, compute_cov=False):
        self.compute_cov = compute_cov
        self.prefix = prefix + '/'
        self.embeddings = []  # (B x d)
        self.emb2emb_dists__batch_estimate = []  # (1)
        self.uniformity = []  # (1)

    def compute(self, compute_dist_emb2emb=False):
        if len(self.embeddings) == 0:
            return {}
        embeddings = torch.cat(self.embeddings, dim=0)  # (B_total x d)
        emb2emb_dists__batch_estimate = torch.stack(self.emb2emb_dists__batch_estimate, dim=0)  # (N_batch)
        uniformity = torch.stack(self.uniformity, dim=0)  # (N_batch)

        metrics = {
            self.prefix + "ch_std": embeddings.std(dim=0).mean(),
            self.prefix + "uniformity": uniformity.mean().log(),  # see https://arxiv.org/pdf/2005.10242.pdf / https://arxiv.org/pdf/2105.00470.pdf
            self.prefix + "dist_emb2emb__batch_estimate": emb2emb_dists__batch_estimate.mean()  # emb_avg_batch_centroid_dist
        }
        if compute_dist_emb2emb:
            cdist = torch.cdist(embeddings.unsqueeze(0), embeddings.unsqueeze(0), p=2).squeeze(0)  # (B_total x B_total)
            metrics[self.prefix + "dist_emb2emb"] = cdist.mean()
        if self.compute_cov:
            B, d = embeddings.size()
            diag = torch.eye(d, device=embeddings.device).bool()
            cov = embeddings - embeddings.mean(dim=0, keepdim=True)  # (B x d)
            cov = (cov.T @ cov) / (B - 1)  # (d x d)
            metrics[self.prefix + "cov_offdiagonal"] = cov[~diag].pow_(2).sum() / d
        return metrics

    def update(self, embeddings):
        """

        :param embeddings: B x d
        :return:
        """
        if embeddings is None:
            return

        self.embeddings.append(embeddings)

        # ----- avg pairwise distances btw. per-sample centroids (centroid_sample2sample_dist) -----
        cdist = torch.cdist(embeddings.unsqueeze(0), embeddings.unsqueeze(0), p=2).squeeze(0)  # (B x B)
        self.emb2emb_dists__batch_estimate.append(cdist.mean())
        self.uniformity.append(torch.exp(-2 * (cdist ** 2)).mean())  # (1)


class LocalEmbeddingSpaceMetrics:
    """
    Note: __std means the std of that value btw. different samples, the value itself is then the mean of different samples

    -- within sample diversity --
     -> ch_std_per_sample (+ ch_std_per_sample__std)
     -> dist_emb2cent (+ dist_emb2cent_dist__std)
     -> dist_emb2emb (+ dist_emb2emb__weighted)

    -- across sample diversity --
     -> ch_std
     -> ch_std_sample_centroids
     -> dist_cent2cent (dist_cent2cent__batch_estimate)
    """
    def __init__(self, prefix, compute_cov=False):
        self.compute_cov = compute_cov
        self.prefix = prefix + '/'
        self.means = []  # (B x d)
        self.vars = []  # (B x d)
        self.cov_offdiagonal_per_sample = []  # (B)
        self.cov_offdiagonal = []  # (1)
        self.per_sample_uniformity = []  # (B)
        self.emb2cent_dists__means = []  # (1)
        self.emb2cent_dists__vars = []  # (1)
        self.emb2emb_dists = []  # (1)
        self.emb2emb_dists_weighted = []  # (1)
        self.cent2cent_dists__batch_estimate = []  # (1)

    def compute(self, compute_dist_cent2cent=False):
        if len(self.means) == 0:
            return {}
        means = torch.cat(self.means, dim=0)  # (B_total x d)
        vars = torch.cat(self.vars, dim=0)  # (B_total x d)
        emb2cent_dists__means = torch.stack(self.emb2cent_dists__means, dim=0)  # (N_batch)
        emb2cent_dists__vars = torch.stack(self.emb2cent_dists__vars, dim=0)  # (N_batch)
        emb2emb_dists = torch.stack(self.emb2emb_dists, dim=0)  # (N_batch)
        cent2cent_dists__batch_estimate = torch.stack(self.cent2cent_dists__batch_estimate, dim=0)  # N_batch
        per_sample_uniformity = torch.cat(self.per_sample_uniformity, dim=0)  # B_total

        var_sample_centroids = means.var(dim=0) if means.shape[0] > 1 else 0.  # (d)
        ch_std_per_sample = torch.sqrt(vars).mean(1)  # (B_total)

        metrics = {
            self.prefix + "ch_std": torch.sqrt(var_sample_centroids + vars.mean(dim=0)).mean(),  # total_std
            self.prefix + "ch_std_per_sample": ch_std_per_sample.mean(),  # per_sample_std
            self.prefix + "ch_std_per_sample__std": ch_std_per_sample.std(),
            self.prefix + "ch_std_sample_centroids": torch.sqrt(var_sample_centroids).mean(),  # sample_means_std,
            self.prefix + "per_sample_uniformity": per_sample_uniformity.mean(),
            self.prefix + "per_sample_uniformity__std": per_sample_uniformity.std(),
            self.prefix + "dist_emb2cent": emb2cent_dists__means.mean(),  # emb_avg_centroid_dist
            self.prefix + "dist_emb2cent__std": compute_std_from_sample_vars(emb2cent_dists__means, emb2cent_dists__vars),
            self.prefix + "dist_emb2emb": emb2emb_dists.mean(),  # emb_avg_dist,
            self.prefix + "dist_cent2cent__batch_estimate": cent2cent_dists__batch_estimate.mean(),  # emb_avg_batch_centroid_dist
            self.prefix + "dist_cent2dataset": compute_avg_dist_to_mean(means)  # centroid_sample2dataset_dist
        }
        if compute_dist_cent2cent:
            mean_cdist = torch.cdist(means.unsqueeze(0), means.unsqueeze(0), p=2).squeeze(0)  # (B_total x B_total)
            metrics[self.prefix + "dist_cent2cent"] = mean_cdist.mean()
        if self.compute_cov:
            cov_offdiagonal_per_sample = torch.cat(self.cov_offdiagonal_per_sample, dim=0)  # (B_total)
            cov_offdiagonal = torch.stack(self.cov_offdiagonal, dim=0)  # (num_batche)

            B, d = means.size()
            diag = torch.eye(d, device=means.device).bool()
            cov_of_centroids = means - means.mean(dim=0, keepdim=True)  # (B x d)
            cov_of_centroids = (cov_of_centroids.T @ cov_of_centroids) / (B - 1)  # (d x d)

            metrics[self.prefix + "cov_offdiagonal__batch_estimate"] = cov_offdiagonal.mean()
            metrics[self.prefix + "cov_offdiagonal_per_sample"] = cov_offdiagonal_per_sample.mean()
            metrics[self.prefix + "cov_offdiagonal_per_sample__std"] = cov_offdiagonal_per_sample.std()
            metrics[self.prefix + "cov_offdiagonal_sample_centroids"] = cov_of_centroids[~diag].pow_(2).sum() / d
        if len(self.emb2emb_dists_weighted) > 0:
            emb2emb_dists_weighted = torch.stack(self.emb2emb_dists_weighted, dim=0)  # (N_batch)
            metrics[self.prefix + "dist_emb2emb__weighted"] = emb2emb_dists_weighted.mean()  # emb_weighted_avg_dist
        return metrics

    def update(self, embeddings, mask, weights=None):
        """

        :param embeddings: B x N x d
        :param mask: B x N
        :return:
        """
        if embeddings is None:
            return None

        B, N, d = embeddings.size()
        N_local = mask.float().sum(dim=1)  # (B)
        avg_correction_factor = N_local / N  # (B)

        # ----- per-channel mean, variance, and covariance of embeddings (respect masks)-----
        mean = (mask[:, :, None] * embeddings).mean(dim=1) / avg_correction_factor[:, None]  # (B x d)
        expanded_means = mean.unsqueeze(1).expand(B, N, d)  # (B x N x d)
        var_embeddings = embeddings.clone()
        var_embeddings[~mask] = expanded_means[~mask]
        avg_correction_factor_var = (mask.float().sum(dim=1) - 1) / (N - 1)  # (B)
        var = var_embeddings.new_zeros(B, d)
        var_samples_mask = N_local > 1  # (B)
        var[var_samples_mask] = var_embeddings[var_samples_mask].var(dim=1) / avg_correction_factor_var[var_samples_mask, None]  # (B x d)

        if self.compute_cov:
            embeddings_for_cov = embeddings - expanded_means  # (B x N x d)
            # set to zero to ignore
            embeddings_for_cov[~mask, :] = 0.
            # per-sample cov
            cov = torch.bmm(embeddings_for_cov.transpose(-1, -2), embeddings_for_cov) / (N_local[:, None, None] - 1)  # (B x d x d)
            # set cov of samples with single local embeddings to 0
            cov[mask.sum(-1) <= 1] = 0.
            cov_of_centroids = mean - mean.mean(dim=0, keepdim=True)  # (B x d)
            cov_of_centroids = (cov_of_centroids.T @ cov_of_centroids) / (B - 1)  # (d x d)
            cov_total = torch.mean(cov, dim=0) + cov_of_centroids  # (d x d)
            B, d, _ = cov.size()
            diag = torch.eye(d, device=cov.device).bool()
            self.cov_offdiagonal_per_sample.append((cov[:, ~diag].pow_(2).sum(-1) / d))  # (B)
            self.cov_offdiagonal.append((cov_total[~diag].pow_(2).sum() / d))  # (1)
        self.means.append(mean)
        self.vars.append(var)

        # ----- avg distances of embeddings to per-sample centroids (emb2cent) -----
        # (B x N)
        pdists = F.pairwise_distance(embeddings.reshape(B * N, d), expanded_means.reshape(B * N, d), p=2).view(B, N)
        avg_dist_to_mean = (mask * pdists).mean(dim=1) / avg_correction_factor  # (B)
        self.emb2cent_dists__means.append(avg_dist_to_mean.mean())
        self.emb2cent_dists__vars.append(avg_dist_to_mean.var())
        del pdists

        # ----- avg pairwise distances btw. embeddings (emb2emb) -----
        cdist = torch.cdist(embeddings, embeddings)  # (B x N x N)
        cdist = cdist * mask[:, :, None] * mask[:, None, :]

        self.emb2emb_dists.append((
                    (cdist.mean(dim=2) / avg_correction_factor[:, None]).mean(dim=1) / avg_correction_factor).mean())
        if weights is not None:
            weighted_cdist = (weights[:, None, :] * cdist).sum(dim=2)  # (B x N)
            self.emb2emb_dists_weighted.append((weights * weighted_cdist).sum(dim=1).mean())

        uniformity = torch.exp(-2 * (cdist ** 2)) * mask[:, :, None] * mask[:, None, :]  # (B x N x N)
        # (B)
        uniformity = (uniformity.mean(2) / avg_correction_factor[:, None]).mean(dim=1) / avg_correction_factor
        self.per_sample_uniformity.append(uniformity.log())

        del cdist

        # ----- avg pairwise distances btw. per-sample centroids (centroid_sample2sample_dist) -----
        mean_cdist = torch.cdist(mean.unsqueeze(0), mean.unsqueeze(0), p=2).squeeze(0)  # (B x B)
        self.cent2cent_dists__batch_estimate.append(mean_cdist.mean())

        return mean


class GlobalEmbeddingPairMetrics:
    """
    - rmse
    - dist (dist__std) -> avg (normalized) l2 distance
    """
    def __init__(self, prefix):
        self.prefix = prefix + '/'
        self.batch_mse = []  # (1)
        self.dists__means = []  # (1)
        self.dists__vars = []  # (1)

    def compute(self):
        if len(self.batch_mse) == 0:
            return {}
        batch_mse = torch.stack(self.batch_mse, dim=0)  # (N_batch)
        dists__means = torch.stack(self.dists__means, dim=0)  # (N_batch)
        dists__vars = torch.stack(self.dists__vars, dim=0)  # (N_batch)

        return {
            self.prefix + "rmse": torch.sqrt(batch_mse.mean()),
            self.prefix + "dist": dists__means.mean(),
            self.prefix + "dist__std": compute_std_from_sample_vars(dists__means, dists__vars)
        }

    def update(self, embeddings_1, embeddings_2):
        """

        :param embeddings_1: (B x d)
        :param embeddings_2: (B x d)
        :return:
        """
        if embeddings_1 is None or embeddings_2 is None:
            return

        # -- MSE --
        mse = F.mse_loss(embeddings_1, embeddings_2, reduction='none').sum(-1)  # (B)
        self.batch_mse.append(mse.mean())  # (1)
        dists = torch.sqrt(mse)  # (B)
        self.dists__means.append(dists.mean())
        self.dists__vars.append(dists.var())


class LocalEmbeddingPairMetrics:
    """
    avg per-sample distances
    - sample_rmse (sample_rmse__std, sample_rmse__weighted)
    - sample_min_dist, sample_max_dist, sample_std_dist

    """
    def __init__(self, prefix):
        self.prefix = prefix + '/'
        self.sample_rmse__means = []  # (1)
        self.sample_rmse__vars = []  # (1)
        self.sample_rmse__weighted = []  # (1)
        self.sample_min_dist = []  # (1)
        self.sample_max_dist = []  # (1)
        self.sample_std_dist = []  # (1)

    def compute(self):
        if len(self.sample_rmse__means) == 0:
            return {}
        sample_rmse__means = torch.stack(self.sample_rmse__means, dim=0)  # (N_batch)
        sample_rmse__vars = torch.stack(self.sample_rmse__vars, dim=0)  # (N_batch)
        sample_min_dist = torch.stack(self.sample_min_dist, dim=0)  # (N_batch)
        sample_max_dist = torch.stack(self.sample_max_dist, dim=0)  # (N_batch)
        sample_std_dist = torch.stack(self.sample_std_dist, dim=0)  # (N_batch)

        metrics = {
            self.prefix + "sample_rmse": sample_rmse__means.mean(),  # l2att_emb_mse
            self.prefix + "sample_rmse__std": compute_std_from_sample_vars(sample_rmse__means, sample_rmse__vars),
            self.prefix + "sample_min_dist": sample_min_dist.mean(),  # l2att_emb_min_dist
            self.prefix + "sample_max_dist": sample_max_dist.mean(),  # l2att_emb_max_dist
            self.prefix + "sample_std_dist": sample_std_dist.mean(),  # l2att_emb_std_dist
        }
        if len(self.sample_rmse__weighted) > 0:
            sample_rmse__weighted = torch.stack(self.sample_rmse__weighted, dim=0)  # (N_batch)
            metrics[self.prefix + "sample_rmse__weighted"] = sample_rmse__weighted.mean()  # l2att_emb_weighted_mse
        return metrics

    def update(self, embeddings_1, embeddings_2, mask, weights=None):
        """

        :param embeddings_1: (B x N x d)
        :param embeddings_2: (B x N x d)
        :param mask:
        :return:
        """
        if embeddings_1 is None or embeddings_2 is None:
            return

        B, N, d = embeddings_1.size()
        # correction factor (due to mask) when taking the mean over N dimension
        avg_correction_factor = mask.float().sum(dim=1) / N  # (B)

        # -- RMSE --
        mse = F.mse_loss(embeddings_1, embeddings_2, reduction='none').sum(-1) * mask  # (B x N)
        rmse = torch.sqrt(mse.mean(dim=1) / avg_correction_factor)  # (B)
        if weights is not None:
            weighted_rmse = torch.sqrt((weights * mse).sum(dim=1))  # (B)
        self.sample_rmse__means.append(rmse.mean())
        self.sample_rmse__vars.append(rmse.var())
        if weights is not None:
            self.sample_rmse__weighted.append(weighted_rmse.mean())

        # -- min/max/std RMSE --
        dists = torch.sqrt(mse)  # (B x N)
        dists_for_min = dists.clone()  # (B x N)
        dists_for_min[~mask] = float('inf')
        self.sample_min_dist.append(dists_for_min.min(dim=1)[0].mean())
        self.sample_max_dist.append(dists.max(dim=1)[0].mean())
        self.sample_std_dist.append(torch.sqrt(dists.var(dim=1) / avg_correction_factor).mean())


class LocalDistributionPairMetrics:
    """
    Note: ab_emb_centroid_dist computed on distribution avg with GlobalEmbeddingPair
    - dist_emb2emb (dist_emb2emb__weighted)
    - 1NN_emb2emb_2to1, 1NN_emb2emb_1to2
    - assignment_2to2, assignment_1to1 => 1 = assigned to same modality, 0 = assigned to other modality, 0.5 = perfect alignment of modalities
    """
    def __init__(self, prefix):
        self.prefix = prefix + '/'
        self.emb2emb_dists = []  # (1)
        self.emb2emb_dists__weighted = []  # (1)
        self.emb2emb_1NN_2to1 = []  # (1)
        self.emb2emb_1NN_1to2 = []  # (1) a2b_emb_avg_1NN_dist
        self.assignment_2to2 = []  # (1)
        self.assignment_1to1 = []  # (1)

    def compute(self):
        if len(self.emb2emb_dists) == 0:
            return {}
        emb2emb_dists = torch.stack(self.emb2emb_dists, dim=0)  # (N_batch)
        emb2emb_1NN_2to1 = torch.stack(self.emb2emb_1NN_2to1, dim=0)  # (N_batch)
        emb2emb_1NN_1to2 = torch.stack(self.emb2emb_1NN_1to2, dim=0)  # (N_batch)
        assignment_2to2 = torch.stack(self.assignment_2to2, dim=0)  # (N_batch)
        assignment_1to1 = torch.stack(self.assignment_1to1, dim=0)  # (N_batch)

        metrics = {
            self.prefix + "dist_emb2emb": emb2emb_dists.mean(),  # ab_emb_avg_dist
            self.prefix + "1NN_emb2emb_2to1": emb2emb_1NN_2to1.mean(),  # b2a_emb_avg_1NN_dist
            self.prefix + "1NN_emb2emb_1to2": emb2emb_1NN_1to2.mean(),  # a2b_emb_avg_1NN_dist
            self.prefix + "assignment_2to2": assignment_2to2.mean(),
            self.prefix + "assignment_1to1": assignment_1to1.mean()
        }
        if len(self.emb2emb_dists__weighted) > 0:
            emb2emb_dists__weighted = torch.stack(self.emb2emb_dists__weighted, dim=0)  # (N_batch)
            metrics[self.prefix + "dist_emb2emb__weighted"] = emb2emb_dists__weighted.mean()  # ab_emb_weighted_avg_dist
        return metrics

    def update(self,
               embeddings_1, embeddings_2,
               mean_1, mean_2,
               mask_1=None, mask_2=None,
               weights_1=None, weights_2=None):
        if embeddings_1 is None or embeddings_2 is None:
            return

        B, N_1, d = embeddings_1.size()
        _, N_2, _ = embeddings_2.size()
        # correction factor (due to mask) when taking the mean over N dimension
        avg_correction_factor_1 = mask_1.float().sum(dim=1) / N_1  # (B)
        avg_correction_factor_2 = mask_2.float().sum(dim=1) / N_2  # (B)

        # ----- avg pairwise distance btw embeddings ----
        cdist = torch.cdist(embeddings_1, embeddings_2)  # (B x N_1 x N_2)
        cdist = cdist * mask_1[:, :, None] * mask_2[:, None, :]
        self.emb2emb_dists.append((
                (cdist.mean(dim=2) / avg_correction_factor_2[:, None]).mean(dim=1) / avg_correction_factor_1).mean())
        if weights_1 is not None or weights_2 is not None:
            if weights_2 is not None:
                weighted_cdist = (weights_2[:, None, :] * cdist).sum(dim=2)  # (B x N_1)
            else:
                weighted_cdist = cdist.mean(dim=2) / avg_correction_factor_2[:, None]  # (B x N_1)
            if weights_1 is not None:
                weighted_cdist = (weights_1 * weighted_cdist).sum(dim=1)  # (B)
            else:
                weighted_cdist = cdist.mean(dim=1) / avg_correction_factor_1  # (B)
            self.emb2emb_dists__weighted.append(weighted_cdist.mean())

        # ----- 1NN dist -----
        cdist_b2a = cdist
        cdist_a2b = cdist.clone().transpose(-1, -2)  # (B x N_2 x N_1)
        cdist_b2a.masked_fill(~mask_2[:, None, :], float('inf'))  # inf for N2 (ignore in min operation)
        min_dist_b2a = cdist_b2a.min(dim=2)[0]  # (B x N_1)
        cdist_a2b.masked_fill(~mask_1[:, None, :], float('inf'))  # inf for N1 (ignore in min operation)
        min_dist_a2b = cdist_a2b.min(dim=2)[0]  # (B x N_2)

        self.emb2emb_1NN_2to1.append((min_dist_b2a.mean(dim=1) / avg_correction_factor_1).mean())
        self.emb2emb_1NN_1to2.append((min_dist_a2b.mean(dim=1) / avg_correction_factor_2).mean())

        #if weights_1 is not None:
        #    batch_means['b2a_emb_weighted_avg_1NN_dist'] = (weights_1 * min_dist_b2a).sum(dim=1).mean()
        #if weights_2 is not None:
        #    batch_means['a2b_emb_weighted_avg_1NN_dist'] = (weights_2 * min_dist_a2b).sum(dim=1).mean()

        # min_dist_b2a[~mask_1] = float('inf')  # inf for N1 (ignore in min operation)
        # batch_means['ab_emb_min_dist'] = min_dist_b2a.min(dim=1)[0].mean()

        # ----- centroid assignment: avg dist to other centroid / avg dist to same centroid TODO -----
        expanded_means_b2a = mean_1.unsqueeze(1).expand(B, N_2, d)
        dists_b2a = F.pairwise_distance(embeddings_2.reshape(B * N_2, d), expanded_means_b2a.reshape(B * N_2, d), p=2)\
            .view(B, N_2)
        scores_b2a = torch.reciprocal(dists_b2a ** 2)
        expanded_means_b2b = mean_2.unsqueeze(1).expand(B, N_2, d)
        dists_b2b = F.pairwise_distance(embeddings_2.reshape(B * N_2, d), expanded_means_b2b.reshape(B * N_2, d), p=2) \
            .view(B, N_2)
        scores_b2b = torch.reciprocal(dists_b2b ** 2)
        assignment_b2b = (scores_b2b / (scores_b2b + scores_b2a))  # (B x N_2)
        self.assignment_2to2.append(((mask_2 * assignment_b2b).mean(dim=1) / avg_correction_factor_2).mean())

        expanded_means_a2b = mean_2.unsqueeze(1).expand(B, N_1, d)
        dists_a2b = F.pairwise_distance(embeddings_1.reshape(B * N_1, d), expanded_means_a2b.reshape(B * N_1, d), p=2) \
            .view(B, N_1)
        scores_a2b = torch.reciprocal(dists_a2b ** 2)
        expanded_means_a2a = mean_1.unsqueeze(1).expand(B, N_1, d)
        dists_a2a = F.pairwise_distance(embeddings_1.reshape(B * N_1, d), expanded_means_a2a.reshape(B * N_1, d), p=2) \
            .view(B, N_1)
        scores_a2a = torch.reciprocal(dists_a2a ** 2)
        assignment_a2a = (scores_a2a / (scores_a2a + scores_a2b))  # (B x N_1)
        self.assignment_1to1.append(((mask_1 * assignment_a2a).mean(dim=1) / avg_correction_factor_1).mean())


class EmbeddingMetrics(Metric):
    def __init__(self, compute_cov=False):
        super(EmbeddingMetrics, self).__init__(compute_on_step=False)
        self.compute_cov = compute_cov
        self._init()

    def _init(self):
        # single space metrics
        self.yl_a_metrics = LocalEmbeddingSpaceMetrics('yl_a', compute_cov=self.compute_cov)
        self.yl_b_metrics = LocalEmbeddingSpaceMetrics('yl_b', compute_cov=self.compute_cov)

        self.zl_a_metrics = LocalEmbeddingSpaceMetrics('zl_a', compute_cov=self.compute_cov)
        self.zl_b_metrics = LocalEmbeddingSpaceMetrics('zl_b', compute_cov=self.compute_cov)
        self.zl_a__avg_metrics = GlobalEmbeddingSpaceMetrics('zl_a_avg', compute_cov=self.compute_cov)
        self.zl_b__avg_metrics = GlobalEmbeddingSpaceMetrics('zl_b_avg', compute_cov=self.compute_cov)
        self.zl_a_with_b_metrics = LocalEmbeddingSpaceMetrics('zl_a&b', compute_cov=self.compute_cov)

        self.yg_a_metrics = GlobalEmbeddingSpaceMetrics('yg_a', compute_cov=self.compute_cov)
        self.yg_b_metrics = GlobalEmbeddingSpaceMetrics('yg_b', compute_cov=self.compute_cov)
        self.zg_a_metrics = GlobalEmbeddingSpaceMetrics('zg_a', compute_cov=self.compute_cov)
        self.zg_b_metrics = GlobalEmbeddingSpaceMetrics('zg_b', compute_cov=self.compute_cov)

        # two space metrics
        self.zl_ab_metrics = LocalDistributionPairMetrics('zl_a||zl_b')
        self.zl_ab_avg_metrics = GlobalEmbeddingPairMetrics('zl_a_avg;zl_b_avg')

        # alignment metrics
        self.zl_b2a_metrics = LocalEmbeddingPairMetrics('zl_a;zl_b2a')
        self.zl_a2b_metrics = LocalEmbeddingPairMetrics('zl_b;zl_a2b')

        self.zg_ab_metrics = GlobalEmbeddingPairMetrics('zg_a;zg_b')

    def update(self, embeddings: EmbeddingsData):
        embeddings = embeddings.detach().normalize()
        mask_a = prepare_mask(embeddings.zl_a, embeddings.mask_a)
        mask_b = prepare_mask(embeddings.zl_b, embeddings.mask_b)
        self.yl_a_metrics.update(embeddings.yl_a, mask=mask_a)
        self.yl_b_metrics.update(embeddings.yl_b, mask=mask_b)
        mean_zl_a = self.zl_a_metrics.update(embeddings.zl_a, mask=mask_a, weights=embeddings.weights_a)
        mean_zl_b = self.zl_b_metrics.update(embeddings.zl_b, mask=mask_b, weights=embeddings.weights_b)
        self.zl_a__avg_metrics.update(mean_zl_a)
        self.zl_b__avg_metrics.update(mean_zl_b)
        if embeddings.zl_a is not None and embeddings.zl_b is not None:
            self.zl_a_with_b_metrics.update(torch.cat([embeddings.zl_a, embeddings.zl_b], dim=1),
                                            mask=torch.cat([mask_a, mask_b], dim=1))

        self.yg_a_metrics.update(embeddings.yg_a)
        self.yg_b_metrics.update(embeddings.yg_b)
        self.zg_a_metrics.update(embeddings.zg_a)
        self.zg_b_metrics.update(embeddings.zg_b)

        self.zl_ab_metrics.update(embeddings.zl_a, embeddings.zl_b,
                                  mean_1=mean_zl_a, mean_2=mean_zl_b,
                                  mask_1=mask_a, mask_2=mask_b,
                                  weights_1=embeddings.weights_a, weights_2=embeddings.weights_b)
        self.zl_ab_avg_metrics.update(mean_zl_a, mean_zl_b)

        self.zl_b2a_metrics.update(embeddings.zl_a, embeddings.zl_b2a, mask=mask_a, weights=embeddings.weights_a)
        self.zl_a2b_metrics.update(embeddings.zl_b, embeddings.zl_a2b, mask=mask_b, weights=embeddings.weights_b)

        self.zg_ab_metrics.update(embeddings.zg_a, embeddings.zg_b)

    def compute(self):
        metrics = {}
        with torch.no_grad():
            metrics.update(self.yl_a_metrics.compute())
            metrics.update(self.yl_b_metrics.compute())
            metrics.update(self.zl_a_metrics.compute())
            metrics.update(self.zl_b_metrics.compute())
            metrics.update(self.zl_a__avg_metrics.compute())
            metrics.update(self.zl_b__avg_metrics.compute())
            metrics.update(self.zl_a_with_b_metrics.compute())
            metrics.update(self.yg_a_metrics.compute())
            metrics.update(self.yg_b_metrics.compute())
            metrics.update(self.zg_a_metrics.compute())
            metrics.update(self.zg_b_metrics.compute())

            metrics.update(self.zl_ab_metrics.compute())
            metrics.update(self.zl_ab_avg_metrics.compute())
            metrics.update(self.zl_b2a_metrics.compute())
            metrics.update(self.zl_a2b_metrics.compute())
            metrics.update(self.zg_ab_metrics.compute())

        return metrics

    def reset(self):
        self._init()


class DownstreamEmbeddingMetrics(Metric):
    def __init__(self, compute_cov=False):
        super(DownstreamEmbeddingMetrics, self).__init__(compute_on_step=False)
        self.compute_cov = compute_cov
        self._init()

    def _init(self):
        # single space metrics
        self.yl_a_metrics = LocalEmbeddingSpaceMetrics('yl_a', compute_cov=self.compute_cov)
        self.yg_a_metrics = GlobalEmbeddingSpaceMetrics('yg_a', compute_cov=self.compute_cov)

    def update(self, embeddings: LocalScanDownstreamEmbeddings):
        embeddings = embeddings.detach().normalize()
        self.yl_a_metrics.update(embeddings.yl_a, mask=prepare_mask(embeddings.yl_a, mask=None))
        self.yg_a_metrics.update(embeddings.yg_a)

    def compute(self):
        metrics = {}
        with torch.no_grad():
            metrics.update(self.yl_a_metrics.compute())
            metrics.update(self.yg_a_metrics.compute())

        return metrics

    def reset(self):
        self._init()


def prepare_mask(local, mask):
    if local is None:
        return None
    if mask is None:
        B, N, _ = local.size()
        mask = local.new_ones((B, N), dtype=bool)
    else:
        mask = mask.binary_mask
    return mask
