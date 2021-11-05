import logging
import os
import sys
import warnings
from collections import OrderedDict
from typing import Optional, Any

import torch
from pytorch_lightning import Trainer
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts

# --- Fix for pl_bolts import ---
import torch._six

from models.objectives.global_alignment import get_global_alignment_objective
from models.objectives.local_alignment import get_local_alignment_objective
from models.pretraining.pretraining_utils import load_encoder, BiModalBatchCollator, BiModalTransform, \
    EmbeddingsData, AttentionData, VAL_INPUT_PREFIX, ModelInputData, \
    BiModalModelConfig, ModalityEmbeddingsData
from metrics.bimodal_alignment_metrics import BiModalAlignmentMetrics
from common.config_utils import prepare_config

torch._six.PY3 = sys.version_info[0] == 3
# --- end fix ---
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.utils.data import DataLoader

from data.dataloading_utils import DatasetTransformWrapper, load_dataset
from models.components.attention import CrossAttention, CrossAttentionOutput
from models.components.utils import EncoderOutput
from models.components.fc import MLP, SequenceMLP

log = logging.getLogger(__name__)


class StepsPerEpochMixin:
    @property
    def steps_per_epoch(self) -> int:
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return batches // effective_accum if batches % effective_accum == 0 else (batches // effective_accum) + 1

    @property
    def total_steps(self):
        return self.steps_per_epoch * self.config.max_epochs


class BiModalModelRepresentationLearner(StepsPerEpochMixin, pl.LightningModule):
    def __init__(self,
                 config: BiModalModelConfig,
                 dataset: Optional[str] = None, num_workers=4, do_load_dataset=True):
        super(BiModalModelRepresentationLearner, self).__init__()
        config: BiModalModelConfig = prepare_config(config, BiModalModelConfig, log)

        self.save_hyperparameters("config", "dataset", "num_workers")
        self.config = config
        self.learning_rate = config.learning_rate
        self.dataset = dataset
        self.num_workers = num_workers
        self.run_dir = None
        self.predictions_sub_folder = None

        if dataset is not None and do_load_dataset:
            # try to get dataset stats which may be required for the encoders:
            train_dataset = load_dataset(self.dataset)['train']
            dataset_stats = train_dataset.stats
        else:
            dataset_stats = {}

        # ----- Modality Models (Encoders) -----
        if self.config.d_zg is None:
            self.config.d_zg = self.config.d_z
        self.input_name_a = self.config.encoder_a.modality
        self.input_name_b = self.config.encoder_b.modality
        self.model_a = ModalityModel(load_encoder(self.config.encoder_a,
                                                  dataset_stats=dataset_stats.get(self.input_name_a, None)),
                                     d_z=self.config.d_z, d_zg=self.config.d_zg, d_hidden=self.config.d_hidden,
                                     norm=self.config.projection_norm, projection_dropout_prob=self.config.projection_dropout_prob)
        self.model_b = ModalityModel(load_encoder(self.config.encoder_b,
                                                  dataset_stats=dataset_stats.get(self.input_name_b, None)),
                                     d_z=self.config.d_z, d_zg=self.config.d_zg, d_hidden=self.config.d_hidden,
                                     norm=self.config.projection_norm, projection_dropout_prob=self.config.projection_dropout_prob)

        # ----- Alignment Model -----
        self.ll_attention = CrossAttention(self.config.d_z,
                                           similarity=self.config.attention_sim,
                                           project_keys=self.config.project_attention_keys,
                                           project_values=self.config.project_attention_values,
                                           project_output=self.config.project_attention_outputs,
                                           output_norm=self.config.attention_norm,
                                           dropout_prob=self.config.attended_dropout_prob,
                                           attention_probs_dropout_prob=self.config.attention_prob_dropout_prob,
                                           symmetric=self.config.symmetric_attention,
                                           temperature=self.config.attention_temperature,
                                           temperature_trainable=self.config.attention_temperature_trainable,)

        # ----- Local weights (for weighting local losses and l2g aggregator) -----
        self.lweights_a = LocalWeightsModule(computation_mode=self.config.l_weights_a,
                                             stop_weights_grad=self.config.l_weights_stop_grad)
        self.lweights_b = LocalWeightsModule(computation_mode=self.config.l_weights_b,
                                             stop_weights_grad=self.config.l_weights_stop_grad)

        # ----- Loss -----
        self.loss_module = LossModule(config,
                                      self.model_a.encoder.max_region_size, self.model_b.encoder.max_region_size)

        # ----- Dataset and transforms -----
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.transform = BiModalTransform(self.input_name_a, self.input_name_b,
                                          self.model_a.encoder.transform, self.model_b.encoder.transform)
        self.val_transform = BiModalTransform(self.input_name_a, self.input_name_b,
                                              self.model_a.encoder.transform, self.model_b.encoder.transform,
                                              self.model_a.encoder.val_transform, self.model_b.encoder.val_transform,
                                              val=True, augment_on_validation=config.augment_on_validation)

        # ----- Metrics -----
        self.val_metrics = BiModalAlignmentMetrics(compute_retrieval_metrics=config.compute_retrieval_metrics,
                                                   compute_embedding_stats=config.compute_embedding_stats,
                                                   compute_attention_stats=config.compute_attention_stats)
        if config.compute_metrics_for_train:
            self.train_metrics = BiModalAlignmentMetrics(compute_retrieval_metrics=config.compute_retrieval_metrics,
                                                         compute_embedding_stats=config.compute_embedding_stats,
                                                         compute_attention_stats=config.compute_attention_stats)

        # ----- Load weights from other model -----
        if config.load_checkpoint_path is not None:
            loaded_model = BiModalModelRepresentationLearner.load_from_checkpoint(config.load_checkpoint_path, strict=True)
            for component_name in config.component_loaded_from_checkpoint:
                self_component = getattr(self, component_name)
                loaded_component = getattr(loaded_model, component_name)
                self_component.load_state_dict(loaded_component.state_dict(), strict=True)
                log.info(f'Loaded weights for "{component_name}" from checkpoint {config.load_checkpoint_path}')

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        ignores = {'chexpert_bin_scan', 'chexpert_bin_report', 'chexpert_bin_joint'}
        new_state_dict = []
        for key, value in state_dict.items():
            prefix = key.split('.')[0]
            if prefix in ignores:
                continue
            else:
                new_state_dict.append((key, value))
        super(BiModalModelRepresentationLearner, self).load_state_dict(state_dict=OrderedDict(new_state_dict),
                                                                       strict=strict)

    def shared_step(self, batch, batch_idx,
                    return_attention_probs=False,
                    return_embeddings=False):
        losses = {}

        # ----- Encode both modalities (including projections) -----
        x_a, x_b = batch[self.input_name_a], batch[self.input_name_b]
        modality_a: ModalityEmbeddingsData = self.model_a(x_a)
        modality_b: ModalityEmbeddingsData = self.model_b(x_b)

        embeddings = EmbeddingsData.from_modalities(modality_a, modality_b)

        attention_results = None
        if self.loss_module.compute_local:
            # ----- l-l attention (alignment model) -----
            attention_results = self.ll_attention(embeddings.zl_a, embeddings.zl_b,
                                                  mask_a=embeddings.mask_a, mask_b=embeddings.mask_b,
                                                  a2b=self.loss_module.compute_attended_b,
                                                  b2a=self.loss_module.compute_attended_a)

            # ------ compute local weigths -----
            # (N x N_a)
            l_weights_a = self.lweights_a(modality_a)
            # (B x N_b)
            l_weights_b = self.lweights_b(modality_b)

            embeddings.zl_b2a = attention_results.attended_b2a
            embeddings.zl_a2b = attention_results.attended_a2b
            embeddings.weights_a = l_weights_a
            embeddings.weights_b = l_weights_b

        # ----- loss -----
        total_loss, metrics = self.loss_module(embeddings,
                                               attention_results,
                                               losses)

        # ----- return losses and attention probs (if required) -----
        attention_probs = AttentionData.from_attention_output(attention_results).detach() \
            if return_attention_probs else None
        if not return_embeddings:
            embeddings = None

        return total_loss, metrics, attention_probs, embeddings

    def training_step(self, batch, batch_idx):
        compute_stats = self.config.compute_metrics_for_train \
                        and (self.config.compute_embedding_stats or self.config.compute_attention_stats)

        total_loss, metrics, attention_probs, embeddings = \
            self.shared_step(
                batch,
                batch_idx,
                return_attention_probs=compute_stats,
                return_embeddings=compute_stats
            )

        self.log_dict({f'train/{name}': metric for name, metric in metrics.items()})
        self.log('train/total_loss', total_loss, prog_bar=True)
        if self.config.compute_metrics_for_train:
            self.train_metrics(embeddings, attention_probs)

        return total_loss

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        for i_optim, optim in enumerate(self.trainer.optimizers):
            for i, group in enumerate(optim.param_groups):
                if len(optim.param_groups) == 1:
                    self.log(f'optim_{i_optim}/lr', group['lr'], prog_bar=True, on_step=True, on_epoch=False)
                else:
                    self.log(f'optim_{i_optim}/lr_{i}', group['lr'], on_step=True, on_epoch=False)

    def on_train_epoch_end(self, outputs=None):
        if self.config.compute_metrics_for_train:
            self.log_dict({f'train/{name}': metric for name, metric in self.train_metrics.compute().items()})
            self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        compute_stats = self.config.compute_embedding_stats or self.config.compute_attention_stats

        total_loss, metrics, attention_probs, embeddings = \
            self.shared_step(batch, batch_idx,
                             return_attention_probs=compute_stats,
                             return_embeddings=compute_stats)

        self.log_dict({f'val/{name}': metric for name, metric in metrics.items()})
        self.log('val/total_loss', total_loss)
        self.val_metrics(embeddings, attention_probs)

        return total_loss

    def on_validation_epoch_end(self):
        self.log_dict({f'val/{name}': metric for name, metric in self.val_metrics.compute().items()})
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        compute_stats = self.config.compute_embedding_stats or self.config.compute_attention_stats

        total_loss, metrics, attention_probs, embeddings = \
            self.shared_step(batch, batch_idx,
                             return_attention_probs=compute_stats,
                             return_embeddings=compute_stats)

        self.log_dict({f'test/{name}': metric for name, metric in metrics.items()})
        self.log('test/total_loss', total_loss)
        self.val_metrics(embeddings, attention_probs)

        return total_loss

    def on_test_epoch_end(self):
        self.log_dict({f'test/{name}': metric for name, metric in self.val_metrics.compute().items()})
        self.val_metrics.reset()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        assert self.run_dir is not None and self.predictions_sub_folder is not None
        x_a = self.model_a.encoder.get_x(**batch[self.input_name_a])
        x_b = self.model_b.encoder.get_x(**batch[self.input_name_b])
        inputs = ModelInputData(
            patient_id=batch['patient_id'], study_id=batch['study_id'],
            x_a=x_a.to_dict(), x_b=x_b.to_dict(),
            chexpert_bin_labels=batch.get('chexpert_bin_labels')
        )
        _, _, attention_probs, embeddings = \
            self.shared_step(batch, batch_idx,
                             return_attention_probs=True,
                             return_embeddings=True)

        predictions_dir = os.path.join(self.run_dir, 'predictions', self.predictions_sub_folder)
        inputs.save(predictions_dir, batch_idx)
        embeddings.save(predictions_dir, batch_idx)
        attention_probs.save(predictions_dir, batch_idx)

    def configure_optimizers(self):
        if self.config.optimizer == 'Adam':
            optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()),
                             lr=self.learning_rate, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'AdamW':
            optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                              lr=self.learning_rate, weight_decay=self.config.weight_decay)
        else:
            raise ValueError(self.config.optimizer)
        schedulers = []
        assert all(schedulder in ('linear_warmup_cosine_annealing', 'reduce_on_plateau',
                                  'cosine_annealing_per_epoch', 'cosine_annealing_warm_restarts')
                   for schedulder in self.config.lr_scheduler)
        if 'linear_warmup_cosine_annealing' in self.config.lr_scheduler:
            schedulers.append({
                'scheduler': LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=self.config.warmup_steps,
                    max_epochs=self.total_steps,
                    warmup_start_lr=self.learning_rate / 100
            ),
                'interval': 'step',
                'frequency': 1
            })
            log.info(f'Using LR scheduler linear_warmup_cosine_annealing')
        if 'cosine_annealing_per_epoch' in self.config.lr_scheduler:
            schedulers.append({
                'scheduler': CosineAnnealingLR(
                    optimizer,
                    T_max= self.steps_per_epoch,
                    eta_min=0),
                'interval': 'step',
                'frequency': 1
            })
            log.info(f'Using LR scheduler cosine_annealing_per_epoch')
        if 'cosine_annealing_warm_restarts' in self.config.lr_scheduler:
            schedulers.append({
                'scheduler': CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=self.steps_per_epoch,
                    eta_min=0),
                'interval': 'step',
                'frequency': 1
            })
            log.info(f'Using LR scheduler cosine_annealing_warm_restarts')
        if 'reduce_on_plateau' in self.config.lr_scheduler:
            schedulers.append({
                'scheduler': ReduceLROnPlateau(optimizer, patience=self.config.lr_reduce_patience, factor=self.config.lr_reduce_factor),
                'monitor': 'val/total_loss',
                'interval': 'epoch',
                'frequency': 1
            })
            log.info(f'Using LR scheduler reduce_on_plateau')

        optimizers = [optimizer]
        return optimizers, schedulers

    def setup(self, stage: str = None):
        dataset = load_dataset(self.dataset)
        self.train_dataset = dataset['train']
        self.val_dataset = dataset['validation']
        self.test_dataset = dataset['test']

    def batch_collator(self, val=False):
        return BiModalBatchCollator(self.input_name_a, self.input_name_b,
                                    self.model_a.encoder.batch_collator, self.model_b.encoder.batch_collator,
                                    val=val, augment_on_validation=self.config.augment_on_validation)

    def train_dataloader(self):
        dataloader = DataLoader(DatasetTransformWrapper(self.train_dataset, self.transform),
                                batch_size=self.config.batch_size, collate_fn=self.batch_collator(),
                                num_workers=self.num_workers, pin_memory=True, drop_last=True, shuffle=True)
        log.info(f'Training dataset contains {len(self.train_dataset)} samples, '
                 f'iterated using {len(dataloader)} batches of size {self.config.batch_size}')
        return dataloader

    def val_dataloader(self):
        return DataLoader(DatasetTransformWrapper(self.val_dataset, self.val_transform),
                          batch_size=self.config.batch_size, collate_fn=self.batch_collator(val=True),
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(DatasetTransformWrapper(self.test_dataset, self.val_transform),
                          batch_size=self.config.batch_size, collate_fn=self.batch_collator(val=True),
                          num_workers=self.num_workers, pin_memory=True)

    def get_dataloader(self, dataset='test'):
        if dataset == 'test':
            return self.test_dataloader()
        elif dataset == 'validation':
            return self.val_dataloader()
        elif dataset == 'train':
            return self.train_dataloader()
        else:
            raise ValueError(dataset)

    def get_representation_dim(self, modality: str, representation_type='global'):
        assert representation_type in ('global', 'local')
        if representation_type == 'global':
            assert modality in ('a', 'b', 'a&b')

            if modality == 'a&b':
                return self.model_a.encoder.d_g + self.model_b.encoder.d_g
            elif modality == 'a':
                return self.model_a.encoder.d_g
            else:
                return self.model_b.encoder.d_g
        elif representation_type == 'local':
            assert modality in ('a', 'b')

            if modality == 'a':
                return self.model_a.encoder.d_l
            else:
                return self.model_b.encoder.d_l

    def get_representation(self, batch, modality: str, representation_type='global', val=False):
        assert representation_type in ('global', 'local')
        val_prefix = VAL_INPUT_PREFIX if val else ''
        if representation_type == 'global':
            assert modality in ('a', 'b', 'a&b')
            if modality == 'a&b':
                encoded_a = self.model_a.encoder(**batch[val_prefix + self.input_name_a],
                                                 return_local=False, return_global=True).global_features
                encoded_b = self.model_b.encoder(**batch[val_prefix + self.input_name_b],
                                                 return_local=False, return_global=True).global_features
                return torch.cat([encoded_a, encoded_b], dim=-1)
            elif modality == 'a':
                return self.model_a.encoder(**batch[val_prefix + self.input_name_a],
                                            return_local=False, return_global=True).global_features
            else:
                return self.model_b.encoder(**batch[val_prefix + self.input_name_b],
                                            return_local=False, return_global=True).global_features
        else:
            assert modality in ('a', 'b')
            if modality == 'a':
                encoded = self.model_a.encoder(**batch[val_prefix + self.input_name_a],
                                               return_local=True, return_global=False)
            else:
                encoded = self.model_b.encoder(**batch[val_prefix + self.input_name_b],
                                               return_local=True, return_global=False)
            return encoded.local_features, encoded.local_structure_size


class ModalityModel(nn.Module):
    def __init__(self, encoder: nn.Module, d_z: int, d_zg: int, d_hidden: int,
                 norm='batch', projection_dropout_prob: float = 0.):
        super(ModalityModel, self).__init__()
        self.encoder = encoder
        self.project_local = SequenceMLP(encoder.d_l, d_z, d_hidden=d_hidden,
                                         norm=norm, norm_before_act=True,
                                         dropout_prob=projection_dropout_prob)
        self.project_global = MLP(encoder.d_g, d_zg, d_hidden=d_hidden,
                                  norm=norm, norm_before_act=True,
                                  dropout_prob=projection_dropout_prob)
        self.d_z = d_z
        self.d_zg = d_zg

    def forward(self, x, model_mode='online', compute_local=True, compute_global=True):
        assert model_mode in ('online', 'frozen')

        if model_mode == 'frozen':
            with torch.no_grad:
                return self._do_forward(x, compute_local=compute_local, compute_global=compute_global).detach()
        elif model_mode == 'online':
            return self._do_forward(x, compute_local=compute_local, compute_global=compute_global)

    def _do_forward(self, x, compute_local=True, compute_global=True, return_encoder_output=False) -> ModalityEmbeddingsData:
        encoded: EncoderOutput = self.encoder(**x, return_local=compute_local, return_global=compute_global)
        # (B x N x d_yl), (B x d_yg)
        yl, yg = encoded.local_features, encoded.global_features
        zl = self.project_local(yl) if compute_local else None  # (B x N x d)
        zg = self.project_global(yg) if compute_global else None  # (B x d)

        return ModalityEmbeddingsData(
            yg=yg, yl=yl, zg=zg, zl=zl,
            mask=encoded.local_mask, weights=encoded.local_weights,
            local_size=encoded.local_structure_size
        )


class LossModule(nn.Module):
    def __init__(self, config: BiModalModelConfig, max_region_size_a, max_region_size_b):
        super(LossModule, self).__init__()
        self.config = config

        # ----- Local Alignment Losses -----
        ll_alignments_a = []
        for i in range(len(self.config.ll_alignments_a)):
            assert str(i) in self.config.ll_alignments_a
            ll_alignment_a_conf = self.config.ll_alignments_a[str(i)]
            ll_alignment_a = get_local_alignment_objective(
                ll_alignment_a_conf,
                max_structure_size=max_region_size_a,
                d_z=self.config.d_z)
            ll_alignments_a.append(ll_alignment_a)
            log.info(f'Instantiated objective {type(ll_alignment_a)} for ll_alignment_a number {i}')
        self.ll_alignments_a = nn.ModuleList(ll_alignments_a)

        ll_alignments_b = []
        for i in range(len(self.config.ll_alignments_b)):
            assert str(i) in self.config.ll_alignments_b
            ll_alignment_b_conf = self.config.ll_alignments_b[str(i)]
            ll_alignment_b = get_local_alignment_objective(
                ll_alignment_b_conf,
                max_structure_size=max_region_size_b,
                d_z=self.config.d_z)
            ll_alignments_b.append(ll_alignment_b)
            log.info(f'Instantiated objective {type(ll_alignment_b)} for ll_alignment_b number {i}')
        self.ll_alignments_b = nn.ModuleList(ll_alignments_b)

        # ----- Global Loss -----
        self.g_alignment = get_global_alignment_objective(self.config.g_alignment,
                                                          d_zg=self.config.d_zg, d_hidden=self.config.d_hidden)
        log.info(f'Instantiated objective {type(self.g_alignment)} for g_alignment')

        assert len(self.ll_alignments_a) > 0 or len(self.ll_alignments_b) > 0 \
               or self.g_alignment is not None, 'No objectives have been defined'

        # ----- Loss weights and some consistency checks -----
        self.loss_weights = self.config.loss_weights

        self.compute_global = self.g_alignment is not None
        self.compute_attended_a = len(self.ll_alignments_a) > 0
        self.compute_attended_b = len(self.ll_alignments_b) > 0
        self.compute_local = self.compute_attended_a or self.compute_attended_b

    def forward(self, embeddings: EmbeddingsData, attention: CrossAttentionOutput, losses: dict):
        metrics = {}

        losses.update(self.local_loss(embeddings, attention))
        losses.update(self.global_loss(embeddings, metrics))

        total_loss, detached_losses = self.combine_losses(losses, self.loss_weights)
        metrics.update({f'{name}_loss': loss for name, loss in detached_losses.items()})
        return total_loss, metrics

    def global_loss(self, embeddings, metrics):
        losses = {}
        if self.g_alignment is not None:
            losses['global_a2b'], losses['global_b2a'], g_metrics = \
                self.g_alignment(embeddings.zg_a, embeddings.zg_b)
            metrics.update({'global_' + name: metric for name, metric in g_metrics.items()})
        return losses

    def local_loss(self, embeddings, attention: CrossAttentionOutput):
        losses = {}
        for i, ll_alignment_a in enumerate(self.ll_alignments_a):
            prefix = f'local_a_{i}_' if len(self.ll_alignments_a) > 1 else 'local_a_'
            losses[prefix + 'l2att'], losses[prefix + 'att2l'] = ll_alignment_a(
                embeddings.zl_a,
                embeddings.zl_b2a,
                mask=embeddings.mask_a,
                weights=embeddings.weights_a,
                structure_size=embeddings.local_size_a,
            )
        for i, ll_alignment_b in enumerate(self.ll_alignments_b):
            prefix = f'local_b_{i}_' if len(self.ll_alignments_b) > 1 else 'local_b_'
            losses[prefix + 'l2att'], losses[prefix + 'att2l'] = ll_alignment_b(
                embeddings.zl_b,
                embeddings.zl_a2b,
                mask=embeddings.mask_b,
                weights=embeddings.weights_b,
                structure_size=embeddings.local_size_b,
            )
        return losses

    def combine_losses(self, losses, weights):
        assert len(losses) > 0
        # -- keep losses for debug --
        detached_losses = {
            key: loss.detach() for key, loss in losses.items() if loss is not None
        }
        # -- weight the losses and sum them --
        total_loss = sum(weights[key] * loss for key, loss in losses.items() if loss is not None)
        return total_loss, detached_losses


class LocalWeightsModule(nn.Module):
    def __init__(self, computation_mode, stop_weights_grad):
        super(LocalWeightsModule, self).__init__()
        assert computation_mode in (None, 'from_aggregation')
        self.computation_mode = computation_mode
        self.stop_weights_grad = stop_weights_grad

    def forward(self, modality_a: ModalityEmbeddingsData):
        if self.computation_mode is None:
            return None
        elif self.computation_mode == 'from_aggregation':
            assert modality_a.weights is not None, 'No weights have been computed during aggregation'
            l_weights = modality_a.weights
        else:
            raise ValueError(self.computation_mode)

        if self.stop_weights_grad:
            l_weights = l_weights.detach()
        return l_weights
