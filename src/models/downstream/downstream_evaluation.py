import logging
from dataclasses import dataclass
from typing import Any, Optional, Collection

import pytorch_lightning as pl
import torch
from omegaconf import MISSING, OmegaConf
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader
from torchmetrics import F1, AveragePrecision, AUROC, IoU, Accuracy, Precision, Recall

from data.dataloading_utils import load_dataset, DatasetTransformWrapper
from models.components.utils import EncoderConfig
from models.downstream.classification import ClassificationHead
from models.downstream.detection import YOLOv3WithResNetBackbone
from models.downstream.segmentation import LinearSegmentation, UNetSegmentationForResNet
from models.image.scan_encoder import ScanEncoderConfig
from models.pretraining.bimodal_alignment_model import BiModalModelRepresentationLearner, load_encoder
from metrics.detection_metrics import BBoxMeanAPMetric, FrocMetric
from common.config_utils import prepare_config

log = logging.getLogger(__name__)


@dataclass
class EvaluationModelConfig:
    task_type: str = MISSING
    eval_name: str = MISSING
    task: str = MISSING
    dataset: str = MISSING
    data_augmentation: Optional[Any] = False

    freeze_encoder: bool = True
    frozen_warmup_steps: int = 200  # ConVIRT: 200 steps
    warmup_lr: float = 1e-3   # ConVIRT: 1e-3

    batch_size: int = 64  # ConVIRT: 64
    learning_rate: float = 1e-4  # ConVIRT: 1e-4
    weight_decay: float = 1e-6  # ConVIRT: 1e-6
    lr_reduce_patience: int = 3  # ConVIRT: 3
    lr_reduce_factor: float = 0.5  # ConVIRT: 0.5

    max_epochs: int = 100


@dataclass
class ClassificationEvaluationModelConfig(EvaluationModelConfig):
    task_type: str = 'classification'
    task: str = MISSING

    nonlinear: bool = False
    d_hidden: int = 512  # only relevant if nonlinear = True
    dropout_prob: float = 0.2


class ClassificationEvaluator(pl.LightningModule):
    def __init__(
            self,
            config: ClassificationEvaluationModelConfig,
            encoder_config: EncoderConfig,
            encoder: Optional[nn.Module] = None,
            num_workers=4,
    ):
        super(ClassificationEvaluator, self).__init__()
        config = prepare_config(config, ClassificationEvaluationModelConfig, log)
        encoder_config = OmegaConf.create(encoder_config)
        self.config = config
        self.num_workers = num_workers

        self.save_hyperparameters('config', 'encoder_config', 'num_workers')

        # try to get dataset stats which may be required for the encoder transforms:
        train_dataset = load_dataset(config.dataset)['train']
        dataset_stats = train_dataset.stats.get(self.input_name, {})

        if encoder is None:
            encoder = load_encoder(encoder_config, dataset_stats=dataset_stats)
        encoder.update_data_augmentation(config.data_augmentation, dataset_stats=dataset_stats)
        self.encoder = encoder

        self.classifier = ClassificationHead(
            dataset_stats=train_dataset.stats,
            classifier_task=self.config.task,
            d_y=self.encoder.d_g,
            d_hidden=self.config.d_hidden,
            dropout_prob=self.config.dropout_prob,
            nonlinear=self.config.nonlinear
        )
        self.train_acc_metric = Accuracy()
        self.val_acc_metric = Accuracy(compute_on_step=False)
        self.test_acc_metric = Accuracy(compute_on_step=False)
        self.val_auroc_metric = AUROC(compute_on_step=False, num_classes=self.classifier.num_labels)
        self.test_auroc_metric = AUROC(compute_on_step=False, num_classes=self.classifier.num_labels)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.transform = self.encoder.transform
        self.val_transform = self.encoder.val_transform

        self.encoder_frozen = False
        if config.freeze_encoder or config.frozen_warmup_steps > 0:
            self.encoder_frozen = True

    def on_train_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int):
        if not self.config.freeze_encoder and self.trainer.total_batch_idx >= self.config.frozen_warmup_steps:
            self.encoder_frozen = False
            for optim in self.trainer.optimizers:
                for group in optim.param_groups:
                    group['lr'] = self.config.learning_rate

        if self.encoder_frozen:
            self.encoder.eval()

    def training_step(self, batch, batch_idx):
        probs, labels, loss = self.shared_step(batch)
        acc = self.train_acc_metric(probs, labels)

        self.log(f'{self.config.eval_name}_train/loss', loss, prog_bar=True)
        self.log(f'{self.config.eval_name}_train/acc_step', acc, prog_bar=True)
        self.log(f'{self.config.eval_name}_train/acc_epoch', self.train_acc_metric, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        probs, labels, loss = self.shared_step(batch)
        self.val_acc_metric(probs, labels)
        self.val_auroc_metric(probs, labels)

        self.log(f'{self.config.eval_name}_val/loss', loss, prog_bar=True, sync_dist=True)
        self.log(f'{self.config.eval_name}_val/acc', self.val_acc_metric)
        self.log(f'{self.config.eval_name}_val/auroc', self.val_auroc_metric)

        return loss

    def test_step(self, batch, batch_idx):
        probs, labels, loss = self.shared_step(batch)
        self.test_acc_metric(probs, labels)
        self.test_auroc_metric(probs, labels)

        self.log(f'{self.config.eval_name}_test/loss', loss, sync_dist=True)
        self.log(f'{self.config.eval_name}_test/acc', self.test_acc_metric)
        self.log(f'{self.config.eval_name}_test/auroc', self.test_auroc_metric)

        return loss

    def shared_step(self, batch, return_probs=True):
        labels = self.classifier.get_labels(batch)

        if self.encoder_frozen:
            with torch.no_grad():
                representations = self.encoder(**batch, return_local=False, return_global=True).global_features
        else:
            representations = self.encoder(**batch, return_local=False, return_global=True).global_features

        return self.classifier(representations, labels, return_probs=return_probs)

    def configure_optimizers(self):
        params = self.classifier.parameters() if self.config.freeze_encoder else self.parameters()
        if not self.config.freeze_encoder and self.config.frozen_warmup_steps > 0:
            lr = self.config.warmup_lr
        else:
            lr = self.config.learning_rate
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=self.config.weight_decay)

        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, patience=self.config.lr_reduce_patience,
                                           factor=self.config.lr_reduce_factor,
                                           mode='max', verbose=True),
            'monitor': f'{self.config.eval_name}_val/auroc',
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer], [scheduler]

    def setup(self, stage: str):
        if stage == 'fit' or stage is None:
            dataset = load_dataset(self.config.dataset)
            self.train_dataset = dataset['train']
            self.val_dataset = dataset['validation']
            self.test_dataset = dataset['test']

    def batch_collator(self):
        return self.encoder.batch_collator

    def train_dataloader(self):
        return DataLoader(DatasetTransformWrapper(self.train_dataset, self.transform),
                          batch_size=self.config.batch_size, collate_fn=self.batch_collator(),
                          num_workers=self.num_workers, pin_memory=True, drop_last=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(DatasetTransformWrapper(self.val_dataset, self.val_transform),
                          batch_size=self.config.batch_size, collate_fn=self.batch_collator(),
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(DatasetTransformWrapper(self.test_dataset, self.val_transform),
                          batch_size=self.config.batch_size, collate_fn=self.batch_collator(),
                          num_workers=self.num_workers, pin_memory=True)

    @staticmethod
    def for_bimodal_model(checkpoint_path: str,
                          eval_config: ClassificationEvaluationModelConfig,
                          num_workers,
                          encoder: str = 'a'):
        assert encoder in ('a', 'b')

        pretrained_model: BiModalModelRepresentationLearner = \
            BiModalModelRepresentationLearner.load_from_checkpoint(checkpoint_path, do_load_dataset=False, strict=True)

        if encoder == 'a':
            encoder = pretrained_model.model_a.encoder
        else:
            encoder = pretrained_model.model_b.encoder

        eval_model = ClassificationEvaluator(config=OmegaConf.to_container(eval_config),
                                             encoder_config=OmegaConf.to_container(encoder.config),
                                             encoder=encoder, num_workers=num_workers)
        return eval_model, pretrained_model.config, pretrained_model.dataset

    @staticmethod
    def for_encoder(encoder_config: EncoderConfig,
                    eval_config: ClassificationEvaluationModelConfig,
                    num_workers):
        return ClassificationEvaluator(config=OmegaConf.to_container(eval_config),
                                       encoder_config=OmegaConf.to_container(encoder_config), num_workers=num_workers)


@dataclass
class SegmentationEvaluationModelConfig(EvaluationModelConfig):
    task_type: str = 'segmentation'
    task: str = MISSING

    loss_fn: str = 'bce'
    segmentation_head: str = 'unet'  # linear or unet or linear_unpooled_superpixel
    extracted_layer: Optional[str] = None
    reinit_upsampling: bool = False
    compute_ap_auroc: bool = False


class SegmentationEvaluator(pl.LightningModule):
    def __init__(
            self,
            config: SegmentationEvaluationModelConfig,
            encoder_config: EncoderConfig,
            encoder: Optional[nn.Module] = None,
            num_workers=4,
    ):
        super(SegmentationEvaluator, self).__init__()
        config = prepare_config(config, SegmentationEvaluationModelConfig, log)
        encoder_config = prepare_config(encoder_config, ScanEncoderConfig, log)
        self.config = config
        self.num_workers = num_workers

        self.save_hyperparameters('config', 'encoder_config', 'num_workers')
        self.input_name = encoder_config.modality

        # try to get dataset stats which may be required for the encoder transforms:
        train_dataset = load_dataset(config.dataset)['train']
        dataset_stats = train_dataset.stats.get(self.input_name, {})

        if encoder is None:
            encoder = load_encoder(encoder_config, dataset_stats=dataset_stats)
        encoder.update_data_augmentation(config.data_augmentation, dataset_stats=dataset_stats)
        self.transform = encoder.transform
        self.val_transform = encoder.val_transform

        if config.segmentation_head == 'linear':
            extracted_layer = config.extracted_layer
            if extracted_layer is None:
                extracted_layer = encoder.config.region_feature_layer
            self.segmentation_model = LinearSegmentation(encoder.feature_extractor,
                                                         extracted_layer,
                                                         config.task,
                                                         dataset_stats=train_dataset.stats,
                                                         loss_fn=config.loss_fn)
        elif config.segmentation_head == 'unet':
            self.segmentation_model = UNetSegmentationForResNet(encoder.feature_extractor,
                                                                config.task,
                                                                dataset_stats=train_dataset.stats,
                                                                reinit_upsampling=config.reinit_upsampling,
                                                                loss_fn=config.loss_fn)
        else:
            raise ValueError(config.segmentation_head)

        self.train_dice_metrics = nn.ModuleDict({
            target_name: F1(multiclass=False, average='micro')
            for target_name in self.segmentation_model.segmentation_head.target_names
        })
        self.val_dice_metrics = nn.ModuleDict({
            target_name: F1(compute_on_step=False, multiclass=False, average='micro')
            for target_name in self.segmentation_model.segmentation_head.target_names
        })
        self.val_pos_dice_metrics = nn.ModuleDict({
            target_name: F1(compute_on_step=False, multiclass=False, average='samples')
            for target_name in self.segmentation_model.segmentation_head.target_names
        })
        self.val_ap_metrics = nn.ModuleDict({
            target_name: AveragePrecision(compute_on_step=False,
                                          num_classes=1,
                                          pos_label=1)
            for target_name in self.segmentation_model.segmentation_head.target_names
        })
        self.test_dice_metrics = nn.ModuleDict({
            target_name: F1(compute_on_step=False, multiclass=False, average='micro')
            for target_name in self.segmentation_model.segmentation_head.target_names
        })
        self.test_pos_dice_metrics = nn.ModuleDict({
            target_name: F1(compute_on_step=False, multiclass=False, average='samples')
            for target_name in self.segmentation_model.segmentation_head.target_names
        })
        self.test_precision_metrics = nn.ModuleDict({
            target_name: Precision(compute_on_step=False, multiclass=False, average='micro')
            for target_name in self.segmentation_model.segmentation_head.target_names
        })
        self.test_recall_metrics = nn.ModuleDict({
            target_name: Recall(compute_on_step=False, multiclass=False, average='micro')
            for target_name in self.segmentation_model.segmentation_head.target_names
        })
        self.test_iou_metrics = nn.ModuleDict({
            target_name: IoU(compute_on_step=False, num_classes=2, ignore_index=0)
            for target_name in self.segmentation_model.segmentation_head.target_names
        })
        self.test_ap_metrics = nn.ModuleDict({
            target_name: AveragePrecision(compute_on_step=False,
                                          num_classes=1,
                                          pos_label=1)
            for target_name in self.segmentation_model.segmentation_head.target_names
        })
        self.test_auroc_metrics = nn.ModuleDict({
            target_name: AUROC(compute_on_step=False,
                               num_classes=1,
                               pos_label=1)
            for target_name in self.segmentation_model.segmentation_head.target_names
        })

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.encoder_frozen = False
        if config.freeze_encoder or config.frozen_warmup_steps > 0:
            self.encoder_frozen = True

    @property
    def backbone(self):
        return self.segmentation_model.backbone

    def on_train_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int):
        if not self.config.freeze_encoder and self.trainer.total_batch_idx >= self.config.frozen_warmup_steps:
            self.encoder_frozen = False
            for optim in self.trainer.optimizers:
                for group in optim.param_groups:
                    group['lr'] = self.config.learning_rate

        if self.encoder_frozen:
            self.segmentation_model.backbone.eval()

    def training_step(self, batch, batch_idx):
        probs, targets, loss, metrics = self.segmentation_model(**batch, frozen_backbone=self.encoder_frozen)

        self.log(f'{self.config.eval_name}_train/loss', loss, prog_bar=True)

        for target_name, dice_metric in self.train_dice_metrics.items():
            dice_metric(probs[target_name], targets[target_name])
            self.log(f'{self.config.eval_name}_train/{target_name}_dice_step', dice_metric)
            self.log(f'{self.config.eval_name}_train/{target_name}_dice_epoch', dice_metric, on_step=False, on_epoch=True)

        self.log_dict({
            f'{self.config.eval_name}_train/{metric_name}': metric
            for metric_name, metric in metrics.items()
        })

        return loss

    def _get_pos_samples(self, probs, targets):
        pos_probs = {}
        pos_targets = {}
        for target_name, target in targets.items():
            assert target.ndim == 3
            pos_mask = (target > 0).any(-1).any(-1)
            pos_targets[target_name] = target[pos_mask, :, :]
            pos_probs[target_name] = probs[target_name][pos_mask, :, :]
        return pos_probs, pos_targets

    def validation_step(self, batch, batch_idx):
        probs, targets, loss, metrics = self.segmentation_model(**batch)
        pos_probs, pos_targets = self._get_pos_samples(probs, targets)

        self.log(f'{self.config.eval_name}_val/loss', loss, prog_bar=True, sync_dist=True)

        for target_name, dice_metric in self.val_dice_metrics.items():
            dice_metric(probs[target_name], targets[target_name].bool())
            self.log(f'{self.config.eval_name}_val/{target_name}_dice', dice_metric)
        for target_name, pos_dice_metric in self.val_pos_dice_metrics.items():
            if pos_targets[target_name].shape[0] > 0:
                pos_dice_metric(pos_probs[target_name], pos_targets[target_name].bool())
                self.log(f'{self.config.eval_name}_val/{target_name}_pos_dice', pos_dice_metric)
        for target_name, ap_metric in self.val_ap_metrics.items():
            ap_metric(probs[target_name], targets[target_name])
            self.log(f'{self.config.eval_name}_val/{target_name}_AP', ap_metric)

        return loss

    def on_validation_epoch_end(self) -> None:
        avg_dice = torch.stack([dice_metric.compute() for dice_metric in self.val_dice_metrics.values()]).mean()
        self.log(f'{self.config.eval_name}_val/avg_dice', avg_dice)

    def test_step(self, batch, batch_idx):
        probs, targets, loss, metrics = self.segmentation_model(**batch)
        pos_probs, pos_targets = self._get_pos_samples(probs, targets)

        self.log_dict({
            f'{self.config.eval_name}_test/{metric_name}': metric
            for metric_name, metric in metrics.items()
        })
        self.log(f'{self.config.eval_name}_test/loss', loss)

        for target_name, dice_metric in self.test_dice_metrics.items():
            dice_metric(probs[target_name], targets[target_name])
            self.log(f'{self.config.eval_name}_test/{target_name}_dice', dice_metric)
        for target_name, pos_dice_metric in self.test_pos_dice_metrics.items():
            if pos_targets[target_name].shape[0] > 0:
                pos_dice_metric(pos_probs[target_name], pos_targets[target_name])
                self.log(f'{self.config.eval_name}_test/{target_name}_pos_dice', pos_dice_metric)
        for target_name, precision_metric in self.test_precision_metrics.items():
            precision_metric(probs[target_name], targets[target_name])
            self.log(f'{self.config.eval_name}_test/{target_name}_precision', precision_metric)
        for target_name, recall_metric in self.test_recall_metrics.items():
            recall_metric(probs[target_name], targets[target_name])
            self.log(f'{self.config.eval_name}_test/{target_name}_recall', recall_metric)
        for target_name, iou_metric in self.test_iou_metrics.items():
            iou_metric(probs[target_name], targets[target_name])
            self.log(f'{self.config.eval_name}_test/{target_name}_IoU', iou_metric)

        if self.config.compute_ap_auroc:
            for target_name in self.test_ap_metrics.keys():
                prob, target = probs[target_name], targets[target_name]
                # treat each pixel segmentation as its own prediction for AP and AUROC
                if prob.ndim > target.ndim:
                    B, n_classes, H, W = probs.size()
                    prob = prob.permute(0, 2, 3, 1).reshape(-1, n_classes)
                    target = target.view(-1)
                else:
                    prob = prob.view(-1)
                    target = target.view(-1)
                self.test_ap_metrics[target_name](prob, target)
                self.test_auroc_metrics[target_name](prob, target)

                self.log_dict({
                    f'{self.config.eval_name}_test/{target_name}_AP': ap_metric
                    for target_name, ap_metric in self.test_ap_metrics.items()
                })
                self.log_dict({
                    f'{self.config.eval_name}_test/{target_name}_AUROC': auroc_metric
                    for target_name, auroc_metric in self.test_auroc_metrics.items()
                })

        return loss

    def on_test_epoch_end(self) -> None:
        avg_dice = torch.stack([dice_metric.compute() for dice_metric in self.test_dice_metrics.values()]).mean()
        self.log(f'{self.config.eval_name}_test/avg_dice', avg_dice)

    def configure_optimizers(self):
        params = self.segmentation_model.non_backbone_params() if self.config.freeze_encoder else self.parameters()
        if not self.config.freeze_encoder and self.config.frozen_warmup_steps > 0:
            lr = self.config.warmup_lr
        else:
            lr = self.config.learning_rate
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=self.config.weight_decay)

        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, patience=self.config.lr_reduce_patience,
                                           factor=self.config.lr_reduce_factor,
                                           mode='min', verbose=True),
            'monitor': f'{self.config.eval_name}_val/loss',
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer], [scheduler]

    def setup(self, stage: str):
        if stage == 'fit' or stage is None:
            dataset = load_dataset(self.config.dataset)
            self.train_dataset = dataset['train']
            self.val_dataset = dataset['validation']
            self.test_dataset = dataset['test']

    def train_dataloader(self):
        return DataLoader(DatasetTransformWrapper(self.train_dataset, self.transform),
                          batch_size=self.config.batch_size,
                          num_workers=self.num_workers, pin_memory=True, drop_last=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(DatasetTransformWrapper(self.val_dataset, self.val_transform),
                          batch_size=self.config.batch_size,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(DatasetTransformWrapper(self.test_dataset, self.val_transform),
                          batch_size=self.config.batch_size,
                          num_workers=self.num_workers, pin_memory=True)

    @staticmethod
    def for_bimodal_model(checkpoint_path: str,
                          eval_config: SegmentationEvaluationModelConfig,
                          num_workers,
                          encoder: str = 'a',
                          feature_layer: Optional[str] = None):
        assert encoder in ('a', 'b')

        pretrained_model: BiModalModelRepresentationLearner = \
            BiModalModelRepresentationLearner.load_from_checkpoint(checkpoint_path, do_load_dataset=False, strict=True)

        if encoder == 'a':
            encoder = pretrained_model.model_a.encoder
        else:
            encoder = pretrained_model.model_b.encoder

        if feature_layer is not None:
            encoder.update_region_feature_layer(feature_layer)

        eval_model = SegmentationEvaluator(config=OmegaConf.to_container(eval_config),
                                           encoder_config=OmegaConf.to_container(encoder.config),
                                           encoder=encoder, num_workers=num_workers)
        return eval_model, pretrained_model.config, pretrained_model.dataset

    @staticmethod
    def for_encoder(encoder_config: EncoderConfig,
                    eval_config: SegmentationEvaluationModelConfig,
                    num_workers):
        return SegmentationEvaluator(config=OmegaConf.to_container(eval_config),
                                     encoder_config=OmegaConf.to_container(encoder_config), num_workers=num_workers)


@dataclass
class DetectionEvaluationModelConfig(EvaluationModelConfig):
    task_type: str = 'detection'
    task: str = MISSING

    extracted_layers: Collection[str] = ('conv3', 'conv4', 'conv5')
    # None => use anchors from dataset_stats
    anchors: Any = (((64.64, 48.6), (84.24, 106.92), (201.42, 176.04)),
                    ((16.2, 32.94), (33.48, 24.3), (31.86, 64.26)),
                    ((5.4, 7.02), (8.64, 16.2), (17.82, 12.42)))

    # YOLOv3, FasterRCNN, FasterRCNN-FPN
    detection_head: str = 'YOLOv3'


class DetectionEvaluator(pl.LightningModule):
    def __init__(
            self,
            config: DetectionEvaluationModelConfig,
            encoder_config: EncoderConfig,
            encoder: Optional[nn.Module] = None,
            num_workers=4,
    ):
        super(DetectionEvaluator, self).__init__()
        config = prepare_config(config, DetectionEvaluationModelConfig, log)
        encoder_config = prepare_config(encoder_config, ScanEncoderConfig, log)
        self.config = config
        self.num_workers = num_workers

        self.save_hyperparameters('config', 'encoder_config', 'num_workers')
        self.input_name = encoder_config.modality

        # try to get dataset stats which may be required for the encoder transforms:
        train_dataset = load_dataset(config.dataset)['train']
        dataset_stats = train_dataset.stats.get(self.input_name, {})

        if encoder is None:
            encoder = load_encoder(encoder_config, dataset_stats=dataset_stats)
        encoder.update_data_augmentation(config.data_augmentation, dataset_stats=dataset_stats)
        self.batch_collator = encoder.batch_collator
        self.transform = encoder.transform
        self.val_transform = encoder.val_transform

        if config.detection_head == 'YOLOv3':
            self.detection_model = YOLOv3WithResNetBackbone(encoder.feature_extractor,
                                                            img_size=encoder.config.input_size,
                                                            task=config.task,
                                                            extracted_layers=config.extracted_layers,
                                                            anchors=config.anchors,
                                                            dataset_stats=train_dataset.stats)
        else:
            raise ValueError(config.detection_head)

        self.val_mAP_metric = BBoxMeanAPMetric(self.detection_model.class_names)
        self.test_mAP_metric = BBoxMeanAPMetric(self.detection_model.class_names)
        self.val_pos_mAP_metric = BBoxMeanAPMetric(self.detection_model.class_names)
        self.test_pos_mAP_metric = BBoxMeanAPMetric(self.detection_model.class_names)

        if len(self.detection_model.class_names) == 1:
            self.val_froc = FrocMetric()
            self.test_froc = FrocMetric()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.encoder_frozen = False
        if config.freeze_encoder or config.frozen_warmup_steps > 0:
            self.encoder_frozen = True

    @property
    def backbone(self):
        return self.detection_model.backbone

    def on_train_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int):
        if not self.config.freeze_encoder and self.trainer.total_batch_idx >= self.config.frozen_warmup_steps:
            self.encoder_frozen = False
            for optim in self.trainer.optimizers:
                for group in optim.param_groups:
                    group['lr'] = self.config.learning_rate

        if self.encoder_frozen:
            self.detection_model.backbone.eval()

    def training_step(self, batch, batch_idx):
        _, _, loss, detailed_losses = self.detection_model(**batch, frozen_backbone=self.encoder_frozen)

        self.log(f'{self.config.eval_name}_train/loss', loss, prog_bar=True)
        self.log_dict({f'{self.config.eval_name}_train/{name}': value for name, value in detailed_losses.items()})

        return loss

    def _get_pos_samples(self, predictions, targets):
        pos_predictions = []
        pos_targets = []
        for predicted, target in zip(predictions, targets):
            if len(target['classes']) > 0:
                pos_predictions.append(predicted)
                pos_targets.append(target)
        return pos_predictions, pos_targets

    def validation_step(self, batch, batch_idx):
        output, targets, loss, detailed_losses = self.detection_model(**batch,
                                                             frozen_backbone=self.encoder_frozen,
                                                             return_predictions=True)
        pos_outputs, pos_targets = self._get_pos_samples(output, targets)

        self.log(f'{self.config.eval_name}_val/loss', loss, prog_bar=True, sync_dist=True)
        self.log_dict({f'{self.config.eval_name}_val/{name}': value for name, value in detailed_losses.items()})

        self.val_mAP_metric.update(output, targets)
        self.val_pos_mAP_metric.update(pos_outputs, pos_targets)

        if len(self.detection_model.class_names) == 1:
            self.val_froc.update(output, targets)

        return loss

    def on_validation_epoch_end(self):
        self.log_dict({f'{self.config.eval_name}_val/{name}': value
                       for name, value in self.val_mAP_metric.compute().items()})
        self.val_mAP_metric.reset()

        self.log_dict({f'{self.config.eval_name}_val/pos_{name}': value
                       for name, value in self.val_pos_mAP_metric.compute().items()})
        self.val_pos_mAP_metric.reset()

        if len(self.detection_model.class_names) == 1:
            self.log(f'{self.config.eval_name}_val/froc', self.val_froc.compute())
            self.val_froc.reset()

    def test_step(self, batch, batch_idx):
        output, targets, loss, detailed_losses = self.detection_model(**batch,
                                                             frozen_backbone=self.encoder_frozen,
                                                             return_predictions=True)
        pos_outputs, pos_targets = self._get_pos_samples(output, targets)

        self.log(f'{self.config.eval_name}_test/loss', loss, prog_bar=True, sync_dist=True)
        self.log_dict({f'{self.config.eval_name}_test/{name}': value for name, value in detailed_losses.items()})

        self.test_mAP_metric.update(output, targets)
        self.test_pos_mAP_metric.update(pos_outputs, pos_targets)
        if len(self.detection_model.class_names) == 1:
            self.test_froc.update(output, targets)

        return loss

    def on_test_epoch_end(self):
        self.log_dict({f'{self.config.eval_name}_test/{name}': value
                       for name, value in self.test_mAP_metric.compute().items()})
        self.test_mAP_metric.reset()

        self.log_dict({f'{self.config.eval_name}_test/pos_{name}': value
                       for name, value in self.test_pos_mAP_metric.compute().items()})
        self.test_pos_mAP_metric.reset()

        if len(self.detection_model.class_names) == 1:
            self.log(f'{self.config.eval_name}_test/froc', self.test_froc.compute())
            self.test_froc.reset()

    def configure_optimizers(self):
        params = self.detection_model.non_backbone_params() if self.config.freeze_encoder else self.parameters()
        if not self.config.freeze_encoder and self.config.frozen_warmup_steps > 0:
            lr = self.config.warmup_lr
        else:
            lr = self.config.learning_rate
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=self.config.weight_decay)

        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, patience=self.config.lr_reduce_patience,
                                           factor=self.config.lr_reduce_factor,
                                           mode='min', verbose=True),
            'monitor': f'{self.config.eval_name}_val/loss',
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer], [scheduler]

    def setup(self, stage: str):
        if stage == 'fit' or stage is None:
            dataset = load_dataset(self.config.dataset)
            self.train_dataset = dataset['train']
            self.val_dataset = dataset['validation']
            self.test_dataset = dataset['test']

    def train_dataloader(self):
        return DataLoader(DatasetTransformWrapper(self.train_dataset, self.transform),
                          batch_size=self.config.batch_size, collate_fn=self.batch_collator,
                          num_workers=self.num_workers, pin_memory=True, drop_last=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(DatasetTransformWrapper(self.val_dataset, self.val_transform),
                          batch_size=self.config.batch_size, collate_fn=self.batch_collator,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(DatasetTransformWrapper(self.test_dataset, self.val_transform),
                          batch_size=self.config.batch_size, collate_fn=self.batch_collator,
                          num_workers=self.num_workers, pin_memory=True)

    @staticmethod
    def for_bimodal_model(checkpoint_path: str,
                          eval_config: DetectionEvaluationModelConfig,
                          num_workers,
                          encoder: str = 'a'):
        assert encoder in ('a', 'b')
        pretrained_model: BiModalModelRepresentationLearner = \
            BiModalModelRepresentationLearner.load_from_checkpoint(checkpoint_path, do_load_dataset=False, strict=True)

        if encoder == 'a':
            encoder = pretrained_model.model_a.encoder
        else:
            encoder = pretrained_model.model_b.encoder

        eval_model = DetectionEvaluator(config=OmegaConf.to_container(eval_config),
                                        encoder_config=OmegaConf.to_container(encoder.config),
                                        encoder=encoder, num_workers=num_workers)
        return eval_model, pretrained_model.config, pretrained_model.dataset

    @staticmethod
    def for_encoder(encoder_config: EncoderConfig,
                    eval_config: DetectionEvaluationModelConfig,
                    num_workers):
        return DetectionEvaluator(config=OmegaConf.to_container(eval_config),
                                  encoder_config=OmegaConf.to_container(encoder_config),
                                  num_workers=num_workers)


def create_downstream_evaluator_for_bimodal_model(checkpoint_path: str,
                          eval_config: EvaluationModelConfig,
                          num_workers,
                          encoder: str = 'a',
                          feature_layer: Optional[str] = None):
    if eval_config.task_type == 'classification':
        return ClassificationEvaluator.for_bimodal_model(checkpoint_path, eval_config,
                                                         num_workers=num_workers, encoder=encoder)
    elif eval_config.task_type == 'segmentation':
        return SegmentationEvaluator.for_bimodal_model(checkpoint_path, eval_config,
                                                       num_workers=num_workers, encoder=encoder,
                                                       feature_layer=feature_layer)
    elif eval_config.task_type == 'detection':
        return DetectionEvaluator.for_bimodal_model(checkpoint_path, eval_config,
                                                    num_workers=num_workers, encoder=encoder)
    else:
        raise ValueError(eval_config.task_type)


def create_downstream_evaluator_for_encoder(encoder_config: EncoderConfig,
                                            eval_config: EvaluationModelConfig,
                                            num_workers):
    if eval_config.task_type == 'classification':
        return ClassificationEvaluator.for_encoder(encoder_config, eval_config, num_workers=num_workers)
    elif eval_config.task_type == 'segmentation':
        return SegmentationEvaluator.for_encoder(encoder_config, eval_config, num_workers=num_workers)
    elif eval_config.task_type == 'detection':
        return DetectionEvaluator.for_encoder(encoder_config, eval_config, num_workers=num_workers)
    else:
        raise ValueError(eval_config.task_type)
