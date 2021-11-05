import logging
import math
from itertools import chain
from typing import List, Union

import torch
from segmentation_models_pytorch.losses import FocalLoss, DiceLoss
from torch import nn
from torch.nn import functional as F, BCEWithLogitsLoss

from models.image.resnet import ResNetFeatureExtractor
from models.image.resunet import ResUNetFeatureExtractor
from models.image.unet import UNetFeatureExtractor

log = logging.getLogger(__name__)


class SegmentationHead(nn.Module):
    def __init__(self, segmentation_task: str, d_y: int,
                 d_hidden=512, dropout_prob=0.2, nonlinear=False, bn_before=False, dataset_stats=None,
                 loss_fn='bce'):
        super(SegmentationHead, self).__init__()

        pos_weights = dataset_stats.get('segmentation_pos_weights')
        if segmentation_task == 'SCR_hear_lungs_clavicles':
            self.target_names = ['heart', 'lungs', 'clavicles']
        elif segmentation_task == 'rsna_pneunomia_detection':
            self.target_names = ['opacity']
        elif segmentation_task == 'covid_rural':
            self.target_names = ['opacity']
        elif segmentation_task == 'NIH_CXR_pathology_detection':
            self.target_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate',
                                 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
        elif segmentation_task == 'SIIM_pneumothorax_segmentation':
            self.target_names = ['pneumothorax']
        elif segmentation_task == 'object_cxr':
            self.target_names = ['foreign_object']
        else:
            raise ValueError(segmentation_task)
        self.num_logits = len(self.target_names)
        if bn_before:
            self.bn_1 = nn.BatchNorm1d(d_y)
        else:
            self.bn_1 = None
        self.nonlinear = nonlinear
        if nonlinear:
            assert d_hidden is not None
            self.project = nn.Conv1d(d_y, d_hidden, kernel_size=1, bias=False)
            self.relu = nn.ReLU(inplace=True)
            self.bn_2 = nn.BatchNorm1d(d_hidden)
        else:
            d_hidden = d_y

        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Conv1d(d_hidden, self.num_logits, kernel_size=1, bias=True)

        self.loss_fn = MultiLabelBinarySegmentationLoss(self.target_names,
                                                        pos_weights=pos_weights,
                                                        loss_fn=loss_fn)

    def forward(self, yl, struc_size=None, targets=None, target_shape=None, return_probs=True):
        """

        :param yl: (B x N x d)
        :param labels:
        :param return_probs:
        :return:
        """
        if struc_size is not None:
            B, N, d = yl.size()
            assert N == math.prod(struc_size)
            x = yl.permute(0, 2, 1)  # (B x d x N)
        else:
            B, d, *struc_size = yl.size()
            x = yl.view(B, d, -1)

        if self.bn_1 is not None:
            x = self.bn_1(x)

        if self.nonlinear:
            x = self.dropout(x)
            x = self.project(x)
            x = self.bn_2(x)
            x = self.relu(x)
        x = self.dropout(x)
        x = self.classifier(x)  # (B x num_classes x N)
        logits = x.view(B, self.num_logits, *struc_size)

        if target_shape is None and targets is not None:
            target_shape = targets[list(targets.keys())[0]].size()[1:]
        if target_shape is not None and target_shape != struc_size:
            logits = F.interpolate(logits, size=target_shape, mode='bilinear', align_corners=False)

        predictions, targtes, loss = self.loss_fn(logits, targets, return_probs=return_probs)
        return self.loss_fn.stacked_to_dict(predictions.detach()), self.loss_fn.stacked_to_dict(targtes.detach()), loss


class CombinedLoss(nn.Module):
    def __init__(self, loss_a, loss_b):
        super(CombinedLoss, self).__init__()
        self.loss_a = loss_a
        self.loss_b = loss_b

    def forward(self, logits, targets):
        return self.loss_a(logits, targets) + self.loss_b(logits, targets)


class MultiLabelBinarySegmentationLoss(nn.Module):
    def __init__(self, label_names: List[str], pos_weights: dict = None, loss_fn='bce'):
        super(MultiLabelBinarySegmentationLoss, self).__init__()
        self.num_logits = len(label_names)
        self.label_names = label_names
        if pos_weights is not None:
            log.info('Using pos_weights for segmentation')
            pos_weights = torch.tensor([pos_weights[label_name] for label_name in label_names]).view(-1, 1, 1)
        else:
            pos_weights = None

        if loss_fn == 'bce':
            self.loss_fn = BCEWithLogitsLoss(pos_weight=pos_weights)
        elif loss_fn == 'dice':
            self.loss_fn = DiceLoss(mode='binary')
        elif loss_fn == 'bcedice':
            self.loss_fn = CombinedLoss(BCEWithLogitsLoss(pos_weight=pos_weights), DiceLoss(mode='binary'))
        elif loss_fn == 'focal':
            self.loss_fn = FocalLoss(mode='binary')

    def stacked_to_dict(self, stacked_tensor: torch.Tensor) -> dict:
        return {label_name: stacked_tensor[:, i].contiguous() for i, label_name in enumerate(self.label_names)}

    def forward(self, logits, targets=None, return_probs=True):
        if targets is not None:
            # B x N_targtes
            if isinstance(targets, dict):
                targets = torch.stack([targets[label_name] for label_name in self.label_names], dim=1)
            else:
                assert isinstance(targets, torch.Tensor)
            loss = self.loss_fn(logits, targets.type_as(logits))
        predictions = torch.sigmoid(logits) if return_probs else logits
        if targets is not None:
            return predictions, targets, loss
        else:
            return predictions


class UNetSegmentationForResNet(nn.Module):
    """
    Builds a new trainable UNet on top of encoder.
    If it is already a UNet (ResUNetFeatureExtractor), then only a segmentation head is added.
    If is it a UNet and reinit_upsampling is True then only downsampling part is used and upsampling is newly initialized

    Based on https://github.com/kevinlu1211/pytorch-unet-resnet-50-encoder/blob/master/u_net_resnet_50_encoder.py
    """
    def __init__(self, backbone: Union[ResNetFeatureExtractor, ResUNetFeatureExtractor], segmentation_task: str, loss_fn: str, dataset_stats=None, reinit_upsampling=False):
        super(UNetSegmentationForResNet, self).__init__()
        assert isinstance(backbone, (ResNetFeatureExtractor, ResUNetFeatureExtractor))

        if isinstance(backbone, ResNetFeatureExtractor):
            self.resunet = ResUNetFeatureExtractor(backbone, extracted_layers=[ResUNetFeatureExtractor.LAST_LAYER])
        elif isinstance(backbone, ResUNetFeatureExtractor) and reinit_upsampling:
            self.resunet = ResUNetFeatureExtractor(backbone.backbone, extracted_layers=[ResUNetFeatureExtractor.LAST_LAYER])
        elif isinstance(backbone, ResUNetFeatureExtractor) and not reinit_upsampling:
            backbone.set_extracted_feature_layers([ResUNetFeatureExtractor.LAST_LAYER])
            self.resunet = backbone

        self.segmentation_head = SegmentationHead(segmentation_task, d_y=self.resunet.d[ResUNetFeatureExtractor.LAST_LAYER], nonlinear=False, dropout_prob=0.,
                                                  dataset_stats=dataset_stats, bn_before=False, loss_fn=loss_fn)

    def forward(self, scan: torch.Tensor, segmentation_masks: dict = None, return_probs=True, frozen_backbone=False, **kwargs):
        x = self.resunet(scan, frozen_backbone=frozen_backbone)[ResUNetFeatureExtractor.LAST_LAYER]
        predictions, targtes, loss = self.segmentation_head(x, targets=segmentation_masks, return_probs=return_probs)
        return predictions, targtes, loss, {}

    @property
    def backbone(self):
        return self.resunet.backbone

    def backbone_params(self):
        return self.resunet.backbone_params()

    def non_backbone_params(self):
        return chain(self.resunet.non_backbone_params(), self.segmentation_head.parameters())


class LinearSegmentation(nn.Module):
    def __init__(self, backbone: Union[ResNetFeatureExtractor, UNetFeatureExtractor, ResUNetFeatureExtractor],
                 extracted_layer: str,
                 segmentation_task: str,
                 loss_fn: str,
                 dataset_stats=None, bn=False):
        super(LinearSegmentation, self).__init__()

        self.extracted_layer = extracted_layer
        backbone.set_extracted_feature_layers([extracted_layer])
        self.backbone = backbone
        self.segmentation_head = SegmentationHead(
            segmentation_task=segmentation_task,
            d_y=self.backbone.d[extracted_layer],
            dropout_prob=0.0,
            bn_before=bn,
            nonlinear=False,
            dataset_stats=dataset_stats,
            loss_fn=loss_fn
        )

    def forward(self, scan: torch.Tensor, segmentation_masks: dict = None, return_probs=True, frozen_backbone=False, **kwargs):
        if frozen_backbone:
            with torch.no_grad():
                local_features = self.backbone(scan)[self.extracted_layer]
        else:
            local_features = self.backbone(scan)[self.extracted_layer]

        predictions, targtes, loss = self.segmentation_head(
            local_features,
            targets=segmentation_masks,
            return_probs=return_probs
        )
        return predictions, targtes, loss, {}

    def backbone_params(self):
        return self.backbone.parameters()

    def non_backbone_params(self):
        return self.segmentation_head.parameters()
