import logging
import math
import os
from dataclasses import dataclass, field
from typing import Tuple, List, Union, Collection, Optional, Any, Mapping, Dict

import torch
from omegaconf import MISSING
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.models import ResNet

from models.components.aggregation import get_aggregator
from models.components.utils import EncoderInterface, EncoderOutput, EncoderConfig, AttentionMask
from models.image.resnet import ResNetFeatureExtractor
from models.image.resunet import ResUNetFeatureExtractor
from models.image.scan_transforms import ScanAugmentationConfig, ScanDataTransform
from common.dataclass_utils import TensorDataclassMixin
from common.user_config import USER_CONFIG
from models.image.unet import UNetFeatureExtractor

log = logging.getLogger(__name__)


def load_scan_feature_extractor(backbone_architecture: str,
                                backbone_model: Optional[Tuple[str, str]],
                                backbone_checkpoint: Optional[str],
                                extracted_layers: List[str],
                                backbone_pretrained=True,
                                **kwargs) -> nn.Module:

    if backbone_architecture in ('resnet', 'resunet'):
        assert isinstance(backbone_model, Collection) and len(backbone_model) == 2, backbone_model
        repo_or_dir, model = backbone_model
        if backbone_checkpoint is not None:
            backbone_pretrained = False
        torchvision_resnet_backbone = torch.hub.load(repo_or_dir, model, pretrained=backbone_pretrained)
        assert isinstance(torchvision_resnet_backbone, ResNet), f'Currently only torchvision ResNet backbones are supported: {type(torchvision_resnet_backbone)}'

        if backbone_checkpoint is not None:
            if backbone_checkpoint.startswith('baseline:'):
                model_name = backbone_checkpoint[len('baseline:'):]
                model_path = os.path.join(USER_CONFIG.models.base_path, 'baselines', model_name)
                backbone_checkpoint = os.path.join(model_path, 'checkpoints', 'backbone_weights.pt')
            # don't load FC layer
            new_state_dict = {k: v for k, v in torch.load(backbone_checkpoint).items() if not k.startswith('fc.')}
            old_state_dict = {k: v for k, v in torchvision_resnet_backbone.state_dict().items() if k.startswith('fc.')}
            state_dict = dict(old_state_dict, **new_state_dict)
            torchvision_resnet_backbone.load_state_dict(state_dict)

        if backbone_architecture == 'resnet':
            return ResNetFeatureExtractor(backbone=torchvision_resnet_backbone, extracted_layers=extracted_layers, **kwargs)
        elif backbone_architecture == 'resunet':
            return ResUNetFeatureExtractor.for_torchvision_resnet(backbone=torchvision_resnet_backbone,
                                                                  extracted_layers=extracted_layers,
                                                                  **kwargs)
    elif backbone_architecture == 'unet':
        if backbone_checkpoint is not None:
            raise NotImplementedError
        if backbone_pretrained:
            raise NotImplementedError
        else:
            assert backbone_model is None  # no torchvision model
            return UNetFeatureExtractor(extracted_layers, **kwargs)
    else:
        raise ValueError(backbone_architecture)


@dataclass
class ScanEncoderConfig(EncoderConfig):
    _encoder_cls_: str = 'ScanEncoderModel'
    modality: str = 'scan'

    backbone_architecture: str = MISSING
    backbone_model: Optional[Any] = None
    backbone_checkpoint: Optional[str] = None
    backbone_pretrained: bool = False
    backbone_args: Dict[str, Any] = field(default_factory=dict)
    input_size: Collection[int] = (MISSING, MISSING)
    region_feature_layer: str = MISSING
    global_feature_layer: str = MISSING
    global_aggregator: str = 'avg'  # as used in ResNet
    local_weights_from_aggregator: bool = False

    add_region_pos_encodings: bool = False
    concat_global_with_regions: bool = False

    data_augmentation: ScanAugmentationConfig = ScanAugmentationConfig()


class ScanEncoderModel(EncoderInterface, nn.Module):
    def __init__(self, config: ScanEncoderConfig, dataset_stats=None):
        super(ScanEncoderModel, self).__init__()

        self.config = config
        log.info(f'Instantiating ScanEncoderModel with backbone model {config.backbone_model} '
                 f'({config.backbone_pretrained=})')

        # ----- Feature Extractor -----
        backbone_checkpoint = config.backbone_checkpoint if hasattr(config, 'backbone_checkpoint') else None
        extracted_layers = [config.region_feature_layer]
        if config.global_feature_layer == 'local':
            self.global_from_local = True
        else:
            extracted_layers.append(config.global_feature_layer)
            self.global_from_local = False
            assert not config.local_weights_from_aggregator
        self.feature_extractor = load_scan_feature_extractor(config.backbone_architecture, config.backbone_model,
                                                             extracted_layers=extracted_layers,
                                                             backbone_pretrained=config.backbone_pretrained,
                                                             backbone_checkpoint=backbone_checkpoint,
                                                             **config.backbone_args)
        log.info(f'Backbone model {type(self.feature_extractor)}')

        # ----- Local Features -----
        self.d_l = self.feature_extractor.d[config.region_feature_layer]
        self.superpixel_aggregator = None

        region_downscale_factor = self.feature_extractor.downscale_factors[config.region_feature_layer]
        self._max_region_size = (int(config.input_size[0] / region_downscale_factor),
                                 int(config.input_size[1] / region_downscale_factor))
        H, W = self._max_region_size

        if config.add_region_pos_encodings:
            self.region_pos_embeddings = nn.Parameter(torch.randn(self.d_l, H, W))
            log.info(f'Local features (d_l = {self.d_l}): '
                     f'{self._max_region_size} patches from {config.region_feature_layer} with pos encodings')
        else:
            self.region_pos_embeddings = None
            log.info(f'Local features (d_l = {self.d_l}): '
                     f'{self._max_region_size} patches from {config.region_feature_layer}')

        # ----- Global Features -----
        self.d_g = self.d_l if self.global_from_local else self.feature_extractor.d[config.global_feature_layer]
        if config.concat_global_with_regions:
            self.d_l = self.d_l + self.d_g
        self.global_aggregator = get_aggregator(config.global_aggregator, d=self.d_g)
        log.info(f'Global features (d_g= {self.d_g}): aggregated with {type(self.global_aggregator)} from {config.global_feature_layer}')

        self.batch_collator = scan_batch_collator
        self.transform = ScanDataTransform(config.data_augmentation,
                                           image_size=config.input_size, dataset_stats=dataset_stats)
        self.val_transform = ScanDataTransform(config.data_augmentation,
                                               image_size=config.input_size, dataset_stats=dataset_stats,
                                               val=True)

    def update_data_augmentation(self, data_augmentation_config: Optional[Any], dataset_stats):
        if not data_augmentation_config:
            data_augmentation_config = ScanAugmentationConfig(augment=False)
        elif isinstance(data_augmentation_config, bool) and data_augmentation_config:
            data_augmentation_config = ScanAugmentationConfig()
        elif not isinstance(data_augmentation_config, ScanAugmentationConfig):
            data_augmentation_config = ScanAugmentationConfig(**data_augmentation_config)
        self.config.data_augmentation = data_augmentation_config
        self.transform = ScanDataTransform(data_augmentation_config,
                                           image_size=self.config.input_size, dataset_stats=dataset_stats)
        self.val_transform = ScanDataTransform(data_augmentation_config,
                                               image_size=self.config.input_size, dataset_stats=dataset_stats,
                                               val=True)

    def update_region_feature_layer(self, feature_layer: Optional[str] = None):
        if feature_layer is not None and feature_layer != self.config.region_feature_layer:
            self.feature_extractor.add_extracted_feature_layer(feature_layer)
            self.config.region_feature_layer = feature_layer

    def forward(self, scan: torch.Tensor,
                return_local=True, return_global=True, **kwargs) -> EncoderOutput:
        """
        Note: the input_scan comes from the collated output of ScanDataTransform
        :param scan: (B x 3 x H x W)
        :param return_local:
        :param return_global:
        :param kwargs: ignored
        :return:
        """
        B = scan.shape[0]
        extracted_features = self.feature_extractor(scan)
        _, _, H_x, W_x = scan.size()

        # ----- Local Features -----
        if return_local or (return_global and self.global_from_local):
            region_features = extracted_features[self.config.region_feature_layer]  # (B x d_l x H x W)
            _, _, H, W = region_features.size()
            assert (H, W) == self._max_region_size, f'{(H, W)} != expected size {self._max_region_size}'

            if self.region_pos_embeddings is not None:
                # (B x d_l x H x W)
                region_features = (region_features + self.region_pos_embeddings[None, :, :H, :W]) / math.sqrt(2)

            region_features = region_features.view(B, self.d_l, -1).transpose(-1, -2)  # (B x (H*W) x d_l)
            region_struc_size = (H, W)
        else:
            region_features = None
            region_struc_size = None

        # ----- Global Features -----
        if return_global or (return_local and self.config.concat_global_with_regions):
            if self.global_from_local:
                if self.config.local_weights_from_aggregator:
                    # (B x d_g)
                    global_features, local_weights = self.global_aggregator(region_features, return_weights=True)
                else:
                    # (B x d_g)
                    global_features = self.global_aggregator(region_features)
                    local_weights = None
            else:
                global_features = extracted_features[self.config.global_feature_layer]  # (B x d_g x H_g x W_g)
                global_features = global_features.view(B, self.d_g, -1).transpose(-1, -2)  # (B x (H_g x W_g) x d_g)
                global_features = self.global_aggregator(global_features)  # (B x d_g)
                local_weights = None
        else:
            global_features = None
            local_weights = None

        if self.config.concat_global_with_regions and return_local:
            region_features = torch.cat([region_features, global_features[:, None, :]])
        return EncoderOutput(local_features=region_features, global_features=global_features,
                             local_structure_size=region_struc_size, local_weights=local_weights)

    def get_x(self, scan: torch.Tensor, dicom_id, view_index, view, **kwargs) -> 'ScanInputInfo':
        return ScanInputInfo(dicom_id=dicom_id, view_index=view_index, scan_view=view,
                             scan_normalized=scan[:, 0],
                             scan=(scan[:, 0] * self.val_transform.pixel_std) + self.val_transform.pixel_mean,
                             local_regions_shape=self.max_region_size)

    @property
    def max_region_size(self):
        return self._max_region_size

    @staticmethod
    def load(config_or_checkpoint: Union[ScanEncoderConfig, str], dataset_stats=None):
        if isinstance(config_or_checkpoint, str):
            return ScanEncoderModel.load_pretrained(config_or_checkpoint)
        else:
            return ScanEncoderModel(config_or_checkpoint, dataset_stats=dataset_stats)


@dataclass
class ScanInputInfo(TensorDataclassMixin):
    dicom_id: Optional[List[str]]
    view_index: List[int]
    scan_view: List[str]
    scan: torch.Tensor  # (B x H x W)
    scan_normalized: torch.Tensor  # (B x H x W)
    local_regions_shape: Tuple[int, int]

    @staticmethod
    def is_scan_input(dict_input: dict):
        return 'scan' in dict_input

    @staticmethod
    def from_dict(dict_input):
        return ScanInputInfo(**dict_input)

    def __getitem__(self, i):
        return ScanInputInfo(
            dicom_id=self.dicom_id[i],
            view_index=self.view_index[i],
            scan_view=self.scan_view[i],
            scan=self.scan[i],
            scan_normalized=self.scan_normalized[i],
            local_regions_shape=self.local_regions_shape,  # same for all samples
        )


def scan_batch_collator(batch: list):
    elem = batch[0]
    if isinstance(elem, Mapping):
        return {key: [d[key] for d in batch] if key == 'detection_targets'
                     else default_collate([d[key] for d in batch])
                for key in elem}
    else:
        return default_collate(batch)
