import logging
from collections import OrderedDict
from functools import partial
from itertools import chain
from typing import List, Collection, Set, Tuple, Dict

import torch
from torch import nn
from torchvision.models import ResNet
from torch.nn import functional as F
from transformers import apply_chunking_to_forward

from models.image.resnet import ResNetFeatureExtractor

log = logging.getLogger(__name__)

class ConvBlock(nn.Module):
    """
    Based on https://github.com/kevinlu1211/pytorch-unet-resnet-50-encoder/blob/master/u_net_resnet_50_encoder.py

    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) if with_nonlinearity else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    Based on https://github.com/kevinlu1211/pytorch-unet-resnet-50-encoder/blob/master/u_net_resnet_50_encoder.py
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Based on https://github.com/kevinlu1211/pytorch-unet-resnet-50-encoder/blob/master/u_net_resnet_50_encoder.py

    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class ResUNetFeatureExtractor(nn.Module):
    LAST_LAYER = 'up5'
    LAYERS = {'input', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'bridge', 'up1', 'up2', 'up3', 'up4', 'up5'}
    """
    Based on https://github.com/kevinlu1211/pytorch-unet-resnet-50-encoder/blob/master/u_net_resnet_50_encoder.py
    """
    def __init__(self, backbone: ResNetFeatureExtractor, extracted_layers: List[str], up_dims: Dict[str, int] = None):
        """

        :param backbone:
        :param extracted_layers: list of layers to be extracted.
            Concatenation of multiple layer is possible using "+" (without spaces) between layer names,
            e.g. "up5+conv5" is the concatenation of up5 and conv5. All layers are intepolated to size of the first specified layer.
            Linear projected layer can be specified by adding ":projection_dim" at the end of a layer name, e.g. "conv5:64".
            This can also be done when layers are concatenated, e.g.: "up5+conv5:64" or "up5:32+conv5:32".
        """
        super(ResUNetFeatureExtractor, self).__init__()
        assert isinstance(backbone, ResNetFeatureExtractor)

        self.d = dict(backbone.d)  # init downsampling d from ResNet backbone
        default_up_dims = {
            'bridge': 2048,
            'up1': 1024,
            'up2': 512,
            'up3': 256,
            'up4': 128,
            'up5': 64,
        }
        up_dims = dict(default_up_dims, **up_dims) if up_dims is not None else default_up_dims
        self.d.update(up_dims)
        log.info(f'Using the following dims in ResUNet: {self.d}')
        self.downscale_factors = {
            'input': 1,
            'conv1': 2,
            'conv2': 4,
            'conv3': 8,
            'conv4': 16,
            'conv5': 32,
            'bridge': 32,
            'up1': 16,
            'up2': 8,
            'up3': 4,
            'up4': 2,
            'up5': 1,
        }
        self.projection_layers = nn.ModuleDict()
        self.extracted_layers: Set[str] = set()
        self.extracted_concat_layers: Set[Tuple[str, ...]] = set()
        self.set_extracted_feature_layers(extracted_layers, add_projector=True)

        backbone.set_extracted_feature_layers(['input', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5'])
        self.backbone = backbone
        self.bridge = Bridge(self.d['conv5'], self.d['bridge'])
        self.up_blocks = nn.ModuleList([
            UpBlockForUNetWithResNet50(in_channels=self.d['up1'] + self.d['conv4'], out_channels=self.d['up1'],
                                       up_conv_in_channels=self.d['bridge'], up_conv_out_channels=self.d['up1']),
            UpBlockForUNetWithResNet50(in_channels=self.d['up2'] + self.d['conv3'], out_channels=self.d['up2'],
                                       up_conv_in_channels=self.d['up1'], up_conv_out_channels=self.d['up2']),
            UpBlockForUNetWithResNet50(in_channels=self.d['up3'] + self.d['conv2'], out_channels=self.d['up3'],
                                       up_conv_in_channels=self.d['up2'], up_conv_out_channels=self.d['up3']),
            UpBlockForUNetWithResNet50(in_channels=self.d['up4'] + self.d['conv1'], out_channels=self.d['up4'],
                                       up_conv_in_channels=self.d['up3'], up_conv_out_channels=self.d['up4']),
            UpBlockForUNetWithResNet50(in_channels=self.d['up5'] + self.d['input'], out_channels=self.d['up5'],  # concatenated with input
                                       up_conv_in_channels=self.d['up4'], up_conv_out_channels=self.d['up5'])
        ])

    def set_extracted_feature_layers(self, extracted_layers: Collection[str], add_projector=False):
        self.extracted_layers: Set[str] = set()
        self.extracted_concat_layers: Set[Tuple[str, ...]] = set()
        for layer in extracted_layers:
            self.add_extracted_feature_layer(layer, add_projector=add_projector)

    def add_extracted_feature_layer(self, feature_layer, add_projector=False):
        if '+' in feature_layer:
            # => concat layer
            single_layers = tuple(feature_layer.split('+'))
            for single_layer in single_layers:
                self._add_extracted_single_layer(single_layer, add_projector=add_projector)
            self.extracted_concat_layers.add(single_layers)
            self.d[feature_layer] = sum(self.d[single_layer] for single_layer in single_layers)
            # first layer defined feature map size
            self.downscale_factors[feature_layer] = self.downscale_factors[single_layers[0]]
        else:
            self._add_extracted_single_layer(feature_layer, add_projector=add_projector)

    def _add_extracted_single_layer(self, layer, add_projector=False):
        if ':' in layer:
            assert len(layer.split(':')) == 2
            projection_layer = layer
            layer, projection_dim = layer.split(':')
            projection_dim = int(projection_dim)

            self.d[projection_layer] = projection_dim
            self.downscale_factors[projection_layer] = self.downscale_factors[layer]
            if projection_layer not in self.projection_layers:
                assert add_projector
                self.projection_layers[projection_layer] = nn.Conv2d(self.d[layer], projection_dim, kernel_size=1)

        assert layer in ResUNetFeatureExtractor.LAYERS
        self.extracted_layers.add(layer)

    def forward(self, x, frozen_backbone=False):
        extracted_features = OrderedDict()
        if frozen_backbone:
            with torch.no_grad():
                x, down_outputs, extracted_features = self.downsample(x, extracted_features)
        else:
            x, down_outputs, extracted_features = self.downsample(x, extracted_features)

        x = self.bridge(x)
        if 'bridge' in self.extracted_layers:
            extracted_features['bridge'] = x

        extracted_features = self.upsample(x, down_outputs, extracted_features)

        if len(self.projection_layers) > 0:
            extracted_features = self.project_layers(extracted_features)
        if len(self.extracted_concat_layers) > 0:
            extracted_features = self.concat_layers(extracted_features)

        return extracted_features

    def downsample(self, x, extracted_features):
        backbone_features = self.backbone(x)

        x = backbone_features['conv5']
        down_outputs = [backbone_features[feature] for feature in ('input', 'conv1', 'conv2', 'conv3', 'conv4')]
        for layer, value in backbone_features.items():
            if layer in self.extracted_layers:
                extracted_features[layer] = value
        return x, down_outputs, extracted_features

    def upsample(self, x, down_outputs, extracted_features):
        for i, block in enumerate(self.up_blocks):
            x = block(x, down_outputs.pop())

            if f'up{i + 1}' in self.extracted_layers:
                extracted_features[f'up{i + 1}'] = x
        return extracted_features

    def project_layers(self, extracted_features):
        for projection_layer, projector in self.projection_layers.items():
            layer, projection_dim = projection_layer.split(':')
            extracted_features[projection_layer] = projector(extracted_features[layer])
        return extracted_features

    def concat_layers(self, extracted_features):
        for layer in self.extracted_concat_layers:
            first_layer, *concat_layers = layer
            first_features = extracted_features[first_layer]
            _, _, H, W = first_features.shape
            concat_features = []
            for concat_layer in concat_layers:
                # extract features and interpolate to first_layer size
                features = extracted_features[concat_layer]
                if features.shape[2:] != (H, W):
                    if features.shape[1] <= 64:
                        features = F.interpolate(features, size=(H, W), mode='bilinear', align_corners=False)
                    else:
                        def _interpolate(_features):
                            return F.interpolate(_features, size=(H, W), mode='bilinear', align_corners=False)

                        features = apply_chunking_to_forward(
                            _interpolate,
                            64,  # chunk_size
                            1,  # chunk_dim
                            features,
                        )
                concat_features.append(features)

            extracted_features['+'.join(layer)] = torch.cat([first_features] + concat_features, dim=1)

        return extracted_features

    def backbone_params(self):
        return self.backbone.parameters()

    def non_backbone_params(self):
        return chain(self.bridge.parameters(), self.up_blocks.parameters(), self.projection_layers.parameters())

    @staticmethod
    def for_torchvision_resnet(backbone: ResNet, extracted_layers: List[str], up_dims=None):
        resnet = ResNetFeatureExtractor(backbone=backbone,
                                        extracted_layers=['input', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5'])
        return ResUNetFeatureExtractor(resnet, extracted_layers, up_dims=up_dims)
