from collections import OrderedDict
from itertools import islice
from typing import List

from torch import nn
from torchvision.models import ResNet


class ResNetFeatureExtractor(nn.Module):
    LAYERS = {'input', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5'}
    def __init__(self, backbone: ResNet, extracted_layers: List[str]):
        super(ResNetFeatureExtractor, self).__init__()
        assert len(extracted_layers) > 0

        self.d = {
            'input': 3,
            'conv1': backbone.conv1.out_channels,  # 64,
            'conv2': 256,
            'conv3': 512,
            'conv4': 1024,
            'conv5': 2048
        }
        self.downscale_factors = {
            'input': 1,
            'conv1': 2,
            'conv2': 4,
            'conv3': 8,
            'conv4': 16,
            'conv5': 32
        }

        self.layer_name_to_index = {
            'input': -1,
            'conv1': 2,  # 0: conv1, 1: bn1, 2: relu
            'conv2': 4,  # 3: maxpool, 4: layer1
            'conv3': 5,  # layer2
            'conv4': 6,  # layer3
            'conv5': 7   # layer4
        }

        self.backbone_layers = nn.ModuleDict(list(backbone.named_children())[:-2])  # avg pool and final fc layers are never used
        self.set_extracted_feature_layers(extracted_layers)

    def set_extracted_feature_layers(self, extracted_layers: List[str]):
        self.extracted_layers_by_index = {
            self.layer_name_to_index[extracted_name]: extracted_name for extracted_name in extracted_layers
        }
        self.num_encoder_layers = max(self.layer_name_to_index[layer_name] for layer_name in extracted_layers) + 1

    def add_extracted_feature_layer(self, feature_layer):
        if feature_layer not in self.extracted_layers_by_index.values():
            layer_index = self.layer_name_to_index[feature_layer]
            self.num_encoder_layers = max(self.num_encoder_layers, layer_index + 1)
            self.extracted_layers_by_index[layer_index] = feature_layer

    def forward(self, x):
        extracted_features = OrderedDict()

        # save input feature map if required (index = -1)
        extracted_name = self.extracted_layers_by_index.get(-1)
        if extracted_name is not None:
            extracted_features[extracted_name] = x

        for index, layer in enumerate(islice(self.backbone_layers.values(), self.num_encoder_layers)):
            # apply layer
            x = layer(x)

            # extract the output feature map of this layer if required
            extracted_name = self.extracted_layers_by_index.get(index)

            if extracted_name is not None:
                extracted_features[extracted_name] = x

        return extracted_features