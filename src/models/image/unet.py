from collections import OrderedDict
from typing import List, Collection

from pl_bolts.models.vision import UNet
from torch import nn


class UNetFeatureExtractor(nn.Module):
    def __init__(self, extracted_layers: List[str], backbone: UNet = None):
        super(UNetFeatureExtractor, self).__init__()

        if backbone is None:
            backbone = UNet(num_classes=1)  # classifier layer will not be used
        self.layers = backbone.layers[:-1]  # remove classifier layer
        self.num_layers = backbone.num_layers

        self.d = {
            'input': 3,
            'conv': 64,
            'down1': 128,
            'down2': 256,
            'down3': 512,
            'down4': 1024,
            'up1': 512,
            'up2': 256,
            'up3': 128,
            'up4': 64,
        }
        self.downscale_factors = {
            'input': 1,
            'conv': 1,
            'down1': 2,
            'down2': 4,
            'down3': 8,
            'down4': 16,
            'up1': 8,
            'up2': 4,
            'up3': 2,
            'up4': 1,
        }

        assert all(layer in set(self.d.keys()) for layer in extracted_layers)
        self.extracted_layers = set(extracted_layers)

    def set_extracted_feature_layers(self, extracted_layers: Collection[str]):
        self.extracted_layers = set(extracted_layers)

    def forward(self, x):
        extracted_features = OrderedDict()
        if 'input' in self.extracted_layers:
            extracted_features['input'] = x

        xi = [self.layers[0](x)]
        if 'conv' in self.extracted_layers:
            extracted_features['conv'] = xi[0]

        # Down path
        for i, layer in enumerate(self.layers[1:self.num_layers]):
            xi.append(layer(xi[-1]))
            if f'down{i+1}' in self.extracted_layers:
                extracted_features[f'down{i+1}'] = xi[-1]

        # Up path
        for i, layer in enumerate(self.layers[self.num_layers:]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
            if f'up{i+1}' in self.extracted_layers:
                extracted_features[f'up{i+1}'] = xi[-1]

        return extracted_features
