import collections
import logging
from collections import Sequence
from dataclasses import dataclass
from typing import Tuple, Optional, Collection, List, Dict, Any

import PIL
import torch
import torchvision.transforms as transforms
import numpy as np

from PIL import ImageFile
import torchvision.transforms.functional as F

log = logging.getLogger(__name__)


@dataclass
class ScanAugmentationConfig:
    augment: bool = True

    random_crop: bool = True
    crop_scale_range: Collection[float] = (0.6, 1.0)  # ConvVIRT: (0.6, 1.0), note: crops should be large enough

    random_horizontal_flip: bool = True
    horizontal_flip_prob: float = 0.5  # maybe None is better??? ConvVIRT: 0.5

    random_affine: bool = True  # ConvVIRT: True
    rotation_angle_range: Collection[float] = (-20, 20)  # ConvVIRT: (-20, 20)
    horizontal_translation_fraction: float = 0.1  # ConvVIRT: 0.1
    vertical_translation_fraction: float = 0.1  # ConvVIRT: 0.1
    scaling_range: Collection[float] = (0.95, 1.05)  # ConvVIRT: (0.95, 1.05)

    random_jitter: bool = True
    jitter_prob: Optional[float] = 1.0  # SimCLR/BYOL: 0.8, ConvVIRT: 1.0
    brightness_jitter_ratio_range:  Collection[float] = (0.6, 1.4)  # ConvVIRT: (0.6, 1.4)
    contrast_jitter_ratio_range: Collection[float] = (0.6, 1.4)  # ConvVIRT: (0.6, 1.4)

    gaussian_blur: bool = True
    gaussian_blur_sigma_range: Collection[float] = (0.1, 3.0)  # ConvVIRT: (0.1, 3.0)

    val_resize_mode: str = 'pad'  # center_crop, pad, resize


class ScanDataTransform:
    def __init__(self, config: ScanAugmentationConfig, image_size: Tuple[int, int], dataset_stats=None, val=False):
        """

        :param config:
        :param image_size: (H, W)
        :param dataset_stats:
        :param val:
        """
        super(ScanDataTransform, self).__init__()
        if not image_size[0] == image_size[1]:
            raise NotImplementedError('Currently only square target size is supported')

        assert config.val_resize_mode in ('center_crop', 'pad', 'resize')

        augment = False if val else config.augment

        if dataset_stats is None:
            dataset_stats = {}

        self.pixel_mean = dataset_stats.get('pixel_mean', 0.5)
        self.pixel_std = dataset_stats.get('pixel_std', 0.5)
        log.info(f'Using mean={self.pixel_mean} and std={self.pixel_std} for image normalization')

        data_transforms = []
        if augment and config.random_crop:
            data_transforms.append(RandomResizedCrop(size=image_size, scale=tuple(config.crop_scale_range)))
        else:
            if config.val_resize_mode == 'resize':
                data_transforms.append(Resize(image_size))
            elif config.val_resize_mode == 'center_crop':
                data_transforms.extend([
                    Resize(int(1.1 * max(image_size))),
                    CenterCrop(image_size)
                ])
            elif config.val_resize_mode == 'pad':
                data_transforms.extend([
                    PadToSquare(),
                    Resize(image_size),
                ])
        if augment and config.random_horizontal_flip:
            data_transforms.append(RandomHorizontalFlip(p=config.horizontal_flip_prob))
        if augment and config.random_affine:
            data_transforms.append(RandomAffine(degrees=config.rotation_angle_range,
                                   translate=(config.horizontal_translation_fraction,
                                              config.vertical_translation_fraction),
                                   scale=config.scaling_range))
        if augment and config.random_jitter:
            data_transforms.append(
                transforms.RandomApply([ColorJitter(brightness=tuple(config.brightness_jitter_ratio_range),
                                                    contrast=tuple(config.contrast_jitter_ratio_range))],
                                       p=config.jitter_prob))
        if augment and config.gaussian_blur is not None:
            # note: some implementation just use 3
            kernel_size = int(0.1 * max(image_size))
            if kernel_size % 2 == 0:
                kernel_size += 1
            data_transforms.append(GaussianBlur(kernel_size, sigma=tuple(config.gaussian_blur_sigma_range)))

        # final conversion and normalization
        data_transforms.append(ToTensor())
        data_transforms.append(Normalize(self.pixel_mean, self.pixel_std))

        self.transforms = transforms.Compose(data_transforms)

    def __call__(self, sample: Dict[str, Any]):
        """

        :param scan: list of dicts with
            - view: str
            - scan: numpy array
            The value comes from the "scan" part of each sample in the dataset.
        :return: dict with
            - input_scan: torch.Tensor
            The return value will be collated and then passed to the ScanEncoder.
        """
        # make sure truncated images do not raise an exception (we want to load them anyway)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        scan = sample['scan']
        if isinstance(scan, (list, tuple)):
            num_scans = len(scan)
            if len(scan) > 1:
                view_index = torch.randint(num_scans, size=()).item()
            else:
                view_index = 0
            scan: PIL.Image = scan[view_index]
            # also select views from other fields that are lists or tuples
            for key, data in sample.items():
                if isinstance(data, (list, tuple)):
                    if len(data) != num_scans:
                        raise ValueError(f'The sample field "{key}" is a list or tuple '
                                         f'that does not fit the number of scans in the sample.')
                    sample[key] = data[view_index]
            sample['view_index'] = view_index
            sample['scan'] = scan

        sample = resize_masks(sample)
        sample = self.transforms(sample)

        # expand greyscale image to color image
        sample['scan'] = sample['scan'].expand(3, -1, -1)
        return sample


class PadToSquare:
    def __call__(self, inputs):
        # pad to square
        W, H = _get_image_size(_get_scan(inputs))
        max_size = max(W, H)
        horizontal_pad = int((max_size - W) / 2)
        vertical_pad = int((max_size - H) / 2)

        def pad_boxes(boxes):
            boxes = np.array(boxes, dtype=float)
            boxes[:, 0] += horizontal_pad  # x1
            boxes[:, 1] += vertical_pad  # y1
            return boxes

        return _apply(inputs,
                      F.pad, [horizontal_pad, vertical_pad],
                      apply_to_masks=True, apply_to_detection=True, detection_box_fn=pad_boxes)


class CenterCrop(transforms.CenterCrop):
    def forward(self, inputs):
        return _apply(inputs,
                      F.center_crop, self.size,
                      apply_to_masks=True, apply_to_detection=True)


class Resize(transforms.Resize):
    def forward(self, inputs):
        width, height = _get_image_size(_get_scan(inputs))
        target_height, target_width = self.size

        def resize_boxes(boxes):
            boxes = np.array(boxes, dtype=float)

            scale_h = float(target_height) / float(height)
            scale_w = float(target_width) / float(width)

            boxes[:, 0] *= scale_w  # x1
            boxes[:, 1] *= scale_h  # y1
            boxes[:, 2] *= scale_w  # w
            boxes[:, 3] *= scale_h  # h
            return boxes

        return _apply(inputs,
                      F.resize, self.size, interpolation=self.interpolation,
                      mask_kwargs={'interpolation': F.InterpolationMode.NEAREST},
                      apply_to_masks=True, apply_to_detection=True, detection_box_fn=resize_boxes)


class RandomResizedCrop(transforms.RandomResizedCrop):
    def forward(self, inputs):
        i, j, h, w = self.get_params(_get_scan(inputs), self.scale, self.ratio)
        return _apply(inputs,
                      F.resized_crop, i, j, h, w, self.size, interpolation=self.interpolation,
                      mask_kwargs={'interpolation': F.InterpolationMode.NEAREST},
                      apply_to_masks=True, apply_to_detection=True)


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def forward(self, inputs):
        if torch.rand(1) < self.p:
            fn = F.hflip
        else:
            fn = lambda x: x
        return _apply(inputs, fn,
                      apply_to_masks=True, apply_to_detection=True)


class RandomAffine(transforms.RandomAffine):
    def forward(self, inputs):
        img = _get_scan(inputs)
        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]

        img_size = _get_image_size(img)

        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)

        return _apply(inputs, F.affine,
                      *ret, interpolation=self.interpolation, fill=fill,
                      apply_to_masks=True, apply_to_detection=True)


class ColorJitter(transforms.ColorJitter):
    def forward(self, inputs):
        if isinstance(inputs, dict):
            inputs['scan'] = super(ColorJitter, self).forward(inputs['scan'])
        else:
            inputs = super(ColorJitter, self).forward(inputs)
        return inputs


class GaussianBlur(transforms.GaussianBlur):
    def forward(self, inputs):
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        return _apply(inputs, F.gaussian_blur, self.kernel_size, [sigma, sigma],
                      apply_to_masks=False, apply_to_detection=False)


class ToTensor(transforms.ToTensor):
    def __call__(self, inputs):
        if isinstance(inputs, dict):
            inputs['scan'] = F.to_tensor(inputs['scan'])
            if 'segmentation_masks' in inputs:
                inputs['segmentation_masks'] = {name: ToTensor.convert_mask(mask)
                                                for name, mask in inputs['segmentation_masks'].items()}
            if 'mask' in inputs:
                inputs['mask'] = ToTensor.convert_mask(inputs['mask']).bool()
            if 'detection_targets' in inputs:
                inputs['detection_targets'] = ToTensor.convert_detection_targets(inputs['detection_targets'])
        else:
            inputs = F.to_tensor(inputs)
        return inputs

    @staticmethod
    def convert_mask(mask: PIL.Image):
        return torch.as_tensor(np.array(mask), dtype=torch.int64)

    @staticmethod
    def convert_detection_targets(detection_targets):
        assert isinstance(detection_targets, dict)
        assert set(detection_targets.keys()) == {'boxes', 'classes'}, detection_targets.keys()

        boxes = torch.as_tensor(np.array(detection_targets['boxes']), dtype=torch.float)
        assert boxes.ndim == 2
        num_boxes = boxes.size()[0]
        assert boxes.size()[1] == 4, boxes.size()[1]
        classes = torch.as_tensor(np.array(detection_targets['classes']), dtype=torch.int64)
        assert classes.ndim == 1 and classes.size()[0] == num_boxes, f'Expected {num_boxes}, but was {classes.size()}'
        return {'boxes': boxes, 'classes': classes}


class Normalize(transforms.Normalize):
    def forward(self, inputs):
        return _apply(inputs, F.normalize, self.mean, self.std, self.inplace,
                      apply_to_masks=False, apply_to_detection=False)


def resize_masks(inputs):
    if isinstance(inputs, dict):
        width, height = _get_scan(inputs).size

        def _resize(img: PIL.ImageFile):
            if tuple(img.size) == (width, height):
                return img
            else:
                return F.resize(img, size=[height, width], interpolation=F.InterpolationMode.NEAREST)

        if 'segmentation_masks' in inputs:
            inputs['segmentation_masks'] = {name: _resize(data)
                                            for name, data in inputs['segmentation_masks'].items()}
        if 'mask' in inputs:
            inputs['mask'] = _resize(inputs['mask'])
    return inputs


def _apply(inputs, fn, *args, apply_to_masks=True, apply_to_detection=True, detection_box_fn=None, mask_kwargs=None, **kwargs):
    if isinstance(inputs, dict):
        inputs['scan'] = fn(inputs['scan'], *args, **kwargs)

        mask_kwargs = kwargs if mask_kwargs is None else dict(kwargs, **mask_kwargs)
        if 'segmentation_masks' in inputs and apply_to_masks:
            inputs['segmentation_masks'] = {name: fn(data, *args, **mask_kwargs)
                                            for name, data in inputs['segmentation_masks'].items()}
        if 'mask' in inputs and apply_to_masks:
            inputs['mask'] = fn(inputs['mask'], *args, **mask_kwargs)
        if 'detection_targets' in inputs and apply_to_detection:
            assert detection_box_fn is not None, 'Augmentation does not support object detection targets'
            inputs['detection_targets']['boxes'] = detection_box_fn(inputs['detection_targets']['boxes'])
        return inputs
    else:
        return fn(inputs, *args, **kwargs)


def _get_scan(inputs):
    if isinstance(inputs, dict):
        return inputs['scan']
    else:
        return inputs


def _get_image_size(img):
    if isinstance(img, torch.Tensor):
        return [img.shape[-1], img.shape[-2]]
    else:
        return img.size
