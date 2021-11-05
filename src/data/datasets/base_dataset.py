import csv
import json
import logging
import os
from abc import abstractmethod
from itertools import islice
from typing import List, Tuple, Dict, Any

import PIL
from torch.utils.data import Dataset
import numpy as np

from data.datasets.processing_utils import PIXEL_STATS_FILE_NAME


log = logging.getLogger(__name__)


class DetectionDataset(Dataset):
    def __init__(self, root_path, csv_path, stats):
        self.root_path = root_path
        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            self.samples = list(islice(reader, 1, None))
        self.stats = stats

    @classmethod
    def load_from_disk(cls, dataset_path: str, **kwargs):
        with open(os.path.join(dataset_path, PIXEL_STATS_FILE_NAME), "r") as f:
            stats = json.load(f)

        return {
            split: cls(dataset_path, os.path.join(dataset_path, f'{split}.csv'), split_stats, **kwargs)
            for split, split_stats in stats.items()
        }

    @abstractmethod
    def _get_img_path(self, id):
        raise NotImplementedError

    @classmethod
    def _extract_id_boxes_classes(cls, csv_row: List[str]) -> Tuple[str, str, str]:
        id, boxes, classes, *_ = csv_row
        return id, boxes, classes

    def __getitem__(self, item):
        id, boxes, classes = self._extract_id_boxes_classes(self.samples[item])

        scan = PIL.Image.open(self._get_img_path(id))
        boxes = [[float(coord) for coord in box.split(';')] for box in boxes.split('|') if len(box) > 0]
        boxes = np.array(boxes, dtype=float) if len(boxes) > 0 else np.zeros((0, 4))
        assert boxes.shape[1] == 4, boxes.shape
        classes = np.array([0 for _ in boxes], dtype=int)  # only 1 class => index = 0
        assert boxes.shape[0] == classes.shape[0]

        return {
            "id": id,
            "scan": scan,
            "detection_targets": {"boxes": boxes, "classes": classes}
        }

    def __len__(self):
        return len(self.samples)


class SegmentationDataset(Dataset):
    def __init__(self, root_path, csv_path, stats):
        self.root_path = root_path
        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            self.samples = list(islice(reader, 1, None))
        self.stats = stats

    @classmethod
    def load_from_disk(cls, dataset_path: str, **kwargs):
        with open(os.path.join(dataset_path, PIXEL_STATS_FILE_NAME), "r") as f:
            stats = json.load(f)

        return {
            split: cls(dataset_path, os.path.join(dataset_path, f'{split}.csv'), split_stats, **kwargs)
            for split, split_stats in stats.items()
        }

    @abstractmethod
    def _get_img_path(self, id: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    def _get_mask_paths(self, id: Any) -> Dict[str, str]:
        raise NotImplementedError

    @classmethod
    def _extract_id(cls, csv_row: List[str]) -> Any:
        return csv_row[0]

    def __getitem__(self, item):
        id = self._extract_id(self.samples[item])

        scan = PIL.Image.open(self._get_img_path(id))
        segmentation_masks = {
            mask_name: PIL.Image.open(mask_path) for mask_name, mask_path in self._get_mask_paths(id).items()
        }

        return {
            "id": id,
            "scan": scan,
            "segmentation_masks": segmentation_masks
        }

    def __len__(self):
        return len(self.samples)
