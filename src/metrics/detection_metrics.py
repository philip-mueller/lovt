from collections import namedtuple
from typing import List, Dict

import numpy as np
import torch
from mean_average_precision import MeanAveragePrecision2d
from pytorch_lightning.metrics import Metric
from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin


class BBoxMeanAPMetric(Metric):
    """
    https://github.com/bes-dev/mean_average_precision
    """
    def __init__(self, class_names, iou_thresholds=(0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75),
                 extra_reported_thresholds=(0.5, 0.75)):
        super(BBoxMeanAPMetric, self).__init__(compute_on_step=False)

        self.iou_thresholds = np.array(iou_thresholds)
        self.extra_reported_thresholds = []
        for extra_thres in extra_reported_thresholds:
            found_close = False
            for thres in self.iou_thresholds:
                if np.isclose(thres, extra_thres):
                    self.extra_reported_thresholds.append(thres)
                    found_close = True
            if not found_close:
                raise ValueError(f'{extra_thres} not found in {self.iou_thresholds}')
        self.metric = MeanAveragePrecision2d(len(class_names))
        self.class_names = class_names

    def reset(self):
        self.metric.reset()

    def update(self, predictions: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]]):
        for predicted, target in zip(predictions, targets):
            pred_boxes = predicted['boxes'].detach().cpu().numpy()
            pred_scores = predicted['conf'].detach().cpu().numpy()
            pred_classes = predicted['classes'].detach().cpu().numpy()
            target_classes = target['classes'].detach().cpu().numpy()
            target_boxes = target['boxes'].detach().cpu().numpy()

            # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
            gt = np.zeros((len(target_boxes), 7))
            gt[:, 0:2] = target_boxes[:, 0:2]
            gt[:, 2:4] = target_boxes[:, 0:2] + target_boxes[:, 2:4]
            gt[:, 4] = target_classes
            # [xmin, ymin, xmax, ymax, class_id, confidence]
            preds = np.zeros((len(pred_boxes), 6))
            preds[:, 0:2] = pred_boxes[:, 0:2]
            preds[:, 2:4] = pred_boxes[:, 0:2] + pred_boxes[:, 2:4]
            preds[:, 4] = pred_classes
            preds[:, 5] = pred_scores
            self.metric.add(preds, gt)

    def compute(self):
        computed_metrics = self.metric.value(iou_thresholds=self.iou_thresholds,
                                             mpolicy="soft",
                                             recall_thresholds=np.arange(0., 1.01, 0.01))
        metrics = {'mAP': computed_metrics['mAP']}

        for c, class_name in enumerate(self.class_names):
            metrics[f'{class_name}_mAP'] = np.mean([computed_metrics[t][c]['ap'] for t in self.iou_thresholds])

        for t in self.extra_reported_thresholds:
            metrics[f'mAP@{t}'] = np.mean([computed_metrics[t][c]['ap'] for c in range(len(self.class_names))])

        return metrics


Object = namedtuple('Object',
                    ['image_path', 'object_id', 'object_type', 'coordinates'])
Prediction = namedtuple('Prediction',
                        ['image_index', 'probability', 'coordinates'])


class FrocMetric(DeviceDtypeModuleMixin, Metric):
    """
    See https://github.com/hlk-1135/object-CXR/blob/master/froc.py
    """
    def __init__(self, fps=(0.125, 0.25, 0.5, 1, 2, 4, 8)):
        super(FrocMetric, self).__init__()

        self.fps = fps

        self.preds: List[Prediction] = []
        self.target_boxes = []  # 1 element per sample
        self.num_target_boxes = 0

    def update(self, predictions: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]]):
        for predicted, target in zip(predictions, targets):
            pred_boxes = predicted['boxes'].detach().cpu().numpy()
            pred_probs = predicted['conf'].detach().cpu().numpy()
            target_boxes = target['boxes'].detach().cpu().numpy()
            target_boxes[:, 2:] = target_boxes[:, :2] + target_boxes[:, 2:]  # convert to x1, y1, x2, y2 box

            image_id = len(self.target_boxes)
            self.target_boxes.append(target_boxes)
            self.num_target_boxes += len(target_boxes)
            assert self.target_boxes[image_id] is target_boxes

            coords = pred_boxes[:, :2] + pred_boxes[:, 2:] / 2  # convert to center coord

            for coord, prob in zip(coords, pred_probs):
                self.preds.append(Prediction(image_id, prob, coord))

    def reset(self):
        self.preds = []
        self.target_boxes = []
        self.num_target_boxes = 0

    def compute(self):
        num_images = len(self.target_boxes)
        # sort prediction by probabiliyt
        preds = sorted(self.preds, key=lambda x: x.probability, reverse=True)

        # compute hits and false positives
        hits = 0
        false_positives = 0
        fps_idx = 0
        object_hitted = set()
        fps = self.fps
        froc = []
        for i in range(len(preds)):
            is_inside = False
            pred = preds[i]
            for box_index, box in enumerate(self.target_boxes[pred.image_index]):
                box_id = (pred.image_index, box_index)
                if inside_object(pred, box):
                    is_inside = True
                    if box_id not in object_hitted:
                        hits += 1
                        object_hitted.add(box_id)

            if not is_inside:
                false_positives += 1

            if false_positives / num_images >= fps[fps_idx]:
                sensitivity = hits / self.num_target_boxes
                froc.append(sensitivity)
                fps_idx += 1

                if len(fps) == len(froc):
                    break

        if len(froc) == 0:
            if self.num_target_boxes == 0:
                froc_metric = np.array(1.0)
            else:
                froc_metric = np.array(float(hits) / self.num_target_boxes)
        else:
            while len(froc) < len(fps):
                froc.append(froc[-1])
            froc_metric = np.mean(froc)
        return torch.tensor(froc_metric, device=self.device, dtype=torch.float)


def inside_object(pred, box):
    x1, y1, x2, y2 = box
    x, y = pred.coordinates
    return x1 <= x <= x2 and y1 <= y <= y2
