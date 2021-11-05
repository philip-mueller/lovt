import csv
import logging
import math
from typing import List, Tuple

import PIL
import datasets
import numpy as np
from pyarrow import compute


log = logging.getLogger(__name__)
PIXEL_STATS_FILE_NAME = 'dataset_statistics.json'


def compute_pixel_stats(dataset: datasets.Dataset, pixel_mean_column='scan_pixel_mean', pixel_var_column='scan_pixel_var'):
    log.info('Computing pixel stats...')
    data = dataset.data
    if dataset._indices is not None:
        data = data.fast_gather(dataset._indices.column(0).to_pylist())
    means = data.column(pixel_mean_column)
    variances = data.column(pixel_var_column)

    if len(dataset) > 0:
        mean = compute.mean(means).as_py()
        if len(dataset) > 1:
            var = compute.mean(variances).as_py() + compute.variance(means).as_py()
        else:
            var = compute.mean(variances).as_py()
        std = math.sqrt(var)
    else:
        mean = 0.
        std = 0.

    log.info(f'Pixel stats computed (mean={mean}, std={std})')
    return mean, std


def write_detection_csv_file(target_path, samples: List[Tuple[str, List[Tuple[float, float, float, float]], List[int]]]):
    log.info('Writing csv file')
    with open(target_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['ID', 'boxes', 'classes'])
        for sample_id, boxes, classes in samples:
            assert len(boxes) == len(classes)
            boxes = '|'.join(f'{box[0]};{box[1]};{box[2]};{box[3]}' for box in boxes)
            classes = '|'.join(str(int(cls)) for cls in classes)

            writer.writerow([sample_id, boxes, classes])


def write_segmentation_csv_file(target_path, samples: List[str]):
    log.info('Writing csv file')
    with open(target_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['ID'])
        for sample_id in samples:
            writer.writerow([sample_id])


def compute_detection_dataset_stats(patient_ids, means, vars, boxes: List[Tuple[float, float, float, float]]):
    mean = np.mean(means)
    var = np.mean(vars) + np.var(means)

    return {'num_samples': len(means), 'scan': {'pixel_mean': float(mean), 'pixel_std': math.sqrt(float(var))},
            'patient_ids': patient_ids}


def create_segmentation_mask(boxes, img_size, target_file):
    mask = np.zeros(img_size, dtype=np.uint8)
    for x, y, w, h in boxes:
        mask[round(y):round(y+h), round(x):round(x+w)] = 1.

    PIL.Image.fromarray(mask * 255).save(target_file, format='GIF')


def random_split_indices(num_samples, test_percentage, val_percentage):
    indices = np.random.permutation(num_samples)
    num_test = round(test_percentage * num_samples)
    num_val = round(val_percentage * num_samples)
    test_indices = indices[:num_test]
    val_indices = indices[num_test:num_test + num_val]
    train_indices = indices[num_test + num_val:]
    return train_indices, val_indices, test_indices
