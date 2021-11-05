import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import PIL.Image
import click
import cv2
import pandas as pd

from pytorch_lightning import seed_everything
from skimage import exposure
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np


if __name__ == '__main__':
    datasets_path = Path(os.path.realpath(__file__)).absolute().parent.parent
    root_path = datasets_path.parent.parent
    sys.path.append(str(root_path))


from data.datasets.base_dataset import DetectionDataset, SegmentationDataset
from data.datasets.processing_utils import PIXEL_STATS_FILE_NAME, write_detection_csv_file, \
    compute_detection_dataset_stats

log = logging.getLogger(__name__)


def prepare_object_cxr_dataset(path: str, val_percentage: float = 0.1):
    assert val_percentage < 1.

    log.info('Loading targets')
    train_df = pd.read_csv(os.path.join(path, 'input', 'train.csv'))
    train_df, validation_df = train_test_split(train_df, test_size=val_percentage)
    test_df = pd.read_csv(os.path.join(path, 'input', 'dev.csv'))

    log.info('Processing Splits')
    train_stats = process_split(train_df, path, os.path.join(path, 'train.csv'), is_dev=False)
    val_stats = process_split(validation_df, path, os.path.join(path, 'validation.csv'), is_dev=False)
    test_stats = process_split(test_df, path, os.path.join(path, 'test.csv'), is_dev=True)

    stats = {
        'train': train_stats,
        'validation': val_stats,
        'test': test_stats,
    }
    with open(os.path.join(path, PIXEL_STATS_FILE_NAME), "w") as f:
        json.dump(stats, f)
    log.info('Done.')


def process_split(df: pd.DataFrame, path, csv_path, is_dev):
    df = df.fillna('')
    samples = []
    means = []
    vars = []
    src_imgs_folder = os.path.join(path, 'input', 'dev' if is_dev else 'train')
    processed_imgs_folder = os.path.join(path, 'images_processed')
    os.makedirs(processed_imgs_folder, exist_ok=True)
    masks_folder = os.path.join(path, 'segmentation_masks')
    os.makedirs(masks_folder, exist_ok=True)
    for _, sample in tqdm(df.iterrows()):
        img_id = sample['image_name'][:-len('.jpg')]
        src_path = os.path.join(src_imgs_folder, f'{img_id}.jpg')
        target_path = os.path.join(processed_imgs_folder, f'{img_id}.jpg')
        pixel_mean, pixel_var, img_size = process_image(src_path, target_path)
        img_height, img_width = img_size
        means.append(pixel_mean)
        vars.append(pixel_var)

        boxes, labels = sample_to_bboxes(sample)
        samples.append((img_id, boxes, labels))

        mask_path = os.path.join(masks_folder, f'{img_id}.gif')
        create_mask(sample, mask_path, img_height, img_width)

    all_boxes = [box for _, boxes, _ in samples for box in boxes]
    stats = compute_detection_dataset_stats(None, means, vars, all_boxes)
    write_detection_csv_file(csv_path, samples)

    return stats


def process_image(source_path, target_path):
    img = np.array(PIL.Image.open(source_path).convert('L'), dtype=float) / 255.
    img = exposure.equalize_hist(img)
    pixel_mean = img.mean()
    pixel_var = img.var()
    img_size = img.shape
    img = (255 * img).astype(np.uint8)
    img = PIL.Image.fromarray(img).convert('L')
    img.save(target_path, format='JPEG')

    return pixel_mean, pixel_var, img_size


def create_mask(sample, target_file, height, width):
    annotations = sample['annotation'].split(';') if len(sample['annotation']) > 0 else []
    mask = np.zeros((height, width), dtype=np.uint8)

    for annot in annotations:
        annot_type, *coords = [int(item) for item in annot.split()]
        if annot_type in (0, 1):  # box or ellipse
            assert len(coords) == 4, f'{annot_type}: {coords}'
            x1, y1, x2, y2 = coords
            mask[round(y1):round(y2), round(x1):round(x2)] = 1.
        else:
            assert annot_type == 2, annot_type
            xs = coords[::2]
            ys = coords[1::2]
            cv2.fillPoly(mask, [np.array([[y, x] for x, y in zip(xs, ys)], dtype='int32')], 1)

    PIL.Image.fromarray(mask * 255).save(target_file, format='GIF')


def sample_to_bboxes(sample):
    boxes = []
    annotations = sample['annotation'].split(';') if len(sample['annotation']) > 0 else []
    for annot in annotations:
        annot_type, *coords = [int(item) for item in annot.split()]
        if annot_type in (0, 1):  # box or ellipse
            x1, y1, x2, y2 = coords
        else:
            assert annot_type == 2
            xs = coords[::2]
            ys = coords[1::2]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
        boxes.append((x1, y1, y2 - y1, x2 - x1))
    labels = [1 for _ in boxes]
    return boxes, labels


class ObjectCxrDetectionDataset(DetectionDataset):
    def _get_img_path(self, id):
        return os.path.join(self.root_path, 'images_processed', f'{id}.jpg')


class ObjectCxrSegmentationDataset(SegmentationDataset):
    def _get_img_path(self, id):
        return os.path.join(self.root_path, 'images_processed', f'{id}.jpg')

    def _get_mask_paths(self, id) -> Dict[str, str]:
        return {'foreign_object': os.path.join(self.root_path, 'segmentation_masks', f'{id}.gif')}


@click.command()
@click.argument('object_cxr_root_path')
@click.option('--val_percentage', default=0.2)
@click.option('--seed', default=None)
def prepare_object_cxr_dataset_cmd(object_cxr_root_path, val_percentage, seed):
    if seed is not None:
        seed_everything(seed)
    log.info(f'Preparing Object CXR dataset...')
    log.info(f'Using path "{object_cxr_root_path}"')

    prepare_object_cxr_dataset(object_cxr_root_path, val_percentage=val_percentage)
    log.info(f'Finished dataset preparation.')


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    prepare_object_cxr_dataset_cmd()
