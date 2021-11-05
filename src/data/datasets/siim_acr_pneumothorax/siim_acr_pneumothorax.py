import json
import logging
import math
import os
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict

import PIL
import click
import pandas as pd
import numpy as np
import pydicom
from pytorch_lightning import seed_everything
from skimage import exposure
from tqdm import tqdm

if __name__ == '__main__':
    datasets_path = Path(os.path.realpath(__file__)).absolute().parent.parent
    root_path = datasets_path.parent.parent
    sys.path.append(str(root_path))


from data.datasets.base_dataset import SegmentationDataset
from data.datasets.processing_utils import random_split_indices, write_detection_csv_file, PIXEL_STATS_FILE_NAME, \
    write_segmentation_csv_file

"""
Dataset sources:
https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview
https://www.kaggle.com/seesee/siim-train-test
"""

IMG_SIZE = 1024

log = logging.getLogger(__name__)


def prepare_siim_dataset(path, val_percentage: float, test_percentage: float):
    img_source_folder = os.path.join(path, 'dicom-images-train')
    img_target_folder = os.path.join(path, 'processed_images')
    os.makedirs(img_target_folder, exist_ok=True)
    mask_folder = os.path.join(path, 'segmentation_masks')
    os.makedirs(mask_folder, exist_ok=True)

    images_dict = parse_labels(path)
    image_ids = list(images_dict.keys())
    img_path_dict = find_images(img_source_folder)
    pixel_means = []
    pixel_vars = []
    pos_weights = []
    for image_id in tqdm(image_ids):
        pixel_mean, pixel_var = process_image(img_path_dict, img_target_folder, image_id)
        pixel_means.append(pixel_mean)
        pixel_vars.append(pixel_var)

    for image_id in tqdm(image_ids):
        encoded_pixels_list = images_dict[image_id]
        pos_weights.append(create_mask(mask_folder, image_id, encoded_pixels_list))

    num_samples = len(image_ids)
    train_indices, val_indices, test_indices = random_split_indices(num_samples, test_percentage, val_percentage)

    log.info('Processing Splits...')
    train_stats = write_split(os.path.join(path, 'train.csv'), train_indices, image_ids, pixel_means, pixel_vars, pos_weights)
    val_stats = write_split(os.path.join(path, 'validation.csv'), val_indices, image_ids, pixel_means, pixel_vars, pos_weights)
    test_stats = write_split(os.path.join(path, 'test.csv'), test_indices, image_ids, pixel_means, pixel_vars, pos_weights)
    stats = {
        'train': train_stats,
        'validation': val_stats,
        'test': test_stats,
    }
    with open(os.path.join(path, PIXEL_STATS_FILE_NAME), "w") as f:
        json.dump(stats, f)
    log.info('Done.')


def write_split(target_path, indices, image_ids, means, vars, pos_percentage):
    image_ids = [image_ids[i] for i in indices]
    write_segmentation_csv_file(target_path, image_ids)

    means = [means[i] for i in indices]
    vars = [vars[i] for i in indices]
    pos_percentage = [pos_percentage[i] for i in indices]
    mean = np.mean(means)
    var = np.mean(vars) + np.var(means)
    pos_percentage = np.mean(pos_percentage)
    neg_percentage = 1 - pos_percentage
    pos_weights = neg_percentage / pos_percentage

    return {'num_samples': len(means), 'scan': {'pixel_mean': float(mean), 'pixel_std': math.sqrt(float(var))},
            'segmentation_pos_weights': {'pneumothorax': pos_weights}}


def find_images(images_path):
    img_path_dict = {}
    for path in glob(f'{images_path}/*/*/*.dcm'):
        image_id = path.split('/')[-1][:-len('.dcm')]
        img_path_dict[image_id] = path

    return img_path_dict


def parse_labels(path):
    rles_df = pd.read_csv(os.path.join(path, 'train-rle.csv'))
    rles_df.columns = ['ImageId', 'EncodedPixels']

    images_dict = defaultdict(list)
    for _, row in rles_df.iterrows():
        image_id = row['ImageId']
        encoded_pixels = row['EncodedPixels']

        images_dict[image_id].append(encoded_pixels)

    return images_dict


def process_image(img_path_dict, target_folder: str, image_id):
    src_file = img_path_dict[image_id]
    target_file = os.path.join(target_folder, f'{image_id}.png')

    dcm_data = pydicom.read_file(src_file)
    img = dcm_data.pixel_array.astype(float) / 255.
    assert img.shape == (IMG_SIZE, IMG_SIZE)
    img = exposure.equalize_hist(img)
    pixel_mean = img.mean()
    pixel_var = img.var()

    img = (255 * img).astype(np.uint8)
    img = PIL.Image.fromarray(img).convert('L')
    img.save(target_file, format='PNG')

    return pixel_mean, pixel_var


def create_mask(mask_folder, image_id, encoded_pixels_list):
    """
    Multiple annotations will be combined using OR (mask pixel will be True if any annotation is True at that pixel).
    :param mask_folder:
    :param image_id:
    :param encoded_pixels_list:
    :return:
    """

    target_file = os.path.join(mask_folder, f'{image_id}.gif')

    # adapted based on from https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/data
    mask = np.zeros(IMG_SIZE * IMG_SIZE, dtype=np.uint8)
    for rle in encoded_pixels_list:
        if rle.strip() == '-1':
            continue
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            current_position += start
            mask[current_position:current_position + lengths[index]] = 1
            current_position += lengths[index]

    mask = mask.reshape(IMG_SIZE, IMG_SIZE).T  # (H x W)
    PIL.Image.fromarray(mask * 255).save(target_file, format='GIF')

    return mask.astype(float).sum() / (IMG_SIZE * IMG_SIZE)


class SIIMSegmentationDataset(SegmentationDataset):
    def _get_img_path(self, id):
        return os.path.join(self.root_path, 'processed_images', f'{id}.png')

    def _get_mask_paths(self, id) -> Dict[str, str]:
        return {'pneumothorax': os.path.join(self.root_path, 'segmentation_masks', f'{id}.gif')}


@click.command()
@click.argument('siim_root_path')
@click.option('--val_percentage', default=0.2)
@click.option('--test_percentage', default=0.2)
@click.option('--seed', default=None)
def prepare_siim_dataset_cmd(siim_root_path, val_percentage, test_percentage, seed):
    if seed is not None:
        seed_everything(seed)
    log.info(f'Preparing SIIM ACR pneumothorax segmentation dataset...')
    log.info(f'Using path "{siim_root_path}"')

    prepare_siim_dataset(siim_root_path, val_percentage=val_percentage, test_percentage=test_percentage)
    log.info(f'Finished dataset preparation.')


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    prepare_siim_dataset_cmd()
