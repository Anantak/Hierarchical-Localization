import argparse
import torch
from pathlib import Path
from typing import Dict, List, Union, Optional
import h5py
from types import SimpleNamespace
import cv2
import numpy as np
from tqdm import tqdm
import pprint
import collections.abc as collections
import PIL.Image
import glob

from . import extractors, logger
from .utils.base_model import dynamic_load
from .utils.parsers import parse_image_lists
from .utils.io import read_image, list_h5_names


class ImageDataset(torch.utils.data.Dataset):
    default_conf = {
        'grayscale': False,
        'resize_max': None,
        'resize_force': False,
        'interpolation': 'cv2_area',  # pil_linear is more accurate but slower
    }

    def __init__(self, image_paths, conf):
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.names = [p.as_posix() if isinstance(p, Path) else p for p in image_paths]

    def __getitem__(self, idx):
        name = self.names[idx]
        image = read_image(Path(name), self.conf.grayscale)
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]

        if self.conf.resize_max and (self.conf.resize_force or max(size) > self.conf.resize_max):
            scale = self.conf.resize_max / max(size)
            size_new = tuple(int(round(x * scale)) for x in size)
            image = resize_image(image, size_new, self.conf.interpolation)

        if self.conf.grayscale:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.

        data = {'image': image, 'original_size': np.array(size)}
        return data

    def __len__(self):
        return len(self.names)
