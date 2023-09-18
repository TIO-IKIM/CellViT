# -*- coding: utf-8 -*-
# MoNuSeg Dataset
#
# Dataset information: https://monuseg.grand-challenge.org/Home/
# Please Prepare Dataset as described here: docs/readmes/monuseg.md
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import logging
from pathlib import Path
from typing import Callable, Union, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from cell_segmentation.datasets.pannuke import PanNukeDataset
from einops import rearrange

logger = logging.getLogger()
logger.addHandler(logging.NullHandler())


class MoNuSegDataset(Dataset):
    def __init__(
        self,
        dataset_path: Union[Path, str],
        transforms: Callable = None,
        patching: bool = False,
        overlap: int = 0,
    ) -> None:
        """MoNuSeg Dataset

        Args:
            dataset_path (Union[Path, str]): Path to dataset
            transforms (Callable, optional): Transformations to apply on images. Defaults to None.
            patching (bool, optional): If patches with size 256px should be used Otherwise, the entire MoNuSeg images are loaded. Defaults to False.
            overlap: (bool, optional): If overlap should be used for patch sampling. Overlap in pixels.
                Recommended value other than 0 is 64. Defaults to 0.
        Raises:
            FileNotFoundError: If no ground-truth annotation file was found in path
        """
        self.dataset = Path(dataset_path).resolve()
        self.transforms = transforms
        self.masks = []
        self.img_names = []
        self.patching = patching
        self.overlap = overlap

        image_path = self.dataset / "images"
        label_path = self.dataset / "labels"
        self.images = [f for f in sorted(image_path.glob("*.png")) if f.is_file()]
        self.masks = [f for f in sorted(label_path.glob("*.npy")) if f.is_file()]

        # sanity_check
        for idx, image in enumerate(self.images):
            image_name = image.stem
            mask_name = self.masks[idx].stem
            if image_name != mask_name:
                raise FileNotFoundError(f"Annotation for file {image_name} is missing")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict, str]:
        """Get one item from dataset

        Args:
            index (int): Item to get

        Returns:
            Tuple[torch.Tensor, dict, str]: Trainings-Batch
                * torch.Tensor: Image
                * dict: Ground-Truth values: keys are "instance map", "nuclei_binary_map" and "hv_map"
                * str: filename
        """
        img_path = self.images[index]
        img = np.array(Image.open(img_path)).astype(np.uint8)

        mask_path = self.masks[index]
        mask = np.load(mask_path, allow_pickle=True)
        mask = mask.astype(np.int64)

        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        hv_map = PanNukeDataset.gen_instance_hv_map(mask)
        np_map = mask.copy()
        np_map[np_map > 0] = 1

        # torch convert
        img = torch.Tensor(img).type(torch.float32)
        img = img.permute(2, 0, 1)
        if torch.max(img) >= 5:
            img = img / 255

        if self.patching and self.overlap == 0:
            img = rearrange(img, "c (h i) (w j) -> c h w i j", i=256, j=256)
        if self.patching and self.overlap != 0:
            img = img.unfold(1, 256, 256 - self.overlap).unfold(
                2, 256, 256 - self.overlap
            )

        masks = {
            "instance_map": torch.Tensor(mask).type(torch.int64),
            "nuclei_binary_map": torch.Tensor(np_map).type(torch.int64),
            "hv_map": torch.Tensor(hv_map).type(torch.float32),
        }

        return img, masks, Path(img_path).name

    def __len__(self) -> int:
        """Length of Dataset

        Returns:
            int: Length of Dataset
        """
        return len(self.images)

    def set_transforms(self, transforms: Callable) -> None:
        """Set the transformations, can be used tp exchange transformations

        Args:
            transforms (Callable): PyTorch transformations
        """
        self.transforms = transforms
