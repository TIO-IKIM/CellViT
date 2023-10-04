# -*- coding: utf-8 -*-
# PanNuke Dataset
#
# Dataset information: https://arxiv.org/abs/2108.11195
# Please Prepare Dataset as described here: docs/readmes/pannuke.md # TODO: write own documentation
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


import logging
from pathlib import Path
from typing import Callable, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cell_segmentation.datasets.base_cell import CellDataset
from cell_segmentation.datasets.pannuke import PanNukeDataset

logger = logging.getLogger()
logger.addHandler(logging.NullHandler())


class CoNicDataset(CellDataset):
    """Lizzard dataset

    This dataset is always cached

    Args:
        dataset_path (Union[Path, str]): Path to Lizzard dataset. Structure is described under ./docs/readmes/cell_segmentation.md
        folds (Union[int, list[int]]): Folds to use for this dataset
        transforms (Callable, optional): PyTorch transformations. Defaults to None.
        stardist (bool, optional): Return StarDist labels. Defaults to False
        regression (bool, optional): Return Regression of cells in x and y direction. Defaults to False
        **kwargs are irgnored
    """

    def __init__(
        self,
        dataset_path: Union[Path, str],
        folds: Union[int, list[int]],
        transforms: Callable = None,
        stardist: bool = False,
        regression: bool = False,
        **kwargs,
    ) -> None:
        if isinstance(folds, int):
            folds = [folds]

        self.dataset = Path(dataset_path).resolve()
        self.transforms = transforms
        self.images = []
        self.masks = []
        self.img_names = []
        self.folds = folds
        self.stardist = stardist
        self.regression = regression
        for fold in folds:
            image_path = self.dataset / f"fold{fold}" / "images"
            fold_images = [f for f in sorted(image_path.glob("*.png")) if f.is_file()]

            # sanity_check: mask must exist for image
            for fold_image in fold_images:
                mask_path = (
                    self.dataset / f"fold{fold}" / "labels" / f"{fold_image.stem}.npy"
                )
                if mask_path.is_file():
                    self.images.append(fold_image)
                    self.masks.append(mask_path)
                    self.img_names.append(fold_image.name)

                else:
                    logger.debug(
                        "Found image {fold_image}, but no corresponding annotation file!"
                    )

        # load everything in advance to speedup, as the dataset is rather small
        self.loaded_imgs = []
        self.loaded_masks = []
        for idx in range(len(self.images)):
            img_path = self.images[idx]
            img = np.array(Image.open(img_path)).astype(np.uint8)

            mask_path = self.masks[idx]
            mask = np.load(mask_path, allow_pickle=True)
            inst_map = mask[()]["inst_map"].astype(np.int32)
            type_map = mask[()]["type_map"].astype(np.int32)
            mask = np.stack([inst_map, type_map], axis=-1)
            self.loaded_imgs.append(img)
            self.loaded_masks.append(mask)

        logger.info(f"Created Pannuke Dataset by using fold(s) {self.folds}")
        logger.info(f"Resulting dataset length: {self.__len__()}")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict, str, str]:
        """Get one dataset item consisting of transformed image,
        masks (instance_map, nuclei_type_map, nuclei_binary_map, hv_map) and tissue type as string

        Args:
            index (int): Index of element to retrieve

        Returns:
            Tuple[torch.Tensor, dict, str, str]:
                torch.Tensor: Image, with shape (3, H, W), shape is arbitrary for Lizzard (H and W approx. between 500 and 2000)
                dict:
                    "instance_map": Instance-Map, each instance is has one integer starting by 1 (zero is background), Shape (256, 256)
                    "nuclei_type_map": Nuclei-Type-Map, for each nucleus (instance) the class is indicated by an integer. Shape (256, 256)
                    "nuclei_binary_map": Binary Nuclei-Mask, Shape (256, 256)
                    "hv_map": Horizontal and vertical instance map.
                        Shape: (H, W, 2). First dimension is horizontal (horizontal gradient (-1 to 1)),
                        last is vertical (vertical gradient (-1 to 1)) Shape (256, 256, 2)
                    "dist_map": Probability distance map. Shape (256, 256)
                    "stardist_map": Stardist vector map. Shape (n_rays, 256, 256)
                    [Optional if regression]
                    "regression_map": Regression map. Shape (2, 256, 256). First is vertical, second horizontal.
                str: Tissue type
                str: Image Name
        """
        img_path = self.images[index]
        img = self.loaded_imgs[index]
        mask = self.loaded_masks[index]

        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        inst_map = mask[:, :, 0].copy()
        type_map = mask[:, :, 1].copy()
        np_map = mask[:, :, 0].copy()
        np_map[np_map > 0] = 1
        hv_map = PanNukeDataset.gen_instance_hv_map(inst_map)

        # torch convert
        img = torch.Tensor(img).type(torch.float32)
        img = img.permute(2, 0, 1)
        if torch.max(img) >= 5:
            img = img / 255

        masks = {
            "instance_map": torch.Tensor(inst_map).type(torch.int64),
            "nuclei_type_map": torch.Tensor(type_map).type(torch.int64),
            "nuclei_binary_map": torch.Tensor(np_map).type(torch.int64),
            "hv_map": torch.Tensor(hv_map).type(torch.float32),
        }
        if self.stardist:
            dist_map = PanNukeDataset.gen_distance_prob_maps(inst_map)
            stardist_map = PanNukeDataset.gen_stardist_maps(inst_map)
            masks["dist_map"] = torch.Tensor(dist_map).type(torch.float32)
            masks["stardist_map"] = torch.Tensor(stardist_map).type(torch.float32)
        if self.regression:
            masks["regression_map"] = PanNukeDataset.gen_regression_map(inst_map)

        return img, masks, "Colon", Path(img_path).name

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

    def load_cell_count(self):
        """Load Cell count from cell_count.csv file. File must be located inside the fold folder
        and named "cell_count.csv"

        Example file beginning:
            Image,Neutrophil,Epithelial,Lymphocyte,Plasma,Eosinophil,Connective
            consep_1_0000.png,0,117,0,0,0,0
            consep_1_0001.png,0,95,1,0,0,8
            consep_1_0002.png,0,172,3,0,0,2
            ...
        """
        df_placeholder = []
        for fold in self.folds:
            csv_path = self.dataset / f"fold{fold}" / "cell_count.csv"
            cell_count = pd.read_csv(csv_path, index_col=0)
            df_placeholder.append(cell_count)
        self.cell_count = pd.concat(df_placeholder)
        self.cell_count = self.cell_count.reindex(self.img_names)

    def get_sampling_weights_cell(self, gamma: float = 1) -> torch.Tensor:
        """Get sampling weights calculated by cell type statistics

        Args:
            gamma (float, optional): Gamma scaling factor, between 0 and 1.
                1 means total balancing, 0 means original weights. Defaults to 1.

        Returns:
            torch.Tensor: Weights for each sample
        """
        assert 0 <= gamma <= 1, "Gamma must be between 0 and 1"
        assert hasattr(self, "cell_count"), "Please run .load_cell_count() in advance!"
        binary_weight_factors = np.array([1069, 4189, 4356, 3103, 1025, 4527])
        k = np.sum(binary_weight_factors)
        cell_counts_imgs = np.clip(self.cell_count.to_numpy(), 0, 1)
        weight_vector = k / (gamma * binary_weight_factors + (1 - gamma) * k)
        img_weight = (1 - gamma) * np.max(cell_counts_imgs, axis=-1) + gamma * np.sum(
            cell_counts_imgs * weight_vector, axis=-1
        )
        img_weight[np.where(img_weight == 0)] = np.min(
            img_weight[np.nonzero(img_weight)]
        )

        return torch.Tensor(img_weight)

    # def get_sampling_weights_cell(self, gamma: float = 1) -> torch.Tensor:
    #     """Get sampling weights calculated by cell type statistics

    #     Args:
    #         gamma (float, optional): Gamma scaling factor, between 0 and 1.
    #             1 means total balancing, 0 means original weights. Defaults to 1.

    #     Returns:
    #         torch.Tensor: Weights for each sample
    #     """
    #     assert 0 <= gamma <= 1, "Gamma must be between 0 and 1"
    #     assert hasattr(self, "cell_count"), "Please run .load_cell_count() in advance!"
    #     binary_weight_factors = np.array([4012, 222017, 93612, 24793, 2999, 98783])
    #     k = np.sum(binary_weight_factors)
    #     cell_counts_imgs = self.cell_count.to_numpy()
    #     weight_vector = k / (gamma * binary_weight_factors + (1 - gamma) * k)
    #     img_weight = (1 - gamma) * np.max(cell_counts_imgs, axis=-1) + gamma * np.sum(
    #         cell_counts_imgs * weight_vector, axis=-1
    #     )
    #     img_weight[np.where(img_weight == 0)] = np.min(
    #         img_weight[np.nonzero(img_weight)]
    #     )

    #     return torch.Tensor(img_weight)
