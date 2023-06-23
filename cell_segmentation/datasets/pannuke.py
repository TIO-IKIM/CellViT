# -*- coding: utf-8 -*-
# PanNuke Dataset
#
# Dataset information: https://arxiv.org/abs/2003.10778
# Please Prepare Dataset as described here
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
import yaml
from PIL import Image
from scipy.ndimage import measurements

from cell_segmentation.datasets.base_cell import CellDataset
from cell_segmentation.utils.tools import get_bounding_box

logger = logging.getLogger()
logger.addHandler(logging.NullHandler())


class PanNukeDataset(CellDataset):
    def __init__(
        self,
        dataset_path: Union[Path, str],
        folds: Union[int, list[int]],
        transforms: Callable = None,
    ) -> None:
        """PanNuke dataset

        Args:
            dataset_path (Union[Path, str]): Path to PanNuke dataset. Structure is described under ./docs/readmes/cell_segmentation.md
            folds (Union[int, list[int]]): Folds to use for this dataset
            transforms (Callable, optional): PyTorch transformations. Defaults to None.
        """
        if isinstance(folds, int):
            folds = [folds]

        self.dataset = Path(dataset_path).resolve()
        self.transforms = transforms
        self.images = []
        self.masks = []
        self.types = {}
        self.img_names = []
        self.folds = folds

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
            fold_types = pd.read_csv(self.dataset / f"fold{fold}" / "types.csv")
            fold_type_dict = fold_types.set_index("img")["type"].to_dict()
            self.types = {
                **self.types,
                **fold_type_dict,
            }  # careful - should all be named differently

        logger.info(f"Created Pannuke Dataset by using fold(s) {self.folds}")
        logger.info(f"Resulting dataset length: {self.__len__()}")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict, str, str]:
        """Get one dataset item consisting of transformed image,
        masks (instance_map, nuclei_type_map, nuclei_binary_map, hv_map) and tissue type as string

        Args:
            index (int): Index of element to retrieve

        Returns:
            Tuple[torch.Tensor, dict, str, str]:
                torch.Tensor: Image, with shape (3, H, W), in this case (3, 256, 256)
                dict:
                    "instance_map": Instance-Map, each instance is has one integer starting by 1 (zero is background), Shape (256, 256)
                    "nuclei_type_map": Nuclei-Type-Map, for each nucleus (instance) the class is indicated by an integer. Shape (256, 256)
                    "nuclei_binary_map": Binary Nuclei-Mask, Shape (256, 256)
                    "hv_map": Horizontal and vertical instance map.
                        Shape: (H, W, 2). First dimension is horizontal (horizontal gradient (-1 to 1)),
                        last is vertical (vertical gradient (-1 to 1)) Shape (256, 256, 2)
                str: Tissue type
                str: Image Name
        """
        img_path = self.images[index]
        img = np.array(Image.open(img_path)).astype(np.uint8)

        mask_path = self.masks[index]
        mask = np.load(mask_path, allow_pickle=True)
        inst_map = mask[()]["inst_map"].astype(np.int32)
        type_map = mask[()]["type_map"].astype(np.int32)
        mask = np.stack([inst_map, type_map], axis=-1)

        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        tissue_type = self.types[img_path.name]
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

        return img, masks, tissue_type, Path(img_path).name

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

        Example file beginning:
            Image,Neoplastic,Inflammatory,Connective,Dead,Epithelial
            0_0.png,4,2,2,0,0
            0_1.png,8,1,1,0,0
            0_10.png,17,0,1,0,0
            0_100.png,10,0,11,0,0
            ...
        """
        df_placeholder = []
        for fold in self.folds:
            csv_path = self.dataset / f"fold{fold}" / "cell_count.csv"
            cell_count = pd.read_csv(csv_path, index_col=0)
            df_placeholder.append(cell_count)
        self.cell_count = pd.concat(df_placeholder)
        self.cell_count = self.cell_count.reindex(self.img_names)

    def get_sampling_weights_tissue(self, gamma: float = 1) -> torch.Tensor:
        """Get sampling weights calculated by tissue type statistics

        For this, a file named "weight_config.yaml" with the content:
            tissue:
                tissue_1: xxx
                tissue_2: xxx (name of tissue: count)
                ...
        Must exists in the dataset main folder (parent path, not inside the folds)

        Args:
            gamma (float, optional): Gamma scaling factor, between 0 and 1.
                1 means total balancing, 0 means original weights. Defaults to 1.

        Returns:
            torch.Tensor: Weights for each sample
        """
        assert 0 <= gamma <= 1, "Gamma must be between 0 and 1"
        with open(
            (self.dataset / "weight_config.yaml").resolve(), "r"
        ) as run_config_file:
            yaml_config = yaml.safe_load(run_config_file)
            tissue_counts = dict(yaml_config)["tissue"]

        # calculate weight for each tissue
        weights_dict = {}
        k = np.sum(list(tissue_counts.values()))
        for tissue, count in tissue_counts.items():
            w = k / (gamma * count + (1 - gamma) * k)
            weights_dict[tissue] = w

        weights = []
        for idx in range(self.__len__()):
            img_idx = self.img_names[idx]
            type_str = self.types[img_idx]
            weights.append(weights_dict[type_str])

        return torch.Tensor(weights)

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
        binary_weight_factors = np.array([4191, 4132, 6140, 232, 1528])
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

    def get_sampling_weights_cell_tissue(self, gamma: float = 1) -> torch.Tensor:
        """Get combined sampling weights by calculating tissue and cell sampling weights,
        normalizing them and adding them up to yield one score.

        Args:
            gamma (float, optional): Gamma scaling factor, between 0 and 1.
                1 means total balancing, 0 means original weights. Defaults to 1.

        Returns:
            torch.Tensor: Weights for each sample
        """
        assert 0 <= gamma <= 1, "Gamma must be between 0 and 1"
        tw = self.get_sampling_weights_tissue(gamma)
        cw = self.get_sampling_weights_cell(gamma)
        weights = tw / torch.max(tw) + cw / torch.max(cw)

        return weights

    @staticmethod
    def gen_instance_hv_map(inst_map: np.ndarray) -> np.ndarray:
        """Obtain the horizontal and vertical distance maps for each
        nuclear instance.

        Args:
            inst_map (np.ndarray): Instance map with each instance labelled as a unique integer
                Shape: (H, W)
        Returns:
            np.ndarray: Horizontal and vertical instance map.
                Shape: (H, W, 2). First dimension is horizontal (horizontal gradient (-1 to 1)),
                last is vertical (vertical gradient (-1 to 1))
        """
        orig_inst_map = inst_map.copy()  # instance ID map

        x_map = np.zeros(orig_inst_map.shape[:2], dtype=np.float32)
        y_map = np.zeros(orig_inst_map.shape[:2], dtype=np.float32)

        inst_list = list(np.unique(orig_inst_map))
        inst_list.remove(0)  # 0 is background
        for inst_id in inst_list:
            inst_map = np.array(orig_inst_map == inst_id, np.uint8)
            inst_box = get_bounding_box(inst_map)

            # expand the box by 2px
            # Because we first pad the ann at line 207, the bboxes
            # will remain valid after expansion
            if inst_box[0] >= 2:
                inst_box[0] -= 2
            if inst_box[2] >= 2:
                inst_box[2] -= 2
            if inst_box[1] <= orig_inst_map.shape[0] - 2:
                inst_box[1] += 2
            if inst_box[3] <= orig_inst_map.shape[0] - 2:
                inst_box[3] += 2

            # improvement
            inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

            if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
                continue

            # instance center of mass, rounded to nearest pixel
            inst_com = list(measurements.center_of_mass(inst_map))

            inst_com[0] = int(inst_com[0] + 0.5)
            inst_com[1] = int(inst_com[1] + 0.5)

            inst_x_range = np.arange(1, inst_map.shape[1] + 1)
            inst_y_range = np.arange(1, inst_map.shape[0] + 1)
            # shifting center of pixels grid to instance center of mass
            inst_x_range -= inst_com[1]
            inst_y_range -= inst_com[0]

            inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

            # remove coord outside of instance
            inst_x[inst_map == 0] = 0
            inst_y[inst_map == 0] = 0
            inst_x = inst_x.astype("float32")
            inst_y = inst_y.astype("float32")

            # normalize min into -1 scale
            if np.min(inst_x) < 0:
                inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
            if np.min(inst_y) < 0:
                inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
            # normalize max into +1 scale
            if np.max(inst_x) > 0:
                inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
            if np.max(inst_y) > 0:
                inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

            ####
            x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            x_map_box[inst_map > 0] = inst_x[inst_map > 0]

            y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            y_map_box[inst_map > 0] = inst_y[inst_map > 0]

        hv_map = np.dstack([x_map, y_map])
        return hv_map

    # def fix_mirror_padding(inst_map):
    #     """Deal with duplicated instances due to mirroring in interpolation
    #     during shape augmentation (scale, rotation etc.).
    #     """
    #     current_max_id = np.amax(inst_map)
    #     inst_list = list(np.unique(inst_map))
    #     inst_list.remove(0)  # 0 is background
    #     for inst_id in inst_list:
    #         inst_map = np.array(inst_map == inst_id, np.uint8)
    #         remapped_ids = measurements.label(inst_map)[0]
    #         remapped_ids[remapped_ids > 1] += current_max_id
    #         inst_map[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
    #         current_max_id = np.amax(inst_map)
    #     return inst_map
