# -*- coding: utf-8 -*-
# Base cell segmentation dataset, based on torch Dataset implementation
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import logging
from typing import Callable

import torch
from torch.utils.data import Dataset

logger = logging.getLogger()
logger.addHandler(logging.NullHandler())

from abc import abstractmethod


class CellDataset(Dataset):
    def set_transforms(self, transforms: Callable) -> None:
        self.transforms = transforms

    @abstractmethod
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
        pass

    @abstractmethod
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

    @abstractmethod
    def get_sampling_weights_cell(self, gamma: float = 1) -> torch.Tensor:
        """Get sampling weights calculated by cell type statistics

        Args:
            gamma (float, optional): Gamma scaling factor, between 0 and 1.
                1 means total balancing, 0 means original weights. Defaults to 1.

        Returns:
            torch.Tensor: Weights for each sample
        """

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
