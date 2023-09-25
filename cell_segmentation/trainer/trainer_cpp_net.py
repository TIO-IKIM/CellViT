# -*- coding: utf-8 -*-
# CPP Trainer Class
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

# import wandb
from matplotlib import pyplot as plt
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from base_ml.base_early_stopping import EarlyStopping
from models.segmentation.cell_segmentation.cellvit_cpp_net import (
    DataclassCPPStorage,
)

from cell_segmentation.trainer.trainer_stardist import CellViTStarDistTrainer
from models.segmentation.cell_segmentation.cellvit import CellViT

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)


class CellViTCPPTrainer(CellViTStarDistTrainer):
    """CellViTCPP trainer class

    Args:
        model (CellViTCPP): CellViTCPP model that should be trained
        loss_fn_dict (dict): Dictionary with loss functions for each branch with a dictionary of loss functions.
            Name of branch as top-level key, followed by a dictionary with loss name, loss fn and weighting factor
            Example:
            {
                "dist_map": {"bce": {loss_fn(Callable), weight_factor(float)}, "dice": {loss_fn(Callable), weight_factor(float)}},
                "stardist_map": {"bce": {loss_fn(Callable), weight_factor(float)}, "dice": {loss_fn(Callable), weight_factor(float)}},
                "stardist_map_refined": {"bce": {loss_fn(Callable), weight_factor(float)}, "dice": {loss_fn(Callable), weight_factor(float)}},
                "nuclei_type_map": {"bce": {loss_fn(Callable), weight_factor(float)}, "dice": {loss_fn(Callable), weight_factor(float)}}
                "tissue_types": {"ce": {loss_fn(Callable), weight_factor(float)}}
            }
            Required Keys are:
                * dist_map
                * stardist_map
                * stardist_map_refined
                * nuclei_type_map
                * tissue types
        optimizer (Optimizer): Optimizer
        scheduler (_LRScheduler): Learning rate scheduler
        device (str): Cuda device to use, e.g., cuda:0.
        logger (logging.Logger): Logger module
        logdir (Union[Path, str]): Logging directory
        num_classes (int): Number of nuclei classes
        dataset_config (dict): Dataset configuration. Required Keys are:
            * "tissue_types": describing the present tissue types with corresponding integer
            * "nuclei_types": describing the present nuclei types with corresponding integer
        experiment_config (dict): Configuration of this experiment
        early_stopping (EarlyStopping, optional):  Early Stopping Class. Defaults to None.
        log_images (bool, optional): If images should be logged to WandB. Defaults to False.
        magnification (int, optional): Image magnification. Please select either 40 or 20. Defaults to 40.
        mixed_precision (bool, optional): If mixed-precision should be used. Defaults to False.
    """

    def __init__(
        self,
        model: CellViT,
        loss_fn_dict: dict,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        device: str,
        logger: logging.Logger,
        logdir: Union[Path, str],
        num_classes: int,
        dataset_config: dict,
        experiment_config: dict,
        early_stopping: EarlyStopping = None,
        log_images: bool = False,
        magnification: int = 40,
        mixed_precision: bool = False,
    ):
        super().__init__(
            model=model,
            loss_fn_dict=loss_fn_dict,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            logdir=logdir,
            num_classes=num_classes,
            dataset_config=dataset_config,
            experiment_config=experiment_config,
            early_stopping=early_stopping,
            log_images=log_images,
            magnification=magnification,
            mixed_precision=mixed_precision,
        )

    def unpack_predictions(
        self, predictions: dict, skip_postprocessing: bool = False
    ) -> DataclassCPPStorage:
        """Unpack the given predictions. Main focus lays on reshaping and postprocessing predictions, e.g. separating instances

        Args:
            predictions (dict): Dictionary with the following keys:
                * tissue_types: Logit tissue prediction output. Shape: (batch_size, num_tissue_classes)
                * stardist_map: Stardist output for vector prediction. Shape: (batch_size, n_rays, H, W)
                * stardist_map_refined: Stardist output for vector prediction, but refined by CPP-Net. Shape: (batch_size, n_rays, H, W)
                * dist_map: Logit output for distance map. Shape: (batch_size, 1, H, W)
                * (Optional)
                * nuclei_type_map: Logit output for nuclei instance-prediction. Shape: (batch_size, num_nuclei_classes, H, W)
            skip_postprocessing (bool, optional): If true, postprocesssing for separating nuclei and creating maps is skipped.
                Helpfull for speeding up training. Defaults to False.
        Returns:
            DataclassCPPStorage: Processed network output
        """
        predictions["nuclei_type_map"] = F.softmax(
            predictions["nuclei_type_map"], dim=1
        )  # shape: (batch_size, num_nuclei_classes, H, W)
        predictions["dist_map_sigmoid"] = F.sigmoid(predictions["dist_map"])
        # postprocessing: apply NMS and StarDist postprocessing to generate binary and multiclass cell detections

        if not skip_postprocessing:
            (
                instance_map,
                predictions["instance_types"],
                instance_types_nuclei,
            ) = self.model.calculate_instance_map(
                predictions["dist_map_sigmoid"],
                predictions["stardist_map_refined"],
                predictions["nuclei_type_map"],
            )
            instance_map = instance_map.to(self.device)
            instance_types_nuclei = instance_types_nuclei.to(self.device)
            predictions["instance_map"] = instance_map
            predictions["instance_types_nuclei"] = instance_types_nuclei

        predictions = DataclassCPPStorage(
            **predictions,
            batch_size=predictions["nuclei_type_map"].shape[0],
        )

        return predictions

    def unpack_masks(self, masks: dict, tissue_types: list) -> DataclassCPPStorage:
        """Unpack the given masks. Main focus lays on reshaping and postprocessing masks to generate one dict

        Args:
            masks (dict): Required keys are:
                * instance_map: Pixel-wise nuclear instance segmentations. Shape: (batch_size, H, W)
                * nuclei_type_map: Nuclei instance-prediction and segmentation (not binary, each instance has own integer). Shape: (batch_size, H, W)
                * nuclei_binary_map: Binary nuclei segmentations. Shape: (batch_size, H, W)
                * dist_map: Probability distance map. Shape: (batch_size, H, W)
                * stardist_map: Stardist output. Shape: (batch_size, n_rays H, W)

            tissue_types (list): List of string names of ground-truth tissue types

        Returns:
            DataclassCPPStorage: Output ground truth values

        """
        nuclei_type_maps = torch.squeeze(masks["nuclei_type_map"]).type(torch.int64)
        gt_nuclei_type_maps_onehot = F.one_hot(
            nuclei_type_maps, num_classes=self.num_classes
        ).type(
            torch.float32
        )  # background + nuclei types

        # # assemble ground truth dictionary
        gt = {
            "nuclei_type_map": gt_nuclei_type_maps_onehot.permute(0, 3, 1, 2).to(
                self.device
            ),  # shape: (batch_size, num_nuclei_classes, H, W)
            "stardist_map": masks["stardist_map"].to(
                self.device
            ),  # shape: (batch_size, nrays, H, W)
            "stardist_map_refined": masks["stardist_map"].to(
                self.device
            ),  # shape: (batch_size, nrays, H, W)
            "dist_map": masks["dist_map"].to(self.device)[
                :, None, :, :
            ],  # shape: (batch_size, 1, H, W), TODO: check if None is necessary because of shape?
            "instance_map": masks["instance_map"].to(
                self.device
            ),  # shape: (batch_size, H, W) -> each instance has one integer
            "instance_types_nuclei": (
                gt_nuclei_type_maps_onehot * masks["instance_map"][..., None]
            )
            .permute(0, 3, 1, 2)
            .to(
                self.device
            ),  # shape: (batch_size, num_nuclei_classes, H, W) -> instance has one integer, for each nuclei class
            "tissue_types": torch.Tensor([self.tissue_types[t] for t in tissue_types])
            .type(torch.LongTensor)
            .to(self.device),  # shape: batch_size
        }
        gt = DataclassCPPStorage(**gt, batch_size=gt["tissue_types"].shape[0])
        return gt

    @staticmethod
    def generate_example_image(
        imgs: Union[torch.Tensor, np.ndarray],
        predictions: dict,
        ground_truth: dict,
        num_nuclei_classes: int,
        num_images: int = 2,
    ) -> plt.Figure:
        # TODO: implement
        return None
