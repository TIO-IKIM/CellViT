# -*- coding: utf-8 -*-
# StarDist Experiment Class
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import inspect
import os
import sys

import yaml

from base_ml.base_trainer import BaseTrainer

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from pathlib import Path
from typing import Callable, Tuple, Union

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    ExponentialLR,
    ReduceLROnPlateau,
    SequentialLR,
    _LRScheduler,
)
from torch.utils.data import Dataset
from torchinfo import summary

from base_ml.base_loss import retrieve_loss_fn
from cell_segmentation.experiments.experiment_cellvit_pannuke import (
    ExperimentCellVitPanNuke,
)
from cell_segmentation.trainer.trainer_stardist_debug import CellViTStarDistTrainerDebug

# from models.segmentation.cell_segmentation.cpp_net_stardist import (
#     StarDistRN50
# )
from models.segmentation.cell_segmentation.cpp_net_stardist_rn50 import StarDistRN50


class ExperimentCellViTStarDist(ExperimentCellVitPanNuke):
    def load_dataset_setup(self, dataset_path: Union[Path, str]) -> None:
        """Load the configuration of the PanNuke cell segmentation dataset.

        The dataset must have a dataset_config.yaml file in their dataset path with the following entries:
            * tissue_types: describing the present tissue types with corresponding integer
            * nuclei_types: describing the present nuclei types with corresponding integer

        Args:
            dataset_path (Union[Path, str]): Path to dataset folder
        """
        dataset_config_path = Path(dataset_path) / "dataset_config.yaml"
        with open(dataset_config_path, "r") as dataset_config_file:
            yaml_config = yaml.safe_load(dataset_config_file)
            self.dataset_config = dict(yaml_config)

    def get_loss_fn(self, loss_fn_settings: dict) -> dict:
        loss_fn_dict = {}
        loss_fn_dict["stardist_map"] = {
            "L1LossWeighted": {
                "loss_fn": retrieve_loss_fn("L1LossWeighted"),
                "weight": 1,
            },
        }
        loss_fn_dict["dist_map"] = {
            "bceweighted": {
                "loss_fn": retrieve_loss_fn("BCEWithLogitsLoss"),
                "weight": 1,
            },
        }
        loss_fn_dict["nuclei_type_map"] = {
            "bce": {"loss_fn": retrieve_loss_fn("xentropy_loss"), "weight": 1},
            "dice": {"loss_fn": retrieve_loss_fn("dice_loss"), "weight": 1},
        }
        return loss_fn_dict

    def get_scheduler(self, scheduler_type: str, optimizer: Optimizer) -> _LRScheduler:
        """Get the learning rate scheduler for CellViT

        The configuration of the scheduler is given in the "training" -> "scheduler" section.
        Currenlty, "constant", "exponential" and "cosine" schedulers are implemented.

        Required parameters for implemented schedulers:
            - "constant": None
            - "exponential": gamma (optional, defaults to 0.95)
            - "cosine": eta_min (optional, defaults to 1-e5)
            - "reducelronplateau": everything hardcoded right now, uses vall los for checking
        Args:
            scheduler_type (str): Type of scheduler as a string. Currently implemented:
                - "constant" (lowering by a factor of ten after 25 epochs, increasing after 50, decreasing again after 75)
                - "exponential" (ExponentialLR with given gamma, gamma defaults to 0.95)
                - "cosine" (CosineAnnealingLR, eta_min as parameter, defaults to 1-e5)
            optimizer (Optimizer): Optimizer

        Returns:
            _LRScheduler: PyTorch Scheduler
        """
        implemented_schedulers = [
            "constant",
            "exponential",
            "cosine",
            "reducelronplateau",
        ]
        if scheduler_type.lower() not in implemented_schedulers:
            self.logger.warning(
                f"Unknown Scheduler - No scheduler from the list {implemented_schedulers} select. Using default scheduling."
            )
        if scheduler_type.lower() == "constant":
            scheduler = SequentialLR(
                optimizer=optimizer,
                schedulers=[
                    ConstantLR(optimizer, factor=1, total_iters=25),
                    ConstantLR(optimizer, factor=0.1, total_iters=25),
                    ConstantLR(optimizer, factor=1, total_iters=25),
                    ConstantLR(optimizer, factor=0.1, total_iters=1000),
                ],
                milestones=[24, 49, 74],
            )
        elif scheduler_type.lower() == "exponential":
            scheduler = ExponentialLR(
                optimizer,
                gamma=self.run_conf["training"]["scheduler"].get("gamma", 0.95),
            )
        elif scheduler_type.lower() == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.run_conf["training"]["epochs"],
                eta_min=self.run_conf["training"]["scheduler"].get("eta_min", 1e-5),
            )
        elif scheduler_type.lower() == "reducelronplateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                min_lr=0.0000001,
                patience=10,
                threshold=1e-20,
            )
        else:
            scheduler = super().get_scheduler(optimizer)
        return scheduler

    def get_datasets(
        self,
        dataset_name: str,
        train_transforms: Callable = None,
        val_transforms: Callable = None,
    ) -> Tuple[Dataset, Dataset]:
        """Retrieve training dataset and validation dataset

        Args:
            dataset_name (str): Name of dataset to use
            train_transforms (Callable, optional): PyTorch transformations for train set. Defaults to None.
            val_transforms (Callable, optional): PyTorch transformations for validation set. Defaults to None.

        Returns:
            Tuple[Dataset, Dataset]: Training dataset and validation dataset
        """
        self.run_conf["data"]["stardist"] = True
        train_dataset, val_dataset = super().get_datasets(
            dataset_name=dataset_name,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
        )
        return train_dataset, val_dataset

    def get_train_model(
        self,
        pretrained_encoder: Union[Path, str] = None,
        pretrained_model: Union[Path, str] = None,
        backbone_type: str = "default",
        shared_decoders: bool = False,
        **kwargs,
    ) -> nn.Module:
        self.seed_run(self.default_conf["random_seed"])
        # model = UNetStarDist(
        #     n_channels=3,
        #     n_cls=6
        # )
        model = StarDistRN50()
        # model = CellViTSAMStarDist(
        #     model_path="/homes/fhoerst/histo-projects/CellViT/models/pretrained/SAM/sam_vit_b.pth",
        #     num_nuclei_classes=6,
        #     num_tissue_classes=19,
        #     vit_structure="SAM-B"
        # )
        # model = CellViTSAMStarDist(
        #     model_path="/homes/fhoerst/histo-projects/CellViT/models/pretrained/SAM/sam_vit_h.pth",
        #     num_nuclei_classes=6,
        #     num_tissue_classes=19,
        #     vit_structure="SAM-H"
        # )
        # model = CellViTSAMStarDist(
        #     model_path="/homes/fhoerst/histo-projects/CellViT/models/pretrained/SAM/sam_vit_l.pth",
        #     num_nuclei_classes=6,
        #     num_tissue_classes=19,
        #     vit_structure="SAM-L"
        # )
        # model = StarDistViTSAM(
        #     model_path="/homes/fhoerst/histo-projects/CellViT/models/pretrained/SAM/sam_vit_b.pth",
        #     num_nuclei_classes=6,
        #     num_tissue_classes=19,
        #     vit_structure="SAM-B"
        # )

        self.logger.info(f"\nModel: {model}")
        model = model.to("cpu")
        self.logger.info(
            f"\n{summary(model, input_size=(1, 3, 256, 256), device='cpu')}"
        )

        return model

    def get_trainer(self) -> BaseTrainer:
        """Return Trainer matching to this network

        Returns:
            BaseTrainer: Trainer
        """
        return CellViTStarDistTrainerDebug
