# -*- coding: utf-8 -*-
# CellVit Experiment Class
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import copy
import datetime
import inspect
import os
import shutil
import sys

import yaml

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import uuid
from pathlib import Path
from typing import Callable, Tuple, Union

import albumentations as A
import torch
import torch.nn as nn
import wandb
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    ExponentialLR,
    SequentialLR,
    _LRScheduler,
)
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
    Subset,
    WeightedRandomSampler,
)
from torchinfo import summary
from wandb.sdk.lib.runid import generate_id

from base_ml.base_early_stopping import EarlyStopping
from base_ml.base_experiment import BaseExperiment
from base_ml.base_loss import retrieve_loss_fn
from base_ml.base_trainer import BaseTrainer
from cell_segmentation.datasets.base_cell import CellDataset
from cell_segmentation.datasets.dataset_coordinator import select_dataset
from cell_segmentation.trainer.trainer_cellvit import CellViTTrainer
from models.segmentation.cell_segmentation.cellvit import (
    CellViT,
    CellViTSAM,
    CellViT256,
)
from models.segmentation.cell_segmentation.cellvit_shared import (
    CellViTShared,
    CellViT256Shared,
    CellViTSAMShared,
)
from utils.tools import close_logger


class ExperimentCellVitPanNuke(BaseExperiment):
    def __init__(self, default_conf: dict, checkpoint=None) -> None:
        super().__init__(default_conf, checkpoint)
        self.load_dataset_setup(dataset_path=self.default_conf["data"]["dataset_path"])

    def run_experiment(self) -> tuple[Path, dict, nn.Module, dict]:
        """Main Experiment Code"""
        ### Setup
        # close loggers
        self.close_remaining_logger()

        # get the config for the current run
        self.run_conf = copy.deepcopy(self.default_conf)
        self.run_conf["dataset_config"] = self.dataset_config
        self.run_name = f"{datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')}_{self.run_conf['logging']['log_comment']}"

        wandb_run_id = generate_id()
        resume = None
        if self.checkpoint is not None:
            wandb_run_id = self.checkpoint["wandb_id"]
            resume = "must"
            self.run_name = self.checkpoint["run_name"]

        # initialize wandb
        run = wandb.init(
            project=self.run_conf["logging"]["project"],
            tags=self.run_conf["logging"].get("tags", []),
            name=self.run_name,
            notes=self.run_conf["logging"]["notes"],
            dir=self.run_conf["logging"]["wandb_dir"],
            mode=self.run_conf["logging"]["mode"].lower(),
            group=self.run_conf["logging"].get("group", str(uuid.uuid4())),
            allow_val_change=True,
            id=wandb_run_id,
            resume=resume,
            settings=wandb.Settings(start_method="fork"),
        )

        # get ids
        self.run_conf["logging"]["run_id"] = run.id
        self.run_conf["logging"]["wandb_file"] = run.id

        # overwrite configuration with sweep values are leave them as they are
        if self.run_conf["run_sweep"] is True:
            self.run_conf["logging"]["sweep_id"] = run.sweep_id
            self.run_conf["logging"]["log_dir"] = str(
                Path(self.default_conf["logging"]["log_dir"])
                / f"sweep_{run.sweep_id}"
                / f"{self.run_name}_{self.run_conf['logging']['run_id']}"
            )
            self.overwrite_sweep_values(self.run_conf, run.config)
        else:
            self.run_conf["logging"]["log_dir"] = str(
                Path(self.default_conf["logging"]["log_dir"]) / self.run_name
            )

        # update wandb
        wandb.config.update(
            self.run_conf, allow_val_change=True
        )  # this may lead to the problem

        # create output folder, instantiate logger and store config
        self.create_output_dir(self.run_conf["logging"]["log_dir"])
        self.logger = self.instantiate_logger()
        self.logger.info("Instantiated Logger. WandB init and config update finished.")
        self.logger.info(f"Run ist stored here: {self.run_conf['logging']['log_dir']}")
        self.store_config()

        self.logger.info(
            f"Cuda devices: {[torch.cuda.device(i) for i in range(torch.cuda.device_count())]}"
        )
        ### Machine Learning
        device = f"cuda:{self.run_conf['gpu']}"
        self.logger.info(f"Using GPU: {device}")
        self.logger.info(f"Using device: {device}")

        # loss functions
        loss_fn_dict = self.get_loss_fn(self.run_conf.get("loss", {}))
        self.logger.info("Loss functions:")
        self.logger.info(loss_fn_dict)

        # model
        model = self.get_train_model(
            pretrained_encoder=self.run_conf["model"].get("pretrained_encoder", None),
            pretrained_model=self.run_conf["model"].get("pretrained", None),
            backbone_type=self.run_conf["model"].get("backbone", "default"),
            shared_decoders=self.run_conf["model"].get("shared_decoders", False),
            regression_loss=self.run_conf["model"].get("regression_loss", False),
        )
        model.to(device)

        # optimizer
        optimizer = self.get_optimizer(
            model,
            self.run_conf["training"]["optimizer"],
            self.run_conf["training"]["optimizer_hyperparameter"],
        )

        # scheduler
        scheduler = self.get_scheduler(
            optimizer=optimizer,
            scheduler_type=self.run_conf["training"]["scheduler"]["scheduler_type"],
        )

        # early stopping (no early stopping for basic setup)
        early_stopping = None
        if "early_stopping_patience" in self.run_conf["training"]:
            if self.run_conf["training"]["early_stopping_patience"] is not None:
                early_stopping = EarlyStopping(
                    patience=self.run_conf["training"]["early_stopping_patience"],
                    strategy="maximize",
                )

        ### Data handling
        train_transforms, val_transforms = self.get_transforms(
            self.run_conf["transformations"],
            input_shape=self.run_conf["data"].get("input_shape", 256),
        )

        train_dataset, val_dataset = self.get_datasets(
            train_transforms=train_transforms,
            val_transforms=val_transforms,
        )

        # load sampler
        training_sampler = self.get_sampler(
            train_dataset=train_dataset,
            strategy=self.run_conf["training"].get("sampling_strategy", "random"),
            gamma=self.run_conf["training"].get("sampling_gamma", 1),
        )

        # define dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.run_conf["training"]["batch_size"],
            sampler=training_sampler,
            num_workers=16,
            pin_memory=False,
            worker_init_fn=self.seed_worker,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=128,
            num_workers=16,
            pin_memory=True,
            worker_init_fn=self.seed_worker,
        )

        # start Training
        self.logger.info("Instantiate Trainer")
        trainer_fn = self.get_trainer()
        trainer = trainer_fn(
            model=model,
            loss_fn_dict=loss_fn_dict,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=self.logger,
            logdir=self.run_conf["logging"]["log_dir"],
            num_classes=self.run_conf["data"]["num_nuclei_classes"],
            dataset_config=self.dataset_config,
            early_stopping=early_stopping,
            experiment_config=self.run_conf,
            log_images=self.run_conf["logging"].get("log_images", False),
            magnification=self.run_conf["data"].get("magnification", 40),
            mixed_precision=self.run_conf["training"].get("mixed_precision", False),
        )

        # Load checkpoint if provided
        if self.checkpoint is not None:
            self.logger.info("Checkpoint was provided. Restore ...")
            trainer.resume_checkpoint(self.checkpoint)

        # Call fit method
        self.logger.info("Calling Trainer Fit")
        trainer.fit(
            epochs=self.run_conf["training"]["epochs"],
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            metric_init=self.get_wandb_init_dict(),
            unfreeze_epoch=self.run_conf["training"]["unfreeze_epoch"],
            eval_every=self.run_conf["training"].get("eval_every", 1),
        )

        # Select best model if not provided by early stopping
        checkpoint_dir = Path(self.run_conf["logging"]["log_dir"]) / "checkpoints"
        if not (checkpoint_dir / "model_best.pth").is_file():
            shutil.copy(
                checkpoint_dir / "latest_checkpoint.pth",
                checkpoint_dir / "model_best.pth",
            )

        # At the end close logger
        self.logger.info(f"Finished run {run.id}")
        close_logger(self.logger)

        return self.run_conf["logging"]["log_dir"]

    def load_dataset_setup(self, dataset_path: Union[Path, str]) -> None:
        """Load the configuration of the cell segmentation dataset.

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
        """Create a dictionary with loss functions for all branches

        Branches: "nuclei_binary_map", "hv_map", "nuclei_type_map", "tissue_types"

        Args:
            loss_fn_settings (dict): Dictionary with the loss function settings. Structure
            branch_name(str):
                loss_name(str):
                    loss_fn(str): String matching to the loss functions defined in the LOSS_DICT (base_ml.base_loss)
                    weight(float): Weighting factor as float value
                    (optional) args:  Optional parameters for initializing the loss function
                            arg_name: value

            If a branch is not provided, the defaults settings (described below) are used.

            For further information, please have a look at the file configs/examples/cell_segmentation/train_cellvit.yaml
            under the section "loss"

            Example:
                  nuclei_binary_map:
                    bce:
                        loss_fn: xentropy_loss
                        weight: 1
                    dice:
                        loss_fn: dice_loss
                        weight: 1

        Returns:
            dict: Dictionary with loss functions for each branch. Structure:
                branch_name(str):
                    loss_name(str):
                        "loss_fn": Callable loss function
                        "weight": weight of the loss since in the end all losses of all branches are added together for backward pass
                    loss_name(str):
                        "loss_fn": Callable loss function
                        "weight": weight of the loss since in the end all losses of all branches are added together for backward pass
                branch_name(str)
                ...

        Default loss dictionary:
            nuclei_binary_map:
                bce:
                    loss_fn: xentropy_loss
                    weight: 1
                dice:
                    loss_fn: dice_loss
                    weight: 1
            hv_map:
                mse:
                    loss_fn: mse_loss_maps
                    weight: 1
                msge:
                    loss_fn: msge_loss_maps
                    weight: 1
            nuclei_type_map
                bce:
                    loss_fn: xentropy_loss
                    weight: 1
                dice:
                    loss_fn: dice_loss
                    weight: 1
            tissue_types
                ce:
                    loss_fn: nn.CrossEntropyLoss()
                    weight: 1
        """
        loss_fn_dict = {}
        if "nuclei_binary_map" in loss_fn_settings.keys():
            loss_fn_dict["nuclei_binary_map"] = {}
            for loss_name, loss_sett in loss_fn_settings["nuclei_binary_map"].items():
                parameters = loss_sett.get("args", {})
                loss_fn_dict["nuclei_binary_map"][loss_name] = {
                    "loss_fn": retrieve_loss_fn(loss_sett["loss_fn"], **parameters),
                    "weight": loss_sett["weight"],
                }
        else:
            loss_fn_dict["nuclei_binary_map"] = {
                "bce": {"loss_fn": retrieve_loss_fn("xentropy_loss"), "weight": 1},
                "dice": {"loss_fn": retrieve_loss_fn("dice_loss"), "weight": 1},
            }
        if "hv_map" in loss_fn_settings.keys():
            loss_fn_dict["hv_map"] = {}
            for loss_name, loss_sett in loss_fn_settings["hv_map"].items():
                parameters = loss_sett.get("args", {})
                loss_fn_dict["hv_map"][loss_name] = {
                    "loss_fn": retrieve_loss_fn(loss_sett["loss_fn"], **parameters),
                    "weight": loss_sett["weight"],
                }
        else:
            loss_fn_dict["hv_map"] = {
                "mse": {"loss_fn": retrieve_loss_fn("mse_loss_maps"), "weight": 1},
                "msge": {"loss_fn": retrieve_loss_fn("msge_loss_maps"), "weight": 1},
            }
        if "nuclei_type_map" in loss_fn_settings.keys():
            loss_fn_dict["nuclei_type_map"] = {}
            for loss_name, loss_sett in loss_fn_settings["nuclei_type_map"].items():
                parameters = loss_sett.get("args", {})
                loss_fn_dict["nuclei_type_map"][loss_name] = {
                    "loss_fn": retrieve_loss_fn(loss_sett["loss_fn"], **parameters),
                    "weight": loss_sett["weight"],
                }
        else:
            loss_fn_dict["nuclei_type_map"] = {
                "bce": {"loss_fn": retrieve_loss_fn("xentropy_loss"), "weight": 1},
                "dice": {"loss_fn": retrieve_loss_fn("dice_loss"), "weight": 1},
            }
        if "tissue_types" in loss_fn_settings.keys():
            loss_fn_dict["tissue_types"] = {}
            for loss_name, loss_sett in loss_fn_settings["tissue_types"].items():
                parameters = loss_sett.get("args", {})
                loss_fn_dict["tissue_types"][loss_name] = {
                    "loss_fn": retrieve_loss_fn(loss_sett["loss_fn"], **parameters),
                    "weight": loss_sett["weight"],
                }
        else:
            loss_fn_dict["tissue_types"] = {
                "ce": {"loss_fn": nn.CrossEntropyLoss(), "weight": 1},
            }
        if "regression_loss" in loss_fn_settings.keys():
            loss_fn_dict["regression_map"] = {}
            for loss_name, loss_sett in loss_fn_settings["regression_loss"].items():
                parameters = loss_sett.get("args", {})
                loss_fn_dict["regression_map"][loss_name] = {
                    "loss_fn": retrieve_loss_fn(loss_sett["loss_fn"], **parameters),
                    "weight": loss_sett["weight"],
                }
        elif "regression_loss" in self.run_conf["model"].keys():
            loss_fn_dict["regression_map"] = {
                "mse": {"loss_fn": retrieve_loss_fn("mse_loss_maps"), "weight": 1},
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

        Args:
            scheduler_type (str): Type of scheduler as a string. Currently implemented:
                - "constant" (lowering by a factor of ten after 25 epochs, increasing after 50, decreasimg again after 75)
                - "exponential" (ExponentialLR with given gamma, gamma defaults to 0.95)
                - "cosine" (CosineAnnealingLR, eta_min as parameter, defaults to 1-e5)
            optimizer (Optimizer): Optimizer

        Returns:
            _LRScheduler: PyTorch Scheduler
        """
        implemented_schedulers = ["constant", "exponential", "cosine"]
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
        else:
            scheduler = super().get_scheduler(optimizer)
        return scheduler

    def get_datasets(
        self,
        train_transforms: Callable = None,
        val_transforms: Callable = None,
    ) -> Tuple[Dataset, Dataset]:
        """Retrieve training dataset and validation dataset

        Args:
            train_transforms (Callable, optional): PyTorch transformations for train set. Defaults to None.
            val_transforms (Callable, optional): PyTorch transformations for validation set. Defaults to None.

        Returns:
            Tuple[Dataset, Dataset]: Training dataset and validation dataset
        """
        if (
            "val_split" in self.run_conf["data"]
            and "val_folds" in self.run_conf["data"]
        ):
            raise RuntimeError(
                "Provide either val_splits or val_folds in configuration file, not both."
            )
        if (
            "val_split" not in self.run_conf["data"]
            and "val_folds" not in self.run_conf["data"]
        ):
            raise RuntimeError(
                "Provide either val_split or val_folds in configuration file, one is necessary."
            )
        if (
            "val_split" not in self.run_conf["data"]
            and "val_folds" not in self.run_conf["data"]
        ):
            raise RuntimeError(
                "Provide either val_split or val_fold in configuration file, one is necessary."
            )
        if "regression_loss" in self.run_conf["model"].keys():
            self.run_conf["data"]["regression_loss"] = True

        full_dataset = select_dataset(
            dataset_name="pannuke",
            split="train",
            dataset_config=self.run_conf["data"],
            transforms=train_transforms,
        )
        if "val_split" in self.run_conf["data"]:
            generator_split = torch.Generator().manual_seed(
                self.default_conf["random_seed"]
            )
            val_splits = float(self.run_conf["data"]["val_split"])
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset,
                lengths=[1 - val_splits, val_splits],
                generator=generator_split,
            )
            val_dataset.dataset = copy.deepcopy(full_dataset)
            val_dataset.dataset.set_transforms(val_transforms)
        else:
            train_dataset = full_dataset
            val_dataset = select_dataset(
                dataset_name="pannuke",
                split="validation",
                dataset_config=self.run_conf["data"],
                transforms=val_transforms,
            )

        return train_dataset, val_dataset

    def get_train_model(
        self,
        pretrained_encoder: Union[Path, str] = None,
        pretrained_model: Union[Path, str] = None,
        backbone_type: str = "default",
        shared_decoders: bool = False,
        regression_loss: bool = False,
        **kwargs,
    ) -> CellViT:
        """Return the CellViT training model

        Args:
            pretrained_encoder (Union[Path, str]): Path to a pretrained encoder. Defaults to None.
            pretrained_model (Union[Path, str], optional): Path to a pretrained model. Defaults to None.
            backbone_type (str, optional): Backbone Type. Currently supported are default (None, ViT256, SAM-B, SAM-L, SAM-H). Defaults to None
            shared_decoders (bool, optional): If shared skip decoders should be used. Defaults to False.
            regression_loss (bool, optional): If regression loss is used. Defaults to False

        Returns:
            CellViT: CellViT training model with given setup
        """
        # reseed needed, due to subprocess seeding compatibility
        self.seed_run(self.default_conf["random_seed"])

        # check for backbones
        implemented_backbones = ["default", "vit256", "sam-b", "sam-l", "sam-h"]
        if backbone_type.lower() not in implemented_backbones:
            raise NotImplementedError(
                f"Unknown Backbone Type - Currently supported are: {implemented_backbones}"
            )
        if backbone_type.lower() == "default":
            if shared_decoders:
                model_class = CellViTShared
            else:
                model_class = CellViT
            model = model_class(
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                embed_dim=self.run_conf["model"]["embed_dim"],
                input_channels=self.run_conf["model"].get("input_channels", 3),
                depth=self.run_conf["model"]["depth"],
                num_heads=self.run_conf["model"]["num_heads"],
                extract_layers=self.run_conf["model"]["extract_layers"],
                drop_rate=self.run_conf["training"].get("drop_rate", 0),
                attn_drop_rate=self.run_conf["training"].get("attn_drop_rate", 0),
                drop_path_rate=self.run_conf["training"].get("drop_path_rate", 0),
                regression_loss=regression_loss,
            )

            if pretrained_model is not None:
                self.logger.info(
                    f"Loading pretrained CellViT model from path: {pretrained_model}"
                )
                cellvit_pretrained = torch.load(pretrained_model)
                self.logger.info(model.load_state_dict(cellvit_pretrained, strict=True))
                self.logger.info("Loaded CellViT model")

        if backbone_type.lower() == "vit256":
            if shared_decoders:
                model_class = CellViT256Shared
            else:
                model_class = CellViT256
            model = model_class(
                model256_path=pretrained_encoder,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                drop_rate=self.run_conf["training"].get("drop_rate", 0),
                attn_drop_rate=self.run_conf["training"].get("attn_drop_rate", 0),
                drop_path_rate=self.run_conf["training"].get("drop_path_rate", 0),
                regression_loss=regression_loss,
            )
            model.load_pretrained_encoder(model.model256_path)
            if pretrained_model is not None:
                self.logger.info(
                    f"Loading pretrained CellViT model from path: {pretrained_model}"
                )
                cellvit_pretrained = torch.load(pretrained_model, map_location="cpu")
                self.logger.info(model.load_state_dict(cellvit_pretrained, strict=True))
            model.freeze_encoder()
            self.logger.info("Loaded CellVit256 model")
        if backbone_type.lower() in ["sam-b", "sam-l", "sam-h"]:
            if shared_decoders:
                model_class = CellViTSAMShared
            else:
                model_class = CellViTSAM
            model = model_class(
                model_path=pretrained_encoder,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                vit_structure=backbone_type,
                drop_rate=self.run_conf["training"].get("drop_rate", 0),
                regression_loss=regression_loss,
            )
            model.load_pretrained_encoder(model.model_path)
            if pretrained_model is not None:
                self.logger.info(
                    f"Loading pretrained CellViT model from path: {pretrained_model}"
                )
                cellvit_pretrained = torch.load(pretrained_model, map_location="cpu")
                self.logger.info(model.load_state_dict(cellvit_pretrained, strict=True))
            model.freeze_encoder()
            self.logger.info(f"Loaded CellViT-SAM model with backbone: {backbone_type}")

        self.logger.info(f"\nModel: {model}")
        model = model.to("cpu")
        self.logger.info(
            f"\n{summary(model, input_size=(1, 3, 256, 256), device='cpu')}"
        )

        return model

    def get_wandb_init_dict(self) -> dict:
        pass

    def get_transforms(
        self, transform_settings: dict, input_shape: int = 256
    ) -> Tuple[Callable, Callable]:
        """Get Transformations (Albumentation Transformations). Return both training and validation transformations.

        The transformation settings are given in the following format:
            key: dict with parameters
        Example:
            colorjitter:
                p: 0.1
                scale_setting: 0.5
                scale_color: 0.1

        For further information on how to setup the dictionary and default (recommended) values is given here:
        configs/examples/cell_segmentation/train_cellvit.yaml

        Training Transformations:
            Implemented are:
                - A.RandomRotate90: Key in transform_settings: randomrotate90, parameters: p
                - A.HorizontalFlip: Key in transform_settings: horizontalflip, parameters: p
                - A.VerticalFlip: Key in transform_settings: verticalflip, parameters: p
                - A.Downscale: Key in transform_settings: downscale, parameters: p, scale
                - A.Blur: Key in transform_settings: blur, parameters: p, blur_limit
                - A.GaussNoise: Key in transform_settings: gaussnoise, parameters: p, var_limit
                - A.ColorJitter: Key in transform_settings: colorjitter, parameters: p, scale_setting, scale_color
                - A.Superpixels: Key in transform_settings: superpixels, parameters: p
                - A.ZoomBlur: Key in transform_settings: zoomblur, parameters: p
                - A.RandomSizedCrop: Key in transform_settings: randomsizedcrop, parameters: p
                - A.ElasticTransform: Key in transform_settings: elastictransform, parameters: p
            Always implemented at the end of the pipeline:
                - A.Normalize with given mean (default: (0.5, 0.5, 0.5)) and std (default: (0.5, 0.5, 0.5))

        Validation Transformations:
            A.Normalize with given mean (default: (0.5, 0.5, 0.5)) and std (default: (0.5, 0.5, 0.5))

        Args:
            transform_settings (dict): dictionay with the transformation settings.
            input_shape (int, optional): Input shape of the images to used. Defaults to 256.

        Returns:
            Tuple[Callable, Callable]: Train Transformations, Validation Transformations

        """
        transform_list = []
        transform_settings = {k.lower(): v for k, v in transform_settings.items()}
        if "RandomRotate90".lower() in transform_settings:
            p = transform_settings["randomrotate90"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(A.RandomRotate90(p=p))
        if "HorizontalFlip".lower() in transform_settings.keys():
            p = transform_settings["horizontalflip"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(A.HorizontalFlip(p=p))
        if "VerticalFlip".lower() in transform_settings:
            p = transform_settings["verticalflip"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(A.VerticalFlip(p=p))
        if "Downscale".lower() in transform_settings:
            p = transform_settings["downscale"]["p"]
            scale = transform_settings["downscale"]["scale"]
            if p > 0 and p <= 1:
                transform_list.append(
                    A.Downscale(p=p, scale_max=scale, scale_min=scale)
                )
        if "Blur".lower() in transform_settings:
            p = transform_settings["blur"]["p"]
            blur_limit = transform_settings["blur"]["blur_limit"]
            if p > 0 and p <= 1:
                transform_list.append(A.Blur(p=p, blur_limit=blur_limit))
        if "GaussNoise".lower() in transform_settings:
            p = transform_settings["gaussnoise"]["p"]
            var_limit = transform_settings["gaussnoise"]["var_limit"]
            if p > 0 and p <= 1:
                transform_list.append(A.GaussNoise(p=p, var_limit=var_limit))
        if "ColorJitter".lower() in transform_settings:
            p = transform_settings["colorjitter"]["p"]
            scale_setting = transform_settings["colorjitter"]["scale_setting"]
            scale_color = transform_settings["colorjitter"]["scale_color"]
            if p > 0 and p <= 1:
                transform_list.append(
                    A.ColorJitter(
                        p=p,
                        brightness=scale_setting,
                        contrast=scale_setting,
                        saturation=scale_color,
                        hue=scale_color / 2,
                    )
                )
        if "Superpixels".lower() in transform_settings:
            p = transform_settings["superpixels"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(
                    A.Superpixels(
                        p=p,
                        p_replace=0.1,
                        n_segments=200,
                        max_size=int(input_shape / 2),
                    )
                )
        if "ZoomBlur".lower() in transform_settings:
            p = transform_settings["zoomblur"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(A.ZoomBlur(p=p, max_factor=1.05))
        if "RandomSizedCrop".lower() in transform_settings:
            p = transform_settings["randomsizedcrop"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(
                    A.RandomSizedCrop(
                        min_max_height=(input_shape / 2, input_shape),
                        height=input_shape,
                        width=input_shape,
                        p=p,
                    )
                )
        if "ElasticTransform".lower() in transform_settings:
            p = transform_settings["elastictransform"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(
                    A.ElasticTransform(p=p, sigma=25, alpha=0.5, alpha_affine=15)
                )

        if "normalize" in transform_settings:
            mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
            std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        transform_list.append(A.Normalize(mean=mean, std=std))

        train_transforms = A.Compose(transform_list)
        val_transforms = A.Compose([A.Normalize(mean=mean, std=std)])

        return train_transforms, val_transforms

    def get_sampler(
        self, train_dataset: CellDataset, strategy: str = "random", gamma: float = 1
    ) -> Sampler:
        """Return the sampler (either RandomSampler or WeightedRandomSampler)

        Args:
            train_dataset (CellDataset): Dataset for training
            strategy (str, optional): Sampling strategy. Defaults to "random" (random sampling).
                Implemented are "random", "cell", "tissue", "cell+tissue".
            gamma (float, optional): Gamma scaling factor, between 0 and 1.
                1 means total balancing, 0 means original weights. Defaults to 1.

        Raises:
            NotImplementedError: Not implemented sampler is selected

        Returns:
            Sampler: Sampler for training
        """
        if strategy.lower() == "random":
            sampling_generator = torch.Generator().manual_seed(
                self.default_conf["random_seed"]
            )
            sampler = RandomSampler(train_dataset, generator=sampling_generator)
            self.logger.info("Using RandomSampler")
        else:
            # this solution is not accurate when a subset is used since the weights are calculated on the whole training dataset
            if isinstance(train_dataset, Subset):
                ds = train_dataset.dataset
            else:
                ds = train_dataset
            ds.load_cell_count()
            if strategy.lower() == "cell":
                weights = ds.get_sampling_weights_cell(gamma)
            elif strategy.lower() == "tissue":
                weights = ds.get_sampling_weights_tissue(gamma)
            elif strategy.lower() == "cell+tissue":
                weights = ds.get_sampling_weights_cell_tissue(gamma)
            else:
                raise NotImplementedError(
                    "Unknown sampling strategy - Implemented are cell, tissue and cell+tissue"
                )

            if isinstance(train_dataset, Subset):
                weights = torch.Tensor([weights[i] for i in train_dataset.indices])

            sampling_generator = torch.Generator().manual_seed(
                self.default_conf["random_seed"]
            )
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(train_dataset),
                replacement=True,
                generator=sampling_generator,
            )

            self.logger.info(f"Using Weighted Sampling with strategy: {strategy}")
            self.logger.info(f"Unique-Weights: {torch.unique(weights)}")

        return sampler

    def get_trainer(self) -> BaseTrainer:
        """Return Trainer matching to this network

        Returns:
            BaseTrainer: Trainer
        """
        return CellViTTrainer
