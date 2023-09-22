# -*- coding: utf-8 -*-
# CPP-Net Experiment Class
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import inspect
import os
import sys


from base_ml.base_trainer import BaseTrainer

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from torchinfo import summary

from base_ml.base_loss import retrieve_loss_fn
from cell_segmentation.experiments.experiment_stardist_pannuke import (
    ExperimentCellViTStarDist,
)
from cell_segmentation.trainer.trainer_cpp_net import CellViTCPPTrainer
from models.segmentation.cell_segmentation.cellvit_cpp_net import (
    CellViT256CPP,
    CellViTCPP,
    CellViTSAMCPP,
)


class ExperimentCellViTCPP(ExperimentCellViTStarDist):
    def get_loss_fn(self, loss_fn_settings: dict) -> dict:
        """Create a dictionary with loss functions for all branches

        Branches: "dist_map", "stardist_map", "stardist_map_refined", "nuclei_type_map", "tissue_types"

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
                  nuclei_type_map:
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
            dist_map:
                bceweighted:
                    loss_fn: BCEWithLogitsLoss
                    weight: 1
            stardist_map:
                L1LossWeighted:
                    loss_fn: L1LossWeighted
                    weight: 1
            nuclei_type_map
                bce:
                    loss_fn: xentropy_loss
                    weight: 1
                dice:
                    loss_fn: dice_loss
                    weight: 1
            tissue_type has no default loss and might be skipped
        """
        loss_fn_dict = {}
        if "dist_map" in loss_fn_settings.keys():
            loss_fn_dict["dist_map"] = {}
            for loss_name, loss_sett in loss_fn_settings["dist_map"].items():
                parameters = loss_sett.get("args", {})
                loss_fn_dict["dist_map"][loss_name] = {
                    "loss_fn": retrieve_loss_fn(loss_sett["loss_fn"], **parameters),
                    "weight": loss_sett["weight"],
                }
        else:
            loss_fn_dict["dist_map"] = {
                "bceweighted": {
                    "loss_fn": retrieve_loss_fn("BCEWithLogitsLoss"),
                    "weight": 1,
                },
            }
        if "stardist_map" in loss_fn_settings.keys():
            loss_fn_dict["stardist_map"] = {}
            for loss_name, loss_sett in loss_fn_settings["stardist_map"].items():
                parameters = loss_sett.get("args", {})
                loss_fn_dict["stardist_map"][loss_name] = {
                    "loss_fn": retrieve_loss_fn(loss_sett["loss_fn"], **parameters),
                    "weight": loss_sett["weight"],
                }
        else:
            loss_fn_dict["stardist_map"] = {
                "L1LossWeighted": {
                    "loss_fn": retrieve_loss_fn("L1LossWeighted"),
                    "weight": 1,
                },
            }
        if "stardist_map_refined" in loss_fn_settings.keys():
            loss_fn_dict["stardist_map_refined"] = {}
            for loss_name, loss_sett in loss_fn_settings[
                "stardist_map_refined"
            ].items():
                parameters = loss_sett.get("args", {})
                loss_fn_dict["stardist_map_refined"][loss_name] = {
                    "loss_fn": retrieve_loss_fn(loss_sett["loss_fn"], **parameters),
                    "weight": loss_sett["weight"],
                }
        else:
            loss_fn_dict["stardist_map_refined"] = {
                "L1LossWeighted": {
                    "loss_fn": retrieve_loss_fn("L1LossWeighted"),
                    "weight": 1,
                },
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
        # skip default tissue loss!
        return loss_fn_dict

    def get_train_model(
        self,
        pretrained_encoder: Union[Path, str] = None,
        pretrained_model: Union[Path, str] = None,
        backbone_type: str = "default",
        shared_decoders: bool = False,
        **kwargs,
    ) -> nn.Module:
        """Return the CellViTStarDist training model

        Args:
            pretrained_encoder (Union[Path, str]): Path to a pretrained encoder. Defaults to None.
            pretrained_model (Union[Path, str], optional): Path to a pretrained model. Defaults to None.
            backbone_type (str, optional): Backbone Type. Currently supported are default (None, ViT256, SAM-B, SAM-L, SAM-H). Defaults to None
            shared_decoders (bool, optional): If shared skip decoders should be used. Defaults to False.

        Returns:
            nn.Module: StarDist training model with given setup
        """
        # reseed needed, due to subprocess seeding compatibility
        self.seed_run(self.default_conf["random_seed"])

        # check for backbones
        implemented_backbones = [
            "default",
            "vit256",
            "sam-b",
            "sam-l",
            "sam-h",
        ]
        if backbone_type.lower() not in implemented_backbones:
            raise NotImplementedError(
                f"Unknown Backbone Type - Currently supported are: {implemented_backbones}"
            )
        if backbone_type.lower() == "default":
            if shared_decoders:
                raise NotImplementedError(
                    "Shared decoders are not implemented for StarDist"
                )
            else:
                model_class = CellViTCPP
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
                nrays=self.run_conf["model"].get("nrays", 32),
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
                raise NotImplementedError(
                    "Shared decoders are not implemented for StarDist"
                )
            else:
                model_class = CellViT256CPP
            model = model_class(
                model256_path=pretrained_encoder,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                drop_rate=self.run_conf["training"].get("drop_rate", 0),
                attn_drop_rate=self.run_conf["training"].get("attn_drop_rate", 0),
                drop_path_rate=self.run_conf["training"].get("drop_path_rate", 0),
                nrays=self.run_conf["model"].get("nrays", 32),
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
                raise NotImplementedError(
                    "Shared decoders are not implemented for StarDist"
                )
            else:
                model_class = CellViTSAMCPP
            model = model_class(
                model_path=pretrained_encoder,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                vit_structure=backbone_type,
                drop_rate=self.run_conf["training"].get("drop_rate", 0),
                nrays=self.run_conf["model"].get("nrays", 32),
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

    def get_trainer(self) -> BaseTrainer:
        """Return Trainer matching to this network

        Returns:
            BaseTrainer: Trainer
        """
        return CellViTCPPTrainer
