# -*- coding: utf-8 -*-
# StarDist Inference Method for Patch-Wise Inference on a test set
# Without merging WSI
#
# Aim is to calculate metrics as defined for the PanNuke dataset
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import argparse
import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

from base_ml.base_experiment import BaseExperiment

BaseExperiment.seed_run(1232)

from collections import OrderedDict


# from scipy.io import savemat
import torch.nn.functional as F

from cell_segmentation.inference.inference_stardist_experiment_pannuke import (
    InferenceCellViTStarDist,
)
from models.segmentation.cell_segmentation.cellvit_cpp_net import (
    CellViTCPP,
    CellViT256CPP,
    CellViTSAMCPP,
)


class InferenceCellViTCPP(InferenceCellViTStarDist):
    def get_model(self, model_type: str) -> CellViTCPP:
        """Return the trained model for inference

        Args:
            model_type (str): Name of the model. Must either be one of:
                CellViTCPP, CellViT256CPP, CellViTSAMCPP

        Returns:
            CellViTCPP: Model
        """
        implemented_models = ["CellViTCPP", "CellViT256CPP", "CellViTSAMCPP"]
        if model_type not in implemented_models:
            raise NotImplementedError(
                f"Unknown model type. Please select one of {implemented_models}"
            )
        if model_type in ["CellViTCPP"]:
            model = CellViTCPP(
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

        elif model_type in ["CellViT256CPP"]:
            model = CellViT256CPP(
                model256_path=None,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                drop_rate=self.run_conf["training"].get("drop_rate", 0),
                attn_drop_rate=self.run_conf["training"].get("attn_drop_rate", 0),
                drop_path_rate=self.run_conf["training"].get("drop_path_rate", 0),
                nrays=self.run_conf["model"].get("nrays", 32),
            )
        elif model_type in ["CellViTSAMCPP"]:
            model = CellViTSAMCPP(
                model_path=None,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                vit_structure=self.run_conf["model"]["backbone"],
                drop_rate=self.run_conf["training"].get("drop_rate", 0),
                nrays=self.run_conf["model"].get("nrays", 32),
            )

        return model

    def unpack_predictions(self, predictions: dict, model: CellViTCPP) -> OrderedDict:
        """Unpack the given predictions. Main focus lays on reshaping and postprocessing predictions, e.g. separating instances

        Args:
            predictions (dict): Dictionary with the following keys:
                * tissue_types: Logit tissue prediction output. Shape: (batch_size, num_tissue_classes)
                * stardist_map: Stardist output for vector prediction. Shape: (batch_size, n_rays, H, W)
                * dist_map: Logit output for distance map. Shape: (batch_size, 1, H, W)
                * nuclei_type_map: Logit output for nuclei instance-prediction. Shape: (batch_size, num_nuclei_classes, H, W)
            model (CellViTStarDist): model

        Returns:
            OrderedDict: Processed network output. Keys are:
                * nuclei_type_map: Softmax output for nuclei instance-prediction. Shape: (batch_size, num_nuclei_classes, H, W)
                * stardist_map: Stardist output for vector prediction. Shape: (batch_size, n_rays, H, W)
                * dist_map: Probability distance map. Shape: (batch_size, 1, H, W)
                * instance_map: Pixel-wise nuclear instance segmentation predictions. Shape: (batch_size, H, W)
                * instance_types: Dictionary, Pixel-wise nuclei type predictions
                * instance_types_nuclei: Pixel-wise nuclear instance segmentation predictions, for each nuclei type. Shape: (batch_size, num_nuclei_classes, H, W)
        """
        predictions["nuclei_type_map"] = F.softmax(
            predictions["nuclei_type_map"], dim=1
        )  # shape: (batch_size, num_nuclei_classes, H, W)
        predictions["dist_map_sigmoid"] = F.sigmoid(predictions["dist_map"])
        # postprocessing: apply NMS and StarDist postprocessing to generate binary and multiclass cell detections

        (
            instance_map,
            predictions["instance_types"],
            instance_types_nuclei,
        ) = model.calculate_instance_map(
            predictions["dist_map_sigmoid"],
            predictions["stardist_map_refined"],
            predictions["nuclei_type_map"],
        )
        instance_map = instance_map.to(self.device)
        instance_types_nuclei = instance_types_nuclei.to(self.device)
        predictions["instance_map"] = instance_map
        predictions["instance_types_nuclei"] = instance_types_nuclei

        return predictions


# CLI
class InferenceCellViTParser:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Perform CellViT inference for given run-directory with model checkpoints and logs",
        )

        parser.add_argument(
            "--run_dir",
            type=str,
            default="/homes/fhoerst/histo-projects/CellViT/results/PanNuke/Revision/CellViTStarDist/Common-Loss/SAM-H/Shared-decoder/CPP-Net-Setting/2023-09-17T065947_CellViTSAMStarDist-H-Shared-Fold-3",  # TODO: remove
            help="Logging directory of a training run.",
            # required=True,
        )
        parser.add_argument(
            "--checkpoint_name",
            type=str,
            help="Name of the checkpoint.  Either select 'best_checkpoint.pth',"
            "'latest_checkpoint.pth' or one of the intermediate checkpoint names,"
            "e.g., 'checkpoint_100.pth'",
            default="model_best.pth",
        )
        parser.add_argument(
            "--gpu", type=int, help="Cuda-GPU ID for inference", default=4
        )
        parser.add_argument(
            "--magnification",
            type=int,
            help="Dataset Magnification. Either 20 or 40. Default: 40",
            choices=[20, 40],
            default=40,
        )

        self.parser = parser

    def parse_arguments(self) -> dict:
        opt = self.parser.parse_args()
        return vars(opt)


if __name__ == "__main__":
    configuration_parser = InferenceCellViTParser()
    configuration = configuration_parser.parse_arguments()
    print(configuration)
    inf = InferenceCellViTCPP(
        run_dir=configuration["run_dir"],
        checkpoint_name=configuration["checkpoint_name"],
        gpu=configuration["gpu"],
        magnification=configuration["magnification"],
    )
    model, dataloader, conf = inf.setup_patch_inference()

    inf.run_patch_inference(model, dataloader, conf)
