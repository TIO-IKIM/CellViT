# -*- coding: utf-8 -*-
# CellViT Trainer Class
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2

# import wandb
from matplotlib import pyplot as plt
from torchmetrics.functional import dice
from torchmetrics.functional.classification import binary_jaccard_index

# from cell_segmentation.datasets.conic import CoNicDataclass
from base_ml.base_loss import L1LossWeighted, XentropyLoss, DiceLoss
from base_ml.base_trainer import BaseTrainer
from cell_segmentation.utils.metrics import get_fast_pq, remap_label
from cell_segmentation.utils.tools import get_bounding_box
from stardist import dist_to_coord, non_maximum_suppression, polygons_to_label
from torch.utils.data import DataLoader

from utils.tools import AverageMeter

import tqdm


class CellViTStarDistTrainerDebug(BaseTrainer):
    """CellViTStarDist trainer class

    Args:
        model (CellViTStarDist): CellViTStarDist model that should be trained
        loss_fn_dict (dict): Dictionary with loss functions for each branch with a dictionary of loss functions.
            Name of branch as top-level key, followed by a dictionary with loss name, loss fn and weighting factor
            Example:
            {
                "dist_map": {"bce": {loss_fn(Callable), weight_factor(float)}, "dice": {loss_fn(Callable), weight_factor(float)}},
                "stardist_map": {"bce": {loss_fn(Callable), weight_factor(float)}, "dice": {loss_fn(Callable), weight_factor(float)}},
                "nuclei_type_map": {"bce": {loss_fn(Callable), weight_factor(float)}, "dice": {loss_fn(Callable), weight_factor(float)}}
                "tissue_types": {"ce": {loss_fn(Callable), weight_factor(float)}}
            }
            Required Keys are:
                * dist_map
                * stardist_map
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
        regression_loss (bool, optional): Use regressive loss for predicting vector components.
            Adds two additional channels to the binary and hv decoder. Defaults to False.
            Currently not implemented!
    """

    def __init__(
        self,
        model,
        loss_fn_dict,
        optimizer,
        scheduler,
        device,
        logger,
        logdir,
        num_classes,
        dataset_config,
        experiment_config,
        early_stopping=None,
        log_images=False,
        magnification=40,
        mixed_precision=False,
    ):
        super().__init__(
            model=model,
            loss_fn=None,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            logdir=logdir,
            experiment_config=experiment_config,
            early_stopping=early_stopping,
            accum_iter=1,
            log_images=log_images,
            mixed_precision=False,
        )
        self.loss_fn_dict = loss_fn_dict
        self.num_classes = num_classes
        self.dataset_config = dataset_config
        self.tissue_types = dataset_config["tissue_types"]
        self.reverse_tissue_types = {v: k for k, v in self.tissue_types.items()}
        self.nuclei_types = dataset_config["nuclei_types"]
        self.magnification = magnification

        # setup logging objects
        self.loss_avg_tracker = {"Total_Loss": AverageMeter("Total_Loss", ":.4f")}
        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                self.loss_avg_tracker[f"{branch}_{loss_name}"] = AverageMeter(
                    f"{branch}_{loss_name}", ":.4f"
                )
        self.batch_avg_tissue_acc = AverageMeter("Batch_avg_tissue_ACC", ":4.f")

    def train_epoch(
        self, epoch: int, train_dataloader: DataLoader, unfreeze_epoch: int = 50
    ) -> Tuple[dict, dict]:
        """Training logic for a training epoch

        Args:
            epoch (int): Current epoch number
            train_dataloader (DataLoader): Train dataloader
            unfreeze_epoch (int, optional): Epoch to unfreeze layers
        Returns:
            Tuple[dict, dict]: wandb logging dictionaries
                * Scalar metrics
                * Image metrics
        """
        self.model.train()
        # if epoch >= unfreeze_epoch:
        #     self.model.unfreeze_encoder()

        train_example_img = None

        # reset metrics
        self.loss_avg_tracker["Total_Loss"].reset()
        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                self.loss_avg_tracker[f"{branch}_{loss_name}"].reset()
        self.batch_avg_tissue_acc.reset()

        # randomly select a batch that should be displayed
        if self.log_images:
            select_example_image = int(torch.randint(0, len(train_dataloader), (1,)))
        else:
            select_example_image = None
        train_loop = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))

        for batch_idx, batch in train_loop:
            return_example_images = batch_idx == select_example_image
            batch_metrics, example_img = self.train_step(
                batch,
                batch_idx,
                len(train_dataloader),
                return_example_images=return_example_images,
            )
            train_loop.set_postfix(
                {
                    "Loss": np.round(self.loss_avg_tracker["Total_Loss"].avg, 3),
                }
            )

        scalar_metrics = {
            "Loss/Train": self.loss_avg_tracker["Total_Loss"].avg,
        }

        self.logger.info(
            f"{'Training epoch stats:' : <25} "
            f"Loss: {self.loss_avg_tracker['Total_Loss'].avg:.4f} - "
        )

        image_metrics = {"Example-Predictions/Train": train_example_img}
        return scalar_metrics, image_metrics

    def train_step(
        self,
        batch: object,
        batch_idx: int,
        num_batches: int,
        return_example_images: bool,
    ) -> Tuple[dict, Union[plt.Figure, None]]:
        """Training step

        Args:
            batch (object): Training batch, consisting of images ([0]), masks ([1]), tissue_types ([2]) and figure filenames ([3])
            batch_idx (int): Batch index
            num_batches (int): Total number of batches in epoch
            return_example_images (bool): If an example preciction image should be returned

        Returns:
            Tuple[dict, Union[plt.Figure, None]]:
                * Batch-Metrics: dictionary with the following keys:
                * Example prediction image
        """
        # unpack batch
        imgs = batch[0].to(self.device)  # imgs shape: (batch_size, 3, H, W)
        masks = batch[1]
        masks.pop(
            "hv_map"
        )  # keys: instance_map', 'nuclei_type_map', 'nuclei_binary_map', 'dist_map', 'stardist_map'
        # tissue_types = batch[2]  # list[str]

        predictions_ = self.model.forward(imgs)
        dist_map = F.sigmoid(predictions_["dist_map"])
        stardist_map = predictions_["stardist_map"]
        nuclei_type_map = F.softmax(predictions_["nuclei_type_map"], dim=1)

        gt_dist_map = masks["dist_map"].to(self.device)
        gt_stardist_map = masks["stardist_map"].to(self.device)
        gt_nuclei_type_map = (
            F.one_hot(
                torch.squeeze(masks["nuclei_type_map"]).type(torch.int64), num_classes=6
            )
            .permute(0, 3, 1, 2)
            .to(self.device)
        )

        total_loss = self.calculate_loss(
            dist_map,
            stardist_map,
            nuclei_type_map,
            gt_dist_map,
            gt_stardist_map,
            gt_nuclei_type_map,
        )

        total_loss.backward()
        if (
            ((batch_idx + 1) % self.accum_iter == 0)
            or ((batch_idx + 1) == num_batches)
            or (self.accum_iter == 1)
        ):
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.model.zero_grad()
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

        batch_metrics = {}

        return_example_images = None

        return batch_metrics, return_example_images

    def validation_epoch(
        self, epoch: int, val_dataloader: DataLoader
    ) -> Tuple[dict, dict, float]:
        """Validation logic for a validation epoch

        Args:
            epoch (int): Current epoch number
            val_dataloader (DataLoader): Validation dataloader

        Returns:
            Tuple[dict, dict, float]: wandb logging dictionaries
                * Scalar metrics
                * Image metrics
                * Early stopping metric
        """
        self.model.eval()

        binary_dice_scores = []
        binary_jaccard_scores = []
        pq_scores = []
        val_example_img = None

        # reset metrics
        self.loss_avg_tracker["Total_Loss"].reset()
        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                self.loss_avg_tracker[f"{branch}_{loss_name}"].reset()
        self.batch_avg_tissue_acc.reset()

        # randomly select a batch that should be displayed
        if self.log_images:
            select_example_image = int(torch.randint(0, len(val_dataloader), (1,)))
        else:
            select_example_image = None

        val_loop = tqdm.tqdm(enumerate(val_dataloader), total=len(val_dataloader))

        with torch.no_grad():
            for batch_idx, batch in val_loop:
                return_example_images = batch_idx == select_example_image
                batch_metrics, example_img = self.validation_step(
                    batch, batch_idx, return_example_images
                )
                if example_img is not None:
                    val_example_img = example_img
                binary_dice_scores = (
                    binary_dice_scores + batch_metrics["binary_dice_scores"]
                )
                binary_jaccard_scores = (
                    binary_jaccard_scores + batch_metrics["binary_jaccard_scores"]
                )
                pq_scores = pq_scores + batch_metrics["pq_scores"]
                val_loop.set_postfix(
                    {
                        "Loss": np.round(self.loss_avg_tracker["Total_Loss"].avg, 3),
                        "Dice": np.round(np.nanmean(binary_dice_scores), 3),
                        "Pred-Acc": np.round(self.batch_avg_tissue_acc.avg, 3),
                    }
                )

        # calculate global metrics
        binary_dice_scores = np.array(binary_dice_scores)
        binary_jaccard_scores = np.array(binary_jaccard_scores)
        pq_scores = np.array(pq_scores)

        scalar_metrics = {
            "Loss/Validation": self.loss_avg_tracker["Total_Loss"].avg,
            "Binary-Cell-Dice-Mean/Validation": np.nanmean(binary_dice_scores),
            "Binary-Cell-Jacard-Mean/Validation": np.nanmean(binary_jaccard_scores),
            "bPQ/Validation": np.nanmean(pq_scores),
        }

        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                scalar_metrics[
                    f"{branch}_{loss_name}/Validation"
                ] = self.loss_avg_tracker[f"{branch}_{loss_name}"].avg

        self.logger.info(
            f"{'Validation epoch stats:' : <25} "
            f"Loss: {self.loss_avg_tracker['Total_Loss'].avg:.4f} - "
            f"Binary-Cell-Dice: {np.nanmean(binary_dice_scores):.4f} - "
            f"Binary-Cell-Jacard: {np.nanmean(binary_jaccard_scores):.4f} - "
            f"PQ-Score: {np.nanmean(pq_scores):.4f} - "
        )

        image_metrics = {"Example-Predictions/Validation": val_example_img}

        return scalar_metrics, image_metrics, np.nanmean(pq_scores)

    def validation_step(
        self,
        batch: object,
        batch_idx: int,
        return_example_images: bool,
    ):
        """Validation step

        Args:
            batch (object): Training batch, consisting of images ([0]), masks ([1]), tissue_types ([2]) and figure filenames ([3])
            batch_idx (int): Batch index
            return_example_images (bool): If an example preciction image should be returned

        Returns:
            Tuple[dict, Union[plt.Figure, None]]:
                * Batch-Metrics: dictionary, structure not fixed yet
                * Example prediction image
        """
        # unpack batch, for shape compare train_step method
        imgs = batch[0].to(self.device)
        masks = batch[1]
        # tissue_types = batch[2]

        self.model.zero_grad()
        self.optimizer.zero_grad()

        predictions_ = self.model.forward(imgs)
        dist_map = F.sigmoid(predictions_["dist_map"])
        stardist_map = predictions_["stardist_map"]
        nuclei_type_map = F.softmax(predictions_["nuclei_type_map"], dim=1)

        gt_dist_map = masks["dist_map"].to(self.device)
        gt_stardist_map = masks["stardist_map"].to(self.device)
        gt_nuclei_type_map = gt_nuclei_type_map = (
            F.one_hot(
                torch.squeeze(masks["nuclei_type_map"]).type(torch.int64), num_classes=6
            )
            .permute(0, 3, 1, 2)
            .to(self.device)
        )
        gt_binary_map = masks["nuclei_binary_map"].to(self.device)
        gt_instance_map = masks["instance_map"].to(self.device)

        _ = self.calculate_loss(
            dist_map,
            stardist_map,
            nuclei_type_map,
            gt_dist_map,
            gt_stardist_map,
            gt_nuclei_type_map,
        )

        # get metrics for this batch
        batch_metrics = self.calculate_step_metric_validation(
            dist_map, stardist_map, nuclei_type_map, gt_binary_map, gt_instance_map
        )
        return_example_images = None

        return batch_metrics, return_example_images

    def calculate_loss(
        self,
        dist_map,
        stardist_map,
        nuclei_type_map,
        gt_dist_map,
        gt_stardist_map,
        gt_nuclei_type_map,
    ) -> torch.Tensor:
        """Calculate the loss

        Args:
            predictions (CoNicDataclass): Processed network output
            gt (CoNicDataclass): Ground truth values

        Returns:
            torch.Tensor: Loss
        """
        prob_loss_fn = nn.BCELoss()
        prob_loss = prob_loss_fn(dist_map.squeeze(), gt_dist_map)
        stardist_loss_fn = L1LossWeighted()
        stardist_loss = stardist_loss_fn(stardist_map, gt_stardist_map, gt_dist_map)
        type_map_loss_x = XentropyLoss()
        type_map_loss_dice = DiceLoss()
        type_map_loss = type_map_loss_x(
            nuclei_type_map, gt_nuclei_type_map
        ) + type_map_loss_dice(nuclei_type_map, gt_nuclei_type_map)

        total_loss = 1 * prob_loss + 1 * stardist_loss + 1 * type_map_loss
        self.loss_avg_tracker["Total_Loss"].update(total_loss.detach().cpu().numpy())

        return total_loss

    def calculate_step_metric_train(
        self,
        dist_map,
        stardist_map,
        nuclei_type_map,
        gt_dist_map,
        gt_stardist_map,
        gt_nuclei_type_map,
    ) -> dict:
        """Calculate the metrics for the training step

        Args:
            predictions (CoNicDataclass): Processed network output
            gt (CoNicDataclass): Ground truth values

        Returns:
            dict: Dictionary with metrics. Structure not fixed yet
        """
        binary_dice_scores = []
        binary_jaccard_scores = []

        batch_metrics = {
            "binary_dice_scores": binary_dice_scores,
            "binary_jaccard_scores": binary_jaccard_scores,
        }

        return batch_metrics

    def calculate_step_metric_validation(
        self, dist_map, stardist_map, nuclei_type_map, gt_binary_map, gt_instance_map
    ) -> dict:
        instance_preds, _, _ = self.postprocess(dist_map, stardist_map, nuclei_type_map)

        binary_dice_scores = []
        binary_jaccard_scores = []
        pq_scores = []

        for i in range(instance_preds.shape[0]):
            pred_binary_map = (
                torch.clip(instance_preds[i], min=0, max=1)
                .type(torch.uint8)
                .to(self.device)
            )
            target_binary_map = gt_binary_map[i].type(torch.uint8).to(self.device)
            cell_dice = (
                dice(preds=pred_binary_map, target=target_binary_map, ignore_index=0)
                .detach()
                .cpu()
            )
            cell_jaccard = (
                binary_jaccard_index(
                    preds=pred_binary_map,
                    target=target_binary_map,
                )
                .detach()
                .cpu()
            )
            binary_dice_scores.append(float(cell_dice))
            binary_jaccard_scores.append(float(cell_jaccard))

            # pq values
            remapped_instance_pred = remap_label(instance_preds[i])
            remapped_gt = remap_label(gt_instance_map[i].detach().cpu())
            [_, _, pq], _ = get_fast_pq(true=remapped_gt, pred=remapped_instance_pred)
            pq_scores.append(pq)

        batch_metrics = {
            "binary_dice_scores": binary_dice_scores,
            "binary_jaccard_scores": binary_jaccard_scores,
            "pq_scores": pq_scores,
        }
        return batch_metrics

    def postprocess(self, dist_map, stardist_map, nuclei_type_map):
        instance_preds_list = []
        type_preds = []

        dists_pred = stardist_map.detach().cpu().squeeze().numpy()
        probs_pred = dist_map.detach().cpu().squeeze().numpy()

        for i in range(dists_pred.shape[0]):
            dists = dists_pred[i, :, :, :]
            probs = probs_pred[i, :, :]
            dists = np.transpose(dists, (1, 2, 0))
            points, _, dists = non_maximum_suppression(dists, probs)
            _ = dist_to_coord(dists, points)
            binary_star_label = polygons_to_label(dists, points, (256, 256))
            instance_preds = remap_label(binary_star_label)

            # old code
            pred_type = (
                nuclei_type_map[i, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
            )

            inst_id_list = np.unique(instance_preds)[1:]  # exlcude background
            inst_info_dict = {}
            for inst_id in inst_id_list:
                inst_map = instance_preds == inst_id
                rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
                inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
                inst_map = inst_map[
                    inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
                ]
                inst_map = inst_map.astype(np.uint8)
                inst_moment = cv2.moments(inst_map)
                inst_contour = cv2.findContours(
                    inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                # * opencv protocol format may break
                inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
                # < 3 points dont make a contour, so skip, likely artifact too
                # as the contours obtained via approximation => too small or sthg
                if inst_contour.shape[0] < 3:
                    continue
                if len(inst_contour.shape) != 2:
                    continue  # ! check for trickery shape
                inst_centroid = [
                    (inst_moment["m10"] / inst_moment["m00"]),
                    (inst_moment["m01"] / inst_moment["m00"]),
                ]
                inst_centroid = np.array(inst_centroid)
                inst_contour[:, 0] += inst_bbox[0][1]  # X
                inst_contour[:, 1] += inst_bbox[0][0]  # Y
                inst_centroid[0] += inst_bbox[0][1]  # X
                inst_centroid[1] += inst_bbox[0][0]  # Y
                inst_info_dict[inst_id] = {  # inst_id should start at 1
                    "bbox": inst_bbox,
                    "centroid": inst_centroid,
                    "contour": inst_contour,
                    "type_prob": None,
                    "type": None,
                }

            #### * Get class of each instance id, stored at index id-1 (inst_id = number of deteced nucleus)
            for inst_id in list(inst_info_dict.keys()):
                rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
                inst_map_crop = instance_preds[rmin:rmax, cmin:cmax]
                inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
                inst_map_crop = inst_map_crop == inst_id
                inst_type = inst_type_crop[inst_map_crop]
                type_list, type_pixels = np.unique(inst_type, return_counts=True)
                type_list = list(zip(type_list, type_pixels))
                type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
                inst_type = type_list[0][0]
                if inst_type == 0:  # ! pick the 2nd most dominant if exist
                    if len(type_list) > 1:
                        inst_type = type_list[1][0]
                type_dict = {v[0]: v[1] for v in type_list}
                type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
                inst_info_dict[inst_id]["type"] = int(inst_type)
                inst_info_dict[inst_id]["type_prob"] = float(type_prob)

            type_preds.append(inst_info_dict)
            instance_preds_list.append(instance_preds)

        instance_preds = torch.Tensor(np.stack(instance_preds_list))

        batch_size, h, w = instance_preds.shape
        instance_type_nuclei_maps = torch.zeros((batch_size, h, w, 6))
        for i in range(batch_size):
            instance_type_nuclei_map = torch.zeros((h, w, 6))
            instance_map = instance_preds[i]
            type_pred = type_preds[i]
            for nuclei, spec in type_pred.items():
                nuclei_type = spec["type"]
                instance_type_nuclei_map[:, :, nuclei_type][
                    instance_map == nuclei
                ] = nuclei

            instance_type_nuclei_maps[i, :, :, :] = instance_type_nuclei_map

        instance_type_nuclei_maps = instance_type_nuclei_maps.permute(0, 3, 1, 2)
        instance_type_nuclei_maps = torch.Tensor(instance_type_nuclei_maps)

        return instance_preds, type_preds, instance_type_nuclei_maps
