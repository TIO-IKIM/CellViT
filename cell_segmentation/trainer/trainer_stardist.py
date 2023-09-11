# -*- coding: utf-8 -*-
# CellViT Trainer Class
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

# import wandb
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from torchmetrics.functional import dice
from torchmetrics.functional.classification import binary_jaccard_index

from cell_segmentation.trainer.trainer_cellvit import CellViTTrainer
from cell_segmentation.utils.metrics import get_fast_pq, remap_label


class CellViTStarDistTrainer(CellViTTrainer):
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
        tissue_types = batch[2]  # list[str]

        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # make predictions
                predictions_ = self.model.forward(imgs)

                # reshaping and postprocessing
                predictions = self.unpack_predictions(predictions=predictions_)
                gt = self.unpack_masks(masks=masks, tissue_types=tissue_types)

                # calculate loss
                total_loss = self.calculate_loss(predictions, gt)

                # backward pass
                self.scaler.scale(total_loss).backward()

                if (
                    ((batch_idx + 1) % self.accum_iter == 0)
                    or ((batch_idx + 1) == num_batches)
                    or (self.accum_iter == 1)
                ):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.model.zero_grad()
        else:
            predictions_ = self.model.forward(imgs)
            predictions = self.unpack_predictions(predictions=predictions_)
            gt = self.unpack_masks(masks=masks, tissue_types=tissue_types)

            # calculate loss
            total_loss = self.calculate_loss(predictions, gt)

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

        batch_metrics = self.calculate_step_metric_train(predictions, gt)

        if return_example_images:
            return_example_images = self.generate_example_image(
                imgs, predictions, gt, num_images=4, num_nuclei_classes=self.num_classes
            )
        else:
            return_example_images = None

        return batch_metrics, return_example_images

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
        tissue_types = batch[2]

        self.model.zero_grad()
        self.optimizer.zero_grad()

        if self.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                predictions_ = self.model.forward(imgs)
                # reshaping and postprocessing
                predictions = self.unpack_predictions(predictions=predictions_)
                gt = self.unpack_masks(masks=masks, tissue_types=tissue_types)
                # calculate loss
                _ = self.calculate_loss(predictions, gt)

        else:
            predictions_ = self.model.forward(imgs)
            # reshaping and postprocessing
            predictions = self.unpack_predictions(predictions=predictions_)
            gt = self.unpack_masks(masks=masks, tissue_types=tissue_types)
            # calculate loss
            _ = self.calculate_loss(predictions, gt)

        # get metrics for this batch
        batch_metrics = self.calculate_step_metric_validation(predictions, gt)

        if return_example_images:
            try:
                return_example_images = self.generate_example_image(
                    imgs,
                    predictions,
                    gt,
                    num_images=4,
                    num_nuclei_classes=self.num_classes,
                )
            except AssertionError:
                self.logger.error(
                    "AssertionError for Example Image. Please check. Continue without image."
                )
                return_example_images = None
        else:
            return_example_images = None

        return batch_metrics, return_example_images

    def unpack_predictions(self, predictions: dict) -> OrderedDict:
        """Unpack the given predictions. Main focus lays on reshaping and postprocessing predictions, e.g. separating instances

        Args:
            predictions (dict): Dictionary with the following keys:
                * tissue_types: Logit tissue prediction output. Shape: (batch_size, num_tissue_classes)
                * stardist_map: Stardist output for vector prediction. Shape: (batch_size, n_rays, H, W)
                * dist_map: Logit output for distance map. Shape: (batch_size, 1, H, W)
                * nuclei_type_map: Logit output for nuclei instance-prediction. Shape: (batch_size, num_nuclei_classes, H, W)

        Returns:
            OrderedDict: Processed network output. Keys are:
                * tissue_types: Logit tissue prediction output. Shape: (batch_size, num_tissue_classes)
                * nuclei_type_map: Softmax output for nuclei instance-prediction. Shape: (batch_size, num_nuclei_classes, H, W)
                * stardist_map: Stardist output for vector prediction. Shape: (batch_size, n_rays, H, W)
                * dist_map: Probability distance map. Shape: (batch_size, 1, H, W)
                * instance_map: Pixel-wise nuclear instance segmentation predictions. Shape: (batch_size, H, W)
                * instance_types: Dictionary, Pixel-wise nuclei type predictions
                * instance_types_nuclei: Pixel-wise nuclear instance segmentation predictions, for each nuclei type. Shape: (batch_size, num_nuclei_classes, H, W)
        """
        predictions["tissue_types"] = predictions["tissue_types"].to(self.device)
        predictions["nuclei_type_map"] = F.softmax(
            predictions["nuclei_type_map"], dim=1
        )
        predictions["dist_map"] = F.sigmoid(predictions["dist_map"])
        # postprocessing: apply NMS and StarDist postprocessing to generate binary and multiclass cell detections
        (
            predictions["instance_map"],
            predictions["instance_types"],
        ) = self.model.calculate_instance_map(
            predictions["dist_map"],
            predictions["stardist_map"],
            predictions["nuclei_type_map"],
        )
        predictions["instance_types_nuclei"] = self.model.generate_instance_nuclei_map(
            predictions["instance_map"], predictions["instance_types"]
        ).to(self.device)

        return predictions

    def unpack_masks(self, masks: dict, tissue_types: list) -> dict:
        """Unpack the given masks. Main focus lays on reshaping and postprocessing masks to generate one dict

        Args:
            masks (dict): Required keys are:
                * instance_map: Pixel-wise nuclear instance segmentations. Shape: (batch_size, H, W)
                * nuclei_type_map: Nuclei instance-prediction and segmentation (not binary, each instance has own integer). Shape: (batch_size, H, W)
                * nuclei_binary_map: Binary nuclei segmentations. Shape: (batch_size, H, W)
                * dist_map: Probability distance map. Shape: (batch_size, H, W, 2)
                * stardist_map: Stardist output. Shape: (batch_size, n_rays H, W)

            tissue_types (list): List of string names of ground-truth tissue types

        Returns:
            dict: Output ground truth values, with keys:
                * nuclei_type_map: One-hot encoded nuclei type maps Shape: (batch_size, num_nuclei_classes, H, W)
                * stardist_map: Stardist output. Shape: (batch_size, n_rays H, W)
                * dist_map:  Probability distance map. Shape: (batch_size, H, W)
                * instance_map: Pixel-wise nuclear instance segmentations. Shape: (batch_size, H, W) -> each instance has one integer
                * instance_types_nuclei: Shape: (batch_size, num_nuclei_classes, H, W) -> instance has one integer, for each nuclei class
                * tissue_types: Tissue types, as torch.Tensor with integer values. Shape: batch_size
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
            "dist_map": masks["dist_map"].to(self.device),  # shape: (batch_size, H, W)
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

        return gt

    def calculate_loss(self, predictions: OrderedDict, gt: dict) -> torch.Tensor:
        """Calculate the loss

        Args:
            predictions (OrderedDict): Processed network output. Keys are:
                * tissue_types: Logit tissue prediction output. Shape: (batch_size, num_tissue_classes)
                * nuclei_type_map: Softmax output for nuclei instance-prediction. Shape: (batch_size, num_nuclei_classes, H, W)
                * stardist_map: Stardist output for vector prediction. Shape: (batch_size, n_rays, H, W)
                * dist_map: Probability distance map. Shape: (batch_size, 1, H, W)
                * instance_map: Pixel-wise nuclear instance segmentation predictions. Shape: (batch_size, H, W)
                * instance_types: Dictionary, Pixel-wise nuclei type predictions
                * instance_types_nuclei: Pixel-wise nuclear instance segmentation predictions, for each nuclei type. Shape: (batch_size, num_nuclei_classes, H, W)
            gt (dict): Output ground truth values, with keys:
                * nuclei_type_map: One-hot encoded nuclei type maps Shape: (batch_size, num_nuclei_classes, H, W)
                * stardist_map: Stardist output. Shape: (batch_size, n_rays H, W)
                * dist_map:  Probability distance map.. Shape: (batch_size, H, W)
                * instance_map: Pixel-wise nuclear instance segmentations. Shape: (batch_size, H, W) -> each instance has one integer
                * instance_types_nuclei: Shape: (batch_size, num_nuclei_classes, H, W) -> instance has one integer, for each nuclei class
                * tissue_types: Tissue types, as torch.Tensor with integer values. Shape: batch_size

        Returns:
            torch.Tensor: Loss
        """
        total_loss = 0
        for branch, pred in predictions.items():
            if branch in [
                "instance_map",
                "instance_types",
                "instance_types_nuclei",
            ]:
                continue
            if branch not in self.loss_fn_dict:
                # self.logger.debug
                continue
            branch_loss_fns = self.loss_fn_dict[branch]
            for loss_name, loss_setting in branch_loss_fns.items():
                loss_fn = loss_setting["loss_fn"]
                weight = loss_setting["weight"]
                if loss_name.lower() == "msge":
                    loss_value = loss_fn(
                        input=pred,
                        target=gt[branch],
                        focus=gt["nuclei_binary_map"],
                        device=self.device,
                    )
                elif loss_name.lower() == "bceweighted" and loss_fn.apply_mask:
                    loss_value = loss_fn(
                        input=pred, target=gt[branch], mask=gt["dist_map"]
                    )
                elif loss_name.lower() == "maeweighted" and loss_fn.apply_mask:
                    loss_value = loss_fn(
                        input=pred, target=gt[branch], mask=gt["dist_map"]
                    )
                else:
                    loss_value = loss_fn(input=pred, target=gt[branch])
                total_loss = total_loss + weight * loss_value
                self.loss_avg_tracker[f"{branch}_{loss_name}"].update(
                    loss_value.detach().cpu().numpy()
                )
        self.loss_avg_tracker["Total_Loss"].update(total_loss.detach().cpu().numpy())

        return total_loss

    def calculate_step_metric_train(self, predictions: dict, gt: dict) -> dict:
        """Calculate the metrics for the training step

        Args:
            predictions (OrderedDict): Processed network output. Keys are:
                * tissue_types: Logit tissue prediction output. Shape: (batch_size, num_tissue_classes)
                * nuclei_type_map: Softmax output for nuclei instance-prediction. Shape: (batch_size, num_nuclei_classes, H, W)
                * stardist_map: Stardist output for vector prediction. Shape: (batch_size, n_rays, H, W)
                * dist_map: Probability distance map. Shape: (batch_size, 1, H, W)
                * instance_map: Pixel-wise nuclear instance segmentation predictions. Shape: (batch_size, H, W)
                * instance_types: Dictionary, Pixel-wise nuclei type predictions
                * instance_types_nuclei: Pixel-wise nuclear instance segmentation predictions, for each nuclei type. Shape: (batch_size, num_nuclei_classes, H, W)
            gt (dict): Output ground truth values, with keys:
                * nuclei_type_map: One-hot encoded nuclei type maps Shape: (batch_size, num_nuclei_classes, H, W)
                * stardist_map: Stardist output. Shape: (batch_size, n_rays H, W)
                * dist_map:  Probability distance map.. Shape: (batch_size, H, W)
                * instance_map: Pixel-wise nuclear instance segmentations. Shape: (batch_size, H, W) -> each instance has one integer
                * instance_types_nuclei: Shape: (batch_size, num_nuclei_classes, H, W) -> instance has one integer, for each nuclei class
                * tissue_types: Tissue types, as torch.Tensor with integer values. Shape: batch_size

        Returns:
            dict: Dictionary with metrics. Structure not fixed yet
        """
        # preparation and device movement
        predictions["tissue_types_classes"] = F.softmax(
            predictions["tissue_types"], dim=-1
        )
        pred_tissue = (
            torch.argmax(predictions["tissue_types_classes"], dim=-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        predictions["instance_map"] = predictions["instance_map"].detach().cpu()
        predictions["instance_types_nuclei"] = (
            predictions["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
        )
        gt["tissue_types"] = gt["tissue_types"].detach().cpu().numpy().astype(np.uint8)
        gt["instance_types_nuclei"] = (
            gt["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
        )
        tissue_detection_accuracy = accuracy_score(
            y_true=gt["tissue_types"], y_pred=pred_tissue
        )
        self.batch_avg_tissue_acc.update(tissue_detection_accuracy)

        binary_dice_scores = []
        binary_jaccard_scores = []

        for i in range(len(pred_tissue)):
            # binary dice score: Score for cell detection per image, without background
            pred_binary_map = (
                torch.clip(predictions["instance_map"][i], min=0, max=1)
                .type(torch.uint8)
                .to(self.device)
            )
            target_binary_map = (
                torch.clip(gt["instance_map"][i], min=0, max=1)
                .type(torch.uint8)
                .to(self.device)
            )
            cell_dice = (
                dice(preds=pred_binary_map, target=target_binary_map, ignore_index=0)
                .detach()
                .cpu()
            )
            binary_dice_scores.append(float(cell_dice))

            # binary aji
            cell_jaccard = (
                binary_jaccard_index(
                    preds=pred_binary_map,
                    target=target_binary_map,
                )
                .detach()
                .cpu()
            )
            binary_jaccard_scores.append(float(cell_jaccard))

        batch_metrics = {
            "binary_dice_scores": binary_dice_scores,
            "binary_jaccard_scores": binary_jaccard_scores,
            "tissue_pred": pred_tissue,
            "tissue_gt": gt["tissue_types"],
        }

        return batch_metrics

    def calculate_step_metric_validation(self, predictions: dict, gt: dict) -> dict:
        """Calculate the metrics for the validation step

        Args:
            predictions (OrderedDict): OrderedDict: Processed network output. Keys are:
                * tissue_types: Logit tissue prediction output. Shape: (batch_size, num_tissue_classes)
                * nuclei_type_map: Softmax output for nuclei instance-prediction. Shape: (batch_size, num_nuclei_classes, H, W)
                * stardist_map: Stardist output for vector prediction. Shape: (batch_size, n_rays, H, W)
                * dist_map: Probability distance map. Shape: (batch_size, 1, H, W)
                * instance_map: Pixel-wise nuclear instance segmentation predictions. Shape: (batch_size, H, W)
                * instance_types: Dictionary, Pixel-wise nuclei type predictions
                * instance_types_nuclei: Pixel-wise nuclear instance segmentation predictions, for each nuclei type. Shape: (batch_size, num_nuclei_classes, H, W)
            gt (dict): Output ground truth values, with keys:
                * nuclei_type_map: One-hot encoded nuclei type maps Shape: (batch_size, num_nuclei_classes, H, W)
                * stardist_map: Stardist output. Shape: (batch_size, n_rays H, W)
                * dist_map:  Probability distance map.. Shape: (batch_size, H, W)
                * instance_map: Pixel-wise nuclear instance segmentations. Shape: (batch_size, H, W) -> each instance has one integer
                * instance_types_nuclei: Shape: (batch_size, num_nuclei_classes, H, W) -> instance has one integer, for each nuclei class
                * tissue_types: Tissue types, as torch.Tensor with integer values. Shape: batch_size

        Returns:
            dict: Dictionary with metrics. Structure not fixed yet
        """
        # preparation and device movement
        predictions["tissue_types_classes"] = F.softmax(
            predictions["tissue_types"], dim=-1
        )
        pred_tissue = (
            torch.argmax(predictions["tissue_types_classes"], dim=-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        predictions["instance_map"] = predictions["instance_map"].detach().cpu()
        predictions["instance_types_nuclei"] = (
            predictions["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
        )
        instance_maps_gt = gt["instance_map"].detach().cpu()
        gt["tissue_types"] = gt["tissue_types"].detach().cpu().numpy().astype(np.uint8)
        gt["instance_types_nuclei"] = (
            gt["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
        )

        tissue_detection_accuracy = accuracy_score(
            y_true=gt["tissue_types"], y_pred=pred_tissue
        )
        self.batch_avg_tissue_acc.update(tissue_detection_accuracy)

        binary_dice_scores = []
        binary_jaccard_scores = []
        cell_type_pq_scores = []
        pq_scores = []

        for i in range(len(pred_tissue)):
            # binary dice score: Score for cell detection per image, without background
            pred_binary_map = (
                torch.clip(predictions["instance_map"][i], min=0, max=1)
                .type(torch.uint8)
                .to(self.device)
            )
            target_binary_map = (
                torch.clip(gt["instance_map"][i], min=0, max=1)
                .type(torch.uint8)
                .to(self.device)
            )
            cell_dice = (
                dice(preds=pred_binary_map, target=target_binary_map, ignore_index=0)
                .detach()
                .cpu()
            )
            binary_dice_scores.append(float(cell_dice))

            # binary aji
            cell_jaccard = (
                binary_jaccard_index(
                    preds=pred_binary_map,
                    target=target_binary_map,
                )
                .detach()
                .cpu()
            )
            binary_jaccard_scores.append(float(cell_jaccard))
            # pq values
            remapped_instance_pred = remap_label(predictions["instance_map"][i])
            remapped_gt = remap_label(instance_maps_gt[i])
            [_, _, pq], _ = get_fast_pq(true=remapped_gt, pred=remapped_instance_pred)
            pq_scores.append(pq)

            # pq values per class (skip background)
            nuclei_type_pq = []
            for j in range(0, self.num_classes):
                pred_nuclei_instance_class = remap_label(
                    predictions["instance_types_nuclei"][i][..., j]
                )
                target_nuclei_instance_class = remap_label(
                    gt["instance_types_nuclei"][i][..., j]
                )

                # if ground truth is empty, skip from calculation
                if len(np.unique(target_nuclei_instance_class)) == 1:
                    pq_tmp = np.nan
                else:
                    [_, _, pq_tmp], _ = get_fast_pq(
                        pred_nuclei_instance_class,
                        target_nuclei_instance_class,
                        match_iou=0.5,
                    )
                nuclei_type_pq.append(pq_tmp)

            cell_type_pq_scores.append(nuclei_type_pq)

        batch_metrics = {
            "binary_dice_scores": binary_dice_scores,
            "binary_jaccard_scores": binary_jaccard_scores,
            "pq_scores": pq_scores,
            "cell_type_pq_scores": cell_type_pq_scores,
            "tissue_pred": pred_tissue,
            "tissue_gt": gt["tissue_types"],
        }

        return batch_metrics

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
