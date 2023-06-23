# -*- coding: utf-8 -*-
# UNETR2d Inference Method for Patch-Wise Inference on MoNuSeg dataset
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

from pathlib import Path
from typing import Union

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from einops import rearrange
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from skimage.color import rgba2rgb
from torch.utils.data import DataLoader
from torchmetrics.functional import dice
from torchmetrics.functional.classification import binary_jaccard_index
from torchvision import transforms

from cell_segmentation.datasets.monuseg import MoNuSegDataset
from cell_segmentation.utils.metrics import (
    cell_detection_scores,
    get_fast_pq,
    remap_label,
)
from cell_segmentation.utils.post_proc import calculate_instances
from cell_segmentation.utils.tools import pair_coordinates
from models.segmentation.cell_segmentation.unetr2d import (
    UNETR2d,
    UNETR2dSAM,
    UNETR2dSAMUnshared,
    UNETR2dUnshared,
    UNETR2dVIT256,
    UNETR2dVIT256Unshared,
)
from utils.logger import Logger
from utils.tools import unflatten_dict


class MoNuSegInference:
    def __init__(
        self,
        model_path: Union[Path, str],
        dataset_path: Union[Path, str],
        outdir: Union[Path, str],
        gpu: int,
        patching: bool = False,
        magnification: int = 40,
    ) -> None:
        """Cell Segmentation Inference class for MoNuSeg dataset

        Args:
            model_path (Union[Path, str]): Path to model checkpoint
            dataset_path (Union[Path, str]): Path to dataset
            outdir (Union[Path, str]): Output directory
            gpu (int): CUDA GPU id to use
            patching (bool, optional): If dataset should be pacthed to 256px. Defaults to False.
            magnification (int, optional): Dataset magnification. Defaults to 40.
        """
        self.model_path = Path(model_path)
        self.device = f"cuda:{gpu}"
        self.outdir = Path(outdir)
        self.outdir.mkdir(exist_ok=True, parents=True)
        self.magnification = magnification
        self.__instantiate_logger()
        self.__load_model()
        self.__load_inference_transforms()
        self.inference_dataset = MoNuSegDataset(
            dataset_path=dataset_path,
            transforms=self.inference_transforms,
            patching=patching,
        )
        self.inference_dataloader = DataLoader(
            self.inference_dataset,
            batch_size=1,
            num_workers=8,
            pin_memory=False,
            shuffle=False,
        )

    def __instantiate_logger(self) -> None:
        """Instantiate logger

        Logger is using no formatters. Logs are stored in the run directory under the filename: inference.log
        """
        logger = Logger(
            level="INFO",
            log_dir=self.outdir,
            comment="inference_monuseg",
            use_timestamp=False,
            formatter="%(message)s",
        )
        self.logger = logger.create_logger()

    def __load_model(self) -> None:
        """Load model and checkpoint and load the state_dict"""
        self.logger.info(f"Loading model: {self.model_path}")

        model_checkpoint = torch.load(self.model_path, map_location="cpu")

        # unpack checkpoint
        self.run_conf = unflatten_dict(model_checkpoint["config"], ".")
        self.model = self.__get_model(model_type=model_checkpoint["arch"])
        self.logger.info(
            self.model.load_state_dict(model_checkpoint["model_state_dict"])
        )
        self.model.eval()
        self.model.to(self.device)

    def __get_model(
        self, model_type: str
    ) -> Union[
        UNETR2d,
        UNETR2dUnshared,
        UNETR2dVIT256,
        UNETR2dVIT256Unshared,
        UNETR2dSAM,
        UNETR2dSAMUnshared,
    ]:
        """Return the trained model for inference

        Args:
            model_type (str): Name of the model. Must either be one of:
                UNETR2d, UNETR2dUnshared, UNETR2dVIT256, UNETR2dVIT256Unshared, UNETR2dSAM, UNETR2dSAMUnshared

        Returns:
            Union[UNETR2d, UNETR2dUnshared, UNETR2dVIT256, UNETR2dVIT256Unshared, UNETR2dSAM, UNETR2dSAMUnshared]: Model
        """
        implemented_models = [
            "UNETR2d",
            "UNETR2dUnshared",
            "UNETR2dVIT256",
            "UNETR2dVIT256Unshared",
            "UNETR2dSAM",
            "UNETR2dSAMUnshared",
        ]
        if model_type not in implemented_models:
            raise NotImplementedError(
                f"Unknown model type. Please select one of {implemented_models}"
            )
        if model_type in ["UNETR2d", "UNETR2dUnshared"]:
            if model_type == "UNETR2d":
                model_class = UNETR2d
            elif model_type == "UNETR2dUnshared":
                model_class = UNETR2dUnshared
            model = model_class(
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                embed_dim=self.run_conf["model"]["embed_dim"],
                input_channels=self.run_conf["model"].get("input_channels", 3),
                depth=self.run_conf["model"]["depth"],
                num_heads=self.run_conf["model"]["num_heads"],
                extract_layers=self.run_conf["model"]["extract_layers"],
            )

        elif model_type in ["UNETR2dVIT256", "UNETR2dVIT256Unshared"]:
            if model_type == "UNETR2dVIT256":
                model_class = UNETR2dVIT256
            elif model_type == "UNETR2dVIT256Unshared":
                model_class = UNETR2dVIT256Unshared
            model = model_class(
                model256_path=None,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
            )
        elif model_type in ["UNETR2dSAM", "UNETR2dSAMUnshared"]:
            if model_type == "UNETR2dSAM":
                model_class = UNETR2dSAM
            elif model_type == "UNETR2dSAMUnshared":
                model_class = UNETR2dSAMUnshared
            model = model_class(
                model_path=None,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                vit_structure=self.run_conf["model"]["backbone"],
            )
        return model

    def __load_inference_transforms(self) -> None:
        """Load the inference transformations from the run_configuration"""
        self.logger.info("Loading inference transformations")

        transform_settings = self.run_conf["transformations"]
        if "normalize" in transform_settings:
            mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
            std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        self.inference_transforms = A.Compose([A.Normalize(mean=mean, std=std)])

    def run_inference(self, generate_plots: bool = False) -> None:
        self.model.eval()

        # setup score tracker
        image_names = []  # image names as str
        binary_dice_scores = []  # binary dice scores per image
        binary_jaccard_scores = []  # binary jaccard scores per image
        pq_scores = []  # pq-scores per image
        dq_scores = []  # dq-scores per image
        sq_scores = []  # sq-scores per image
        f1_ds = []  # f1-scores per image
        prec_ds = []  # precision per image
        rec_ds = []  # recall per image

        inference_loop = tqdm.tqdm(
            enumerate(self.inference_dataloader), total=len(self.inference_dataloader)
        )

        with torch.no_grad():
            for image_idx, batch in inference_loop:
                image_metrics = self.inference_step(
                    model=self.model, batch=batch, generate_plots=generate_plots
                )
                image_names.append(image_metrics["image_name"])
                binary_dice_scores.append(image_metrics["binary_dice_score"])
                binary_jaccard_scores.append(image_metrics["binary_jaccard_score"])
                pq_scores.append(image_metrics["pq_score"])
                dq_scores.append(image_metrics["dq_score"])
                sq_scores.append(image_metrics["sq_score"])
                f1_ds.append(image_metrics["f1_d"])
                prec_ds.append(image_metrics["prec_d"])
                rec_ds.append(image_metrics["rec_d"])

        # average metrics for dataset
        binary_dice_scores = np.array(binary_dice_scores)
        binary_jaccard_scores = np.array(binary_jaccard_scores)
        pq_scores = np.array(pq_scores)
        dq_scores = np.array(dq_scores)
        sq_scores = np.array(sq_scores)
        f1_ds = np.array(f1_ds)
        prec_ds = np.array(prec_ds)
        rec_ds = np.array(rec_ds)

        dataset_metrics = {
            "Binary-Cell-Dice-Mean": float(np.nanmean(binary_dice_scores)),
            "Binary-Cell-Jacard-Mean": float(np.nanmean(binary_jaccard_scores)),
            "bPQ": float(np.nanmean(pq_scores)),
            "bDQ": float(np.nanmean(dq_scores)),
            "bSQ": float(np.nanmean(sq_scores)),
            "f1_detection": float(np.nanmean(f1_ds)),
            "precision_detection": float(np.nanmean(prec_ds)),
            "recall_detection": float(np.nanmean(rec_ds)),
        }
        self.logger.info(f"{20*'*'} Binary Dataset metrics {20*'*'}")
        [self.logger.info(f"{f'{k}:': <25} {v}") for k, v in dataset_metrics.items()]

    def inference_step(self, model, batch, generate_plots: bool = False):
        img = batch[0].to(self.device)
        if len(img.shape) > 4:
            img = img[0]
            img = rearrange(img, "c i j w h -> (i j) c w h")
        mask = batch[1]
        image_name = list(batch[2])

        model.zero_grad()

        predictions_ = model.forward(img)
        if img.shape[0] != 1:
            predictions_ = self.post_process_patching(predictions_)
        predictions = self.get_cell_predictions(predictions_)

        mask["instance_types"] = calculate_instances(
            torch.unsqueeze(mask["nuclei_binary_map"], dim=-1), mask["instance_map"]
        )

        image_metrics = self.calculate_step_metric(
            predictions=predictions, gt=mask, image_name=image_name
        )

        scores = [
            float(image_metrics["binary_dice_score"].detach().cpu()),
            float(image_metrics["binary_jaccard_score"].detach().cpu()),
            image_metrics["pq_score"],
        ]
        if generate_plots:
            if img.shape[0] != 1:
                img = torch.permute(img, (0, 2, 3, 1))
                img = rearrange(img, "(i j) h w c -> (i h) (j w) c", i=4, j=4)
                img = torch.unsqueeze(img, dim=0)
                img = torch.permute(img, (0, 3, 1, 2))

            self.plot_results(
                img=img,
                predictions=predictions,
                ground_truth=mask,
                img_name=image_name[0],
                outdir=self.outdir,
                scores=scores,
            )

        return image_metrics

    def calculate_step_metric(self, predictions, gt, image_name):
        predictions["instance_map"] = predictions["instance_map"].detach().cpu()
        instance_maps_gt = gt["instance_map"].detach().cpu()

        pred_binary_map = torch.argmax(predictions["nuclei_binary_map"], dim=-1)
        target_binary_map = gt["nuclei_binary_map"].to(self.device)

        # save predictions as mask
        pred_arr = pred_binary_map.detach().cpu().numpy().squeeze()
        pred_img = Image.fromarray((pred_arr * 255).astype(np.uint8))
        mask_outdir = self.outdir / "masks"
        mask_outdir.mkdir(exist_ok=True, parents=True)
        pred_img.save(mask_outdir / image_name[0])

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
        remapped_instance_pred = remap_label(predictions["instance_map"])
        remapped_gt = remap_label(instance_maps_gt)
        [dq, sq, pq], _ = get_fast_pq(true=remapped_gt, pred=remapped_instance_pred)

        # detection scores
        true_centroids = np.array(
            [v["centroid"] for k, v in gt["instance_types"][0].items()]
        )
        pred_centroids = np.array(
            [v["centroid"] for k, v in predictions["instance_types"].items()]
        )
        if true_centroids.shape[0] == 0:
            true_centroids = np.array([[0, 0]])
        if pred_centroids.shape[0] == 0:
            pred_centroids = np.array([[0, 0]])

        if self.magnification == 40:
            pairing_radius = 12
        else:
            pairing_radius = 6
        paired, unpaired_true, unpaired_pred = pair_coordinates(
            true_centroids, pred_centroids, pairing_radius
        )
        f1_d, prec_d, rec_d = cell_detection_scores(
            paired_true=paired[:, 0],
            paired_pred=paired[:, 1],
            unpaired_true=unpaired_true,
            unpaired_pred=unpaired_pred,
        )

        image_metrics = {
            "image_name": image_name,
            "binary_dice_score": cell_dice,
            "binary_jaccard_score": cell_jaccard,
            "pq_score": pq,
            "dq_score": dq,
            "sq_score": sq,
            "f1_d": f1_d,
            "prec_d": prec_d,
            "rec_d": rec_d,
        }

        return image_metrics

    def convert_binary_type(self, instance_types: dict) -> dict:
        cleaned_instance_types = {}
        for key, elem in instance_types.items():
            if elem["type"] == 0:
                continue
            else:
                elem["type"] = 0
                cleaned_instance_types[key] = elem

        return cleaned_instance_types

    def get_cell_predictions(self, predictions: dict):
        predictions = self.model.reshape_model_output(predictions, self.device)
        predictions["nuclei_binary_map"] = F.softmax(
            predictions["nuclei_binary_map"], dim=-1
        )
        predictions["nuclei_type_map"] = F.softmax(
            predictions["nuclei_type_map"], dim=-1
        )
        (
            predictions["instance_map"],
            predictions["instance_types"],
        ) = self.model.calculate_instance_map(
            predictions, magnification=self.magnification
        )
        predictions["instance_types"] = self.convert_binary_type(
            predictions["instance_types"][0]
        )

        return predictions

    def post_process_patching(self, predictions):
        predictions["nuclei_binary_map"] = rearrange(
            predictions["nuclei_binary_map"], "(i j) d w h ->d (i w) (j h)", i=4, j=4
        )
        predictions["hv_map"] = rearrange(
            predictions["hv_map"], "(i j) d w h -> d (i w) (j h)", i=4, j=4
        )
        predictions["nuclei_type_map"] = rearrange(
            predictions["nuclei_type_map"], "(i j) d w h -> d (i w) (j h)", i=4, j=4
        )

        predictions["nuclei_binary_map"] = torch.unsqueeze(
            predictions["nuclei_binary_map"], dim=0
        )
        predictions["hv_map"] = torch.unsqueeze(predictions["hv_map"], dim=0)
        predictions["nuclei_type_map"] = torch.unsqueeze(
            predictions["nuclei_type_map"], dim=0
        )

        return predictions

    def plot_results(
        self,
        img,
        predictions,
        ground_truth,
        img_name,
        outdir,
        scores,
    ) -> None:
        # create folder
        outdir = Path(outdir) / "plots"
        outdir.mkdir(exist_ok=True, parents=True)

        h = ground_truth["hv_map"].shape[1]
        w = ground_truth["hv_map"].shape[2]

        # process image and other maps
        sample_image = img.permute(0, 2, 3, 1).contiguous().cpu().numpy()

        pred_sample_binary_map = (
            predictions["nuclei_binary_map"][:, :, :, 1].detach().cpu().numpy()
        )[0]
        pred_sample_instance_maps = (
            predictions["instance_map"].detach().cpu().numpy()[0]
        )

        gt_sample_binary_map = (
            ground_truth["nuclei_binary_map"].detach().cpu().numpy()[0]
        )
        gt_sample_instance_map = ground_truth["instance_map"].detach().cpu().numpy()[0]

        binary_cmap = plt.get_cmap("Greys_r")
        instance_map = plt.get_cmap("viridis")

        inv_normalize = transforms.Normalize(
            mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5], std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
        )
        inv_samples = inv_normalize(torch.tensor(sample_image).permute(0, 3, 1, 2))
        sample_image = inv_samples.permute(0, 2, 3, 1).detach().cpu().numpy()[0]

        # start overlaying on image
        placeholder = np.zeros((2 * h, 4 * w, 3))
        # orig image
        placeholder[:h, :w, :3] = sample_image
        placeholder[h : 2 * h, :w, :3] = sample_image
        # binary prediction
        # bw_image = (255*rgba2rgb(
        #     binary_cmap(pred_sample_binary_map * 255)
        # )).astype(np.uint8)
        # bw_image = Image.fromarray(bw_image)
        # bw_image.save(outdir / img_name)
        placeholder[:h, w : 2 * w, :3] = rgba2rgb(
            binary_cmap(gt_sample_binary_map * 255)
        )
        placeholder[h : 2 * h, w : 2 * w, :3] = rgba2rgb(
            binary_cmap(pred_sample_binary_map)
        )

        # instance_predictions
        placeholder[:h, 2 * w : 3 * w, :3] = rgba2rgb(
            instance_map(
                (gt_sample_instance_map - np.min(gt_sample_instance_map))
                / (
                    np.max(gt_sample_instance_map)
                    - np.min(gt_sample_instance_map + 1e-10)
                )
            )
        )
        placeholder[h : 2 * h, 2 * w : 3 * w, :3] = rgba2rgb(
            instance_map(
                (pred_sample_instance_maps - np.min(pred_sample_instance_maps))
                / (
                    np.max(pred_sample_instance_maps)
                    - np.min(pred_sample_instance_maps + 1e-10)
                )
            )
        )
        gt_contours_polygon = [
            v["contour"] for v in ground_truth["instance_types"][0].values()
        ]
        gt_contours_polygon = [
            list(zip(poly[:, 0], poly[:, 1])) for poly in gt_contours_polygon
        ]
        gt_contour_colors_polygon = ["#70c6ff" for i in range(len(gt_contours_polygon))]
        gt_cell_image = Image.fromarray((sample_image * 255).astype(np.uint8)).convert(
            "RGB"
        )
        gt_drawing = ImageDraw.Draw(gt_cell_image)
        add_patch = lambda poly, color: gt_drawing.polygon(poly, outline=color, width=2)
        [
            add_patch(poly, c)
            for poly, c in zip(gt_contours_polygon, gt_contour_colors_polygon)
        ]
        placeholder[:h, 3 * w : 4 * w, :3] = np.asarray(gt_cell_image) / 255
        # pred
        pred_contours_polygon = [
            v["contour"] for v in predictions["instance_types"].values()
        ]
        pred_contours_polygon = [
            list(zip(poly[:, 0], poly[:, 1])) for poly in pred_contours_polygon
        ]
        pred_contour_colors_polygon = [
            "#70c6ff" for i in range(len(pred_contours_polygon))
        ]
        pred_cell_image = Image.fromarray(
            (sample_image * 255).astype(np.uint8)
        ).convert("RGB")
        pred_drawing = ImageDraw.Draw(pred_cell_image)
        add_patch = lambda poly, color: pred_drawing.polygon(
            poly, outline=color, width=2
        )
        [
            add_patch(poly, c)
            for poly, c in zip(pred_contours_polygon, pred_contour_colors_polygon)
        ]
        placeholder[h : 2 * h, 3 * w : 4 * w, :3] = np.asarray(pred_cell_image) / 255

        # plotting
        test_image = Image.fromarray((placeholder * 255).astype(np.uint8))
        test_image.save(outdir / f"raw_{img_name}")
        fig, axs = plt.subplots(figsize=(3, 2), dpi=1200)
        axs.imshow(placeholder)
        axs.set_xticks(np.arange(w / 2, 4 * w, w))
        axs.set_xticklabels(
            [
                "Image",
                "Binary-Cells",
                "Instances",
                "Countours",
            ],
            fontsize=6,
        )
        axs.xaxis.tick_top()

        axs.set_yticks(np.arange(h / 2, 2 * h, h))
        axs.set_yticklabels(["GT", "Pred."], fontsize=6)
        axs.tick_params(axis="both", which="both", length=0)
        grid_x = np.arange(w, 3 * w, w)
        grid_y = np.arange(h, 2 * h, h)

        for x_seg in grid_x:
            axs.axvline(x_seg, color="black")
        for y_seg in grid_y:
            axs.axhline(y_seg, color="black")

        if scores is not None:
            axs.text(
                20,
                1.85 * h,
                f"Dice: {str(np.round(scores[0], 2))}\nJac.: {str(np.round(scores[1], 2))}\nbPQ: {str(np.round(scores[2], 2))}",
                bbox={"facecolor": "white", "pad": 2, "alpha": 0.5},
                fontsize=4,
            )
        fig.suptitle(f"Patch Predictions for {img_name}", fontsize=6)
        fig.tight_layout()
        fig.savefig(outdir / f"pred_{img_name}")
        plt.close()


# CLI
class InferenceUNETR2dMoNuSegParser:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Perform UNETR2d inference for MoNuSeg dataset",
        )

        parser.add_argument(
            "--model",
            type=str,
            help="Model checkpoint file that is used for inference",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            help="Path to MoNuSeg dataset.",
        )
        parser.add_argument(
            "--outdir",
            type=str,
            help="Path to output directory to store results.",
        )
        parser.add_argument(
            "--gpu", type=int, help="Cuda-GPU ID for inference. Default: 0", default=0
        )
        parser.add_argument(
            "--magnification",
            type=int,
            help="Dataset Magnification. Either 20 or 40. Default: 40",
            choices=[20, 40],
            default=40,
        )
        parser.add_argument(
            "--patching",
            type=bool,
            help="Patch to 256px images. Default: False",
            default=False,
        )
        parser.add_argument(
            "--plots",
            type=bool,
            help="Generate result plots. Default: False",
            default=False,
        )

        self.parser = parser

    def parse_arguments(self) -> dict:
        opt = self.parser.parse_args()
        return vars(opt)


if __name__ == "__main__":
    configuration_parser = InferenceUNETR2dMoNuSegParser()
    configuration = configuration_parser.parse_arguments()
    print(configuration)

    inf = MoNuSegInference(
        model_path=configuration["model"],
        dataset_path=configuration["dataset"],
        outdir=configuration["outdir"],
        gpu=configuration["gpu"],
        patching=configuration["patching"],
        magnification=configuration["magnification"],
    )
    inf.run_inference(generate_plots=configuration["plots"])
