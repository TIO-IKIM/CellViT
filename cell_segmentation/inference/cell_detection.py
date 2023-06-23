# -*- coding: utf-8 -*-
# UNETR2d Inference Method for Patch-Wise Inference on a patches test set/Whole WSI
#
# Detect Cells with our Networks
# Patches dataset needs to have the follwoing requirements:
# Patch-Size must be 1024, with overlap of 64
#
# We provide preprocessing code here: ./preprocessing/patch_extraction/main_extraction.py
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

import argparse
import logging
import uuid
import warnings
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
import ujson
from einops import rearrange
from pandarallel import pandarallel

# from PIL import Image
from shapely import strtree
from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import Polygon

# from skimage.color import rgba2rgb
from torch.utils.data import DataLoader
from torchvision import transforms as T

from cell_segmentation.datasets.cell_graph_datamodel import CellGraphDataWSI
from cell_segmentation.utils.template_geojson import (
    get_template_point,
    get_template_segmentation,
)
from datamodel.wsi_datamodel import WSI
from models.segmentation.cell_segmentation.unetr2d import (
    UNETR2d,
    UNETR2dSAM,
    UNETR2dSAMUnshared,
    UNETR2dUnshared,
    UNETR2dVIT256,
    UNETR2dVIT256Unshared,
)
from preprocessing.encoding.datasets.patched_wsi_inference import PatchedWSIInference
from utils.file_handling import load_wsi_files_from_csv
from utils.logger import Logger
from utils.tools import unflatten_dict

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
pandarallel.initialize(progress_bar=False, nb_workers=12)

# color setup
COLOR_DICT = {
    1: [255, 0, 0],
    2: [34, 221, 77],
    3: [35, 92, 236],
    4: [254, 255, 0],
    5: [255, 159, 68],
}

TYPE_NUCLEI_DICT = {
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial",
}


class CellSegmentationInference:
    def __init__(self, model_path: Union[Path, str], gpu: int) -> None:
        """Cell Segmentation Inference class.

        After setup, a WSI can be processed by calling process_wsi method

        Args:
            model_path (Union[Path, str]): Path to model checkpoint
            gpu (int): CUDA GPU id to use
        """
        self.model_path = Path(model_path)
        self.device = f"cuda:{gpu}"
        self.__instantiate_logger()
        self.__load_model()
        self.__load_inference_transforms()

    def __instantiate_logger(self) -> None:
        """Instantiate logger

        Logger is using no formatters. Logs are stored in the run directory under the filename: inference.log
        """
        logger = Logger(
            level="INFO",
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

    def __load_inference_transforms(self):
        """Load the inference transformations from the run_configuration"""
        self.logger.info("Loading inference transformations")

        transform_settings = self.run_conf["transformations"]
        if "normalize" in transform_settings:
            mean = transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
            std = transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        self.inference_transforms = T.Compose(
            [T.ToTensor(), T.Normalize(mean=mean, std=std)]
        )

    def process_wsi(
        self,
        wsi: WSI,
        subdir_name: str = None,
        patch_size: int = 1024,
        overlap: int = 64,
        geojson: bool = False,
    ) -> None:
        """Process WSI file

        Args:
            wsi (WSI): WSI object
            subdir_name (str, optional): If provided, a subdir with the given name is created in the cell_detection folder.
                Helpful if you need to store different cell detection results next to each other. Defaults to None (no subdir).
            patch_size (int, optional): Patch-Size. Default to 1024.
            overlap (int, optional): Overlap between patches. Defaults to 64.
            geosjon (bool, optional): If a geojson export should be performed. Defaults to False.
        """
        self.logger.info(f"Processing WSI: {wsi.name}")

        wsi_inference_dataset = PatchedWSIInference(
            wsi, transform=self.inference_transforms
        )
        wsi_inference_dataloader = DataLoader(
            dataset=wsi_inference_dataset,
            batch_size=8,
            num_workers=16,
            shuffle=False,
            collate_fn=wsi_inference_dataset.collate_batch,
            pin_memory=False,
        )
        dataset_config = self.run_conf["dataset_config"]
        nuclei_types = dataset_config["nuclei_types"]

        if subdir_name is not None:
            outdir = Path(wsi.patched_slide_path) / "cell_detection" / subdir_name
        else:
            outdir = Path(wsi.patched_slide_path) / "cell_detection"
        outdir.mkdir(exist_ok=True, parents=True)

        cell_dict_wsi = []  # for storing all cell information
        cell_dict_detection = []  # for storing only the centroids

        graph_data = {
            "cell_tokens": [],
            "positions": [],
            "contours": [],
            "metadata": {"wsi_metadata": wsi.metadata, "nuclei_types": nuclei_types},
        }
        processed_patches = []

        with torch.no_grad():
            for batch in tqdm.tqdm(
                wsi_inference_dataloader, total=len(wsi_inference_dataloader)
            ):
                patches = batch[0].to(self.device)

                metadata = batch[1]
                predictions_ = self.model.forward(patches, retrieve_tokens=True)

                # reshape, apply softmax to segmentation maps
                predictions = self.model.reshape_model_output(predictions_, self.device)
                instance_types, tokens = self.get_cell_predictions_with_tokens(
                    predictions, magnification=wsi.metadata["magnification"]
                )

                # unpack each patch from batch
                for idx, (patch_instance_types, patch_metadata) in enumerate(
                    zip(instance_types, metadata)
                ):
                    # add global patch metadata
                    patch_cell_detection = {}
                    patch_cell_detection["patch_metadata"] = patch_metadata
                    patch_cell_detection["type_map"] = dataset_config["nuclei_types"]

                    processed_patches.append(
                        f"{patch_metadata['row']}_{patch_metadata['col']}"
                    )

                    # calculate coordinate on highest magnifications
                    wsi_scaling_factor = patch_metadata["wsi_metadata"]["downsampling"]
                    patch_size = patch_metadata["wsi_metadata"]["patch_size"]
                    x_global = int(
                        patch_metadata["row"] * patch_size * wsi_scaling_factor
                        - (patch_metadata["row"] + 0.5) * overlap
                    )
                    y_global = int(
                        patch_metadata["col"] * patch_size * wsi_scaling_factor
                        - (patch_metadata["col"] + 0.5) * overlap
                    )

                    # extract cell information
                    for cell in patch_instance_types.values():
                        if cell["type"] == nuclei_types["Background"]:
                            continue
                        offset_global = np.array([x_global, y_global])
                        centroid_global = cell["centroid"] + np.flip(offset_global)
                        contour_global = cell["contour"] + np.flip(offset_global)
                        bbox_global = cell["bbox"] + offset_global
                        cell_dict = {
                            "bbox": bbox_global.tolist(),
                            "centroid": centroid_global.tolist(),
                            "contour": contour_global.tolist(),
                            "type_prob": cell["type_prob"],
                            "type": cell["type"],
                            "patch_coordinates": [
                                patch_metadata["row"],
                                patch_metadata["col"],
                            ],
                            "cell_status": get_cell_position_marging(
                                cell["bbox"], 1024, 64
                            ),
                            "offset_global": offset_global.tolist()
                            # optional: Local positional information
                            # "bbox_local": cell["bbox"].tolist(),
                            # "centroid_local": cell["centroid"].tolist(),
                            # "contour_local": cell["contour"].tolist(),
                        }
                        cell_detection = {
                            "bbox": bbox_global.tolist(),
                            "centroid": centroid_global.tolist(),
                            "type": cell["type"],
                        }
                        if np.max(cell["bbox"]) == 1024 or np.min(cell["bbox"]) == 0:
                            position = get_cell_position(cell["bbox"], 1024)
                            cell_dict["edge_position"] = True
                            cell_dict["edge_information"] = {}
                            cell_dict["edge_information"]["position"] = position
                            cell_dict["edge_information"][
                                "edge_patches"
                            ] = get_edge_patch(
                                position, patch_metadata["row"], patch_metadata["col"]
                            )
                        else:
                            cell_dict["edge_position"] = False

                        cell_dict_wsi.append(cell_dict)
                        cell_dict_detection.append(cell_detection)

                        # get the cell token
                        bb_index = cell["bbox"] / self.model.patch_size
                        bb_index[0, :] = np.floor(bb_index[0, :])
                        bb_index[1, :] = np.ceil(bb_index[1, :])
                        bb_index = bb_index.astype(np.uint8)
                        cell_token = tokens[
                            idx,
                            bb_index[0, 1] : bb_index[1, 1],
                            bb_index[0, 0] : bb_index[1, 0],
                            :,
                        ]
                        cell_token = torch.mean(
                            rearrange(cell_token, "H W D -> (H W) D"), dim=0
                        )

                        graph_data["cell_tokens"].append(cell_token)
                        graph_data["positions"].append(torch.Tensor(centroid_global))
                        graph_data["contours"].append(torch.Tensor(contour_global))

        # post processing
        self.logger.info(f"Detected cells before cleaning: {len(cell_dict_wsi)}")
        keep_idx = self.post_process_edge_cells(cell_list=cell_dict_wsi)
        cell_dict_wsi = [cell_dict_wsi[idx_c] for idx_c in keep_idx]
        cell_dict_detection = [cell_dict_detection[idx_c] for idx_c in keep_idx]
        graph_data["cell_tokens"] = [
            graph_data["cell_tokens"][idx_c] for idx_c in keep_idx
        ]
        graph_data["positions"] = [graph_data["positions"][idx_c] for idx_c in keep_idx]
        graph_data["contours"] = [graph_data["contours"][idx_c] for idx_c in keep_idx]
        self.logger.info(f"Detected cells after cleaning: {len(keep_idx)}")

        self.logger.info(
            f"Processed all patches. Storing final results: {str(outdir / f'cells.json')} and cell_detection.json"
        )
        cell_dict_wsi = {
            "wsi_metadata": wsi.metadata,
            "processed_patches": processed_patches,
            "type_map": dataset_config["nuclei_types"],
            "cells": cell_dict_wsi,
        }
        with open(str(outdir / "cells.json"), "w") as outfile:
            ujson.dump(cell_dict_wsi, outfile, indent=2)
        if geojson:
            self.logger.info("Converting segmentation to geojson")
            geojson_list = self.convert_geojson(cell_dict_wsi["cells"], True)
            with open(str(str(outdir / "cells.geojson")), "w") as outfile:
                ujson.dump(geojson_list, outfile, indent=2)

        cell_dict_detection = {
            "wsi_metadata": wsi.metadata,
            "processed_patches": processed_patches,
            "type_map": dataset_config["nuclei_types"],
            "cells": cell_dict_detection,
        }
        with open(str(outdir / "cell_detection.json"), "w") as outfile:
            ujson.dump(cell_dict_detection, outfile, indent=2)
        if geojson:
            self.logger.info("Converting detection to geojson")
            geojson_list = self.convert_geojson(cell_dict_wsi["cells"], False)
            with open(str(str(outdir / "cell_detection.geojson")), "w") as outfile:
                ujson.dump(geojson_list, outfile, indent=2)

        self.logger.info(
            f"Create cell graph with embeddings and save it under: {str(outdir / 'cells.pt')}"
        )
        graph = CellGraphDataWSI(
            x=torch.stack(graph_data["cell_tokens"]),
            positions=torch.stack(graph_data["positions"]),
            contours=graph_data["contours"],
            metadata=graph_data["metadata"],
        )
        torch.save(graph, outdir / "cells.pt")

        cell_stats_df = pd.DataFrame(cell_dict_wsi["cells"])
        cell_stats = dict(cell_stats_df.value_counts("type"))
        nuclei_types_inverse = {v: k for k, v in nuclei_types.items()}
        verbose_stats = {nuclei_types_inverse[k]: v for k, v in cell_stats.items()}
        self.logger.info(f"Finished with cell detection for WSI {wsi.name}")
        self.logger.info("Stats:")
        self.logger.info(f"{verbose_stats}")

    def get_cell_predictions_with_tokens(
        self, predictions: dict, magnification: int = 40
    ) -> Tuple[List[dict], torch.Tensor]:
        """Take the raw predictions, apply softmax and calculate type instances

        Args:
            predictions (dict): Network predictions with tokens. Keys:
            magnification (int, optional): WSI magnification. Defaults to 40.

        Returns:
            Tuple[List[dict], torch.Tensor]:
                * List[dict]: List with a dictionary for each batch element with cell seg results
                    Contains bbox, contour, 2D-position, type and type_prob for each cell
                * List[dict]: Network tokens on cpu device with shape (batch_size, num_tokens_h, num_tokens_w, embd_dim)
        """
        predictions["nuclei_binary_map"] = F.softmax(
            predictions["nuclei_binary_map"], dim=-1
        )
        predictions["nuclei_type_map"] = F.softmax(
            predictions["nuclei_type_map"], dim=-1
        )

        # get the instance types
        (
            _,
            instance_types,
        ) = self.model.calculate_instance_map(predictions, magnification=magnification)
        # get the tokens
        tokens = predictions["tokens"].to("cpu")

        return instance_types, tokens

    def post_process_edge_cells(self, cell_list: List[dict]) -> List[int]:
        """Use the CellPostProcessor to remove multiple cells and merge due to overlap

        Args:
            cell_list (List[dict]): List with cell-dictionaries. Required keys:
                * bbox
                * centroid
                * contour
                * type_prob
                * type
                * patch_coordinates
                * cell_status
                * offset_global

        Returns:
            List[int]: List with integers of cells that should be kept
        """
        cell_processor = CellPostProcessor(cell_list, self.logger)
        cleaned_cells = cell_processor.post_process_cells()

        return list(cleaned_cells.index.values)

    def convert_geojson(
        self, cell_list: list[dict], polygons: bool = False
    ) -> List[dict]:
        """Convert a list of cells to a geojson object

        Either a segmentation object (polygon) or detection points are converted

        Args:
            cell_list (list[dict]): Cell list with dict entry for each cell.
                Required keys for detection:
                    * type
                    * centroid
                Required keys for segmentation:
                    * type
                    * contour
            polygons (bool, optional): If polygon segmentations (True) or detection points (False). Defaults to False.

        Returns:
            List[dict]: Geojson like list
        """
        if polygons:
            cell_segmentation_df = pd.DataFrame(cell_list)
            detected_types = sorted(cell_segmentation_df.type.unique())
            geojson_placeholder = []
            for cell_type in detected_types:
                cells = cell_segmentation_df[cell_segmentation_df["type"] == cell_type]
                contours = cells["contour"].to_list()
                final_c = []
                for c in contours:
                    c.append(c[0])
                    final_c.append([c])

                cell_geojson_object = get_template_segmentation()
                cell_geojson_object["id"] = str(uuid.uuid4())
                cell_geojson_object["geometry"]["coordinates"] = final_c
                cell_geojson_object["properties"]["classification"][
                    "name"
                ] = TYPE_NUCLEI_DICT[cell_type]
                cell_geojson_object["properties"]["classification"][
                    "color"
                ] = COLOR_DICT[cell_type]
                geojson_placeholder.append(cell_geojson_object)
        else:
            cell_detection_df = pd.DataFrame(cell_list)
            detected_types = sorted(cell_detection_df.type.unique())
            geojson_placeholder = []
            for cell_type in detected_types:
                cells = cell_detection_df[cell_detection_df["type"] == cell_type]
                centroids = cells["centroid"].to_list()
                cell_geojson_object = get_template_point()
                cell_geojson_object["id"] = str(uuid.uuid4())
                cell_geojson_object["geometry"]["coordinates"] = centroids
                cell_geojson_object["properties"]["classification"][
                    "name"
                ] = TYPE_NUCLEI_DICT[cell_type]
                cell_geojson_object["properties"]["classification"][
                    "color"
                ] = COLOR_DICT[cell_type]
                geojson_placeholder.append(cell_geojson_object)
        return geojson_placeholder


class CellPostProcessor:
    def __init__(self, cell_list: List[dict], logger: logging.Logger) -> None:
        """POst-Processing a list of cells from one WSI

        Args:
            cell_list (List[dict]): List with cell-dictionaries. Required keys:
                * bbox
                * centroid
                * contour
                * type_prob
                * type
                * patch_coordinates
                * cell_status
                * offset_global
            logger (logging.Logger): Logger
        """
        self.logger = logger
        self.logger.info("Initializing Cell-Postprocessor - This may take a while")
        self.cell_df = pd.DataFrame(cell_list)
        self.cell_df = self.cell_df.parallel_apply(convert_coordinates, axis=1)

        self.mid_cells = self.cell_df[
            self.cell_df["cell_status"] == 0
        ]  # cells in the mid
        self.cell_df_margin = self.cell_df[
            self.cell_df["cell_status"] != 0
        ]  # cells either torching the border or margin

    def post_process_cells(self) -> pd.DataFrame:
        """Main Post-Processing coordinator, entry point

        Returns:
            pd.DataFrame: DataFrame with post-processed and cleaned cells
        """
        self.logger.info("Finding edge-cells")
        cleaned_edge_cells = self._clean_edge_cells()
        self.logger.info("Removal of cells detected multiple times")
        cleaned_edge_cells = self._remove_overlap(cleaned_edge_cells)

        # merge with mid cells
        postprocessed_cells = pd.concat(
            [self.mid_cells, cleaned_edge_cells]
        ).sort_index()

        return postprocessed_cells

    def _clean_edge_cells(self) -> pd.DataFrame:
        """Create a DataFrame that just contains all margin cells (cells inside the margin, not touching the border)
        and border/edge cells (touching border) with no overlapping equivalent (e.g, if patch has no neighbour)

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        edge_cells = pd.DataFrame(
            columns=self.cell_df_margin.columns
        )  # cells torching the border without having an overlap from other patches
        margin_cells = pd.DataFrame(
            columns=self.cell_df_margin.columns
        )  # cells at the margin

        for idx, cell_info in tqdm.tqdm(
            self.cell_df_margin.iterrows(), total=len(self.cell_df_margin)
        ):
            if cell_info["edge_position"]:
                # edge cell
                edge_information = dict(cell_info["edge_information"])
                edge_patch = edge_information["edge_patches"][0]
                cells = self.cell_df_margin[
                    self.cell_df_margin["patch_coordinates"]
                    == f"{edge_patch[0]}_{edge_patch[1]}"
                ]
                # i think we need to transform the coordinates first
                if len(cells) == 0:
                    # the edge patch does not exist, insert at for cleaned edge cells
                    edge_cells.loc[idx, :] = cell_info
            else:
                # margin cell
                margin_cells.loc[idx, :] = cell_info

        cleaned_edge_cells = pd.concat([margin_cells, edge_cells])
        return cleaned_edge_cells.sort_index()

    def _remove_overlap(self, cleaned_edge_cells: pd.DataFrame) -> pd.DataFrame:
        """Remove overlapping cells from provided DataFrame

        Args:
            cleaned_edge_cells (pd.DataFrame): DataFrame that should be cleaned

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        merged_cells = pd.DataFrame(columns=self.cell_df_margin.columns)

        poly_list = []
        for idx, cell_info in cleaned_edge_cells.iterrows():
            poly = Polygon(cell_info["contour"])
            if not poly.is_valid:
                self.logger.debug("Found invalid polygon - Fixing with buffer 0")
                poly = poly.buffer(0)
            poly.uid = idx
            poly_list.append(poly)

        # use an strtree for fast querying
        tree = strtree.STRtree(poly_list)

        # keep track of checked cells
        iterated_cells = []

        for query_poly in tqdm.tqdm(poly_list, total=len(poly_list)):
            if query_poly not in iterated_cells:
                intersected_polygons = tree.query(
                    query_poly
                )  # this also contains a self-intersection
                if (
                    len(intersected_polygons) > 1
                ):  # we have more at least one intersection with another cell
                    submergers = []  # all cells that overlap with query
                    for inter_poly in intersected_polygons:
                        if (
                            inter_poly.uid != query_poly.uid
                            and inter_poly.uid not in iterated_cells
                        ):
                            if (
                                query_poly.intersection(inter_poly).area
                                / query_poly.area
                                > 0.5
                                or query_poly.intersection(inter_poly).area
                                / inter_poly.area
                                > 0.5
                            ):
                                submergers.append(inter_poly)
                                iterated_cells.append(inter_poly.uid)
                    # catch block: empty list -> some cells are touching, but not overlapping strongly enough
                    if len(submergers) == 0:
                        merged_cells.loc[query_poly.uid, :] = cleaned_edge_cells.loc[
                            query_poly.uid
                        ]
                    else:  # merging strategy: take the biggest cell, other merging strategies needs to get implemented
                        selected_poly_index = np.argmax(
                            np.array([p.area for p in submergers])
                        )
                        selected_poly_uid = submergers[selected_poly_index].uid
                        merged_cells.loc[selected_poly_uid, :] = cleaned_edge_cells.loc[
                            selected_poly_uid
                        ]
                else:
                    # no intersection, just add
                    merged_cells.loc[query_poly.uid, :] = cleaned_edge_cells.loc[
                        query_poly.uid
                    ]
                iterated_cells.append(query_poly.uid)

        return merged_cells.sort_index()


def convert_coordinates(row: pd.Series) -> pd.Series:
    """Convert a row from x,y type to one string representation of the patch position for fast querying
    Repr: x_y

    Args:
        row (pd.Series): Row to be processed

    Returns:
        pd.Series: Processed Row
    """
    x, y = row["patch_coordinates"]
    row["patch_row"] = x
    row["patch_col"] = y
    row["patch_coordinates"] = f"{x}_{y}"
    return row


def get_cell_position(bbox: np.ndarray, patch_size: int = 1024) -> List[int]:
    """Get cell position as a list

    Entry is 1, if cell touches the border: [top, right, down, left]

    Args:
        bbox (np.ndarray): Bounding-Box of cell
        patch_size (int, optional): Patch-size. Defaults to 1024.

    Returns:
        List[int]: List with 4 integers for each position
    """
    # bbox = 2x2 array in h, w style
    # bbox[0,0] = upper position (height)
    # bbox[1,0] = lower dimension (height)
    # boox[0,1] = left position (width)
    # bbox[1,1] = right position (width)
    # bbox[:,0] -> x dimensions
    top, left, down, right = False, False, False, False
    if bbox[0, 0] == 0:
        top = True
    if bbox[0, 1] == 0:
        left = True
    if bbox[1, 0] == patch_size:
        down = True
    if bbox[1, 1] == patch_size:
        right = True
    position = [top, right, down, left]
    position = [int(pos) for pos in position]

    return position


def get_cell_position_marging(
    bbox: np.ndarray, patch_size: int = 1024, margin: int = 64
) -> int:
    """Get the status of the cell, describing the cell position

    A cell is either in the mid (0) or at one of the borders (1-8)

    # Numbers are assigned clockwise, starting from top left
    # i.e., top left = 1, top = 2, top right = 3, right = 4, bottom right = 5 bottom = 6, bottom left = 7, left = 8
    # Mid status is denoted by 0

    Args:
        bbox (np.ndarray): Bounding Box of cell
        patch_size (int, optional): Patch-Size. Defaults to 1024.
        margin (int, optional): Margin-Size. Defaults to 64.

    Returns:
        int: Cell Status
    """
    cell_status = None
    if np.max(bbox) > patch_size - margin or np.min(bbox) < margin:
        if bbox[0, 0] < margin:
            # top left, top or top right
            if bbox[0, 1] < margin:
                # top left
                cell_status = 1
            elif bbox[1, 1] > patch_size - margin:
                # top right
                cell_status = 3
            else:
                # top
                cell_status = 2
        elif bbox[1, 1] > patch_size - margin:
            # top right, right or bottom right
            if bbox[1, 0] > patch_size - margin:
                # bottom right
                cell_status = 5
            else:
                # right
                cell_status = 4
        elif bbox[1, 0] > patch_size - margin:
            # bottom right, bottom, bottom left
            if bbox[0, 1] < margin:
                # bottom left
                cell_status = 7
            else:
                # bottom
                cell_status = 6
        elif bbox[0, 1] < margin:
            # bottom left, left, top left, but only left is left
            cell_status = 8
    else:
        cell_status = 0

    return cell_status


def get_edge_patch(position, row, col):
    # row starting on bottom or on top?
    if position == [1, 0, 0, 0]:
        # top
        return [[row - 1, col]]
    if position == [1, 1, 0, 0]:
        # top and right
        return [[row - 1, col], [row - 1, col + 1], [row, col + 1]]
    if position == [0, 1, 0, 0]:
        # right
        return [[row, col + 1]]
    if position == [0, 1, 1, 0]:
        # right and down
        return [[row, col + 1], [row + 1, col + 1], [row + 1, col]]
    if position == [0, 0, 1, 0]:
        # down
        return [[row + 1, col]]
    if position == [0, 0, 1, 1]:
        # down and left
        return [[row + 1, col], [row + 1, col - 1], [row, col - 1]]
    if position == [0, 0, 0, 1]:
        # left
        return [[row, col - 1]]
    if position == [1, 0, 0, 1]:
        # left and top
        return [[row, col - 1], [row - 1, col - 1], [row - 1, col]]


# CLI
class InferenceWSIParser:
    """Parser"""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Perform UNETR2d inference for given run-directory with model checkpoints and logs",
        )
        requiredNamed = parser.add_argument_group("required named arguments")
        requiredNamed.add_argument(
            "--model",
            type=str,
            help="Model checkpoint file that is used for inference",
            default="/homes/fhoerst/histo-projects/DigitalHistologyHub/outputs/cell_seg_runs/baseline-results/ViT256/2023-05-17T153418_UNetr2D-Vit-256-baseline-fold-3/checkpoints/model_best.pth",
            # required=True,
        )
        parser.add_argument(
            "--gpu", type=int, help="Cuda-GPU ID for inference. Default: 0", default=0
        )
        parser.add_argument(
            "--magnification",
            type=float,
            help="Network magnification. Is used for checking patch magnification such that we use the correct resolution for network. Default: 40",
            default=40,
        )
        parser.add_argument(
            "--outdir_subdir",
            type=str,
            help="If provided, a subdir with the given name is created in the cell_detection folder where the results are stored. Default: None",
            default=None,
        )
        parser.add_argument(
            "--geojson",
            action="store_true",
            help="Set this flag to export results as additional geojson files for loading them into Software like QuPath.",
        )

        # subparsers for either loading a WSI or a WSI folder

        # WSI
        subparsers = parser.add_subparsers(
            dest="command",
            description="Main run command for either performing inference on single WSI-file or on whole dataset",
        )
        subparser_wsi = subparsers.add_parser(
            "process_wsi", description="Process a single WSI file"
        )
        subparser_wsi.add_argument(
            "--wsi_path",
            type=str,
            help="Path to WSI file",
        )
        subparser_wsi.add_argument(
            "--patched_slide_path",
            type=str,
            help="Path to patched WSI file (specific WSI file, not parent path of patched slide dataset)",
        )

        # Dataset
        subparser_dataset = subparsers.add_parser(
            "process_dataset",
            description="Process a whole dataset",  # TODO: link dataset documentation
        )
        subparser_dataset.add_argument(
            "--wsi_paths", type=str, help="Path to the folder where all WSI are stored"
        )
        subparser_dataset.add_argument(
            "--patch_dataset_path",
            type=str,
            help="Path to the folder where the patch dataset is stored",
        )
        subparser_dataset.add_argument(
            "--filelist",
            type=str,
            help="Filelist with WSI to process. Must be a .csv file with one row denoting the filenames (named 'Filename')."
            "If not provided, all WSI files with given ending in the filelist are processed.",
            default=None,
        )
        subparser_dataset.add_argument(
            "--wsi_extension",
            type=str,
            help="The extension types used for the WSI files, see configs.python.config (WSI_EXT)",
            default="svs",
        )

        self.parser = parser

    def parse_arguments(self) -> dict:
        opt = self.parser.parse_args()
        return vars(opt)


def check_wsi(wsi: WSI, magnification: float = 40.0):
    """Check if provided patched WSI is having the right settings

    Args:
        wsi (WSI): WSI to check
        magnification (float, optional): Check magnification. Defaults to 40.0.

    Raises:
        RuntimeError: The magnification is not matching to the network input magnification.
        RuntimeError: The patch-size is not devisible by 256.
        RunTimeError: The patch-size is not 1024
        RunTimeError: The overlap is not 64px sized
    """
    if wsi.metadata["magnification"] is not None:
        patch_magnification = float(wsi.metadata["magnification"])
    else:
        patch_magnification = float(
            float(wsi.metadata["base_magnification"]) / wsi.metadata["downsampling"]
        )
    patch_size = int(wsi.metadata["patch_size"])

    if patch_magnification != magnification:
        raise RuntimeError(
            "The magnification is not matching to the network input magnification."
        )
    if (patch_size % 256) != 0:
        raise RuntimeError("The patch-size must be devisible by 256.")
    if wsi.metadata["patch_size"] != 1024:
        raise RuntimeError("The patch-size must be 1024.")
    if wsi.metadata["patch_overlap"] != 64:
        raise RuntimeError("The patch-overlap must be 64")


if __name__ == "__main__":
    configuration_parser = InferenceWSIParser()
    configuration = configuration_parser.parse_arguments()
    command = configuration["command"]

    cell_segmentation = CellSegmentationInference(
        model_path=configuration["model"], gpu=configuration["gpu"]
    )

    if command.lower() == "process_wsi":
        cell_segmentation.logger.info("Processing single WSI file")
        wsi_path = Path(configuration["wsi_path"])
        wsi_name = wsi_path.stem
        wsi_file = WSI(
            name=wsi_name,
            patient=wsi_name,
            slide_path=wsi_path,
            patched_slide_path=configuration["patched_slide_path"],
        )
        check_wsi(wsi=wsi_file, magnification=configuration["magnification"])
        cell_segmentation.process_wsi(
            wsi_file,
            subdir_name=configuration["outdir_subdir"],
            geojson=configuration["geojson"],
        )

    elif command.lower() == "process_dataset":
        cell_segmentation.logger.info("Processing whole dataset")
        if configuration["filelist"] is not None:
            if Path(configuration["filelist"]).suffix != ".csv":
                raise ValueError("Filelist must be a .csv file!")
            cell_segmentation.logger.info(
                f"Loading files from filelist {configuration['filelist']}"
            )
            wsi_filelist = load_wsi_files_from_csv(
                csv_path=configuration["filelist"],
                wsi_extension=configuration["wsi_extension"],
            )
        else:
            cell_segmentation.logger.info(
                f"Loading all files from folder {configuration['wsi_paths']}. No filelist provided."
            )
            wsi_filelist = [
                f
                for f in sorted(
                    Path(configuration["wsi_paths"]).glob(
                        f"**/*.{configuration['wsi_extension']}"
                    )
                )
            ]
        for i, wsi_path in enumerate(wsi_filelist):
            wsi_name = wsi_path.stem
            patched_slide_path = Path(configuration["patch_dataset_path"]) / wsi_name
            cell_segmentation.logger.info(f"File {i+1}/{len(wsi_filelist)}: {wsi_name}")
            wsi_file = WSI(
                name=wsi_name,
                patient=wsi_name,
                slide_path=wsi_path,
                patched_slide_path=patched_slide_path,
            )
            check_wsi(wsi=wsi_file, magnification=configuration["magnification"])
            cell_segmentation.process_wsi(
                wsi_file,
                subdir_name=configuration["outdir_subdir"],
                geojson=configuration["geojson"],
            )
