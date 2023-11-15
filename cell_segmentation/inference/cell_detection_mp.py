# -*- coding: utf-8 -*-
# CellViT Inference Method for Patch-Wise Inference on a patches test set/Whole WSI
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
# @ Erik Ylipää, erik.ylipaa@gmail.com
# Linköping University
# Luleå, Sweden


from dataclasses import dataclass
from functools import partial
import inspect
from io import BytesIO
import os
import queue
import sys
import multiprocessing
from multiprocessing.pool import ThreadPool
import zipfile
from time import sleep

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

from cellvit.cell_segmentation.utils.post_proc import DetectionCellPostProcessor


import argparse
import logging
import uuid
import warnings
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Literal, OrderedDict, Tuple, Union, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
import ujson
from einops import rearrange

# from PIL import Image
from shapely import strtree
from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import Polygon, MultiPolygon


# from skimage.color import rgba2rgb
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
#from torch.profiler import profile, record_function, ProfilerActivity

from cellvit.cell_segmentation.datasets.cell_graph_datamodel import CellGraphDataWSI
from cellvit.cell_segmentation.utils.template_geojson import (
    get_template_point,
    get_template_segmentation,
)
from cellvit.datamodel.wsi_datamodel import WSI
from cellvit.models.segmentation.cell_segmentation.cellvit import (
    CellViT,
    CellViT256,
    CellViT256Unshared,
    CellViTSAM,
    CellViTSAMUnshared,
    CellViTUnshared,
)
from cellvit.preprocessing.encoding.datasets.patched_wsi_inference import PatchedWSIInference
from cellvit.utils.file_handling import load_wsi_files_from_csv
from cellvit.utils.logger import Logger
from cellvit.utils.tools import unflatten_dict

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
#pandarallel.initialize(progress_bar=False, nb_workers=12)



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

# This file will be used to indicate that a image has been processed
FLAG_FILE_NAME = ".cell_detection_done"

def load_wsi(wsi_path, overwrite=False):
    try:
        wsi_name = wsi_path.stem
        patched_slide_path = Path(configuration["patch_dataset_path"]) / wsi_name
        flag_file_path = patched_slide_path / "cell_detection" / FLAG_FILE_NAME
        if not overwrite and flag_file_path.exists():
            return
        wsi_file = WSI(
            name=wsi_name,
            patient=wsi_name,
            slide_path=wsi_path,
            patched_slide_path=patched_slide_path,
        )
        check_wsi(wsi=wsi_file, magnification=configuration["magnification"])
        return wsi_file
    except BaseException as e:
        e.wsi_file = wsi_path
        return e
            

class InferenceWSIDataset(Dataset):
    def __init__(self, wsi_filelist, n_workers: int = 0, overwrite=False, transform: Callable = None):
        self.wsi_files = []

        # This index will contain a repeat of all the wsi objects the number of
        # patches they have. This means that it will be as long as the total number
        # of patches in all WSI files. One can simply get the desired patch by 
        # subscripting into this list to get the correct WSI file object and 
        # pertinent metadata
        self.wsi_index = []
        self.transform = transform

        pb = tqdm.trange(len(wsi_filelist), desc='Loading WSI file list')
        already_processed_files = []
        if n_workers > 0:
            #Since this is mostly and IO-bound task, we use a thread pool
            #with multiprocessing.Pool(n_workers) as pool:
            with ThreadPool(n_workers) as pool:
                load_wsi_partial = partial(load_wsi, overwrite=overwrite)
                for wsi_file in pool.imap(load_wsi_partial, wsi_filelist):
                    if isinstance(wsi_file, BaseException):
                        logging.warn(f"Could not load file {wsi_file.wsi_file}, caught exception {str(wsi_file)}")
                    elif wsi_file is None:
                        already_processed_files.append(wsi_file)
                    else:                    
                        self.wsi_files.append(wsi_file)
                        n_patches = wsi_file.get_number_patches()
                        indexing_info = [(wsi_file, i) for i in range(n_patches)]
                        self.wsi_index.extend(indexing_info)
                    pb.update()
        else:
            for wsi_file_path in wsi_filelist:
                wsi_file = load_wsi(wsi_file_path, overwrite)
                if isinstance(wsi_file, BaseException):
                    logging.warn(f"Could not load file {wsi_file.wsi_file}, caught exception {str(wsi_file)}")
                elif wsi_file is None:
                    already_processed_files.append(wsi_file)
                else:                    
                    self.wsi_files.append(wsi_file)
                    n_patches = wsi_file.get_number_patches()
                    indexing_info = [(wsi_file, i) for i in range(n_patches)]
                    self.wsi_index.extend(indexing_info)
                pb.update()
        

    def __len__(self):
        return len(self.wsi_index)

    def __getitem__(self, item):
        wsi_file, local_idx = self.wsi_index[item]
        patch, metadata = wsi_file.get_patch(local_idx, self.transform)
        return patch, local_idx, wsi_file, metadata

    def get_n_files(self):
        return len(self.wsi_files)


def wsi_patch_collator(batch):
    patches, local_idx, wsi_file, metadata = zip(*batch) # Transpose the batch
    patches = torch.stack(patches)
    return patches, local_idx, wsi_file, metadata


def f_post_processing_worker(wsi_file, wsi_work_list, postprocess_arguments):
    local_idxs, predictions_records, metadata = zip(*wsi_work_list)
    # Merge the prediction records into a single dictionary again.
    predictions = defaultdict(list)
    for record in predictions_records:
        for k,v in record.items():
            predictions[k].append(v)
    predictions_stacked = {k: torch.stack(v).to(torch.float32) for k,v in predictions.items()}
    postprocess_predictions(predictions_stacked, metadata, wsi_file, postprocess_arguments)


@dataclass
class PostprocessArguments:
    n_images: int
    num_nuclei_classes: int
    dataset_config: Dict
    overlap: int 
    patch_size: int
    geojson: bool
    subdir_name: str
    logger: Logger
    n_workers: int = 0
    wait_time: float = 2.


def postprocess_predictions(predictions, metadata, wsi, postprocessing_args: PostprocessArguments):
    # logger = postprocessing_args.logger
    logger = logging.getLogger()
    num_nuclei_classes = postprocessing_args.num_nuclei_classes
    dataset_config = postprocessing_args.dataset_config
    overlap = postprocessing_args.overlap
    patch_size = postprocessing_args.patch_size
    geojson = postprocessing_args.geojson
    subdir_name = postprocessing_args.subdir_name

    if subdir_name is not None:
        outdir = Path(wsi.patched_slide_path) / "cell_detection" / subdir_name
    else:
        outdir = Path(wsi.patched_slide_path) / "cell_detection"
    outdir.mkdir(exist_ok=True, parents=True)

    outfile = outdir / "cell_detection.zip"

    instance_types, tokens = get_cell_predictions_with_tokens(num_nuclei_classes,
        predictions, magnification=wsi.metadata["magnification"]
    )

    processed_patches = []
    # unpack each patch from batch
    cell_dict_wsi = []  # for storing all cell information
    cell_dict_detection = []  # for storing only the centroids
    nuclei_types = dataset_config["nuclei_types"]

    graph_data = {
            "cell_tokens": [],
            "positions": [],
            "contours": [],
            "metadata": {"wsi_metadata": wsi.metadata, "nuclei_types": nuclei_types},
        }

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
        # wsi_scaling_factor = patch_metadata["wsi_metadata"]["downsampling"]
        # patch_size = patch_metadata["wsi_metadata"]["patch_size"]
        wsi_scaling_factor = wsi.metadata["downsampling"]
        patch_size = wsi.metadata["patch_size"]
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
            bb_index = cell["bbox"] / patch_size
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
    logger.info(f"Detected cells before cleaning: {len(cell_dict_wsi)}")
    keep_idx = post_process_edge_cells(cell_list=cell_dict_wsi, logger=logger)
    cell_dict_wsi = [cell_dict_wsi[idx_c] for idx_c in keep_idx]
    cell_dict_detection = [cell_dict_detection[idx_c] for idx_c in keep_idx]
    graph_data["cell_tokens"] = [
    graph_data["cell_tokens"][idx_c] for idx_c in keep_idx
    ]
    graph_data["positions"] = [graph_data["positions"][idx_c] for idx_c in keep_idx]
    graph_data["contours"] = [graph_data["contours"][idx_c] for idx_c in keep_idx]
    logger.info(f"Detected cells after cleaning: {len(keep_idx)}")

    logger.info(
    f"Processed all patches. Storing final results: {str(outdir / f'cells.json')} and cell_detection.json"
    )
    cell_dict_wsi = {
    "wsi_metadata": wsi.metadata,
    "processed_patches": processed_patches,
    "type_map": dataset_config["nuclei_types"],
    "cells": cell_dict_wsi,
    }
    
    with zipfile.ZipFile(outfile, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("cells.json", ujson.dumps(cell_dict_wsi, outfile, indent=2))
        
        if geojson:
            logger.info("Converting segmentation to geojson")
        
        geojson_list = convert_geojson(cell_dict_wsi["cells"], True)
        zf.writestr("cells.geojson", ujson.dumps(geojson_list, outfile, indent=2))

        cell_dict_detection = {
        "wsi_metadata": wsi.metadata,
        "processed_patches": processed_patches,
        "type_map": dataset_config["nuclei_types"],
        "cells": cell_dict_detection,
        }
        zf.writestr("cell_detection.json", ujson.dumps(cell_dict_detection, outfile, indent=2))
        if geojson:
            logger.info("Converting detection to geojson")
        geojson_list = convert_geojson(cell_dict_wsi["cells"], False)
        zf.writestr("cell_detection.geojson", ujson.dumps(geojson_list, outfile, indent=2))

        logger.info(
        f"Create cell graph with embeddings and save it under: {str(outdir / 'cells.pt')}"
        )
        graph = CellGraphDataWSI(
        x=torch.stack(graph_data["cell_tokens"]),
        positions=torch.stack(graph_data["positions"]),
        contours=graph_data["contours"],
        metadata=graph_data["metadata"],
        )
        torch_bytes_io = BytesIO()
        #torch.save(graph, outdir / "cells.pt")
        torch.save(graph, torch_bytes_io)
        zf.writestr("cells.pt", torch_bytes_io.getvalue())

    flag_file = outdir / FLAG_FILE_NAME
    flag_file.touch()

    cell_stats_df = pd.DataFrame(cell_dict_wsi["cells"])
    cell_stats = dict(cell_stats_df.value_counts("type"))
    nuclei_types_inverse = {v: k for k, v in nuclei_types.items()}
    verbose_stats = {nuclei_types_inverse[k]: v for k, v in cell_stats.items()}
    logger.info(f"Finished with cell detection for WSI {wsi.name}")
    logger.info("Stats:")
    logger.info(f"{verbose_stats}")


def post_process_edge_cells(cell_list: List[dict], logger) -> List[int]:
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
    cell_processor = CellPostProcessor(cell_list, logger)
    cleaned_cells_idx = cell_processor.post_process_cells()

    return sorted(cell_record["index"] for cell_record in cleaned_cells_idx)


def convert_geojson(cell_list: list[dict], polygons: bool = False) -> List[dict]:
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


def calculate_instance_map(num_nuclei_classes: int, predictions: OrderedDict, magnification: Literal[20, 40] = 40
    ) -> Tuple[torch.Tensor, List[dict]]:
        """Calculate Instance Map from network predictions (after Softmax output)

        Args:
            predictions (dict): Dictionary with the following required keys:
                * nuclei_binary_map: Binary Nucleus Predictions. Shape: (batch_size, H, W, 2)
                * nuclei_type_map: Type prediction of nuclei. Shape: (batch_size, H, W, 6)
                * hv_map: Horizontal-Vertical nuclei mapping. Shape: (batch_size, H, W, 2)
            magnification (Literal[20, 40], optional): Which magnification the data has. Defaults to 40.

        Returns:
            Tuple[torch.Tensor, List[dict]]:
                * torch.Tensor: Instance map. Each Instance has own integer. Shape: (batch_size, H, W)
                * List of dictionaries. Each List entry is one image. Each dict contains another dict for each detected nucleus.
                    For each nucleus, the following information are returned: "bbox", "centroid", "contour", "type_prob", "type"
        """
        cell_post_processor = DetectionCellPostProcessor(nr_types=num_nuclei_classes, magnification=magnification, gt=False)
        instance_preds = []
        type_preds = []
        max_nuclei_type_predictions = predictions["nuclei_type_map"].argmax(dim=-1, keepdims=True).detach()
        max_nuclei_type_predictions = max_nuclei_type_predictions.cpu() # This is a costly operation because this map is rather large
        max_nuclei_location_predictions = predictions["nuclei_binary_map"].argmax(dim=-1, keepdims=True).detach().cpu()
        
        for i in range(predictions["nuclei_binary_map"].shape[0]):
            # Broke this out to profile better
            pred_map = np.concatenate(
                [
                    max_nuclei_type_predictions[i],
                    max_nuclei_location_predictions[i],
                    predictions["hv_map"][i].detach().cpu(),
                ],
                axis=-1,
            )
            instance_pred = cell_post_processor.post_process_cell_segmentation(pred_map)
            instance_preds.append(instance_pred[0])
            type_preds.append(instance_pred[1])

        return torch.Tensor(np.stack(instance_preds)), type_preds


def get_cell_predictions_with_tokens(num_nuclei_classes: int, 
         predictions: dict, magnification: int = 40
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
        ) = calculate_instance_map(num_nuclei_classes, predictions, magnification=magnification)
        # get the tokens
        tokens = predictions["tokens"]

        return instance_types, tokens


class CellSegmentationInference:
    def __init__(
        self,
        model_path: Union[Path, str],
        gpu: int,
        enforce_mixed_precision: bool = False,
    ) -> None:
        """Cell Segmentation Inference class.

        After setup, a WSI can be processed by calling process_wsi method

        Args:
            model_path (Union[Path, str]): Path to model checkpoint
            gpu (int): CUDA GPU id to use
            enforce_mixed_precision (bool, optional): Using PyTorch autocasting with dtype float16 to speed up inference. Also good for trained amp networks.
                Can be used to enforce amp inference even for networks trained without amp. Otherwise, the network setting is used.
                Defaults to False.
        """
        self.model_path = Path(model_path)
        if gpu >= 0:
            self.device = f"cuda:{gpu}"
        else:
            self.device = "cpu"
        self.__instantiate_logger()
        self.__load_model()
        self.__load_inference_transforms()
        self.__setup_amp(enforce_mixed_precision=enforce_mixed_precision)

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
        CellViT,
        CellViTUnshared,
        CellViT256,
        CellViTUnshared,
        CellViTSAM,
        CellViTSAMUnshared,
    ]:
        """Return the trained model for inference

        Args:
            model_type (str): Name of the model. Must either be one of:
                CellViT, CellViTUnshared, CellViT256, CellViT256Unshared, CellViTSAM, CellViTSAMUnshared

        Returns:
            Union[CellViT, CellViTUnshared, CellViT256, CellViT256Unshared, CellViTSAM, CellViTSAMUnshared]: Model
        """
        implemented_models = [
            "CellViT",
            "CellViTUnshared",
            "CellViT256",
            "CellViT256Unshared",
            "CellViTSAM",
            "CellViTSAMUnshared",
        ]
        if model_type not in implemented_models:
            raise NotImplementedError(
                f"Unknown model type. Please select one of {implemented_models}"
            )
        if model_type in ["CellViT", "CellViTUnshared"]:
            if model_type == "CellViT":
                model_class = CellViT
            elif model_type == "CellViTUnshared":
                model_class = CellViTUnshared
            model = model_class(
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                embed_dim=self.run_conf["model"]["embed_dim"],
                input_channels=self.run_conf["model"].get("input_channels", 3),
                depth=self.run_conf["model"]["depth"],
                num_heads=self.run_conf["model"]["num_heads"],
                extract_layers=self.run_conf["model"]["extract_layers"],
            )

        elif model_type in ["CellViT256", "CellViT256Unshared"]:
            if model_type == "CellViT256":
                model_class = CellViT256
            elif model_type == "CellViTVIT256Unshared":
                model_class = CellViT256Unshared
            model = model_class(
                model256_path=None,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
            )
        elif model_type in ["CellViTSAM", "CellViTSAMUnshared"]:
            if model_type == "CellViTSAM":
                model_class = CellViTSAM
            elif model_type == "CellViTSAMUnshared":
                model_class = CellViTSAMUnshared
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

    def __setup_amp(self, enforce_mixed_precision: bool = False) -> None:
        """Setup automated mixed precision (amp) for inference.

        Args:
            enforce_mixed_precision (bool, optional): Using PyTorch autocasting with dtype float16 to speed up inference. Also good for trained amp networks.
                Can be used to enforce amp inference even for networks trained without amp. Otherwise, the network setting is used.
                Defaults to False.
        """
        if enforce_mixed_precision:
            self.mixed_precision = enforce_mixed_precision
        else:
            self.mixed_precision = self.run_conf["training"].get(
                "mixed_precision", False
            )

    def process_wsi(
        self,
        wsi: WSI,
        subdir_name: str = None,
        patch_size: int = 1024,
        overlap: int = 64,
        batch_size: int = 8,
        geojson: bool = False,
    ) -> None:
        """Process WSI file

        Args:
            wsi (WSI): WSI object
            subdir_name (str, optional): If provided, a subdir with the given name is created in the cell_detection folder.
                Helpful if you need to store different cell detection results next to each other. Defaults to None (no subdir).
            patch_size (int, optional): Patch-Size. Default to 1024.
            overlap (int, optional): Overlap between patches. Defaults to 64.
            batch_size (int, optional): Batch-size for inference. Defaults to 8.
            geosjon (bool, optional): If a geojson export should be performed. Defaults to False.
        """
        self.logger.info(f"Processing WSI: {wsi.name}")

        wsi_inference_dataset = PatchedWSIInference(
            wsi, transform=self.inference_transforms
        )

        num_workers = int(3 / 4 * os.cpu_count())
        if num_workers is None:
            num_workers = 16
        num_workers = int(np.clip(num_workers, 1, 2 * batch_size))

        wsi_inference_dataloader = DataLoader(
            dataset=wsi_inference_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
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

        predicted_batches = []
        with torch.no_grad():
            for batch in tqdm.tqdm(
                wsi_inference_dataloader, total=len(wsi_inference_dataloader)
            ):
                patches = batch[0].to(self.device)

                metadata = batch[1]
                if self.mixed_precision:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        predictions_ = self.model(patches, retrieve_tokens=True)
                else:
                    predictions_ = self.model(patches, retrieve_tokens=True)
                # reshape, apply softmax to segmentation maps
                #predictions = self.model.reshape_model_output(predictions_, self.device)
                predictions = self.model.reshape_model_output(predictions_, 'cpu')
                predicted_batches.append((predictions, metadata))
        
        postprocess_predictions(predicted_batches, self.model.num_nuclei_classes, wsi, self.logger, dataset_config, overlap, patch_size, geojson, outdir)

    def process_wsi_filelist(self, 
                             wsi_filelist,
                            subdir_name: str = None,
                            patch_size: int = 1024,
                            overlap: int = 64,
                            batch_size: int = 8,
                            torch_compile: bool = False,
                            geojson: bool = False,
                            n_postprocess_workers: int = 0,
                            n_dataloader_workers: int = 4,
                            overwrite: bool = False):
        if torch_compile:
                self.logger.info("Model will be compiled using torch.compile. First batch will take a lot more time to compute.")
                self.model = torch.compile(self.model)

        dataset = InferenceWSIDataset(wsi_filelist, transform=self.inference_transforms, overwrite=overwrite, n_workers=n_postprocess_workers)
        self.logger.info(f"Loaded dataset with {dataset.get_n_files()} images")

        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=wsi_patch_collator, num_workers=n_dataloader_workers)
        #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        post_process_arguments = PostprocessArguments(n_images=dataset.get_n_files(),
                                                     num_nuclei_classes=self.model.num_nuclei_classes, 
                                                     dataset_config=self.run_conf['dataset_config'], 
                                                     overlap=overlap, 
                                                     patch_size=patch_size, 
                                                     geojson=geojson, 
                                                     subdir_name=subdir_name,
                                                     n_workers=n_postprocess_workers,
                                                     logger=self.logger)
        if n_postprocess_workers > 0:
            self._process_wsi_filelist_multiprocessing(dataloader, 
                                                       post_process_arguments)
        else:
            self._process_wsi_filelist_singleprocessing(dataloader, 
                                                       post_process_arguments)


        #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    def _process_wsi_filelist_singleprocessing(self,
                                              dataloader,
                                              post_process_arguments):
        wsi_work_map = {}
        
        with torch.no_grad():
            try:
                for batch in tqdm.tqdm(dataloader, desc="Processing patches"):
                    patches, local_idxs, wsi_files, metadatas = batch
                    patches = patches.to(self.device)

                    if self.mixed_precision:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            predictions_ = self.model(patches, retrieve_tokens=True)
                    else:
                        predictions_ = self.model(patches, retrieve_tokens=True)
                    # reshape, apply softmax to segmentation maps
                    #predictions = self.model.reshape_model_output(predictions_, self.device)
                    predictions = self.model.reshape_model_output(predictions_, 'cpu')
                    # We break out the predictions into records (one dict per patch instead of all patches in one dict)
                    prediction_records = [{k: v[i] for k,v in predictions.items()} for i in range(len(local_idxs))]

                    for i, wsi_file in enumerate(wsi_files):
                        wsi_name = wsi_file.name
                        if wsi_name not in wsi_work_map:
                            wsi_work_map[wsi_name] = []
                        (wsi_work_list) = wsi_work_map[wsi_name]
                        work_package = (local_idxs[i], prediction_records[i], metadatas[i])
                        (wsi_work_list).append(work_package)
                        if len((wsi_work_list)) == wsi_file.get_number_patches():
                            local_idxs, predictions_records, metadata = zip(*wsi_work_list)
                            # Merge the prediction records into a single dictionary again.
                            predictions = defaultdict(list)
                            for record in predictions_records:
                                for k,v in record.items():
                                    predictions[k].append(v)
                            predictions_stacked = {k: torch.stack(v).to(torch.float32) for k,v in predictions.items()}
                            postprocess_predictions(predictions_stacked, metadata, wsi_file, post_process_arguments)
                            del wsi_work_map[wsi_name]
                                        
            except KeyboardInterrupt:
                pass

    def _process_wsi_filelist_multiprocessing(self,
                                              dataloader,
                                              post_process_arguments: PostprocessArguments):
        
        pbar_batches = tqdm.trange(len(dataloader), desc="Processing patch-batches")
        pbar_postprocessing = tqdm.trange(post_process_arguments.n_images, desc="Postprocessed images")

        wsi_work_map = {}

        with torch.no_grad():
            with multiprocessing.Pool(post_process_arguments.n_workers) as pool:
                try:
                    results = []
                    
                    for batch in dataloader:
                        patches, local_idxs, wsi_files, metadatas = batch
                        patches = patches.to(self.device)
                        
                        if self.mixed_precision:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                predictions_ = self.model(patches, retrieve_tokens=True)
                        else:
                            predictions_ = self.model(patches, retrieve_tokens=True)
                        # reshape, apply softmax to segmentation maps
                        #predictions = self.model.reshape_model_output(predictions_, self.device)
                        predictions = self.model.reshape_model_output(predictions_, 'cpu')
                        pbar_batches.update()
                        
                        # We break out the predictions into records (one dict per patch instead of all patches in one dict)
                        prediction_records = [{k: v[i] for k,v in predictions.items()} for i in range(len(local_idxs))]
                        
                        for i, wsi_file in enumerate(wsi_files):
                            wsi_name = wsi_file.name
                            if wsi_name not in wsi_work_map:
                                wsi_work_map[wsi_name] = []
                            wsi_work_list = wsi_work_map[wsi_name]
                            work_package = (local_idxs[i], prediction_records[i], metadatas[i])
                            wsi_work_list.append(work_package)
                            if len((wsi_work_list)) == wsi_file.get_number_patches():
                                while len(results) >= post_process_arguments.n_workers:
                                    n_working = len(results)
                                    results = [result for result in results if not result.ready()]
                                    n_done = n_working - len(results)
                                    pbar_postprocessing.update(n_done)
                                    pbar_batches.set_description(f"Processing patch-batches (waiting on postprocessing workers)")
                                    sleep(post_process_arguments.wait_time)
                                result = pool.apply_async(f_post_processing_worker, (wsi_file, wsi_work_list, post_process_arguments))
                                pbar_batches.set_description(f"Processing patch-batches")
                                results.append(result)
                                del wsi_work_map[wsi_name]
                    self.logger.info("Model predictions done, waiting for postprocessing to finish.")
                    pool.close()
                    pool.join()
                except KeyboardInterrupt:
                    pool.terminate()
                    pool.join()

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
        ) = calculate_instance_map(self.model.num_nuclei_classes, predictions, magnification=magnification)
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
        self.logger.info("Initializing Cell-Postprocessor")

        for index, cell_dict in enumerate(cell_list):
            # TODO: Shouldn't it be the other way around? Column = x, Row = Y
            x,y = cell_dict["patch_coordinates"]
            cell_dict["patch_row"] = x
            cell_dict["patch_col"] = y
            cell_dict["patch_coordinates"] = f"{x}_{y}"
            cell_dict["index"] = index

        #self.cell_df = pd.DataFrame(cell_list)
        self.cell_records = cell_list

        #xs, ys = zip(*self.cell_df["patch_coordinates"])
        
        #self.cell_df["patch_row"] = xs
        #self.cell_df["patch_col"] = ys
        #self.cell_df["patch_coordinates"] = [f"{x}_{y}" for x,y in zip(xs, ys)]
        # The call to DataFrame.apply below was exceedingly slow, the list comprehension above is _much_ faster
        #self.cell_df = self.cell_df.apply(convert_coordinates, axis=1)
        self.mid_cells = [cell_record for cell_record in self.cell_records if cell_record["cell_status"] == 0]
        self.margin_cells = [cell_record for cell_record in self.cell_records if cell_record["cell_status"] != 0]
        
    def post_process_cells(self) -> List[Dict]:
        """Main Post-Processing coordinator, entry point

        Returns:
            List[Dict]: List of records (dictionaries) with post-processed and cleaned cells
        """
        self.logger.info("Finding edge-cells for merging")
        cleaned_edge_cells = self._clean_edge_cells()
        self.logger.info("Removal of cells detected multiple times")
        cleaned_edge_cells = self._remove_overlap(cleaned_edge_cells)

        # merge with mid cells
        postprocessed_cells = self.mid_cells + cleaned_edge_cells

        return postprocessed_cells

    def _clean_edge_cells(self) -> List[Dict]:
        """Create a record list that just contains all margin cells (cells inside the margin, not touching the border)
        and border/edge cells (touching border) with no overlapping equivalent (e.g, if patch has no neighbour)

        Returns:
            List[Dict]: Cleaned record list
        """

        margin_cells = [record for record in self.cell_records if record["edge_position"] == 0]
        edge_cells = [record for record in self.cell_records if record["edge_position"] == 1]
        
        existing_patches = list(set(record["patch_coordinates"] for record in self.margin_cells))

        edge_cells_unique = []

        for record in edge_cells:
            edge_information = record["edge_information"]
            edge_patch = edge_information["edge_patches"][0]
            edge_patch = f"{edge_patch[0]}_{edge_patch[1]}"
            if edge_patch not in existing_patches:
                edge_cells_unique.append(record)

        cleaned_edge_cells = margin_cells + edge_cells_unique

        return cleaned_edge_cells

    def _remove_overlap(self, cleaned_edge_cells: List[Dict]) -> List[Dict]:
        """Remove overlapping cells from provided cell record list

        Args:
            cleaned_edge_cells (List[Dict]): List[Dict] that should be cleaned

        Returns:
            List[Dict]: Cleaned cell records
        """
        merged_cells = cleaned_edge_cells

        for iteration in range(20):
            poly_list = []
            for i, cell_info in enumerate(merged_cells):
                poly = Polygon(cell_info["contour"])
                if not poly.is_valid:
                    self.logger.debug("Found invalid polygon - Fixing with buffer 0")
                    multi = poly.buffer(0)
                    if isinstance(multi, MultiPolygon):
                        if len(multi) > 1:
                            poly_idx = np.argmax([p.area for p in multi])
                            poly = multi[poly_idx]
                            poly = Polygon(poly)
                        else:
                            poly = multi[0]
                            poly = Polygon(poly)
                    else:
                        poly = Polygon(multi)
                poly.uid = i
                poly_list.append(poly)

            # use an strtree for fast querying
            tree = strtree.STRtree(poly_list)

            merged_idx = deque()
            iterated_cells = set()
            overlaps = 0

            for query_poly in poly_list:
                if query_poly.uid not in iterated_cells:
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
                                    > 0.01
                                    or query_poly.intersection(inter_poly).area
                                    / inter_poly.area
                                    > 0.01
                                ):
                                    overlaps = overlaps + 1
                                    submergers.append(inter_poly)
                                    iterated_cells.add(inter_poly.uid)
                        # catch block: empty list -> some cells are touching, but not overlapping strongly enough
                        if len(submergers) == 0:
                            merged_idx.append(query_poly.uid)
                        else:  # merging strategy: take the biggest cell, other merging strategies needs to get implemented
                            selected_poly_index = np.argmax(
                                np.array([p.area for p in submergers])
                            )
                            selected_poly_uid = submergers[selected_poly_index].uid
                            merged_idx.append(selected_poly_uid)
                    else:
                        # no intersection, just add
                        merged_idx.append(query_poly.uid)
                    iterated_cells.add(query_poly.uid)

            self.logger.info(
                f"Iteration {iteration}: Found overlap of # cells: {overlaps}"
            )
            if overlaps == 0:
                self.logger.info("Found all overlapping cells")
                break
            elif iteration == 20:
                self.logger.info(
                    f"Not all doubled cells removed, still {overlaps} to remove. For perfomance issues, we stop iterations now. Please raise an issue in git or increase number of iterations."
                )
            
            merged_cells = [cleaned_edge_cells[i] for i in merged_idx]
        return merged_cells


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
            description="Perform CellViT inference for given run-directory with model checkpoints and logs",
        )
        requiredNamed = parser.add_argument_group("required named arguments")
        requiredNamed.add_argument(
            "--model",
            type=str,
            help="Model checkpoint file that is used for inference",
            required=True,
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
            "--enforce_amp",
            action="store_true",
            help="Whether to use mixed precision for inference (enforced). Otherwise network default training settings are used."
            " Default: False",
        )
        parser.add_argument(
            "--torch_compile",
            action="store_true",
            help="Whether to use torch.compile to compile the model before inference. Has an large overhead for single predictions but leads to a significant speedup when predicting on multiple images."
            " Default: False",
        )

        parser.add_argument(
            "--batch_size",
            type=int,
            help="Inference batch-size. Default: 8",
            default=8,
        )

        parser.add_argument(
            "--n_postprocess_workers",
            type=int,
            help="Number of processes to dedicate to post processing. Set to 0 to disable multiprocessing for post processing.  Default: 8",
            default=8,
        )

        parser.add_argument(
            "--n_dataloader_workers",
            type=int,
            help="Number of workers to use for the pytorch patch dataloader. Default: 4",
            default=4,
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

        parser.add_argument(
            "--overwrite",
            action="store_true",
            help=f"If set, include all found pre-processed files even if they include a \"{FLAG_FILE_NAME}\" file.",
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
            description="Process a whole dataset",
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
        model_path=configuration["model"],
        gpu=configuration["gpu"],
        enforce_mixed_precision=configuration["enforce_amp"],
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
            batch_size=configuration["batch_size"],
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
            #if not configuration["overwrite"]:
            #    wsi_filelist = filter_processed_file(wsi_filelist)

        cell_segmentation.process_wsi_filelist(
            wsi_filelist,
            subdir_name=configuration["outdir_subdir"],
            geojson=configuration["geojson"],
            batch_size=configuration["batch_size"],
            torch_compile=configuration["torch_compile"],
            n_postprocess_workers=configuration["n_postprocess_workers"],
            n_dataloader_workers=configuration["n_dataloader_workers"],
            overwrite=configuration["overwrite"]
        )