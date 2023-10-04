# -*- coding: utf-8 -*-
# Process a batch of patches
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from openslide import OpenSlide
from PIL import Image
from shapely.geometry import Polygon
from preprocessing.patch_extraction import logger
from preprocessing.patch_extraction.src.utils.patch_util import (
    DeepZoomGeneratorOS,
    calculate_background_ratio,
    get_intersected_labels,
    macenko_normalization,
    pad_tile,
    patch_to_tile_size,
    standardize_brightness,
)
from utils.tools import module_exists


def process_batch(
    batch: List[Tuple[int, int, float]],
    *,
    wsi_file: Union[Path, str],
    wsi_metadata: dict,
    patch_size: int,
    patch_overlap: int,
    level: int,
    polygons: List[Polygon],
    region_labels: List[str],
    label_map: Dict[str, int],
    min_intersection_ratio: float = 0.0,
    save_only_annotated_patches: bool = False,
    adjust_brightness: bool = False,
    normalize_stains: bool = False,
    normalization_vector_path: Union[str, Path] = None,
    store_masks: bool = False,
    overlapping_labels: bool = False,
    context_scales: List[int] = None,
) -> Tuple[List[np.ndarray], List[dict], List[np.ndarray], dict[int, List[np.ndarray]]]:
    """Calcultes batch results for a list of coordinates

    Patches are extracted according to their coordinate with given patch-settings (size, overlap).
    Patch annotation masks can be stored, as well as context patches with the same shape retrieved.
    Optionally, stains can be nornalized according to macenko normalization.

    Args:
        batch (List[Tuple[int, int, float]]): A batch of patch coordinates (row, col, backgropund ratio)
        wsi_file (Union[Path, str]): Path to the WSI file from which the patches should be extracted from
        wsi_metadata (dict): Dictionary with important WSI metadata
        patch_size (int): The size of the patches in pixel that will be retrieved from the WSI, e.g. 256 for 256px
        patch_overlap (int): The amount pixels that should overlap between two different patches
        level (int): The tile level for sampling.
        polygons (List[Polygon]): Annotations of this WSI as a list of polygons (referenced to highest level of WSI).
            If no annotations, pass an empty list [].
        region_labels (List[str]):  List of labels for the annotations provided as polygons parameter.
            If no annotations, pass an empty list [].
        label_map (Dict[str, int]): Dictionary mapping the label names to an integer. Please ensure that background label has integer 0!
        min_intersection_ratio(float, optional): Minimum ratio of intersection between annotation class and patch to be considered as class instance. Defaults to 0.0.
        save_only_annotated_patches (bool, optional): If true only patches containing annotations will be stored. Defaults to False.
        adjust_brightness (bool, optional): Normalize brightness in a batch by clipping to 90%. Defaults to False.
        normalize_stains (bool, optional): Uses Macenko normalization on patches. Defaults to False.
        normalization_vector_path (Union[str, Path], optional): The path to a JSON file where the normalization vectors are stored. Defaults to None.
        store_masks (bool, optional): Set to store masks per patch. Defaults to False.
        overlapping_labels (bool, optional): Per default, labels (annotations) are mutually exclusive.
            If labels overlap, they are overwritten according to the label_map.json ordering (highest number = highest priority).
            True means that the mask array is 3D with shape (patch_size, patch_size, len(label_map)), otherwise just (patch_size, patch_size).
            Defaults to False.
        context_scales (List[int], optional): Define context scales for context patches. Context patches are centered around a central patch.
            The context-patch size is equal to the patch-size, but downsampling is different. Defaults to None.

    Returns:
        Tuple[List[np.ndarray], List[dict], List[np.ndarray], dict[int, List[np.ndarray]]]:

        - List[np.ndarray]: List with patches as numpy arrays with shape (patch_size, patch_size, 3)
        - List[dict]: List with metadata dictionary for each patch
        - List[np.ndarray]: List with patch masks if store_masks is True. Shape is (256, 256) for non-overlapping labels
            and (256, 256, num_classes) for overlapping labels. If masks should not be stored, returns an empty list
        - dict[int, List[np.ndarray]]: Each key is a downsampling value for the context patch
            and the entries are numpy array for context patches with shape [patch_size, patch_size, 3].
            If no context patches, returns an empty dict
    """
    logger.debug(f"Started process {multiprocessing.current_process().name}")

    # Where the results of this batch will be stored
    patches, metadata, patch_masks = [], [], []

    # context patch
    context_tiles = {}
    if context_scales is not None:
        context_patches = {scale: [] for scale in context_scales}
    else:
        context_patches = {}

    # reopen slide
    if module_exists("cucim", error="ignore"):
        from cucim import CuImage

        from preprocessing.patch_extraction.src.cucim_deepzoom import (
            DeepZoomGeneratorCucim,
        )

        generator_module = DeepZoomGeneratorCucim
        image_loader = CuImage
    else:
        generator_module = DeepZoomGeneratorOS
        image_loader = OpenSlide

    slide = OpenSlide(str(wsi_file))
    slide_cu = image_loader(str(wsi_file))
    tile_size = patch_to_tile_size(patch_size, patch_overlap)

    tiles = generator_module(
        osr=slide,
        cucim_slide=slide_cu,
        tile_size=tile_size,
        overlap=patch_overlap,
        limit_bounds=True,
    )

    if context_scales is not None:
        for scale in context_scales:
            overlap_context = int((scale - 1) * patch_size / 2) + patch_overlap
            context_tiles[scale] = generator_module(
                osr=slide,
                cucim_slide=slide_cu,
                tile_size=tile_size,  # tile_size,
                overlap=overlap_context,  # (1-scale) * tile_size / 2,
                limit_bounds=True,
            )
    # patches_count = 0

    for row, col, _ in batch:
        # OpenSlide: Address of the tile within the level as a (column, row) tuple
        new_tile = np.array(tiles.get_tile(level, (col, row)), dtype=np.uint8)

        # calculate background ratio for every patch
        background_ratio = calculate_background_ratio(new_tile, patch_size)

        # patch_label
        if background_ratio > 1 - min_intersection_ratio:
            intersected_labels = []  # Zero means background
            ratio = []
            patch_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
        else:
            intersected_labels, ratio, patch_mask = get_intersected_labels(
                tile_size=tile_size,
                patch_overlap=patch_overlap,
                col=col,
                row=row,
                polygons=polygons,
                label_map=label_map,
                min_intersection_ratio=min_intersection_ratio,
                region_labels=region_labels,
                overlapping_labels=overlapping_labels,
                store_masks=store_masks,
            )
        if len(intersected_labels) == 0 and save_only_annotated_patches:
            continue
        if store_masks:
            patch_masks.append(patch_mask)

        patches.append(pad_tile(new_tile, patch_size, col, row))
        patch_metadata = {
            "row": row,
            "col": col,
            "background_ratio": float(background_ratio),
            "intersected_labels": intersected_labels,
            "label_ratio": ratio,
            "wsi_metadata": wsi_metadata,
        }

        if context_scales is not None:
            patch_metadata["context_scales"] = []
            for scale in context_scales:
                context_patch = np.array(
                    context_tiles[scale].get_tile(level, (col, row)),
                    dtype=np.uint8,  # TODO change back to level
                )
                context_patch = pad_tile(context_patch, patch_size * scale, col, row)
                context_patch = np.array(
                    Image.fromarray(context_patch).resize((patch_size, patch_size)),
                    dtype=np.uint8,
                )
                context_patches[scale].append(context_patch)
                patch_metadata["context_scales"].append(scale)

        metadata.append(patch_metadata)

    if len(patches) > 0:
        if adjust_brightness:
            patches = standardize_brightness(patches)
            for scale, scale_patch in context_patches.items():
                context_patches[scale] = standardize_brightness(scale_patch)
        if normalize_stains:
            patches, _, _ = macenko_normalization(
                patches, normalization_vector_path=normalization_vector_path
            )
            for scale, scale_patch in context_patches.items():
                context_patches[scale], _, _ = macenko_normalization(
                    scale_patch, normalization_vector_path=normalization_vector_path
                )

    logger.debug(
        f"Process {multiprocessing.current_process().name} finished, found "
        f"{len(patches)} patches."
    )
    return patches, metadata, patch_masks, context_patches
