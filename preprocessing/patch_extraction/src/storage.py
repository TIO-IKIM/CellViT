# -*- coding: utf-8 -*-
# Storage class to store a processed WSI and its batches
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import json
import multiprocessing
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import yaml
from PIL import Image

from preprocessing.patch_extraction import logger


class Storage:
    """Storage class to store all WSI related files

    Generates the following folder structure for storage:

        * Output-Path/WSI-Name
            * metadata.yaml: Metadata of the WSI
            * annotation_masks: preview images of annotations
            * patches: store extracted patches with each path "wsi_name_row_col.png"
            * metadata: store metadata for each path "wsi_name_row_col.yaml"
            * thumbnails: WSI thumbnails
            * tissue masks: Masks of tissue detection
            * Optional: context: context folder with subfolder for each context scale
            * Optional: masks: Masks for each patch as .npy files (numpy arrays)

    Args:
        wsi_name (str): Name of the WSI, as string. Just the name without suffix and no path!
        output_path (Union[Path, str]): Path to the folder where the resulting dataset should be stored.
        metadata (dict): Metadata of the WSI. Is stored in parent directory
        mask_images (dict[str, Image]): Masks generated during tissue detection stored in dict
            with keys equals the mask name and values equals the PIL image
        mask_images_annotations (dict[str, Image]): Annotation masks for provided annotations for the complete WSI.
            Masks are equal to the tissue masks sizes. Keys are the mask names and values are the PIL images.
        thumbnails (dict[str, Image]): Dictionary with thumbnails and corresponding thumbnail names.
            Names are keys, PIL Images are values
        store_masks (bool, optional): Set to store masks per patch. Defaults to False.
        save_context (bool, optional): If context patches are provided. Defaults to False.
        context_scales (List[int], optional): List with context scales. Defaults to None.
    """

    def __init__(
        self,
        wsi_name: str,
        output_path: Union[Path, str],
        metadata: dict,
        mask_images: dict,
        mask_images_annotations: dict,
        thumbnails: dict,
        store_masks: bool = False,
        save_context: bool = False,
        context_scales: List[int] = None,
    ) -> None:
        self.wsi_name = wsi_name
        self.output_path = Path(output_path)
        self.save_context = save_context

        self.wsi_path = self.output_path / self.wsi_name
        self.wsi_path.mkdir(parents=True, exist_ok=True)
        self.patches_path = self.wsi_path / "patches"
        self.patches_path.mkdir(parents=True, exist_ok=True)
        self.patch_metadata_path = self.wsi_path / "metadata"
        self.patch_metadata_path.mkdir(parents=True, exist_ok=True)
        self.thumbnail_path = self.wsi_path / "thumbnails"
        self.thumbnail_path.mkdir(parents=True, exist_ok=True)
        self.tissue_mask_path = self.wsi_path / "tissue_masks"
        self.tissue_mask_path.mkdir(parents=True, exist_ok=True)
        self.annotation_mask_path = self.wsi_path / "annotation_masks"
        self.annotation_mask_path.mkdir(parents=True, exist_ok=True)

        if self.save_context:
            assert (
                context_scales is not None
            ), "Please provide at least one context scale"
            self.context_path = self.wsi_path / "context"
            self.context_path.mkdir(parents=True, exist_ok=True)
            for scale in context_scales:
                (self.context_path / str(scale)).mkdir(parents=True, exist_ok=True)

        self.store_masks = store_masks
        if self.store_masks:
            self.masks_path = self.wsi_path / "masks"
            self.masks_path.mkdir(parents=True, exist_ok=True)

        self.metadata = metadata

        self.save_meta_data()
        self.save_masks(mask_images)
        self.save_annotation_mask(mask_images_annotations)
        self.save_thumbnails(thumbnails)

        # patch_stats
        patch_distribution = self.metadata["label_map"]
        self.patch_distribution = {v: 0 for k, v in patch_distribution.items()}
        self.patch_list = []

    def save_meta_data(self) -> None:
        """
        Store arbitrary meta data in a yaml file on wsi output folder
        """
        # ensure folder exists
        with open(self.wsi_path / "metadata.yaml", "w") as outfile:
            yaml.dump(
                self.metadata,
                outfile,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )

    def save_masks(self, mask_images: dict):
        """Save tissue masks

        Args:
            mask_images (dict[str, Image]): Masks generated during tissue detection stored in dict
                with keys equals the mask name and values equals the PIL image
        """
        assert "mask" in mask_images.keys()

        for mask_name, mask in mask_images.items():
            mask_path = self.tissue_mask_path / f"{mask_name}.png"
            mask.save(str(mask_path))
        mask_images["mask"].save(self.wsi_path / "mask.png")

    def save_annotation_mask(self, mask_images_annotations: dict):
        """Save annotation masks

        Args:
            mask_images_annotations (dict[str, Image]): Annotation masks for provided annotations for the complete WSI.
                Masks are equal to the tissue masks sizes. Keys are the mask names and values are the PIL images.
        """
        for mask_name, mask in mask_images_annotations.items():
            mask_path = self.annotation_mask_path / f"{mask_name}.png"
            mask_path_eps = self.annotation_mask_path / f"{mask_name}.eps"
            mask.save(str(mask_path))
            mask.save(str(mask_path_eps))

    def save_thumbnails(self, thumbnails: dict):
        """Save thumbnails of WSI

        Args:
            thumbnails (dict[str, Image]): Dictionary with thumbnails and corresponding thumbnail names.
                Names are keys, PIL Images are values
        """
        assert "thumbnail" in thumbnails.keys()

        for sample_factor, thumbnail in thumbnails.items():
            thumbnail_path = self.thumbnail_path / f"thumbnail_{sample_factor}.png"
            thumbnail.save(str(thumbnail_path))
        thumbnails["thumbnail"].save(self.wsi_path / "thumbnail.png")

    def save(
        self,
        patch_result: Tuple[
            List[np.ndarray], List[dict], List[np.ndarray], List[Dict[int, np.ndarray]]
        ],
    ) -> None:
        """Saving a batch of patches with metadata and optional masks and context patches.

        Args:
            patch_result (Tuple[List[np.ndarray], List[dict], List[np.ndarray], List[Dict[int, np.ndarray]]]):

                Tuple containing:
                * List[np.ndarray]: List with patches as numpy arrays with shape (patch_size, patch_size, 3)
                * List[dict]: List with metadata dictionary for each patch
                * List[np.ndarray]: List with patch masks if store_masks is True. Shape is (256, 256) for non-overlapping labels
                    and (256, 256, num_classes) for overlapping labels.
                * List[Dict[int, np.ndarray]: Each key is a downsampling value for the context patch
                    and the entries are numpy array for context patches with shape [patch_size, patch_size, 3].
        """
        patches, patch_metadata, patch_masks, context = patch_result
        logger.debug(
            f"Process {multiprocessing.current_process().name} is saving {len(patches)}."
        )

        self.save_to_disk(
            patches,
            patch_metadata,
            patch_masks if len(patch_masks) > 0 else None,
            context,
        )

    def save_to_disk(
        self,
        patches: List[np.ndarray],
        patches_metadata: List[dict],
        patch_masks: List[np.ndarray] = None,
        contexts: List[Dict[int, np.ndarray]] = None,
    ) -> None:
        """Saving the unpacked batch

        Args:
            patches (List[np.ndarray]): List with patches as numpy arrays with shape (patch_size, patch_size, 3)
            patches_metadata (List[dict]): List with metadata dictionary for each patch
            patch_masks (List[np.ndarray], optional): List with patch masks if store_masks is True. Shape is (256, 256) for non-overlapping labels
                    and (256, 256, num_classes) for overlapping labels. Defaults to None.
            contexts (List[Dict[int, np.ndarray]], optional): Each key is a downsampling value for the context patch
                    and the entries are numpy array for context patches with shape [patch_size, patch_size, 3]. Defaults to None.
        """
        # zip is not an option, since the contexts structure is different
        for i, patch in enumerate(patches):
            # Construct the new PNG filename
            patch_metadata = patches_metadata[i]
            row = patch_metadata["row"]
            col = patch_metadata["col"]
            patch_fname = f"{self.wsi_name}_{row}_{col}.png"
            patch_yaml_name = f"{self.wsi_name}_{row}_{col}.yaml"

            # Save the patch
            Image.fromarray(patch).save(self.patches_path / patch_fname)

            # Save the metadata
            with open(self.patch_metadata_path / patch_yaml_name, "w") as yaml_file:
                yaml.dump(
                    patch_metadata, yaml_file, default_flow_style=False, sort_keys=False
                )

            # Save the Mask
            if patch_masks is not None and self.store_masks:
                np.save(
                    str(self.masks_path / f"{Path(patch_fname).stem}_mask.npy"),
                    patch_masks[i].squeeze(),
                )
                patch_metadata["mask"] = f"./masks/{Path(patch_fname).stem}_mask.npy"

            # increase patch_distribution count
            for patch_label in patch_metadata["intersected_labels"]:
                self.patch_distribution[patch_label] += 1

            patch_metadata.pop("wsi_metadata")
            patch_metadata["metadata_path"] = f"./metadata/{patch_yaml_name}"

            # Save context patches if non empty
            if self.save_context:
                patch_metadata["context_scales"] = {}
                for scale, context_images in contexts.items():
                    context_name = f"{Path(patch_fname).stem}_context_{scale}.png"
                    Image.fromarray(context_images[i]).save(
                        self.context_path / str(scale) / context_name
                    )
                    patch_metadata["context_scales"][
                        scale
                    ] = f"./context/{context_name}"

            # append all metadata to WSI data
            self.patch_list.append({patch_fname: patch_metadata})

    def clean_up(self):
        """Clean-Up function, called after WSI has been processed. Appends WSI to `processed.json` file
        and generated a metadata file in root folder called `patch_metadata.json` with merged metadata for all patches
        in one file.
        """
        try:
            with open(str(self.output_path / "processed.json"), "r") as processed_list:
                try:
                    processed_files = json.load(processed_list)
                    processed_files["processed_files"].append(self.wsi_name)
                except JSONDecodeError:
                    processed_files = {"processed_files": [self.wsi_name]}
        except FileNotFoundError:
            processed_files = {"processed_files": [self.wsi_name]}
        with open(str(self.output_path / "processed.json"), "w") as processed_list:
            json.dump(processed_files, processed_list, indent=2)

        # count patches per class
        self.metadata["patch_distribution"] = self.patch_distribution
        self.save_meta_data()

        # save patch metadata file
        with open(self.wsi_path / "patch_metadata.json", "w") as outfile:
            json.dump(self.patch_list, outfile, indent=2)
