# -*- coding: utf-8 -*-
# WSI Model
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


import json
from pathlib import Path
from typing import Union, List, Callable, Tuple

from dataclasses import dataclass, field
import numpy as np
import yaml
import logging
import torch
from PIL import Image


@dataclass
class WSI:
    """WSI object

    Args:
        name (str): WSI name
        patient (str): Patient name
        slide_path (Union[str, Path]): Full path to the WSI file.
        patched_slide_path (Union[str, Path], optional): Full path to preprocessed WSI files (patches). Defaults to None.
        embedding_name (Union[str, Path], optional): Defaults to None.
        label (Union[str, int, float, np.ndarray], optional): Label of the WSI. Defaults to None.
        logger (logging.logger, optional): Logger module for logging information. Defaults to None.
    """

    name: str
    patient: str
    slide_path: Union[str, Path]
    patched_slide_path: Union[str, Path] = None
    embedding_name: Union[str, Path] = None
    label: Union[str, int, float, np.ndarray] = None
    logger: logging.Logger = None

    # unset attributes used in this class
    metadata: dict = field(init=False, repr=False)
    all_patch_metadata: List[dict] = field(init=False, repr=False)
    patches_list: List = field(init=False, repr=False)
    patch_transform: Callable = field(init=False, repr=False)

    # name without ending (e.g. slide1 instead of slide1.svs)
    def __post_init__(self):
        """Post-Processing object"""
        super().__init__()
        # define paramaters that are used, but not defined at startup

        # convert string to path
        self.slide_path = Path(self.slide_path).resolve()
        if self.patched_slide_path is not None:
            self.patched_slide_path = Path(self.patched_slide_path).resolve()
            # load metadata
            self._get_metadata()
            self._get_wsi_patch_metadata()
            self.patch_transform = None  # hardcode to None (should not be a parameter, but should be defined)

        if self.logger is not None:
            self.logger.debug(self.__repr__())

    def _get_metadata(self) -> None:
        """Load metadata yaml file"""
        self.metadata_path = self.patched_slide_path / "metadata.yaml"
        with open(self.metadata_path.resolve(), "r") as metadata_yaml:
            try:
                self.metadata = yaml.safe_load(metadata_yaml)
            except yaml.YAMLError as exc:
                print(exc)
        self.metadata["label_map_inverse"] = {
            v: k for k, v in self.metadata["label_map"].items()
        }

    def _get_wsi_patch_metadata(self) -> None:
        """Load patch_metadata json file and convert to dict and lists"""
        with open(self.patched_slide_path / "patch_metadata.json", "r") as json_file:
            metadata = json.load(json_file)
            self.patches_list = [str(list(elem.keys())[0]) for elem in metadata]
            self.all_patch_metadata = {
                str(list(elem.keys())[0]): elem[str(list(elem.keys())[0])]
                for elem in metadata
            }

    def load_patch_metadata(self, patch_name: str) -> dict:
        """Return the metadata of a patch with given name (including patch suffix, e.g., wsi_1_1.png)

        This function assumes that metadata path is a subpath of the patches dataset path

        Args:
            patch_name (str): Name of patch

        Returns:
            dict: metadata
        """
        patch_metadata_path = self.all_patch_metadata[patch_name]["metadata_path"]
        patch_metadata_path = self.patched_slide_path / patch_metadata_path

        # open
        with open(patch_metadata_path, "r") as metadata_yaml:
            patch_metadata = yaml.safe_load(metadata_yaml)
        patch_metadata["name"] = patch_name

        return patch_metadata

    def set_patch_transform(self, transform: Callable) -> None:
        """Set the transformation function to process a patch

        Args:
            transform (Callable): Transformation function
        """
        self.patch_transform = transform

    # patch processing
    def process_patch_image(
        self, patch_name: str, transform: Callable = None
    ) -> Tuple[torch.Tensor, dict]:
        """Process one patch: Load from disk, apply transformation if needed. ToTensor is applied automatically

        Args:
            patch_name (Path): Name of patch to load, including patch suffix, e.g., wsi_1_1.png
            transform (Callable, optional): Optional Patch-Transformation
        Returns:
            Tuple[torch.Tensor, dict]:

            * torch.Tensor: patch as torch.tensor (:,:,3)
            * dict: patch metadata as dictionary
        """
        patch = Image.open(self.patched_slide_path / "patches" / patch_name)
        if transform:
            patch = transform(patch)

        metadata = self.load_patch_metadata(patch_name)
        return patch, metadata

    def get_number_patches(self) -> int:
        """Return the number of patches for this WSI

        Returns:
            int: number of patches
        """
        return int(len(self.patches_list))

    def get_patches(
        self, transform: Callable = None
    ) -> Tuple[torch.Tensor, list, list]:
        """Get all patches for one image

        Args:
            transform (Callable, optional): Optional Patch-Transformation

        Returns:
            Tuple[torch.Tensor, list]:

            * patched image: Shape of torch.Tensor(num_patches, 3, :, :)
            * coordinates as list metadata_dictionary

        """
        if self.logger is not None:
            self.logger.warning(f"Loading {self.get_number_patches()} patches!")
        patches = []
        metadata = []
        for patch in self.patches_list:
            transformed_patch, meta = self.process_patch_image(patch, transform)
            patches.append(transformed_patch)
            metadata.append(meta)
        patches = torch.stack(patches)

        return patches, metadata

    def load_embedding(self) -> torch.Tensor:
        """Load embedding from subfolder patched_slide_path/embedding/

        Raises:
            FileNotFoundError: If embedding is not given

        Returns:
            torch.Tensor: WSI embedding
        """
        embedding_path = (
            self.patched_slide_path / "embeddings" / f"{self.embedding_name}.pt"
        )
        if embedding_path.is_file():
            embedding = torch.load(embedding_path)
            return embedding
        else:
            raise FileNotFoundError(
                f"Embeddings for WSI {self.slide_path} cannot be found in path {embedding_path}"
            )
