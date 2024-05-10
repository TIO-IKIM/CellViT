# -*- coding: utf-8 -*-
# CLI
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


import argparse
import json
import logging
from copy import copy
from pathlib import Path
from typing import Any, List, Optional, Tuple

import yaml
from pydantic import BaseModel, validator

from base_ml.base_cli import ABCParser
from configs.python.config import ANNOTATION_EXT, LOGGING_EXT, WSI_EXT
from utils.logger import Logger


class PreProcessingYamlConfig(BaseModel):
    """For explanation, see PreProcessingParser"""

    # Set all to optional to allow selecting from yaml and argparse cli

    # dataset paths
    wsi_paths: Optional[str]
    output_path: Optional[str]
    wsi_extension: Optional[str]
    wsi_filelist: Optional[str]

    # basic setups
    patch_size: Optional[int]
    patch_overlap: Optional[float]
    target_mpp: Optional[float]
    target_mag: Optional[float]
    downsample: Optional[int]
    level: Optional[int]
    context_scales: Optional[List[int]]
    check_resolution: Optional[float]
    processes: Optional[int]
    overwrite: Optional[bool]

    # annotation specific settings
    annotation_paths: Optional[str]
    annotation_extension: Optional[str]
    incomplete_annotations: Optional[bool]
    label_map_file: Optional[str]
    save_only_annotated_patches: Optional[bool]
    exclude_classes: Optional[List[str]]
    store_masks: Optional[bool]
    generate_thumbnails: Optional[bool]
    overlapping_labels: Optional[bool]

    # macenko stain normalization
    normalize_stains: Optional[bool]
    normalization_vector_json: Optional[str]
    adjust_brightness: Optional[bool]

    # finding patches
    min_intersection_ratio: Optional[float]
    tissue_annotation: Optional[str]
    tissue_annotation_intersection_ratio: Optional[float] 
    masked_otsu: Optional[bool]
    otsu_annotation: Optional[str]
    filter_patches: Optional[bool]
    apply_prefilter: Optional[bool]

    # other
    log_path: Optional[str]
    log_level: Optional[str]
    hardware_selection: Optional[str]
    wsi_properties: Optional[dict]


class PreProcessingConfig(BaseModel):
    """Storing the preprocessing configuration

    All string that describe paths are converted to pathlib.Path objects.

    Args:
        wsi_paths (str): Path to the folder where all WSI are stored or path to a single WSI-file.
        output_path (str): Path to the folder where the resulting dataset should be stored.
        wsi_extension (str, optional): The extension of the WSI-files. Defaults to "svs".
        wsi_filelist (str, optional): Path to a csv-filelist with WSI files (separator: `,`), if provided just these files are used. Must include full paths to WSIs, including suffixes.
            Can be used as an replacement for the wsi_paths option. If both are provided, yields an error. Defaults to None.
        patch_size (int, optional): The size of the patches in pixel that will be retrieved from the WSI, e.g. 256 for 256px. Defaults to 256.
        patch_overlap (float, optional): The percentage amount pixels that should overlap between two different patches.
            Please Provide as integer between 0 and 100, indicating overlap in percentage.
            Defaults to 0.
        target_mpp (float, optional): If this parameter is provided, the output level of the WSI
            corresponds to the level that is at the target microns per pixel of the WSI.
            Alternative to target_mag, downsaple and level. Highest priority, overwrites all other setups for magnifcation, downsample, or level.
        target_mag (float, optional): If this parameter is provided, the output level of the WSI
            corresponds to the level that is at the target magnification of the WSI.
            Alternative to target_mpp, downsaple and level. High priority, just target_mpp has a higher priority, overwrites downsample and level if provided. Defaults to None.
        downsample (int, optional): Each WSI level is downsampled by a factor of 2, downsample
            expresses which kind of downsampling should be used with
            respect to the highest possible resolution. Defaults to 0.
        level (int, optional): The tile level for sampling, alternative to downsample. Defaults to None.
        context_scales ([List[int], optional): Define context scales for context patches. Context patches are centered around a central patch.
            The context-patch size is equal to the patch-size, but downsampling is different.
            Defaults to None.
        check_resolution (float, optional): If a float value is supplies, the program checks whether
            the resolution of all images corresponds to the given value.
            Defaults to None.
        processes (int, optional): The number of processes to use. Defaults to 24
        overwrite (bool, optional): Overwrite the patches that have already been created in
            case they already exist. Removes dataset. Handle with care! If false, skips already processed files from "processed.json". Defaults to False.
        annotation_paths (str, optional): Path to the subfolder where the annotations are
            stored or path to a file. Defaults to None.
        annotation_extension (str, optional): The extension types used for the annotation files. Defaults to None.
        incomplete_annotations (bool, optional): Set to allow WSI without annotation file. Defaults to False.
        label_map_file (str, optional): The path to a json file that contains the mapping between
            he annotation labels and some integers; an example can be found in examples. Defaults to None.
        label_map (dict, optional): Field to store the label mapping defined in the label map file. Gets overwriten by creation - to a dictionary with str: int. Do not pass values.
            Defaults to None.
        save_only_annotated_patches (bool, optional): If true only patches containing annotations will be stored. Defaults to False.
        exclude_classes (List[str], optional): Can be used to exclude annotation classes. Defaults to [].
        store_masks (bool, optional): Set to store masks per patch. Defaults to false.
        overlapping_labels (bool, optional): Per default, labels (annotations) are mutually exclusive.
            If labels overlap, they are overwritten according to the label_map.json ordering (highest number = highest priority).
            True means that the mask array is 3D with shape [patch_size, patch_size, len(label_map)], otherwise just [patch_size, patch_size].
            Defaults to False.
        normalize_stains (bool, optional): Uses Macenko normalization on a portion of the whole slide images. Defaults to False.
        normalization_vector_json (str, optional): The path to a JSON file where the normalization vectors are stored. Defaults to None.
        adjust_brightness (bool, optional): Normalize brightness in a batch by clipping to 90 percent. Not recommended, but kept for legacy reasons. Defaults to False.
        min_intersection_ratio (float, optional): The minimum intersection between the tissue mask and the patch.
            Must be between 0 and 1. 0 means that all patches are extracted. Defaults to 0.01.
        tissue_annotation (str, optional): Can be used to name a polygon annotation to determine the tissue area
            If a tissue annotation is provided, no Otsu-thresholding is performed. Defaults to None.
        tissue_annotation_intersection_ratio (float, optional): Intersection ratio with tissue annotation. Helpful, if ROI annotation is passed, which should not interfere with background ratio.
            If not provided, the default min_intersection_ratio with the background is used. Defaults to None.
        masked_otsu (bool, optional): Use annotation to mask the thumbnail before otsu-thresholding is used. Defaults to False.
        otsu_annotation (bool, optional): Can be used to name a polygon annotation to determine the area
            for masked otsu thresholding. Seperate multiple labels with ' ' (whitespace). Defaults to None.
        filter_patches (bool, optional): Post-extraction patch filtering to sort out artefacts, marker and other non-tissue patches with a DL model. Time consuming.
            Defaults to False.
        apply_prefilter (bool, optional): Pre-extraction mask filtering to remove marker from mask before applying otsu. Defaults to False.
        log_path (str, optional): Path where log files should be stored. Otherwise, log files are stored in the output folder. Defaults to None.
        log_level (str, optional): Set the logging level. Defaults to "info".
        hardware_selection (str, optional): Select hardware device (just if available, otherwise always cucim). Defaults to "cucim".
        wsi_properties (dict, optional): Dictionary with manual WSI metadata, but just applies if metadata cannot be derived from OpenSlide (e.g., for .tiff files). Supported keys are slide_mpp and magnification

    Raises:
        ValueError: Patch-size must be positive
        ValueError: At least 1 process is needed
        ValueError: Batch must contain at least 1 patch, recommended are 100-500.
        ValueError: Background ratio must be between 0 and 1.
        ValueError: Matching annotation type
        ValueError: Matching logging level
        ValueError: Matching WSI extension

    """

    # dataset paths
    output_path: str
    wsi_paths: Optional[str]
    wsi_filelist: Optional[str]
    wsi_extension: Optional[str] = "svs"

    # basic setups
    patch_size: Optional[int] = 256
    patch_overlap: Optional[float] = 0
    downsample: Optional[int] = 1
    target_mpp: Optional[float]
    target_mag: Optional[float]
    level: Optional[int]
    context_scales: Optional[List[int]]
    check_resolution: Optional[float]
    processes: Optional[int] = 24
    overwrite: Optional[bool] = False

    # annotation specific settings
    annotation_paths: Optional[str]
    annotation_extension: Optional[str]
    incomplete_annotations: Optional[bool] = False
    label_map_file: Optional[str]
    label_map: Optional[dict]
    save_only_annotated_patches: Optional[bool] = False
    exclude_classes: Optional[List[str]] = []
    store_masks: Optional[bool] = False
    overlapping_labels: Optional[bool] = False

    # macenko stain normalization
    normalize_stains: Optional[bool] = False
    normalization_vector_json: Optional[str]
    adjust_brightness: Optional[bool] = False

    # finding patches
    min_intersection_ratio: Optional[float] = 0.01
    tissue_annotation: Optional[str]
    tissue_annotation_intersection_ratio: Optional[float]
    masked_otsu: Optional[bool] = False
    otsu_annotation: Optional[str]
    filter_patches: Optional[bool] = False
    apply_prefilter: Optional[bool] = False

    # other
    log_path: Optional[str]
    log_level: Optional[str] = "info"
    hardware_selection: Optional[str] = "cucim"
    wsi_properties: Optional[dict]

    def __init__(__pydantic_self__, **data: Any) -> None:
        super().__init__(**data)
        __pydantic_self__.__post_init_post_parse__()

    # validators
    @validator("patch_size")
    def patch_size_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Patch-Size in pixels must be positive")
        return v

    @validator("patch_overlap")
    def overlap_percentage(cls, v):
        if v < 0 and v >= 100:
            raise ValueError(
                "Patch-Overlap in percentage must be between 0 and 100 (100 not included)"
            )
        return v

    @validator("processes")
    def processes_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("At least 1 process is needed")
        return v

    @validator("min_intersection_ratio")
    def min_intersection_ratio_range_check(cls, v):
        if v < 0 and v > 1:
            raise ValueError("Background ratio must be between 0 and 1")
        return v

    @validator("annotation_extension")
    def annotation_extension_selector(cls, v):
        if v not in ANNOTATION_EXT:
            raise ValueError(
                f"The extension types used for the annotation files is wrong, the options are: {ANNOTATION_EXT}"
            )
        return v

    @validator("log_level")
    def log_level_check(cls, v):
        if v not in LOGGING_EXT:
            raise ValueError(f"Wrong logging level. Options are {LOGGING_EXT}")
        return v.upper()

    @validator("wsi_extension")
    def wsi_extension_selector(cls, v):
        if v not in WSI_EXT:
            raise ValueError(
                f"The extension types used for the WSI files is wrong, the options are: {WSI_EXT}"
            )
        return v

    def __post_init_post_parse__(self):
        """Post processing after parsing.

        Converting paths to `Pathlib` object, convert strings and stored dict.

        Raises:
            RuntimeError: Please provide either wsi_paths or wsi_filelist argument
            ValueError: A label map file must be used if annotations are passed
            ValueError: Checking for right label_map format (.json) file.
        """
        if (self.wsi_paths is None and self.wsi_filelist is None) or (
            self.wsi_paths is not None and self.wsi_filelist is not None
        ):
            raise RuntimeError(
                "Please provide either wsi_paths or wsi_filelist argument!"
            )

        self.output_path = Path(self.output_path).resolve()

        if self.wsi_paths is not None:
            self.wsi_paths = Path(self.wsi_paths).resolve()
        if self.wsi_filelist is not None:
            self.wsi_filelist = Path(self.wsi_filelist).resolve()

        if self.annotation_paths is not None:
            self.annotation_paths = Path(self.annotation_paths).resolve()
            if self.label_map_file is None:
                raise ValueError(
                    "Please provide label_map_file if annoations should be used"
                )
            else:
                self.label_map_file = Path(self.label_map_file).resolve()
                if self.label_map_file.suffix != ".json":
                    raise ValueError("Please provide label_map_file as json file")
                with open(str(self.label_map_file)) as json_file:
                    label_map = json.load(json_file)
                    self.label_map = {k.lower(): v for k, v in label_map.items()}
        if self.label_map_file is None or self.label_map is None:
            self.label_map = {"background": 0}
        if self.log_path is None:
            self.log_path = self.output_path
        if self.otsu_annotation is not None:
            self.otsu_annotation = self.otsu_annotation.lower()
        if self.tissue_annotation is not None:
            self.tissue_annotation = self.tissue_annotation.lower()
        if len(self.exclude_classes) > 0:
            self.exclude_classes = [f.lower() for f in self.exclude_classes]
        if self.tissue_annotation_intersection_ratio is None:
            self.tissue_annotation_intersection_ratio = self.min_intersection_ratio
        else:
            if self.tissue_annotation_intersection_ratio < 0 and self.tissue_annotation_intersection_ratio > 1:
                raise RuntimeError("Tissue_annotation_intersection_ratio must be between 0 and 1")

class PreProcessingParser(ABCParser):
    """Configuration Parser for Preprocessing"""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        # dataset paths
        parser.add_argument(
            "--wsi_paths",
            type=str,
            help="Path to the folder where all WSI are stored or path to a single WSI-file.",
        )
        parser.add_argument(
            "--wsi_filelist",
            type=str,
            help="Path to a csv-filelist with WSI files (separator: `,`), if provided just these files are used."
            "Must include full paths to WSIs, including suffixes."
            "Can be used as an replacement for the wsi_paths option."
            "If both are provided, yields an error.",
        )
        parser.add_argument(
            "--output_path",
            type=str,
            help="Path to the folder where the resulting dataset should be stored.",
        )
        parser.add_argument(
            "--wsi_extension",
            type=str,
            choices=WSI_EXT,
            help="The extension types used for the WSI files, the "
            "options are: " + str(WSI_EXT),
        )

        parser.add_argument(
            "--config",
            type=str,
            help="Path to a config file. The config file can hold the same parameters as the CLI. "
            "Parameters provided with the CLI are always having precedence over the parameters in the config file.",
        )

        # basic setup
        parser.add_argument(
            "--patch_size",
            type=int,
            help="The size of the patches in pixel that will be retrieved from the WSI, e.g. 256 for 256px",
        )
        parser.add_argument(
            "--patch_overlap",
            type=float,
            help="The percentage amount pixels that should overlap between two different patches. "
            "Please Provide as integer between 0 and 100, indicating overlap in percentage.",
        )
        parser.add_argument(
            "--target_mpp",
            type=float,
            help="If this parameter is provided, the output level of the WSI "
            "corresponds to the level that is at the target microns per pixel of the WSI. "
            "Alternative to target_mag, downsaple and level. Highest priority, overwrites all other setups for magnifcation, downsample, or level.",
        )
        parser.add_argument(
            "--target_mag",
            type=float,
            help="If this parameter is provided, the output level of the WSI "
            "corresponds to the level that is at the target magnification of the WSI. "
            "Alternative to target_mpp, downsaple and level. High priority, just target_mpp has a higher priority, overwrites downsample and level if provided.",
        )
        parser.add_argument(
            "--downsample",
            type=int,
            help="Each WSI level is downsampled by a factor of 2, downsample "
            "expresses which kind of downsampling should be used with "
            "respect to the highest possible resolution. Medium priority, gets overwritten by target_mag and target_mpp if provided, "
            "but overwrites level.",
        )
        parser.add_argument(
            "--level",
            type=int,
            help="The tile level for sampling, alternative to downsample. "
            "Lowest priority, gets overwritten by target_mag and downsample if they are provided. ",
        )
        parser.add_argument(
            "--context_scales",
            nargs="*",
            type=int,
            help="Define context scales for context patches. Context patches are centered around a central patch. "
            "The context-patch size is equal to the patch-size, but downsampling is different",
        )
        parser.add_argument(
            "--check_resolution",
            type=float,
            help="If a float value is supplies, the program checks whether "
            "the resolution of all images corresponds to the given "
            "value",
        )
        parser.add_argument(
            "--processes",
            type=int,
            help="The number of processes to use.",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            default=None,
            help="Overwrite the patches that have already been created in "
            "case they already exist. Removes dataset. Handle with care!",
        )

        # annotation specific settings
        parser.add_argument(
            "--annotation_paths",
            type=str,
            help="Path to the subfolder where the XML/JSON annotations are "
            "stored or path to a file",
        )
        parser.add_argument(
            "--annotation_extension",
            type=str,
            choices=ANNOTATION_EXT,
            help="The extension types used for the annotation files, the "
            "options are: " + str(ANNOTATION_EXT),
        )
        parser.add_argument(
            "--incomplete_annotations",
            action="store_true",
            default=None,
            help="Set to allow WSI without annotation file",
        )
        parser.add_argument(
            "--label_map_file",
            type=str,
            help="The path to a json file that contains the mapping between"
            " the annotation labels and some integers; an example can "
            "be found in examples",
        )
        parser.add_argument(
            "--save_only_annotated_patches",
            action="store_true",
            default=None,
            help="If true only patches containing annotations will be stored",
        )
        parser.add_argument(
            "--exclude_classes",
            action="append",
            default=None,
            help="Can be used to exclude annotation classes",
        )
        parser.add_argument(
            "--store_masks",
            action="store_true",
            default=None,
            help="Set to store masks per patch. Defaults to false",
        )
        parser.add_argument(
            "--overlapping_labels",
            action="store_true",
            default=None,
            help="Per default, labels (annotations) are mutually exclusive. "
            "If labels overlap, they are overwritten according to the label_map.json ordering"
            " (highest number = highest priority)",
        )

        # macenko stain normalization
        parser.add_argument(
            "--normalize_stains",
            action="store_true",
            default=None,
            help="Uses Macenko normalization on a portion of the whole " "slide image",
        )
        parser.add_argument(
            "--normalization_vector_json",
            type=str,
            help="The path to a JSON file where the normalization vectors are stored",
        )
        parser.add_argument(
            "--adjust_brightness",
            action="store_true",
            default=None,
            help="Normalize brightness in a batch by clipping to 90 percent. Not recommended, but kept for legacy reasons",
        )

        # finding patches
        parser.add_argument(
            "--min_intersection_ratio",
            type=float,
            help="The minimum intersection between the tissue mask and the patch. "
            "Must be between 0 and 1. 0 means that all patches are extracted.",
        )
        parser.add_argument(
            "--tissue_annotation",
            type=str,
            help="Can be used to name a polygon annotation to determine the tissue area. "
            "If a tissue annotation is provided, no Otsu-thresholding is performed",
        )
        parser.add_argument(
            "--tissue_annotation_intersection_ratio",
            type=float,
            help="Intersection ratio with tissue annotation. Helpful, if ROI annotation is passed, "
            "which should not interfere with background ratio. If not provided, the default min_intersection_ratio with the background is used."
        )
        parser.add_argument(
            "--masked_otsu",
            action="store_true",
            default=None,
            help="Use annotation to mask the thumbnail before otsu-thresholding is used",
        )
        parser.add_argument(
            "--otsu_annotation",
            type=str,
            help="Can be used to name a polygon annotation to determine the area "
            "for masked otsu thresholding. Seperate multiple labels with ' ' (whitespace)",
        )
        parser.add_argument(
            "--filter_patches",
            action="store_true",
            default=None,
            help="Post-extraction patch filtering to sort out artefacts, marker and other non-tissue patches with a DL model. Time consuming. Defaults to False.",
        )
        parser.add_argument(
            "--apply_prefilter",
            action="store_true",
            default=None,
            help="Pre-extraction mask filtering to remove marker from mask before applying otsu. Defaults to False.",
        )

        # other
        parser.add_argument(
            "--log_path",
            type=str,
            help="Path where log files should be stored. Otherwise, log files are stored in the output folder",
        )
        parser.add_argument(
            "--log_level",
            type=str,
            choices=LOGGING_EXT,
            help=f"Set the logging level. Options are {LOGGING_EXT}",
        )
        parser.add_argument(
            "--hardware_selection",
            type=str,
            choices=["cucim", "openslide"],
            help="Select hardware device (just if available, otherwise always cucim). Defaults to cucim.",
        )
        parser.add_argument(
            "--wsi_properties",
            type=dict,
            help="Dictionary with manual WSI metadata, but just applies if metadata cannot be derived from OpenSlide (e.g., for .tiff files). Supported keys are slide_mpp and magnification",
        )

        self.parser = parser

    def get_config(self) -> Tuple[PreProcessingConfig, logging.Logger]:
        """Setup function for the CLI-configuration.

        At first, all CLI arguments are loaded. Then the provided configuration file
        (needs to be a `.yaml` file) is loaded. CLI arguments are having a higher priority than
        arguments stored in the configuration file.
        The configuration is stored as an :obj:`~preprocessing.src.cli.PreProcessingConfig` object.
        A logger object is instantiated and returned.

        Raises:
            ValueError: The provided configuration file must be a yaml file.

        Returns:
            - PreProcessingConfig: Preprocessing configuration
            - logging.Logger: Logging object
        """
        opt = self.parser.parse_args()

        if opt.config is not None:
            opt_dict = vars(opt)
            if Path(opt.config).suffix != ".yaml":
                raise ValueError("Please provide config file as `.yaml` file")
            with open(opt.config, "r") as config_file:
                yaml_config = yaml.safe_load(config_file)
                yaml_config = PreProcessingYamlConfig(**yaml_config)

                # convert to dict and override missing values
                yaml_config_dict = dict(yaml_config)

                for k, v in opt_dict.items():
                    if v is None:
                        if yaml_config_dict[k] is not None:
                            opt_dict[k] = yaml_config_dict[k]
                opt_dict = {k: v for k, v in opt_dict.items() if v is not None}

        else:
            opt_dict = vars(opt)
            opt_dict = {k: v for k, v in opt_dict.items() if v is not None}

        # generate final setup
        self.preprocessconfig = PreProcessingConfig(**opt_dict)
        
        # create logger
        preprocess_logger = Logger(
            level=self.preprocessconfig.log_level.upper(),
            log_dir=self.preprocessconfig.log_path,
            comment="preprocessing",
            use_timestamp=True,
        )
        self.logger = preprocess_logger.create_logger()
        self.logger.debug("Parsed CLI without errors. Logger instantiated.")

        return self.preprocessconfig, self.logger

    def store_config(self) -> None:
        """Store the config file in the logging directory to keep track of the configuration."""
        # get dict and convert paths to str
        config_repr = self.preprocessconfig.dict()
        config_repr_str = {
            k: str(v) for k, v in config_repr.items() if isinstance(v, Path)
        }
        for k, v in config_repr_str.items():
            config_repr[k] = v
        # store in log directory
        with open(self.preprocessconfig.log_path / "config.yaml", "w") as yaml_file:
            yaml.dump(config_repr, yaml_file, sort_keys=False)

        self.logger.debug(
            f"Stored config under: {str(self.preprocessconfig.log_path / 'config.yaml')}"
        )


class MacenkoYamlConfig(PreProcessingYamlConfig):
    wsi_path: Optional[str]
    save_json_path: Optional[str]


class MacenkoConfig(PreProcessingConfig):
    save_json_path: str


class MacenkoParser(ABCParser):
    """Macenko Vector Calculation CLI"""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        # dataset paths
        parser.add_argument(
            "--wsi_path",
            type=str,
            help="Path to a single WSI-file.",
        )
        parser.add_argument(
            "--wsi_extension",
            type=str,
            choices=WSI_EXT,
            help="The extension types used for the WSI file, the "
            "options are: " + str(WSI_EXT),
        )
        parser.add_argument(
            "--save_json_path",
            type=str,
            help="The path to a JSON file where the normalization vectors are going to be stored",
        )

        parser.add_argument(
            "--config",
            type=str,
            help="Path to a config file. The config file can hold the same parameters as the CLI. "
            "Parameters provided with the CLI are always having precedence over the parameters in the config file.",
        )

        # basic setup
        parser.add_argument(
            "--patch_size",
            type=int,
            help="The size of the patches in pixel that will be retrieved from the WSI, e.g. 256 for 256px",
        )
        parser.add_argument(
            "--patch_overlap",
            type=float,
            help="The percentage amount pixels that should overlap between two different patches. "
            "Please Provide as integer between 0 and 100, indicating overlap in percentage.",
        )
        parser.add_argument(
            "--downsample",
            type=int,
            help="Each WSI level is downsampled by a factor of 2, downsample "
            "expresses which kind of downsampling should be used with "
            "respect to the highest possible resolution. Medium priority, gets overwritten by target_mag if provided, "
            "but overwrites level.",
        )
        parser.add_argument(
            "--target_mag",
            type=float,
            help="If this parameter is provided, the output level of the WSI "
            "corresponds to the level that is at the target magnification of the WSI. "
            "Alternative to downsaple and level. Highest priority, overwrites downsample and level if provided.",
        )
        parser.add_argument(
            "--level",
            type=int,
            help="The tile level for sampling, alternative to downsample. "
            "Lowest priority, gets overwritten by target_mag and downsample if they are provided. ",
        )
        # annotations
        parser.add_argument(
            "--annotation_paths",
            type=str,
            help="Path to the subfolder where the XML/JSON annotations are "
            "stored or path to a file",
        )
        parser.add_argument(
            "--annotation_extension",
            type=str,
            choices=ANNOTATION_EXT,
            help="The extension types used for the annotation files, the "
            "options are: " + str(ANNOTATION_EXT),
        )
        parser.add_argument(
            "--label_map_file",
            type=str,
            help="The path to a json file that contains the mapping between"
            " the annotation labels and some integers; an example can "
            "be found in examples",
        )
        parser.add_argument(
            "--save_only_annotated_patches",
            action="store_true",
            default=None,
            help="If true only patches containing annotations will be stored",
        )
        parser.add_argument(
            "--exclude_classes",
            action="append",
            default=None,
            help="Can be used to exclude annotation classes",
        )

        # appearance
        parser.add_argument(
            "--adjust_brightness",
            action="store_true",
            default=None,
            help="Normalize brightness in a batch by clipping to 90 percen0. Not recommended, but kept for legacy reasonst",
        )

        # finding patches
        parser.add_argument(
            "--min_intersection_ratio",
            type=float,
            help="The minimum intersection between the tissue mask and the patch. "
            "Must be between 0 and 1. 0 means that all patches are extracted.",
        )
        parser.add_argument(
            "--tissue_annotation",
            type=str,
            help="Can be used to name a polygon annotation to determine the tissue area. "
            "If a tissue annotation is provided, no Otsu-thresholding is performed",
        )
        parser.add_argument(
            "--masked_otsu",
            action="store_true",
            default=None,
            help="Use annotation to mask the thumbnail before otsu-thresholding is used",
        )
        parser.add_argument(
            "--otsu_annotation",
            type=str,
            help="Can be used to name a polygon annotation to determine the area "
            "for masked otsu thresholding. Seperate multiple labels with ' ' (whitespace)",
        )

        # other
        parser.add_argument(
            "--log_path",
            type=str,
            help="Path where log files should be stored. Otherwise, log files are stored in the output folder",
        )
        parser.add_argument(
            "--log_level",
            type=str,
            choices=LOGGING_EXT,
            help=f"Set the logging level. Options are {LOGGING_EXT}",
        )
        parser

        self.parser = parser

        self.default_dict = {
            "check_resolution": False,
            "processes": 1,
            "overwrite": False,
            "store_masks": False,
            "overlapping_labels": False,
            "normalization_vector_json": None,
            "normalize_stains": False,
        }

    def get_config(self) -> Tuple[MacenkoConfig, logging.Logger]:
        opt = self.parser.parse_args()
        if opt.config is not None:
            if Path(opt.config).suffix != ".yaml":
                raise ValueError("Please provide config file as `.yaml` file")
            with open(opt.config, "r") as config_file:
                yaml_config = yaml.safe_load(config_file)
                yaml_config = MacenkoYamlConfig(**yaml_config)

        # convert to dict and override missing values
        opt_dict = vars(opt)
        yaml_config_dict = dict(yaml_config)

        for k, v in opt_dict.items():
            if v is None:
                if yaml_config_dict[k] is not None:
                    opt_dict[k] = yaml_config_dict[k]
        opt_dict = {k: v for k, v in opt_dict.items() if v is not None}

        opt_dict["wsi_paths"] = copy(opt_dict["wsi_path"])
        opt_dict.pop("wsi_path")

        # overwrite hard coded options
        for k, v in self.default_dict.items():
            opt_dict[k] = v

        assert (
            Path(opt_dict["save_json_path"]).suffix == ".json"
        ), "Output path must be a .json file"

        opt_dict["output_path"] = str(Path(opt_dict["save_json_path"]).parent)

        self.preprocessconfig = MacenkoConfig(**opt_dict)
        # create logger
        preprocess_logger = Logger(
            level=self.preprocessconfig.log_level.upper(),
            log_dir=self.preprocessconfig.log_path,
            comment="preprocessing",
            use_timestamp=True,
        )
        self.logger = preprocess_logger.create_logger()
        self.logger.debug("Parsed CLI without errors. Logger instantiated.")

        return self.preprocessconfig, self.logger

    def store_config(self) -> None:
        """Store the config file in the logging directory to keep track of the configuration."""
        pass
