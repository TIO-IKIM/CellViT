# -*- coding: utf-8 -*-
# Main Patch Extraction Class for a WSI/Dataset
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import json
import multiprocessing
import os
import random
from functools import partial
from logging import getLogger
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from shutil import rmtree
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from openslide import OpenSlide
from shapely.affinity import scale
from shapely.geometry import Polygon
from tqdm import tqdm

from preprocessing.patch_extraction import logger
from preprocessing.patch_extraction.src.cli import PreProcessingConfig
from preprocessing.patch_extraction.src.process_batch import process_batch
from preprocessing.patch_extraction.src.storage import Storage
from preprocessing.patch_extraction.src.utils.exceptions import (
    UnalignedDataException,
    WrongParameterException,
)
from preprocessing.patch_extraction.src.utils.patch_util import (
    DeepZoomGeneratorOS,
    chunks,
    compute_interesting_patches,
    generate_thumbnails,
    get_files_from_dir,
    get_regions_json,
    get_regions_xml,
    is_power_of_two,
    macenko_normalization,
    pad_tile,
    patch_to_tile_size,
    standardize_brightness,
    target_mag_to_downsample,
)
from utils.tools import end_timer, module_exists, start_timer


def _worker_init(q: multiprocessing.Queue, log_level: str = "DEBUG") -> None:
    """Setting up workers and define loggers for them

    Args:
        q (multiprocessing.Queue): MP Queue
        log_level (str, optional): Logger Level. Defaults to "DEBUG".
    """
    # All records from worker processes go to qh and then into q
    qh = QueueHandler(q)
    curr_logger = getLogger("__main__")
    curr_logger.setLevel(log_level.upper())
    curr_logger.addHandler(qh)


class PreProcessor(object):
    """PreProcessor class. Provides methods to preprocess a whole dataset containing WSI but also just single WSI.

    The configuration is passed via the `slide_processor_config' variable.
    For further configuration options, please see the :obj:`~preprocessing.src.cli.PreProcessingConfig`
    documentation.
    During initialization, all WSI inside the provided inout path with matching extension are loaded,
    as well as annotations if provided and the output path is created.


    Args:
        slide_processor_config (PreProcessingConfig): Preprocessing configuration

    Todo:
        * TODO: Check the docstring link above
        * TODO: Zoomed thumbnail
    """

    def __init__(self, slide_processor_config: PreProcessingConfig) -> None:
        self.config = slide_processor_config
        self.files, self.annotation_files = [], []
        self.num_files = 0

        # paths
        self.setup_output_path(self.config.output_path)
        self._set_wsi_paths(self.config.wsi_paths, self.config.wsi_extension)
        self._set_annotations_paths(
            self.config.annotation_paths,
            self.config.annotation_extension,
            self.config.incomplete_annotations,
        )

        # hardware
        self._set_hardware()

        # convert overlap from percentage to pixels
        self.config.patch_overlap = int(
            np.floor(self.config.patch_size / 2 * self.config.patch_overlap / 100)
        )

        if self.config.context_scales is not None:
            self.save_context = True
        else:
            self.save_context = False

        # set seed
        random.seed(42)

        logger.info(f"Data store directory: {str(self.config.output_path)}")
        logger.info(f"Images found: {self.num_files}")
        logger.info(f"Annotations found: {len(self.annotation_files)}")
        if len(self.config.exclude_classes) != 0:
            logger.warning(f"Excluding classes: {self.config.exclude_classes}")

    @staticmethod
    def setup_output_path(output_path: Union[str, Path]) -> None:
        """Create output path

        Args:
            output_path (Union[str, Path]): Output path for WSI
        """
        if output_path is not None:
            output_path = Path(output_path)
            output_path.mkdir(exist_ok=True, parents=True)

    def _set_wsi_paths(
        self, wsi_paths: Union[str, Path, List], wsi_extension: str
    ) -> None:
        """Set the path(s) to the WSI files. Find all wsi files with given extension

        Args:
            wsi_paths (Union[str, Path, List]): Path to the folder where all WSI are stored or path to a single WSI-file.
            wsi_extension (str): Extension of WSI. Please provide without ".",
                e.g. `svs` would be valid, but `.svs`invalid.
        """
        self.files = get_files_from_dir(wsi_paths, wsi_extension)
        self.files = sorted(self.files, key=lambda x: x.name)
        self.num_files = len(self.files)

    def _set_annotations_paths(
        self,
        annotation_paths: Union[Path, str],
        annotation_extension: str,
        incomplete_annotations: bool = True,
    ) -> None:
        """Set the path to the annotation files. Find all annotations

        Args:
            annotation_paths (Union[Path, str]): Path to the subfolder where the annotations are
                stored or path to a file.
            annotation_extension (str): Extension of Annotations. Please provide without ".",
                e.g. `json` would be valid, but `.json`invalid.
            incomplete_annotations (bool, optional): Set to allow wsi without annotation file. Defaults to True.

        Raises:
            UnalignedDataException: Checking if all annotations have been found when `incomplete_annotations=False`
        """
        if annotation_paths is not None:
            files_list = get_files_from_dir(
                annotation_paths, file_type=annotation_extension
            )
            self.annotation_files = sorted(files_list, key=lambda x: x.name)
            # filter to match WSI files
            self.annotation_files = [
                a for f in self.files for a in self.annotation_files if f.stem == a.stem
            ]
            if not incomplete_annotations:
                if [f.stem for f in self.files] != [
                    f.stem for f in self.annotation_files
                ]:
                    raise UnalignedDataException(
                        "Requested to read annotations but the names of the WSI files does not "
                        "correspond to the number of annotation files. We assume the annotation "
                        "files to have the same name as the WSI files. Otherwise use incomplete_annotations=True"
                    )

    def _set_hardware(self):
        """Either load CuCIM (GPU-accelerated) or OpenSlide"""
        if module_exists("cucim", error="ignore"):
            from cucim import CuImage

            from preprocessing.patch_extraction.src.cucim_deepzoom import (
                DeepZoomGeneratorCucim,
            )

            self.deepzoomgenerator = DeepZoomGeneratorCucim
            self.image_loader = CuImage
        else:
            self.deepzoomgenerator = DeepZoomGeneratorOS
            self.image_loader = OpenSlide

    def sample_patches_dataset(self) -> None:
        """Main functiuon to create a dataset. Sample the complete dataset.

        This function acts as an entrypoint to the patch-processing.
        When this function is called, all WSI that have been detected are processed.
        Depending on the selected configuration, either already processed WSI will be removed or newly processed.
        The processed WSI are stored in the file `processed.json` in the output-folder.
        """
        # perform logical check
        self._check_patch_params(
            patch_size=self.config.patch_size,
            patch_overlap=self.config.patch_overlap,
            downsample=self.config.downsample,
            level=self.config.level,
            min_background_ratio=self.config.min_intersection_ratio,
        )
        # remove database or check to continue from checkpoint
        self._check_overwrite(self.config.overwrite)

        # multiprocessing
        pool_init = self._return_pool_args(self.config.processes, self.config.log_level)
        pool = multiprocessing.Pool(**pool_init)

        total_count = 0
        start_time = start_timer()
        for i, wsi_file in enumerate(self.files):
            try:
                logger.info(f"{(os.get_terminal_size()[0]-33)*'*'}")
            except Exception:
                pass
            logger.info(f"{i+1}/{len(self.files)}: {wsi_file.name}")

            # prepare wsi, espeically find patches
            (
                (n_cols, n_rows),
                (wsi_metadata, mask_images, mask_images_annotations, thumbnails),
                func,
                divided,
            ) = self._prepare_wsi(wsi_file)

            # setup storage
            store = Storage(
                wsi_name=wsi_file.stem,
                output_path=self.config.output_path,
                metadata=wsi_metadata,
                mask_images=mask_images,
                mask_images_annotations=mask_images_annotations,
                thumbnails=thumbnails,
                store_masks=self.config.store_masks,
                save_context=self.config.context_scales is not None,
                context_scales=self.config.context_scales,
            )
            logger.info("Start extracting patches...")

            # Multiprocessing
            asyncs = []
            for batch in divided:
                asyncs.append(pool.apply_async(func, [batch], callback=store.save))
            patch_count = sum([len(a.get()[0]) for a in tqdm(asyncs)])

            if patch_count == 0:
                logger.warning(f"No patches sampled from {wsi_file.name}")
            logger.info(f"Total patches sampled: {patch_count}")

            store.clean_up()
            total_count += patch_count

        pool.close()  # do not accept any more tasks
        pool.join()  # wait for the completion of all scheduled jobs

        logger.info(f"Patches saved to: {self.config.output_path.resolve()}")
        logger.info(f"Total patches sampled for all WSI: {total_count}")

        end_timer(start_time)

    @staticmethod
    def _return_pool_args(processes: Optional[int], log_level: str = "DEBUG") -> Dict:
        """Setting up the multiprocessing pool

        Args:
            processes (Optional[int]): Number of processes to use
            log_level (str, optional): Log level for subprocesses. Defaults to "DEBUG".

        Returns:
            Dict:
        Todo:
            *TODO: how to store dictionary keys and values in docstring (maybe necessary keys. etc)
        """
        # Create a multiprocessing pool
        if processes is None or processes > multiprocessing.cpu_count():
            # Compute number of processes
            processes = int((3.0 / 4.0) * multiprocessing.cpu_count())
        # Handle logger for multiple processes
        q: multiprocessing.Queue = multiprocessing.Queue()
        ql = QueueListener(q, *logger.handlers[1:])
        ql.start()
        logger.info(f"Using {processes} processes.")
        return {
            "processes": processes,
            "initializer": _worker_init,
            "initargs": [q, log_level],
        }

    @staticmethod
    def _check_patch_params(
        patch_size: int,
        patch_overlap: int,
        downsample: int = None,
        target_mag: float = None,
        level: int = None,
        min_background_ratio: float = 1.0,
    ) -> None:
        """Sanity Check for parameters

        See `Raises`section for further comments about the sanity check.

        Args:
            patch_size (int): The size of the patches in pixel that will be retrieved from the WSI, e.g. 256 for 256px
            patch_overlap (int): The amount pixels that should overlap between two different patches.
            downsample (int, optional): Downsampling factor from the highest level (largest resolution). Defaults to None.
            target_mag (float, optional): If this parameter is provided, the output level of the wsi
            corresponds to the level that is at the target magnification of the wsi.
            Alternative to downsaple and level. Defaults to None.
            level (int, optional): The tile level for sampling, alternative to downsample. Defaults to None.
            min_background_ratio (float, optional): Minimum background selection ratio. Defaults to 1.0.

        Raises:
            WrongParameterException: Either downsample, level, or target_magnification must have been selected.
            WrongParameterException: Downsampling must be a power of two.
            WrongParameterException: Negative overlap is not allowed.
            WrongParameterException: Overlap should not be larger than half of the patch size.
            WrongParameterException: Background Percentage must be between 0 and 1.
        """
        if downsample is None and level is None and target_mag is None:
            raise WrongParameterException(
                "Both downsample and level are none, "
                "please fill one of the two parameters."
            )
        if downsample is not None and not is_power_of_two(downsample):
            raise WrongParameterException("Downsample can only be a power of two.")

        if downsample is not None and level is not None:
            logger.warning(
                "Both downsample and level are set, "
                "we will use downsample and ignore level."
            )

        if patch_overlap < 0:
            raise WrongParameterException("Negative overlap not allowed.")
        if patch_overlap > patch_size // 2:
            raise WrongParameterException(
                "An overlap greater than half the patch size yields a tile size of zero."
            )

        if min_background_ratio < 0.0 or min_background_ratio > 1.0:
            raise WrongParameterException(
                "The parameter min_background_ratio should be a "
                "float value between 0 and 1 representing the "
                "maximum percentage of background allowed."
            )

    def _check_overwrite(self, overwrite: bool = False) -> None:
        """Performs data cleanage, depending on overwrite.

        If true, overwrites the patches that have already been created in
        case they already exist. If false, skips already processed files from `processed.json`
        in the provided output path (created during class initialization)

        Args:
            overwrite (bool, optional): Overwrite flag. Defaults to False.
        """
        if overwrite:
            logger.info("Removing complete dataset! This may take a while.")
            subdirs = [f for f in Path(self.config.output_path).iterdir() if f.is_dir()]
            for subdir in subdirs:
                rmtree(subdir.resolve(), ignore_errors=True)
            if (Path(self.config.output_path) / "processed.json").exists():
                os.remove(Path(self.config.output_path) / "processed.json")
            self.setup_output_path(self.config.output_path)
        else:
            try:
                with open(
                    str(Path(self.config.output_path) / "processed.json"), "r"
                ) as processed_list:
                    processed_files = json.load(processed_list)["processed_files"]
                    logger.info(
                        f"Found {len(processed_files)} files. Continue to process {len(self.files)-len(processed_files)}/{len(self.files)} files."
                    )
                    self._drop_processed_files(processed_files)
            except FileNotFoundError:
                logger.info("Empty output folder. Processing all files")

    def _drop_processed_files(self, processed_files: list[str]) -> None:
        """Drop processed file from `processed.json` file from dataset.

        Args:
            processed_files (list[str]): List with processed filenames
        """
        for file in self.files:
            if file.stem in processed_files:
                self.files.remove(file)
                logger.info(f"Dropped: {file.stem}")

    def _check_wsi_resolution(self, slide_properties: dict[str, str]) -> None:
        """Check if the WSI resolution is the same for all files in the dataset. Just returns a warning message if not.

        Args:
            slide_properties (dict[str, str]): Dictionary withn slide properties. Must contain "openslide.mpp-x" and ""openslide.mpp-y" as keys.
        """
        if (
            self.config.check_resolution is not None
            and "openslide.mpp-x" in slide_properties
            and "openslide.mpp-y" in slide_properties
            and (
                round(float(slide_properties["openslide.mpp-x"]), 4)
                != self.config.check_resolution
                or round(float(slide_properties["openslide.mpp-y"]), 4)
                != self.config.check_resolution
            )
        ):
            if self.config.check_resolution is True:
                logger.warning(
                    f"The resolution of the current file does not correspond to the given "
                    f"resolution {self.config.check_resolution}. The resolutions are "
                    f'{slide_properties["openslide.mpp-x"]} and '
                    f'{slide_properties["openslide.mpp-y"]}.'
                )

    def _prepare_wsi(
        self, wsi_file: str
    ) -> Tuple[
        Tuple[int, int], Tuple[dict, dict, dict, dict], Callable, List[List[Tuple]]
    ]:
        """Prepare a WSI for preprocessing

        First, some sanity checks are performed and the target level for DeepZoomGenerator is calculated.
        We are not using OpenSlides default DeepZoomGenerator, but rather one based on the cupy library which is much faster
        (cf https://github.com/rapidsai/cucim). One core element is to find all patches that are non-background patches.
        For this, a tissue mask is generated. At this stage, no patches are extracted!

        For further documentation (i.e., configuration settings), see the class documentation [link].

        Args:
            wsi_file (str): Name of the wsi file

        Raises:
            WrongParameterException: The level resulting from target magnification or downsampling factor must exists to extract patches.

        Returns:
            Tuple[Tuple[int, int], Tuple[dict, dict, dict, dict], Callable, List[List[Tuple]]]:

            - Tuple[int, int]: Number of rows, cols of the WSI at the given level
            - dict: Dictionary with Metadata of the WSI
            - dict[str, Image]: Masks generated during tissue detection stored in dict with keys equals the mask name and values equals the PIL image
            - dict[str, Image]: Annotation masks for provided annotations for the complete WSI. Masks are equal to the tissue masks sizes.
                Keys are the mask names and values are the PIL images.
            - dict[str, Image]: Thumbnail images with different downsampling and resolutions.
                Keys are the thumbnail names and values are the PIL images.
            - callable: Batch-Processing function performing the actual patch-extraction task
            - List[List[Tuple]]: Divided List with batches of batch-size. Each batch-element contains the row, col position of a patch together with bg-ratio.

        Todo:
            * TODO: Check if this works out for non-GPU devices
            * TODO: Class documentation link
        """
        logger.info(f"Computing patches for {wsi_file.name}")

        # load slide (OS and CuImage/OS)
        slide = OpenSlide(str(wsi_file))
        slide_cu = self.image_loader(str(wsi_file))

        # Check whether the resolution of the current image is the same as the given one
        self._check_wsi_resolution(slide.properties)

        # Zoom Recap:
        # - Row and column of the tile within the Deep Zoom level (t_)
        # - Pixel coordinates within the Deep Zoom level (z_)
        # - Pixel coordinates within the slide level (l_)
        # - Pixel coordinates within slide level 0 (l0_)
        # Tile size is the amount of pixels that are taken from the image (without overlaps)
        tile_size = patch_to_tile_size(
            self.config.patch_size, self.config.patch_overlap
        )

        tiles = self.deepzoomgenerator(
            osr=slide,
            cucim_slide=slide_cu,
            tile_size=tile_size,
            overlap=self.config.patch_overlap,
            limit_bounds=True,
        )

        # target mag has precedence before downsample!
        if self.config.target_mag is not None:
            self.config.downsample = target_mag_to_downsample(
                float(slide.properties.get("openslide.objective-power")),  # slide,
                self.config.target_mag,
                self.config.downsample,
            )
        if self.config.downsample is not None:
            # Each level is downsampled by a factor of 2
            # downsample expresses the desired downsampling, we need to count how many times the
            # downsampling is performed to find the level
            # e.g. downsampling of 8 means 2 * 2 * 2 = 3 times
            # we always need to remove 1 level more than necessary, so 4
            # so we can just use the bit length of the numbers, since 8 = 1000 and len(1000) = 4
            level = tiles.level_count - self.config.downsample.bit_length()
        else:
            self.config.downsample = 2 ** (tiles.level_count - level - 1)
        if level >= tiles.level_count:
            raise WrongParameterException(
                "Requested level does not exist. Number of slide levels:",
                tiles.level_count,
            )

        # store level!
        self.curr_wsi_level = level

        # initialize annotation objects
        region_labels: List[str] = []
        polygons: List[Polygon] = []
        polygons_downsampled: List[Polygon] = []
        tissue_region: List[Polygon] = []

        # load the annotation if provided
        if len(self.annotation_files) > 0:
            (
                region_labels,
                polygons,
                polygons_downsampled,
                tissue_region,
            ) = self.get_wsi_annotations(
                wsi_file=wsi_file,
                tissue_annotation=self.config.tissue_annotation,
                downsample=self.config.downsample,
                exclude_classes=self.config.exclude_classes,
            )

        # get the interesting coordinates: no background, filtered by annotation etc.
        # original number of tiles of total wsi
        n_cols, n_rows = tiles.level_tiles[level]
        if self.config.min_intersection_ratio == 0.0 and tissue_region is None:
            # Create a list of all coordinates of the grid -> Whole WSI with background is loaded
            interesting_coords = [
                (row, col, 1.0) for row in range(n_rows) for col in range(n_cols)
            ]
        else:
            (
                interesting_coords,
                mask_images,
                mask_images_annotations,
            ) = compute_interesting_patches(
                polygons=polygons,
                slide=slide,
                tiles=tiles,
                target_level=level if level is not None else 1,
                target_patch_size=self.config.patch_size,
                target_overlap=self.config.patch_overlap,
                mask_otsu=self.config.masked_otsu,
                label_map=self.config.label_map,
                region_labels=region_labels,
                tissue_annotation=tissue_region,
                otsu_annotation=self.config.otsu_annotation,
                min_intersection_ratio=self.config.min_intersection_ratio,
            )
        if len(interesting_coords) == 0:
            logger.warning(f"No patches sampled from {wsi_file.name}")
        wsi_metadata = {
            "orig_n_tiles_cols": n_cols,
            "orig_n_tiles_rows": n_rows,
            "base_magnification": slide.properties.get("openslide.objective-power"),
            "downsampling": self.config.downsample,
            "label_map": self.config.label_map,
            "patch_overlap": self.config.patch_overlap * 2,
            "patch_size": self.config.patch_size,
            "stain_normalization": self.config.normalize_stains,
            "magnification": self.config.target_mag,
            "level": level,
        }

        # divide the list into chunks
        divided = list(chunks(interesting_coords, self.config.patches_per_batch))
        logger.info(f"{wsi_file.name}: Processing {len(interesting_coords)} patches.")
        func = partial(
            process_batch,
            wsi_file=wsi_file,
            wsi_metadata=wsi_metadata,
            patch_size=self.config.patch_size,
            patch_overlap=self.config.patch_overlap,
            level=level,
            polygons=polygons_downsampled,
            region_labels=region_labels,
            label_map=self.config.label_map,
            min_intersection_ratio=self.config.min_intersection_ratio,
            save_only_annotated_patches=self.config.save_only_annotated_patches,
            adjust_brightness=self.config.adjust_brightness,
            normalize_stains=self.config.normalize_stains,
            normalization_vector_path=self.config.normalization_vector_json,
            store_masks=self.config.store_masks,
            overlapping_labels=self.config.overlapping_labels,
            context_scales=self.config.context_scales,
        )

        logger.info("Generate thumbnails")
        thumbnails = generate_thumbnails(slide, sample_factors=[32, 64, 128])

        return (
            (n_cols, n_rows),
            (wsi_metadata, mask_images, mask_images_annotations, thumbnails),
            func,
            divided,
        )

    def save_normalization_vector(
        self, wsi_file: Path, save_json_path: Union[Path, str]
    ) -> None:
        """Save the Macenko Normalization Vector for a WSI in the given file

        Args:
            wsi_file (Path): Path to WSI file, must be within the dataset
            save_json_path (Union[Path, str]): Path to JSON-File where to Macenko-Vectors should be stored.
        """
        # check input
        assert (
            wsi_file in self.files
        ), "WSI-File must be in the Preprocessing WSI dataset!"

        save_json_path = Path(save_json_path)
        assert save_json_path.suffix == ".json", "Output path must be a .json file"

        # perform logical check
        self._check_patch_params(
            patch_size=self.config.patch_size,
            patch_overlap=self.config.patch_overlap,
            downsample=self.config.downsample,
            level=self.config.level,
            min_background_ratio=self.config.min_intersection_ratio,
        )

        (
            (_, _),
            (_, _, _, _),
            _,
            divided,
        ) = self._prepare_wsi(wsi_file)
        # convert divided back to batch
        batch = [item for sublist in divided for item in sublist]

        # open slide
        slide = OpenSlide(str(wsi_file))
        slide_cu = self.image_loader(str(wsi_file))
        tile_size = patch_to_tile_size(
            self.config.patch_size, self.config.patch_overlap
        )

        # extract all patches
        patches = []
        tiles = self.deepzoomgenerator(
            osr=slide,
            cucim_slide=slide_cu,
            tile_size=tile_size,
            overlap=self.config.patch_overlap,
            limit_bounds=True,
        )

        for row, col, _ in batch:
            new_tile = np.array(
                tiles.get_tile(self.curr_wsi_level, (col, row)), dtype=np.uint8
            )
            patches.append(pad_tile(new_tile, self.config.patch_size, col, row))

        if self.config.adjust_brightness:
            patches = standardize_brightness(patches)

        _, stain_vectors, max_sat = macenko_normalization(patches)

        if stain_vectors is not None and max_sat is not None:
            logger.info(f"H&E vector:\n {stain_vectors}")
            logger.info(f"max saturation vector:\n {max_sat}")
            norm_vectors = {}
            norm_vectors["stain_vectors"] = stain_vectors.tolist()
            norm_vectors["max_sat"] = max_sat.tolist()

            save_json_path.parent.mkdir(exist_ok=True, parents=True)
            with save_json_path.open("w") as outfile:
                json.dump(norm_vectors, outfile, indent=2)
            logger.info(f"Normalization vectors stored at {save_json_path}.")

        else:
            logger.warning("The vectors are None and thus they will not be stored.")

    def get_wsi_annotations(
        self,
        wsi_file: Union[Path, str],
        tissue_annotation: str = None,
        downsample: int = 1,
        exclude_classes: List[str] = [],
    ) -> Tuple[List[str], List[Polygon], List[Polygon], List[Polygon]]:
        """Retrieve annotations for a given WSI file

        Loads annotations for a given wsi file. The annotations is downscaled with a given downsaling factor.
        All loaded annotations are converted to shapely polygons.
        If annotations classes should be excluded, please pass a list with exclusion annotations names.
        To retrieve tissue annotations for selecting the tissue area,
        pass a string with the annotation name of the tissue annotation.

        Args:
            wsi_file (Union[Path, str]): Name of WSI file to retrieve annotations from
            tissue_annotation (str, optional): Name of tissue annotation to get a tissue polygon. Defaults to None.
            downsample (int, optional): Downsampling factor to downsample polygons. Defaults to 1.
            exclude_classes (List[str], optional): Annotation classes to exclude. Defaults to [].

        Raises:
            Exception: Raises exception if a tissue region is passed, but not found

        Returns:
            Tuple[List[str], List[Polygon], List[Polygon], List[Polygon]]:

            - List[str]: Polygon labels matching to the returned polygons
            - List[Polygon]: Loaded polygons
            - List[Polygon]: Loaded polygons, scaled by downsampling factor
            - List[Polygon]: Loaded tissue polygons to indicate tissue region (not downscaled!)
        """
        region_labels: List[str] = []
        polygons: List[Polygon] = []
        polygons_downsampled: List[Polygon] = []
        tissue_region: List[Polygon] = []

        # Expect filename of annotations to match WSI file name
        annotation_file = self.get_annotation_file_by_name(wsi_file.stem)
        if annotation_file is not None:
            if self.config.annotation_extension == "xml":
                polygons, region_labels = get_regions_xml(
                    path=annotation_file,
                    exclude_classes=exclude_classes,
                )
            elif self.config.annotation_extension == "json":
                polygons, region_labels = get_regions_json(
                    path=annotation_file, exclude_classes=exclude_classes
                )
            # downsample polygons to match the images
            polygons_downsampled = [
                scale(
                    poly,
                    xfact=1 / downsample,
                    yfact=1 / downsample,
                    origin=(0, 0),
                )
                for poly in polygons
            ]

            if tissue_annotation is not None:
                for poly, region_label in zip(polygons, region_labels):
                    if region_label == tissue_annotation:
                        tissue_region.append(poly)
                if len(region_labels) == 0:
                    raise Exception(
                        f"Tissue annotation ('{tissue_annotation}') is provided but cannot be found in given annotation files. "
                        "If no tissue annotation is existance for this file, consider using otsu_annotation as a non-strict way for passing tissue-annotations."
                    )

        return region_labels, polygons, polygons_downsampled, tissue_region

    def get_annotation_file_by_name(self, wsi_file_stem: str) -> Union[Path, None]:
        """Returns the annoation file as path when the file_stem matches - else return None

        Args:
            wsi_file_stem (str): Name of WSI file without extension.

        Returns:
            Union[Path, None]: The path to annotation file or None.
        """
        for file in self.annotation_files:
            if file.stem == wsi_file_stem:
                return file
        return None
