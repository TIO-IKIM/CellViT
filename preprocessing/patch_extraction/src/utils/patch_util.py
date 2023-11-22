# -*- coding: utf-8 -*-
# Utility functions regarding patches
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


import json
import math
import warnings
from pathlib import Path
from typing import Generator, List, Tuple, Union

import cv2
import geojson
import numpy as np
import openslide
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image, ImageDraw
from rasterio.features import rasterize
from shapely.affinity import translate
from shapely.geometry import Polygon, shape
from shapely.validation import make_valid

from preprocessing.patch_extraction import logger
from preprocessing.patch_extraction.src.utils.exceptions import WrongParameterException
from preprocessing.patch_extraction.src.utils.masking import (
    convert_polygons_to_mask,
    generate_tissue_mask,
)
from preprocessing.patch_extraction.src.utils.plotting import generate_polygon_overview



def get_files_from_dir(
    file_path: Union[Path, str, List], file_type: str = "svs"
) -> List[Path]:
    """Returns the names of all the files in the provided file path with matching extension

    Args:
        file_path (Union[Path, str, List]): A path object
        file_type (str, optional): The desired file extension. Defaults to "svs".

    Returns:
        List[Path]: A list of valid paths with the given file extension
    """
    if not isinstance(file_path, list):
        file_path = [file_path]
    all_files = []
    for curr_path in file_path:
        # Could be that the path itself is a WSI
        curr_path = Path(curr_path)
        if curr_path.suffix[1:] == file_type and curr_path.is_file():
            all_files += [curr_path]
        else:
            all_files += [
                curr_file
                for curr_file in curr_path.glob("*." + file_type)
                if curr_file.is_file()
            ]
            # Could also be (class) folder in folder
            if len(all_files) == 0:
                all_files += [
                    curr_file
                    for curr_file in curr_path.glob("**/*" + file_type)
                    if curr_file.is_file()
                ]

    return all_files


def is_power_of_two(n: int) -> bool:
    """Checks if input integer is power of two

    Args:
        n (int): Integer to check

    Returns:
        bool: True if power of two, else False
    """
    return (n != 0) and (n & (n - 1) == 0)


def patch_to_tile_size(
    patch_size: int, overlap: int, rescaling_factor: float = 1.0
) -> Tuple[int, int]:
    """Given patch size and overlap, it returns the size of the tile of the WSI image.

    The tile size must be reduced on both sides from the overlap, since OpenSlide is adding the overlap to the tile size.
    This function ensures that the resulting patch has the same dimensions as configured and is not extended by the overlap.

    Args:
        patch_size (int): Configured Patch Size (in pixels)
        overlap (int): The amount of overlap between adjecent patches (in pixels)
        recaling_factor (float, optional): Rescaling of tiles after extraction to generate matching microns per pixel.
            Defaults to 1.0 (no rescaling)

    Returns:
        int: Resulting tile size reduced by overlap (with optional rescaling)
        int: Resulting overlap (with optional rescaling)
    """
    if rescaling_factor != 1.0:
        patch_size = rescaling_factor * patch_size
        patch_size = int(np.ceil(patch_size / 2) * 2)
        overlap = rescaling_factor * overlap
        overlap = int(np.ceil(overlap / 2) * 2)
    return patch_size - overlap * 2, overlap
    #return patch_size, overlap

def target_mag_to_downsample(base_mag: float, target_mag: float) -> int:
    """Convert the target magnification to a specific downsampling factor based on the base magnification of an image.

    Resulting downsampling factor must be a power of 2

    Args:
        base_mag (float): Base magnification of WSI
        target_mag (float): Target magnification for patches

    Raises:
        WrongParameterException: Raised when calculating error occurs

    Returns:
        int: Resulting downsampling

    Examples:
        The target magnification is 5 and the base magnification is 40. Then the downsampling
        factor is 40/5 = 8. The downsampling factor must be a power of 2
    """
    try:
        base_mag = int(round(base_mag))
        if target_mag is not None:
            if np.log2((base_mag / target_mag)).is_integer():
                downsample = int(base_mag / target_mag)
            else:
                raise WrongParameterException(
                    "Cannot derive downsampling, because base/target must be a power of 2"
                )
    except KeyError:
        raise WrongParameterException("No base magnification in metadata")

    return downsample


def target_mpp_to_downsample(
    base_mpp: float, target_mpp: float, tolerance: float = 0.1
) -> Tuple[int, float]:
    """Calculate the downsampling factor needed to reach a target microns-per-pixel (mpp) resolution.

    Args:
        base_mpp (float): The base mpp resolution.
        target_mpp (float): The target mpp resolution.
        tolerance (float, optional): The relative tolerance for checking if the target is a power of the base. Defaults to 0.1.

    Raises:
        WrongParameterException: Raised when the target mpp is not a power of the base mpp, with a specified tolerance.

    Returns:
        int: The downsampling factor required to achieve the target mpp resolution.
        float: Rescaling factor: If mpp is not on a pyramid level, we need to rescale manually.
            If rescaling == 1, no rescaling is necessary.
    """
    if target_mpp >= base_mpp:
        exponent_fraction = np.log2((target_mpp / base_mpp))
        nearest_integer = round(exponent_fraction)
        is_close = np.isclose(exponent_fraction, nearest_integer, rtol=tolerance)
        if is_close:
            downsample = round(target_mpp / base_mpp)
            rescaling_factor = 1.0
        else:
            logger.warning(
                f"Requested mpp resolution ({target_mpp}) is not a power of the base resultion {base_mpp}. "
                "We perform rescaling, but this may not be accurate and is very slow!"
            )
            downsample = int(np.floor(target_mpp / base_mpp))
            rescaling_factor = target_mpp / base_mpp
    else:
        logger.warning(
            f"Requested mpp resolution ({target_mpp}) is not a power of the base resultion {base_mpp}. "
            "We perform rescaling, but this may not be accurate and is very slow!"
        )
        downsample = 1
        rescaling_factor = target_mpp / base_mpp
    return downsample, rescaling_factor


def get_regions_xml(
    path: Union[str, Path], exclude_classes: List[str] = None
) -> Tuple[List[np.ndarray], List[str]]:
    # TODO: Implement the functionality, write tests and docstring
    raise NotImplementedError(
        "The function is not implemented yet, but will be published in a future release!"
    )


# def get_regions_xml(
#     path: Union[str, Path], exclude_classes: List[str] = None
# ) -> Tuple[List[np.ndarray], List[str]]:
#     """
#     Parses the XML at the given path, assuming annotation format importable by ImageScope.

#     :param path: The path where the XML file is located
#     :type path: Union[str, Path]
#     :param exclude_classes: A list of annotation classes to be excluded, default to None.
#     :type exclude_classes: str, optional
#     :return: A list of polygons (each described as an ndarray) and a list of their corresponding
#     labels
#     :rtype: Tuple[List[np.ndarray], List[str]]
#     """
#     xml = minidom.parse(str(path))
#     # The first region marked is always the tumour delineation
#     regions_ = xml.getElementsByTagName("Region")
#     polygons, region_labels = [], []
#     for region in regions_:
#         vertices = region.getElementsByTagName("Vertex")
#         attribute = region.getElementsByTagName("Attribute")
#         if len(attribute) > 0:
#             r_label = attribute[0].attributes["Value"].value
#         else:
#             r_label = region.getAttribute("Text")
#         if exclude_classes is not None and r_label in exclude_classes:
#             # exclude label: next loop
#             continue
#         region_labels.append(r_label)

#         # Store x, y coordinates into a 2D array in format [x1, y1], [x2, y2], ...
#         coords = np.zeros((len(vertices), 2))

#         for i, vertex in enumerate(vertices):
#             coords[i][0] = vertex.attributes["X"].value
#             coords[i][1] = vertex.attributes["Y"].value

#         polygons.append(Polygon(coords))
#     return polygons, region_labels


def get_regions_json(
    path: Union[str, Path], exclude_classes: List[str] = None, tissue_annotation: str = None
) -> Tuple[List[Polygon], List[str]]:
    """Retrieve annotation regions from json file (as obtained by QuPath).

    Annotations are returned as a list of polygons and labels as a list of strings.
    Regions with the same label are not merged.

    The QuPath export script can be found here preprocessing/help/qupath_export.groovy # TODO: link

    Args:
        path (Union[str, Path]): Path to json annotation file.
        exclude_classes (List[str], optional): List of classes that should be excluded. Names must match (Case-Sensitive)!
            Defaults to None.

    Returns:
        Tuple[List[Polygon], List[str]]: List of Polygons and Labels
    """
    with open(path) as f:
        gj = geojson.load(f)
    region_labels = []
    all_geometries = []
    for feature in gj:
        if feature.type == "Feature":
            if hasattr(feature, "geometry"):
                new_g = shape(feature["geometry"]).buffer(0)
                all_geometries.append(new_g)
        else:
            raise NotImplementedError("Unknown entry in geojson. Check and prepare appropriately")
        
    # all_geometries = [shape(feature["geometry"]).buffer(0) for feature in gj]

    # filter and add label
    geometries = []
    tissue_region = []
    for feature, geom in zip(gj, all_geometries):
        if "classification" in feature.properties:
            label = feature.properties["classification"]["name"].lower()
            if tissue_annotation is not None and label == tissue_annotation:
                tissue_region.append(geom)
            if exclude_classes is not None and label in exclude_classes:
                continue
            region_labels.append(label)
        else:
            region_labels.append("background")
        geometries.append(geom)

    return geometries, region_labels, tissue_region


def compute_interesting_patches(
    slide: openslide.OpenSlide,
    tiles: DeepZoomGenerator,
    target_level: int,
    target_patch_size: int,
    target_overlap: int,
    tissue_annotation_intersection_ratio: float,
    label_map: dict[str, int],
    rescaling_factor: float = 1.0,
    full_tile_size: int = 2000,
    polygons: List[Polygon] = None,
    region_labels: List[str] = None,
    tissue_annotation: List[Polygon] = None,
    mask_otsu: bool = False,
    otsu_annotation: Union[List[str], str] = "object",
    apply_prefilter: bool = False,
) -> Tuple[List[Tuple[int, int, float]], dict, dict]:
    """Compute interesting patches for a WSI.

    For the given WSI file, first tissue detection is performed to generate a tissue mask. The processing steps are
    performed on a downscaled version of the WSI, which size is defined by full_tile_size. The tissue mask is either
    generated from scratch using otsu-thresholding, by masked otsu-thresholding using the otsu-annotation
    (e.g., a coarse bounding box of tissue to remove marker or just select one tissue specimen on a slide) or by
    using a tissue annotation (then without otsu!). Afterward, the interesting patch coordinates on the target level of
    the image pyramid are calculated.
    Returned are the kist of coordinates with background ratio, the tissue masks (dictionary with mask name and PIL.Images)
    and resulting annotation masks (dictionary with annotation name and PIL.Images). All returned images are having the same shape
    as the tile used for patch detection.

    Args:
        slide (openslide.OpenSlide): Slide as OpenSlide object. Used to get metadata
        tiles (DeepZoomGenerator): Slide as DeepZoomGenerator object. Used to get tile for patch-detection.
            Can also be used with CuCIM deepzoom (DeepZoomGeneratorCucim)
        target_level (int): The tile level for sampling.
        target_patch_size (int): The size of the patches in pixel that will be retrieved from the WSI, e.g. 256 for 256px
            (please clean with patch_to_tile_size if overlap is used)
        target_overlap (int): The amount pixels that should overlap between two different patches
        tissue_annotation_intersection_ratio (float): The minimum intersection between the tissue mask and the patch to not consider as background.
        label_map (dict[str, int]): Dictionary mapping the label names to an integer. Please ensure that background label has integer 0!
        recaling_factor (float, optional): Rescaling of tiles after extraction to generate matching microns per pixel.
            Defaults to 1.0 (no rescaling)
        full_tile_size (int, optional): Tile size for masking. Defaults to 2000.
        polygons (List[Polygon], optional): Annotations of this WSI as a list of polygons (referenced to highest level of WSI). Defaults to None.
        region_labels (List[str], optional): List of labels for the annotations provided as polygons parameter. Defaults to None.
        tissue_annotation (List[Polygon], optional): List with tissue polygons.
            If provided, it has precedence over otsu thresholding. Defaults to None.
        mask_otsu (bool, optional): If a mask should be used for otso thresholding. Defaults to False.
        otsu_annotation (Union[List[str], str], optional): List with annotation names or string with annotation name to use for a masked otsu thresholding.
            Defaults to "object".
        apply_prefilter (bool, optional): If a prefilter should be used to remove markers before applying otsu. Defaults to False.

    Returns:
        Tuple[List[Tuple[int, int, float]], dict, dict]:

        - List[Tuple[int, int, float]]: List with coordinates and background ratios of patches. Each list element
            consists of row-position (int), col-position(int) and background ratio (float).
        - dict[str, Image]: Masks generated during tissue detection stored in dict with keys equals the mask name and values equals the PIL image
        - dict[str, Image]: Annotation masks for provided annotations for the complete WSI. Masks are equal to the tissue masks sizes.
            Keys are the mask names and values are the PIL images.
    """
    if polygons is not None:
        assert len(polygons) == len(
            region_labels
        ), "Polygon list and polygon labels are not having the same length"
    if target_patch_size % 2 != 0:
        logger.warning(
            f"The given patch size {target_patch_size} is not divisible by two. "
            f"This could cause a shift in the coordinate grid during background detection and patch extraction. "
            f"This will not result in an error, but could cause severe side effects and should not be used."
        )
    if target_overlap > 0 and not is_power_of_two(target_overlap):
        logger.warning(
            f"The given overlap {target_overlap} is not a power of 2. "
            f"For a better background selection please consider using a power of 2."
        )

    # Set the target level as default
    largest_single_level = target_level
    # Go through the dimensions of the DeepZoom
    for i, (width, height) in enumerate(tiles.level_dimensions):
        # If it gets too big, take the one we have
        if i > target_level:
            break
        # If we get to a level with sufficiently large sizes, take the level
        if width > full_tile_size or height > full_tile_size:
            largest_single_level = i
            break
    levels_diff = target_level - largest_single_level

    # Get the tile
    # get_thumbnail returns a PIL image with (width, height)
    # while np array use (height=rows,width=columns)
    full_tile = np.array(
        slide.get_thumbnail(tiles.level_dimensions[largest_single_level]),
        dtype=np.uint8,
    )
    diff_height = abs(
        full_tile.shape[0] - tiles.level_dimensions[largest_single_level][1]
    )
    diff_width = abs(
        full_tile.shape[1] - tiles.level_dimensions[largest_single_level][0]
    )

    assert diff_height < 2 and diff_width < 2, (
        f"{full_tile.shape[:-1]} and {tiles.level_dimensions[largest_single_level]} have a total "
        f"difference of {diff_width + diff_height}."
    )

    if tissue_annotation is None or len(tissue_annotation) == 0:
        tissue_mask = generate_tissue_mask(
            tissue_tile=full_tile,
            mask_otsu=mask_otsu,
            polygons=polygons,
            region_labels=region_labels,
            otsu_annotation=otsu_annotation,
            downsample=int(2 ** (tiles.level_count - largest_single_level - 1)),
            apply_prefilter=apply_prefilter,
        )
    else:
        logger.info("Using tissue geometry for background seperation")
        if mask_otsu is True:
            logger.warning(
                "Mask-Otsu is set to true, but tissue annotation has precedence"
            )
        tissue_mask = convert_polygons_to_mask(
            polygons=tissue_annotation,
            reference_size=full_tile.shape,
            downsample=int(2 ** (tiles.level_count - largest_single_level - 1)),
        )

    # Now work with the level/patch size that we actually want
    # Get the number of tiles at the level
    n_cols, n_rows = tiles.level_tiles[target_level]

    downsample_overlap, downsample_patch_size = (
        target_overlap * 1.0,
        target_patch_size * 1.0,
    )
    # downsample_tile_size, _ = patch_to_tile_size(target_patch_size, target_overlap)
    downsample_tile_size = downsample_patch_size * 1.0 #downsample_patch_size * 1.0 #downsample_tile_size * 1.0

    for _ in range(levels_diff):
        downsample_overlap /= 2.0
        downsample_patch_size /= 2.0
        downsample_tile_size /= 2.0

    if downsample_tile_size < 1:
        logger.warning(
            "The tile size is too small to correctly compute the background mask. "
            "Please choose a larger full_tile_size or a larger tile size."
        )
        return [(row, col, 1.0) for row in range(n_rows) for col in range(n_cols)]

    if downsample_overlap % 2 != 0 or downsample_patch_size % 2 != 0:
        logger.warning(
            f"The given setting results in the following specifications for background detection: "
            f"downsampling-patch-size={downsample_patch_size}, downsampling-overlap: {downsample_overlap} "
            f"This could cause a slight shift between the background detection grid and the actual patches extracted. "
            f"Grid is interpolated and rearranged, but side effects are possible."
        )

    pixel_missmatch = int(math.ceil(downsample_patch_size)) - downsample_patch_size
    
    downsample_overlap = int(math.ceil(downsample_overlap))
    downsample_patch_size = int(math.ceil(downsample_patch_size))
    downsample_tile_size = int(math.ceil(downsample_tile_size))

    tissue_mask_image = Image.fromarray(tissue_mask * 255).convert("RGB")
    tissue_mask_image_grid = tissue_mask_image.copy()
    tissue_grid = Image.fromarray(full_tile.copy()).convert("RGB")
    draw = ImageDraw.Draw(tissue_mask_image_grid)
    draw_tissue = ImageDraw.Draw(tissue_grid)

    interesting_patches = []

    offset_grid_row = 0
    offset_grid_col = 0
    for row in range(n_rows):
        for col in range(n_cols):
            (
                down_row_init,
                down_row_end,
                down_col_init,
                down_col_end,
            ) = compute_patch_location_in_level(
                row=row,
                col=col,
                tile_size=downsample_tile_size,
                grid_size=tiles.level_tiles[target_level],
                overlap=downsample_overlap,
                pixel_missmatch=pixel_missmatch
            )            
            # Get the current tile
            # Was passiert wenn down_row_end größer als shape wird?
            curr_tile = tissue_mask[
                down_row_init:down_row_end, down_col_init:down_col_end
            ]

            # The total background is the number of current white pixels plus the pixels that will
            # be added by padding
            total_background_pixels = (
                np.sum(curr_tile)
                + (downsample_patch_size**2)
                - (curr_tile.shape[0] * curr_tile.shape[1])
            )
            background_ratio = total_background_pixels / (downsample_patch_size**2)
            if background_ratio <= 1 - tissue_annotation_intersection_ratio:
                interesting_patches.append((row, col, background_ratio))
                draw.rectangle(
                    [down_col_init, down_row_init, down_col_end, down_row_end],
                    outline="green",
                    width=2,
                )
                draw_tissue.rectangle(
                    [down_col_init, down_row_init, down_col_end, down_row_end],
                    outline="green",
                    width=2,
                )

    mask_images = {
        "mask_nogrid": tissue_mask_image,
        "mask": tissue_mask_image_grid,
        "tissue_grid": tissue_grid,
    }
    if len(polygons) != 0:
        mask_images_annotations = generate_polygon_overview(
            polygons=polygons,
            region_labels=region_labels,
            label_map=label_map,
            downsample=int(2 ** (tiles.level_count - largest_single_level - 1)),
            image=Image.fromarray(full_tile.copy()),
            tissue_grid=tissue_grid,
            reference_size=full_tile.shape,
        )
    else:
        mask_images_annotations = {}

    return interesting_patches, mask_images, mask_images_annotations


def compute_patch_location_in_level(
    row: int,
    col: int,
    tile_size: int,
    grid_size: Tuple[int, int],
    overlap: int,
    pixel_missmatch: int = 0
) -> Tuple[int, int, int, int]:
    """Convert the row and col position of a patch into absolute coordinates.

    Coordinates returned are in the format [row upper left, col upper left, row lower right, col lower right].
    Function considers overlap to use generated overlap grid.
    This is necessary to get the coordinates of a patch on a complete slide, by considering the overlap of the slide.
    Due to overlap, the coordinatze grid is shifted, starting from the upper left corner (0,0).

    Args:
        row (int): Row-position of patch on target level
        col (int): Col-position of patch on target level
        tile_size (int): Tile size, according to the target level
            (because the patch-size of target level is usually different to thumbnail used for patch calculations)
        grid_size (Tuple[int, int]): Number of cols, rows of the image at the target level.
            Be careful: Differing from usually ordering, cols and rows are switched in their position!
        overlap (int): Overlap as integer
        pixel_missmatch (int, optional): Missmatch between ceiling integer and grid. Defaults to 0 
        
    Returns:
        Tuple[int, int, int, int]: Position of patch in absolute pixel values [row upper left, col upper left, row lower right, col lower right]
    """
    top, right, bottom, left = compute_overlap(row, col, grid_size, overlap)
    # Find location of the downsampled patch
    row_init = max(0, row * tile_size - top)
    col_init = max(0, col * tile_size - left)
    row_end = (row + 1) * tile_size + bottom
    col_end = (col + 1) * tile_size + right
    if pixel_missmatch != 0:
        correction_row = int(math.floor(pixel_missmatch*row))
        correction_col = int(math.floor(pixel_missmatch*col))
        row_init = row_init - correction_row
        col_init = col_init - correction_col
        row_end = row_end - correction_row
        col_end = col_end - correction_col
    return row_init, row_end, col_init, col_end

def compute_overlap(
    row: int,
    col: int,
    grid_size: Tuple[int, int],
    overlap: int,
) -> Tuple[int, int, int, int]:
    """Calculate top/left and bottom/right overlap according to the position of the tile

    Args:
        row (int): Row position
        col (int): Col position
        grid_size (Tuple[int, int]): Number of cols, rows of the image at the target level.
            Be careful: Differing from usually ordering, cols and rows are switched in their position!
        overlap (int): Overlap in pixels.

    Returns:
        Tuple[int, int, int, int]: _description_
        # TODO: return overlap docstring
    """
    # Checked - True!
    left, top = tuple(overlap * int(t != 0) for t in (col, row))
    right, bottom = tuple(
        overlap * int(t != t_lim - 1) for t, t_lim in zip((col, row), grid_size)
    )
    return top, right, bottom, left


def generate_thumbnails(
    slide: OpenSlide,
    slide_mpp: float,
    sample_factors: List[int] = [32, 64, 128],
    mpp_factors: List[float] = [5, 10],
) -> dict:
    """Generates a dictionary with thumbnails and corresponding thumbnail names

    Args:
        slide (OpenSlide): Slide to retrieve thumbnails from.
        sample_factors (List[int], optional): Sample factors for downsampling of original size (highest level). Defaults to [32, 64, 128].
        mpp_factors (List[float], optional): List of microns per pixels to retrieve. Defaults to [5, 10].

    Returns:
        dict[str, Image]: dictionary with thumbnails and corresponding thumbnail names.
            Names are keys, PIL Images are values.

    Todo:

        * TODO: Error handling if mpp are not available
    """
    # dict: str, PIL
    logger.debug(f"Save thumbnails of image at different scales: {sample_factors}")

    thumbnails = {}
    # downsampling
    for sample_factor in sample_factors:
        thumbnail = slide.get_thumbnail(
            (
                int(slide.properties["openslide.level[0].width"]) / sample_factor,
                int(slide.properties["openslide.level[0].height"]) / sample_factor,
            )
        )
        thumbnails[f"downsample_{sample_factor}"] = thumbnail
    # slide_mpp = float(slide.properties["openslide.mpp-x"])
    # matching microns per pixel
    for mpp in mpp_factors:
        sample_factor = round(mpp / slide_mpp)
        thumbnail = slide.get_thumbnail(
            (
                int(slide.properties["openslide.level[0].width"]) / sample_factor,
                int(slide.properties["openslide.level[0].height"]) / sample_factor,
            )
        )
        thumbnails[f"mpp_{mpp}"] = thumbnail
    # thumbnail with 5 mpp
    sample_factor = round(5 / slide_mpp)
    thumbnail = slide.get_thumbnail(
        (
            int(slide.properties["openslide.level[0].width"]) / sample_factor,
            int(slide.properties["openslide.level[0].height"]) / sample_factor,
        )
    )
    thumbnails["thumbnail"] = thumbnail

    return thumbnails


def chunks(lst: Union[List, Tuple], n: int) -> Generator:
    """Yield successive n-sized chunks from list.

    Args:
        lst (Union[List, Tuple]):  A list to split into chunks
        n (int): The maximum size of each chunk

    Yields:
        Generator: A splitted list as generator
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def pad_tile(new_tile: np.ndarray, patch_size: int, x: int, y: int) -> np.ndarray:
    """Returns a patch of the expected size. OpenSlide does not create patches of the correct size
    at the borders, so the output of OpenSlide is taken and if it does not have the correct size,
    it is padded.


    Args:
        new_tile (np.ndarray): The tile as returned by OpenSlide
        patch_size (int):  The desired size of all patches
        x (int): The row position of the patch, used to determine where the padding should be added
        y (int): The column position of the patch, used to determine where the padding should be added

    Returns:
        np.ndarray: A patch of size (patch_size, patch_size)
    """
    if np.shape(new_tile) == (patch_size, patch_size, 3):
        return new_tile

    logger.debug("Padding Tile")
    missing_x = patch_size - np.shape(new_tile)[0]
    missing_y = patch_size - np.shape(new_tile)[1]
    if y == 0:
        x_tup = (missing_x, 0)
    else:
        x_tup = (0, missing_x)
    if x == 0:
        y_tup = (missing_y, 0)
    else:
        y_tup = (0, missing_y)
    return np.pad(new_tile, pad_width=(x_tup, y_tup, (0, 0)), constant_values=255)


class NormalizeParameters:
    """Macenko Stain Normalization.

    Contains vectors used as reference for the Macenko normalization.
    The H&E reference matrix should have shape (3, 2) where the rows are RGB positions and the
    columns are the two components (first column Haematoxylin, second column Eosin).
    The maximum saturation (stain concentration) should have shape (2, 1) and it represents the
    maximum saturation/concentration that the Haematoxylin and Eosin will have in optical density
    before the conversion to RGB.

    Args:
        normalization_vector_path (Union[Path, str], optional): Path to a normalization vector, must be a json file. Defaults to None.
    """

    def __init__(self, normalization_vector_path: Union[Path, str] = None) -> None:
        if normalization_vector_path is not None:
            with open(str(Path(normalization_vector_path).resolve()), "r") as norm_file:
                json_file = json.load(norm_file)
                self.set_max_sat(np.array(json_file["max_sat"]))
                self.set_he_ref(np.array(json_file["stain_vectors"]))
        else:
            self._MAX_SAT = np.array([[1.9713055649557338], [0.741354425035508]])
            self._HE_REF = np.array(
                [
                    [0.5001340654085598, 0.004804369872676684],
                    [0.7272425313652708, 0.7330272758823506],
                    [0.47008958421915664, 0.6801822776599128],
                ]
            )

    def set_he_ref(self, new_ref: np.ndarray) -> None:
        """Set the reference H&E matrix.

        Args:
            new_ref (np.ndarray): A (3, 2) ndarray representing some reference values for a H&E WSI
        """
        if new_ref.shape == (3, 2):
            self._HE_REF = new_ref
        else:
            logger.warning(
                f"The chosen reference H&E stain vector does not have the required shape (3, "
                f"2) but has shape {new_ref.shape}."
            )

    def get_he_ref(self) -> np.ndarray:
        """Returns the reference H&E vector.


        Returns:
            np.ndarray: A (3, 2) ndarray representing some reference values for a H&E WSI
        """
        return self._HE_REF

    def set_max_sat(self, new_ref: np.ndarray) -> None:
        """Set the desired maximum saturation.

        Args:
            new_ref (np.ndarray): A (2, 1) ndarray representing the desired maximum saturation.
        """
        if new_ref.shape == (2, 1):
            self._MAX_SAT = new_ref
        else:
            logger.warning(
                f"The chosen reference H&E stain vector does not have the required shape (2,) "
                f"but has shape {new_ref.shape}."
            )

    def get_max_sat(self) -> np.ndarray:
        """Returns the maximum saturation.


        Returns:
            np.ndarray: A (2, 1) ndarray representing the desired maximum saturation.
        """
        return self._MAX_SAT


def standardize_brightness(stains: List[np.ndarray]) -> List[np.ndarray]:
    """Standardize brightness of the list of patches (take 90th percentile)

    Compare: https://github.com/wanghao14/Stain_Normalization/blob/master/stain_utils.py

    Args:
        stains (List[np.ndarray]): List of patches

    Returns:
        List[np.ndarray]: List of patches with normalized brightness
    """

    stacked_stains = np.stack(stains, axis=0)
    # Take the 90th percentile of the brightness
    return [
        p.astype(np.uint8)
        for p in np.clip(
            stacked_stains * 255.0 / np.percentile(stacked_stains, 90), 0, 255
        )
    ]


def macenko_normalization(
    patches: List[np.ndarray],
    beta: float = 0.15,
    alpha: int = 1,
    light_intensity: int = 255,
    normalization_vector_path: Union[Path, str] = None,
) -> Tuple[List[np.ndarray], Union[None, np.ndarray], Union[None, np.ndarray]]:
    """Normalizes the stain appareance of the patches using Macenko.

    References:
    ---------
    - A method for normalizing histology slides for quantitative analysis. M. Macenko et al.,
    ISBI 2009
    - https://github.com/schaugf/HEnorm_python/blob/master/normalizeStaining.py
    - https://github.com/CODAIT/deep-histopath/blob/c8baf8d47b6c08c0f6c7b1fb6d5dd6b77e711c33/
    deephistopath/preprocessing.py
    - https://towardsdatascience.com/stain-estimation-on-microscopy-whole-slide-images-2b5a57062268

    Args:
        patches (List[np.ndarray]): A list of patches
        beta (float, optional):  Which portion of the transparent values should be removed. Defaults to 0.15.
        alpha (int, optional): Which percentiles of the values to consider (alpha, 1 - alpha). Defaults to 1.
        light_intensity (int, optional):  The highest luminosity value. Defaults to 255.
        normalization_vector_path (Union[Path, str], optional): Path to a normalization file (needs to be a json file). Defaults to None.

    Returns:
        Tuple[List[np.ndarray], Union[None, np.ndarray], Union[None, np.ndarray]]: A stained-normalized list of patches,
            the mixing stain vectors and the 99th percentile saturation vectors
    """
    # Load normalization parameters
    normalization_vector_patch = NormalizeParameters(
        normalization_vector_path=normalization_vector_path
    )

    # Get all the patches from the dictionary
    h, w, c = patches[0].shape
    # Stack all the patches on a new dimension and reshape it such that it is just a vector of RGB
    # intensities
    stacked_patches = np.stack(patches, axis=0).reshape(-1, 3)
    # Histo images: The more stain there is, the darker the color
    # According to Beer-Lambert law the light will be attenuated exponentially with the density of the tissue

    # Convert to OD (optical density)
    OD = RGB_to_OD(stacked_patches, light_intensity)

    # Remove transparent pixels up to the value beta
    non_trasparent = (OD > beta).any(axis=1)

    # Compute eigenvectors
    try:
        _, eig_vecs = np.linalg.eigh(np.cov(OD[non_trasparent], rowvar=False))
    except np.linalg.LinAlgError:
        logger.warning(
            "numpy.linalg.LinAlgError: Eigenvalues did not converge. The normalization "
            "could not be performed, try using a higher level with more patches."
        )
        return patches, None, None

    # Take the two largest eigenvectors (strip off the residual stain component)
    eig_vecs = eig_vecs[:, 1:3]
    # We assume they need to be positive
    if eig_vecs[0, 0] < 0:
        eig_vecs[:, 0] *= -1
    if eig_vecs[0, 1] < 0:
        eig_vecs[:, 1] *= -1
    # Project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    # N x 3 * 3 x 2 = N x 2
    T_hat = OD[non_trasparent].dot(eig_vecs)
    # The stain vectors are represented by the maximum and minimum angle
    # We first compute the angle: phi = arctan2(y, x) -> angle between x axis and line passing through (y, x)
    phi = np.arctan2(T_hat[:, 1], T_hat[:, 0])
    # And estimate the maximum and minimum using percentiles
    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100 - alpha)

    # Trasform angles back to 3D coordinates
    v_min = eig_vecs.dot(np.array([(np.cos(min_phi), np.sin(min_phi))]).T)
    v_max = eig_vecs.dot(np.array([(np.cos(max_phi), np.sin(max_phi))]).T)

    # A heuristic to make the vector corresponding to Haematoxylin first and the
    # one corresponding to Eosin second (Haematoxylin is larger)
    if v_min[0] > v_max[0]:
        stain_vectors = np.array([v_min[:, 0], v_max[:, 0]]).T
    else:
        stain_vectors = np.array([v_max[:, 0], v_min[:, 0]]).T

    # Determine concentrations/saturations of the individual stains
    # Note: Here, we solve
    #    OD = VS
    #     S = V^{-1}OD
    # where 'OD' is the matrix of optical density values of our image,
    # 'V' is the matrix of stain vectors, and 'S' is the matrix of stain
    # saturations.  Since this is an overdetermined system, we use the
    # least squares solver, rather than a direct solve.
    # OD[non_trasparent].T: Rows correspond to channels (RGB), columns to OD values
    # 3 x (patch_size * patch_size)
    # We want to obtain a matrix with two values (H&E), but we have three
    # 2 x (patch_size * patch_size)
    sat = np.linalg.lstsq(stain_vectors, OD[non_trasparent].T, rcond=None)[0]

    # Find maximum saturation with percentiles
    # 2 values (H&E)
    max_sat = np.percentile(sat, 99, axis=1, keepdims=True)
    # Normalization: (sat / max_sat) * ref_sat
    # Divide by maximum and multiply by the new reference maximum
    sat = np.multiply(np.divide(sat, max_sat), normalization_vector_patch.get_max_sat())

    # Recreate the image using reference mixing matrix
    # Multiply our matrix by the reference matrix and convert back to RGB
    # Substitute the values that were not transparent with the new matrix
    OD[non_trasparent] = -normalization_vector_patch.get_he_ref().dot(sat).T
    i_norm = np.multiply(light_intensity, np.exp(OD)).reshape((len(patches), h, w, c))
    # Round and clip the values
    np.clip(np.round(i_norm), a_min=0, a_max=255, out=i_norm)
    # logger.debug(f'Restoring the original vector {i_norm.shape} took:'
    #              f' {timedelta(seconds=timer() - s)}.')
    return [p.astype(np.uint8) for p in i_norm], stain_vectors, max_sat


def RGB_to_OD(img: np.ndarray, light_intensity: int = 255) -> np.ndarray:
    """Converts an RGB array to OD.

    Args:
        img (np.ndarray): An RGB array
        light_intensity (int, optional): The highest luminosity value. Defaults to 255.

    Returns:
        np.ndarray: The optical density array
    """
    img[img == 0] = 1  # Avoid 0 division
    return -np.log((img.astype(np.float64)) / light_intensity)


def calculate_background_ratio(patch: np.ndarray, patch_size: int) -> float:
    """Calculates the background ratio of a patch

    Args:
        patch (np.ndarray): Patch as numpy array, could also be with overlap
        patch_size (int): Size of the patch, but without overlap

    Returns:
        float: background ratio (between 0 and 1)
    """
    hsv_img = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_RGB2HSV)
    gray_mask = cv2.inRange(hsv_img, (0, 0, 70), (180, 10, 255))
    black_mask = cv2.inRange(hsv_img, (0, 0, 0), (180, 255, 85))
    background_mask = gray_mask | black_mask
    background_mask = (background_mask / 255).astype(np.uint8)
    total_background_pixels = (
        np.sum(background_mask) + (patch_size**2) - (patch.shape[0] * patch.shape[1])
    )
    background_ratio = total_background_pixels / (patch_size**2)

    return background_ratio


def get_intersected_labels(
    tile_size: int,
    patch_overlap: int,
    row: int,
    col: int,
    label_map: dict[str, int],
    polygons: List[Polygon],
    region_labels: List[str],
    min_intersection_ratio: float = 0.0,
    store_masks: bool = False,
    overlapping_labels: bool = False,
) -> Tuple[List[int], List[float], np.ndarray]:
    """Return intersected labels for a given patch

    Returns a list of integer labels with intersected regions and a list with the intersection ratio.
    Optionally, also the annotation mask can be returned. Per default, labels (annotations) are mutually exclusive.
    If labels overlap, they are overwritten according to the label_map.json ordering (highest number = highest priority).
    True means that the mask array is binary 3D with shape (patch_size, patch_size, len(label_map)), otherwise just (patch_size, patch_size)
    with integer values corresponding to the integers in the label_map file for the annotation label.

    Args:
        tile_size (int): Tile size
        patch_overlap (int): Overlap as integer
        row (int): Row-position of patch
        col (int): Col-position of patch
        label_map (dict[str, int]): Dictionary mapping the label names to an integer. Please ensure that background label has integer 0!
        polygons (List[Polygon]):  Annotations of this WSI as a list of polygons (referenced to highest level of WSI).
        region_labels (List[str]): List of labels for the annotations provided as polygons parameter
        min_intersection_ratio(float, optional): Minimum ratio of intersection between annotation class and patch to be considered as class instance. Defaults to 0.0.
        store_masks (bool, optional): Set to store masks per patch. Defaults to False.
        overlapping_labels (bool, optional): Per default, labels (annotations) are mutually exclusive.
            If labels overlap, they are overwritten according to the label_map.json ordering (highest number = highest priority).
            True means that the mask array is 3D with shape (patch_size, patch_size, label_map), otherwise just (patch_size, patch_size).
            Defaults to False.
    Returns:
        Tuple[List[int], List[float], np.ndarray]: 
           Labels as list
           Ratio for each label
           Mask 
    """
    assert isinstance(polygons, list)

    # Create a polygon that has exactly the size (patch_size, patch_size)
    # even if it cannot be taken like that from the actual WSI
    # We do it like this such that we can always create intersections that have the right
    # size
    row_init = row * tile_size - patch_overlap
    col_init = col * tile_size - patch_overlap
    row_end = (row + 1) * tile_size + patch_overlap
    col_end = (col + 1) * tile_size + patch_overlap
    tile_poly = Polygon(
        [
            [col_init, row_init],
            [col_end, row_init],
            [col_end, row_end],
            [col_init, row_end],
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # tree = STRtree(polygons)

    # Get all the polygons for which we have an intersection
    # found_polys_index = [
    #     i for i in tree.query(tile_poly) if polygons[i].intersects(tile_poly)
    # ]
    found_polys_index = [
        i for i in range(len(polygons)) if polygons[i].intersects(tile_poly)
    ]

    curr_labels = []
    ratios = []
    intersected_polygons = []
    for idx in found_polys_index:
        ratio = tile_poly.intersection(polygons[idx]).area / tile_poly.area
        if ratio > min_intersection_ratio:
            ratios.append(ratio)
            curr_labels.append(label_map[region_labels[idx]])
            intersected_polygons.append(polygons[idx])

    if store_masks:
        poly_label_pair = [
            (
                translate(
                    make_valid(poly).intersection(tile_poly),
                    xoff=-col_init,
                    yoff=-row_init,
                ),
                label_map[region_labels[polygons.index(poly)]],
            )
            for poly in intersected_polygons
        ]  # translated polygon together with label
        polygon_mask_patch = polygon_to_patch_mask(
            poly_label_pair=poly_label_pair,
            patch_size=tile_size,
            mask_size=tile_size,
            overlapping_labels=overlapping_labels,
            num_labels=len(label_map.keys()),
        )
    else:
        polygon_mask_patch = None

    return (
        curr_labels,
        ratios,
        polygon_mask_patch,
    )


def polygon_to_patch_mask(
    poly_label_pair: List[Tuple[Polygon, int]],
    patch_size: int,
    mask_size: int,
    overlapping_labels: bool = False,
    num_labels: int = None,
) -> np.ndarray:
    """Convert polygon to a binary mask

    Args:
        poly_label_pair (List[Tuple[Polygon, int]]): Tuple with Polygons and matching label as integer
        patch_size (int): Patch size
        mask_size (int): Mask output size
        overlapping_labels (bool, optional): Per default, labels (annotations) are mutually exclusive.
            If labels overlap, they are overwritten according to the label_map.json ordering (highest number = highest priority).
            True means that the mask array is 3D with shape (patch_size, patch_size, num_labels), otherwise just (patch_size, patch_size).
            Defaults to False.
        num_labels (int, optional): Maximum number of labels, needed for overlapping labels. Defaults to None.

    Returns:
        np.ndarray: Mask, either with integer values and shape (patch_size, patch_size) for non overlapping labels or binary 3D array
            with shape (patch_size, patch_size, num_labels) with 3 axis corresponding the the labels
    """
    if not overlapping_labels:
        if len(poly_label_pair) == 0:
            # background mask
            return np.zeros((mask_size, mask_size), dtype=np.uint8)
        if len(poly_label_pair) > 1:
            # sort by label map
            index_sorted = np.argsort([elem[1] for elem in poly_label_pair])
            poly_label_pair = [poly_label_pair[idx] for idx in index_sorted]

        # problem: precedence: what to do if labels are overlapping? -> Now they need to be mutually exclusive...
        label_mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
        # All touched produces a smoother contour
        rasterize(poly_label_pair, out=label_mask, all_touched=True)

        # See implementation by O. Ester for further use-cases
    else:
        # TODO: Make a sanity check and test
        assert isinstance(num_labels, int)
        if len(poly_label_pair) == 0:
            # background mask
            return np.zeros((mask_size, mask_size, num_labels), dtype=np.uint8)
        else:
            label_mask = np.zeros((mask_size, mask_size, num_labels), dtype=np.uint8)
            for poly, label in poly_label_pair:
                label_submask = np.zeros((patch_size, patch_size), dtype=np.uint8)
                rasterize((poly, 1), out=label_mask, all_touched=True)
                label_mask[:, :, label] = label_submask

    return label_mask


# ignore kwargs for OpenSlide DeepZoomGenerator
class DeepZoomGeneratorOS(DeepZoomGenerator):
    def __init__(self, osr, tile_size=254, overlap=1, limit_bounds=False, **kwargs):
        """Overwrite DeepZoomGenerator of OpenSlide

            DeepZoomGenerator gets overwritten to provide matching API with CuCim
            No Change in functionality

        Args:
            osr (OpenSlide): OpenSlide Image. Needed for OS compatibility and for retrieving metadata.
            tile_size (int, optional): the width and height of a single tile.  For best viewer
                          performance, tile_size + 2 * overlap should be a power
                          of two.. Defaults to 254.
            overlap (int, optional): the number of extra pixels to add to each interior edge
                          of a tile. Defaults to 1.
            limit_bounds (bool, optional): True to render only the non-empty slide region. Defaults to False.
        """
        super().__init__(osr, tile_size, overlap, limit_bounds)
