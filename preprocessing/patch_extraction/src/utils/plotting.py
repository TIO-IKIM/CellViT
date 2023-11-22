# -*- coding: utf-8 -*-
# Plotting functions for preprocessing
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import math
import os
import warnings
from typing import List, Tuple

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import Polygon

from configs.python.config import COLOR_DEFINITIONS
from preprocessing.patch_extraction.src.utils.masking import get_filtered_polygons



def generate_polygon_overview(
    polygons: Tuple[List[Polygon], Polygon],
    region_labels: List[str],
    label_map: dict[str, int],
    reference_size: tuple[int],
    downsample: int = 1,
    image: Image = None,
    tissue_grid: Image = None,
) -> dict:
    """Generate a polygon overview.

    Creates overview images with annotation for each unique label region labels.
    Annotations with the same label are merged. Returns up to two figures for each unique annotation.
    One image is on white background, one overlay on optional background image (e.g., tissue image)

    Args:
        polygons (Tuple[List[Polygon], Polygon]): List of polygons to use.
        region_labels (List[str]): List of labels for the annotations provided as polygons parameter.
        label_map (dict[str, int]): Dictionary mapping the label names to an integer. Please ensure that background label has integer 0!
            Is used here to get the color from preprocessing.src.config.py
        reference_size (tuple[int]): Shape of resulting mask image. Shape should be (height, width, channels).
        downsample (int, optional):  Set the factor by which the polygon should be scaled down. Defaults to 1.
        image (Image, optional): Image as background. Defaults to None.
        tissue_grid (Image, optional): Image with tissue grid as background. Defaults to None.
    Returns:
        dict: Dictionary with annotation names as strings and PIL.Images as keys.
    """

    mask_container = {}
    image_container = {}
    areas = {}
    region_label_set = set(region_labels)

    # save white basic image
    white_bg = Image.fromarray(255 * np.ones(shape=reference_size, dtype=np.uint8))
    white_bg.save("tmp.tif")

    if image is None:
        src = 255 * np.ones(shape=reference_size, dtype=np.uint8)
        image = Image.fromarray(src)
    # draw individual images
    for label in region_label_set:
        label_image = image.copy()
        white_image = white_bg.copy()
        if tissue_grid is not None:
            label_tissue_grid = tissue_grid.copy()
        else:
            label_tissue_grid = None
        label_polygon = get_filtered_polygons(
            polygons, region_labels, label, downsample
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with rasterio.open("tmp.tif") as src:
                out_image, out_transform = rasterio_mask(src, label_polygon, crop=False)
                # check polygon draw
                label_polygon_list = []
                for poly in label_polygon:
                    if poly.type == "MultiPolygon":
                        labels = [x for x in poly.geoms]
                        label_polygon_list = label_polygon_list + labels
                    else:
                        label_polygon_list = label_polygon_list + [poly]
                poly_outline_image = label_image.copy()
                poly_outline_image_draw = ImageDraw.Draw(poly_outline_image)
                [
                    poly_outline_image_draw.polygon(
                        list(lp.exterior.coords),
                        outline=COLOR_DEFINITIONS[label_map[label]],
                        width=5,
                    )
                    for lp in label_polygon_list
                ]
                # [poly_outline_image_draw.polygon(list(lp.interiors), outline=COLOR_DEFINITIONS[label_map[label]] , width=5) for lp in label_polygon_list if len(list(lp.interiors)) > 2]
                # TODO: interiors are wrong, needs to be fixed (check file ID_1004_LOC_4_TIME_2_BUYUE2088_STATUS_0_UID_27 for an example with interiors)

                mask = out_image.transpose(1, 2, 0)
                mask = (mask / 255).astype(np.uint8)
                area = np.sum(mask[:, :, 0])

                opacity_mask = Image.fromarray(mask[:, :, 0] * 180)
                polygon_image = Image.fromarray(
                    (COLOR_DEFINITIONS[label_map[label]] * mask).astype(np.uint8)
                )
                label_image.paste(polygon_image, (0, 0), opacity_mask)
                white_image.paste(polygon_image, (0, 0), opacity_mask)
                mask_container[label] = {
                    "polygon_image": polygon_image,
                    "polygon_outline": label_polygon_list,
                    "mask": opacity_mask,
                }
                if label_tissue_grid is not None:
                    label_tissue_grid.paste(polygon_image, (0, 0), opacity_mask)
                    mask_container[label]["tissue_grid"] = label_tissue_grid
                    image_container[f"{label}_grid"] = label_tissue_grid
                image_container[label] = label_image
                image_container[f"{label}_clean"] = white_image
                image_container[f"{label}_ouline"] = poly_outline_image
                areas[area] = label

    os.remove("tmp.tif")

    # draw all masks on one image, sorted by areas
    sorted_labels = [areas[k] for k in sorted(areas, reverse=True)]

    final_image = image.copy()
    final_white = white_bg.copy()
    final_outline = image.copy()
    final_grid = tissue_grid.copy()
    final_outline_draw = ImageDraw.Draw(final_outline)
    for label in sorted_labels:
        polygon_image = mask_container[label]["polygon_image"]
        opacity_mask = mask_container[label]["mask"]
        polygon_outline = mask_container[label]["polygon_outline"]
        final_image.paste(polygon_image, (0, 0), opacity_mask)
        final_grid.paste(polygon_image, (0, 0), opacity_mask)
        final_white.paste(polygon_image, (0, 0), opacity_mask)
        [
            final_outline_draw.polygon(
                list(lp.exterior.coords),
                outline=COLOR_DEFINITIONS[label_map[label]],
                width=5,
            )
            for lp in polygon_outline
        ]
        # [final_outline_draw.polygon(list(lp.interiors), outline=COLOR_DEFINITIONS[label_map[label]] , width=5) for lp in polygon_outline if len(list(lp.interiors)) > 2]

    image_container["all_overlaid"] = final_image
    image_container["all_overlaid_clean"] = final_white
    image_container["all_overlaid_outline"] = final_outline
    image_container["all_overlaid_grid"] = final_grid
    
    return image_container


class DashedImageDraw(ImageDraw.ImageDraw):
    def thick_line(self, xy, direction, fill=None, width=0):
        # xy – Sequence of 2-tuples like [(x, y), (x, y), ...]
        # direction – Sequence of 2-tuples like [(x, y), (x, y), ...]
        if xy[0] != xy[1]:
            self.line(xy, fill=fill, width=width)
        else:
            x1, y1 = xy[0]
            dx1, dy1 = direction[0]
            dx2, dy2 = direction[1]
            if dy2 - dy1 < 0:
                x1 -= 1
            if dx2 - dx1 < 0:
                y1 -= 1
            if dy2 - dy1 != 0:
                if dx2 - dx1 != 0:
                    k = -(dx2 - dx1) / (dy2 - dy1)
                    a = 1 / math.sqrt(1 + k**2)
                    b = (width * a - 1) / 2
                else:
                    k = 0
                    b = (width - 1) / 2
                x3 = x1 - math.floor(b)
                y3 = y1 - int(k * b)
                x4 = x1 + math.ceil(b)
                y4 = y1 + int(k * b)
            else:
                x3 = x1
                y3 = y1 - math.floor((width - 1) / 2)
                x4 = x1
                y4 = y1 + math.ceil((width - 1) / 2)
            self.line([(x3, y3), (x4, y4)], fill=fill, width=1)
        return

    def dashed_line(self, xy, dash=(2, 2), fill=None, width=0):
        # xy – Sequence of 2-tuples like [(x, y), (x, y), ...]
        for i in range(len(xy) - 1):
            x1, y1 = xy[i]
            x2, y2 = xy[i + 1]
            x_length = x2 - x1
            y_length = y2 - y1
            length = math.sqrt(x_length**2 + y_length**2)
            dash_enabled = True
            postion = 0
            while postion <= length:
                for dash_step in dash:
                    if postion > length:
                        break
                    if dash_enabled:
                        start = postion / length
                        end = min((postion + dash_step - 1) / length, 1)
                        self.thick_line(
                            [
                                (
                                    round(x1 + start * x_length),
                                    round(y1 + start * y_length),
                                ),
                                (
                                    round(x1 + end * x_length),
                                    round(y1 + end * y_length),
                                ),
                            ],
                            xy,
                            fill,
                            width,
                        )
                    dash_enabled = not dash_enabled
                    postion += dash_step
        return

    def dashed_rectangle(self, xy, dash=(2, 2), outline=None, width=0):
        # xy - Sequence of [(x1, y1), (x2, y2)] where (x1, y1) is top left corner and (x2, y2) is bottom right corner
        x1, y1 = xy[0]
        x2, y2 = xy[1]
        halfwidth1 = math.floor((width - 1) / 2)
        halfwidth2 = math.ceil((width - 1) / 2)
        min_dash_gap = min(dash[1::2])
        end_change1 = halfwidth1 + min_dash_gap + 1
        end_change2 = halfwidth2 + min_dash_gap + 1
        odd_width_change = (width - 1) % 2
        self.dashed_line(
            [(x1 - halfwidth1, y1), (x2 - end_change1, y1)], dash, outline, width
        )
        self.dashed_line(
            [(x2, y1 - halfwidth1), (x2, y2 - end_change1)], dash, outline, width
        )
        self.dashed_line(
            [
                (x2 + halfwidth2, y2 + odd_width_change),
                (x1 + end_change2, y2 + odd_width_change),
            ],
            dash,
            outline,
            width,
        )
        self.dashed_line(
            [
                (x1 + odd_width_change, y2 + halfwidth2),
                (x1 + odd_width_change, y1 + end_change2),
            ],
            dash,
            outline,
            width,
        )
        return
