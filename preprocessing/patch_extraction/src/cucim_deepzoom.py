# -*- coding: utf-8 -*-
# Support for Deep Zoom images.
#
# This module provides functionality for generating Deep Zoom images from
# CuImage objects
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import numpy as np
from cucim import CuImage
from cucim.clara.cache import preferred_memory_capacity
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image


class DeepZoomGeneratorCucim(DeepZoomGenerator):
    """Create a DeepZoomGenerator, but instead of utilizing OpenSlide,
    use cucim to read regions.

    Args:
        osr (OpenSlide): OpenSlide Image. Needed for OS compatibility and for retrieving metadata.
        cucim_slide (CuImage): CuImage slide. Used for retrieving image data.
        tile_size (int, optional): the width and height of a single tile.  For best viewer
                      performance, tile_size + 2 * overlap should be a power
                      of two.. Defaults to 254.
        overlap (int, optional): the number of extra pixels to add to each interior edge
                      of a tile. Defaults to 1.
        limit_bounds (bool, optional): True to render only the non-empty slide region. Defaults to False.
    """

    def __init__(
        self,
        osr: OpenSlide,
        cucim_slide: CuImage,
        tile_size: int = 254,
        overlap: int = 1,
        limit_bounds=False,
    ):
        super().__init__(osr, tile_size, overlap, limit_bounds)

        self._cucim_slide = cucim_slide
        self.memory_capacity = preferred_memory_capacity(
            self._cucim_slide, patch_size=(tile_size, tile_size)
        )
        self.cache = CuImage.cache(
            "per_process", memory_capacity=self.memory_capacity, record_stat=True
        )

    def get_tile(self, level: int, address: tuple[int]) -> Image:
        """Return an RGB PIL.Image for a tile

        Args:
            level (int): the Deep Zoom level
            address (tuple(int)): the address of the tile within the level as a (col, row)
                   tuple

        Returns:
            Image: PIL Image
        """
        args, z_size = self._get_tile_info(level, address)

        tile = self._cucim_slide.read_region(
            location=args[0],
            level=args[1],
            size=args[2],
        )
        tile = Image.fromarray(np.array(tile), mode="RGB")  # CuImage is RGB

        # Scale to the correct size
        if tile.size != z_size:
            # Image.Resampling added in Pillow 9.1.0
            # Image.LANCZOS removed in Pillow 10
            tile.thumbnail(z_size, getattr(Image, "Resampling", Image).LANCZOS)

        return tile
