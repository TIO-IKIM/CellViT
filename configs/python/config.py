# -*- coding: utf-8 -*-
# Internal Config
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from typing import List

WSI_EXT: List[str] = [
    "svs",
    "tiff",
    "tif",
    "bif",
    "scn",
    "ndpi",
    "vms",
    "vmu",
]  # mirax not tested yet
ANNOTATION_EXT: List[str] = ["json"]
LOGGING_EXT: List[str] = ["critical", "error", "warning", "info", "debug"]
BACKBONES: List[str] = ["ResNet50", "ResNet50Bottleneck", "ResNet18", "ResNet34"]

# Currently: 30 Colors
COLOR_DEFINITIONS: dict[int, tuple[int]] = {
    0: (239, 71, 111),
    1: (255, 209, 102),
    2: (6, 214, 160),
    3: (7, 59, 76),
    4: (255, 190, 11),
    5: (251, 86, 7),
    6: (255, 0, 110),
    7: (131, 56, 236),
    8: (58, 134, 255),
    9: (249, 65, 68),
    10: (243, 114, 44),
    11: (248, 150, 30),
    12: (249, 132, 74),
    13: (249, 199, 79),
    14: (144, 190, 109),
    15: (67, 170, 139),
    16: (77, 144, 142),
    17: (87, 117, 144),
    18: (39, 125, 161),
    19: (116, 0, 184),
    20: (105, 48, 195),
    21: (128, 255, 219),
    22: (166, 138, 100),
    23: (65, 72, 51),
    24: (51, 61, 41),
    25: (60, 22, 66),
    26: (8, 99, 117),
    27: (29, 211, 176),
    28: (175, 252, 65),
    29: (178, 255, 158),
    30: (17, 138, 178),
}
