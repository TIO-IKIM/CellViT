# -*- coding: utf-8 -*-
# Prepare MoNuSeg Dataset By converting and resorting files
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from PIL import Image
import xml.etree.ElementTree as ET
from skimage import draw
import numpy as np
from pathlib import Path
from typing import Union
import argparse


def convert_monuseg(
    input_path: Union[Path, str], output_path: Union[Path, str]
) -> None:
    """Convert the MoNuSeg dataset to a new format (1000 -> 1024, tiff to png and xml to npy)

    Args:
        input_path (Union[Path, str]): Input dataset
        output_path (Union[Path, str]): Output path
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # testing and training
    parts = ["testing", "training"]
    for part in parts:
        print(f"Prepare: {part}")
        input_path_part = input_path / part
        output_path_part = output_path / part
        output_path_part.mkdir(exist_ok=True, parents=True)
        (output_path_part / "images").mkdir(exist_ok=True, parents=True)
        (output_path_part / "labels").mkdir(exist_ok=True, parents=True)

        # images
        images = [f for f in sorted((input_path_part / "images").glob("*.tif"))]
        for img_path in images:
            loaded_image = Image.open(img_path)
            resized = loaded_image.resize(
                (1024, 1024), resample=Image.Resampling.LANCZOS
            )
            new_img_path = output_path_part / "images" / f"{img_path.stem}.png"
            resized.save(new_img_path)
        # masks
        annotations = [f for f in sorted((input_path_part / "labels").glob("*.xml"))]
        for annot_path in annotations:
            binary_mask = np.transpose(np.zeros((1000, 1000)))

            # extract xml file
            tree = ET.parse(annot_path)
            root = tree.getroot()
            child = root[0]

            for x in child:
                r = x.tag
                if r == "Regions":
                    element_idx = 1
                    for y in x:
                        y_tag = y.tag

                        if y_tag == "Region":
                            regions = []
                            vertices = y[1]
                            coords = np.zeros((len(vertices), 2))
                            for i, vertex in enumerate(vertices):
                                coords[i][0] = vertex.attrib["X"]
                                coords[i][1] = vertex.attrib["Y"]
                            regions.append(coords)
                            vertex_row_coords = regions[0][:, 0]
                            vertex_col_coords = regions[0][:, 1]
                            fill_row_coords, fill_col_coords = draw.polygon(
                                vertex_col_coords, vertex_row_coords, binary_mask.shape
                            )
                            binary_mask[fill_row_coords, fill_col_coords] = element_idx

                            element_idx = element_idx + 1
            inst_image = Image.fromarray(binary_mask)
            resized_mask = np.array(
                inst_image.resize((1024, 1024), resample=Image.Resampling.NEAREST)
            )
            new_mask_path = output_path_part / "labels" / f"{annot_path.stem}.npy"
            np.save(new_mask_path, resized_mask)
    print("Finished")


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Convert the MoNuSeg dataset",
)
parser.add_argument(
    "--input_path",
    type=str,
    help="Input path of the original MoNuSeg dataset",
    required=True,
)
parser.add_argument(
    "--output_path",
    type=str,
    help="Output path to store the processed MoNuSeg dataset",
    required=True,
)

if __name__ == "__main__":
    opt = parser.parse_args()
    configuration = vars(opt)

    input_path = Path(configuration["input_path"])
    output_path = Path(configuration["output_path"])

    convert_monuseg(input_path=input_path, output_path=output_path)
