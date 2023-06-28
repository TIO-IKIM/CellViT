# -*- coding: utf-8 -*-
# Prepare MoNuSeg Dataset By converting and resorting files
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
from cell_segmentation.utils.metrics import remap_label


def process_fold(fold, input_path, output_path) -> None:
    fold_path = Path(input_path) / f"fold{fold}"
    output_fold_path = Path(output_path) / f"fold{fold}"
    output_fold_path.mkdir(exist_ok=True, parents=True)
    (output_fold_path / "images").mkdir(exist_ok=True, parents=True)
    (output_fold_path / "labels").mkdir(exist_ok=True, parents=True)

    print(f"Fold: {fold}")
    print("Loading large numpy files, this may take a while")
    images = np.load(fold_path / "images.npy")
    masks = np.load(fold_path / "masks.npy")

    print("Process images")
    for i in tqdm(range(len(images)), total=len(images)):
        outname = f"{fold}_{i}.png"
        out_img = images[i]
        im = Image.fromarray(out_img.astype(np.uint8))
        im.save(output_fold_path / "images" / outname)

    print("Process masks")
    for i in tqdm(range(len(images)), total=len(images)):
        outname = f"{fold}_{i}.npy"

        # need to create instance map and type map with shape 256x256
        mask = masks[i]
        inst_map = np.zeros((256, 256))
        num_nuc = 0
        for j in range(5):
            # copy value from new array if value is not equal 0
            layer_res = remap_label(mask[:, :, j])
            # inst_map = np.where(mask[:,:,j] != 0, mask[:,:,j], inst_map)
            inst_map = np.where(layer_res != 0, layer_res + num_nuc, inst_map)
            num_nuc = num_nuc + np.max(layer_res)
        inst_map = remap_label(inst_map)

        type_map = np.zeros((256, 256)).astype(np.int32)
        for j in range(5):
            layer_res = ((j + 1) * np.clip(mask[:, :, j], 0, 1)).astype(np.int32)
            type_map = np.where(layer_res != 0, layer_res, type_map)

        outdict = {"inst_map": inst_map, "type_map": type_map}
        np.save(output_fold_path / "labels" / outname, outdict)


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Perform CellViT inference for given run-directory with model checkpoints and logs",
)
parser.add_argument(
    "--input_path",
    type=str,
    help="Input path of the original PanNuke dataset",
    required=True,
)
parser.add_argument(
    "--output_path",
    type=str,
    help="Output path to store the processed PanNuke dataset",
    required=True,
)

if __name__ == "__main__":
    opt = parser.parse_args()
    configuration = vars(opt)

    input_path = Path(configuration["input_path"])
    output_path = Path(configuration["output_path"])

    for fold in [0, 1, 2]:
        process_fold(fold, input_path, output_path)
