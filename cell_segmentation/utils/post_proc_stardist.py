# -*- coding: utf-8 -*-
from typing import Tuple

import cv2
import numpy as np
import torch
from stardist import non_maximum_suppression, polygons_to_label

from cell_segmentation.utils.metrics import remap_label
from cell_segmentation.utils.tools import get_bounding_box


class StarDistPostProcessor:
    """StarDist Postprocessing strategy, based on Non-Maximum-Suppression Algorithm

    Args:
        nr_types (int, optional): Number of nuclei types. Defaults to 6.
        image_shape (Tuple, optional): Image-shape (h, w). Defaults to (256, 256).
    """

    def __init__(self, nr_types: int = 6, image_shape: Tuple = (256, 256)) -> None:
        self.nr_types = nr_types
        self.image_shape = image_shape

    def post_proc_stardist(
        self,
        dist_map: np.ndarray,
        stardist_map: np.ndarray,
        pred_type: np.ndarray,
    ) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """Perform postprocessing based on probability map, stardist_map and pred-type (nuclei type prediction)

        Args:
            dist_map (np.ndarray): Probability distance map of one image. Shape: (H, W)
            stardist_map (np.ndarray): Stardist probabilities of one image. Shape: (n_rays, H, W)
            pred_type (np.ndarray): Nuclei type map probabilities of one image. (num_nuclei_types, H, W)

        Returns:
            Tuple[torch.Tensor, dict, torch.Tensor]:
                * instance-predictions with shape (H, W) and each integer indicating one instance
                * dict with results for this image.
                For each nucleus, the following information are returned: "bbox", "centroid", "contour", "type_prob", "type"
                * nuclei-instance predictions with shape (num_nuclei_types, H, W)
        """

        dists = np.transpose(stardist_map, (1, 2, 0))
        pred_type = np.transpose(pred_type, (1, 2, 0))
        pred_type = np.argmax(pred_type, axis=-1)  # argmax to find type

        points, _, dists = non_maximum_suppression(dists, dist_map)
        binary_star_label = polygons_to_label(dists, points, self.image_shape)
        instance_preds = remap_label(binary_star_label)

        inst_id_list = np.unique(instance_preds)[1:]  # exlcude background
        inst_info_dict = {}
        for inst_id in inst_id_list:
            inst_map = instance_preds == inst_id
            rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            inst_map = inst_map[
                inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
            ]
            inst_map = inst_map.astype(np.uint8)
            inst_moment = cv2.moments(inst_map)
            inst_contour = cv2.findContours(
                inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # * opencv protocol format may break
            inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
            # < 3 points dont make a contour, so skip, likely artifact too
            # as the contours obtained via approximation => too small or sthg
            if inst_contour.shape[0] < 3:
                continue
            if len(inst_contour.shape) != 2:
                continue  # ! check for trickery shape
            inst_centroid = [
                (inst_moment["m10"] / inst_moment["m00"]),
                (inst_moment["m01"] / inst_moment["m00"]),
            ]
            inst_centroid = np.array(inst_centroid)
            inst_contour[:, 0] += inst_bbox[0][1]  # X
            inst_contour[:, 1] += inst_bbox[0][0]  # Y
            inst_centroid[0] += inst_bbox[0][1]  # X
            inst_centroid[1] += inst_bbox[0][0]  # Y
            inst_info_dict[inst_id] = {  # inst_id should start at 1
                "bbox": inst_bbox,
                "centroid": inst_centroid,
                "contour": inst_contour,
                "type_prob": None,
                "type": None,
            }

        #### * Get class of each instance id, stored at index id-1 (inst_id = number of deteced nucleus)
        for inst_id in list(inst_info_dict.keys()):
            rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
            inst_map_crop = instance_preds[rmin:rmax, cmin:cmax]
            inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
            inst_map_crop = inst_map_crop == inst_id
            inst_type = inst_type_crop[inst_map_crop]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0:  # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            type_dict = {v[0]: v[1] for v in type_list}
            type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
            inst_info_dict[inst_id]["type"] = int(inst_type)
            inst_info_dict[inst_id]["type_prob"] = float(type_prob)

        instance_type_nuclei_map = torch.zeros((*self.image_shape, self.nr_types))
        for nuclei, spec in inst_info_dict.items():
            nuclei_type = spec["type"]
            instance_type_nuclei_map[:, :, nuclei_type][
                instance_preds == nuclei
            ] = nuclei

        return (
            torch.Tensor(instance_preds),
            inst_info_dict,
            instance_type_nuclei_map.permute(2, 0, 1),
        )
