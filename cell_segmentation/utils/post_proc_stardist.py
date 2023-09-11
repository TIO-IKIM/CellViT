# -*- coding: utf-8 -*-
import warnings
from typing import List, Sequence

import cv2
import numpy as np
from numba import njit
from scipy.spatial import KDTree
from skimage.morphology import disk, erosion

from .tools import get_bboxes, get_bounding_box, intersection, polygons_to_label


def noop(*args, **kargs):
    pass


# TODO: document everything and write all docstrings

warnings.warn = noop


class StarDistPostProcessor:
    def __init__(
        self,
        nr_types: int = None,
        score_thresh: float = 0.5,
        iou_thresh: float = 0.5,
        trim_bboxes: bool = True,
    ) -> None:
        self.nr_types = nr_types
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.trim_bboxes = trim_bboxes

    def post_proc_stardist(
        self,
        dist_map: np.ndarray,
        stardist_map: np.ndarray,
        pred_type: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Run post-processing for stardist outputs.

        NOTE: This is not the original cpp version.
        This is a python re-implementation of the stardidst post-processing
        pipeline that uses non-maximum-suppression. Here, critical parts of the
        nms are accelerated with `numba` and `scipy.spatial.KDtree`.

        NOTE:
        This implementaiton of the stardist post-processing is actually nearly 2x
        faster than the original version if `trim_bboxes` is set to True. The resulting
        segmentation is not an exact match but the differences are mostly neglible.

        Parameters
        ----------
            dist_map : np.ndarray
                Predicted distance transform. Shape: (H, W).
            stardist_map : np.ndarray
                Predicted radial distances. Shape: (n_rays, H, W).
            thresh : float, default=0.4
                Threshold for the regressed distance transform.
            trim_bboxes : bool, default=True
                If True, The non-zero pixels are computed only from the cell contours
                which prunes down the pixel search space drastically.

        Returns
        -------
            np.ndarray:
                Instance labelled mask. Shape: (H, W).
        """
        if (
            not dist_map.ndim == 2
            and not stardist_map.ndim == 3
            and not dist_map.shape == stardist_map.shape[:2]
        ):
            raise ValueError(
                "Illegal input shapes. Make sure that: "
                f"`dist_map` has to have shape: (H, W). Got: {dist_map.shape} "
                f"`stardist_map` has to have shape (H, W, nrays). Got: {stardist_map.shape}"
            )

        dist = np.asarray(stardist_map).transpose(1, 2, 0)
        prob = np.asarray(dist_map)

        # threshold the edt distance transform map
        mask = self._ind_prob_thresh(prob)

        # get only the mask contours to trim down bbox search space
        if self.trim_bboxes:
            fp = disk(2)
            mask -= erosion(mask, fp)

        points = np.stack(np.where(mask), axis=1)

        # Get only non-zero pixels of the transforms
        dist = dist[mask > 0]
        scores = prob[mask > 0]

        # sort descendingly
        ind = np.argsort(scores)[::-1]
        dist = dist[ind]
        scores = scores[ind]
        points = points[ind]

        # get bounding boxes
        x1, y1, x2, y2, areas, max_dist = get_bboxes(dist, points)
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # consider only boxes above score threshold
        score_cond = scores >= self.score_thresh
        boxes = boxes[score_cond]
        scores = scores[score_cond]
        areas = areas[score_cond]

        # run nms
        inds = self.nms_stardist(
            boxes,
            points,
            scores,
            areas,
            max_dist,
        )

        # get the centroids
        points = points[inds]
        scores = scores[inds]
        dist = dist[inds]
        pred_inst = polygons_to_label(dist, points, prob=scores, shape=dist_map.shape)

        inst_id_list = np.unique(pred_inst)[1:]  # exlcude background
        inst_info_dict = {}

        pred_type = np.argmax(pred_type, axis=0)[..., None]
        pred_type = pred_type.astype(np.int32)

        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id
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
            inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
            inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
            inst_map_crop = (
                inst_map_crop == inst_id
            )  # TODO: duplicated operation, may be expensive
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

        return pred_inst, inst_info_dict

    def nms_stardist(
        self,
        boxes: np.ndarray,
        points: np.ndarray,
        scores: np.ndarray,
        areas: np.ndarray,
        max_dist: float,
    ) -> np.ndarray:
        """Non maximum suppression for stardist bboxes.

        NOTE: This implementation relies on `scipy.spatial` `KDTree`

        NOTE: This version of nms is faster than the original one in stardist repo
        and is fully written in python. The differenecs in the resulting instance
        segmentation masks are neglible.

        Parameters
        ----------
            boxes : np.ndarray
                An array of bbox coords in pascal VOC format (x0, y0, x1, y1).
                Shape: (n_points, 4). Dtype: float64.
            points : np.ndarray
                The yx-coordinates of the non-zero points. Shape (n_points, 2). Dtype: int64
            scores : np.ndarray
                The probability values at the point coordinates. Shape (n_points,).
                Dtype: float32/float64.
            areas : np.ndarray
                The areas of the bounding boxes at the point coordinates. Shape (n_points,).
                Dtype: float32/float64.
            radius_outer : np.ndarray
                The radial distances to background at each point. Shape (n_points, )
            max_dist : float
                The maximum radial distance of all the radial distances
            score_threshold : float, default=0.5
                Threshold for the probability distance map.
            iou_threshold : float, default=0.5
                Threshold for the IoU metric deciding whether to suppres a bbox.

        Returns
        -------
            np.ndarray:
                The indices of the bboxes that are not suppressed. Shape: (n_kept, ).
        """
        keep = []

        if len(boxes) == 0:
            return np.zeros(0, dtype=np.int64)

        kdtree = KDTree(points, leafsize=16)

        suppressed = np.full(len(boxes), False)
        for current_idx in range(len(scores)):
            # If already visited or discarded
            if suppressed[current_idx]:
                continue

            # If score is already below threshold then break
            if scores[current_idx] < self.score_thresh:
                break

            # Query the points
            query = kdtree.query_ball_point(points[current_idx], max_dist)
            suppressed = self._suppress_bbox(
                np.array(query), current_idx, boxes, areas, suppressed, self.iou_thresh
            )

            # Add the current box
            keep.append(current_idx)

        return np.array(keep)

    def _ind_prob_thresh(self, prob: np.ndarray, b: int = 2) -> np.ndarray:
        """Index based thresholding."""
        if b is not None and np.isscalar(b):
            b = ((b, b),) * prob.ndim

        ind_thresh = prob > self.score_thresh
        if b is not None:
            _ind_thresh = np.zeros_like(ind_thresh)
            ss = tuple(
                slice(_bs[0] if _bs[0] > 0 else None, -_bs[1] if _bs[1] > 0 else None)
                for _bs in b
            )
            _ind_thresh[ss] = True
            ind_thresh &= _ind_thresh
        return ind_thresh.astype("int32")

    @staticmethod
    @njit
    def _suppress_bbox(
        query: Sequence[int],
        current_idx: int,
        boxes: np.ndarray,
        areas: np.ndarray,
        suppressed: List[bool],
        iou_thresh: float,
    ) -> np.ndarray:
        """Inner loop of the stardist nms algorithm where bboxes are suppressed.

        NOTE: Numba compiled only for performance.
            Parallelization had only a negative effect on run-time on.
            12-core hyperthreaded Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz.
        """
        for i in range(len(query)):
            query_idx = query[i]

            if suppressed[query_idx]:
                continue

            overlap = intersection(boxes[current_idx], boxes[query_idx])
            iou = overlap / min(areas[current_idx] + 1e-10, areas[query_idx] + 1e-10)
            suppressed[query_idx] = iou > iou_thresh

        return suppressed
