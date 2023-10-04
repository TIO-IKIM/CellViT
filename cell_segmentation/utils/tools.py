# -*- coding: utf-8 -*-
# Helpful functions Pipeline
#
# Adapted from HoverNet
# HoverNet Network (https://doi.org/10.1016/j.media.2019.101563)
# Code Snippet adapted from HoverNet implementation (https://github.com/vqdang/hover_net)
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


import math
from typing import Tuple

import numpy as np
import scipy
from numba import njit, prange
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from skimage.draw import polygon


def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


@njit
def cropping_center(x, crop_shape, batch=False):
    """Crop an input image at the centre.

    Args:
        x: input array
        crop_shape: dimensions of cropped array

    Returns:
        x: cropped array

    """
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1], ...]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1], ...]
    return x


def remove_small_objects(pred, min_size=64, connectivity=1):
    """Remove connected components smaller than the specified size.

    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided.

    Args:
        pred: input labelled array
        min_size: minimum size of instance in output array
        connectivity: The connectivity defining the neighborhood of a pixel.

    Returns:
        out: output array with instances removed under min_size

    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out


def pair_coordinates(
    setA: np.ndarray, setB: np.ndarray, radius: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the Munkres or Kuhn-Munkres algorithm to find the most optimal
    unique pairing (largest possible match) when pairing points in set B
    against points in set A, using distance as cost function.

    Args:
        setA (np.ndarray): np.array (float32) of size Nx2 contains the of XY coordinate
                    of N different points
        setB (np.ndarray): np.array (float32) of size Nx2 contains the of XY coordinate
                    of N different points
        radius (float): valid area around a point in setA to consider
                a given coordinate in setB a candidate for match

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            pairing: pairing is an array of indices
                where point at index pairing[0] in set A paired with point
                in set B at index pairing[1]
            unparedA: remaining point in set A unpaired
            unparedB: remaining point in set B unpaired
    """
    # * Euclidean distance as the cost matrix
    pair_distance = scipy.spatial.distance.cdist(setA, setB, metric="euclidean")

    # * Munkres pairing with scipy library
    # the algorithm return (row indices, matched column indices)
    # if there is multiple same cost in a row, index of first occurence
    # is return, thus the unique pairing is ensured
    indicesA, paired_indicesB = linear_sum_assignment(pair_distance)

    # extract the paired cost and remove instances
    # outside of designated radius
    pair_cost = pair_distance[indicesA, paired_indicesB]

    pairedA = indicesA[pair_cost <= radius]
    pairedB = paired_indicesB[pair_cost <= radius]

    pairing = np.concatenate([pairedA[:, None], pairedB[:, None]], axis=-1)
    unpairedA = np.delete(np.arange(setA.shape[0]), pairedA)
    unpairedB = np.delete(np.arange(setB.shape[0]), pairedB)

    return pairing, unpairedA, unpairedB


def fix_duplicates(inst_map: np.ndarray) -> np.ndarray:
    """Re-label duplicated instances in an instance labelled mask.

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled mask. Shape (H, W).

    Returns
    -------
        np.ndarray:
            The instance labelled mask without duplicated indices.
            Shape (H, W).
    """
    current_max_id = np.amax(inst_map)
    inst_list = list(np.unique(inst_map))
    if 0 in inst_list:
        inst_list.remove(0)

    for inst_id in inst_list:
        inst = np.array(inst_map == inst_id, np.uint8)
        remapped_ids = ndimage.label(inst)[0]
        remapped_ids[remapped_ids > 1] += current_max_id
        inst_map[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
        current_max_id = np.amax(inst_map)

    return inst_map


def polygons_to_label_coord(
    coord: np.ndarray, shape: Tuple[int, int], labels: np.ndarray = None
) -> np.ndarray:
    """Render polygons to image given a shape.

    Parameters
    ----------
        coord.shape : np.ndarray
            Shape: (n_polys, n_rays)
        shape : Tuple[int, int]
            Shape of the output mask.
        labels : np.ndarray, optional
            Sorted indices of the centroids.

    Returns
    -------
        np.ndarray:
            Instance labelled mask. Shape: (H, W).
    """
    coord = np.asarray(coord)
    if labels is None:
        labels = np.arange(len(coord))

    assert coord.ndim == 3 and coord.shape[1] == 2 and len(coord) == len(labels)

    lbl = np.zeros(shape, np.int32)

    for i, c in zip(labels, coord):
        rr, cc = polygon(*c, shape)
        lbl[rr, cc] = i + 1

    return lbl


def ray_angles(n_rays: int = 32):
    """Get linearly spaced angles for rays."""
    return np.linspace(0, 2 * np.pi, n_rays, endpoint=False)


def dist_to_coord(
    dist: np.ndarray, points: np.ndarray, scale_dist: Tuple[int, int] = (1, 1)
) -> np.ndarray:
    """Convert list of distances and centroids from polar to cartesian coordinates.

    Parameters
    ----------
        dist : np.ndarray
            The centerpoint pixels of the radial distance map. Shape (n_polys, n_rays).
        points : np.ndarray
            The centroids of the instances. Shape: (n_polys, 2).
        scale_dist : Tuple[int, int], default=(1, 1)
            Scaling factor.

    Returns
    -------
        np.ndarray:
            Cartesian cooridnates of the polygons. Shape (n_polys, 2, n_rays).
    """
    dist = np.asarray(dist)
    points = np.asarray(points)
    assert (
        dist.ndim == 2
        and points.ndim == 2
        and len(dist) == len(points)
        and points.shape[1] == 2
        and len(scale_dist) == 2
    )
    n_rays = dist.shape[1]
    phis = ray_angles(n_rays)
    coord = (dist[:, np.newaxis] * np.array([np.sin(phis), np.cos(phis)])).astype(
        np.float32
    )
    coord *= np.asarray(scale_dist).reshape(1, 2, 1)
    coord += points[..., np.newaxis]
    return coord


def polygons_to_label(
    dist: np.ndarray,
    points: np.ndarray,
    shape: Tuple[int, int],
    prob: np.ndarray = None,
    thresh: float = -np.inf,
    scale_dist: Tuple[int, int] = (1, 1),
) -> np.ndarray:
    """Convert distances and center points to instance labelled mask.

    Parameters
    ----------
        dist : np.ndarray
            The centerpoint pixels of the radial distance map. Shape (n_polys, n_rays).
        points : np.ndarray
            The centroids of the instances. Shape: (n_polys, 2).
        shape : Tuple[int, int]:
            Shape of the output mask.
        prob : np.ndarray, optional
            The centerpoint pixels of the regressed distance transform.
            Shape: (n_polys, n_rays).
        thresh : float, default=-np.inf
            Threshold for the regressed distance transform.
        scale_dist : Tuple[int, int], default=(1, 1)
            Scaling factor.

    Returns
    -------
        np.ndarray:
            Instance labelled mask. Shape (H, W).
    """
    dist = np.asarray(dist)
    points = np.asarray(points)
    prob = np.inf * np.ones(len(points)) if prob is None else np.asarray(prob)

    assert dist.ndim == 2 and points.ndim == 2 and len(dist) == len(points)
    assert len(points) == len(prob) and points.shape[1] == 2 and prob.ndim == 1

    ind = prob > thresh
    points = points[ind]
    dist = dist[ind]
    prob = prob[ind]

    ind = np.argsort(prob, kind="stable")
    points = points[ind]
    dist = dist[ind]

    coord = dist_to_coord(dist, points, scale_dist=scale_dist)

    return polygons_to_label_coord(coord, shape=shape, labels=ind)


@njit(cache=True, fastmath=True)
def intersection(boxA: np.ndarray, boxB: np.ndarray):
    """Compute area of intersection of two boxes.

    Parameters
    ----------
        boxA : np.ndarray
            First boxes
        boxB : np.ndarray
            Second box

    Returns
    -------
        float64:
            Area of intersection
    """
    xA = max(boxA[..., 0], boxB[..., 0])
    xB = min(boxA[..., 2], boxB[..., 2])
    dx = xB - xA
    if dx <= 0:
        return 0.0

    yA = max(boxA[..., 1], boxB[..., 1])
    yB = min(boxA[..., 3], boxB[..., 3])
    dy = yB - yA
    if dy <= 0.0:
        return 0.0

    return dx * dy


@njit(parallel=True)
def get_bboxes(
    dist: np.ndarray, points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Get bounding boxes from the non-zero pixels of the radial distance maps.

    This is basically a translation from the stardist repo cpp code to python

    NOTE: jit compiled and parallelized with numba.

    Parameters
    ----------
        dist : np.ndarray
            The non-zero values of the radial distance maps. Shape: (n_nonzero, n_rays).
        points : np.ndarray
            The yx-coordinates of the non-zero points. Shape (n_nonzero, 2).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        Returns the x0, y0, x1, y1 bbox coordinates, bbox areas and the maximum
        radial distance in the image.
    """
    n_polys = dist.shape[0]
    n_rays = dist.shape[1]

    bbox_x1 = np.zeros(n_polys)
    bbox_x2 = np.zeros(n_polys)
    bbox_y1 = np.zeros(n_polys)
    bbox_y2 = np.zeros(n_polys)

    areas = np.zeros(n_polys)
    angle_pi = 2 * math.pi / n_rays
    max_dist = 0

    for i in prange(n_polys):
        max_radius_outer = 0
        py = points[i, 0]
        px = points[i, 1]

        for k in range(n_rays):
            d = dist[i, k]
            y = py + d * np.sin(angle_pi * k)
            x = px + d * np.cos(angle_pi * k)

            if k == 0:
                bbox_x1[i] = x
                bbox_x2[i] = x
                bbox_y1[i] = y
                bbox_y2[i] = y
            else:
                bbox_x1[i] = min(x, bbox_x1[i])
                bbox_x2[i] = max(x, bbox_x2[i])
                bbox_y1[i] = min(y, bbox_y1[i])
                bbox_y2[i] = max(y, bbox_y2[i])

            max_radius_outer = max(d, max_radius_outer)

        areas[i] = (bbox_x2[i] - bbox_x1[i]) * (bbox_y2[i] - bbox_y1[i])
        max_dist = max(max_dist, max_radius_outer)

    return bbox_x1, bbox_y1, bbox_x2, bbox_y2, areas, max_dist
