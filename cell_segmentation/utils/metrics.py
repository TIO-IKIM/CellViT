# -*- coding: utf-8 -*-
# Implemented Metrics for Cell detection
#
# This code is based on the following repository: https://github.com/TissueImageAnalytics/PanNuke-metrics
#
# Implemented metrics are:
#
# Instance Segmentation Metrics
# Binary PQ
# Multiclass PQ
# Neoplastic PQ
# Non-Neoplastic PQ
# Inflammatory PQ
# Dead PQ
# Inflammatory PQ
# Dead PQ
#
# Detection and Classification Metrics
# Precision, Recall, F1
#
# Other
# dice1, dice2, aji, aji_plus
#
# Binary PQ (bPQ): Assumes all nuclei belong to same class and reports the average PQ across tissue types.
# Multi-Class PQ (mPQ): Reports the average PQ across the classes and tissue types.
# Neoplastic PQ: Reports the PQ for the neoplastic class on all tissues.
# Non-Neoplastic PQ: Reports the PQ for the non-neoplastic class on all tissues.
# Inflammatory PQ: Reports the PQ for the inflammatory class on all tissues.
# Connective PQ: Reports the PQ for the connective class on all tissues.
# Dead PQ: Reports the PQ for the dead class on all tissues.
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from typing import List
import numpy as np
from scipy.optimize import linear_sum_assignment


def get_fast_pq(true, pred, match_iou=0.5):
    """
    `match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique
    (1 prediction instance to 1 GT instance mapping).

    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing.

    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.

    Fast computation requires instance IDs are in contiguous orderding
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand
    and `by_size` flag has no effect on the result.

    Returns:
        [dq, sq, pq]: measurement statistic

        [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                      pairing information to perform measurement

    """
    assert match_iou >= 0.0, "Cant' be negative"

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    # if there is no background, fixing by adding it
    if 0 not in pred_id_list:
        pred_id_list = [0] + pred_id_list

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1.0e-6)  # good practice?
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


#####


def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


####


def binarize(x):
    """
    convert multichannel (multiclass) instance segmetation tensor
    to binary instance segmentation (bg and nuclei),

    :param x: B*B*C (for PanNuke 256*256*5 )
    :return: Instance segmentation
    """
    out = np.zeros([x.shape[0], x.shape[1]])
    count = 1
    for i in range(x.shape[2]):
        x_ch = x[:, :, i]
        unique_vals = np.unique(x_ch)
        unique_vals = unique_vals.tolist()
        unique_vals.remove(0)
        for j in unique_vals:
            x_tmp = x_ch == j
            x_tmp_c = 1 - x_tmp
            out *= x_tmp_c
            out += count * x_tmp
            count += 1
    out = out.astype("int32")
    return out


def get_tissue_idx(tissue_indices, idx):
    for i in range(len(tissue_indices)):
        if tissue_indices[i].count(idx) == 1:
            tiss_idx = i
    return tiss_idx


def cell_detection_scores(
    paired_true, paired_pred, unpaired_true, unpaired_pred, w: List = [1, 1]
):
    tp_d = paired_pred.shape[0]
    fp_d = unpaired_pred.shape[0]
    fn_d = unpaired_true.shape[0]

    # tp_tn_dt = (paired_pred == paired_true).sum()
    # fp_fn_dt = (paired_pred != paired_true).sum()
    prec_d = tp_d / (tp_d + fp_d)
    rec_d = tp_d / (tp_d + fn_d)

    f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

    return f1_d, prec_d, rec_d


def cell_type_detection_scores(
    paired_true,
    paired_pred,
    unpaired_true,
    unpaired_pred,
    type_id,
    w: List = [2, 2, 1, 1],
    exhaustive: bool = True,
):
    type_samples = (paired_true == type_id) | (paired_pred == type_id)

    paired_true = paired_true[type_samples]
    paired_pred = paired_pred[type_samples]

    tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
    tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
    fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
    fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

    if not exhaustive:
        ignore = (paired_true == -1).sum()
        fp_dt -= ignore

    fp_d = (unpaired_pred == type_id).sum()  #
    fn_d = (unpaired_true == type_id).sum()

    prec_type = (tp_dt + tn_dt) / (tp_dt + tn_dt + w[0] * fp_dt + w[2] * fp_d)
    rec_type = (tp_dt + tn_dt) / (tp_dt + tn_dt + w[1] * fn_dt + w[3] * fn_d)

    f1_type = (2 * (tp_dt + tn_dt)) / (
        2 * (tp_dt + tn_dt) + w[0] * fp_dt + w[1] * fn_dt + w[2] * fp_d + w[3] * fn_d
    )
    return f1_type, prec_type, rec_type
