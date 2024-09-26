import numpy as np
from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.metrics import r2_score
import skimage
import scipy
import numpy as np
from typing import Tuple, List

# from misc.utils import remap_label, get_bounding_box

####### CONIC metrics #######


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


def get_bounding_box(img):
    """Get the bounding box coordinates of a binary input- assumes a single object.

    Args:
        img: input binary image.

    Returns:
        bounding box coordinates

    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def get_multi_pq_info(true, pred, nr_classes=6, match_iou=0.5):
    """Get the statistical information needed to compute multi-class PQ.

    CoNIC multiclass PQ is achieved by considering nuclei over all images at the same time,
    rather than averaging image-level results, like was done in MoNuSAC. This overcomes issues
    when a nuclear category is not present in a particular image.

    Args:
        true (ndarray): HxWx2 array. First channel is the instance segmentation map
            and the second channel is the classification map.
        pred: HxWx2 array. First channel is the instance segmentation map
            and the second channel is the classification map.
        nr_classes (int): Number of classes considered in the dataset.
        match_iou (float): IoU threshold for determining whether there is a detection.

    Returns:
        statistical info per class needed to compute PQ.

    """

    assert match_iou >= 0.0, "Cant' be negative"

    true_inst = true[..., 0]
    pred_inst = pred[..., 0]
    ###
    true_class = true[..., 1]
    pred_class = pred[..., 1]

    pq = []
    for idx in range(nr_classes):
        pred_class_tmp = pred_class == idx + 1
        pred_inst_oneclass = pred_inst * pred_class_tmp
        pred_inst_oneclass = skimage.segmentation.relabel_sequential(
            pred_inst_oneclass
        )[0]
        ##
        true_class_tmp = true_class == idx + 1
        true_inst_oneclass = true_inst * true_class_tmp
        true_inst_oneclass = skimage.segmentation.relabel_sequential(
            true_inst_oneclass
        )[0]

        pq_oneclass_info = get_pq(true_inst_oneclass, pred_inst_oneclass)

        # add (in this order) tp, fp, fn iou_sum
        pq_oneclass_stats = [
            pq_oneclass_info[1][0],
            pq_oneclass_info[1][1],
            pq_oneclass_info[1][2],
            pq_oneclass_info[2],
        ]
        pq.append(pq_oneclass_stats)

    return pq


def get_pq(true, pred, match_iou=0.5):  # , remap=True
    """Get the panoptic quality result.

    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` beforehand. Here, the `by_size` flag
    has no effect on the result.

    Args:
        true (ndarray): HxW ground truth instance segmentation map
        pred (ndarray): HxW predicted instance segmentation map
        match_iou (float): IoU threshold level to determine the pairing between
            GT instances `p` and prediction instances `g`. `p` and `g` is a pair
            if IoU > `match_iou`. However, pair of `p` and `g` must be unique
            (1 prediction instance to 1 GT instance mapping). If `match_iou` < 0.5,
            Munkres assignment (solving minimum weight matching in bipartite graphs)
            is caculated to find the maximal amount of unique pairing. If
            `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
            the number of pairs is also maximal.
        remap (bool): whether to ensure contiguous ordering of instances.

    Returns:
        [dq, sq, pq]: measurement statistic

        [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                      pairing information to perform measurement

        paired_iou.sum(): sum of IoU within true positive predictions

    """
    assert match_iou >= 0.0, "Cant' be negative"
    # ensure instance maps are contiguous
    # if remap:
    #     pred = skimage.segmentation.relabel_sequential(pred)[0]
    #     true = skimage.segmentation.relabel_sequential(true)[0]

    true = np.copy(true)
    pred = np.copy(pred)
    true = true.astype("int32")
    pred = pred.astype("int32")
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))
    # prefill with value
    pairwise_iou = np.zeros([len(true_id_list), len(pred_id_list)], dtype=np.float64)

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask_lab = true == true_id
        rmin1, rmax1, cmin1, cmax1 = get_bounding_box(t_mask_lab)
        t_mask_crop = t_mask_lab[rmin1:rmax1, cmin1:cmax1]
        t_mask_crop = t_mask_crop.astype("int")
        p_mask_crop = pred[rmin1:rmax1, cmin1:cmax1]
        pred_true_overlap = p_mask_crop[t_mask_crop > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask_lab = pred == pred_id
            p_mask_lab = p_mask_lab.astype("int")

            # crop region to speed up computation
            rmin2, rmax2, cmin2, cmax2 = get_bounding_box(p_mask_lab)
            rmin = min(rmin1, rmin2)
            rmax = max(rmax1, rmax2)
            cmin = min(cmin1, cmin2)
            cmax = max(cmax1, cmax2)
            t_mask_crop2 = t_mask_lab[rmin:rmax, cmin:cmax]
            p_mask_crop2 = p_mask_lab[rmin:rmax, cmin:cmax]

            total = (t_mask_crop2 + p_mask_crop2).sum()
            inter = (t_mask_crop2 * p_mask_crop2).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou

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
    dq = tp / ((tp + 0.5 * fp + 0.5 * fn) + 1.0e-6)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return (
        [dq, sq, dq * sq],
        [tp, fp, fn],
        paired_iou.sum(),
        [paired_true, unpaired_pred, unpaired_true],
    )


################################
####### HOVERNET metrics #######


def get_fast_aji(true, pred):
    """
    AJI version distributed by MoNuSeg, has no permutation problem but suffered from
    over-penalisation similar to DICE2

    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no
    effect on the result.
    """
    true = np.copy(true)  # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

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
    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise
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
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter

    # to handle cases where true is not empty but pred is
    if pairwise_inter.shape[1] == 0:
        pairwise_inter = np.array(0).reshape((1, 1))
        pairwise_union = (t_mask).sum().reshape((1, 1))

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care
    # about reusing pred instance multiple times
    try:
        paired_pred = np.argmax(pairwise_iou, axis=1)
        pairwise_iou = np.max(pairwise_iou, axis=1)

        # exlude those dont have intersection
        paired_true = np.nonzero(pairwise_iou > 0.0)[0]
        paired_pred = paired_pred[paired_true]
        # print(paired_true.shape, paired_pred.shape)
        overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
        overall_union = (pairwise_union[paired_true, paired_pred]).sum()
        #
        paired_true = list(paired_true + 1)  # index to instance ID
        paired_pred = list(paired_pred + 1)
        # add all unpaired GT and Prediction into the union
        unpaired_true = np.array(
            [idx for idx in true_id_list[1:] if idx not in paired_true]
        )
        unpaired_pred = np.array(
            [idx for idx in pred_id_list[1:] if idx not in paired_pred]
        )
        for true_id in unpaired_true:
            overall_union += true_masks[true_id].sum()
        for pred_id in unpaired_pred:
            overall_union += pred_masks[pred_id].sum()
        #
        aji_score = overall_inter / (overall_union + 1.0e-6)
    except ValueError:
        aji_score = float("NaN")
    return aji_score


def get_fast_aji_plus(true, pred):
    """
    AJI+, an AJI version with maximal unique pairing to obtain overall intersecion.
    Every prediction instance is paired with at most 1 GT instance (1 to 1) mapping, unlike AJI
    where a prediction instance can be paired against many GT instances (1 to many).
    Remaining unpaired GT and Prediction instances will be added to the overall union.
    The 1 to 1 mapping prevents AJI's over-penalisation from happening.

    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no
    effect on the result.
    """
    true = np.copy(true)  # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

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
    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise
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
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter
    #
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    #### Munkres pairing to find maximal unique pairing
    paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
    ### extract the paired cost and remove invalid pair
    paired_iou = pairwise_iou[paired_true, paired_pred]
    # now select all those paired with iou != 0.0 i.e have intersection
    paired_true = paired_true[paired_iou > 0.0]
    paired_pred = paired_pred[paired_iou > 0.0]
    paired_inter = pairwise_inter[paired_true, paired_pred]
    paired_union = pairwise_union[paired_true, paired_pred]
    paired_true = list(paired_true + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    overall_inter = paired_inter.sum()
    overall_union = paired_union.sum()
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()
    #
    aji_score = overall_inter / (overall_union + +1.0e-6)
    return aji_score


def _get_fast_pq_legacy(true, pred, match_iou=0.5):
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
    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1.0e-6)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


def get_fast_dice_2(true, pred):
    """
    Ensemble dice: compute for each nucleus and then average
    """
    true = np.copy(true)
    pred = np.copy(pred)
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))

    overall_total = 0
    overall_inter = 0

    true_masks = [np.zeros(true.shape)]
    for t in true_id[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [np.zeros(true.shape)]
    for p in pred_id[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    for true_idx in range(1, len(true_id)):
        t_mask = true_masks[true_idx]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        try:  # blinly remove background
            pred_true_overlap_id.remove(0)
        except ValueError:
            pass  # just mean no background
        for pred_idx in pred_true_overlap_id:
            p_mask = pred_masks[pred_idx]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            overall_total += total
            overall_inter += inter

    return 2 * overall_inter / overall_total


def get_dice_1(true, pred):
    """
    Traditional dice
    """
    # cast to binary 1st
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    inter = true * pred
    denom = true + pred
    return 2.0 * np.sum(inter) / (np.sum(denom) + 1.0e-6)


def get_dice_2(true, pred):
    true = np.copy(true)
    pred = np.copy(pred)
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))
    # remove background aka id 0
    true_id.remove(0)
    pred_id.remove(0)

    total_markup = 0
    total_intersect = 0
    for t in true_id:
        t_mask = np.array(true == t, np.uint8)
        for p in pred_id:
            p_mask = np.array(pred == p, np.uint8)
            intersect = p_mask * t_mask
            if intersect.sum() > 0:
                total_intersect += intersect.sum()
                total_markup += t_mask.sum() + p_mask.sum()
    return 2 * total_intersect / total_markup


def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.

    Args:
        pred (ndarray): the 2d array contain instances where each instances is marked
            by non-zero integer.
        by_size (bool): renaming such that larger nuclei have a smaller id (on-top).

    Returns:
        new_pred (ndarray): Array with continguous ordering of instances.

    """
    pred_id = list(np.unique(pred))
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


def pair_coordinates(setA, setB, radius):
    """
    Use the Munkres or Kuhn-Munkres algorithm to find the most optimal
    unique pairing (largest possible match) when pairing points in set B
    against points in set A, using distance as cost function

    Args:
        setA, setB: np.array (float32) of size Nx2 contains the of XY coordinate
                    of N different points
        radius: valid area around a point in setA to consider
                a given coordinate in setB a candidate for match
    Return:
        pairing: pairing is an array of indices
        where point at index pairing[0] in set A paired with point
        in set B at index pairing[1]
        unparedA, unpairedB: remaining poitn in set A and set B unpaired
    """

    # * Euclidean distance as the cost matrix
    setA_tile = np.expand_dims(setA, axis=1)
    setB_tile = np.expand_dims(setB, axis=0)
    setA_tile = np.repeat(setA_tile, setB.shape[0], axis=1)
    setB_tile = np.repeat(setB_tile, setA.shape[0], axis=0)
    pair_distance = (setA_tile - setB_tile) ** 2
    # set A is row, and set B is paired against set A
    pair_distance = np.sqrt(np.sum(pair_distance, axis=-1))

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

    unpairedA = [idx for idx in range(setA.shape[0]) if idx not in list(pairedA)]
    unpairedB = [idx for idx in range(setB.shape[0]) if idx not in list(pairedB)]

    pairing = np.array(list(zip(pairedA, pairedB)))
    unpairedA = np.array(unpairedA, dtype=np.int64)
    unpairedB = np.array(unpairedB, dtype=np.int64)

    return pairing, unpairedA, unpairedB


################################
########### extra misc #########


def cell_border_remove(pred, true):
    # remove cells from borders for statistics
    whitesquare = np.zeros(pred.shape).astype(np.uint8)
    whitesquare[0:4, :] = 1
    whitesquare[pred.shape[0] - 4 : pred.shape[0], :] = 1
    whitesquare[:, 0:4] = 1
    whitesquare[:, pred.shape[0] - 4 : pred.shape[0]] = 1
    # reconst = reconstruction(whitesquare*(pred>0), pred>0,'dilation')

    reconst = np.zeros(pred.shape)

    inst_list = list(np.unique(pred))
    inst_list.remove(0)  # 0 is background

    for inst_id in inst_list:
        inst_map = np.array(pred == inst_id, np.uint8)
        if np.sum(whitesquare * inst_map) > 0:
            reconst = reconst + inst_map

    prednobo = pred * (1 - reconst)

    reconst = np.zeros(true.shape)

    inst_list = list(np.unique(true))
    inst_list.remove(0)  # 0 is background

    for inst_id in inst_list:
        inst_map = np.array(true == inst_id, np.uint8)
        if np.sum(whitesquare * inst_map) > 0:
            reconst = reconst + inst_map

    truenobo = true * (1 - reconst)

    return prednobo, truenobo

