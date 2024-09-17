import gc
import numpy as np
import cv2
import skimage
from skimage import morphology as morph
from skimage.morphology import (
    erosion,
    disk,
    dilation,
    reconstruction,
    skeletonize,
    square,
)
from skimage.segmentation import watershed
from skimage.measure import label
from scipy import ndimage
from cisca.imutils import get_sobel_kernel


def extract_counts(cell_instances, multiclass=False, cell_type_pred=None, classes=None):
    """
    Extract counts and cell type information from cell instances.

    Parameters:
    - cell_instances (ndarray): Labeled array of cell instances.
    - multiclass (bool): Whether to perform multiclass cell type extraction.
    - cell_type_pred (ndarray): Predicted cell types.
    - classes (iterable): List of classes for multiclass extraction.

    Returns:
    - tuple: Contains processed cell instances, cell count, and optionally cell type information.
    """

    def count_classes(y, classes):
        return tuple(np.sum(np.array(list(y.values())) == i) for i in classes)

    n_cells = len(np.unique(cell_instances)) - 1  # removing background label
    # cls = dict(zip(range(1, n_cells + 1), res["class_id"]))

    if multiclass:
        cell_type_dict = {}
        cell_type = np.zeros(cell_instances.shape, np.uint16)

        cell_type_instances = np.squeeze(cell_type_pred) * (
            np.expand_dims(cell_instances > 0, -1)
        )
        for r in skimage.measure.regionprops(cell_instances):
            m = cell_instances[r.slice] == r.label
            counts = np.sum(cell_type_instances[r.slice][m], 0)
            class_id = np.argmax(counts)
            if class_id == 0:
                class_id = np.argmax(counts[1:]) + 1
            cell_type[r.slice][m] = class_id
            cell_type_dict[r.label] = class_id

        class_count = count_classes(cell_type_dict, classes=classes)

        return cell_instances, n_cells, cell_type, cell_type_dict, class_count

    else:
        return cell_instances, n_cells


def postprocess(
    y_pred,
    watershed_line=False,
    multiclass=False,
    th_lambda=1,
    erosion_rad=[2],
    min_resid_size=3,
    magnification="40x",
    th=0.57,
    small_objects_threshold=None,
    distmapweight=0.5,
    with_counts=True,
):
    """
    Post-process the neural network output to get the final segmentation.

    Parameters:
    - y_pred (ndarray or list): Output of the network.
    - mode (str): Postprocessing modality to use.
    - watershed_line (bool): Whether to use watershed line.
    - multiclass (bool): Whether to perform multiclass postprocessing.
    - th_lambda (float): Threshold lambda value.
    - erosion_rad (list): List of erosion radii.
    - min_resid_size (int): Minimum residue size.
    - magnification (str): Resolution of the input image.
    - th (float): Threshold value.
    - small_objects_threshold (int): Threshold for small object removal.
    - distmapweight (float): Weight for distance map.

    Returns:
    - tuple: Processed cell instances and counts.
    """

    if small_objects_threshold is None:
        small_objects_threshold = 80 if magnification == "40x" else 24
    struct_rad = 5 if magnification == "40x" else 3
    erosion_rad = (
        erosion_rad if magnification == "40x" else [1]
    )  
    min_resid_size = min_resid_size if magnification == "40x" else 2
    mask_strel = square(3) if magnification == "40x" else disk(1)
    
    cell_instance_pred = y_pred[0] * 255
    dist_map_pred = y_pred[1]
    n_contour_classes = cell_instance_pred.shape[-1]
    n_grads = dist_map_pred.shape[-1]
    diag = n_grads == 4

    if n_contour_classes == 2:  # only fg/bg
        foreground_pred = np.squeeze(cell_instance_pred[..., 0] / 255)
    else:  #  contour/body/bg or contour/body/bg/gap
        foreground_pred = np.squeeze(
            cell_instance_pred[..., 0] / 255 + cell_instance_pred[..., 1] / 255
        )

    h_dir_raw = np.squeeze(dist_map_pred[..., 0])
    v_dir_raw = np.squeeze(dist_map_pred[..., 1])
    if diag:
        tl_dir_raw = np.squeeze(dist_map_pred[..., 2])
        bl_dir_raw = np.squeeze(dist_map_pred[..., 3])

    # print('**************gradient processing***************')

    foreground_pred_th = np.copy(foreground_pred)
    foreground_pred_th[foreground_pred_th >= 0.5] = 1
    foreground_pred_th[foreground_pred_th < 0.5] = 0

    foreground_pred_th = label(foreground_pred_th)
    foreground_pred_th[foreground_pred_th > 0] = 1  # back ground is 0 already

    h_dir = cv2.normalize(
        h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_dir = cv2.normalize(
        v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    if diag:  # also diagonal gradients are present
        tl_dir = cv2.normalize(
            tl_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
        bl_dir = cv2.normalize(
            bl_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

    if diag:
        mh, mv, mtl, mbl = get_sobel_kernel(5, diag=True)
    else:
        mh, mv = get_sobel_kernel(5, diag=False)

    sobelh = cv2.filter2D(h_dir, cv2.CV_32F, mh, anchor=(-1, -1))
    sobelv = cv2.filter2D(v_dir, cv2.CV_32F, mv, anchor=(-1, -1))
    sobelh = 1 - (
        cv2.normalize(
            sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )
    sobelv = 1 - (
        cv2.normalize(
            sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )

    if diag:
        sobeltl = cv2.filter2D(tl_dir, cv2.CV_32F, mtl, anchor=(-1, -1))
        sobelbl = cv2.filter2D(bl_dir, cv2.CV_32F, mbl, anchor=(-1, -1))
        sobeltl = 1 - (
            cv2.normalize(
                sobeltl,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )
        sobelbl = 1 - (
            cv2.normalize(
                sobelbl,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )
        contour_map = np.amax(np.stack((sobelh, sobelv, sobeltl, sobelbl), -1), -1)
    else:
        contour_map = np.amax(np.stack((sobelh, sobelv), -1), -1)

    del sobelh
    del sobelv
    del h_dir
    del v_dir
    del h_dir_raw
    del v_dir_raw

    if diag:
        del sobeltl
        del sobelbl
        del tl_dir
        del bl_dir
        del tl_dir_raw
        del bl_dir_raw

    gc.collect()

    contour_map = contour_map - (1 - foreground_pred_th)
    contour_map[contour_map < 0] = 0

    th = th

    contour_map_th = contour_map.copy()
    contour_map_th[contour_map >= th_lambda * th] = 1
    contour_map_th[contour_map < th_lambda * th] = 0

    dist = (1.0 - contour_map_th) * foreground_pred_th
    ## nuclei values form mountains so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    # print('**************erosion and reconstruction***************')
    cell_instance_pred_prob = cell_instance_pred / 255

    pred_msk = np.zeros_like(cell_instance_pred[..., 0], dtype="uint16")

    if n_contour_classes == 2:  # only fg in first channel
        pred_msk = np.where(
            (cell_instance_pred[..., 0] > cell_instance_pred[..., 1]), 1, pred_msk
        )
    else:  # fg > contour and bg
        pred_msk = np.where(
            (cell_instance_pred[..., 1] > cell_instance_pred[..., 0])
            & (cell_instance_pred[..., 1] > cell_instance_pred[..., 2]),
            1,
            pred_msk,
        )
    # pred_msk = np.where((cell_instance_pred[..., 1] > cell_instance_pred[..., 2]) | (cell_instance_pred[..., 0] > cell_instance_pred[..., 2]), 1, pred_msk)

    pred_msk = pred_msk.astype(np.uint16)
    pred_msk = np.squeeze(pred_msk)

    # print("Setting borders detected with gradients to ZERO")
    pred_msk[
        (
            dilation(skeletonize(contour_map_th), square(struct_rad)).astype(int)
            * contour_map_th.astype(int)
        )
        == 1
    ] = 0

    labelpred_msk = label(pred_msk)
    labelpred_msk = morph.remove_small_objects(
        labelpred_msk, min_size=int(small_objects_threshold / 4), connectivity=1
    )

    pred_msk = (labelpred_msk > 0).astype(np.uint16)
    del labelpred_msk
    gc.collect()

    comb_mask = pred_msk
    comb_mask[comb_mask < 0] = 0

    cell_instance_pred_prob = np.squeeze(cell_instance_pred_prob)

    # erosion and geodesic reconstruction (limited amount of iterations to prevent oversegmentation)
    # t = t0 = timeit.default_timer()
    nb_iter = 0
    img = comb_mask
    img_niter = np.zeros_like(comb_mask, dtype="uint16")

    if erosion_rad[-1] > 0:
        radius = erosion_rad
        num_iterations = len(radius)

        for num_iteration in np.arange(num_iterations):
            # print(nb_iter)
            r = radius[num_iteration]
            # print(r)
            img_ero = erosion(img, disk(r))
            nb_iter = nb_iter + 1
            reconst = reconstruction(img_ero, img, "dilation")
            residues = img - reconst
            img_niter = np.where(residues == 1, nb_iter, img_niter)
            img = img_ero

        img_niter = np.where(img > 0, nb_iter, img_niter)

    if erosion_rad[-1] > 0:
        # print('Processing time: {:.3f} s'.format(timeit.default_timer() - t))

        # residues relabel
        # print('**************relabel residues***************')
        # t = timeit.default_timer()
        img_residue = np.copy(img_niter)
        img_residue[img_residue > 0] = 1
        # changed to 2 to prevent merging after erosion
        img_residue = dilation(img_residue, disk(1))
        img_residue = label(img_residue, connectivity=2, background=0)
        img_residue = erosion(img_residue, disk(1))

        # print('Processing time: {:.3f} s'.format(timeit.default_timer() - t))

        # dynamic reconstruction
        # print('**************dynamic reconstruction***************')
        # t = timeit.default_timer()
        img_rc = np.zeros_like(img_residue, dtype="uint16")
        i = np.max(img_niter)
        # print("i:", i)

        img_rc = np.where(img_niter == i, img_residue, img_rc)
    else:
        img_rc = label(comb_mask, connectivity=2, background=0)

    # print('Processing time: {:.3f} s'.format(timeit.default_timer() - t))

    # apply watershed
    # print('**************watershed transform***************')
    # t = timeit.default_timer()
    nucl_msk = 255 - cell_instance_pred_prob[..., 1] * 255
    mask = np.zeros_like(cell_instance_pred_prob[..., 1], dtype="uint16")

    if n_contour_classes == 2:  # only fg/bg
        mask = np.where(
            cell_instance_pred_prob[..., 0] > cell_instance_pred_prob[..., 1], 1, mask
        )
    else:
        mask = np.where(
            (
                (cell_instance_pred_prob[..., 0] + cell_instance_pred_prob[..., 1])
                > cell_instance_pred_prob[..., 2]
            ),
            1,
            mask,
        )

    img_rc = morph.remove_small_objects(img_rc, min_size=min_resid_size, connectivity=1)
    # mask = ndimage.binary_fill_holes(mask).astype(int)
    mask = (1 - (skimage.morphology.remove_small_objects(1 - mask, min_size=5))).astype(
        int
    )
    mask = ndimage.binary_opening(mask, structure=mask_strel).astype(int)
    # mask = ndimage.binary_opening(mask, structure=disk(1)).astype(int)
    # mask = ndimage.binary_closing(mask, structure=disk(1)).astype(int)
    # mask = ndimage.binary_opening(mask, structure=square(3)).astype(int)
    # mask = ndimage.binary_closing(mask, structure=disk(disk_rad)).astype(int)
    # mask = ndimage.binary_opening(mask, structure=square(3)).astype(int)
    # mask = ndimage.binary_opening(mask, structure=disk(disk_rad)).astype(int)

    nucl_msk = (1 - distmapweight) * nucl_msk + (distmapweight) * 255 * (dist + 1)

    img_rc = label(img_rc, connectivity=2, background=0)
    cell_instances = watershed(
        nucl_msk, img_rc, mask=mask, watershed_line=watershed_line, connectivity=8
    )
    cell_instances = morph.remove_small_objects(
        cell_instances, min_size=small_objects_threshold, connectivity=1
    ) * (
        morph.remove_small_objects(
            cell_instances > 0, min_size=small_objects_threshold, connectivity=1
        )
    )
    cell_instances = skimage.segmentation.relabel_sequential(cell_instances)[0]

    if with_counts:
        return extract_counts(
            cell_instances,
            multiclass,
            y_pred[-1],
            classes=range(1, y_pred[-1].shape[-1]),
        )
    else:
        if multiclass:
            return cell_instances, y_pred[-1]
        else:
            return cell_instances
