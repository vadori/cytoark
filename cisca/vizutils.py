import copy
import colorsys
import random
import cv2
import os
import warnings
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
from skimage.segmentation import find_boundaries
from skimage.color import label2rgb
from skimage.measure import label
import skimage
from .config import DATASETMETA

VIZ_PARAM = DATASETMETA

def make_random_color_labels(input_data, predictions):
    RGB_data = np.zeros(input_data.shape[:3] + (3,), dtype="float32")
    for img_id in range(input_data.shape[0]):
        nlabel = len(np.unique(predictions[img_id]))
        rgb_image = label2rgb(
            predictions,
            colors=np.random.random((nlabel, 3)),
            image=input_data[img_id],
            saturation=0,
            alpha=0.7,
        )  # , bg_label=0
        RGB_data[img_id, :, :, :] = rgb_image
    return rgb_image


def make_color_overlay(input_data):
    """Create a color overlay from 2 channel image data

    Args:
        input_data: stack of input images

    Returns:
        numpy.array: color-adjusted stack of overlays in RGB mode
    """
    RGB_data = np.zeros(input_data.shape[:3] + (3,), dtype="float32")

    # rescale channels to aid plotting
    for img in range(input_data.shape[0]):
        for channel in range(input_data.shape[-1]):
            # get histogram for non-zero pixels
            percentiles = np.percentile(
                input_data[img, :, :, channel][input_data[img, :, :, channel] > 0],
                [5, 95],
            )
            rescaled_intensity = rescale_intensity(
                input_data[img, :, :, channel],
                in_range=(percentiles[0], percentiles[1]),
                out_range="float32",
            )
            RGB_data[img, :, :, channel + 2] = rescaled_intensity

    # create a blank array for red channel
    return RGB_data


def make_outline_overlay(RGB_data, predictions):
    boundaries = np.zeros_like(predictions)
    overlay_data = copy.copy(RGB_data)

    for img in range(predictions.shape[0]):
        # boundary = binary_dilation(find_boundaries(predictions[img, :, :], connectivity=1, mode='inner'), disk(1))
        boundary = find_boundaries(predictions[img, :, :], connectivity=1, mode="inner")
        boundaries[img, boundary > 0] = 1
        # print(boundaries.shape)

    overlay_data[boundaries > 0, :] = 0
    # overlay_data[boundaries <= 0, :] = 255
    # print(overlay_data.shape)

    return overlay_data

# from https://github.com/stardist/stardist/tree/master/stardist/plot/plot.py
def random_label_cmap(n=2**16, h=(0, 1), l=(0.4, 1), s=(0.2, 0.8)):
    import matplotlib
    import colorsys

    # cols = np.random.rand(n,3)
    # cols = np.random.uniform(0.1,1.0,(n,3))
    h, l, s = (
        np.random.uniform(*h, n),
        np.random.uniform(*l, n),
        np.random.uniform(*s, n),
    )
    cols = np.stack(
        [colorsys.hls_to_rgb(_h, _l, _s) for _h, _l, _s in zip(h, l, s)], axis=0
    )
    cols[0] = 0
    return matplotlib.colors.ListedColormap(cols)

# from https://github.com/stardist/stardist/tree/master/stardist/plot/render.py
def _single_color_integer_cmap(color=(0.3, 0.4, 0.5)):
    from matplotlib.colors import Colormap

    assert len(color) in (3, 4)

    class BinaryMap(Colormap):
        def __init__(self, color):
            self.color = np.array(color)
            if len(self.color) == 3:
                self.color = np.concatenate([self.color, [1]])

        def __call__(self, X, alpha=None, bytes=False):
            res = np.zeros(X.shape + (4,), np.float32)
            res[..., -1] = self.color[-1]
            res[X > 0] = np.expand_dims(self.color, 0)
            if bytes:
                return np.clip(256 * res, 0, 255).astype(np.uint8)
            else:
                return res

    return BinaryMap(color)


# from https://github.com/CSBDeep/CSBDeep/blob/master/csbdeep/utils/utils.py
def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    try:
        import numexpr

        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)

    return x


def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


# from https://github.com/stardist/stardist/tree/master/stardist/plot/render.py
def render_label(
    lbl,
    img=None,
    cmap=None,
    cmap_img="gray",
    alpha=0.5,
    alpha_boundary=None,
    normalize_img=True,
):
    """Renders a label image and optionally overlays it with another image. Used for generating simple output images to asses the label quality

    Parameters
    ----------
    lbl: np.ndarray of dtype np.uint16
        The 2D label image
    img: np.ndarray
        The array to overlay the label image with (optional)
    cmap: string, tuple, or callable
        The label colormap. If given as rgb(a)  only a single color is used, if None uses a random colormap
    cmap_img: string or callable
        The colormap of img (optional)
    alpha: float
        The alpha value of the overlay. Set alpha=1 to get fully opaque labels
    alpha_boundary: float
        The alpha value of the boundary (if None, use the same as for labels, i.e. no boundaries are visible)
    normalize_img: bool
        If True, normalizes the img (if given)

    Returns
    -------
    img: np.ndarray
        the (m,n,4) RGBA image of the rendered label

    Example
    -------

    from scipy.ndimage import label, zoom
    img = zoom(np.random.uniform(0,1,(16,16)),(8,8),order=3)
    lbl,_ = label(img>.8)
    u1 = render_label(lbl, img = img, alpha = .7)
    u2 = render_label(lbl, img = img, alpha = 0, alpha_boundary =.8)
    plt.subplot(1,2,1);plt.imshow(u1)
    plt.subplot(1,2,2);plt.imshow(u2)

    """
    from skimage.segmentation import find_boundaries
    from matplotlib import cm

    alpha = np.clip(alpha, 0, 1)

    if alpha_boundary is None:
        alpha_boundary = alpha

    if cmap is None:
        cmap = random_label_cmap()
    elif isinstance(cmap, tuple):
        cmap = _single_color_integer_cmap(cmap)
    else:
        pass

    cmap = cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
    cmap_img = cm.get_cmap(cmap_img) if isinstance(cmap_img, str) else cmap_img

    # render image if given
    if img is None:
        im_img = np.zeros(lbl.shape + (4,), np.float32)
        im_img[..., -1] = 1

    else:
        assert lbl.shape[:2] == img.shape[:2]
        img = normalize(img) if normalize_img else img
        if img.ndim == 2:
            im_img = cmap_img(img)
        elif img.ndim == 3:
            im_img = img[..., :4]
            if img.shape[-1] < 4:
                im_img = np.concatenate(
                    [img, np.ones(img.shape[:2] + (4 - img.shape[-1],))], axis=-1
                )
        else:
            raise ValueError("img should be 2 or 3 dimensional")

    # render label
    im_lbl = cmap(lbl)

    mask_lbl = lbl > 0
    mask_bound = np.bitwise_and(mask_lbl, find_boundaries(lbl, mode="thick"))

    # blend
    im = im_img.copy()

    im[mask_lbl] = alpha * im_lbl[mask_lbl] + (1 - alpha) * im_img[mask_lbl]
    im[mask_bound] = (
        alpha_boundary * im_lbl[mask_bound] + (1 - alpha_boundary) * im_img[mask_bound]
    )

    return im


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


def random_colors(N, bright=True):
    """Generate random colors.

    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def visualize_instances_map(
    input_image, inst_map, cell_type_dict=None, type_colour=None, line_thickness=2
):
    """Generates base image with cells contours overlayed. Colors are random unless cell_type_dict is not None

    Args:
        input_image (ndarray): base image
        inst_map (ndarray): instance labelled map with integers values representing cells
        cell_type_dict (dict, optional): dictionary with keys given by cell labels according to inst_map and values given by cell class index. Defaults to None.
        type_colour (dict, optional): _description_. Defaults to None.
        line_thickness (int, optional): _description_. Defaults to 2.

    Returns:
        ndarray: base image with cells contours overlayed
    """

    if cell_type_dict is not None and type_colour is None:
        raise ValueError("cell_type_dict is not None while type_colour is None")
    else:
        overlay = np.copy((input_image).astype(np.uint8))

        inst_list = list(np.unique(inst_map))  # get list of instances
        inst_list.remove(0)  # remove background

        if len(inst_list) < 10:
            inst_rng_colors = np.array(inst_list).astype(np.uint8)
        else:
            inst_rng_colors = random_colors(len(inst_list))
            inst_rng_colors = np.array(inst_rng_colors) * 255
            inst_rng_colors = inst_rng_colors.astype(np.uint8)

        for inst_idx, inst_id in enumerate(inst_list):
            inst_map_mask = np.array(inst_map == inst_id, np.uint8)  # get single object
            y1, y2, x1, x2 = get_bounding_box(inst_map_mask)
            y1 = y1 - 2 if y1 - 2 >= 0 else y1
            x1 = x1 - 2 if x1 - 2 >= 0 else x1
            x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
            y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2
            inst_map_crop = inst_map_mask[y1:y2, x1:x2]
            if np.sum(inst_map_crop) > 6:
                contours_crop = cv2.findContours(
                    inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                # only has 1 instance per map, no need to check #contour detected by opencv
                contours_crop = np.squeeze(
                    contours_crop[0][0].astype("int32")
                )  # * opencv protocol format may break
                try:
                    contours_crop += np.asarray([[x1, y1]])  # index correction
                    if cell_type_dict is not None:
                        type_id = cell_type_dict[inst_id]
                        inst_colour = type_colour[type_id]
                    else:
                        inst_colour = (inst_rng_colors[inst_idx]).tolist()
                    cv2.drawContours(
                        overlay, [contours_crop], -1, inst_colour, line_thickness
                    )
                except ValueError:
                    print(
                        "Skipping id {} due to multiple connected components associated with this ID, probably due to cutting".format(
                            inst_id
                        )
                    )
        return overlay


def plotandsave(
    input_images,
    imageid,
    cell_instances,
    cell_type,
    cell_type_dict,
    predict_folder,
    foldername="newdata",
    GTpath=None,
    viz_param_key="Conic",
):
    """Plot quadrants with inout image at the center, overlayed instance label map in the top left quadrant,
    overlayed cell classes in the top center quadrant, single cell type in the other quadrants
    """

    input_images = np.squeeze(input_images)

    type_colour = VIZ_PARAM[viz_param_key]["type_colour"]

    if GTpath is not None:
        import glob

        ext = ["png", "tiff"]  # Add image formats here
        files = []
        [files.extend(glob.glob(os.path.join(GTpath, imageid) + "*." + e)) for e in ext]
        if len(files) == 0:
            warnings.warn("GT not found for " + os.path.join(GTpath, imageid))
            firstimage = render_label(
                cell_instances,
                img=tf.image.adjust_brightness(input_images, -0.7) / 255,
                normalize_img=False,
            )
            first_desc = "Prediction (instances)"

        else:
            imagepath = [file for file in files][0]
            if imagepath.endswith(".png"):
                cell_class_mask_GT = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
                if len(np.unique(cell_class_mask_GT)) == 2:  # binary mask
                    cell_class_mask_GT = cell_class_mask_GT > 0
                cell_class_mask_GT = cell_class_mask_GT.astype(int)
            else:
                cell_class_mask_GT = cv2.imread(imagepath, -1).astype(int)

            cell_instances_GT = label(cell_class_mask_GT)
            cell_type_dict_GT = {}

            for r in skimage.measure.regionprops(cell_instances_GT, cell_class_mask_GT):
                cell_type_dict_GT[r.label] = r.intensity_max

            firstimage = visualize_instances_map(
                input_images,
                cell_instances_GT,
                cell_type_dict_GT,
                type_colour=type_colour,
                line_thickness=2,
            )
            first_desc = "Ground Truth"

    else:
        firstimage = render_label(
            cell_instances,
            img=tf.image.adjust_brightness(input_images, -0.7) / 255,
            normalize_img=False,
        )
        first_desc = "Prediction (instances)"

    cell_type_image = visualize_instances_map(
        input_images,
        cell_instances,
        cell_type_dict,
        type_colour=type_colour,
        line_thickness=2,
    )

    cell_type_image1 = visualize_instances_map(
        input_images,
        cell_instances * (cell_type == 1),
        cell_type_dict,
        type_colour=type_colour,
        line_thickness=1,
    )
    cell_type_image2 = visualize_instances_map(
        input_images,
        cell_instances * (cell_type == 2),
        cell_type_dict,
        type_colour=type_colour,
        line_thickness=1,
    )
    cell_type_image3 = visualize_instances_map(
        input_images,
        cell_instances * (cell_type == 3),
        cell_type_dict,
        type_colour=type_colour,
        line_thickness=1,
    )
    cell_type_image4 = visualize_instances_map(
        input_images,
        cell_instances * (cell_type == 4),
        cell_type_dict,
        type_colour=type_colour,
        line_thickness=1,
    )
    cell_type_image5 = visualize_instances_map(
        input_images,
        cell_instances * (cell_type == 5),
        cell_type_dict,
        type_colour=type_colour,
        line_thickness=1,
    )
    cell_type_image6 = visualize_instances_map(
        input_images,
        cell_instances * (cell_type == 6),
        cell_type_dict,
        type_colour=type_colour,
        line_thickness=1,
    )

    # from skimage import color, io
    # nlabel = len(np.unique(cell_instances))
    # rgb_image = color.label2rgb(
    #     cell_instances, colors=np.random.random((nlabel, 3)),image=tf.image.adjust_brightness(input_images,-0.7)/255,saturation=1,alpha=0.5)#, bg_label=0

    # fig, axs = plt.subplots(1, 3, figsize=(14,5))

    # for ax, title, img in zip(
    #     axs.ravel(),
    #     ('input', 'prediction (instances)', 'prediction (class)'),
    #     (normed_sample,
    #      render_label(cell_instances, img=tf.image.adjust_brightness(normed_sample,-0.7)/255, normalize_img=False),
    #      render_label(cell_type, img=tf.image.adjust_brightness(normed_sample,-0.7)/255, normalize_img=False, cmap=random_label_cmap()),
    #     )
    # ):
    #     ax.imshow(img, interpolation=None)
    #     ax.set_title(title)
    #     ax.axis('off')

    fig, axs = plt.subplots(3, 3, figsize=(20, 20))

    for ax, title, img in zip(
        axs.flatten(),
        (
            first_desc,
            "Prediction (class)",
            "Epithelial",
            "Neutrophil",
            "INPUT",
            "Lymphocyte",
            "Plasma",
            "Eosinophil",
            "Connective",
        ),
        (
            firstimage,
            cell_type_image,
            cell_type_image2,
            cell_type_image1,
            input_images,
            cell_type_image3,
            cell_type_image4,
            cell_type_image5,
            cell_type_image6,
        ),
    ):
        ax.imshow(img, interpolation=None)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    directory = os.path.join(predict_folder, foldername)
    filepath1 = os.path.join(directory, "{0}_{1}.png".format(imageid, "overlay"))
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(filepath1, dpi=600)
    plt.close(fig)


# fig, axs = plt.subplots(1, 3, figsize=(14,5))

# for ax, title, img in zip(
#     axs.ravel(),
#     ('input', 'prediction (instances)', 'prediction (class)'),
#     (normed_sample,
#      rgb_image,
#      cell_type_image
#     )
# ):
#     ax.imshow(img, interpolation=None)
#     ax.set_title(title)
#     ax.axis('off')

# plt.tight_layout();
