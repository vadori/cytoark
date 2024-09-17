# ================================================================================================
# Acknowledgments:
# - Based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# ================================================================================================

import cv2
import os
import random
import tensorflow as tf
import numpy as np
from skimage.measure import label
from skimage.morphology import disk, dilation
from cisca.imutils import dir_distance_map, dir_distance_map_fullrange
from deepcell.deepcell_generators import _transform_masks as deepcell_transform_masks


def crop_center(img, cropx, cropy=None):
    """
    Crops a centered rectangular region from an image.

    This function extracts a central region of the specified dimensions from the input image.
    If only one dimension (`cropx`) is provided, the function assumes a square crop and uses the same size for the height (`cropy`).

    Args:
        img (numpy.ndarray): The input image to be cropped. Expected to be a 3D array (height, width, channels).
        cropx (int): The width of the cropped region.
        cropy (int, optional): The height of the cropped region. If None, `cropx` is used for both dimensions, resulting in a square crop.

    Returns:
        numpy.ndarray: The cropped image, with dimensions (`cropy`, `cropx`, channels).

    Raises:
        ValueError: If `cropx` or `cropy` is greater than the dimensions of `img`.

    Examples:
        >>> import numpy as np
        >>> img = np.random.rand(100, 100, 3)  # Example image of shape (100, 100, 3)
        >>> cropped_img = crop_center(img, 50)  # Crop a 50x50 region
        >>> cropped_img = crop_center(img, 50, 30)  # Crop a 50x30 region

    Notes:
        - The function assumes that `img` is a 3-dimensional array, where the first two dimensions are spatial (height and width), and the third dimension represents color channels.
        - The cropping operation is performed by slicing the image array based on calculated start positions for the crop region.
        - Ensure that the specified crop dimensions do not exceed the dimensions of the input image to avoid errors.
    """

    if cropy is None:
        cropy = cropx
    y, x = img.shape[0:2]
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2

    return img[starty : starty + cropy, startx : startx + cropx, ...]


class DataGeneratorCISCA(tf.keras.utils.Sequence):
    """
    A data generator for training, validation, and testing of image-based models with various configurations.

    This class extends `tf.keras.utils.Sequence` to generate batches of data for training, validation, or testing.
    It supports various image processing and augmentation techniques, including cropping, resizing, and transformation.
    The generator can handle multiple types of masks and data formats, and it can return additional information like
    original images and label maps if specified.

    Args:
        list_IDs (list of str): List of image IDs to be used by the generator.
        load_mode (str): Mode of loading data. Can be 'train', 'valid', or 'test'.
        input_shape (tuple): Shape of the input images (height, width).
        center_crop (int or None): Size of the center crop. If None, no cropping is performed.
        n_input_channels (int): Number of channels in the input images.
        n_contour_classes (int): Number of contour classes.
        n_celltype_classes (int): Number of cell type classes.
        dist_regression (bool): Whether to perform distance regression.
        magnification (str): Magnification level for distance maps.
        diag_dist (bool): Whether to include diagonal distance maps.
        batch_size (int): Number of samples per batch.
        steps_per_epoch (int): Number of steps per epoch.
        shuffle (bool): Whether to shuffle the data at the beginning of each epoch.
        random_crop (bool): Whether to apply random cropping to the images.
        random_transformers (callable or None): Function or object for applying random transformations to images and masks.
        with_original (bool): Whether to include original images in the output.
        with_label_map (bool): Whether to include label maps in the output.
        contour_mode (str): Contour mode indicating how contours are represented (e.g., 'RGB', 'GRAY4C', 'BWGAP', 'BW').
        image_folder (str): Path to the folder containing the images.
        rgb_contour_mask_folder (str): Path to the folder containing RGB contour masks.
        gray4c_contour_mask_folder (str): Path to the folder containing grayscale 4-channel contour masks.
        bw_contour_mask_folder (str): Path to the folder containing black-and-white contour masks.
        dir_distance_map_folder (str): Path to the folder containing distance maps.
        instance_mask_folder (str): Path to the folder containing instance masks.
        class_mask_folder (str): Path to the folder containing class masks.

    Methods:
        __len__(): Returns the number of batches per epoch.
        __iter__(): Iterates over the dataset yielding batches of data.
        __getitem__(index): Generates a batch of data at the specified index.
        _on_train_start(): Initializes and shuffles the indexes for training.
        _data_generation(batch_size=None, list_IDs_temp=None): Generates data for a given batch size.
    """

    def __init__(
        self,
        list_IDs,
        load_mode,
        input_shape,
        center_crop,
        n_input_channels,
        n_contour_classes,
        n_celltype_classes,
        dist_regression,
        magnification,
        diag_dist,
        batch_size,
        steps_per_epoch,
        shuffle,
        random_crop,
        random_transformers,
        with_original,
        with_label_map,
        contour_mode,
        image_folder,
        rgb_contour_mask_folder,
        gray4c_contour_mask_folder,
        bw_contour_mask_folder,
        dir_distance_map_folder,
        instance_mask_folder,
        class_mask_folder,
    ):
        self.input_shape = input_shape
        self.n_input_channels = n_input_channels
        self.center_crop = center_crop
        self.dist_regression = dist_regression
        self.magnification = magnification
        self.diag_dist = diag_dist
        self.list_IDs = list_IDs
        self.n_contour_classes = n_contour_classes
        self.n_celltype_classes = n_celltype_classes
        self.multiclass = self.n_celltype_classes > 1
        if self.center_crop is not None:
            self.output_shape = (self.center_crop, self.center_crop)
        else:
            self.output_shape = self.input_shape
        if self.diag_dist:
            self.n_output_channels = (
                self.n_contour_classes
                + 4
                + int(self.multiclass) * (1 + self.n_celltype_classes)
                + 1
            )
        elif self.dist_regression:
            self.n_output_channels = (
                self.n_contour_classes
                + 2
                + int(self.multiclass) * (1 + self.n_celltype_classes)
                + 1
            )
        else:
            self.n_output_channels = self.n_contour_classes + int(self.multiclass) * (
                1 + self.n_celltype_classes
            )
        self.contour_mode = contour_mode
        self.with_original = with_original
        self.with_label_map = with_label_map
        self.load_mode = load_mode
        self.shuffle = shuffle
        self.random_transformers = random_transformers
        self.batch_size = batch_size
        self.image_folder = image_folder
        self.rgb_contour_mask_folder = rgb_contour_mask_folder
        self.gray4c_contour_mask_folder = gray4c_contour_mask_folder
        self.bw_contour_mask_folder = bw_contour_mask_folder
        self.dir_distance_map_folder = dir_distance_map_folder
        self.instance_mask_folder = instance_mask_folder
        self.class_mask_folder = class_mask_folder

        if (self.load_mode == "valid") or (self.load_mode == "test"):
            self.random_crop = False
            self.steps_per_epoch = int(np.ceil(len(self.list_IDs) / self.batch_size))
            sampletocomplete = self.batch_size * self.steps_per_epoch - len(
                self.list_IDs
            )
            self.list_IDs = self.list_IDs + [
                self.list_IDs[i]
                for i in np.random.randint(len(self.list_IDs), size=sampletocomplete)
            ]
        else:
            self.random_crop = random_crop
            self.steps_per_epoch = steps_per_epoch

        self._on_train_start()

        self.train_img = []
        self.train_msk = []
        self.train_label = []
        self.train_class = []
        self.homap = []
        self.vemap = []
        self.tlmap = []
        self.blmap = []

        for img_id in self.list_IDs:
            img = cv2.cvtColor(
                cv2.imread(
                    os.path.join(image_folder, "{0}.png".format(img_id)),
                    cv2.IMREAD_COLOR,
                ),
                cv2.COLOR_BGR2RGB,
            )
            self.train_img.append(img)

            if (self.load_mode == "valid") or (self.load_mode == "train"):
                if self.contour_mode == "GRAY4C":
                    msk = cv2.imread(
                        os.path.join(
                            self.gray4c_contour_mask_folder, "{0}.png".format(img_id)
                        ),
                        cv2.IMREAD_GRAYSCALE,
                    )
                elif self.contour_mode == "RGB":
                    msk = cv2.cvtColor(
                        cv2.imread(
                            os.path.join(
                                self.rgb_contour_mask_folder, "{0}.png".format(img_id)
                            ),
                            cv2.IMREAD_COLOR,
                        ),
                        cv2.COLOR_BGR2RGB,
                    )
                elif "BW" in self.contour_mode:
                    labelmap = cv2.imread(
                        os.path.join(
                            self.instance_mask_folder, "{0}.tiff".format(img_id)
                        ),
                        -1,
                    )
                    msk = np.empty((*labelmap.shape, self.n_contour_classes))
                    if self.contour_mode == "BWGAP":
                        contourmap = cv2.imread(
                            os.path.join(
                                self.bw_contour_mask_folder, "{0}.png".format(img_id)
                            )
                        )
                        msk[:, :, 0] = (contourmap[:, :, 0] > 0) * 255
                        msk[:, :, 1] = (contourmap[:, :, 0] <= 0) * 255
                    else:
                        msk[:, :, 0] = (labelmap > 0) * 255
                        msk[:, :, 1] = (labelmap <= 0) * 255
                self.train_msk.append(msk)

            filename = os.path.join(instance_mask_folder, "{0}.tiff".format(img_id))
            if os.path.exists(filename):
                labelmap = cv2.imread(filename, -1)
            else:
                labelmap = np.zeros(shape=img.shape).astype(int)
                print(
                    f"File {filename} not found, loading zero placeholder as label map"
                )
            self.train_label.append(labelmap)

            if (self.load_mode == "valid") or (self.load_mode == "train"):
                if self.dist_regression and self.random_transformers is None:
                    homap = cv2.imread(
                        os.path.join(
                            self.dir_distance_map_folder, "{0}_h.tiff".format(img_id)
                        ),
                        -1,
                    )
                    vemap = cv2.imread(
                        os.path.join(
                            self.dir_distance_map_folder, "{0}_v.tiff".format(img_id)
                        ),
                        -1,
                    )
                    self.homap.append(homap)
                    self.vemap.append(vemap)
                    if self.diag_dist:
                        tlmap = cv2.imread(
                            os.path.join(
                                self.dir_distance_map_folder,
                                "{0}_tl.tiff".format(img_id),
                            ),
                            -1,
                        )
                        blmap = cv2.imread(
                            os.path.join(
                                self.dir_distance_map_folder,
                                "{0}_bl.tiff".format(img_id),
                            ),
                            -1,
                        )
                        self.tlmap.append(tlmap)
                        self.blmap.append(blmap)

            if self.multiclass:
                filename = os.path.join(class_mask_folder, "{0}.tiff".format(img_id))
                if os.path.exists(filename):
                    cell_class_mask = cv2.imread(filename, -1).astype(int)
                else:
                    cell_class_mask = (labelmap > 0).astype(int)
                    print(
                        f"File {filename} not found, loading binarized labelmap placeholder as cell class mask"
                    )
                self.train_class.append(cell_class_mask)

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(self.steps_per_epoch)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        "Generate one batch of data"

        if self.load_mode:
            indexes = self.indexes[
                index * self.batch_size : (index + 1) * self.batch_size
            ]

            list_IDs_temp = indexes
            batch_size = len(list_IDs_temp)

            return self._data_generation(
                batch_size=batch_size, list_IDs_temp=list_IDs_temp
            )
        else:
            return self._data_generation(batch_size=self.batch_size)

    def _on_train_start(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.load_mode and self.shuffle:
            np.random.shuffle(self.indexes)

    def _data_generation(self, batch_size=None, list_IDs_temp=None):
        X = np.empty((batch_size, *self.input_shape, self.n_input_channels))
        y = np.empty((batch_size, *self.output_shape, self.n_output_channels))
        names = []
        if self.with_original:
            X_orig = np.empty((batch_size, *self.input_shape, self.n_input_channels))
            y_orig = np.empty((batch_size, *self.output_shape, self.n_output_channels))
        if self.with_label_map:
            if self.multiclass:
                y_label_map = np.empty((batch_size, *self.output_shape, 2))
            else:
                y_label_map = np.empty((batch_size, *self.output_shape))

        for i in range(batch_size):
            if (self.load_mode == "valid") or (self.load_mode == "test"):
                img_id = list_IDs_temp[i]

            else:
                img_id = np.random.randint(len(self.list_IDs))

            name = self.list_IDs[img_id]
            names.append(name)
            img = self.train_img[img_id]
            labelmap = self.train_label[img_id]
            if self.multiclass:
                cell_class_mask = self.train_class[img_id]
            if (self.load_mode == "valid") or (self.load_mode == "train"):
                msk = self.train_msk[img_id]
                if self.with_original:
                    homap = cv2.imread(
                        os.path.join(
                            self.dir_distance_map_folder,
                            "{0}_h.tiff".format(self.list_IDs[img_id]),
                        ),
                        -1,
                    )
                    vemap = cv2.imread(
                        os.path.join(
                            self.dir_distance_map_folder,
                            "{0}_v.tiff".format(self.list_IDs[img_id]),
                        ),
                        -1,
                    )
                    msks = msk.copy()
                    if self.contour_mode == "GRAY4C":
                        if self.n_contour_classes == 4:
                            msks = tf.keras.utils.to_categorical(msks, 4)
                        elif self.n_contour_classes == 3:
                            msks2 = tf.keras.utils.to_categorical(msks, 4)
                            msks2[:, :, 2] = msks2[:, :, 2] + msks2[:, :, 3]
                            msks = msks2[:, :, 0:3]
                        elif self.n_contour_classes == 2:
                            msks2 = tf.keras.utils.to_categorical(msks, 4)
                            msks2[:, :, 0] = msks2[:, :, 0] + msks2[:, :, 1]
                            msks2[:, :, 1] = msks2[:, :, 2] + msks2[:, :, 3]
                            msks = msks2[:, :, 0:2]

                    elif self.contour_mode == "RGB":
                        if self.n_contour_classes == 3:
                            msks = tf.keras.utils.to_categorical(msks, 3)[:, :, 0, :]
                        elif self.n_contour_classes == 2:
                            msks2 = tf.keras.utils.to_categorical(msks, 3)[:, :, 0, :]
                            msks2[:, :, 0] = msks2[:, :, 0] + msks2[:, :, 1]
                            msks2[:, :, 1] = msks2[:, :, 2]
                            msks = msks2[:, :, 0:2]

                    labelmapBW = dilation(labelmap > 0, disk(6))
                    weightmask = labelmapBW + (1 - labelmapBW) * 0.05

                    if self.diag_dist:
                        tlmap = cv2.imread(
                            os.path.join(
                                self.dir_distance_map_folder,
                                "{0}_tl.tiff".format(self.list_IDs[img_id]),
                            ),
                            -1,
                        )
                        blmap = cv2.imread(
                            os.path.join(
                                self.dir_distance_map_folder,
                                "{0}_bl.tiff".format(self.list_IDs[img_id]),
                            ),
                            -1,
                        )
                    if self.multiclass:
                        msks = np.dstack(
                            [
                                msks,
                                tf.keras.utils.to_categorical(
                                    cell_class_mask, self.n_celltype_classes + 1
                                ),
                            ]
                        )
                    if self.diag_dist:
                        y_orig[i] = np.dstack(
                            [msks, homap, vemap, tlmap, blmap, weightmask]
                        )

                    elif self.dist_regression:
                        y_orig[i] = np.dstack([msks, homap, vemap, weightmask])

                    else:
                        y_orig[i] = msks

            if (self.load_mode == "valid") or (self.load_mode == "train"):
                if self.multiclass:
                    if self.random_transformers is not None:
                        data = self.random_transformers(
                            image=img, mask=msk, mask2=labelmap, mask3=cell_class_mask
                        )
                    else:
                        data = {"image": img, "mask": msk, "mask3": cell_class_mask}
                else:
                    if self.random_transformers is not None:
                        data = self.random_transformers(
                            image=img, mask=msk, mask2=labelmap
                        )
                    else:
                        data = {"image": img, "mask": msk}
            else:
                if self.multiclass:
                    if self.random_transformers is not None:
                        data = self.random_transformers(
                            image=img, mask2=labelmap, mask3=cell_class_mask
                        )
                    else:
                        data = {"image": img, "mask3": cell_class_mask}
                else:
                    if self.random_transformers is not None:
                        data = self.random_transformers(image=img, mask2=labelmap)
                    else:
                        data = {"image": img}

            X_height = img.shape[0]
            X_width = img.shape[1]

            if self.random_crop:
                random_width = random.randint(0, X_width - self.input_shape[1] - 1)
                random_height = random.randint(0, X_height - self.input_shape[0] - 1)
            else:
                random_width = 0
                random_height = 0

            if (self.load_mode == "test") and self.dist_regression:
                if self.random_transformers is not None:
                    labelmap = label(
                        data["mask2"][
                            random_height : random_height + self.input_shape[0],
                            random_width : random_width + self.input_shape[1],
                        ]
                    )
            else:
                if self.dist_regression:
                    if self.random_transformers is not None:
                        labelmap = label(
                            data["mask2"][
                                random_height : random_height + self.input_shape[0],
                                random_width : random_width + self.input_shape[1],
                            ]
                        )
                        homap, vemap, tlmap, blmap = dir_distance_map(
                            labelmap, magnification=int(self.magnification[0:2])
                        )
                    else:
                        homap = self.homap[img_id]
                        vemap = self.vemap[img_id]
                        homap = homap[
                            random_height : random_height + self.input_shape[0],
                            random_width : random_width + self.input_shape[1],
                        ]
                        vemap = vemap[
                            random_height : random_height + self.input_shape[0],
                            random_width : random_width + self.input_shape[1],
                        ]
                        if self.diag_dist:
                            tlmap = self.tlmap[img_id]
                            blmap = self.blmap[img_id]
                            tlmap = tlmap[
                                random_height : random_height + self.input_shape[0],
                                random_width : random_width + self.input_shape[1],
                            ]
                            blmap = blmap[
                                random_height : random_height + self.input_shape[0],
                                random_width : random_width + self.input_shape[1],
                            ]
                    if self.center_crop is not None:
                        homap = crop_center(homap, self.center_crop)
                        vemap = crop_center(vemap, self.center_crop)
                        if self.diag_dist:
                            tlmap = crop_center(tlmap, self.center_crop)
                            blmap = crop_center(blmap, self.center_crop)

            labelmapBW = dilation(labelmap > 0, disk(6))

            weightmask = labelmapBW + (1 - labelmapBW) * 0.05

            X[i] = (
                data["image"][
                    random_height : random_height + self.input_shape[0],
                    random_width : random_width + self.input_shape[1],
                    :,
                ]
                / 255
            )

            if self.with_original:
                X_orig[i] = img / 255

            if self.with_label_map:
                if self.multiclass:
                    y_label_map[i, :, :, 0] = labelmap
                    y_label_map[i, :, :, 1] = data["mask3"]
                else:
                    y_label_map[i] = labelmap

            if (self.load_mode == "valid") or (self.load_mode == "train"):
                if self.contour_mode == "GRAY4C":
                    if self.n_contour_classes == 4:
                        data["mask"] = tf.keras.utils.to_categorical(data["mask"], 4)
                    elif self.n_contour_classes == 3:
                        msk = tf.keras.utils.to_categorical(data["mask"], 4)
                        msk[:, :, 2] = msk[:, :, 2] + msk[:, :, 3]
                        data["mask"] = msk[:, :, 0:3]
                    elif self.n_contour_classes == 2:
                        msk = tf.keras.utils.to_categorical(data["mask"], 4)
                        msk[:, :, 0] = msk[:, :, 0] + msk[:, :, 1]
                        msk[:, :, 1] = msk[:, :, 2] + msk[:, :, 3]
                        data["mask"] = msk[:, :, 0:2]

                elif self.contour_mode == "RGB":
                    if self.n_contour_classes == 3:
                        data["mask"] = tf.keras.utils.to_categorical(data["mask"], 3)[
                            :, :, 0, :
                        ]
                    elif self.n_contour_classes == 2:
                        msk = tf.keras.utils.to_categorical(data["mask"], 3)[:, :, 0, :]
                        msk[:, :, 0] = msk[:, :, 0] + msk[:, :, 1]
                        msk[:, :, 1] = msk[:, :, 2]
                        data["mask"] = msk[:, :, 0:2]

            if (self.load_mode == "valid") or (self.load_mode == "train"):
                if self.multiclass:
                    data["mask3"] = tf.keras.utils.to_categorical(
                        data["mask3"], self.n_celltype_classes + 1
                    )
                    data["mask"] = np.dstack([data["mask"], data["mask3"]])
                msk = data["mask"][
                    random_height : random_height + self.input_shape[0],
                    random_width : random_width + self.input_shape[1],
                    :,
                ]

                if self.center_crop is not None:
                    msk = crop_center(msk, self.center_crop)
                    weightmask = crop_center(weightmask, self.center_crop)

                if self.diag_dist:
                    y[i] = np.dstack([msk, homap, vemap, tlmap, blmap, weightmask])

                elif self.dist_regression:
                    y[i] = np.dstack([msk, homap, vemap, weightmask])

                else:
                    y[i] = msk

        if self.load_mode == "test":
            return X.astype(np.float32), (y_label_map.astype(np.float32), names)
        elif self.with_original and self.with_label_map:
            return (
                X.astype(np.float32),
                y.astype(np.float32),
                X_orig.astype(np.float32),
                y_orig.astype(np.float32),
                y_label_map.astype(np.float32),
                names,
            )
        elif self.with_original:
            return (
                X.astype(np.float32),
                y.astype(np.float32),
                X_orig.astype(np.float32),
                y_orig.astype(np.float32),
                names,
            )
        elif self.with_label_map:
            return X.astype(np.float32), (
                y.astype(np.float32),
                y_label_map.astype(np.float32),
                names,
            )
        else:
            return X.astype(np.float32), y.astype(np.float32)


