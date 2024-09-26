import os
import datetime
import gc
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.optimizers import Adam

from cisca.ioutils import set_seed, save_json, MyEncoder
from cisca.dataloading import DataGeneratorCISCA
from cisca.config import ConfigCISCA
from cisca.architectures.efficientunet import (
    get_efficient_unet_b2,
    get_efficient_unet_b3,
    get_efficient_unet_b4,
)
from cisca.architectures.unet import get_unetmodel
from cisca.postprocessing.instance_prediction import postprocess
from cisca.imutils import tile_image, untile_image, _normalize_grid
from cisca.metrics.loss import (
    compound_dice_cce_loss,
    dice_coef_wrapper,
    categorical_crossentropy_loss,
    compound_focal_tversky_wcce_loss,
    weighted_categorical_crossentropy_loss,
    focal_tversky_loss,
    stardist_tversky_loss,
    reg_loss,
    mse_loss,
    msge_loss,
    dice_coef_wrapper_cell_type,
    mae_loss,
)


class CISCA(object):
    """
    CISCA framework.

    Parameters
    ----------
    config : :class:`ConfigCISCA`, optional
        Configuration object for the model.
    name : str or None, optional
        Model name. If set to ``None``, a timestamp-based name will be generated.
    basedir : str, optional
        Base directory where model files (such as weights, configuration, and logs) will be stored. Defaults to current directory ('.').
    prepare_for_training : bool, optional
        If ``True``, the model will be initialized and prepared for training.

    Attributes
    ----------
    config : :class:`ConfigCISCA`
        Configuration object containing settings such as dataset information, model parameters, and file paths.
    multiclass : bool
        Whether the model is a multi-class model, determined by the number of cell-type classes.
    model_folder : str
        Path to the directory where the model files will be stored.
    logdir : str
        Directory where logs, weights, and other related model files are stored.
    data_folder : str
        Path to the folder containing training dataset images and masks.
    valid_data_folder : str
        Path to the folder containing validation dataset images and masks.
    image_folder : str
        Path to the folder containing the training images.
    rgb_contour_mask_folder : str
        Path to the folder containing RGB contour masks for training.
    gray4c_contour_mask_folder : str
        Path to the folder containing 4-channel grayscale contour masks for training.
    bw_contour_mask_folder : str
        Path to the folder containing binary contour masks for training.
    instance_mask_folder : str
        Path to the folder containing instance masks for training.
    class_mask_folder : str
        Path to the folder containing class masks for training.
    dir_distance_map_folder : str
        Path to the folder containing distance maps for training.
    valid_image_folder : str
        Path to the folder containing validation images.
    valid_rgb_contour_mask_folder : str
        Path to the folder containing RGB contour masks for validation.
    valid_gray4c_contour_mask_folder : str
        Path to the folder containing 4-channel grayscale contour masks for validation.
    valid_bw_contour_mask_folder : str
        Path to the folder containing binary contour masks for validation.
    valid_instance_mask_folder : str
        Path to the folder containing instance masks for validation.
    valid_class_mask_folder : str
        Path to the folder containing class masks for validation.
    valid_dir_distance_map_folder : str
        Path to the folder containing distance maps for validation.
    predict_folder : str
        Directory where prediction outputs will be saved.
    name : str
        Model name, either provided during initialization or generated from the current timestamp.
    basedir : :class:`pathlib.Path` or None
        Base directory for model-related files. Can be ``None`` if not provided.
    _model_prepared : bool
        Indicates whether the model is prepared for training.
    weights_filepath : str or None
        Path to the model weights file, if any. Set during model initialization if available.
    keras_model : :class:`tensorflow.keras.Model`
        The Keras model constructed by the `_build()` method.
    strategy : :class:`tf.distribute.Strategy`
        TensorFlow strategy used for distributed training.
    """

    def __init__(
        self, config=ConfigCISCA(), name=None, basedir=".", prepare_for_training=False
    ):
        """See class docstring."""

        self.config = config

        self.multiclass = self.config.n_celltype_classes > 1
        self.model_folder = os.path.join(self.config.root_folder, "models","segmentation")
        self.logdir = os.path.join(
            self.model_folder,
            self.config.model_name,
            self.config.model_variant,
            "_".join([self.config.dataset_name, self.config.magnification]),
            "_pretrained" if self.config.pretrained else "",
        )
        self.data_folder = os.path.join(
            self.config.root_folder,
            "datasets",
            self.config.dataset_name,
            self.config.magnification,
            "x".join(
                [str(self.config.raw_input_side), str(self.config.raw_input_side)]
            ),
        )
        self.valid_data_folder = os.path.join(
            self.config.root_folder,
            "datasets",
            self.config.dataset_name,
            self.config.magnification,
            "x".join(
                [
                    str(self.config.valid_raw_input_side),
                    str(self.config.valid_raw_input_side),
                ]
            ),
        )
        self.image_folder = os.path.join(self.data_folder, "image")
        self.rgb_contour_mask_folder = os.path.join(self.data_folder, "rgbmask")
        self.gray4c_contour_mask_folder = os.path.join(self.data_folder, "graymask4")
        self.bw_contour_mask_folder = os.path.join(self.data_folder, "bwmask")
        self.instance_mask_folder = os.path.join(self.data_folder, "label")
        self.class_mask_folder = os.path.join(self.data_folder, "class")
        self.dir_distance_map_folder = os.path.join(self.data_folder, "distmap")
        self.valid_image_folder = os.path.join(self.valid_data_folder, "image")
        self.valid_rgb_contour_mask_folder = os.path.join(
            self.valid_data_folder, "rgbmask"
        )
        self.valid_gray4c_contour_mask_folder = os.path.join(
            self.valid_data_folder, "graymask4"
        )
        self.valid_bw_contour_mask_folder = os.path.join(
            self.valid_data_folder, "bwmask"
        )
        self.valid_instance_mask_folder = os.path.join(self.valid_data_folder, "label")
        self.valid_class_mask_folder = os.path.join(self.valid_data_folder, "class")
        self.valid_dir_distance_map_folder = os.path.join(
            self.valid_data_folder, "distmap"
        )
        self.predict_folder = os.path.join(
            self.config.root_folder,
            "output",
            self.config.model_name,
            self.config.model_variant,
        )

        self.name = (
            name
            if name is not None
            else datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        )
        self.basedir = Path(basedir) if basedir is not None else None
        self._model_prepared = False
        self.weights_filepath = None
        if self.config.train_checkpoint is not None:
            self.weights_filepath = os.path.join(
                self.model_folder,
                self.model_name,
                self.mode_variant,
                self.train_checkpoint,
            )
        self.keras_model, self.strategy = self._build()
        if prepare_for_training:
            self._prepare_for_training()

    def _build(self):
        """
        Build and return the Keras model along with the appropriate TensorFlow distribution strategy.

        This method creates the Keras model based on the provided configuration. If a multi-GPU
        mirrored strategy is enabled (`self.config.mirroredstrategy`), the model is built within
        the scope of the `tf.distribute.MirroredStrategy`. Otherwise, the model is built without
        a distribution strategy.

        The model is constructed based on the backbone specified in the configuration. It supports
        either a standard U-Net model or an EfficientNet-based U-Net variant, depending on the
        configuration.

        If weights are provided through the configuration (`self.weights_filepath`), the model
        will load pre-trained weights from the checkpoint.

        Returns
        -------
        keras_model : :class:`tensorflow.keras.Model`
            The constructed Keras model based on the configuration and selected backbone.
        strategy : :class:`tf.distribute.Strategy` or None
            TensorFlow strategy for distributed training. If multi-GPU strategy is enabled,
            it will return a `MirroredStrategy`; otherwise, it will return `None`.
        """

        if self.config.mirroredstrategy:
            strategy = tf.distribute.MirroredStrategy(
                devices=["GPU:0", "GPU:1"],
                cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(),
            )
            print("Number of devices: {}".format(strategy.num_replicas_in_sync))

            # Open a strategy scope.
            with strategy.scope():
                if not self.config.backbone.startswith("Efficient"):
                    data_kwargs = {
                        "input_shape": self.config.input_shape
                        + (self.config.n_input_channels,),
                        "n_contour_classes": self.config.n_contour_classes,
                        "n_celltype_classes": self.config.n_celltype_classes,
                        "dist_regression": self.config.dist_regression,
                        "diag_dist": self.config.diag_dist,
                        "head_blocks": self.config.head_blocks,
                        "unet_grid": _normalize_grid(self.config.unet_grid, 2),
                        "unet_n_conv_per_depth": self.config.unet_n_conv_per_depth,
                        "unet_n_filter_base": self.config.unet_n_filter_base,
                        "unet_kernel_size": self.config.unet_kernel_size,
                        "unet_activation": self.config.unet_activation,
                        "net_conv_after_unet": self.config.net_conv_after_unet,
                        "unet_last_activation": self.config.unet_last_activation,
                        "unet_n_depth": self.config.unet_n_depth,
                        "unet_pool": self.config.unet_pool,
                        "unet_batch_norm": self.config.unet_batch_norm,
                        "unet_dropout": self.config.unet_dropout,
                        "unet_prefix": self.config.unet_prefix,
                        "backbone": self.config.backbone,
                    }
                    keras_model = get_unetmodel(**data_kwargs)
                else:
                    data_kwargs = {
                        "input_shape": self.config.input_shape
                        + (self.config.n_input_channels,),
                        "n_contour_classes": self.config.n_contour_classes,
                        "n_celltype_classes": self.config.n_celltype_classes,
                        "pretrained": self.config.pretrained,
                        "block_type": self.config.block_type,
                        "concat_input": self.config.concat_input,
                        "trainable": self.config.trainable,
                        "framework": self.config.framework,
                        "encdec_skipconn_attention": self.config.encdec_skipconn_attention,
                        "decdec_skipconn_attention": self.config.decdec_skipconn_attention,
                        "dist_regression": self.config.dist_regression,
                        "diag_dist": self.config.diag_dist,
                        "skipp_conn_decod": self.config.decdec_skipconn,
                    }
                    if self.config.backbone == "EfficientNet-B2":
                        keras_model = get_efficient_unet_b2(**data_kwargs)
                    elif self.config.backbone == "EfficientNet-B3":
                        keras_model = get_efficient_unet_b3(**data_kwargs)
                    else:
                        keras_model = get_efficient_unet_b4(**data_kwargs)

                if self.weights_filepath is not None:
                    print("Loading weights from checkpoint")
                    keras_model.load_weights(self.weights_filepath)
                for layer in keras_model.layers:
                    layer.trainable = True
        else:
            strategy = None
            if not self.config.backbone.startswith("Efficient"):
                data_kwargs = {
                    "input_shape": self.config.input_shape
                    + (self.config.n_input_channels,),
                    "n_contour_classes": self.config.n_contour_classes,
                    "n_celltype_classes": self.config.n_celltype_classes,
                    "dist_regression": self.config.dist_regression,
                    "diag_dist": self.config.diag_dist,
                    "head_blocks": self.config.head_blocks,
                    "unet_grid": _normalize_grid(self.config.unet_grid, 2),
                    "unet_n_conv_per_depth": self.config.unet_n_conv_per_depth,
                    "unet_n_filter_base": self.config.unet_n_filter_base,
                    "unet_kernel_size": self.config.unet_kernel_size,
                    "unet_activation": self.config.unet_activation,
                    "net_conv_after_unet": self.config.net_conv_after_unet,
                    "unet_last_activation": self.config.unet_last_activation,
                    "unet_n_depth": self.config.unet_n_depth,
                    "unet_pool": self.config.unet_pool,
                    "unet_batch_norm": self.config.unet_batch_norm,
                    "unet_dropout": self.config.unet_dropout,
                    "unet_prefix": self.config.unet_prefix,
                    "backbone": self.config.backbone,
                }
                keras_model = get_unetmodel(**data_kwargs)
            else:
                data_kwargs = {
                    "input_shape": self.config.input_shape
                    + (self.config.n_input_channels,),
                    "n_contour_classes": self.config.n_contour_classes,
                    "n_celltype_classes": self.config.n_celltype_classes,
                    "pretrained": self.config.pretrained,
                    "block_type": self.config.block_type,
                    "concat_input": self.config.concat_input,
                    "trainable": self.config.trainable,
                    "framework": self.config.framework,
                    "encdec_skipconn_attention": self.config.encdec_skipconn_attention,
                    "decdec_skipconn_attention": self.config.decdec_skipconn_attention,
                    "dist_regression": self.config.dist_regression,
                    "diag_dist": self.config.diag_dist,
                    "skipp_conn_decod": self.config.decdec_skipconn,
                }
                if self.config.backbone == "EfficientNet-B2":
                    keras_model = get_efficient_unet_b2(**data_kwargs)
                elif self.config.backbone == "EfficientNet-B3":
                    keras_model = get_efficient_unet_b3(**data_kwargs)
                else:
                    keras_model = get_efficient_unet_b4(**data_kwargs)

            if self.weights_filepath is not None:
                print("Loading weights from checkpoint")
                keras_model.load_weights(self.weights_filepath)
            for layer in keras_model.layers:
                layer.trainable = True

        return keras_model, strategy

    def train(self, epochs=None, steps_per_epoch=None):
        """
        Train the neural network with the provided data.

        Parameters
        ----------
        epochs : int, optional
            Number of epochs to train the model. If not provided, the value from the configuration (`self.config.train_epochs`) will be used.
        steps_per_epoch : int, optional
            Number of steps per epoch. If not provided, the value from the configuration (`self.config.steps_per_epoch`) will be used.

        Returns
        -------
        history : :class:`keras.callbacks.History`
            The history object containing details of the training process, such as loss values and metrics for each epoch.
        """

        if steps_per_epoch is not None:
            self.config.update_parameters(steps_per_epoch=steps_per_epoch)
            self._prepare_for_training(steps_per_epoch=steps_per_epoch)

        if epochs is None:
            epochs = self.config.train_epochs

        if not self._model_prepared:
            self._prepare_for_training()

        print(
            "Training for {0} epochs, using {1} per epoch".format(
                epochs, self.config.steps_per_epoch
            )
        )

        history = self.keras_model.fit(
            self.train_generator,
            epochs=epochs,
            verbose=1,
            validation_data=self.valid_generator,
            callbacks=self.callbacks,
        )
        return history

    def _prepare_for_training(
        self, optimizer=None, run_mode=None, steps_per_epoch=None, seed=42
    ):
        """
        Prepare the model for training by compiling it and setting up the necessary callbacks.

        Parameters
        ----------
        optimizer : tf.keras.optimizers.Optimizer or None, optional
            An instance of a Keras optimizer to be used for training. If None, the Adam optimizer
            is used with the learning rate specified in the configuration.

        run_mode : str or None, optional
            Specifies the mode of training. If provided, it updates the configuration's `run_mode`.

        steps_per_epoch : int or None, optional
            Number of steps (batches) to run per epoch during training. If None, it is computed
            based on the dataset size and configuration.

        seed : int, optional
            Random seed for reproducibility. Default is 42.

        Returns
        -------
        None
            The method sets up the model for training, but does not return a value.
        """

        set_seed(seed)

        if run_mode is not None:
            self.config.update_parameters(run_mode=run_mode)

        images_list = pd.read_csv(os.path.join(self.data_folder, "folds.csv"))
        self.partition = (
            images_list.groupby("fold")["img_id"]
            .apply(lambda x: x.values.tolist())
            .to_dict()
        )

        if self.config.run_mode == "USE":
            from random import shuffle

            # training with test set included for applications - i.e. for applying the model to WSIs
            # training_dataset = self.partition[0]+self.partition[2]+[item for sublist in 5*[self.partition[2][0:4]] for item in sublist]+[item for sublist in 5*[self.partition[0][0:8]] for item in sublist]
            self.training_dataset = (
                self.partition[0]
                + self.partition[2]
                + self.partition[0][0:2]
                + self.partition[0][-9:]
                + self.partition[0][-9:]
                + self.partition[0][-9:]
                + self.partition[2][0:1]
            )
            shuffle(self.training_dataset)
        else:
            self.training_dataset = self.partition[0]

        valid_images_list = pd.read_csv(
            os.path.join(self.valid_data_folder, "folds.csv")
        )
        self.valid_partition = (
            valid_images_list.groupby("fold")["img_id"]
            .apply(lambda x: x.values.tolist())
            .to_dict()
        )

        if self.config.run_mode == "USE":
            from random import shuffle
            # training with test set included for applications - i.e. for applying the model to WSIs
            # valid_dataset = self.valid_partition[1]+[item for sublist in 5*[self.valid_partition[1][0:4]] for item in sublist]+[item for sublist in 5*[self.partition[0][0:8]] for item in sublist]
            self.valid_dataset = self.valid_partition[1] + self.valid_partition[1][-32:]
        else:
            self.valid_dataset = self.valid_partition[1]

        if steps_per_epoch is None:
            steps_per_epoch = np.ceil(
                1
                / self.config.batch_size
                * len(self.training_dataset)
                * (self.config.raw_input_side / self.config.input_shape[0]) ** 2
            )

        # update parameters and identifier (string that identifies the config options)
        self.config.update_parameters(steps_per_epoch=steps_per_epoch)
        self.config.update_string_ID()
        self.save_config()

        data_kwargs = {
            "input_shape": self.config.input_shape,
            "center_crop": self.config.center_crop,
            "n_input_channels": self.config.n_input_channels,
            "n_contour_classes": self.config.n_contour_classes,
            "n_celltype_classes": self.config.n_celltype_classes,
            "dist_regression": self.config.dist_regression,
            "magnification": self.config.magnification,
            "diag_dist": self.config.diag_dist,
            "batch_size": self.config.batch_size,
            "steps_per_epoch": self.config.steps_per_epoch,
            "shuffle": self.config.shuffle,
            "random_crop": self.config.random_crop,
            "random_transformers": self.config.random_transformers,
            "with_original": self.config.with_original,
            "with_label_map": self.config.with_label_map,
            "contour_mode": self.config.contour_mode,
            "image_folder": self.image_folder,
            "rgb_contour_mask_folder": self.rgb_contour_mask_folder,
            "gray4c_contour_mask_folder": self.gray4c_contour_mask_folder,
            "bw_contour_mask_folder": self.bw_contour_mask_folder,
            "dir_distance_map_folder": self.dir_distance_map_folder,
            "instance_mask_folder": self.instance_mask_folder,
            "class_mask_folder": self.class_mask_folder,
        }

        self.train_generator = DataGeneratorCISCA(
            self.training_dataset, load_mode="train", **data_kwargs
        )

        data_kwargs = {
            "input_shape": self.config.input_shape,
            "center_crop": self.config.center_crop,
            "n_input_channels": self.config.n_input_channels,
            "n_contour_classes": self.config.n_contour_classes,
            "n_celltype_classes": self.config.n_celltype_classes,
            "dist_regression": self.config.dist_regression,
            "magnification": self.config.magnification,
            "diag_dist": self.config.diag_dist,
            "batch_size": self.config.batch_size,
            "steps_per_epoch": self.config.steps_per_epoch,
            "shuffle": self.config.shuffle,
            "random_crop": self.config.random_crop,
            "random_transformers": self.config.valid_random_transformers,
            "with_original": self.config.with_original,
            "with_label_map": self.config.with_label_map,
            "contour_mode": self.config.contour_mode,
            "image_folder": self.valid_image_folder,
            "rgb_contour_mask_folder": self.valid_rgb_contour_mask_folder,
            "gray4c_contour_mask_folder": self.valid_gray4c_contour_mask_folder,
            "bw_contour_mask_folder": self.valid_bw_contour_mask_folder,
            "dir_distance_map_folder": self.valid_dir_distance_map_folder,
            "instance_mask_folder": self.valid_instance_mask_folder,
            "class_mask_folder": self.valid_class_mask_folder,
        }

        self.valid_generator = DataGeneratorCISCA(
            self.valid_dataset, load_mode="valid", **data_kwargs
        )

        if optimizer is None:
            optimizer = Adam(
                amsgrad=True, clipnorm=0.001, learning_rate=self.config.learning_rate
            )

        train_loss_weights = self.config.train_loss_weights_unnorm / np.sum(
            self.config.train_loss_weights_unnorm
        )
        contour_classes_loss = compound_dice_cce_loss(
            n_contour_classes=self.config.n_contour_classes,
            train_contour_weights=train_loss_weights[0 : self.config.n_contour_classes],
        )
        contour_classes_metrics = [
            dice_coef_wrapper(m) for m in range(self.config.n_contour_classes - 1)
        ] + [categorical_crossentropy_loss(self.config.n_contour_classes)]

        if self.config.dist_regression:
            dist_loss = reg_loss(
                n_contour_classes=self.config.n_contour_classes,
                n_celltype_classes=self.config.n_celltype_classes,
                train_reg_loss_weights=train_loss_weights[
                    self.config.n_contour_classes + int(self.multiclass) :
                ],
                masked_regression=self.config.masked_regression,
                diag_dist=self.config.diag_dist,
            )
            dist_metrics = [
                mse_loss(
                    self.config.n_contour_classes,
                    self.config.n_celltype_classes,
                    self.config.masked_regression,
                ),
                mae_loss(
                    self.config.n_contour_classes,
                    self.config.n_celltype_classes,
                    self.config.masked_regression,
                ),
                msge_loss(
                    self.config.n_contour_classes,
                    self.config.n_celltype_classes,
                    self.config.masked_regression,
                    self.config.diag_dist,
                ),
            ]
            if self.multiclass:
                celltype_classes_loss = compound_focal_tversky_wcce_loss(
                    self.config.n_contour_classes,
                    self.config.n_celltype_classes,
                    train_loss_weights[self.config.n_contour_classes],
                    self.config.train_class_weights,
                    self.config.masked_celltype_classification,
                )
                # celltype_classes_loss = compound_focal_tversky_wcce_dice_loss(self.config.n_contour_classes,self.config.n_celltype_classes, train_loss_weights[self.config.n_contour_classes],self.config.train_class_weights,self.config.train_class_weights_2,self.config.masked_celltype_classification)

                loss = [contour_classes_loss, dist_loss, celltype_classes_loss]
                celltype_classes_metrics = [
                    dice_coef_wrapper_cell_type(m, self.config.n_contour_classes)
                    for m in range(self.config.n_celltype_classes + 1)
                ] + [
                    weighted_categorical_crossentropy_loss(
                        self.config.n_contour_classes,
                        self.config.n_celltype_classes,
                        self.config.train_class_weights,
                        self.config.masked_celltype_classification,
                    ),
                    focal_tversky_loss(
                        self.config.n_contour_classes, self.config.n_celltype_classes
                    ),
                    stardist_tversky_loss(
                        self.config.n_contour_classes, self.config.n_celltype_classes
                    ),
                    celltype_classes_loss,
                ]
                metrics = [
                    contour_classes_metrics,
                    dist_metrics,
                    celltype_classes_metrics,
                ]
            else:
                loss = [contour_classes_loss, dist_loss]
                metrics = [contour_classes_metrics, dist_metrics]
        else:
            if self.multiclass:
                celltype_classes_loss = compound_focal_tversky_wcce_loss(
                    self.config.n_contour_classes,
                    self.config.n_celltype_classes,
                    train_loss_weights[self.config.n_contour_classes],
                    self.config.train_class_weights,
                    self.config.masked_celltype_classification,
                )
                # celltype_classes_loss = compound_focal_tversky_wcce_dice_loss(self.config.n_contour_classes,self.config.n_celltype_classes, train_loss_weights[self.config.n_contour_classes],self.config.train_class_weights,self.config.train_class_weights_2,self.config.masked_celltype_classification)
                loss = [contour_classes_loss, celltype_classes_loss]
                celltype_classes_metrics = [
                    dice_coef_wrapper_cell_type(m, self.config.n_contour_classes)
                    for m in range(self.config.n_celltype_classes + 1)
                ] + [
                    weighted_categorical_crossentropy_loss(
                        self.config.n_contour_classes,
                        self.config.n_celltype_classes,
                        self.config.train_class_weights,
                        self.config.masked_celltype_classification,
                    ),
                    focal_tversky_loss(
                        self.config.n_contour_classes, self.config.n_celltype_classes
                    ),
                    stardist_tversky_loss(
                        self.config.n_contour_classes, self.config.n_celltype_classes
                    ),
                ]
                metrics = [contour_classes_metrics, celltype_classes_metrics]
            else:
                loss = [contour_classes_loss]
                metrics = [contour_classes_metrics]

        if self.config.mirroredstrategy:
            with self.strategy.scope():
                self.keras_model.compile(
                    loss=loss, run_eagerly=False, optimizer=optimizer, metrics=metrics
                )
        else:
            self.keras_model.compile(
                loss=loss, run_eagerly=False, optimizer=optimizer, metrics=metrics
            )

        self.callbacks = []

        # lrSchedule = LearningRateScheduler(
        #     lambda epoch: schedule_steps(
        #         epoch, [(1e-4, 60), (1e-4, 62), (2e-5, 65), (1e-5, 70)]
        #     )
        # )

        log_filepath = os.path.join(
            self.logdir, "log__" + self.config.string_id + ".csv"
        )
        weights_filepath = os.path.join(
            self.logdir,
            "weights__" + self.config.string_id + "__{epoch:02d}__{val_loss:.5f}.hdf5",
        )

        # tracking performance on training and validation test
        csv_logger = tf.keras.callbacks.CSVLogger(
            log_filepath, separator=",", append=False
        )
        self.callbacks.append(csv_logger)

        # stop training if not improving
        earlystopCallback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=self.config.earlystop_patience,
            verbose=1,
            mode="min",
            baseline=None,
            restore_best_weights=True,
        )
        self.callbacks.append(earlystopCallback)

        # save model weights
        saveCallback = tf.keras.callbacks.ModelCheckpoint(
            weights_filepath,
            monitor="val_loss",
            mode="min",
            verbose=0,
            save_best_only=False,
            save_weights_only=True,
            save_freq="epoch",
        )
        self.callbacks.append(saveCallback)

        if self.config.reduce_lr is not None:
            rlrop_params = self.config.reduce_lr
            if "verbose" not in rlrop_params:
                rlrop_params["verbose"] = True
            self.callbacks.insert(
                0, tf.keras.callbacks.ReduceLROnPlateau(**rlrop_params)
            )

        self._model_prepared = True

    def get_data_generator(
        self,
        load_mode="valid",
        partition_index=2,
        batch_size=1,
        shuffle=False,
        steps_per_epoch=None,
        data_folder=None,
        random_transformers=None,
        with_original=None,
        with_label_map=None,
    ):
        """
        Parameters
        ----------
        load_mode : str, optional
            Specifies the data loading mode. Options are 'train', 'valid', or 'test'. Default is 'valid'.
        partition_index : int, optional
            The index of the partition to load data from. Default is 2.
        batch_size : int, optional
            The size of the batch for data generation. Default is 1.
        shuffle : bool, optional
            Whether to shuffle the data. Default is False.
        steps_per_epoch : int or None, optional
            Number of steps per epoch. If None, defaults to `self.config.steps_per_epoch`.
        data_folder : str or None, optional
            Path to the folder containing the dataset. If None, uses `valid_data_folder` for validation/testing or `data_folder` for training.
        random_transformers : list or None, optional
            List of random transformers to apply to the data. If None, uses the default from `self.config`.
        with_original : bool or None, optional
            Whether to include the original images in the generated data. If None, defaults to `self.config.with_original`.
        with_label_map : bool or None, optional
            Whether to include label maps in the generated data. If None, defaults to `self.config.with_label_map`.

        Returns
        -------
        DataGeneratorCISCA
            A data generator object initialized with the provided parameters.
        """

        if steps_per_epoch is None:
            steps_per_epoch = self.config.steps_per_epoch
        if with_original is None:
            with_original = self.config.with_original
        if with_label_map is None:
            with_label_map = self.config.with_label_map
        if data_folder is None:
            if load_mode == "valid" or load_mode == "test":
                data_folder = self.valid_data_folder
            else:
                data_folder = self.data_folder

        if random_transformers is None:
            random_transformers = self.config.valid_random_transformers

        images_list = pd.read_csv(os.path.join(data_folder, "folds.csv"))
        partition = (
            images_list.groupby("fold")["img_id"]
            .apply(lambda x: x.values.tolist())
            .to_dict()
        )
        dataset = partition[partition_index]

        image_folder = os.path.join(data_folder, "image")
        rgb_contour_mask_folder = os.path.join(data_folder, "rgbmask")
        gray4c_contour_mask_folder = os.path.join(data_folder, "graymask4")
        bw_contour_mask_folder = os.path.join(data_folder, "bwmask")
        instance_mask_folder = os.path.join(data_folder, "label")
        class_mask_folder = os.path.join(data_folder, "class")
        dir_distance_map_folder = os.path.join(data_folder, "distmap")

        data_kwargs = {
            "input_shape": self.config.input_shape,
            "center_crop": self.config.center_crop,
            "n_input_channels": self.config.n_input_channels,
            "n_contour_classes": self.config.n_contour_classes,
            "n_celltype_classes": self.config.n_celltype_classes,
            "dist_regression": self.config.dist_regression,
            "magnification": self.config.magnification,
            "diag_dist": self.config.diag_dist,
            "batch_size": batch_size,
            "steps_per_epoch": steps_per_epoch,
            "shuffle": shuffle,
            "random_crop": self.config.random_crop,
            "random_transformers": random_transformers,
            "with_original": with_original,
            "with_label_map": with_label_map,
            "contour_mode": self.config.contour_mode,
            "image_folder": image_folder,
            "rgb_contour_mask_folder": rgb_contour_mask_folder,
            "gray4c_contour_mask_folder": gray4c_contour_mask_folder,
            "bw_contour_mask_folder": bw_contour_mask_folder,
            "dir_distance_map_folder": dir_distance_map_folder,
            "instance_mask_folder": instance_mask_folder,
            "class_mask_folder": class_mask_folder,
        }

        self.test_generator = DataGeneratorCISCA(
            dataset, load_mode=load_mode, **data_kwargs
        )

    def save_config(self):
        """
        Save the current configuration to a JSON file.
        """

        config_filepath = os.path.join(
            self.logdir, "config__" + self.config.string_id + ".json"
        )
        save_json(vars(self.config), config_filepath, cls=MyEncoder)

    def find_and_load_weights(
        self,
        weights_filename=None,
        by_name=False,
        dataset_name="cytoark",
        magnification="40x",
        run_mode="COMPARE",
        train_loss_weights_unnorm=[2, 1, 2, 2, 2],
    ):
        """
        Parameters
        ----------
        weights_filename : str or None, optional
            Filename of the weights to load. If `None`, the method will search for weights
            matching the given criteria. Defaults to `None`.
        by_name : bool, optional
            Whether to load weights by layer name. Defaults to `False`.
        dataset_name : str, optional
            Name of the dataset to match in the weight files. Defaults to `'cytoark'`.
        magnification : str, optional
            Magnification level to match in the weight files. Defaults to `'40x'`.
        run_mode : str, optional
            The run mode to match in the weight files. Defaults to `'COMPARE'`.
        train_loss_weights_unnorm : list, optional
            A list of unnormalized training loss weights to match in the weight files.
            Defaults to `[2, 1, 2, 2, 2]`.

        Returns
        -------
        None
        """

        from itertools import chain

        # get all weight files and sort by validation loss ascending (best first)
        if weights_filename is None:
            weights_files = []
            folder = Path(self.logdir)
            for f in chain(folder.glob("*.hdf5"), folder.glob("*.h5")):
                if all(
                    option in f.name
                    for option in [
                        dataset_name,
                        magnification,
                        run_mode,
                        str(train_loss_weights_unnorm),
                    ]
                ):
                    weights_files.append(f)
            weights_files
            weights_files = list(
                sorted(
                    weights_files,
                    key=lambda f: str(f)[str(f).rfind("__") + len("__") :],
                )
            )
            if len(weights_files) == 0:
                raise ValueError(
                    f"Couldn't find any network weights to load for the following options:dataset_name:{dataset_name},magnification:{magnification},run_mode:{run_mode},train_loss_weights_unnorm:{train_loss_weights_unnorm}."
                )
            weights_chosen = weights_files[0]
            print("Loading network weights from '%s'." % weights_chosen.name)
            self.load_weights(weights_chosen.name, by_name=by_name)
        else:
            weights_chosen = weights_filename
            print("Loading network weights from '%s'." % weights_chosen)
            self.load_weights(weights_chosen, by_name=by_name)

    def load_weights(self, name=None, by_name=False):
        """
        Parameters
        ----------
        name : str, optional
            Name of the HDF5 weight file to load. If not specified, a default name may be used.
        by_name : bool, optional
            Whether to load weights by layer name (default is False). If True, it will only load
            layers that have matching names.

        Returns
        -------
        None
        """

        self.keras_model.load_weights(os.path.join(self.logdir, name), by_name=by_name)

    def tile_predict(
        self,
        X_test,
        input_shape=(256, 256),
        stride_ratio=0.5,
        force_spline=False,
        spline_power=2,
        batch_size=16,
    ):
        """
        Predict output on tiled images.

        Parameters
        ----------
        X_test : numpy.ndarray
            Input image data to be tiled and processed for predictions.
        input_shape : tuple, optional
            The shape of the tiles to create from the input image (default is (256, 256)).
        stride_ratio : float, optional
            The ratio of the stride for overlapping tiles (default is 0.5).
        force_spline : bool, optional
            Whether to apply a spline force when untiling predictions (default is False).
        spline_power : int, optional
            The power parameter for spline interpolation when untiling predictions (default is 2).
        batch_size : int, optional
            The number of tiles to process in each batch during prediction (default is 16).

        Returns
        -------
        y_pred : list
            A list containing the predictions from the model for each output type,
            formatted as numpy arrays.
        """

        # Tile X_test into overlapping tiles
        X_tiles, tiles_info_X = tile_image(
            X_test, model_input_shape=input_shape, stride_ratio=stride_ratio
        )

        del X_test
        gc.collect()

        # Predict on tiles
        tot_tiles = X_tiles.shape[0]
        tot_tiles_proc = 1800
        num_pred = int(np.ceil(tot_tiles / tot_tiles_proc))
        if self.multiclass:
            y_pred = [[], [], []]
        else:
            y_pred = [[], []]
        for i in range(0, num_pred):
            print("Processing batch {0} out of {1}".format(i + 1, num_pred))
            y_pred_subs = self.keras_model.predict(
                X_tiles[
                    i * tot_tiles_proc : np.min((tot_tiles, (i + 1) * tot_tiles_proc))
                ],
                batch_size=batch_size,
            )
            y_pred[0].append(y_pred_subs[0])
            y_pred[1].append(y_pred_subs[1])
            if self.multiclass:
                y_pred[2].append(y_pred_subs[2])
            gc.collect()

        y_pred[0] = np.vstack(y_pred[0])
        y_pred[1] = np.vstack(y_pred[1])
        if self.multiclass:
            y_pred[2] = np.vstack(y_pred[2])

        # Untile predictions
        y_pred = [
            untile_image(
                o,
                tiles_info_X,
                model_input_shape=input_shape,
                power=spline_power,
                force=force_spline,
            )
            for o in y_pred
        ]
        return y_pred

    def predict(
        self,
        input_images=None,
        watershed_line=False,
        th=0.57,
        save_pred=False,
        y_pred_filepath=None,
        small_objects_threshold=None,
    ):
        """
        Perform predictions using the model and optionally apply postprocessing.

        Parameters
        ----------
        input_images : list or None, optional
            A list of input images to be processed. If set to None, the method will load predictions
            from a specified file.

        watershed_line : bool, optional
            If True, applies watershed line processing during postprocessing. Defaults to False.

        th : float, optional
            Threshold value used during postprocessing. Defaults to 0.57.

        save_pred : bool, optional
            If True, saves the raw predictions to the specified file path. Defaults to False.

        y_pred_filepath : str or None, optional
            File path to save or load raw predictions. Must be provided if `save_pred` is True or
            if `input_images` is None.

        small_objects_threshold : float or None, optional
            Threshold value for filtering small objects during postprocessing. Defaults to None.

        Returns
        -------
        processed_predictions : array
            The processed predictions after applying the specified postprocessing steps.

            Notes
        -----
        - This method performs a network forward pass followed by postprocessing or only postprocessing
        if predictions from the forward pass have been previously saved. This is useful for
        testing different postprocessing hyperparameters.

        - Option 1: network forward pass + postprocessing. `input_images` must be provided.
        Optionally set `save_pred` to True and provide a valid `y_pred_filepath` to save raw predictions.

        - Option 2: postprocessing only. `input_images` must be left as None.
        `y_pred_filepath` must be set to a valid file path to retrieve raw predictions for postprocessing.
        """

        if input_images is not None:
            y_pred = self.tile_predict(
                input_images,
                self.config.input_shape,
                stride_ratio=0.5,
                force_spline=False,
                spline_power=2,
                batch_size=1,
            )

            if save_pred:
                if y_pred_filepath is None:
                    raise ValueError(
                        "Filepath must be provided to save the model raw predictions before postprocessing"
                    )
                else:
                    print(
                        "Saving the model raw predictions before postprocessing in: {}".format(
                            y_pred_filepath
                        )
                    )
                    with open(y_pred_filepath, "wb") as handle:
                        pickle.dump(y_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print(
                "Loading the model raw predictions before postprocessing from: {}".format(
                    y_pred_filepath
                )
            )
            with open(y_pred_filepath, "rb") as handle:
                y_pred = pickle.load(handle)

        return postprocess(
            y_pred,
            watershed_line=watershed_line,
            multiclass=self.multiclass,
            th=th,
            magnification=self.config.magnification,
            small_objects_threshold=small_objects_threshold,
        )

    @property
    def _config_class(self):
        return ConfigCISCA
