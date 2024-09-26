# ================================================================================================
# Acknowledgments:
# - Based on https://github.com/CSBDeep/CSBDeep/blob/master/csbdeep/models/config.py
# ================================================================================================


import argparse
import pathlib

DATASETMETA = {
    "conic": {
        "class_names": {
            0: "BACKGROUND",
            1: "Neutrophil",
            2: "Epithelial",
            3: "Lymphocyte",
            4: "Plasma",
            5: "Eosinophil",
            6: "Connective",
        },
        "type_colour": {
            0: (0, 0, 0),
            1: (241, 121, 45),
            2: (0, 255, 0),
            3: (255, 0, 0),
            4: (0, 191, 255),
            5: (0, 80, 201),
            6: (255, 255, 0),
        },
        "magnification": "20x",
    },
    "pannuke": {
        "class_names": {
            0: "BACKGROUND",
            1: "Neoplastic",
            2: "Inflammatory",
            3: "Connective",
            4: "Dead",
            5: "Non-neoplastic/Epithelial",
        },
        "type_colour": {
            0: (0, 0, 0),
            1: (255, 0, 0),
            2: (241, 121, 45),
            3: (255, 255, 0),
            4: (0, 80, 201),
            5: (0, 255, 0),
        },
        "magnification": "40x",
    },
    "cytodark020": {"magnification": "20x", "small_objects_threshold": 20},
    "cytodark040": {"magnification": "40x", "small_objects_threshold": 80},
}


class BaseConfig(argparse.Namespace):
    """
    Base configuration class for managing parameters.

    This class extends `argparse.Namespace` to provide a flexible way to handle
    configuration parameters, allowing for validation and dynamic updates of
    parameters. It is particularly useful for scenarios where configuration
    settings may change at runtime.

    Parameters
    ----------
    allow_new_parameters : bool, optional
        If set to `True`, new parameters can be added during instantiation.
        Defaults to `False`.
    **kwargs : keyword arguments
        Key-value pairs representing configuration parameters to be set on
        the instance. These parameters should match the attributes defined
        within the class.
    """

    def __init__(self, allow_new_parameters=False, **kwargs):
        self.update_parameters(allow_new_parameters, **kwargs)

    def is_valid(self, return_invalid=False):
        return (True, tuple()) if return_invalid else True

    def update_parameters(self, allow_new=False, **kwargs):
        if not allow_new:
            attr_new = []
            for k in kwargs:
                try:
                    getattr(self, k)
                except AttributeError:
                    attr_new.append(k)
            if len(attr_new) > 0:
                raise AttributeError(
                    "Not allowed to add new parameters (%s)" % ", ".join(attr_new)
                )
        for k in kwargs:
            setattr(self, k, kwargs[k])


class ConfigCISCA(BaseConfig):
    """
    Configuration class for the CISCA model, inheriting from BaseConfig.

    This class defines the configuration settings required to instantiate and
    train the CISCA model, including various parameters for the architecture,
    training process, and data handling.

    Parameters
    ----------
    n_rays : int, optional
        Number of radial directions for the star-convex polygon. It is recommended
        to use a power of 2 (default: 32).
    n_channel_in : int, optional
        Number of channels of the input image (default: 1).
    grid : tuple of (int, int), optional
        Subsampling factors for each of the axes. These must be powers of 2.
        The model will predict on a subsampled grid for increased efficiency
        and larger field of view.
    n_classes : int or None, optional
        Number of object classes for multi-class prediction. Set to None to disable.
    backbone : str, optional
        Name of the neural network architecture to be used as the backbone (default: 'unet').
    kwargs : dict, optional
        Additional keyword arguments to overwrite or add configuration attributes.

    Attributes
    ----------
    root_folder : pathlib.Path
        Path to the root directory of the project.
    dataset_name : str
        Name of the dataset (default: "cytoark").
    magnification : str
        Magnification level for image processing (default: "40x").
    raw_input_side : int
        Size of the raw input image (default: 512).
    valid_raw_input_side : int
        Size of the valid raw input image (default: 256).
    model_name : str
        Name of the model (default: "CISCA").
    model_variant : str
        Variant of the model (default: "default").
    input_shape : tuple of (int, int)
        Shape of the input images (default: (256, 256)).
    n_input_channels : int
        Number of input channels for the model (default: 3).
    center_crop : int or None
        Size of the center crop to be applied to images (default: None).
    n_contour_classes : int
        Number of contour classes for segmentation (default: 3).
    n_celltype_classes : int
        Number of cell type classes for classification (default: 1).
    train_class_weights : tuple of float
        Class weights for training (default: (1, 1, 1, 1, 1, 1, 1)).
    dist_regression : bool
        Indicates whether distance regression is used (default: True).
    diag_dist : bool
        Indicates whether diagonal distance is used (default: True).
    masked_regression : bool
        Indicates whether masked regression is enabled (default: True).
    train_epochs : int
        Number of training epochs (default: 100).
    train_batch_size : int
        Batch size for training (default: 8).
    train_learning_rate : float
        Learning rate for training (default: 0.0005).
    train_loss_weights : tuple of float
        Weights for losses relating to probability and distance (default: (2, 1.5, 2, 2, 2, 2)).
    use_gpu : bool
        Indicates whether to use GPU for computations (default: False).
    mirroredstrategy : bool
        Indicates whether to use MirroredStrategy for distributed training (default: True).
    """

    def __init__(self, **kwargs):
        """See class docstring."""

        super().__init__()

        # folders
        self.root_folder = pathlib.Path(__file__).parents[1]
        self.dataset_name = "conic"
        self.magnification = "20x"
        self.raw_input_side = 512
        self.valid_raw_input_side = 256
        self.small_objects_threshold = 26

        # model
        self.model_name = "cisca"
        self.model_variant = "default"
        self.backbone = "unet"  # "EfficientNet-B4"
        self.input_shape = (256, 256)
        self.n_input_channels = 3
        self.center_crop = None  # 80 for hovernet
        self.n_contour_classes = 3
        self.n_celltype_classes = 1
        self.train_class_weights = (1, 1, 1, 1, 1, 1, 1)
        self.train_class_weights_2 = (0, 1, 1, 1, 1, 1, 1)  # (0,1,0,0,1,1,0)
        self.dist_regression = True
        self.diag_dist = True
        self.masked_regression = True
        self.masked_celltype_classification = True
        self.dist_mode = "NORM"  # TIP, NORM, UNNORM
        self.contour_mode = "GRAY4C"  # GRAY4C, RGB, BWGAP, BW

        # training
        self.run_mode = "COMPARE"  # USE
        self.batch_size = 8
        self.steps_per_epoch = 800
        self.random_crop = True
        self.shuffle = True
        self.random_transformers = None  # "aug_mega_hardcore_simp_cytoark()"
        self.valid_random_transformers = None
        self.with_original = False
        self.with_label_map = False
        self.train_checkpoint = None
        self.train_loss_weights_unnorm = [
            2,
            1,
            2,
            5,
            2,
            2,
        ]  # 2, 1, 2, 2, 2 [2, 1.5, 2, 3, 3] # cce contour fg celltype loss mse mge
        self.train_epochs = 100
        self.learning_rate = 0.0005
        self.reduce_lr = {"factor": 0.5, "patience": 10, "min_delta": 0}
        self.earlystop_patience = 20
        self.tensorboard = True
        self.mirroredstrategy = True
        self.pretrained = False

        # update based on backbone
        if kwargs.get("backbone", self.backbone).startswith("Efficient"):
            self.pretrained = kwargs.get("pretrained", True)
            self.encdec_skipconn_attention = kwargs.get(
                "encdec_skipconn_attention", True
            )
            self.decdec_skipconn = kwargs.get("decdec_skipconn", True)
            self.decdec_skipconn_attention = kwargs.get(
                "decdec_skipconn_attention", True
            )
            self.block_type = kwargs.get("block_type", "upsampling")
            self.concat_input = kwargs.get("concat_input", True)
            self.trainable = kwargs.get("trainable", False)
            self.framework = kwargs.get("framework", False)  # TODO: change effeunet!!
            self.pretrained = kwargs.get("pretrained", False)
        elif (kwargs.get("backbone", self.backbone) == "unet") or (
            kwargs.get("backbone", self.backbone) == "attunet"
        ):
            self.unet_grid = kwargs.get("unet_grid", (1, 1))
            self.unet_n_depth = kwargs.get("unet_n_depth", 3)
            self.unet_kernel_size = kwargs.get("unet_kernel_size", (3, 3))
            self.unet_n_filter_base = kwargs.get("unet_n_filter_base", 32)
            self.unet_n_conv_per_depth = kwargs.get("unet_n_conv_per_depth", 2)
            self.unet_pool = kwargs.get("unet_pool", (2, 2))
            self.unet_activation = kwargs.get("unet_activation", "relu")
            self.unet_last_activation = kwargs.get("unet_last_activation", "relu")
            self.unet_batch_norm = kwargs.get("unet_batch_norm", False)
            self.unet_dropout = kwargs.get("unet_dropout", 0.0)
            self.unet_prefix = kwargs.get("unet_prefix", "")
            self.net_conv_after_unet = kwargs.get("net_conv_after_unet", 128)
            self.head_blocks = kwargs.get("head_blocks", 2)
        elif kwargs.get("backbone", self.backbone) == "unetplus":
            self.unet_grid = kwargs.get("unet_grid", (1, 1))
            self.unet_n_depth = kwargs.get("unet_n_depth", 4)
            self.unet_kernel_size = kwargs.get("unet_kernel_size", (3, 3))
            self.unet_n_filter_base = kwargs.get("unet_n_filter_base", 32)
            self.unet_n_conv_per_depth = kwargs.get("unet_n_conv_per_depth", 2)
            self.unet_pool = kwargs.get("unet_pool", (2, 2))
            self.unet_activation = kwargs.get("unet_activation", "elu")
            self.unet_last_activation = kwargs.get("unet_last_activation", "elu")
            # batchnorm is more importnant for resnet blocks
            self.unet_batch_norm = kwargs.get("unet_batch_norm", True)
            self.net_conv_after_unet = kwargs.get("net_conv_after_unet", 128)
            self.head_blocks = kwargs.get("head_blocks", 2)
        elif kwargs.get("backbone", self.backbone) == "mrunet":
            self.unet_grid = kwargs.get("unet_grid", (1, 1))
            self.unet_n_depth = kwargs.get("unet_n_depth", 3)
            self.unet_kernel_size = kwargs.get("unet_kernel_size", (3, 3))
            self.unet_n_filter_base = kwargs.get("unet_n_filter_base", 32)
            self.unet_n_conv_per_depth = kwargs.get("unet_n_conv_per_depth", 2)
            self.unet_pool = kwargs.get("unet_pool", (2, 2))
            self.unet_activation = kwargs.get("unet_activation", "relu")
            self.unet_last_activation = kwargs.get("unet_last_activation", "relu")
            self.unet_batch_norm = kwargs.get("unet_batch_norm", False)
            self.unet_dropout = kwargs.get("unet_dropout", 0.0)
            self.unet_prefix = kwargs.get("unet_prefix", "")
            self.net_conv_after_unet = kwargs.get("net_conv_after_unet", 128)
            self.head_blocks = kwargs.get("head_blocks", 2)
        elif kwargs.get("backbone", self.backbone) == "fpn":
            self.unet_grid = kwargs.get("unet_grid", (1, 1))
            self.unet_n_depth = kwargs.get("unet_n_depth", 4)
            self.unet_kernel_size = kwargs.get("unet_kernel_size", (3, 3))
            self.unet_n_filter_base = kwargs.get("unet_n_filter_base", 32)
            self.unet_n_conv_per_depth = kwargs.get("unet_n_conv_per_depth", 2)
            self.unet_pool = kwargs.get("unet_pool", (2, 2))
            self.unet_activation = kwargs.get("unet_activation", "elu")
            self.unet_last_activation = kwargs.get("unet_last_activation", "elu")
            # batchnorm is more importnant for resnet blocks
            self.unet_batch_norm = kwargs.get("unet_batch_norm", True)
            self.unet_dropout = kwargs.get("unet_dropout", 0.0)
            self.unet_prefix = kwargs.get("unet_prefix", "")
            self.net_conv_after_unet = kwargs.get("net_conv_after_unet", 128)
            self.head_blocks = kwargs.get("head_blocks", 2)
        else:
            # TODO: resnet backbone for 2D model?
            raise ValueError(
                "backbone '%s' not supported." % kwargs.get("backbone", self.backbone)
            )

        # change values of config parameters if given as input parameters
        self.update_parameters(False, **kwargs)

        if self.raw_input_side <= self.input_shape[0]:
            self.random_crop = False
        # self.data_folder = "D:\\NucleiSegmentation\\Projects\\CISCA\\NeuronInstanceSeg-master\\NeuronInstanceSeg-master\\datasets\\conic"
        self.update_string_ID()

    def update_string_ID(self):
        stringList = [
            self.backbone,
            self.dataset_name,
            self.magnification,
            self.run_mode,
            self.n_contour_classes,
            self.n_celltype_classes,
            self.train_class_weights,
            self.train_class_weights_2,
            self.contour_mode,
            self.dist_mode,
            self.raw_input_side,
            self.random_crop,
            self.random_transformers.name
            if self.random_transformers is not None
            else "None",
            self.masked_regression,
            self.masked_celltype_classification,
            self.train_loss_weights_unnorm,
            self.batch_size,
            self.steps_per_epoch,
        ]

        self.string_id = "__".join(str(string) for string in stringList)
        # print(self.string_id)
