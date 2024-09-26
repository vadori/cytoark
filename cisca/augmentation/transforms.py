import random
import numpy as np
import cv2
import pathlib
import os
from imgaug import augmenters as iaa
from cisca.augmentation.randstainna import RandStainNA
from cisca.augmentation.composition import Compose, OneOf
import cisca.augmentation.functional as F
from cisca.augmentation.stain_normalization import (
    VahadaneNormalizer,
    RuifrokNormalizer,
    MacenkoNormalizer,
    ReinhardNormalizer,
)

######## loading all the images to use as references normalization and augmentation
# rootfolder = os.path.join(pathlib.Path().resolve().parents[3],"datasets")
rootfolder = os.path.join(pathlib.Path(__file__).parents[2], "datasets")
# print("The parent directory from current folder is:", rootfolder)

#### loading images for HE
# for normalization
fullname0 = os.path.join(rootfolder, "extra/HE/normalization", "HE1.png")

target_image0 = cv2.cvtColor(cv2.imread(fullname0, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

#### loading images for Nissl
fullname5 = os.path.join(rootfolder, "extra/Nissl", "NisslStain1.jpg")
fullname6 = os.path.join(rootfolder, "extra/Nissl", "NisslStain2.png")
fullname7 = os.path.join(rootfolder, "extra/Nissl", "NisslStain3.jpg")
fullname8 = os.path.join(rootfolder, "extra/Nissl", "NisslStain4.png")
fullname9 = os.path.join(rootfolder, "extra/Nissl", "NisslStain5.jpg")
fullname10 = os.path.join(rootfolder, "extra/Nissl", "NisslStain9.jpg")
target_image5 = cv2.cvtColor(cv2.imread(fullname5, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
target_image6 = cv2.cvtColor(cv2.imread(fullname6, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
target_image7 = cv2.cvtColor(cv2.imread(fullname7, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
target_image8 = cv2.cvtColor(cv2.imread(fullname8, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
target_image9 = cv2.cvtColor(cv2.imread(fullname9, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
target_image10 = cv2.cvtColor(
    cv2.imread(fullname10, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
)
# target_image11 = cv2.cvtColor(cv2.imread(fullname11, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
stain_normalizer5 = VahadaneNormalizer()
stain_normalizer5.fit(target_image5)
stain_normalizer6 = VahadaneNormalizer()
stain_normalizer6.fit(target_image6)
stain_normalizer7 = VahadaneNormalizer()
stain_normalizer7.fit(target_image7)
stain_normalizer8 = VahadaneNormalizer()
stain_normalizer8.fit(target_image8)
stain_normalizer9 = VahadaneNormalizer()
stain_normalizer9.fit(target_image9)
stain_normalizer10 = MacenkoNormalizer()
stain_normalizer10.fit(target_image10)
# stain_normalizer11 = VahadaneNormalizer()
# stain_normalizer11.fit(target_image11)


# target_image1 = cv2.cvtColor(cv2.imread(fullname1, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
# target_image2 = cv2.cvtColor(cv2.imread(fullname2, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
# target_image3 = cv2.cvtColor(cv2.imread(fullname3, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
# target_image4 = cv2.cvtColor(cv2.imread(fullname4, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
stain_normalizer0 = VahadaneNormalizer()
stain_normalizer0.fit(target_image0)
stain_normalizer1 = RuifrokNormalizer()
stain_normalizer1.fit(target_image0)
stain_normalizer2 = MacenkoNormalizer()
stain_normalizer2.fit(target_image0)
stain_normalizer3 = ReinhardNormalizer()
stain_normalizer3.fit(target_image0)

stain_normalizersNISSLaug = [
    stain_normalizer5,
    stain_normalizer10,
    stain_normalizer0,
    stain_normalizer6,
    stain_normalizer7,
    stain_normalizer8,
    stain_normalizer9,
    stain_normalizer10,
    stain_normalizer0,
]

stain_normalizersHEnorm = [
    stain_normalizer0,
    stain_normalizer1,
    stain_normalizer2,
    stain_normalizer3,
]


def to_tuple(param, low=None):
    if isinstance(param, tuple):
        return param
    else:
        return (-param if low is None else low, param)


class BasicTransform:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, **kwargs):
        if random.random() < self.prob:
            params = self.get_params()
            # print(type(self).__name__)
            return {
                k: self.apply(a, **params) if k in self.targets else a
                for k, a in kwargs.items()
            }
        return kwargs

    def apply(self, img, **params):
        raise NotImplementedError

    def get_params(self):
        return {}

    @property
    def targets(self):
        # you must specify targets in subclass
        # for example: ('image', 'mask')
        #              ('image', 'boxes')
        raise NotImplementedError


class BasicIAATransform(BasicTransform):
    def __init__(self, prob=0.5):
        super().__init__(prob)
        self.processor = iaa.Noop()
        self.deterministic_processor = iaa.Noop()

    def __call__(self, **kwargs):
        # print("calledBasicIAATransform")
        self.deterministic_processor = self.processor.to_deterministic()
        return super().__call__(**kwargs)

    def apply(self, img, **params):
        # print("applyBasicIAATransform")
        return self.deterministic_processor.augment_image(img)


class DualTransform(BasicTransform):
    """
    transfrom for segmentation task
    """

    @property
    def targets(self):
        return "image", "mask", "mask2", "mask3"


# class TrialTransform(BasicTransform):
#     """
#     transfrom for segmentation task
#     """
#     @property
#     def targets(self):
#         return 'image', 'mask', 'mask2'


class DualIAATransform(DualTransform, BasicIAATransform):
    pass


class ImageOnlyTransform(BasicTransform):
    """
    transforms applied to image only
    """

    @property
    def targets(self):
        return "image"


class ImageOnlyIAATransform(ImageOnlyTransform, BasicIAATransform):
    pass


class VerticalFlip(DualTransform):
    def apply(self, img, **params):
        return F.vflip(img)


class HorizontalFlip(DualTransform):
    def apply(self, img, **params):
        return F.hflip(img)


class Flip(DualTransform):
    def apply(self, img, d=0):
        return F.random_flip(img, d)

    def get_params(self):
        return {"d": random.randint(-1, 1)}


class Transpose(DualTransform):
    def apply(self, img, **params):
        return F.transpose(img)


class RandomRotate90(DualTransform):
    def apply(self, img, factor=0):
        # print("transfprintRandomRotate90")
        return np.ascontiguousarray(np.rot90(img, factor))

    def get_params(self):
        return {"factor": random.randint(0, 4)}


class Rotate(DualTransform):
    def __init__(self, limit=90, prob=0.5):
        super().__init__(prob)
        self.limit = to_tuple(limit)

    def apply(self, img, angle=0):
        # print("transfprintRotate")
        return F.rotate(img, angle)

    def get_params(self):
        return {"angle": random.uniform(self.limit[0], self.limit[1])}


class ShiftScaleRotate(DualTransform):
    # def __init__(self, shift_limit=0.0625, scale_limit=0.05, rotate_limit=135, prob=0.5):
    def __init__(self, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, prob=0.5):
        super().__init__(prob)
        self.shift_limit = to_tuple(shift_limit)
        self.scale_limit = to_tuple(scale_limit)
        self.rotate_limit = to_tuple(rotate_limit)

    def apply(self, img, angle=0, scale=0, dx=0, dy=0):
        # print("transfprintShiftScaleRotate")
        return F.shift_scale_rotate(img, angle, scale, dx, dy)

    def get_params(self):
        return {
            "angle": random.uniform(self.rotate_limit[0], self.rotate_limit[1]),
            "scale": random.uniform(1 + self.scale_limit[0], 1 + self.scale_limit[1]),
            "dx": round(random.uniform(self.shift_limit[0], self.shift_limit[1])),
            "dy": round(random.uniform(self.shift_limit[0], self.shift_limit[1])),
        }


class CenterCrop(DualTransform):
    def __init__(self, height, width, prob=0.5):
        super().__init__(prob)
        self.height = height
        self.width = width

    def apply(self, img, **params):
        # print("transfprintCenterCrop")
        return F.center_crop(img, self.height, self.width)


class Distort1(DualTransform):
    def __init__(self, distort_limit=0.05, shift_limit=0.05, prob=0.5):
        super().__init__(prob)
        self.shift_limit = to_tuple(shift_limit)
        self.distort_limit = to_tuple(distort_limit)
        self.shift_limit = to_tuple(shift_limit)

    def apply(self, img, k=0, dx=0, dy=0):
        # print("transfprintDistort1")
        return F.distort1(img, k, dx, dy)

    def get_params(self):
        return {
            "k": random.uniform(self.distort_limit[0], self.distort_limit[1]),
            "dx": round(random.uniform(self.shift_limit[0], self.shift_limit[1])),
            "dy": round(random.uniform(self.shift_limit[0], self.shift_limit[1])),
        }


class Distort2(DualTransform):
    def __init__(self, num_steps=5, distort_limit=0.3, prob=0.5):
        super().__init__(prob)
        self.num_steps = num_steps
        self.distort_limit = to_tuple(distort_limit)
        self.prob = prob

    def apply(self, img, stepsx=[], stepsy=[]):
        # print("transfprintDistort2")
        return F.distort2(img, self.num_steps, stepsx, stepsy)

    def get_params(self):
        stepsx = [
            1 + random.uniform(self.distort_limit[0], self.distort_limit[1])
            for i in range(self.num_steps + 1)
        ]
        stepsy = [
            1 + random.uniform(self.distort_limit[0], self.distort_limit[1])
            for i in range(self.num_steps + 1)
        ]
        return {"stepsx": stepsx, "stepsy": stepsy}


class ElasticTransform(DualTransform):
    def __init__(self, alpha=1, sigma=50, alpha_affine=50, prob=0.5):
        super().__init__(prob)
        self.alpha = alpha
        self.alpha_affine = alpha_affine
        self.sigma = sigma

    def apply(self, img, random_state=None):
        # print("transfprintElasticTransform")
        return F.elastic_transform_fast(
            img,
            self.alpha,
            self.sigma,
            self.alpha_affine,
            np.random.RandomState(random_state),
        )

    def get_params(self):
        return {"random_state": np.random.randint(0, 100000)}


class ElasticTransform2(DualTransform):
    def __init__(self, grid=5, amount=10, order=1, axis=(0, 1), prob=0.5):
        super().__init__(prob)
        self.grid = grid
        self.amount = amount
        self.order = order
        self.axis = axis

    def apply(self, img, random_state=None):
        return F.transform_elastic(
            img,
            axis=self.axis,
            grid=self.grid,
            amount=self.amount,
            order=self.order,
            random_state=np.random.RandomState(random_state),
        )

    def get_params(self):
        return {"random_state": np.random.randint(0, 100000)}


class RandomStainNorm(ImageOnlyTransform):
    """apply VahadaneNormalizer to normalize images with respect to target image(s)"""

    def __init__(self, stain, prob=0.5):
        # print("VahadaneStainNorm")
        super().__init__(prob)
        if stain == "Nissl":
            self.stain_normalizers = stain_normalizersNISSLaug
        elif stain == "HE":  # dataset == "HE"
            self.stain_normalizers = stain_normalizersHEnorm

    def apply(self, img):
        idx = random.randint(0, len(self.stain_normalizers) - 1)
        # print(idx)
        return self.stain_normalizers[idx].transform(img)


class VahadaneHEStainBaseNorm(ImageOnlyTransform):
    """apply VahadaneNormalizer to normalize images with respect to target image, only HE"""

    def __init__(self, prob=0.5):
        # print("VahadaneStainNorm")
        super().__init__(prob)
        self.stain_normalizers = stain_normalizersHEnorm

    def apply(self, img):
        # print(idx)
        return self.stain_normalizers[0].transform(img)


# class RuifrokNormalizer(ImageOnlyTransform):
#     """apply affine intensity shift where background is bright"""
#     def __init__(self,  prob=0.5):
#         super().__init__(prob)

#     def apply(self, img):
#         idx = random.randint(0,len(stain_normalizers)-1)
#         return stain_normalizers[idx].transform(img)

# class MacenkoNormalizer(ImageOnlyTransform):
#     """apply affine intensity shift where background is bright"""
#     def __init__(self,  prob=0.5):
#         super().__init__(prob)

#     def apply(self, img):
#         idx = random.randint(0,len(stain_normalizers)-1)
#         #print(idx)
#         return stain_normalizers[idx].transform(img)

# class ReinhardNormalizer(ImageOnlyTransform):
#     """apply affine intensity shift where background is bright"""
#     def __init__(self,  prob=0.5):
#         super().__init__(prob)

#     def apply(self, img):
#         idx = random.randint(0,len(stain_normalizers)-1)
#         #print(idx)
#         return stain_normalizers[idx].transform(img)


class RandStainingNA(ImageOnlyTransform):
    """apply RandStainNA to augment images using statistics extracted from data"""

    def __init__(self, dataset, prob=0.5):
        super().__init__(prob)
        yaml_file = os.path.join(
            pathlib.Path(__file__).parents[2],
            "datasets/extra/HE/augmentation",
            dataset + ".yaml",
        )
        print("Using {} for RandStainingNA augmentation".format(yaml_file))
        if not os.path.isfile(yaml_file):
            raise FileNotFoundError("{} was not found".format(yaml_file))
        self.randstainna = RandStainNA(
            yaml_file=yaml_file,
            std_hyper=-0.4,
            distribution="normal",
            probability=1.0,
            is_train=True,
        )

    def apply(self, img):
        # print(idx)
        return self.randstainna(img)


class HEStaining(ImageOnlyTransform):
    def __init__(self, amount_matrix=0.15, amount_stains=0.4, prob=0.5):
        super().__init__(prob)
        self.amount_matrix = amount_matrix
        self.amount_stains = amount_stains

    def apply(self, img):
        return F.he_staining(
            img, amount_matrix=self.amount_matrix, amount_stains=self.amount_stains
        )


class HueBrightnessSaturation(ImageOnlyTransform):
    """apply affine intensity shift where background is bright"""

    def __init__(self, hue=0.1, brightness=0, saturation=1, prob=0.5):
        super().__init__(prob)
        self.hue = hue
        self.brightness = brightness
        self.saturation = saturation

    def apply(self, img):
        return F.hbs_adjust(
            img, hue=self.hue, brightness=self.brightness, saturation=self.saturation
        )


class HueSaturationValue(ImageOnlyTransform):
    def __init__(
        self, hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=20, prob=0.5
    ):
        # def __init__(self, hue_shift_limit=100, sat_shift_limit=50, val_shift_limit=10, prob=0.5, targets=('image')):
        super().__init__(prob)
        self.hue_shift_limit = to_tuple(hue_shift_limit)
        self.sat_shift_limit = to_tuple(sat_shift_limit)
        self.val_shift_limit = to_tuple(val_shift_limit)

    def apply(self, image, hue_shift=0, sat_shift=0, val_shift=0):
        assert (
            image.dtype == np.uint8
            or image.dtype == np.uint32
            or self.hue_shift_limit < 1
        )
        # print("hue_shift: {0}, sat_shift: {1}, val_shift: {2}".format(hue_shift, sat_shift, val_shift))
        return F.shift_hsv(image, hue_shift, sat_shift, val_shift)

    def get_params(self):
        return {
            "hue_shift": np.random.uniform(
                self.hue_shift_limit[0], self.hue_shift_limit[1]
            ),
            "sat_shift": np.random.uniform(
                self.sat_shift_limit[0], self.sat_shift_limit[1]
            ),
            "val_shift": np.random.uniform(
                self.val_shift_limit[0], self.val_shift_limit[1]
            ),
        }


class RGBShift(ImageOnlyTransform):
    def __init__(self, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, prob=0.5):
        super().__init__(prob)
        self.r_shift_limit = to_tuple(r_shift_limit)
        self.g_shift_limit = to_tuple(g_shift_limit)
        self.b_shift_limit = to_tuple(b_shift_limit)

    def apply(self, image, r_shift=0, g_shift=0, b_shift=0):
        return F.shift_rgb(image, r_shift, g_shift, b_shift)

    def get_params(self):
        return {
            "r_shift": np.random.uniform(self.r_shift_limit[0], self.r_shift_limit[1]),
            "g_shift": np.random.uniform(self.g_shift_limit[0], self.g_shift_limit[1]),
            "b_shift": np.random.uniform(self.b_shift_limit[0], self.b_shift_limit[1]),
        }


class RandomBrightness(ImageOnlyTransform):
    def __init__(self, limit=0.2, prob=0.6):
        super().__init__(prob)
        self.limit = to_tuple(limit)

    def apply(self, img, alpha=1):
        # print(alpha)
        return F.random_brightness(img, alpha)

    def get_params(self):
        return {"alpha": 1.0 + np.random.uniform(self.limit[0], self.limit[1])}


class RandomContrast(ImageOnlyTransform):
    def __init__(self, limit=0.15, prob=0.5):
        super().__init__(prob)
        self.limit = to_tuple(limit)

    def apply(self, img, alpha=1.05):
        return F.random_contrast(img, alpha)

    def get_params(self):
        return {"alpha": 1.0 + np.random.uniform(self.limit[0], self.limit[1])}


class Blur(ImageOnlyTransform):
    def __init__(self, blur_limit=7, prob=0.5):
        super().__init__(prob)
        self.blur_limit = to_tuple(blur_limit, 3)

    def apply(self, image, ksize=3):
        return F.blur(image, ksize)

    def get_params(self):
        return {
            "ksize": np.random.choice(
                np.arange(self.blur_limit[0], self.blur_limit[1] + 1, 2)
            )
        }


class MotionBlur(Blur):
    def apply(self, img, ksize=3):
        return F.motion_blur(img, ksize=ksize)


class MedianBlur(Blur):
    def apply(self, image, ksize=3):
        return F.median_blur(image, ksize)


class Remap(ImageOnlyTransform):
    backgrounds = {
        "gray": [216, 222, 222],
        "dark-gray": [206, 202, 202],
        "purple": [181, 159, 210],
        # 'pink': [211, 151, 204],
        "pink-gray": [235, 211, 235],
        # 'pink2': [208, 145, 202]
    }

    nuclei_max = {
        "gray": [80, 20, 105],
        "pink": [117, 12, 127],
        "purple": [82, 23, 192],
    }

    nuclei_center = {
        "gray": [149, 81, 168],
        "purple": [136, 107, 204],
        "pink": [170, 62, 176],
    }

    def apply(self, img, bg=[], center=[], max=[]):
        # print("transfprintElasticRemap")
        return F.remap_color(img, bg, center, max)

    def get_params(self):
        bg = random.choice(list(Remap.backgrounds.values()))
        center = random.choice(list(Remap.nuclei_center.values()))
        max = random.choice(list(Remap.nuclei_max.values()))
        return {"bg": bg, "center": center, "max": max}


class GaussNoise(ImageOnlyTransform):
    def __init__(self, var_limit=(10, 20), prob=0.5):
        super().__init__(prob)
        self.var_limit = to_tuple(var_limit)

    def apply(self, img, var=30):
        return F.gauss_noise(img, var=var)

    def get_params(self):
        return {"var": np.random.randint(self.var_limit[0], self.var_limit[1])}


class CLAHE(ImageOnlyTransform):
    def __init__(self, clipLimit=1.4, tileGridSize=(8, 8), prob=0.5):
        super().__init__(prob)
        # print("CLAHE")
        self.clipLimit = to_tuple(clipLimit, 0.6)
        self.tileGridSize = tileGridSize

    def apply(self, img, clipLimit=2):
        return F.clahe(img, clipLimit, self.tileGridSize)

    def get_params(self):
        return {"clipLimit": np.random.uniform(self.clipLimit[0], self.clipLimit[1])}


class IAAEmboss(ImageOnlyIAATransform):
    def __init__(self, alpha=(0.05, 0.1), strength=(0.05, 0.1), prob=0.5):
        super().__init__(prob)
        # print("IAAEmboss")
        self.processor = iaa.Emboss(alpha, strength)


class IAASuperpixels(ImageOnlyIAATransform):
    """
    may be slow
    """

    def __init__(self, p_replace=0.1, n_segments=100, prob=0.5):
        super().__init__(prob)
        self.processor = iaa.Superpixels(p_replace=p_replace, n_segments=n_segments)


class IAASharpen(ImageOnlyIAATransform):
    def __init__(self, alpha=(0.05, 0.1), lightness=(0.9, 1.1), prob=0.5):
        super().__init__(prob)
        # print("IAASharpen")
        self.processor = iaa.Sharpen(alpha, lightness)


class IAAAdditiveGaussianNoise(ImageOnlyIAATransform):
    def __init__(self, loc=0, scale=(0.01 * 255, 0.03 * 255), prob=0.5):
        super().__init__(prob)
        self.processor = iaa.AdditiveGaussianNoise(loc, scale)


class IAAPiecewiseAffine(DualIAATransform):
    def __init__(self, scale=(0.03, 0.05), nb_rows=4, nb_cols=4, prob=0.5):
        super().__init__(prob)
        self.processor = iaa.PiecewiseAffine(scale, nb_rows, nb_cols)


class IAAPerspective(DualIAATransform):
    def __init__(self, scale=(0.05, 0.1), prob=0.5):
        super().__init__(prob)
        self.processor = iaa.PerspectiveTransform(scale)


class IAARotate(DualIAATransform):
    def __init__(self, rotate=([-90, 90, 180]), prob=0.5):
        super().__init__(prob)
        self.processor = iaa.Affine(rotate=rotate)


class ChannelShuffle(ImageOnlyTransform):
    def apply(self, img, **params):
        return F.channel_shuffle(img)


class InvertImg(ImageOnlyTransform):
    def apply(self, img, **params):
        return F.invert(img)


class ToThreeChannelGray(ImageOnlyTransform):
    def __init__(self, prob=1.0):
        super().__init__(prob)

    def apply(self, img, **params):
        return F.to_three_channel_gray(img)


class ToGray(ImageOnlyTransform):
    def apply(self, img, **params):
        return F.to_gray(img)


class RandomLine(ImageOnlyTransform):
    def apply(self, img, **params):
        return F.random_polosa(img)


class AddChannel(ImageOnlyTransform):
    def __init__(self, prob=1.0):
        super().__init__(prob)

    def apply(self, img, **params):
        return F.add_channel(img)


class FixMasks(BasicTransform):
    def __init__(self, prob=1.0):
        super().__init__(prob)

    def apply(self, img, **params):
        return F.fix_mask(img)

    @property
    def targets(self):
        return ("mask",)


class ToTensor(BasicTransform):
    def __init__(self, num_classes=1, sigmoid=True):
        super().__init__(1.0)
        self.num_classes = num_classes
        self.sigmoid = sigmoid

    def __call__(self, **kwargs):
        kwargs.update({"image": F.img_to_tensor(kwargs["image"])})
        if "mask" in kwargs.keys():
            kwargs.update(
                {
                    "mask": F.mask_to_tensor(
                        kwargs["mask"], self.num_classes, sigmoid=self.sigmoid
                    )
                }
            )
        return kwargs

    @property
    def targets(self):
        raise NotImplementedError


def augNissl(dataset):
    return Compose(
        [
            OneOf(
                [
                    RandomStainNorm(stain="Nissl", prob=0.5),
                    RandStainingNA(dataset=dataset, prob=0.5),
                ],
                prob=0.7,
            ),
            RandomRotate90(prob=0.9),
            Flip(prob=0.9),
            ElasticTransform2(
                grid=5, amount=10, order=0, axis=(0, 1), prob=0.7
            ),  # order 0 interpolates --> bad
            OneOf(
                [
                    MotionBlur(prob=0.15),
                    MedianBlur(blur_limit=3, prob=0.15),
                    Blur(blur_limit=3, prob=0.15),
                    GaussNoise(prob=0.2),
                    IAASharpen(prob=0.2),
                    IAAEmboss(prob=0.15),
                ],
                prob=0.15,
            ),
            OneOf(
                [CLAHE(prob=0.5), RandomContrast(prob=0.5), RandomBrightness(prob=0.2)],
                prob=0.7,
            ),
            OneOf(
                [
                    HueSaturationValue(prob=0.5),
                    HueBrightnessSaturation(
                        hue=0.1, brightness=0.1, saturation=(1, 1), prob=0.5
                    ),
                ],
                prob=0.65,
            ),
            IAAAdditiveGaussianNoise(prob=0.15),
        ],
        prob=0.99,
        name="augNissl",
    )


# def augHE(dataset):
#     return Compose([
#         OneOf([
#             VahadaneHEStainBaseNorm(prob=0.05),
#             RandomStainNorm(stain="HE",prob=0.375),
#             RandStainingNA(dataset=dataset,prob=0.575)
#         ], prob=0.9),
#         RandomRotate90(prob=0.9),
#         Flip(prob=0.9),
#         ElasticTransform2(grid=5, amount=10, order=0, axis=(0,1), prob=0.9),  # order 0 interpolates --> bad
#         OneOf([
#             MotionBlur(prob=0.15),
#             MedianBlur(blur_limit=3, prob=0.15),
#             Blur(blur_limit=3, prob=0.15),
#             GaussNoise(prob=0.2),
#             IAASharpen(prob=0.2),
#             IAAEmboss(prob=0.15)
#         ], prob=0.3),
#         OneOf([
#             CLAHE(prob=0.5),
#             RandomContrast(prob=0.5),
#             RandomBrightness(prob=0.2)
#         ], prob=0.9),
#         OneOf([
#             HueSaturationValue(prob=0.5),
#             HueBrightnessSaturation(hue=0.1, brightness=0.1, saturation=(1,1),prob=0.5),
#         ], prob=0.9),
#         IAAAdditiveGaussianNoise(prob=0.25),
#         ], prob=0.99,name="augHE")
#         # ,
#         # OneOf([
#         # Rotate(prob=0.5),
#         # ShiftScaleRotate(shift_limit=0, scale_limit=scale_limits, rotate_limit=135, prob=0.5)], prob=0.4),
#         # OneOf([
#         #     Distort1(prob=.5),
#         #     Distort2(prob=.5)
#         # ], prob=0.15)

def augHE(dataset):
    return Compose(
        [
            OneOf(
                [
                    VahadaneHEStainBaseNorm(prob=0.05),
                    RandomStainNorm(stain="HE", prob=0.375),
                    RandStainingNA(dataset=dataset, prob=0.575),
                ],
                prob=0.7,
            ),
            RandomRotate90(prob=0.9),
            Flip(prob=0.9),
            ElasticTransform2(
                grid=5, amount=10, order=0, axis=(0, 1), prob=0.7
            ),  # order 0 interpolates --> bad
            OneOf(
                [
                    MotionBlur(prob=0.15),
                    MedianBlur(blur_limit=3, prob=0.15),
                    Blur(blur_limit=3, prob=0.15),
                    GaussNoise(prob=0.2),
                    IAASharpen(prob=0.2),
                    IAAEmboss(prob=0.15),
                ],
                prob=0.15,
            ),
            OneOf(
                [CLAHE(prob=0.5), RandomContrast(prob=0.5), RandomBrightness(prob=0.2)],
                prob=0.7,
            ),
            OneOf(
                [
                    HueSaturationValue(prob=0.5),
                    HueBrightnessSaturation(
                        hue=0.1, brightness=0.1, saturation=(1, 1), prob=0.5
                    ),
                ],
                prob=0.65,
            ),
            IAAAdditiveGaussianNoise(prob=0.15),
        ],
        prob=0.99,
        name="augHE",
    )
    # ,
    # OneOf([
    # Rotate(prob=0.5),
    # ShiftScaleRotate(shift_limit=0, scale_limit=scale_limits, rotate_limit=135, prob=0.5)], prob=0.4),
    # OneOf([
    #     Distort1(prob=.5),
    #     Distort2(prob=.5)
    # ], prob=0.15)


def augHEOnlyVahadaneNorm(dataset):
    return Compose(
        [
            OneOf(
                [
                    VahadaneHEStainBaseNorm(prob=0.1),
                    RandStainingNA(dataset=dataset, prob=0.575),
                ],
                prob=0.7,
            ),
            RandomRotate90(prob=0.9),
            Flip(prob=0.9),
            ElasticTransform2(
                grid=5, amount=10, order=0, axis=(0, 1), prob=0.7
            ),  # order 0 interpolates --> bad
            OneOf(
                [
                    MotionBlur(prob=0.15),
                    MedianBlur(blur_limit=3, prob=0.15),
                    Blur(blur_limit=3, prob=0.15),
                    GaussNoise(prob=0.2),
                    IAASharpen(prob=0.2),
                    IAAEmboss(prob=0.15),
                ],
                prob=0.15,
            ),
            OneOf(
                [CLAHE(prob=0.5), RandomContrast(prob=0.5), RandomBrightness(prob=0.2)],
                prob=0.7,
            ),
            OneOf(
                [
                    HueSaturationValue(prob=0.5),
                    HueBrightnessSaturation(
                        hue=0.1, brightness=0.1, saturation=(1, 1), prob=0.5
                    ),
                ],
                prob=0.65,
            ),
            IAAAdditiveGaussianNoise(prob=0.15),
        ],
        prob=0.99,
        name="augHEOnlyVahadaneNorm",
    )
    # ,
    # OneOf([
    # Rotate(prob=0.5),
    # ShiftScaleRotate(shift_limit=0, scale_limit=scale_limits, rotate_limit=135, prob=0.5)], prob=0.4),
    # OneOf([
    #     Distort1(prob=.5),
    #     Distort2(prob=.5)
    # ], prob=0.15)
