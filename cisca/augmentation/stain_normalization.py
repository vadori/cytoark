# ================================================================================================
# Acknowledgments:
# - Based on https://github.com/Peter554/StainTools
# ================================================================================================

from __future__ import annotations
import colorsys
import random
import cv2
import numpy as np
from sklearn.decomposition import DictionaryLearning
from skimage import exposure
import warnings

def rgb2od(img: np.ndarray) -> np.ndarray:
    r"""Convert from RGB to optical density (:math:`OD_{RGB}`) space.

    .. math::
        RGB = 255 * exp^{-1*OD_{RGB}}

    Args:
        img (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
            RGB image.

    Returns:
        :class:`numpy.ndarray`:
            Optical density (OD) RGB image.

    Examples:
        >>> from tiatoolbox.utils import transforms, misc
        >>> rgb_img = misc.imread('path/to/image')
        >>> od_img = transforms.rgb2od(rgb_img)

    """
    mask = img == 0
    img[mask] = 1
    return np.maximum(-1 * np.log(img / 255), 1e-6)


def od2rgb(od: np.ndarray) -> np.ndarray:
    r"""Convert from optical density (:math:`OD_{RGB}`) to RGB.

    .. math::
        RGB = 255 * exp^{-1*OD_{RGB}}

    Args:
        od (:class:`numpy.ndarray`):
            Optical density (OD) RGB image.

    Returns:
        :class:`numpy.ndarray`:
            RGB Image.

    Examples:
        >>> from tiatoolbox.utils import transforms, misc
        >>> rgb_img = misc.imread('path/to/image')
        >>> od_img = transforms.rgb2od(rgb_img)
        >>> rgb_img = transforms.od2rgb(od_img)

    """
    od = np.maximum(od, 1e-6)
    return (255 * np.exp(-1 * od)).astype(np.uint8)

def contrast_enhancer(img: np.ndarray, low_p: int = 2, high_p: int = 98) -> np.ndarray:
    """Enhance contrast of the input image using intensity adjustment.

    This method uses both image low and high percentiles.

    Args:
        img (:class:`numpy.ndarray`): input image used to obtain tissue mask.
            Image should be uint8.
        low_p (scalar): low percentile of image values to be saturated to 0.
        high_p (scalar): high percentile of image values to be saturated to 255.
            high_p should always be greater than low_p.

    Returns:
        img (:class:`numpy.ndarray`):
            Image (uint8) with contrast enhanced.

    Raises:
        AssertionError: Internal errors due to invalid img type.

    Examples:
        >>> from tiatoolbox import utils
        >>> img = utils.misc.contrast_enhancer(img, low_p=2, high_p=98)

    """
    # check if image is not uint8
    if img.dtype != np.uint8:
        msg = "Image should be uint8."
        raise AssertionError(msg)
    img_out = img.copy()
    percentiles = np.array(np.percentile(img_out, (low_p, high_p)))
    p_low, p_high = percentiles[0], percentiles[1]
    if p_low >= p_high:
        p_low, p_high = np.min(img_out), np.max(img_out)
    if p_high > p_low:
        img_out = exposure.rescale_intensity(
            img_out,
            in_range=(p_low, p_high),
            out_range=(0.0, 255.0),
        )
    return img_out.astype(np.uint8)

def get_luminosity_tissue_mask(img: np.ndarray, threshold: float) -> np.ndarray:
    """Get tissue mask based on the luminosity of the input image.

    Args:
        img (:class:`numpy.ndarray`):
            Input image used to obtain tissue mask.
        threshold (float):
            Luminosity threshold used to determine tissue area.

    Returns:
        tissue_mask (:class:`numpy.ndarray`):
            Binary tissue mask.

    Examples:
        >>> from tiatoolbox import utils
        >>> tissue_mask = utils.misc.get_luminosity_tissue_mask(img, threshold=0.8)

    """
    img = img.astype("uint8")  # ensure input image is uint8
    img = contrast_enhancer(img, low_p=2, high_p=98)  # Contrast  enhancement
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_lab = img_lab[:, :, 0] / 255.0  # Convert to range [0,1].
    tissue_mask = l_lab < threshold

    # check it's not empty
    if tissue_mask.sum() == 0:
        msg = "Empty tissue mask computed."
        raise ValueError(msg)

    return tissue_mask

def vectors_in_correct_direction(e_vectors: np.ndarray) -> np.ndarray:
    """Points the eigen vectors in the right direction.

    Args:
        e_vectors (:class:`numpy.ndarray`):
            Eigen vectors.

    Returns:
        :class:`numpy.ndarray`:
            Pointing in the correct direction.

    """
    if e_vectors[0, 0] < 0:
        e_vectors[:, 0] *= -1
    if e_vectors[0, 1] < 0:
        e_vectors[:, 1] *= -1

    return e_vectors


def h_and_e_in_right_order(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Rearrange input vectors for H&E in correct order with H as first output.

    Args:
        v1 (:class:`numpy.ndarray`):
            Input vector for stain extraction.
        v2 (:class:`numpy.ndarray`):
            Input vector for stain extraction.

    Returns:
        :class:`numpy.ndarray`:
            Input vectors in the correct order.

    """
    if v1[0] > v2[0]:
        return np.array([v1, v2])

    return np.array([v2, v1])


def dl_output_for_h_and_e(dictionary: np.ndarray) -> np.ndarray:
    """Return correct value for H and E from dictionary learning output.

    Args:
        dictionary (:class:`numpy.ndarray`):
            :class:`sklearn.decomposition.DictionaryLearning` output

    Returns:
        :class:`numpy.ndarray`:
            With correct values for H and E.

    """
    if dictionary[0, 0] < dictionary[1, 0]:
        return dictionary[[1, 0], :]

    return dictionary


class CustomExtractor:
    """Get the user-defined stain matrix.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Examples:
        >>> from tiatoolbox.tools.stainextract import CustomExtractor
        >>> from tiatoolbox.utils import imread
        >>> extractor = CustomExtractor(stain_matrix)
        >>> img = imread('path/to/image')
        >>> stain_matrix = extractor.get_stain_matrix(img)

    """

    def __init__(self, stain_matrix: np.ndarray) -> None:
        """Initialize :class:`CustomExtractor`."""
        self.stain_matrix = stain_matrix
        if self.stain_matrix.shape not in [(2, 3), (3, 3)]:
            msg = "Stain matrix must have shape (2, 3) or (3, 3)."
            raise ValueError(msg)

    def get_stain_matrix(self, _: np.ndarray) -> np.ndarray:
        """Get the user defined stain matrix.

        Returns:
            :class:`numpy.ndarray`:
                User defined stain matrix.

        """
        return self.stain_matrix


class RuifrokExtractor:
    """Reuifrok stain extractor.

    Get the stain matrix as defined in:

    Ruifrok, Arnout C., and Dennis A. Johnston. "Quantification of
    histochemical staining by color deconvolution." Analytical and
    quantitative cytology and histology 23.4 (2001): 291-299.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Examples:
        >>> from tiatoolbox.tools.stainextract import RuifrokExtractor
        >>> from tiatoolbox.utils import imread
        >>> extractor = RuifrokExtractor()
        >>> img = imread('path/to/image')
        >>> stain_matrix = extractor.get_stain_matrix(img)

    """

    def __init__(self) -> None:
        """Initialize :class:`RuifrokExtractor`."""
        self.__stain_matrix = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])

    def get_stain_matrix(self, _: np.ndarray) -> np.ndarray:
        """Get the pre-defined stain matrix.

        Returns:
            :class:`numpy.ndarray`:
                Pre-defined  stain matrix.

        """
        return self.__stain_matrix.copy()


class MacenkoExtractor:
    """Macenko stain extractor.

    Get the stain matrix as defined in:

    Macenko, Marc, et al. "A method for normalizing histology
    slides for quantitative analysis." 2009 IEEE International
    Symposium on Biomedical Imaging: From Nano to Macro. IEEE, 2009.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Args:
        luminosity_threshold (float):
            Threshold used for tissue area selection
        angular_percentile (int):
            Percentile of angular coordinates to be selected
            with respect to the principle, orthogonal eigenvectors.

    Examples:
        >>> from tiatoolbox.tools.stainextract import MacenkoExtractor
        >>> from tiatoolbox.utils import imread
        >>> extractor = MacenkoExtractor()
        >>> img = imread('path/to/image')
        >>> stain_matrix = extractor.get_stain_matrix(img)

    """

    def __init__(
        self,
        luminosity_threshold: float = 0.8,
        angular_percentile: float = 99,
    ) -> None:
        """Initialize :class:`MacenkoExtractor`."""
        self.__luminosity_threshold = luminosity_threshold
        self.__angular_percentile = angular_percentile

    def get_stain_matrix(self, img: np.ndarray) -> np.ndarray:
        """Stain matrix estimation.

        Args:
            img (:class:`numpy.ndarray`):
                Input image used for stain matrix estimation.

        Returns:
            :class:`numpy.ndarray`:
                Estimated stain matrix.

        """
        img = img.astype("uint8")  # ensure input image is uint8
        luminosity_threshold = self.__luminosity_threshold
        angular_percentile = self.__angular_percentile

        # convert to OD and ignore background
        tissue_mask = get_luminosity_tissue_mask(
            img,
            threshold=luminosity_threshold,
        ).reshape((-1,))
        img_od = rgb2od(img).reshape((-1, 3))
        img_od = img_od[tissue_mask]

        # eigenvectors of covariance in OD space (orthogonal as covariance symmetric)
        _, eigen_vectors = np.linalg.eigh(np.cov(img_od, rowvar=False))

        # the two principle eigenvectors
        eigen_vectors = eigen_vectors[:, [2, 1]]

        # make sure vectors are pointing the right way
        eigen_vectors = vectors_in_correct_direction(e_vectors=eigen_vectors)

        # project on this basis.
        proj = np.dot(img_od, eigen_vectors)

        # angular coordinates with respect to the principle, orthogonal eigenvectors
        phi = np.arctan2(proj[:, 1], proj[:, 0])

        # min and max angles
        min_phi = np.percentile(phi, 100 - angular_percentile)
        max_phi = np.percentile(phi, angular_percentile)

        # the two principle colors
        v1 = np.dot(eigen_vectors, np.array([np.cos(min_phi), np.sin(min_phi)]))
        v2 = np.dot(eigen_vectors, np.array([np.cos(max_phi), np.sin(max_phi)]))

        # order of H&E - H first row
        he = h_and_e_in_right_order(v1, v2)

        return he / np.linalg.norm(he, axis=1)[:, None]

class VahadaneExtractor:
    """Vahadane stain extractor.

    Get the stain matrix as defined in:

    Vahadane, Abhishek, et al. "Structure-preserving color normalization
    and sparse stain separation for histological images."
    IEEE transactions on medical imaging 35.8 (2016): 1962-1971.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Args:
        luminosity_threshold (float):
            Threshold used for tissue area selection.
        regularizer (float):
            Regularizer used in dictionary learning.

    Examples:
        >>> from tiatoolbox.tools.stainextract import VahadaneExtractor
        >>> from tiatoolbox.utils import imread
        >>> extractor = VahadaneExtractor()
        >>> img = imread('path/to/image')
        >>> stain_matrix = extractor.get_stain_matrix(img)

    """

    def __init__(
        self,
        luminosity_threshold: float = 0.8,
        regularizer: float = 0.1,
    ) -> None:
        """Initialize :class:`VahadaneExtractor`."""
        self.__luminosity_threshold = luminosity_threshold
        self.__regularizer = regularizer

    def get_stain_matrix(self, img: np.ndarray) -> np.ndarray:
        """Stain matrix estimation.

        Args:
            img (:class:`numpy.ndarray`):
                Input image used for stain matrix estimation

        Returns:
            :class:`numpy.ndarray`:
                Estimated stain matrix.

        """
        img = img.astype("uint8")  # ensure input image is uint8
        luminosity_threshold = self.__luminosity_threshold
        regularizer = self.__regularizer
        # convert to OD and ignore background
        tissue_mask = get_luminosity_tissue_mask(
            img,
            threshold=luminosity_threshold,
        ).reshape((-1,))
        img_od = rgb2od(img).reshape((-1, 3))
        img_od = img_od[tissue_mask]

        # do the dictionary learning
        dl = DictionaryLearning(
            n_components=2,
            alpha=regularizer,
            transform_alpha=regularizer,
            fit_algorithm="lars",
            transform_algorithm="lasso_lars",
            positive_dict=True,
            verbose=False,
            max_iter=3,
            transform_max_iter=1000,
        )
        dictionary = dl.fit_transform(X=img_od.T).T

        # order H and E.
        # H on first row.
        dictionary = dl_output_for_h_and_e(dictionary)

        return dictionary / np.linalg.norm(dictionary, axis=1)[:, None]

def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x,0,1)

    return x

def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)

def random_label_cmap(n=2**16, h = (0,1), l = (.4,1), s =(.2,.8)):
    import matplotlib
    import colorsys
    # cols = np.random.rand(n,3)
    # cols = np.random.uniform(0.1,1.0,(n,3))
    h,l,s = np.random.uniform(*h,n), np.random.uniform(*l,n), np.random.uniform(*s,n)
    cols = np.stack([colorsys.hls_to_rgb(_h,_l,_s) for _h,_l,_s in zip(h,l,s)],axis=0)
    cols[0] = 0
    return matplotlib.colors.ListedColormap(cols)

def _single_color_integer_cmap(color = (.3,.4,.5)):
    from matplotlib.colors import Colormap
    
    assert len(color) in (3,4)
    
    class BinaryMap(Colormap):
        def __init__(self, color):
            self.color = np.array(color)
            if len(self.color)==3:
                self.color = np.concatenate([self.color,[1]])
        def __call__(self, X, alpha=None, bytes=False):
            res = np.zeros(X.shape+(4,), np.float32)
            res[...,-1] = self.color[-1]
            res[X>0] = np.expand_dims(self.color,0)
            if bytes:
                return np.clip(256*res,0,255).astype(np.uint8)
            else:
                return res
    return BinaryMap(color)

def render_label(lbl, img = None, cmap = None, cmap_img = "gray", alpha = 0.5, alpha_boundary = None, normalize_img = True):
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
        im_img = np.zeros(lbl.shape+(4,),np.float32)
        im_img[...,-1] = 1
        
    else:
        assert lbl.shape[:2] == img.shape[:2]
        img = normalize(img) if normalize_img else img
        if img.ndim==2:
            im_img = cmap_img(img)
        elif img.ndim==3:
            im_img = img[...,:4]
            if img.shape[-1]<4:
                im_img = np.concatenate([img, np.ones(img.shape[:2]+(4-img.shape[-1],))], axis = -1)
        else:
            raise ValueError("img should be 2 or 3 dimensional")
            
                
            
    # render label
    im_lbl = cmap(lbl)

    mask_lbl = lbl>0
    mask_bound = np.bitwise_and(mask_lbl,find_boundaries(lbl, mode = "thick"))
    
    # blend
    im = im_img.copy()
    
    im[mask_lbl] = alpha*im_lbl[mask_lbl]+(1-alpha)*im_img[mask_lbl]
    im[mask_bound] = alpha_boundary*im_lbl[mask_bound]+(1-alpha_boundary)*im_img[mask_bound]
        
    return im

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


class StainNormalizer:
    """Stain normalization base class.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Attributes:
        extractor (CustomExtractor, RuifrokExtractor):
            Method specific stain extractor.
        stain_matrix_target (:class:`numpy.ndarray`):
            Stain matrix of target.
        target_concentrations (:class:`numpy.ndarray`):
            Stain concentration matrix of target.
        maxC_target (:class:`numpy.ndarray`):
            99th percentile of each stain.
        stain_matrix_target_RGB (:class:`numpy.ndarray`):
            Target stain matrix in RGB.

    """

    def __init__(self: StainNormalizer) -> None:
        """Initialize :class:`StainNormalizer`."""
        self.extractor = None
        self.stain_matrix_target = None
        self.target_concentrations = None
        self.maxC_target = None
        self.stain_matrix_target_RGB = None

    @staticmethod
    def get_concentrations(img: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
        """Estimate concentration matrix given an image and stain matrix.

        Args:
            img (:class:`numpy.ndarray`):
                Input image.
            stain_matrix (:class:`numpy.ndarray`):
                Stain matrix for haematoxylin and eosin stains.

        Returns:
            numpy.ndarray:
                Stain concentrations of input image.

        """
        od = rgb2od(img).reshape((-1, 3))
        x, _, _, _ = np.linalg.lstsq(stain_matrix.T, od.T, rcond=-1)
        return x.T

    def fit(self: StainNormalizer, target: np.ndarray) -> None:
        """Fit to a target image.

        Args:
            target (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
              Target/reference image.

        """
        self.stain_matrix_target = self.extractor.get_stain_matrix(target)
        self.target_concentrations = self.get_concentrations(
            target,
            self.stain_matrix_target,
        )
        self.maxC_target = np.percentile(
            self.target_concentrations,
            99,
            axis=0,
        ).reshape((1, 2))
        # useful to visualize.
        self.stain_matrix_target_RGB = od2rgb(self.stain_matrix_target)

    def transform(self: StainNormalizer, img: np.ndarray) -> np.ndarray:
        """Transform an image.

        Args:
            img (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
                RGB input source image.

        Returns:
            :class:`numpy.ndarray`:
                RGB stain normalized image.

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stain_matrix_source = self.extractor.get_stain_matrix(img)
                source_concentrations = self.get_concentrations(img, stain_matrix_source)
                max_c_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
                source_concentrations *= self.maxC_target / max_c_source
                trans = 255 * np.exp(
                    -1 * np.dot(source_concentrations, self.stain_matrix_target),
                )
                # ensure between 0 and 255
                trans[trans > 255] = 255  # noqa: PLR2004
                trans[trans < 0] = 0
                res = trans.reshape(img.shape).astype(np.uint8)
            except:
                res = img
        return res
   
class CustomNormalizer(StainNormalizer):
    """Stain Normalization using a user-defined stain matrix.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Args:
        stain_matrix (:class:`numpy.ndarray`):
            User-defined stain matrix. Must be either 2x3 or 3x3.

    Examples:
        >>> from tiatoolbox.tools.stainnorm import CustomNormalizer
        >>> norm = CustomNormalizer(stain_matrix)
        >>> norm.fit(target_img)
        >>> norm_img = norm.transform(source_img)

    """

    def __init__(self: StainNormalizer, stain_matrix: np.ndarray) -> None:
        """Initialize :class:`CustomNormalizer`."""
        super().__init__()

        self.extractor = CustomExtractor(stain_matrix)


class RuifrokNormalizer(StainNormalizer):
    """Ruifrok & Johnston stain normalizer.

    Normalize a patch to the stain appearance of the target image using
    the method of:

    Ruifrok, Arnout C., and Dennis A. Johnston. "Quantification of
    histochemical staining by color deconvolution." Analytical and
    quantitative cytology and histology 23.4 (2001): 291-299.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Examples:
        >>> from tiatoolbox.tools.stainnorm import RuifrokNormalizer
        >>> norm = RuifrokNormalizer()
        >>> norm.fit(target_img)
        >>> norm_img = norm.transform(source_img)

    """

    def __init__(self: StainNormalizer) -> None:
        """Initialize :class:`RuifrokNormalizer`."""
        super().__init__()
        self.extractor = RuifrokExtractor()


class MacenkoNormalizer(StainNormalizer):
    """Macenko stain normalizer.

    Normalize a patch to the stain appearance of the target image using
    the method of:

    Macenko, Marc, et al. "A method for normalizing histology slides for
    quantitative analysis." 2009 IEEE International Symposium on
    Biomedical Imaging: From Nano to Macro. IEEE, 2009.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Examples:
        >>> from tiatoolbox.tools.stainnorm import MacenkoNormalizer
        >>> norm = MacenkoNormalizer()
        >>> norm.fit(target_img)
        >>> norm_img = norm.transform(source_img)

    """

    def __init__(self: StainNormalizer) -> None:
        """Initialize :class:`MacenkoNormalizer`."""
        super().__init__()
        self.extractor = MacenkoExtractor()


class VahadaneNormalizer(StainNormalizer):
    """Vahadane stain normalizer.

    Normalize a patch to the stain appearance of the target image using
    the method of:

    Vahadane, Abhishek, et al. "Structure-preserving color normalization
    and sparse stain separation for histological images." IEEE
    transactions on medical imaging 35.8 (2016): 1962-1971.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Examples:
        >>> from tiatoolbox.tools.stainnorm import VahadaneNormalizer
        >>> norm = VahadaneNormalizer()
        >>> norm.fit(target_img)
        >>> norm_img = norm.transform(source_img)

    """

    def __init__(self: StainNormalizer) -> None:
        """Initialize :class:`VahadaneNormalizer`."""
        super().__init__()
        self.extractor = VahadaneExtractor()


class ReinhardNormalizer:
    """Reinhard colour normalizer.

    Normalize a patch colour to the target image using the method of:

    Reinhard, Erik, et al. "Color transfer between images." IEEE
    Computer graphics and applications 21.5 (2001): 34-41.

    This class contains code inspired by StainTools
    [https://github.com/Peter554/StainTools] written by Peter Byfield.

    Attributes:
        target_means (float):
            Mean of each LAB channel.
        target_stds (float):
            Standard deviation of each LAB channel.

    Examples:
        >>> from tiatoolbox.tools.stainnorm import ReinhardNormalizer
        >>> norm = ReinhardNormalizer()
        >>> norm.fit(target_img)
        >>> norm_img = norm.transform(src_img)

    """

    def __init__(self: ReinhardNormalizer) -> None:
        """Initialize :class:`ReinhardNormalizer`."""
        self.target_means = None
        self.target_stds = None

    def fit(self: ReinhardNormalizer, target: np.ndarray) -> None:
        """Fit to a target image.

        Args:
            target (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
                Target image.

        """
        means, stds = self.get_mean_std(target)
        self.target_means = means
        self.target_stds = stds

    def transform(self: ReinhardNormalizer, img: np.ndarray) -> np.ndarray:
        """Transform an image.

        Args:
            img (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
                Input image.

        Returns:
            :class:`numpy.ndarray` of type :class:`numpy.float`:
                Colour normalized RGB image.

        """
        chan1, chan2, chan3 = self.lab_split(img)
        means, stds = self.get_mean_std(img)
        norm1 = (
            (chan1 - means[0]) * (self.target_stds[0] / stds[0])
        ) + self.target_means[0]
        norm2 = (
            (chan2 - means[1]) * (self.target_stds[1] / stds[1])
        ) + self.target_means[1]
        norm3 = (
            (chan3 - means[2]) * (self.target_stds[2] / stds[2])
        ) + self.target_means[2]
        return self.merge_back(norm1, norm2, norm3)

    @staticmethod
    def lab_split(img: np.ndarray) -> tuple[float, float, float]:
        """Convert from RGB uint8 to LAB and split into channels.

        Args:
            img (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
                Input image.

        Returns:
            tuple:
                - :py:obj:`float`:
                    L channel in LAB colour space.
                - :py:obj:`float`:
                    A channel in LAB colour space.
                - :py:obj:`float`:
                    B channel in LAB colour space.

        """
        img = img.astype("uint8")  # ensure input image is uint8
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img_float = img.astype(np.float32)
        chan1, chan2, chan3 = cv2.split(img_float)
        chan1 /= 2.55  # should now be in range [0,100]
        chan2 -= 128.0  # should now be in range [-127,127]
        chan3 -= 128.0  # should now be in range [-127,127]
        return chan1, chan2, chan3

    @staticmethod
    def merge_back(chan1: float, chan2: float, chan3: float) -> np.ndarray:
        """Take separate LAB channels and merge back to give RGB uint8.

        Args:
            chan1 (float):
                L channel.
            chan2 (float):
                A channel.
            chan3 (float):
                B channel.

        Returns:
            :class:`numpy.ndarray`:
                Merged image.

        """
        chan1 *= 2.55  # should now be in range [0,255]
        chan2 += 128.0  # should now be in range [0,255]
        chan3 += 128.0  # should now be in range [0,255]
        img = np.clip(cv2.merge((chan1, chan2, chan3)), 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    def get_mean_std(
        self: ReinhardNormalizer,
        img: np.ndarray,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Get mean and standard deviation of each channel.

        Args:
            img (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
                Input image.

        Returns:
            tuple:
                - :py:obj:`float` - Means:
                    Mean values for each RGB channel.
                - :py:obj:`float` - Standard deviations:
                    Standard deviation for each RGB channel.

        """
        img = img.astype("uint8")  # ensure input image is uint8
        chan1, chan2, chan3 = self.lab_split(img)
        m1, sd1 = cv2.meanStdDev(chan1)
        m2, sd2 = cv2.meanStdDev(chan2)
        m3, sd3 = cv2.meanStdDev(chan3)
        means = m1, m2, m3
        stds = sd1, sd2, sd3
        return means, stds

