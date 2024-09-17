import numpy as np
import cv2
import scipy.signal
from csbdeep.utils import _raise
from skimage import transform
from skimage.measure import regionprops
from skimage.morphology import remove_small_holes
from skimage.segmentation import find_boundaries
from collections import OrderedDict


def erode_edges(mask, erosion_width):
    """
    Erode the edges of objects within a mask to prevent them from touching each other.
    This is done by progressively shrinking the boundaries of the objects.

    Args:
        mask (numpy.array): A 2D or 3D array where each unique value represents a different labeled object.
        erosion_width (int): The number of pixels by which to erode the edges of each object. If 0, no erosion is applied.

    Returns:
        numpy.array: A new mask where the edges of each object have been eroded by the specified number of pixels.
    """

    if mask.ndim not in {2, 3}:
        raise ValueError(
            "erode_edges expects arrays of ndim 2 or 3." "Got ndim: {}".format(
                mask.ndim
            )
        )
    if erosion_width:
        new_mask = np.copy(mask)
        for _ in range(erosion_width):
            boundaries = find_boundaries(new_mask, mode="inner")
            new_mask[boundaries > 0] = 0
        return new_mask

    return mask


def resize(data, shape, data_format="channels_last", labeled_image=False):
    """
    Resize the input data to the specified shape using OpenCV or skimage depending on the number of channels.

    Args:
        data (np.array): The input data to be resized. This should be a 3D or 4D numpy array with shape
            `[batch, x, y]`, `[x, y, channels]`, or `[batch, x, y, channels]`, depending on `data_format`.
        shape (tuple): The target shape `(x, y)` to resize the data to. The batch and channel dimensions
            are preserved.
        data_format (str, optional): The format of the data's channel axis. It should be one of
            `"channels_first"` (channels at index 1) or `"channels_last"` (channels at the last index).
            Defaults to `"channels_last"`.
        labeled_image (bool, optional): A flag indicating whether the data contains labeled images (such as
            masks or annotations). If `True`, nearest neighbor interpolation is used. If `False`, linear
            interpolation is applied. Defaults to `False`.

    Returns:
        np.array: The resized data as a numpy array, with the same number of channels and batch size
        as the input data. The dtype of the output matches the original data.
    """

    if len(data.shape) not in {3, 4}:
        raise ValueError(
            "Data must have 3 or 4 dimensions, e.g. "
            "[batch, x, y], [x, y, channel] or "
            "[batch, x, y, channel]. Input data only has {} "
            "dimensions.".format(len(data.shape))
        )

    if len(shape) != 2:
        raise ValueError(
            "Shape for resize can only have length of 2, e.g. (x,y)."
            "Input shape has {} dimensions.".format(len(shape))
        )

    original_dtype = data.dtype

    # cv2 resize is faster but does not support multi-channel data
    # If the data is multi-channel, use skimage.transform.resize
    channel_axis = 0 if data_format == "channels_first" else -1
    batch_axis = -1 if data_format == "channels_first" else 0

    # Use skimage for multichannel data
    if data.shape[channel_axis] > 1:
        # Adjust output shape to account for channel axis
        if data_format == "channels_first":
            shape = tuple([data.shape[channel_axis]] + list(shape))
        else:
            shape = tuple(list(shape) + [data.shape[channel_axis]])

        # linear interpolation (order 1) for image data, nearest neighbor (order 0) for labels
        # anti_aliasing introduces spurious labels, include only for image data
        order = 0 if labeled_image else 1
        anti_aliasing = not labeled_image

        def _resize(d):
            return transform.resize(
                d,
                shape,
                mode="constant",
                preserve_range=True,
                order=order,
                anti_aliasing=anti_aliasing,
            )

    # single channel image, resize with cv2
    else:
        shape = tuple(shape)[::-1]  # cv2 expects swapped axes.

        # linear interpolation for image data, nearest neighbor for labels
        # CV2 doesn't support ints for linear interpolation, set to float for image data
        if labeled_image:
            interpolation = cv2.INTER_NEAREST
        else:
            interpolation = cv2.INTER_LINEAR
            data = data.astype("float32")

        def _resize(d):
            return np.expand_dims(
                cv2.resize(np.squeeze(d), shape, interpolation=interpolation),
                axis=channel_axis,
            )

    # Check for batch dimension to loop over
    if len(data.shape) == 4:
        batch = []
        for i in range(data.shape[batch_axis]):
            d = data[i] if batch_axis == 0 else data[..., i]
            batch.append(_resize(d))
        resized = np.stack(batch, axis=batch_axis)
    else:
        resized = _resize(data)

    return resized.astype(original_dtype)


def tile_image(
    image, model_input_shape=(512, 512), stride_ratio=0.75, pad_mode="constant"
):
    """
    Tile a large 4D image into overlapping tiles, suitable for model input, and return
    the tiles along with tiling details for reconstruction.

    Args:
        image (numpy.array): A 4D numpy array of shape (batch_size, height, width, channels),
            representing the input image(s) to tile.
        model_input_shape (tuple): A tuple (height, width) specifying the dimensions of each
            output tile. Default is (512, 512).
        stride_ratio (float): Fraction of the tile size used to determine the stride (i.e., the
            step size between adjacent tiles). A value of 0.75 means 75% overlap. Default is 0.75.
        pad_mode (str): The padding mode to be used when padding the image for tiling. This is
            passed to ``numpy.pad``. Default is "constant".

    Returns:
        tuple:
            - numpy.array: A 4D array of tiled images, with shape
              (number_of_tiles, tile_height, tile_width, channels).
            - dict: A dictionary containing metadata about the tiling, including:
                - "batches": List of batch indices for each tile.
                - "x_starts", "x_ends": Start and end positions of each tile along the x-axis.
                - "y_starts", "y_ends": Start and end positions of each tile along the y-axis.
                - "overlaps_x", "overlaps_y": Overlap ranges for each tile along the x and y axes.
                - "stride_x", "stride_y": Computed strides (in pixels) used for tiling.
                - "tile_size_x", "tile_size_y": Tile dimensions.
                - "stride_ratio": The stride ratio used.
                - "image_shape": The shape of the padded image.
                - "dtype": The data type of the image.
                - "pad_x", "pad_y": Padding applied along the x and y axes.

    """

    if image.ndim != 4:
        raise ValueError("Expected image of rank 4, got {}".format(image.ndim))

    image_size_x, image_size_y = image.shape[1:3]
    tile_size_x = model_input_shape[0]
    tile_size_y = model_input_shape[1]

    def ceil(x):
        return int(np.ceil(x))

    def round_to_even(x):
        return int(np.ceil(x / 2.0) * 2)

    stride_x = min(round_to_even(stride_ratio * tile_size_x), tile_size_x)
    stride_y = min(round_to_even(stride_ratio * tile_size_y), tile_size_y)

    rep_number_x = max(ceil((image_size_x - tile_size_x) / stride_x + 1), 1)
    rep_number_y = max(ceil((image_size_y - tile_size_y) / stride_y + 1), 1)
    new_batch_size = image.shape[0] * rep_number_x * rep_number_y

    tiles_shape = (new_batch_size, tile_size_x, tile_size_y, image.shape[3])
    tiles = np.zeros(tiles_shape, dtype=image.dtype)

    # Calculate overlap of last tile
    overlap_x = (tile_size_x + stride_x * (rep_number_x - 1)) - image_size_x
    overlap_y = (tile_size_y + stride_y * (rep_number_y - 1)) - image_size_y

    # Calculate padding needed to account for overlap and pad image accordingly
    pad_x = (int(np.ceil(overlap_x / 2)), int(np.floor(overlap_x / 2)))
    pad_y = (int(np.ceil(overlap_y / 2)), int(np.floor(overlap_y / 2)))
    pad_null = (0, 0)
    padding = (pad_null, pad_x, pad_y, pad_null)
    image = np.pad(image, padding, pad_mode)

    counter = 0
    batches = []
    x_starts = []
    x_ends = []
    y_starts = []
    y_ends = []
    overlaps_x = []
    overlaps_y = []

    for b in range(image.shape[0]):
        for i in range(rep_number_x):
            for j in range(rep_number_y):
                x_axis = 1
                y_axis = 2

                # Compute the start and end for each tile
                if i != rep_number_x - 1:  # not the last one
                    x_start, x_end = i * stride_x, i * stride_x + tile_size_x
                else:
                    x_start, x_end = (
                        image.shape[x_axis] - tile_size_x,
                        image.shape[x_axis],
                    )

                if j != rep_number_y - 1:  # not the last one
                    y_start, y_end = j * stride_y, j * stride_y + tile_size_y
                else:
                    y_start, y_end = (
                        image.shape[y_axis] - tile_size_y,
                        image.shape[y_axis],
                    )

                # Compute the overlaps for each tile
                if i == 0:
                    overlap_x = (0, tile_size_x - stride_x)
                elif i == rep_number_x - 2:
                    overlap_x = (
                        tile_size_x - stride_x,
                        tile_size_x - image.shape[x_axis] + x_end,
                    )
                elif i == rep_number_x - 1:
                    overlap_x = ((i - 1) * stride_x + tile_size_x - x_start, 0)
                else:
                    overlap_x = (tile_size_x - stride_x, tile_size_x - stride_x)

                if j == 0:
                    overlap_y = (0, tile_size_y - stride_y)
                elif j == rep_number_y - 2:
                    overlap_y = (
                        tile_size_y - stride_y,
                        tile_size_y - image.shape[y_axis] + y_end,
                    )
                elif j == rep_number_y - 1:
                    overlap_y = ((j - 1) * stride_y + tile_size_y - y_start, 0)
                else:
                    overlap_y = (tile_size_y - stride_y, tile_size_y - stride_y)

                tiles[counter] = image[b, x_start:x_end, y_start:y_end, :]
                batches.append(b)
                x_starts.append(x_start)
                x_ends.append(x_end)
                y_starts.append(y_start)
                y_ends.append(y_end)
                overlaps_x.append(overlap_x)
                overlaps_y.append(overlap_y)
                counter += 1

    tiles_info = {}
    tiles_info["batches"] = batches
    tiles_info["x_starts"] = x_starts
    tiles_info["x_ends"] = x_ends
    tiles_info["y_starts"] = y_starts
    tiles_info["y_ends"] = y_ends
    tiles_info["overlaps_x"] = overlaps_x
    tiles_info["overlaps_y"] = overlaps_y
    tiles_info["stride_x"] = stride_x
    tiles_info["stride_y"] = stride_y
    tiles_info["tile_size_x"] = tile_size_x
    tiles_info["tile_size_y"] = tile_size_y
    tiles_info["stride_ratio"] = stride_ratio
    tiles_info["image_shape"] = image.shape
    tiles_info["dtype"] = image.dtype
    tiles_info["pad_x"] = pad_x
    tiles_info["pad_y"] = pad_y

    return tiles, tiles_info


def spline_window(window_size, overlap_left, overlap_right, power=2):
    """
    Generates a spline window with customizable overlaps on both the left and right sides.
    The window is used to taper the edges of a signal smoothly to zero, ensuring continuity
    between overlapping segments. Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2

    Args:
        window_size (int): Total size of the window.
        overlap_left (int): Number of points in the left overlap region to apply the spline taper.
        overlap_right (int): Number of points in the right overlap region to apply the spline taper.
        power (int, optional): Power of the spline function. Default is 2 (squared spline).

    Returns:
        numpy.ndarray: A 1D array representing the spline window, with the size of `window_size`.
                       The values taper off to zero at the overlap regions and remain one in the center.

    """

    def _spline_window(w_size):
        intersection = int(w_size / 4)
        wind_outer = (abs(2 * (scipy.signal.triang(w_size))) ** power) / 2
        wind_outer[intersection:-intersection] = 0

        wind_inner = 1 - (abs(2 * (scipy.signal.triang(w_size) - 1)) ** power) / 2
        wind_inner[:intersection] = 0
        wind_inner[-intersection:] = 0

        wind = wind_inner + wind_outer
        wind = wind / np.amax(wind)
        return wind

    # Create the window for the left overlap
    if overlap_left > 0:
        window_size_l = 2 * overlap_left
        l_spline = _spline_window(window_size_l)[0:overlap_left]

    # Create the window for the right overlap
    if overlap_right > 0:
        window_size_r = 2 * overlap_right
        r_spline = _spline_window(window_size_r)[overlap_right:]

    # Put the two together
    window = np.ones((window_size,))
    if overlap_left > 0:
        window[0:overlap_left] = l_spline
    if overlap_right > 0:
        window[-overlap_right:] = r_spline

    return window


def window_2D(window_size, overlap_x=(32, 32), overlap_y=(32, 32), power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    window_x = spline_window(window_size[0], overlap_x[0], overlap_x[1], power=power)
    window_y = spline_window(window_size[1], overlap_y[0], overlap_y[1], power=power)

    window_x = np.expand_dims(np.expand_dims(window_x, -1), -1)
    window_y = np.expand_dims(np.expand_dims(window_y, -1), -1)

    window = window_x * window_y.transpose(1, 0, 2)
    return window


def untile_image(tiles, tiles_info, power=2, **kwargs):
    """
    Reconstruct an image from its tiled segments, applying optional windowing for smooth blending.

    Args:
        tiles (numpy.array): A 4D array of tiled image segments with shape
            (batches, tile_height, tile_width, channels).
        tiles_info (dict): Dictionary containing information about how the original image
            was split into tiles. It should include the following keys:
            - 'stride_ratio' (float): Ratio of the tile stride to the tile size.
            - 'image_shape' (tuple): Shape of the original image (before tiling).
            - 'batches' (list): List of batch indices corresponding to each tile.
            - 'x_starts' (list): List of x-axis starting coordinates for each tile.
            - 'x_ends' (list): List of x-axis ending coordinates for each tile.
            - 'y_starts' (list): List of y-axis starting coordinates for each tile.
            - 'y_ends' (list): List of y-axis ending coordinates for each tile.
            - 'overlaps_x' (list): List of overlap values for each tile along the x-axis.
            - 'overlaps_y' (list): List of overlap values for each tile along the y-axis.
            - 'tile_size_x' (int): Width of each tile.
            - 'tile_size_y' (int): Height of each tile.
            - 'pad_x' (tuple): Padding applied to the x-axis (left and right) during tiling.
            - 'pad_y' (tuple): Padding applied to the y-axis (top and bottom) during tiling.
        power (int, optional): The power used in the window function for smooth blending
            of overlapping tiles. Default is 2.

    Returns:
        numpy.array: The reconstructed image of shape (batches, height, width, channels)
        where the height and width correspond to the original image size before tiling.

    Notes:
        - If the tile size or stride ratio is too small, window-based blending is skipped,
          and raw tiles are used to reconstruct the image without interpolation.
        - Padding is removed from the final reconstructed image to match the original image size.
    """

    # Define mininally acceptable tile_size and stride_ratio for spline interpolation
    min_tile_size = 32
    min_stride_ratio = 0.5

    stride_ratio = tiles_info["stride_ratio"]
    image_shape = tiles_info["image_shape"]
    batches = tiles_info["batches"]
    x_starts = tiles_info["x_starts"]
    x_ends = tiles_info["x_ends"]
    y_starts = tiles_info["y_starts"]
    y_ends = tiles_info["y_ends"]
    overlaps_x = tiles_info["overlaps_x"]
    overlaps_y = tiles_info["overlaps_y"]
    tile_size_x = tiles_info["tile_size_x"]
    tile_size_y = tiles_info["tile_size_y"]
    stride_ratio = tiles_info["stride_ratio"]
    x_pad = tiles_info["pad_x"]
    y_pad = tiles_info["pad_y"]

    image_shape = [image_shape[0], image_shape[1], image_shape[2], tiles.shape[-1]]
    window_size = (tile_size_x, tile_size_y)
    image = np.zeros(image_shape, dtype=float)

    window_cache = {}
    for x, y in zip(overlaps_x, overlaps_y):
        if (x, y) not in window_cache:
            w = window_2D(window_size, overlap_x=x, overlap_y=y, power=power)
            window_cache[(x, y)] = w

    for tile, batch, x_start, x_end, y_start, y_end, overlap_x, overlap_y in zip(
        tiles, batches, x_starts, x_ends, y_starts, y_ends, overlaps_x, overlaps_y
    ):
        # Conditions under which to use spline interpolation
        # A tile size or stride ratio that is too small gives inconsistent results,
        # so in these cases we skip interpolation and just return the raw tiles
        if (
            min_tile_size <= tile_size_x < image_shape[1]
            and min_tile_size <= tile_size_y < image_shape[2]
            and stride_ratio >= min_stride_ratio
        ):
            window = window_cache[(overlap_x, overlap_y)]
            image[batch, x_start:x_end, y_start:y_end, :] += tile * window
        else:
            image[batch, x_start:x_end, y_start:y_end, :] = tile

    image = image.astype(tiles.dtype)

    x_start = x_pad[0]
    y_start = y_pad[0]
    x_end = image_shape[1] - x_pad[1]
    y_end = image_shape[2] - y_pad[1]

    image = image[:, x_start:x_end, y_start:y_end, :]

    return image


def tile_image_3D(image, model_input_shape=(10, 256, 256), stride_ratio=0.5):
    """
    Tile large image into many overlapping tiles of size "model_input_shape".

    Args:
        image (numpy.array): The 3D image to tile, must be rank 5.
        model_input_shape (tuple): The input size of the model.
        stride_ratio (float): The stride expressed as a fraction of the tile sizet

    Returns:
        tuple(numpy.array, dict): An tuple consisting of an array of tiled
            images and a dictionary of tiling details (for use in un-tiling).

    Raises:
        ValueError: image is not rank 5.
    """
    if image.ndim != 5:
        raise ValueError("Expected image of 5, got {}".format(image.ndim))

    image_size_z, image_size_x, image_size_y = image.shape[1:4]
    tile_size_z = model_input_shape[0]
    tile_size_x = model_input_shape[1]
    tile_size_y = model_input_shape[2]

    def ceil(x):
        return int(np.ceil(x))

    def round_to_even(x):
        return int(np.ceil(x / 2.0) * 2)

    stride_z = min(round_to_even(stride_ratio * tile_size_z), tile_size_z)
    stride_x = min(round_to_even(stride_ratio * tile_size_x), tile_size_x)
    stride_y = min(round_to_even(stride_ratio * tile_size_y), tile_size_y)

    rep_number_z = max(ceil((image_size_z - tile_size_z) / stride_z + 1), 1)
    rep_number_x = max(ceil((image_size_x - tile_size_x) / stride_x + 1), 1)
    rep_number_y = max(ceil((image_size_y - tile_size_y) / stride_y + 1), 1)
    new_batch_size = image.shape[0] * rep_number_z * rep_number_x * rep_number_y

    # catches error caused by interpolation along z axis with rep number = 1
    # TODO - create a better solution or figure out why it doesn't occur in x and y planes
    if rep_number_z == 1:
        stride_z = tile_size_z

    tiles_shape = (
        new_batch_size,
        tile_size_z,
        tile_size_x,
        tile_size_y,
        image.shape[4],
    )
    tiles = np.zeros(tiles_shape, dtype=image.dtype)

    # Calculate overlap of last tile along each axis
    overlap_z = (tile_size_z + stride_z * (rep_number_z - 1)) - image_size_z
    overlap_x = (tile_size_x + stride_x * (rep_number_x - 1)) - image_size_x
    overlap_y = (tile_size_y + stride_y * (rep_number_y - 1)) - image_size_y

    # Calculate padding needed to account for overlap and pad image accordingly
    pad_z = (int(np.ceil(overlap_z / 2)), int(np.floor(overlap_z / 2)))
    pad_x = (int(np.ceil(overlap_x / 2)), int(np.floor(overlap_x / 2)))
    pad_y = (int(np.ceil(overlap_y / 2)), int(np.floor(overlap_y / 2)))
    pad_null = (0, 0)
    padding = (pad_null, pad_z, pad_x, pad_y, pad_null)
    image = np.pad(image, padding, "constant", constant_values=0)

    counter = 0
    batches = []
    z_starts = []
    z_ends = []
    x_starts = []
    x_ends = []
    y_starts = []
    y_ends = []
    overlaps_z = []
    overlaps_x = []
    overlaps_y = []
    z_axis = 1
    x_axis = 2
    y_axis = 3

    for b in range(image.shape[0]):
        for i in range(rep_number_x):
            for j in range(rep_number_y):
                for k in range(rep_number_z):
                    # Compute the start and end for each tile
                    if i != rep_number_x - 1:  # not the last one
                        x_start, x_end = i * stride_x, i * stride_x + tile_size_x
                    else:
                        x_start, x_end = (
                            image.shape[x_axis] - tile_size_x,
                            image.shape[x_axis],
                        )

                    if j != rep_number_y - 1:  # not the last one
                        y_start, y_end = j * stride_y, j * stride_y + tile_size_y
                    else:
                        y_start, y_end = (
                            image.shape[y_axis] - tile_size_y,
                            image.shape[y_axis],
                        )

                    if k != rep_number_z - 1:  # not the last one
                        z_start, z_end = k * stride_z, k * stride_z + tile_size_z
                    else:
                        z_start, z_end = (
                            image.shape[z_axis] - tile_size_z,
                            image.shape[z_axis],
                        )

                    # Compute the overlaps for each tile
                    if i == 0:
                        overlap_x = (0, tile_size_x - stride_x)
                    elif i == rep_number_x - 2:
                        overlap_x = (
                            tile_size_x - stride_x,
                            tile_size_x - image.shape[x_axis] + x_end,
                        )
                    elif i == rep_number_x - 1:
                        overlap_x = ((i - 1) * stride_x + tile_size_x - x_start, 0)
                    else:
                        overlap_x = (tile_size_x - stride_x, tile_size_x - stride_x)

                    if j == 0:
                        overlap_y = (0, tile_size_y - stride_y)
                    elif j == rep_number_y - 2:
                        overlap_y = (
                            tile_size_y - stride_y,
                            tile_size_y - image.shape[y_axis] + y_end,
                        )
                    elif j == rep_number_y - 1:
                        overlap_y = ((j - 1) * stride_y + tile_size_y - y_start, 0)
                    else:
                        overlap_y = (tile_size_y - stride_y, tile_size_y - stride_y)

                    if k == 0:
                        overlap_z = (0, tile_size_z - stride_z)
                    elif k == rep_number_z - 2:
                        overlap_z = (
                            tile_size_z - stride_z,
                            tile_size_z - image.shape[z_axis] + z_end,
                        )
                    elif k == rep_number_z - 1:
                        overlap_z = ((k - 1) * stride_z + tile_size_z - z_start, 0)
                    else:
                        overlap_z = (tile_size_z - stride_z, tile_size_z - stride_z)

                    tiles[counter] = image[
                        b, z_start:z_end, x_start:x_end, y_start:y_end, :
                    ]
                    batches.append(b)
                    x_starts.append(x_start)
                    x_ends.append(x_end)
                    y_starts.append(y_start)
                    y_ends.append(y_end)
                    z_starts.append(z_start)
                    z_ends.append(z_end)
                    overlaps_x.append(overlap_x)
                    overlaps_y.append(overlap_y)
                    overlaps_z.append(overlap_z)
                    counter += 1

    tiles_info = {}
    tiles_info["batches"] = batches
    tiles_info["x_starts"] = x_starts
    tiles_info["x_ends"] = x_ends
    tiles_info["y_starts"] = y_starts
    tiles_info["y_ends"] = y_ends
    tiles_info["z_starts"] = z_starts
    tiles_info["z_ends"] = z_ends
    tiles_info["overlaps_x"] = overlaps_x
    tiles_info["overlaps_y"] = overlaps_y
    tiles_info["overlaps_z"] = overlaps_z
    tiles_info["stride_x"] = stride_x
    tiles_info["stride_y"] = stride_y
    tiles_info["stride_z"] = stride_z
    tiles_info["tile_size_x"] = tile_size_x
    tiles_info["tile_size_y"] = tile_size_y
    tiles_info["tile_size_z"] = tile_size_z
    tiles_info["stride_ratio"] = stride_ratio
    tiles_info["image_shape"] = image.shape
    tiles_info["dtype"] = image.dtype
    tiles_info["pad_x"] = pad_x
    tiles_info["pad_y"] = pad_y
    tiles_info["pad_z"] = pad_z

    return tiles, tiles_info


def window_3D(
    window_size, overlap_z=(5, 5), overlap_x=(32, 32), overlap_y=(32, 32), power=3
):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    window_z = spline_window(window_size[0], overlap_z[0], overlap_z[1], power=power)
    window_x = spline_window(window_size[1], overlap_x[0], overlap_x[1], power=power)
    window_y = spline_window(window_size[2], overlap_y[0], overlap_y[1], power=power)

    window_z = np.expand_dims(np.expand_dims(np.expand_dims(window_z, -1), -1), -1)
    window_x = np.expand_dims(np.expand_dims(np.expand_dims(window_x, -1), -1), -1)
    window_y = np.expand_dims(np.expand_dims(np.expand_dims(window_y, -1), -1), -1)

    window = window_z * window_x.transpose(1, 0, 2, 3) * window_y.transpose(1, 2, 0, 3)

    return window


def untile_image_3D(tiles, tiles_info, power=3, force=False, **kwargs):
    """Untile a set of tiled images back to the original model shape.

    Args:
        tiles (numpy.array): The tiled images image to untile.
        tiles_info (dict): Details of how the image was tiled (from tile_image).
        power (int): The power of the window function
        force (bool): If set to True, forces use spline interpolation regardless of
                      tile size or stride_ratio.

    Returns:
        numpy.array: The untiled image.
    """
    # Define mininally acceptable tile_size and stride_ratios for spline interpolation
    min_tile_size = 32
    min_stride_ratio = 0.5

    if force:
        min_tile_size = 0
        min_stride_ratio = 0

    stride_ratio = tiles_info["stride_ratio"]
    image_shape = tiles_info["image_shape"]
    batches = tiles_info["batches"]

    x_starts = tiles_info["x_starts"]
    x_ends = tiles_info["x_ends"]
    y_starts = tiles_info["y_starts"]
    y_ends = tiles_info["y_ends"]
    z_starts = tiles_info["z_starts"]
    z_ends = tiles_info["z_ends"]

    overlaps_x = tiles_info["overlaps_x"]
    overlaps_y = tiles_info["overlaps_y"]
    overlaps_z = tiles_info["overlaps_z"]

    tile_size_x = tiles_info["tile_size_x"]
    tile_size_y = tiles_info["tile_size_y"]
    tile_size_z = tiles_info["tile_size_z"]
    pad_x = tiles_info["pad_x"]
    pad_y = tiles_info["pad_y"]
    pad_z = tiles_info["pad_z"]

    image_shape = tuple(list(image_shape[:4]) + [tiles.shape[-1]])
    window_size = (tile_size_z, tile_size_x, tile_size_y)
    image = np.zeros(image_shape, dtype=float)

    tile_data_zip = zip(
        tiles,
        batches,
        x_starts,
        x_ends,
        y_starts,
        y_ends,
        z_starts,
        z_ends,
        overlaps_x,
        overlaps_y,
        overlaps_z,
    )

    for (
        tile,
        batch,
        x_start,
        x_end,
        y_start,
        y_end,
        z_start,
        z_end,
        overlap_x,
        overlap_y,
        overlap_z,
    ) in tile_data_zip:
        # Conditions under which to use spline interpolation
        # A tile size or stride ratio that is too small gives inconsistent results,
        # so in these cases we skip interpolation and just return the raw tiles
        if (
            min_tile_size <= tile_size_x < image_shape[2]
            and min_tile_size <= tile_size_y < image_shape[3]
            and min_stride_ratio <= stride_ratio
        ):
            window = window_3D(
                window_size,
                overlap_z=overlap_z,
                overlap_x=overlap_x,
                overlap_y=overlap_y,
                power=power,
            )
            image[batch, z_start:z_end, x_start:x_end, y_start:y_end, :] += (
                tile * window
            )
        else:
            image[batch, z_start:z_end, x_start:x_end, y_start:y_end, :] = tile

    image = image.astype(tiles.dtype)

    x_start = pad_x[0]
    y_start = pad_y[0]
    z_start = pad_z[0]
    x_end = image_shape[2] - pad_x[1]
    y_end = image_shape[3] - pad_y[1]
    z_end = image_shape[1] - pad_z[1]

    image = image[:, z_start:z_end, x_start:x_end, y_start:y_end, :]

    return image


def fill_holes(label_img, size=10, connectivity=1):
    """Fills holes located completely within a given label with pixels of the same value

    Args:
        label_img (numpy.array): a 2D labeled image
        size (int): maximum size for a hole to be filled in
        connectivity (int): the connectivity used to define the hole

    Returns:
        numpy.array: a labeled image with no holes smaller than ``size``
            contained within any label.
    """
    output_image = np.copy(label_img)

    props = regionprops(np.squeeze(label_img.astype("int")), cache=False)
    for prop in props:
        if prop.euler_number < 1:
            patch = output_image[prop.slice]

            filled = remove_small_holes(
                ar=(patch == prop.label), area_threshold=size, connectivity=connectivity
            )

            output_image[prop.slice] = np.where(filled, prop.label, patch)

    return output_image


def get_sobel_kernel(size, diag=True):
    """
    Generate Sobel kernels for edge detection.

    Parameters:
    - size (int): Size of the Sobel kernel, must be an odd number.
    - diag (bool): Whether to include diagonal kernels.

    Returns:
    - tuple: Sobel kernels for horizontal, vertical, and optionally diagonal gradients.
    """

    assert size % 2 == 1, "Must be odd, get size=%d" % size
    if diag:
        h_range = np.arange(-size // 2 + 1, size // 2 + 1, dtype=np.float32)
        v_range = np.arange(-size // 2 + 1, size // 2 + 1, dtype=np.float32)
        h, v = np.meshgrid(h_range, v_range)
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        kernel_tl = kernel_h + kernel_v
        kernel_bl = np.rot90(kernel_tl)
        return kernel_h, kernel_v, kernel_tl, kernel_bl
    else:
        h_range = np.arange(-size // 2 + 1, size // 2 + 1, dtype=np.float32)
        v_range = np.arange(-size // 2 + 1, size // 2 + 1, dtype=np.float32)
        h, v = np.meshgrid(h_range, v_range)
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        return kernel_h, kernel_v


def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def dir_distance_map(gt, magnification):
    """
    Compute directional distance maps for a given ground truth instance map. 

    This function takes an instance segmentation map (`gt`) and computes four directional distance maps:
    horizontal, vertical, top-left diagonal, and bottom-left diagonal. These distance maps represent the 
    normalized distance values from the borders of each instance in the segmentation map. Differently from 
    the full range function, this function limits the computation to the 'tips' of cells.

    Parameters:
    -----------
    gt : numpy.ndarray
        Ground truth instance segmentation map. Each unique non-zero integer represents a different instance, 
        and 0 represents the background.
        
    magnification : int
        The magnification factor of the images. Affects the size of borders used in the distance map calculations.

    Returns:
    --------
    h_map : numpy.ndarray
        A 2D map of the same size as `gt`, containing horizontal distance gradients for each instance.
    
    v_map : numpy.ndarray
        A 2D map of the same size as `gt`, containing vertical distance gradients for each instance.
    
    tl_map : numpy.ndarray
        A 2D map of the same size as `gt`, containing top-left diagonal distance gradients for each instance.
    
    bl_map : numpy.ndarray
        A 2D map of the same size as `gt`, containing bottom-left diagonal distance gradients for each instance.
    """

    orig_ann = gt  # instance ID map

    h_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    v_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    tl_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    bl_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)

    inst_list = list(np.unique(orig_ann))
    inst_list.remove(0)  # 0 is background
    # print(len(inst_list))

    for inst_id in inst_list:
        # inst_id = inst_list[25]

        inst_map = np.array(orig_ann == inst_id, np.uint8)
        inst_box = bounding_box(inst_map)

        inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

        bordersize = 13 if magnification < 40 else 21

        if 1 in inst_map.shape:
            continue
        else:
            start_ = -1
            end_ = 1
            coords = np.indices((inst_map.shape[0], inst_map.shape[1]))

            # hor
            if inst_map.shape[1] > bordersize * 2:
                modelarray = np.ones([inst_map.shape[1]]) * bordersize
                modelarray[0:bordersize] = np.arange(bordersize)
                modelarray[-bordersize:] = np.arange(bordersize + 1, bordersize * 2 + 1)
                distimagehor = np.tile(modelarray, (inst_map.shape[0], 1)) / (
                    bordersize * 2
                )

            else:
                distimagehor = (coords[1]) / (inst_map.shape[1] - 1)

            gradient_imagehor = start_ * (1 - distimagehor) + end_ * distimagehor

            # ver
            if inst_map.shape[0] > bordersize * 2:
                modelarray = np.ones([inst_map.shape[0], 1]) * bordersize
                modelarray[0:bordersize, 0] = np.arange(bordersize)
                modelarray[-bordersize:, 0] = np.arange(
                    bordersize + 1, bordersize * 2 + 1
                )
                distimagever = np.tile(modelarray, inst_map.shape[1]) / (bordersize * 2)

            else:
                distimagever = (coords[0]) / (inst_map.shape[0] - 1)

            gradient_imagever = start_ * (1 - distimagever) + end_ * distimagever

            h_map_box = h_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            h_map_box[inst_map > 0] = gradient_imagehor[inst_map > 0]

            v_map_box = v_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            v_map_box[inst_map > 0] = gradient_imagever[inst_map > 0]

            inst_box_square = inst_box
            if inst_map.shape[0] > inst_map.shape[1]:
                sidesize = inst_map.shape[0]
                diff = inst_map.shape[0] - inst_map.shape[1]
                if (diff % 2) == 0:
                    # print("case1")
                    inst_map = np.pad(
                        inst_map, ((0, 0), (int(diff / 2), int(diff / 2)))
                    )
                    inst_box_square[2] -= int(diff / 2)
                    inst_box_square[3] += int(diff / 2)
                else:
                    # print("case2")
                    inst_map = np.pad(
                        inst_map,
                        ((0, 0), (int(np.floor(diff / 2)), int(np.ceil(diff / 2)))),
                    )
                    inst_box_square[2] -= int(np.floor(diff / 2))
                    inst_box_square[3] += int(np.ceil(diff / 2))
            elif inst_map.shape[0] < inst_map.shape[1]:
                sidesize = inst_map.shape[1]
                diff = inst_map.shape[1] - inst_map.shape[0]
                if (diff % 2) == 0:
                    # print("case3")
                    inst_map = np.pad(
                        inst_map, ((int(diff / 2), int(diff / 2)), (0, 0))
                    )
                    inst_box_square[0] -= int(diff / 2)
                    inst_box_square[1] += int(diff / 2)
                else:
                    # print("case4")
                    inst_map = np.pad(
                        inst_map,
                        ((int(np.floor(diff / 2)), int(np.ceil(diff / 2))), (0, 0)),
                    )
                    inst_box_square[0] -= int(np.floor(diff / 2))
                    inst_box_square[1] += int(np.ceil(diff / 2))
            else:
                # print("case5")
                sidesize = inst_map.shape[1]

            # blocky distance from start vertex

            coordstl = np.indices((sidesize, sidesize))
            y = coordstl[0] + coordstl[1]

            bordersize = bordersize * 1.4

            if (np.max(y[inst_map > 0]) - np.min(y[inst_map > 0])) > bordersize * 2:
                distimagetl = y.copy()
                distimagetl[distimagetl <= (np.min(y[inst_map > 0]) + bordersize)] = (
                    distimagetl[distimagetl <= np.min(y[inst_map > 0]) + bordersize]
                    - np.min(y[inst_map > 0])
                )
                distimagetl[distimagetl >= (np.max(y[inst_map > 0]) - bordersize)] = (
                    distimagetl[distimagetl >= (np.max(y[inst_map > 0]) - bordersize)]
                    - ((np.max(y[inst_map > 0])) - 2 * bordersize)
                )
                distimagetl[
                    (y > (np.min(y[inst_map > 0]) + bordersize))
                    & (y < (np.max(y[inst_map > 0]) - bordersize))
                ] = bordersize

            else:
                distimagetl = y / (2 * (sidesize - 1))

            # print(np.min(distimagetl[inst_map > 0]))

            if (
                np.max(distimagetl[inst_map > 0]) - np.min(distimagetl[inst_map > 0])
            ) == 0:
                distimagetl = distimagetl - np.min(distimagetl[inst_map > 0])
            else:
                distimagetl = (distimagetl - np.min(distimagetl[inst_map > 0])) / (
                    np.max(distimagetl[inst_map > 0])
                    - np.min(distimagetl[inst_map > 0])
                )

            gradient_imagetl = start_ * (1 - distimagetl) + end_ * distimagetl

            # bottom left
            coordsbl = np.indices((sidesize, sidesize))
            coordsbl[0] = sidesize - 1 - coordsbl[0]

            y = coordsbl[0] + coordsbl[1]

            if (np.max(y[inst_map > 0]) - np.min(y[inst_map > 0])) > bordersize * 2:
                distimagebl = y.copy()
                distimagebl[distimagebl <= (np.min(y[inst_map > 0]) + bordersize)] = (
                    distimagebl[distimagebl <= np.min(y[inst_map > 0]) + bordersize]
                    - np.min(y[inst_map > 0])
                )
                distimagebl[distimagebl >= (np.max(y[inst_map > 0]) - bordersize)] = (
                    distimagebl[distimagebl >= (np.max(y[inst_map > 0]) - bordersize)]
                    - ((np.max(y[inst_map > 0])) - 2 * bordersize)
                )
                distimagebl[
                    (y > (np.min(y[inst_map > 0]) + bordersize))
                    & (y < (np.max(y[inst_map > 0]) - bordersize))
                ] = bordersize

            else:
                distimagebl = y / (2 * (sidesize - 1))

            if (
                np.max(distimagebl[inst_map > 0]) - np.min(distimagebl[inst_map > 0])
            ) == 0:
                distimagebl = distimagebl - np.min(distimagebl[inst_map > 0])
            else:
                distimagebl = (distimagebl - np.min(distimagebl[inst_map > 0])) / (
                    np.max(distimagebl[inst_map > 0])
                    - np.min(distimagebl[inst_map > 0])
                )

            # distimagebl = (distimagebl - np.min(distimagebl[inst_map > 0]))/(np.max(distimagebl[inst_map > 0]) - np.min(distimagebl[inst_map > 0]))
            gradient_imagebl = start_ * (1 - distimagebl) + end_ * distimagebl

            if inst_box_square[0] < 0:
                inst_map = inst_map[-inst_box_square[0] :, :]
                gradient_imagetl = gradient_imagetl[-inst_box_square[0] :, :]
                gradient_imagebl = gradient_imagebl[-inst_box_square[0] :, :]
                inst_box_square[0] = 0
            if inst_box_square[2] < 0:
                inst_map = inst_map[:, -inst_box_square[2] :]
                gradient_imagetl = gradient_imagetl[:, -inst_box_square[2] :]
                gradient_imagebl = gradient_imagebl[:, -inst_box_square[2] :]
                inst_box_square[2] = 0
            if inst_box_square[1] > orig_ann.shape[0]:
                inst_map = inst_map[: -(inst_box_square[1] - orig_ann.shape[0]), :]
                gradient_imagetl = gradient_imagetl[
                    : -(inst_box_square[1] - orig_ann.shape[0]), :
                ]
                gradient_imagebl = gradient_imagebl[
                    : -(inst_box_square[1] - orig_ann.shape[0]), :
                ]
                inst_box_square[1] = orig_ann.shape[0]
            if inst_box_square[3] > orig_ann.shape[1]:
                inst_map = inst_map[:, : -(inst_box_square[3] - orig_ann.shape[1])]
                gradient_imagetl = gradient_imagetl[
                    :, : -(inst_box_square[3] - orig_ann.shape[1])
                ]
                gradient_imagebl = gradient_imagebl[
                    :, : -(inst_box_square[3] - orig_ann.shape[1])
                ]
                inst_box_square[3] = orig_ann.shape[1]

            tl_map_box = tl_map[
                inst_box_square[0] : inst_box_square[1],
                inst_box_square[2] : inst_box_square[3],
            ]
            tl_map_box[inst_map > 0] = gradient_imagetl[inst_map > 0]

            bl_map_box = bl_map[
                inst_box_square[0] : inst_box_square[1],
                inst_box_square[2] : inst_box_square[3],
            ]
            bl_map_box[inst_map > 0] = gradient_imagebl[inst_map > 0]
    return h_map, v_map, tl_map, bl_map


def dir_distance_map_fullrange(gt):
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        # due to python indexing, need to add 1 to max
        # else accessing will be 1px in the box, not out
        rmax += 1
        cmax += 1
        return [rmin, rmax, cmin, cmax]

    orig_ann = gt  # instance ID map

    h_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    v_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    tl_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    bl_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)

    inst_list = list(np.unique(orig_ann))
    inst_list.remove(0)  # 0 is background
    # print(len(inst_list))

    for inst_id in inst_list:
        inst_map = np.array(orig_ann == inst_id, np.uint8)
        inst_box = bounding_box(inst_map)

        inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

        if 1 in inst_map.shape:
            continue
        else:
            start_ = -1
            end_ = 1
            coords = np.indices((inst_map.shape[0], inst_map.shape[1]))

            # hor
            distimagehor = (coords[1]) / (inst_map.shape[1] - 1)
            gradient_imagehor = start_ * (1 - distimagehor) + end_ * distimagehor

            # ver
            distimagever = (coords[0]) / (inst_map.shape[0] - 1)
            gradient_imagever = start_ * (1 - distimagever) + end_ * distimagever

            h_map_box = h_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            h_map_box[inst_map > 0] = gradient_imagehor[inst_map > 0]

            v_map_box = v_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            v_map_box[inst_map > 0] = gradient_imagever[inst_map > 0]

            inst_box_square = inst_box
            if inst_map.shape[0] > inst_map.shape[1]:
                sidesize = inst_map.shape[0]
                diff = inst_map.shape[0] - inst_map.shape[1]
                if (diff % 2) == 0:
                    # print("case1")
                    inst_map = np.pad(
                        inst_map, ((0, 0), (int(diff / 2), int(diff / 2)))
                    )
                    inst_box_square[2] -= int(diff / 2)
                    inst_box_square[3] += int(diff / 2)
                else:
                    # print("case2")
                    inst_map = np.pad(
                        inst_map,
                        ((0, 0), (int(np.floor(diff / 2)), int(np.ceil(diff / 2)))),
                    )
                    inst_box_square[2] -= int(np.floor(diff / 2))
                    inst_box_square[3] += int(np.ceil(diff / 2))
            elif inst_map.shape[0] < inst_map.shape[1]:
                sidesize = inst_map.shape[1]
                diff = inst_map.shape[1] - inst_map.shape[0]
                if (diff % 2) == 0:
                    # print("case3")
                    inst_map = np.pad(
                        inst_map, ((int(diff / 2), int(diff / 2)), (0, 0))
                    )
                    inst_box_square[0] -= int(diff / 2)
                    inst_box_square[1] += int(diff / 2)
                else:
                    # print("case4")
                    inst_map = np.pad(
                        inst_map,
                        ((int(np.floor(diff / 2)), int(np.ceil(diff / 2))), (0, 0)),
                    )
                    inst_box_square[0] -= int(np.floor(diff / 2))
                    inst_box_square[1] += int(np.ceil(diff / 2))
            else:
                # print("case5")
                sidesize = inst_map.shape[1]

            # blocky distance from start vertex
            coordstl = np.indices((sidesize, sidesize))
            distimagetl = (coordstl[0] + coordstl[1]) / (2 * (sidesize - 1))
            # print(np.min(distimagetl[inst_map > 0]))

            if (
                np.max(distimagetl[inst_map > 0]) - np.min(distimagetl[inst_map > 0])
            ) == 0:
                distimagetl = distimagetl - np.min(distimagetl[inst_map > 0])
            else:
                distimagetl = (distimagetl - np.min(distimagetl[inst_map > 0])) / (
                    np.max(distimagetl[inst_map > 0])
                    - np.min(distimagetl[inst_map > 0])
                )

            # try:
            #     distimagetl = (distimagetl - np.min(distimagetl[inst_map > 0]))/(np.max(distimagetl[inst_map > 0]) - np.min(distimagetl[inst_map > 0]))
            # except:
            #     print((distimagetl - np.min(distimagetl[inst_map > 0])))
            #     print((np.max(distimagetl[inst_map > 0]) - np.min(distimagetl[inst_map > 0])))

            # distimagetl[inst_map == 0] = 0
            gradient_imagetl = start_ * (1 - distimagetl) + end_ * distimagetl

            # bottom left
            coordsbl = np.indices((sidesize, sidesize))
            coordsbl[0] = sidesize - 1 - coordsbl[0]
            distimagebl = (coordsbl[0] + coordsbl[1]) / (2 * (sidesize - 1))

            if (
                np.max(distimagebl[inst_map > 0]) - np.min(distimagebl[inst_map > 0])
            ) == 0:
                distimagebl = distimagebl - np.min(distimagebl[inst_map > 0])
            else:
                distimagebl = (distimagebl - np.min(distimagebl[inst_map > 0])) / (
                    np.max(distimagebl[inst_map > 0])
                    - np.min(distimagebl[inst_map > 0])
                )

            # distimagebl = (distimagebl - np.min(distimagebl[inst_map > 0]))/(np.max(distimagebl[inst_map > 0]) - np.min(distimagebl[inst_map > 0]))
            gradient_imagebl = start_ * (1 - distimagebl) + end_ * distimagebl

            if inst_box_square[0] < 0:
                inst_map = inst_map[-inst_box_square[0] :, :]
                gradient_imagetl = gradient_imagetl[-inst_box_square[0] :, :]
                gradient_imagebl = gradient_imagebl[-inst_box_square[0] :, :]
                inst_box_square[0] = 0
            if inst_box_square[2] < 0:
                inst_map = inst_map[:, -inst_box_square[2] :]
                gradient_imagetl = gradient_imagetl[:, -inst_box_square[2] :]
                gradient_imagebl = gradient_imagebl[:, -inst_box_square[2] :]
                inst_box_square[2] = 0
            if inst_box_square[1] > orig_ann.shape[0]:
                inst_map = inst_map[: -(inst_box_square[1] - orig_ann.shape[0]), :]
                gradient_imagetl = gradient_imagetl[
                    : -(inst_box_square[1] - orig_ann.shape[0]), :
                ]
                gradient_imagebl = gradient_imagebl[
                    : -(inst_box_square[1] - orig_ann.shape[0]), :
                ]
                inst_box_square[1] = orig_ann.shape[0]
            if inst_box_square[3] > orig_ann.shape[1]:
                inst_map = inst_map[:, : -(inst_box_square[3] - orig_ann.shape[1])]
                gradient_imagetl = gradient_imagetl[
                    :, : -(inst_box_square[3] - orig_ann.shape[1])
                ]
                gradient_imagebl = gradient_imagebl[
                    :, : -(inst_box_square[3] - orig_ann.shape[1])
                ]
                inst_box_square[3] = orig_ann.shape[1]

            tl_map_box = tl_map[
                inst_box_square[0] : inst_box_square[1],
                inst_box_square[2] : inst_box_square[3],
            ]
            tl_map_box[inst_map > 0] = gradient_imagetl[inst_map > 0]

            bl_map_box = bl_map[
                inst_box_square[0] : inst_box_square[1],
                inst_box_square[2] : inst_box_square[3],
            ]
            bl_map_box[inst_map > 0] = gradient_imagebl[inst_map > 0]

    return h_map, v_map, tl_map, bl_map


def distance_map(h_dir, v_dir, tl_dir, bl_dir, gt):
    mh, mv, mtl, mbl = get_sobel_kernel(7, diag=True)
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

    sobeltl = cv2.filter2D(tl_dir, cv2.CV_32F, mtl, anchor=(-1, -1))
    sobelbl = cv2.filter2D(bl_dir, cv2.CV_32F, mbl, anchor=(-1, -1))
    sobeltl = 1 - (
        cv2.normalize(
            sobeltl, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )
    sobelbl = 1 - (
        cv2.normalize(
            sobelbl, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )
    contour_map = np.amax(np.stack((sobelh, sobelv, sobeltl, sobelbl), -1), -1)

    foreground_pred_th = gt > 0

    contour_map = contour_map - (1 - foreground_pred_th)
    contour_map[contour_map < 0] = 0

    contour_map_th = contour_map.copy()
    contour_map_th[contour_map >= 0.5] = 1
    contour_map_th[contour_map < 0.5] = 0

    dist = (1.0 - contour_map_th) * foreground_pred_th

    dist = cv2.GaussianBlur(dist, (3, 3), 0)
    return dist

    # erosion_rad = [4]
    # min_resid_size = 3

    # erosion_rad = erosion_rad if magnification == '40' else [1] # was 1 for cerebellum (slightly >20x)
    # min_resid_size = min_resid_size if magnification == '40' else 2
    # small_objects_threshold = 80 if magnification == '40' else 20
    # struct_rad = 5 if magnification == '40' else 3
    # disk_rad = 2 if magnification == '40' else 1
    # to be used later for predictions
    # marker = foreground_pred_th - contour_map_th
    # marker[marker < 0] = 0
    # marker = ndimage.binary_fill_holes(marker).astype('uint8')
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(struct_rad, struct_rad))
    # marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    # marker = label(marker)
    # marker = morph.remove_small_objects(marker, min_size=int(magnification/4))
    # cell_instances = watershed(dist, marker, mask=foreground_pred_th, watershed_line=False)
    # proc_y_pred = morph.remove_small_objects(cell_instances, min_size=small_objects_threshold, connectivity = 1)


def _is_power_of_2(i):
    assert i > 0
    e = np.log2(i)
    return e == int(e)


def _normalize_grid(grid, n):
    try:
        grid = tuple(grid)
        (
            len(grid) == n
            and all(map(np.isscalar, grid))
            and all(map(_is_power_of_2, grid))
        ) or _raise(TypeError())
        return tuple(int(g) for g in grid)
    except (TypeError, AssertionError):
        raise ValueError(
            "grid = {grid} must be a list/tuple of length {n} with values that are power of 2".format(
                grid=grid, n=n
            )
        )


def render_contour(contour, val=1, dtype="int32", round=False, reference=None):
    if reference is None:
        reference = contour
    xmin, ymin = np.floor(np.min(reference, axis=0)).astype("int")
    xmax, ymax = np.ceil(np.max(reference, axis=0)).astype("int")
    a = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=dtype)
    if round:
        contour = contour.round()
    a = cv2.drawContours(
        a,
        [np.array(contour, dtype=np.int32).reshape((-1, 1, 2))],
        0,
        val,
        -1,
        offset=(-xmin, -ymin),
    )
    return a, (xmin, xmax), (ymin, ymax)


def clip_contour_(contour, size):
    np.clip(contour[..., 0], 0, size[1], out=contour[..., 0])
    np.clip(contour[..., 1], 0, size[0], out=contour[..., 1])


def contours2labels(
    contours,
    size,
    rounded=True,
    clip=True,
    initial_depth=1,
    gap=3,
    dtype="int32",
    ioa_thresh=None,
    sort_by=None,
    sort_descending=True,
    return_indices=False,
):
    """Contours to labels.

    Convert contours to label image.

    Notes:
        - ~137 ms for contours.shape=(1284, 128, 2), size=(1000, 1000).
        - Label images come with channels, as contours may assign pixels to multiple objects.
          Since such multi-assignments cannot be easily encoded in a channel-free label image, channels are used.
          To remove channels refer to `resolve_label_channels`.

    Args:
        contours: Contours of a single image. Array[num_contours, num_points, 2] or List[Array[num_points, 2]].
        size: Label image size. (height, width).
        rounded: Whether to round contour coordinates.
        clip: Whether to clip contour coordinates to given `size`.
        initial_depth: Initial number of channels. More channels are used if necessary.
        gap: Gap between instances.
        dtype: Data type of label image.
        ioa_thresh: Intersection over area threshold. Skip contours that have an intersection over own area
            (i.e. area of contour that already contains a label vs. area of contour) greater `ioa_thresh`,
            compared to the union of all contours painted before. Note that the order of `contours` is
            relevant, as contours are processed iteratively. IoA of 0 means no labels present so far, IoA of 1. means
            the entire contour area is already covered by other contours.
        sort_by: Optional Array used to sort contours. Note, that if this option is used, labels and contour indices no
            longer correspond.
        sort_descending: Whether to sort by descending.
        return_indices: Whether to return indices.

    Returns:
        Array[height, width, channels]. Since contours may assign pixels to multiple objects, the label image comes
        with channels. To remove channels refer to `resolve_label_channels`.
    """
    contours_ = contours
    if sort_by is not None:
        indices = np.argsort(sort_by)
        if sort_descending:
            indices = reversed(indices)
        contours_ = (contours[i] for i in indices)
    labels = np.zeros(tuple(size) + (initial_depth,), dtype=dtype)
    lbl = 1
    keep = []
    for idx, contour in enumerate(contours_):
        if rounded:
            contour = np.round(contour)
        if clip:
            clip_contour_(contour, np.array(size) - 1)
        a, (xmin, xmax), (ymin, ymax) = render_contour(contour, val=lbl, dtype=dtype)
        if ioa_thresh is not None:
            m = a > 0
            crp = (labels[ymin : ymin + a.shape[0], xmin : xmin + a.shape[1]] > 0).any(
                -1
            )
            ioa = crp[m].sum() / m.sum()
            if ioa > ioa_thresh:
                continue
            else:
                keep.append(idx)
        lbl += 1
        s = (
            labels[
                np.maximum(0, ymin - gap) : gap + ymin + a.shape[0],
                np.maximum(0, xmin - gap) : gap + xmin + a.shape[1],
            ]
            > 0
        ).sum((0, 1))
        i = next(
            i
            for i in range(labels.shape[2] + 1)
            if ~(i < labels.shape[2] and np.any(s[i]))
        )
        if i >= labels.shape[2]:
            labels = np.concatenate(
                (labels, np.zeros(size, dtype=dtype)[..., None]), axis=-1
            )
        labels[ymin : ymin + a.shape[0], xmin : xmin + a.shape[1], i] += a
    if return_indices:
        return labels, keep
    return labels


def resolve_label_channels(labels, method="dilation", max_iter=999, kernel=(3, 3)):
    """Resolve label channels.

    Remove channels from a label image.
    Pixels that are assigned to exactly one foreground label remain as is.
    Pixels that are assigned to multiple foreground labels present a conflict, as they cannot be described by a
    channel-less label image. Such conflicts are resolved by `method`.

    Args:
        labels: Label image. Array[h, w, c].
        method: Method to resolve overlapping regions.
        max_iter: Max iteration.
        kernel: Kernel.

    Returns:
        Labels with channels removed. Array[h, w].
    """
    if isinstance(kernel, (tuple, list)):
        kernel = cv2.getStructuringElement(1, kernel)
    mask_sm = np.sum(labels > 0, axis=-1)
    mask = mask_sm > 1  # all overlaps
    if mask.any():
        if method == "dilation":
            mask_ = mask_sm == 1  # all cores
            lbl = np.zeros(labels.shape[:2], dtype="float64")
            lbl[mask_] = labels.max(-1)[mask_]
            for _ in range(max_iter):
                lbl_ = np.copy(lbl)
                m = mask & (lbl <= 0)
                if not np.any(m):
                    break
                lbl[m] = cv2.dilate(lbl, kernel=kernel)[m]
                if np.allclose(lbl_, lbl):
                    break
        else:
            raise ValueError(f"Invalid method: {method}")
    else:
        lbl = labels.max(-1)
    return lbl.astype(labels.dtype)


def labels2contours(
    labels,
    mode=cv2.RETR_EXTERNAL,
    method=cv2.CHAIN_APPROX_NONE,
    flag_fragmented_inplace=False,
    raise_fragmented=True,
    constant=-1,
) -> dict:
    """Labels to contours.

    Notes:
        - If ``flag_fragmented_inplace is True``, ``labels`` may be modified inplace.

    Args:
        labels:
        mode:
        method: Contour method. CHAIN_APPROX_NONE must be used if contours are used for CPN.
        flag_fragmented_inplace: Whether to flag fragmented labels. Flagging sets labels that consist of more than one
            connected component to ``constant``.
        constant: Flagging constant.
        raise_fragmented: Whether to raise ValueError when encountering fragmented labels.

    Returns:
        dict
    """
    crops = []
    contours = OrderedDict()
    contour_list = []
    for channel in np.split(labels, labels.shape[2], 2):
        crops += [(p.label, p.image) + p.bbox[:2] for p in regionprops(channel)]
    for label, crop, oy, ox in crops:
        crop.dtype = np.uint8
        r = cv2.findContours(crop, mode=mode, method=method, offset=(ox, oy))
        if len(r) == 3:  # be compatible with both existing versions of findContours
            _, c, _ = r
        elif len(r) == 2:
            c, _ = r
        else:
            raise NotImplementedError("try different cv2 version")
        try:
            (c,) = c
        except ValueError as ve:
            if flag_fragmented_inplace:
                labels[labels == label] = constant
            elif raise_fragmented:
                raise ValueError("Object labeled with multiple connected components.")
            continue
        if len(c) == 1:
            c = np.concatenate(
                (c, c), axis=0
            )  # min len for other functions to work properly
        contours[label] = c
        contour_list.append(c)
    if labels.shape[2] > 1:
        return OrderedDict(sorted(contours.items()))
    return contours, contour_list
