from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
# from tensorflow.keras.losses import categorical_crossentropy

_epsilon = tf.convert_to_tensor(0.1 * K.epsilon(), K.floatx())

def schedule_steps(epoch, steps):
    for step in steps:
        if step[1] > epoch:
            print("Setting learning rate to {}".format(step[0]))
            return step[0]
    # print("Setting learning rate to {}".format(steps[-1][0]))
    return steps[-1][0]


def dice_coef(y_true, y_pred):
    # y_true = K.cast(y_true, y_pred.dtype)
    y_true_f = K.flatten(y_true)
    # print(tf.reduce_max(y_true_f))
    y_pred_f = K.flatten(y_pred)
    # print(tf.reduce_max(y_pred_f))
    y_true_f = K.cast(y_true_f, y_pred_f.dtype)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    # print("intersection", intersection)
    # print("union", (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1))
    return (2.0 * intersection + 1) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1
    )


def dice_coef_loss(y_true, y_pred):
    return 1 - (dice_coef(y_true, y_pred))


def dice_coef_wrapper(m):
    def dice_coeff_ch(y_true, y_pred):
        return dice_coef(y_true[..., m], y_pred[..., m])

    dice_coeff_ch.__name__ = "dice_coeff_ch" + str(m)
    return dice_coeff_ch


def dice_coef_wrapper_cell_type(m, n_contour_classes):
    idx_start = n_contour_classes

    def dice_coeff_ch_cell_type(y_true, y_pred):
        return dice_coef(y_true[..., idx_start + m], y_pred[..., m])

    dice_coeff_ch_cell_type.__name__ = "dice_coeff_cell_type" + str(m)
    return dice_coeff_ch_cell_type


# can be called on general inputs
def weighted_categorical_crossentropy(train_class_weights, masked_classification=False):
    shape = [1] * 4
    shape[-1] = len(train_class_weights)
    train_class_weights = np.broadcast_to(train_class_weights, shape)
    train_class_weights = K.constant(train_class_weights)

    def _weighted_categorical_crossentropy(y_true, y_pred):
        if masked_classification:
            mask = tf.expand_dims(
                K.cast(tf.reduce_sum(y_true, axis=-1) >= 0, K.floatx()), -1
            )
        else:
            mask = 1
        y_pred /= tf.reduce_sum(y_pred + _epsilon, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, _epsilon, 1.0 - _epsilon)
        # standard definition of cross entropy it sum over classes and average over samples (batches*H*W)
        loss = -tf.reduce_mean(
            tf.reduce_sum(train_class_weights * mask * y_true * K.log(y_pred), axis=-1)
        )
        # loss = - tf.reduce_mean(tf.reduce_sum(train_class_weights*y_true*K.log(y_pred),axis= -1))
        return loss

    return _weighted_categorical_crossentropy


# to be called by CISCA model
def weighted_categorical_crossentropy_loss(
    n_contour_classes, n_celltype_classes, train_class_weights, masked_classification
):
    _wcce = weighted_categorical_crossentropy(
        train_class_weights, masked_classification
    )
    # getting the channels corresponding to the classes of interest (cell types)
    # from the ground truth tensor y_true
    idx_start = n_contour_classes
    idx_end = idx_start + int(n_celltype_classes > 1) * (1 + n_celltype_classes)

    def _weighted_categorical_crossentropy_loss(y_true, y_pred):
        return _wcce(y_true[..., idx_start:idx_end], y_pred)

    return _weighted_categorical_crossentropy_loss


# # can be called on general inputs
# def focal_tversky_old(alpha=0.7, gamma=0.75):
#     def _focal_tversky(y_true,y_pred):
#         y_true_pos = K.flatten(y_true)
#         y_pred_pos = K.flatten(y_pred)
#         true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
#         #print(true_pos)
#         false_neg = tf.reduce_sum(y_true_pos * (1-y_pred_pos))
#         #print(false_neg)
#         false_pos = tf.reduce_sum((1-y_true_pos)*y_pred_pos)
#         #print(false_pos)
#         pt_1 = (true_pos + _epsilon)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + _epsilon)
#         pt_1 = K.clip(pt_1, _epsilon, 1-_epsilon)
#         return K.pow((1-pt_1), gamma)
#     return _focal_tversky

# # can be called on general inputs
# def focal_tversky(alpha=0.7, gamma=0.75):
#     def _focal_tversky(y_true,y_pred):
#         y_true_pos = K.flatten(y_true)
#         y_pred_pos = K.flatten(y_pred)
#         true_pos = tf.reduce_sum(y_true_pos*y_pred_pos)
#         tpfnfp  = tf.reduce_sum(alpha*y_true_pos+(1-alpha)*y_pred_pos)
#         tvindex = (true_pos + _epsilon)/(tpfnfp + _epsilon)
#         tvindex = K.clip(tvindex, _epsilon, 1-_epsilon)
#         return K.pow((1-tvindex), gamma)
#     return _focal_tversky

# # to be called by CISCA model
# def focal_tversky_loss(n_contour_classes,n_celltype_classes,alpha=0.7, gamma=0.75):
#     _ft  = focal_tversky(alpha,gamma)
#     idx_start = n_contour_classes
#     idx_end = idx_start+int(n_celltype_classes>1)*(1+n_celltype_classes)
#     def _focal_tversky_loss(y_true, y_pred):
#         if n_celltype_classes > 1:
#             totalloss = 0
#             for j in np.arange(n_celltype_classes+1):
#                 totalloss += _ft(y_true[...,idx_start+j], y_pred[...,j])
#         else:
#             totalloss = _ft(y_true[...,idx_start:idx_end], y_pred)
#         return totalloss
#         # return _ft(y_true[...,idx_start:idx_end], y_pred)
#     return _focal_tversky_loss


def focal_tversky_loss(n_contour_classes, n_celltype_classes, alpha=0.7, gamma=1):
    idx_start = n_contour_classes
    idx_end = idx_start + int(n_celltype_classes > 1) * (1 + n_celltype_classes)

    def _focal_tversky(y_true, y_pred):
        true_pos = tf.reduce_sum(
            y_true[..., idx_start:idx_end] * y_pred, axis=(0, 1, 2)
        )
        false_neg = tf.reduce_sum(
            y_true[..., idx_start:idx_end] * (1 - y_pred), axis=(0, 1, 2)
        )
        false_pos = tf.reduce_sum(
            (1 - y_true[..., idx_start:idx_end]) * y_pred, axis=(0, 1, 2)
        )
        pt_1 = (true_pos + _epsilon) / (
            true_pos + alpha * false_neg + (1 - alpha) * false_pos + _epsilon
        )
        pt_1 = K.clip(pt_1, _epsilon, 1 - _epsilon)
        return tf.reduce_mean(
            (1 - pt_1) ** gamma
            * (
                _epsilon
                + K.cast(
                    tf.reduce_sum(y_true[..., idx_start:idx_end], axis=(0, 1, 2)) > 0,
                    K.floatx(),
                )
            )
        )

    return _focal_tversky


def stardist_tversky_loss(n_contour_classes, n_celltype_classes, alpha=0.7, gamma=1):
    """focal tversky loss (stardist implementation)

    loss(x,y) = xy / (1-(a+b)) xy + ax + by
    loss(x,y) = xy / ax + by   (if a+b=1)


    Abraham, Nabila, and Naimul Mefraz Khan. "A novel focal tversky loss function with improved attention u-net for lesion segmentation." 2019 IEEE 16th international symposium on biomedical imaging (ISBI 2019). IEEE, 2019.

    Special cases:

        dice loss    -> gamma=1, alpha=0.5
        tversky loss -> gamma=1

    """

    # print(gamma)
    assert alpha >= 0 and alpha <= 1
    idx_start = n_contour_classes
    idx_end = idx_start + int(n_celltype_classes > 1) * (1 + n_celltype_classes)

    def _stardist_tversky_loss(y_true, y_pred):
        inter = y_true[..., idx_start:idx_end] * y_pred
        over = alpha * y_true[..., idx_start:idx_end] + (1 - alpha) * y_pred
        loss = (inter + _epsilon) / (over + _epsilon)
        loss = K.clip(loss, _epsilon, 1 - _epsilon)
        loss = tf.reduce_mean(K.pow(1 - loss, gamma))
        return loss

    return _stardist_tversky_loss


def dice_coef_minority_loss(
    n_contour_classes, n_celltype_classes, train_class_weights_2
):
    idx_start = n_contour_classes

    def _dice_coef_minority_loss(y_true, y_pred):
        _dcl = 0
        for j in np.arange(n_celltype_classes + 1):
            _dcl = _dcl + train_class_weights_2[j] * dice_coef_loss(
                y_true[..., idx_start + j], y_pred[..., j]
            )
        return _dcl

    return _dice_coef_minority_loss


# to be called by CISCA model
def compound_focal_tversky_wcce_loss(
    n_contour_classes,
    n_celltype_classes,
    train_celltype_loss_weight,
    train_class_weights,
    masked_classification=True,
    alpha=0.7,
    gamma=1,
):
    """sum of weighted cce and tversky loss"""
    _wcce = weighted_categorical_crossentropy_loss(
        n_contour_classes,
        n_celltype_classes,
        train_class_weights,
        masked_classification,
    )
    # print("_wcce", _wcce)
    _ft = stardist_tversky_loss(n_contour_classes, n_celltype_classes, alpha, gamma)

    # _ft = stardist_tversky_loss(n_contour_classes,n_celltype_classes,alpha,gamma)
    # print("_ft", _ft)
    def _compound_focal_tversky_wcce(y_true, y_pred):
        # print(train_celltype_loss_weight)
        # print(_ft(y_true, y_pred))
        return train_celltype_loss_weight * (
            _wcce(y_true, y_pred) + _ft(y_true, y_pred)
        )

    return _compound_focal_tversky_wcce


# to be called by CISCA model
def compound_focal_tversky_wcce_dice_loss(
    n_contour_classes,
    n_celltype_classes,
    train_celltype_loss_weight,
    train_class_weights,
    train_class_weights_2,
    masked_classification=True,
    alpha=0.7,
    gamma=1,
):
    """sum of weighted cce and tversky loss"""
    _wcce = weighted_categorical_crossentropy_loss(
        n_contour_classes,
        n_celltype_classes,
        train_class_weights,
        masked_classification,
    )
    # print("_wcce", _wcce)
    _ft = stardist_tversky_loss(n_contour_classes, n_celltype_classes, alpha, gamma)
    # print("_ft", _ft)
    _dcl = dice_coef_minority_loss(
        n_contour_classes, n_celltype_classes, train_class_weights_2
    )

    def _compound_focal_tversky_dice_wcce(y_true, y_pred):
        # print(train_celltype_loss_weight)
        # print(_ft(y_true, y_pred))
        return train_celltype_loss_weight * (
            _wcce(y_true, y_pred) + _ft(y_true, y_pred) + _dcl(y_true, y_pred)
        )

    return _compound_focal_tversky_dice_wcce


def categorical_crossentropy(y_true, y_pred):
    return weighted_categorical_crossentropy((1,))(y_true, y_pred)
    # y_pred = tf.convert_to_tensor(y_pred)
    # y_true = K.cast(y_true, y_pred.dtype)
    # _epsilon = tf.convert_to_tensor(_epsilon, y_pred.dtype.base_dtype)
    # y_pred = K.clip(y_pred, _epsilon, 1. - _epsilon)
    # return tf.reduce_mean(tf.reduce_sum(-y_true*K.log(y_pred),axis=-1))


# to be called by CISCA model
def categorical_crossentropy_loss(n_contour_classes):
    def _categorical_crossentropy(y_true, y_pred):
        return categorical_crossentropy(y_true[..., 0:n_contour_classes], y_pred)

    return _categorical_crossentropy


# to be called by CISCA model
def compound_dice_cce_loss(n_contour_classes, train_contour_weights):
    def _compound_dice_cce(y_true, y_pred):
        _cce = train_contour_weights[0] * categorical_crossentropy_loss(
            n_contour_classes
        )(y_true, y_pred)
        # 4 contour classes --> dice for 0,1,2
        # 3 contour classes --> dice for 0,1
        # 2 contour classes --> dice for 0
        for j in np.arange(n_contour_classes - 1):
            _cce = _cce + train_contour_weights[j + 1] * dice_coef_loss(
                y_true[..., j], y_pred[..., j]
            )
        return _cce

    return _compound_dice_cce


# to be called by CISCA model
def compound_dice_bce_loss(train_contour_weights):
    def _compound_dice_bce(y_true, y_pred):
        _bce = train_contour_weights[0] * binary_crossentropy_loss(y_true, y_pred)
        _dice = train_contour_weights[1] * dice_coef_loss(
            y_true[..., 0], y_pred[..., 0]
        )
        return _bce + _dice

    return _compound_dice_bce


def binary_crossentropy(from_logits=False):
    def _binary_crossentropy(y_true, y_pred):
        if not from_logits:
            # transform back to logits
            y_pred = tf.where(
                tf.equal(y_true, 0),
                tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon),
                y_pred,
            )
            y_pred = tf.where(
                tf.equal(y_true, 1),
                tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon),
                y_pred,
            )
            y_pred = tf.math.log(y_pred / (1 - y_pred))

        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        )

    return _binary_crossentropy


# to be called by CISCA model
def binary_crossentropy_loss(from_logits=False):
    def _binary_crossentropy_loss(y_true, y_pred):
        return binary_crossentropy(from_logits)(y_true[..., 0], y_pred)

    return _binary_crossentropy_loss


# to be called by CISCA model
def reg_loss(
    n_contour_classes,
    n_celltype_classes,
    train_reg_loss_weights,
    masked_regression=True,
    mae=True,
    diag_dist=True,
):
    def _reg_loss(y_true, y_pred):
        idx_shift = n_contour_classes + int(n_celltype_classes > 1) * (
            n_celltype_classes + 1
        )
        if masked_regression:
            # norm_factor = (tf.reduce_mean(y_true[..., -1]) + _epsilon)
            mask = tf.expand_dims(y_true[..., -1], -1)
            if mae:
                _rl = (
                    masked_loss_mae(mask)(y_true[..., idx_shift:-1], y_pred)
                    * train_reg_loss_weights[0]
                    + msge(mask, diag_dist=diag_dist)(y_true[..., idx_shift:-1], y_pred)
                    * train_reg_loss_weights[1]
                )  # /norm_factor
            else:
                _rl = (
                    mse(mask)(y_true[..., idx_shift:-1], y_pred)
                    * train_reg_loss_weights[0]
                    + msge(mask, diag_dist=diag_dist)(y_true[..., idx_shift:-1], y_pred)
                    * train_reg_loss_weights[1]
                )  # /norm_factor
        else:
            # mask absent as last channel in y_true
            _rl = (
                mse()(y_true[..., idx_shift:], y_pred) * train_reg_loss_weights[0]
                + msge(diag_dist=diag_dist)(y_true[..., idx_shift:], y_pred)
                * train_reg_loss_weights[1]
            )
        return _rl

    return _reg_loss


def generic_masked_loss(
    mask, loss, weights=1, norm_by_mask=True, reg_weight=0, reg_penalty=tf.math.abs
):
    def _loss(y_true, y_pred):
        actual_loss = tf.reduce_mean(mask * weights * loss(y_true, y_pred))
        norm_mask = (tf.reduce_mean(mask) + _epsilon) if norm_by_mask else 1
        if reg_weight > 0:
            reg_loss = (1 - mask) * reg_penalty(y_pred)
            return actual_loss / norm_mask + reg_weight * reg_loss
        else:
            return actual_loss / norm_mask

    return _loss


def masked_loss(mask, penalty, reg_weight, norm_by_mask):
    loss = lambda y_true, y_pred: penalty(y_true - y_pred)
    return generic_masked_loss(
        mask, loss, reg_weight=reg_weight, norm_by_mask=norm_by_mask
    )


def masked_loss_mae(mask=1, reg_weight=0.01, norm_by_mask=True):
    return masked_loss(
        mask, tf.math.abs, reg_weight=reg_weight, norm_by_mask=norm_by_mask
    )


def mse(mask=1):
    def _mse(y_true, y_pred):
        loss = y_pred - y_true
        return tf.reduce_mean(K.cast(mask, loss.dtype) * loss**2)

    return _mse


def msge(mask=1, diag_dist=True):
    def _msge(y_true, y_pred):
        pred_grad = get_gradient_hvd(y_pred, diag_dist=diag_dist)
        true_grad = get_gradient_hvd(y_true, diag_dist=diag_dist)
        loss = pred_grad - true_grad
        actual_loss = tf.reduce_mean(K.cast(mask, loss.dtype) * loss**2)
        norm_mask = tf.reduce_mean(K.cast(mask, loss.dtype)) + _epsilon
        return actual_loss / norm_mask

    return _msge


def mae_loss(n_contour_classes, n_celltype_classes, masked_regression=True):
    def _mae_loss(y_true, y_pred):
        idx_shift = n_contour_classes + int(n_celltype_classes > 1) * (
            n_celltype_classes + 1
        )
        # print("idx_shift",idx_shift)
        if masked_regression:
            mask = tf.expand_dims(y_true[..., -1], -1)
            _mse = masked_loss_mae(mask)(y_true[..., idx_shift:-1], y_pred)
            # print("_mse",_mse)
        else:
            # mask absent as last channel in y_true
            _mse = masked_loss_mae()(y_true[..., idx_shift:], y_pred)
        return _mse

    return _mae_loss


# to be called by CISCA model
def mse_loss(n_contour_classes, n_celltype_classes, masked_regression=True):
    def _mse_loss(y_true, y_pred):
        idx_shift = n_contour_classes + int(n_celltype_classes > 1) * (
            n_celltype_classes + 1
        )
        # print("idx_shift",idx_shift)
        if masked_regression:
            mask = tf.expand_dims(y_true[..., -1], -1)
            _mse = mse(mask)(y_true[..., idx_shift:-1], y_pred)
            # print("_mse",_mse)
        else:
            # mask absent as last channel in y_true
            _mse = mse()(y_true[..., idx_shift:], y_pred)
        return _mse

    return _mse_loss


# to be called by CISCA model
def msge_loss(
    n_contour_classes, n_celltype_classes, masked_regression=True, diag_dist=True
):
    def _msge_loss(y_true, y_pred):
        idx_shift = n_contour_classes + int(n_celltype_classes > 1) * (
            n_celltype_classes + 1
        )
        if masked_regression:
            mask = tf.expand_dims(y_true[..., -1], -1)
            _msge = msge(mask, diag_dist=diag_dist)(y_true[..., idx_shift:-1], y_pred)
        else:
            # mask absent as last channel in y_true
            _msge = msge(diag_dist=diag_dist)(y_true[..., idx_shift:], y_pred)
        return _msge

    return _msge_loss


def get_gradient_hvd(distance_map, h_ch=0, v_ch=1, tl_ch=2, bl_ch=3, diag_dist=True):
    """
    Calculate the partial differentiation for distance maps (horizontal, vertical and diagonal)

    The partial differentiation is approximated by calculating the central difference
    which is obtained by using Sobel kernel of size 5x5. The boundary is zero-padded
    when channel is convolved with the Sobel kernel.

    Args:
        distance_map (tensor): tensor of shape NHWC with C should be 2 or 4 depending on presence or absence of diagonal
        distances (2 only vertical and horizontal, 4 also top left and bottom left diagonals)
        h_ch(int) : index within C axis of `distance_map` that corresponds to horizontal channel
        v_ch(int) : index within C axis of `distance_map` that corresponds to vertical channel
        tl_ch(int) : index within C axis of `distance_map` that corresponds to top left channel
        tl_ch(int) : index within C axis of `distance_map` that corresponds to bottom left channel
    """

    def get_sobel_kernel(size):
        assert size % 2 == 1, "Must be odd, get size=%d" % size

        h_range = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
        v_range = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
        h, v = tf.meshgrid(h_range, v_range)
        kernel_h = h / (h * h + v * v + _epsilon)
        kernel_v = v / (h * h + v * v + _epsilon)
        kernel_tl = kernel_h + kernel_v

        mh = tf.reshape(kernel_h, [5, 5, 1, 1])
        mv = tf.reshape(kernel_v, [5, 5, 1, 1])
        mtl = tf.reshape(kernel_tl, [5, 5, 1, 1])
        mbl = tf.reshape(
            tf.image.rot90(tf.reshape(kernel_tl, [1, 5, 5, 1]), 1), [5, 5, 1, 1]
        )

        return mh, mv, mtl, mbl

    mh, mv, mtl, mbl = get_sobel_kernel(5)

    # n_distance_maps = tf.shape(distance_map)[3]

    # print(n_distance_maps)

    if not diag_dist:
        h = tf.expand_dims(distance_map[..., h_ch], axis=-1)
        v = tf.expand_dims(distance_map[..., v_ch], axis=-1)
        dh = tf.nn.conv2d(h, mh, strides=[1, 1, 1, 1], padding="SAME")
        dv = tf.nn.conv2d(v, mv, strides=[1, 1, 1, 1], padding="SAME")
        output = tf.concat([dh, dv], axis=-1)

    else:
        tl = tf.expand_dims(distance_map[..., tl_ch], axis=-1)
        bl = tf.expand_dims(distance_map[..., bl_ch], axis=-1)
        dtl = tf.nn.conv2d(tl, mtl, strides=[1, 1, 1, 1], padding="SAME")
        dbl = tf.nn.conv2d(bl, mbl, strides=[1, 1, 1, 1], padding="SAME")
        output = tf.concat([dtl, dbl], axis=-1)

    return output


# EXTRA FUNCTIONS FOR OTHER METHODS (HOVERNET, DEEPCELL)


def dice_coef_rounded_ch0(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    y_true_f = K.cast(y_true_f, y_pred_f.dtype)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1
    )


def dice_coef_rounded_ch1(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 1]))
    y_pred_f = K.flatten(K.round(y_pred[..., 1]))
    # y_pred = tf.convert_to_tensor(y_pred)
    y_true_f = K.cast(y_true_f, y_pred_f.dtype)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1
    )


def dice_coef_rounded_ch2(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 2]))
    y_pred_f = K.flatten(K.round(y_pred[..., 1]))
    # y_pred = tf.convert_to_tensor(y_pred)
    y_true_f = K.cast(y_true_f, y_pred_f.dtype)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1
    )


def dice_coef_rounded_ch3(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 3]))
    y_pred_f = K.flatten(K.round(y_pred[..., 3]))
    # y_pred = tf.convert_to_tensor(y_pred)
    y_true_f = K.cast(y_true_f, y_pred_f.dtype)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1
    )


def dice_coef_rounded_ch1deepcell(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 1]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    # y_pred = tf.convert_to_tensor(y_pred)
    y_true_f = K.cast(y_true_f, y_pred_f.dtype)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1
    )


###################################### HOVERNET LOSSES ###############################

# true:
# 0 foreground
# 1 background
# 2 background
# 3 class 1
# 4 class 2
# 5 = 2 + 3 class 3
# 6 = 2 + 3 + 1 hor
# 7 ver

def xentropy_loss_hover(true, pred, reduction="mean"):
    epsilon = 10e-8
    # scale preds so that the class probs of each sample sum to 1
    pred = pred / tf.math.reduce_sum(pred, -1, keepdims=True)
    # manual computation of crossentropy
    pred = tf.clip_by_value(pred, epsilon, 1.0 - epsilon)
    loss = -tf.math.reduce_sum((true * tf.math.log(pred)), -1, keepdims=True)
    loss = (
        tf.math.reduce_mean(loss) if reduction == "mean" else tf.math.reduce_sum(loss)
    )
    return loss


def dice_loss_hover(true, pred, smooth=1e-3):
    """`pred` and `true` must be of torch.float32. Assuming of shape NxHxWxC."""
    inse = tf.math.reduce_sum(pred * true, (0, 1, 2))
    l = tf.math.reduce_sum(pred, (0, 1, 2))
    r = tf.math.reduce_sum(true, (0, 1, 2))
    loss = 1.0 - (2.0 * inse + smooth) / (l + r + smooth)
    loss = tf.math.reduce_sum(loss)
    return loss


def nploss_hover(true, pred):
    return xentropy_loss_hover(true[..., 0:2], pred) + dice_loss_hover(
        true[..., 0:2], pred
    )


def hvloss_hover(n_celltype_classes):
    def _hvloss_hover(true, pred):
        # print(pred.shape)
        # print(true[..., idx_shift:].shape)
        # return msge_loss_hover(true, pred,n_celltype_classes)+mse_loss_hover(true, pred,n_celltype_classes)
        return msge_loss_hover_metric(n_celltype_classes)(
            true, pred
        ) + mse_loss_hover_metric(n_celltype_classes)(true, pred)

    return _hvloss_hover


def tploss_hover(n_celltype_classes):
    def _tploss_hover(true, pred):
        return xentropy_loss_hover(
            true[..., 2 : 2 + n_celltype_classes + 1], pred
        ) + dice_loss_hover(true[..., 2 : 2 + n_celltype_classes + 1], pred)

    return _tploss_hover


def mse_loss_hover_metric(n_celltype_classes):
    def _mse_loss_hover_metric(true, pred):
        idx_shift = 2 + int(n_celltype_classes > 1) * (n_celltype_classes + 1)
        return mse_loss_hover(true[..., idx_shift:], pred)

    return _mse_loss_hover_metric


def mse_loss_hover(true, pred):
    # print(true.shape)
    # print(pred.shape)
    loss = pred - true
    loss = tf.math.reduce_mean(loss * loss)
    return loss


def msge_loss_hover_metric(n_celltype_classes):
    def _msge_loss_hover_metric(true, pred):
        return msge_loss_hover(true, pred, n_celltype_classes)

    return _msge_loss_hover_metric


def msge_loss_hover(true, pred, n_celltype_classes):
    idx_shift = 2 + int(n_celltype_classes > 1) * (n_celltype_classes + 1)
    mask = tf.expand_dims(true[..., 0], -1)
    mask = tf.concat([mask, mask], -1)
    true = true[..., idx_shift:]

    def get_gradient_hv(distance_map, h_ch=0, v_ch=1):
        def get_sobel_kernel(size):
            assert size % 2 == 1, "Must be odd, get size=%d" % size

            h_range = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
            v_range = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
            h, v = tf.meshgrid(h_range, v_range)
            kernel_h = h / (h * h + v * v + _epsilon)
            kernel_v = v / (h * h + v * v + _epsilon)

            mh = tf.reshape(kernel_h, [5, 5, 1, 1])
            mv = tf.reshape(kernel_v, [5, 5, 1, 1])

            return mh, mv

        mh, mv = get_sobel_kernel(5)

        h = tf.expand_dims(distance_map[..., h_ch], axis=-1)
        v = tf.expand_dims(distance_map[..., v_ch], axis=-1)
        dh = tf.nn.conv2d(h, mh, strides=[1, 1, 1, 1], padding="SAME")
        dv = tf.nn.conv2d(v, mv, strides=[1, 1, 1, 1], padding="SAME")
        output = tf.concat([dh, dv], axis=-1)

        return output

    pred_grad = get_gradient_hv(pred)
    true_grad = get_gradient_hv(true)
    loss = pred_grad - true_grad
    actual_loss = tf.reduce_sum(K.cast(mask, loss.dtype) * loss**2)
    norm_mask = tf.reduce_sum(K.cast(mask, loss.dtype)) + _epsilon
    return actual_loss / norm_mask

