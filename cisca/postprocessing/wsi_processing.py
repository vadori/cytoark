from cisca.imutils import tile_image, untile_image
import math
import numpy as np
import gc


def tile_predict(
    model,
    X_test,
    input_shape=(256, 256),
    stride_ratio=0.5,
    force_spline=False,
    spline_power=2,
    batch_size=16,
):
    print(X_test.shape)
    # Tile X_test into overlapping tiles
    X_tiles, tiles_info_X = tile_image(
        X_test, model_input_shape=input_shape, stride_ratio=stride_ratio
    )

    del X_test
    gc.collect()

    # Predict on tiles
    tot_tiles = X_tiles.shape[0]
    print(tot_tiles)
    tot_tiles_proc = 1800
    num_pred = int(np.ceil(tot_tiles / tot_tiles_proc))
    y_pred = [[], []]
    for i in range(0, num_pred):
        print("Processing batch {0} out of {1}".format(i + 1, num_pred))
        y_pred_subs = model.predict(
            X_tiles[i * tot_tiles_proc : np.min((tot_tiles, (i + 1) * tot_tiles_proc))],
            batch_size=batch_size,
        )
        print((y_pred_subs[0].shape))
        y_pred[0].append(y_pred_subs[0])
        y_pred[1].append(y_pred_subs[1])
        gc.collect()

    y_pred[0] = np.vstack(y_pred[0])
    y_pred[1] = np.vstack(y_pred[1])
    # print((len(y_pred[0])))
    # print(((y_pred[0][0].shape)))
    # print(((y_pred[0][1].shape)))
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


def tile_predict_hovernet(model, X_test):
    """
    Using 'model' to generate the prediction of image X_test

    Args:
        X_test : input image to be segmented. It will be split into patches
            to run the prediction upon before being assembled back
    """
    step_size = [80, 80]
    msk_size = [80, 80]
    win_size = [270, 270]
    inf_batch_size = 1

    def get_last_steps(length, msk_size, step_size):
        nr_step = math.ceil((length - msk_size) / step_size)
        last_step = (nr_step + 1) * step_size
        return int(last_step), int(nr_step + 1)

    im_h = X_test.shape[0]
    im_w = X_test.shape[1]

    last_h, nr_step_h = get_last_steps(im_h, msk_size[0], step_size[0])
    last_w, nr_step_w = get_last_steps(im_w, msk_size[1], step_size[1])

    diff_h = win_size[0] - step_size[0]
    padt = diff_h // 2
    padb = last_h + win_size[0] - im_h

    diff_w = win_size[1] - step_size[1]
    padl = diff_w // 2
    padr = last_w + win_size[1] - im_w

    X_test = np.lib.pad(X_test, ((padt, padb), (padl, padr), (0, 0)), "reflect")

    #### TODO: optimize this
    sub_patches = []
    # generating subpatches from orginal
    for row in range(0, last_h, step_size[0]):
        for col in range(0, last_w, step_size[1]):
            win = X_test[row : row + win_size[0], col : col + win_size[1]]
            sub_patches.append(win)

    pred_map = []

    for j in np.arange(len(sub_patches)):
        mini_batch = sub_patches[:inf_batch_size]
        sub_patches = sub_patches[inf_batch_size:]
        mini_output = model.predict(np.array(mini_batch), verbose=0)

        if len(mini_output) == 2:
            mini_output = np.concatenate(
                (np.squeeze(mini_output[0])[..., 0:1], np.squeeze(mini_output[1])),
                axis=2,
            )
        else:
            mini_output = np.concatenate(
                (
                    np.squeeze(mini_output[2]),
                    np.squeeze(mini_output[0])[..., 0:1],
                    np.squeeze(mini_output[1]),
                ),
                axis=2,
            )

        # print("mini_output.shape:",mini_output.shape)
        # mini_output = np.split(mini_output, inf_batch_size, axis=0)
        pred_map.append(mini_output)

    # print(len(pred_map))
    # #### Assemble back into full image
    output_patch_shape = np.squeeze(pred_map[0]).shape
    ch = 1 if len(output_patch_shape) == 2 else output_patch_shape[-1]

    #### Assemble back into full image
    pred_map = np.squeeze(np.array(pred_map))
    pred_map = np.reshape(pred_map, (nr_step_h, nr_step_w) + pred_map.shape[1:])
    pred_map = (
        np.transpose(pred_map, [0, 2, 1, 3, 4])
        if ch != 1
        else np.transpose(pred_map, [0, 2, 1, 3])
    )
    pred_map = np.reshape(
        pred_map,
        (
            pred_map.shape[0] * pred_map.shape[1],
            pred_map.shape[2] * pred_map.shape[3],
            ch,
        ),
    )
    pred_map = np.squeeze(pred_map[:im_h, :im_w])  # just crop back to original size

    return pred_map


def tile_predict_framework(
    model,
    X_test,
    input_shape=(256, 256),
    stride_ratio=0.5,
    force_spline=False,
    spline_power=2,
    batch_size=16,
):
    print(X_test.shape)
    # Tile X_test into overlapping tiles
    X_tiles, tiles_info_X = tile_image(
        X_test, model_input_shape=input_shape, stride_ratio=stride_ratio
    )

    del X_test
    gc.collect()

    # Predict on tiles
    tot_tiles = X_tiles.shape[0]
    print("tot_tiles", tot_tiles)
    print("X_tiles.shape[0]", X_tiles.shape)

    inf_batch_size = 1
    y_pred = []

    for j in np.arange(tot_tiles):
        # print(j)

        mini_batch = X_tiles[:inf_batch_size]
        X_tiles = X_tiles[inf_batch_size:]
        # print("X_tiles.shape",X_tiles.shape)
        mini_output = model.predict(mini_batch, verbose=0)
        # print("mini_output.shape:",mini_output.shape)
        # mini_output = np.split(mini_output, inf_batch_size, axis=0)
        y_pred.append(mini_output)

    # print((len(y_pred[0])))
    # print(((y_pred[0][0].shape)))
    # print(((y_pred[0][1].shape)))
    # Untile predictions
    y_pred = np.vstack(y_pred)
    y_pred = untile_image(
        y_pred,
        tiles_info_X,
        model_input_shape=input_shape,
        power=spline_power,
        force=force_spline,
    )
    return y_pred
