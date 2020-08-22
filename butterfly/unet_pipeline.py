import numpy as np

from fastai.vision import get_image_files, open_image
from pathlib import Path
from skimage import io
from shutil import rmtree
from skimage.color import rgb2gray 
from skimage import util


def save_crop_image(image, window_shape, pad_width, step, folder, index=0, is_multichannel=False):
    """
    """
    windows = util.view_as_windows(image, window_shape=window_shape,
                                   step=step)
    grid_shape = windows.shape[0:2]
    img_crop = np.vstack(windows)

    if not folder.is_dir():
        folder.mkdir()
    
    for idx, aux in enumerate(img_crop):
        fname = '%03d_crop-%03d.png' % (index, idx)
        if is_multichannel:
            aux = aux[0]
        io.imsave(folder/fname, util.img_as_ubyte(aux), check_contrast=False)

    return grid_shape


def make_it_squared_and_multiple(image, window_shape=(576, 576, 3), step=512, is_multichannel=False):
    """
    Notes
    -----
        unpad_width should have the size that will be padded later.
    """
    if is_multichannel:
        rows_img, cols_img, cnls_img = image.shape
        rows_win, cols_win, cnls_win = window_shape
    else:
        image = rgb2gray(image)
        rows_img, cols_img = image.shape
        rows_win, cols_win = window_shape

    max_dim_img = max(image.shape)
    min_desired_size = np.ceil(max_dim_img / step) * step

    pad_rows, pad_cols = (np.ceil(
        (min_desired_size - (rows_img, cols_img)) / 2)
                         ).astype('int')

    rows_new_size = rows_img + 2 * pad_rows
    cols_new_size = cols_img + 2 * pad_cols

    aux_rows, aux_cols = pad_rows, pad_cols

    if rows_new_size > min_desired_size:
        pad_rows -= 1
    if cols_new_size > min_desired_size:
        pad_cols -= 1

    if is_multichannel:
        image = np.pad(image, [(pad_rows, aux_rows),
                               (pad_cols, aux_cols),
                               (0, 0)])
    else:
        image = np.pad(image, [(pad_rows, aux_rows),
                               (pad_cols, aux_cols)])
    
    pad_width = ((pad_rows, aux_rows), (pad_cols, aux_cols))

    return image, pad_width


def predict_on_crops(folder, learner):
    predictions = []
    filenames = sorted(get_image_files(folder))
    for filename in filenames:
        img_temp = open_image(filename)
        _, _, img_pred = learner.predict(img_temp)
        predictions.append(img_pred[1].data.numpy())  # [0]: background.
    return predictions


def _aux_predict(predictions, pad_width=16, grid_shape=(10, 10),
                 num_class=2, multichannel=False):
    """
    """
    aux_slice = len(grid_shape) * [slice(pad_width, -pad_width)]
    if multichannel:
        # multiply pad_width with everyone, except depth and colors
        output = np.zeros((predictions.shape[0],
                           *np.asarray(predictions.shape[1:-1]) - 2*pad_width,
                           predictions.shape[-1]))
        for idx, pred in enumerate(predictions):
            output[idx] = pred[(*aux_slice, slice(None))]
    else:
        output = predictions[(slice(None), *aux_slice, 0)]

    if output.ndim == 3:
        output = util.montage(output,
                              fill=0,
                              grid_shape=grid_shape,
                              multichannel=multichannel)
    elif output.ndim == 4:
        output = montage_3d(output,
                            fill=0,
                            grid_shape=grid_shape,
                            multichannel=multichannel)
    return output


def predict(image, window_shape, pad_width, step, learner,
            is_multichannel=False):
    """
    """
    img_shape = image.shape
    image, orig_pad = make_it_squared_and_multiple(image,
                                                   window_shape=window_shape,
                                                   is_multichannel=True)

    folder_temp = Path('.tmp')
    grid_shape = save_crop_image(image,
                                 window_shape=window_shape,
                                 pad_width=pad_width,
                                 step=step,
                                 folder=folder_temp,
                                 index=0,
                                 is_multichannel=is_multichannel)
 
    partial_preds = predict_on_crops(folder=folder_temp, learner=learner)
    prediction = _aux_predict(predictions=np.asarray(partial_preds)[...,
                                                                    np.newaxis],
                              pad_width=pad_width,
                              grid_shape=grid_shape)
    prediction = np.pad(prediction, pad_width=32)  # difference between original shape and result [TODO: a better way to make the padding?]
    rmtree(folder_temp)

    fixing_pad = (orig_pad[0][0] + pad_width, orig_pad[1][0] + pad_width)
    prediction = prediction[fixing_pad[0]:img_shape[0] + fixing_pad[0],
                            fixing_pad[1]:img_shape[1] + fixing_pad[1]] > 0.5
        
    return prediction
