import numpy as np
import warnings

warnings.simplefilter('ignore', UserWarning)
"""
Ignoring user warning until fastai/pytorch upgrade. Current one:

torch/nn/functional.py:3103: UserWarning: The default behavior for
interpolate/upsample with float scale_factor changed in 1.6.0 to align with
other frameworks/libraries, and now uses scale_factor directly, instead of
relying on the computed output size. If you wish to restore the old behavior,
please set recompute_scale_factor=True. See the documentation of nn.Upsample
for details.
"""

from fastai.vision.data import Image
from fastai.vision.learner import load_learner
from pathlib import Path
from skimage.io import imsave
from skimage.util import img_as_float32, img_as_ubyte
from torch import from_numpy
from mothra import binarization, connection


WEIGHTS_GENDER = './models/id_gender_test-3classes.pkl'


def _classification(image_rgb, weights):
    """Helping function. Classifies the input image according to `weights`.

    Parameters
    ----------
    bfly_rgb : 3D array
        RGB image of the Lepidoptera (ruler and tags cropped out).
    weights : str or pathlib.Path
        Path of the file containing weights.

    Returns
    -------
    prediction : int
        Prediction obtained with the given weights.

    Notes
    -----
    If a string is given in `weights`, it will be converted into a pathlib.Path
    object.
    """
    if isinstance(weights, str):
        weights = Path(weights)

    connection.download_weights(weights)

    # parameters here were defined when training the networks.
    learner = load_learner(fname=weights)

    _, prediction, _ = learner.predict(image_rgb)

    return int(prediction)


def predict_gender(image_rgb, weights=WEIGHTS_GENDER):
    """Predicts position and gender of the Lepidoptera in `image_rgb`.

    Parameters
    ----------
    image_rgb : (M, N, 3) ndarray
        RGB input image contaning Lepidoptera, ruler and tags.
    weights : str or pathlib.Path, optional
        Path of the file containing weights.

    Returns
    -------
    prediction : str
        Classification obtained from `image_rgb`, being "female",
        "male", or "upside_down".
    """
    pos_and_gender = {  # TODO: check if gender/position match!
        0: 'female',
        1: 'male',
        2: 'upside_down'
    }
    prediction = _classification(image_rgb, weights)

    return pos_and_gender.get(prediction)


def main(image_rgb):
    """Identifies position and gender of the Lepidoptera in `image_rgb`.

    Parameters
    ---------
    image_rgb : 3D array
        RGB image of the entire picture.

    Returns
    -------
    position : str
        Position of the Lepidoptera: `right-side_up` or `upside_down`.
    gender : str
        Gender of the Lepidoptera, or N/A if position is `upside_down`.
    """
    print('Identifying position and gender...')
    pos_and_gender = predict_gender(image_rgb, weights=WEIGHTS_GENDER)

    if pos_and_gender == 'upside_down':
        position = pos_and_gender
        gender = 'N/A'

        print(f'* Position: {position}\n * Gender: {gender}')

    else:
        position = 'right-side_up'
        gender = pos_and_gender

        print(f'* Position: {position}\n* Gender: {gender}')

    return position, gender