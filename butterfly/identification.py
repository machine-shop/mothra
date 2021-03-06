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

from fastai.vision import load_learner, open_image
from pathlib import Path
from skimage.io import imsave
from skimage.util import img_as_ubyte
from tempfile import NamedTemporaryFile
from butterfly import binarization, connection


def _classification(bfly_rgb, weights):
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
    learner = load_learner(path=weights.parent, file=weights.name)

    with NamedTemporaryFile(suffix='.png', dir='.') as aux_fname:
        imsave(fname=aux_fname.name, arr=img_as_ubyte(bfly_rgb), check_contrast=False)
        bfly_aux = open_image(aux_fname.name)

    _, prediction, _ = learner.predict(bfly_aux)

    return int(prediction)


def predict_position(bfly_rgb, weights='./models/id_position.pkl'):
    """Predicts the position of the Lepidoptera in `bfly_rgb`.

    Parameters
    ----------
    bfly_rgb : 3D array
        RGB image of the Lepidoptera (ruler and tags cropped out).
    weights : str or pathlib.Path, optional
        Path of the file containing weights.

    Returns
    -------
    prediction : str
        Classification obtained from `bfly_rgb`, being "right-side_up" or
        "upside_down".
    """
    position = {
        0: 'upside_down',
        1: 'right-side_up'
    }
    prediction = _classification(bfly_rgb, weights)

    return position.get(prediction)


def predict_gender(bfly_rgb, weights='./models/id_gender.pkl'):
    """Predicts the gender of the Lepidoptera in `bfly_rgb`.

    Parameters
    ----------
    bfly_rgb : 3D array
        RGB image of the Lepidoptera (ruler and tags cropped out).
    weights : str or pathlib.Path, optional
        Path of the file containing weights.

    Returns
    -------
    prediction : str
        Classification obtained from `bfly_rgb`, being "female" or
        "male".
    """
    gender = {
        0: 'female',
        1: 'male'
    }
    prediction = _classification(bfly_rgb, weights)

    return gender.get(prediction)


def main(image_rgb, top_ruler, axes=None):
    """Identifies position and gender of the Lepidoptera in `image_rgb`.

    Parameters
    ---------
    image_rgb : 3D array
        RGB image of the entire picture.
    top_ruler : int
        Top point in the Y axis where the ruler starts.

    Returns
    -------
    position : str
        Position of the Lepidoptera: `right-side_up` or `upside_down`.
    gender : str
        Gender of the Lepidoptera, or N/A if position is `upside_down`.
    """
    label_edge = binarization.find_tags_edge(image_rgb, top_ruler, axes)
    bfly_rgb = image_rgb[:top_ruler, :label_edge]

    print('Identifying position...')
    position = predict_position(bfly_rgb, weights='./models/id_position.pkl')
    print(f'* Position: {position}')

    if position == 'right-side_up':
        print('Identifying gender...')
        gender = predict_gender(bfly_rgb, weights='./models/id_gender.pkl')
        print(f'* Gender: {gender}')
    else:
        gender = 'N/A'

    return position, gender
