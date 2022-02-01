from fastai.vision.learner import load_learner
from pathlib import Path
from mothra import connection


WEIGHTS_GENDER = './models/id_gender_test-3classes.pkl'


def predict_gender(image_rgb, weights=WEIGHTS_GENDER):
    """Predicts position and gender of the lepidopteran in `image_rgb`,
    according to `weights`.

    Parameters
    ----------
    image_rgb : 3D array
        RGB image of the lepidopteran (ruler and tags cropped out).
    weights : str or pathlib.Path
        Path of the file containing weights.

    Returns
    -------
    prediction : string
        Prediction obtained with the given weights, between the classes
        `female`, `male`, or `upside_down`.
    probabilities : 1D array
        Probabilities of prediction returned by the network for each class.

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

    prediction, _, probabilities = learner.predict(image_rgb)
    print(prediction, probabilities)

    return prediction, probabilities


def main(image_rgb):
    """Identifies position and gender of the lepidopteran in `image_rgb`.

    Parameters
    ---------
    image_rgb : 3D array
        RGB image of the entire picture.

    Returns
    -------
    position : str
        Position of the lepidopteran: `right-side_up` or `upside_down`.
    gender : str
        Gender of the lepidopteran, or N/A if position is `upside_down`.
    """
    print('Identifying position and gender...')
    try:
        prediction, probabilities = predict_gender(image_rgb, weights=WEIGHTS_GENDER)

        if prediction == 'upside_down':
            position = prediction
            gender = 'N/A'
            print(f'* Position: {position}\n * Gender: {gender}')
        else:
            position = 'right-side_up'
            gender = prediction
            print(f'* Position: {position}\n* Gender: {gender}')
    except AttributeError:  # 'Compose' object has no attribute 'is_check_args'
        position = 'N/A'
        gender = 'N/A'
        print(f'* Could not calculate position and gender')

    return position, gender
