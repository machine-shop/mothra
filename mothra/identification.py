from fastai.vision.learner import load_learner
from pathlib import Path
from mothra import connection


WEIGHTS_CLASSES = './models/id_gender_test-3classes.pkl'
CLASSES = {0: 'upside_down', 1: 'female', 2: 'male'}


def predicting_classes(image_rgb, weights=WEIGHTS_CLASSES):
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
        Probabilities for the prediction returned by the network for each
        class.

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
    probabilities : 1D array
        Probabilities for the prediction returned by the network for each
        class.
    """
    print('Identifying position and gender...')
    try:
        prediction, probabilities = predicting_classes(image_rgb,
                                                       weights=WEIGHTS_CLASSES)

        # converting probabilities to numpy array and rounding the result
        probabilities = [round(prob, ndigits=4)
                         for prob in probabilities.tolist()]

        if prediction == 'down':
            position = 'upside_down'
            gender = 'N/A'
        else:
            position = 'right-side_up'
            gender = prediction
        print(f'* Position: {position}\n* Gender: {gender}')\

        print('Probabilities:')
        for idx, probability in enumerate(probabilities):
            print(f'* {CLASSES[idx]}: {probability}')
    except AttributeError:  # 'Compose' object has no attribute 'is_check_args'
        position = 'N/A'
        gender = 'N/A'
        probabilities = 'N/A'
        print(f'* Could not calculate position and gender')

    return position, gender, probabilities
