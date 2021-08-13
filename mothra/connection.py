from pooch import retrieve
from urllib import request

import hashlib
import socket


# defining global constants.
URL_MODEL = {
    'id_gender_test-3classes' : 'https://gitlab.com/mothra/mothra-data/-/raw/main/models/id_gender/id_gender_test-3classes.pkl',
    'segmentation_test-4classes' : 'https://gitlab.com/mothra/mothra-data/-/raw/main/models/segmentation/segmentation_test-4classes.pkl'
    }

URL_HASH = {
    'id_gender_test-3classes' : 'https://gitlab.com/mothra/mothra-data/-/raw/main/models/id_gender/SHA256SUM-id_gender_test-3classes',
    'segmentation_test-4classes' : 'https://gitlab.com/mothra/mothra-data/-/raw/main/models/segmentation/SHA256SUM-segmentation_test-4classes'
    }


def _get_model_info(weights):
    """Helper function. Returns info from the model according the filename of
    its weights.

    Parameters
    ----------
    weights : pathlib.Path
        Path of the file containing weights.

    Returns
    -------
    url_model : str
        URL of the file for the latest model.
    url_hash : str
        URL of the hash file for the latest model.
    """
    return (URL_MODEL.get(weights.stem), URL_HASH.get(weights.stem))


def download_weights(weights):
    """Triggers functions to download weights.

    Parameters
    ----------
    weights : str or pathlib.Path
        Path of the file containing weights.

    Returns
    -------
    None
    """
    _, url_hash = _get_model_info(weights)

    # check if weights is in its folder. If not, download the file.
    if not weights.is_file():
        print(f'{weights} not in the path. Downloading...')
        fetch_data(weights)
    # file exists: check if we have the last version; download if not.
    else:
        if has_internet():
            local_hash_val = read_hash_local(weights)
            url_hash_val = read_hash_from_url(url_hash)
            if local_hash_val != url_hash_val:
                print('New training data available. Downloading...')
                fetch_data(weights)

    return None


def fetch_data(weights):
    """Downloads and checks the hash of `weights`, according to its filename.

    Parameters
    ----------
    weights : str
        Weights containing the model file.

    Returns
    -------
    None
    """
    url_model, url_hash = _get_model_info(weights)

    url_hash_val = read_hash_from_url(url_hash)
    retrieve(url=url_model,
             known_hash=f'sha256:{url_hash_val}',
             fname=weights,
             path='.')

    return None


def has_internet():
    """Small script to check if PC is connected to the internet.

    Parameters
    ----------
    None

    Returns
    -------
    True if connected to the internet; False otherwise.
    """
    return socket.gethostbyname(socket.gethostname()) != '127.0.0.1'


def read_hash_local(weights):
    """Reads local SHA256 hash from weights.

    Parameters
    ----------
    weights : str or pathlib.Path
        Path of the file containing weights.

    Returns
    -------
    local_hash : str or None
        SHA256 hash of weights file.

    Notes
    -----
    Returns None if file is not found.
    """
    BUFFER_SIZE = 65536
    sha256 = hashlib.sha256()

    try:
        with open(weights, 'rb') as file_weights:
            while True:
                data = file_weights.read(BUFFER_SIZE)
                if not data:
                    break
                sha256.update(data)
        local_hash = sha256.hexdigest()
    except FileNotFoundError:
        local_hash = None
    return local_hash


def read_hash_from_url(url_hash):
    """Returns the SHA256 hash online for the file in `url_hash`.

    Parameters
    ----------
    url_hash : str
        URL of the hash file for the latest model.

    Returns
    -------
    online_hash : str
        SHA256 hash for the file in `url_hash`.
    """
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    headers = {'User-Agent':user_agent,}

    aux_req = request.Request(url_hash, None, headers)
    response = request.urlopen(aux_req)
    hashes = response.read()

    # expecting only one hash, and not interested in the filename:
    online_hash, _ = hashes.decode('ascii').split()

    return online_hash
