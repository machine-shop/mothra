from pathlib import Path
from pooch import retrieve

import socket


# defining global constants.
URL_MODEL = {
    'id_gender' : 'https://gitlab.com/alexdesiqueira/mothra-models/-/raw/main/models/id_gender/id_gender.pkl',
    'id_position' : 'https://gitlab.com/alexdesiqueira/mothra-models/-/raw/main/models/id_position/id_position.pkl',
    'segmentation' : 'https://gitlab.com/alexdesiqueira/mothra-models/-/raw/main/models/segmentation/segmentation.pkl'
    }

URL_HASH = {
    'id_gender' : 'https://gitlab.com/alexdesiqueira/mothra-models/-/raw/main/models/id_gender/.SHA256SUM_ONLINE-id_gender',
    'id_position' : 'https://gitlab.com/alexdesiqueira/mothra-models/-/raw/main/models/id_position/.SHA256SUM_ONLINE-id_position',
    'segmentation' : 'https://gitlab.com/alexdesiqueira/mothra-models/-/raw/main/models/segmentation/.SHA256SUM_ONLINE-segmentation'
    }

LOCAL_HASH = {
    'id_gender' : Path('./models/SHA256SUM-id_gender'),
    'id_position' : Path('./models/SHA256SUM-id_position'),
    'segmentation' : Path('./models/SHA256SUM-segmentation')
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
    local_hash : pathlib.Path
        Path of the local hash file.
    """
    return (URL_MODEL.get(weights.stem), URL_HASH.get(weights.stem),
            LOCAL_HASH.get(weights.stem))


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
    _, url_hash, local_hash = _get_model_info(weights)
    # check if weights is in its folder. If not, download it.
    if not weights.is_file():
        print(f'{weights} not in the path. Downloading...')
        fetch_data(weights)
    # file exists: check if we have the last versioonlinen; download if not.
    else:
        if has_internet():
            local_hash_val = read_hash_local(filename=local_hash)
            url_hash_val = read_hash_from_url(path=local_hash.parent,
                                                 url_hash=url_hash)
            if local_hash_val != url_hash_val:
                print('New training data available. Downloading...')
                fetch_data(weights)

    return None


def download_hash_from_url(url_hash, filename):
    """Downloads hash from `url_hash`.

    Parameters
    ----------
    url_hash : str
        URL of the SHA256 hash.
    filename : str
        Filename to save the SHA256 hash locally.

    Returns
    -------
    None
    """
    retrieve(url=url_hash, known_hash=None, fname=filename, path='.')
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
    url_model, url_hash, local_hash = _get_model_info(weights)

    # creating filename to save url_hash.
    filename = local_hash.parent/Path(url_hash).name

    download_hash_from_url(url_hash=url_hash, filename=filename)
    local_hash_val = read_hash_local(local_hash)
    retrieve(url=url_model,
             known_hash=f'sha256:{local_hash_val}',
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


def read_hash_local(filename):
    """Reads local SHA256 hash file.

    Parameters
    ----------
    filename : pathlib.Path
        Path of the hash file.

    Returns
    -------
    local_hash : str
        SHA256 hash.

    Notes
    -----
    Returns None if file is not found.
    """
    try:
        with open(filename, 'r') as file_hash:
            hashes = [line for line in file_hash]
        # expecting only one hash, and not interested in the filename:
        local_hash, _ = hashes[0].split()
    except FileNotFoundError:
        local_hash = None
    return local_hash


def read_hash_from_url(path, url_hash):
    """Downloads and returns the SHA256 hash online for the file in `url_hash`.

    Parameters
    ----------
    path : str
        Where to look for the hash file.
    url_hash : str
        URL of the hash file for the latest model.

    Returns
    -------
    online_hash : str
        SHA256 hash for the file in `url_hash`.
    """
    filename = Path(url_hash).name
    latest_hash = Path(f'{path}/{filename}')

    download_hash_from_url(url_hash=url_hash, filename=filename)
    with open(latest_hash, 'r') as file_hash:
        hashes = [line for line in file_hash]

    # expecting only one hash, and not interested in the filename:
    online_hash, _ = hashes[0].split()

    return online_hash
