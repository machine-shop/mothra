from pathlib import Path
from pooch import retrieve

import socket


# defining global variables.
MODEL_ONLINE = 'https://gitlab.com/alexdesiqueira/butterfly-wings-data-unet/-/raw/master/unet_model/unet_butterfly.pkl'
HASH_ONLINE = 'https://gitlab.com/alexdesiqueira/butterfly-wings-data-unet/-/raw/master/unet_model/SHA256SUM'


def download_hash_online(url_hash=HASH_ONLINE, filename='SHA256SUM'):
    """Downloads hash from `url_hash`.

    Parameters
    ----------
    url_hash : str
        URL of the SHA256 hash.
    filename : str
        Filename to save the SHA256 hash.

    Returns
    -------
    None
    """
    retrieve(url=url_hash, known_hash=None, fname=filename, path='.')
    return None


def fetch_data(path='./models/', filename='unet_butterfly.pkl'):
    """Downloads and checks the hash of `filename`.

    Parameters
    ----------
    filename : str
        Filename to save the model file.

    Returns
    -------
    None
    """
    FILE_HASH = Path(f'{path}/SHA256SUM')

    download_hash_online(filename=FILE_HASH)
    LOCAL_HASH = read_hash_local()
    retrieve(url=MODEL_ONLINE, known_hash=f'sha256:{LOCAL_HASH}',
             fname=filename, path='.')

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


def read_hash_local(path='./models/', filename='SHA256SUM'):
    """Reads local SHA256 hash file.

    Parameters
    ----------
    path : str
        Where to look for the hash file.
    filename : str
        Filename of the hash file.

    Returns
    -------
    local_hash : str
        SHA256 hash.

    Notes
    -----
    Returns None if file is not found.
    """
    HASH_LOCAL = Path(f'{path}/{filename}')

    try:
        with open(HASH_LOCAL, 'r') as file_hash:
            hashes = [line for line in file_hash]
        # expecting only one hash, and not interested in the filename:
        local_hash, _ = hashes[0].split()
    except FileNotFoundError:
        local_hash = None
    return local_hash


def read_hash_online(path='./models/', filename='.SHA256SUM_online'):
    """Downloads and returns the SHA256 hash online for `filename`.

    Parameters
    ----------
    path : str
        Where to look for the hash file.
    filename : str
        Filename of the hash file.

    Returns
    -------
    online_hash : str
    """
    HASH_ONLINE = Path(f'{path}/{filename}')

    download_hash_online(filename=HASH_ONLINE)
    with open(HASH_ONLINE, 'r') as file_hash:
        hashes = [line for line in file_hash]

    # expecting only one hash, and not interested in the filename:
    online_hash, _ = hashes[0].split()

    return online_hash
