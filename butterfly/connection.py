from pathlib import Path
from pooch import retrieve

import socket


# defining global variables.
BASE_PATH = './butterfly/misc/'
MODEL_ONLINE = 'https://gitlab.com/alexdesiqueira/butterfly-wings-data-unet/-/raw/master/unet_model/unet_butterfly.pkl'
HASH_ONLINE = 'https://gitlab.com/alexdesiqueira/butterfly-wings-data-unet/-/raw/master/unet_model/SHA256SUM'


def download_hash_online(filename):
    """
    """
    retrieve(url=HASH_ONLINE, known_hash=None, fname=filename,
             path='.')
    return None


def fetch_data(filename='unet_butterfly.pkl'):
    """Downloads and checks the hash of unet_butterfly.pkl.

    Parameters
    ----------
    filename : str
        Filename to save the model file.
    """
    FILE_HASH = Path(f'{BASE_PATH}/SHA256SUM')

    download_hash_online(filename=FILE_HASH)
    LOCAL_HASH = read_hash_local()
    retrieve(url=MODEL_ONLINE, known_hash=f'sha256:{LOCAL_HASH}',
             fname=filename, path=BASE_PATH)

    return None


def has_internet():
    """Small script to check if PC is connected to the internet.
    """
    return socket.gethostbyname(socket.gethostname()) != '127.0.0.1'


def read_hash_local():
    """
    """
    HASH_LOCAL = Path(f'{BASE_PATH}/SHA256SUM')

    try:
        with open(HASH_LOCAL, 'r') as file_hash:
            hashes = [line for line in file_hash]
        # expecting only one hash, and not interested in the filename:
        local_hash, _ = hashes[0].split()
    except FileNotFoundError:
        local_hash = None
    return local_hash


def read_hash_online():
    """
    """
    HASH_ONLINE = Path(f'{BASE_PATH}/.SHA256SUM_online')

    download_hash_online(filename=HASH_ONLINE)
    with open(HASH_ONLINE, 'r') as file_hash:
        hashes = [line for line in file_hash]

    # expecting only one hash, and not interested in the filename:
    online_hash, _ = hashes[0].split()

    return online_hash
