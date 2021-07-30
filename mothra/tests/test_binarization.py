import numpy as np
import pytest

from mothra import binarization
from skimage import draw
from skimage.io import imread
from skimage.util import img_as_bool


@pytest.fixture(scope="module")
def fake_butterfly_layout():
    M, N = (300, 400)

    binary = np.zeros((M, N))
    binary[230:] = 1  # ruler
    binary[30:100, 250:380] = 1  # tag 1
    binary[120:200, 260:370] = 1  # tag 2

    # "butterfly"
    bfly_upper, bfly_lower = 50, 180
    bfly_left, bfly_right = 50, 220
    bfly_height = bfly_lower - bfly_upper
    bfly_width = bfly_right - bfly_left

    rr, cc = draw.polygon(
        [bfly_upper, bfly_upper, bfly_lower],
        [bfly_left, bfly_right, (bfly_left + bfly_right) / 2]
    )
    binary[rr, cc] = 1

    return binary, (bfly_height, bfly_width)


@pytest.fixture(scope="module")
def fake_butterfly_no_tags():
    M, N = (300, 400)

    binary = np.zeros((M, N))
    binary[230:] = 1  # ruler

    # "butterfly"
    bfly_upper, bfly_lower = 50, 180
    bfly_left, bfly_right = 50, 220
    bfly_height = bfly_lower - bfly_upper
    bfly_width = bfly_right - bfly_left

    rr, cc = draw.polygon(
        [bfly_upper, bfly_upper, bfly_lower],
        [bfly_left, bfly_right, (bfly_left + bfly_right) / 2]
    )
    binary[rr, cc] = 1

    return binary, (bfly_height, bfly_width)


def test_find_tags_edge(fake_butterfly_layout):
    butterfly, (rows, cols) = fake_butterfly_layout
    picture_2d = butterfly.astype(np.uint8)
    picture_3d = np.dstack((picture_2d,
                            1/2 * picture_2d,
                            1/4 * picture_2d))  # fake RGB image

    result = binarization.find_tags_edge(picture_3d, 230)
    print(result)
    assert (250 <= result <= 260)  # assert the tags edge is in a proper place


def test_missing_tags(fake_butterfly_no_tags):
    butterfly, (rows, cols) = fake_butterfly_no_tags
    picture_2d = butterfly.astype(np.uint8)
    picture_3d = np.dstack((picture_2d,
                            1/2 * picture_2d,
                            1/4 * picture_2d))  # fake RGB image

    result = binarization.find_tags_edge(picture_3d, 230)
    # such a crop will probably throw off measurement process
    # but the program won't crash
    assert (result >= 399)
