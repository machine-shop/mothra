import numpy as np
import pytest

from mothra import binarization
from skimage import draw

# defining labels for classes in the processed images.
TAGS_LABEL = 1


@pytest.fixture(scope="module")
def fake_bfly_layout():
    """Implements a "fake butterfly" input image containing ruler and
    identification tags, that mimics the others on the actual lepidopteran
    datasets.

    Notes
    -----
    Starting from mothra 1.0, the pipeline processing images will return
    labels corresponding to the classes of the elements in the input image,
    instead of a binary image.
    These classes are 0 (background), 1 (tags), 2 (ruler), 3 (lepidopteran).
    """
    M, N = (300, 400)
    img_classes = np.zeros((M, N))

    # creating tags.
    img_classes[30:100, 250:380] = 1  # tag 1
    img_classes[120:200, 260:370] = 1  # tag 2

    # creating ruler.
    img_classes[230:] = 2

    # creating "butterfly".
    bfly_upper, bfly_lower = 50, 180
    bfly_left, bfly_right = 50, 220
    bfly_height = bfly_lower - bfly_upper
    bfly_width = bfly_right - bfly_left

    rr, cc = draw.polygon(
        [bfly_upper, bfly_upper, bfly_lower],
        [bfly_left, bfly_right, (bfly_left + bfly_right) / 2]
    )
    img_classes[rr, cc] = 3

    return img_classes, (bfly_height, bfly_width)


@pytest.fixture(scope="module")
def fake_bfly_no_tags():
    """Implements a "fake butterfly" input image containing ruler, but no
    tags, that mimics the others on the actual lepidopteran datasets.

    Notes
    -----
    Starting from mothra 1.0, the pipeline processing images will return
    labels corresponding to the classes of the elements in the input image,
    instead of a binary image.
    These classes are 0 (background), 1 (tags), 2 (ruler), 3 (lepidopteran).
    """
    M, N = (300, 400)
    img_classes = np.zeros((M, N))

    # creating ruler.
    img_classes[230:] = 2

    # creating "butterfly".
    bfly_upper, bfly_lower = 50, 180
    bfly_left, bfly_right = 50, 220
    bfly_height = bfly_lower - bfly_upper
    bfly_width = bfly_right - bfly_left

    rr, cc = draw.polygon(
        [bfly_upper, bfly_upper, bfly_lower],
        [bfly_left, bfly_right, (bfly_left + bfly_right) / 2]
    )
    img_classes[rr, cc] = 3

    return img_classes, (bfly_height, bfly_width)


def test_rescale_image(fake_bfly_layout):
    """Testing function binarization.binarization.

    Summary
    -------
    We decimate an input image, rescale it using binarization.binarization
    and compare their sizes.

    Expected
    --------
    The input image and it's decimated/rescaled version should have the same
    size.
    """
    bfly, _ = fake_bfly_layout
    bfly_dec = bfly[::4]
    bfly_dec_rescaled = binarization._rescale_image(image_refer=bfly,
                                                    image_to_rescale=bfly_dec)

    assert (bfly_dec_rescaled.shape == bfly.shape)


def test_find_tags_edge(fake_bfly_layout):
    """Testing function binarization.binarization.

    Summary
    -------
    Since the current segmentation algorithm returns tags, ruler and
    lepidopteran, we obtain the tags from an input image and check if the
    coordinate generated from binarization.find_tags_edge is correct.

    Expected
    --------
    Edge of the tags, returned as an X coordinate, is in a proper place.
    """
    bfly, _ = fake_bfly_layout
    # returning a binary image containing only tags.
    bfly_tags = bfly * (bfly == TAGS_LABEL)

    result = binarization.find_tags_edge(tags_bin=bfly_tags, top_ruler=230)
    assert (250 <= result <= 260)


def test_missing_tags(fake_bfly_no_tags):
    """Testing function binarization.binarization.

    Summary
    -------
    We pass an input image without tags to binarization.find_tags_edge.

    Expected
    --------
    first_tag_edge, in this case, should be the last column of the input
    image. That could probably impair the measurement process, but the
    pipeline won't crash.
    """
    bfly, _ = fake_bfly_no_tags
    # returning a binary image containing no tags.
    bfly_no_tags = bfly * (bfly == TAGS_LABEL)
    print(bfly_no_tags, bfly_no_tags.shape)

    result = binarization.find_tags_edge(tags_bin=bfly_no_tags, top_ruler=230)

    assert (result >= 399)


def test_return_largest_region():
    """Testing function binarization.return_largest_region.

    Summary
    -------
    We pass a binary input image with three regions and check if the result
    image from binarization.return_largest_region contains only the largest
    one.

    Expected
    --------
    Resulting image has only the largest region.
    """
    img_test = np.asarray([[0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
                           [0, 1, 1, 1, 1, 0, 0, 0, 1, 1],
                           [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
                           [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                           [0, 1, 1, 1, 1, 0, 0, 1, 1, 0],
                           [0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                           [1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                           [0, 1, 1, 1, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='bool')

    img_expect = np.asarray([[0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                             [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                             [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                             [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                             [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                             [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='bool')

    img_result = binarization.return_largest_region(img_test)

    assert (img_expect.all() == img_result.all())