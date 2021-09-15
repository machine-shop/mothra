import numpy as np
import pytest

from mothra import binarization
from skimage import draw
from skimage.io import imread
from skimage.util import img_as_bool


# required by fastai while predicting:
def label_func(image):
    """Function used to label images while training. Required by fastai."""
    return path/"labels"/f"{image.stem}{LABEL_EXT}"

# a dirty trick, so pytorch sees label_func
import __main__; __main__.label_func = label_func


# defining labels for classes in the processed images.
TAGS_LABEL = 1

# defining RGB test image.
IMAGE_RGB = './mothra/tests/test_files/test_input/BMNHE_500607.JPG'

# binarized images.
LEPID_SEG = './mothra/tests/test_files/test_input/BMNHE_500607-lepid.png.seg'
RULER_SEG = './mothra/tests/test_files/test_input/BMNHE_500607-ruler.png.seg'
TAGS_SEG = './mothra/tests/test_files/test_input/BMNHE_500607-tags.png.seg'

# prediction weights.
WEIGHTS_BIN = './models/segmentation_test-4classes.pkl'


@pytest.fixture(scope="module")
def fake_lepid_layout():
    """Implements a "fake lepidopteran" input image containing ruler and
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

    # creating "lepidopteran".
    lepid_upper, lepid_lower = 50, 180
    lepid_left, lepid_right = 50, 220
    lepid_height = lepid_lower - lepid_upper
    lepid_width = lepid_right - lepid_left

    rr, cc = draw.polygon(
        [lepid_upper, lepid_upper, lepid_lower],
        [lepid_left, lepid_right, (lepid_left + lepid_right) / 2]
    )
    img_classes[rr, cc] = 3

    return img_classes, (lepid_height, lepid_width)


@pytest.fixture(scope="module")
def fake_lepid_no_tags():
    """Implements a "fake lepidopteran" input image containing ruler, but no
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

    # creating "lepidopteran".
    lepid_upper, lepid_lower = 50, 180
    lepid_left, lepid_right = 50, 220
    lepid_height = lepid_lower - lepid_upper
    lepid_width = lepid_right - lepid_left

    rr, cc = draw.polygon(
        [lepid_upper, lepid_upper, lepid_lower],
        [lepid_left, lepid_right, (lepid_left + lepid_right) / 2]
    )
    img_classes[rr, cc] = 3

    return img_classes, (lepid_height, lepid_width)


def test_rescale_image(fake_lepid_layout):
    """Testing function binarization._rescale_image.

    Summary
    -------
    We decimate an input image, rescale it using binarization._rescale_image
    and compare their sizes.

    Expected
    --------
    The input image and its decimated/rescaled version should have the same
    size.
    """
    lepid, _ = fake_lepid_layout
    lepid_dec = lepid[::4]
    lepid_dec_rescaled = binarization._rescale_image(image_refer=lepid,
                                                    image_to_rescale=lepid_dec)

    assert (lepid_dec_rescaled.shape == lepid.shape)


def test_find_tags_edge(fake_lepid_layout):
    """Testing function binarization.find_tags_edge.

    Summary
    -------
    Since the current segmentation algorithm returns tags, ruler and
    lepidopteran, we obtain the tags from an input image and check if the
    coordinate generated from binarization.find_tags_edge is correct.

    Expected
    --------
    Edge of the tags, returned as an X coordinate, is in a proper place.
    """
    lepid, _ = fake_lepid_layout
    # returning a binary image containing only tags.
    lepid_tags = lepid * (lepid == TAGS_LABEL)

    result = binarization.find_tags_edge(tags_bin=lepid_tags, top_ruler=230)
    assert (250 <= result <= 260)


def test_find_tags_edge_missing_tags(fake_lepid_no_tags):
    """Testing function binarization.find_tags_edge.

    Summary
    -------
    We pass an input image without tags to binarization.find_tags_edge.

    Expected
    --------
    first_tag_edge, in this case, should be the last column of the input
    image. That could probably impair the measurement process, but the
    pipeline won't crash.
    """
    lepid, _ = fake_lepid_no_tags
    # returning a binary image containing no tags.
    lepid_no_tags = lepid * (lepid == TAGS_LABEL)
    print(lepid_no_tags, lepid_no_tags.shape)

    result = binarization.find_tags_edge(tags_bin=lepid_no_tags, top_ruler=230)

    assert (result >= 399)


def test_binarization():
    """Testing function binarization.binarization.

    Summary
    -------
    We pass a test image to binarization.binarization and check if its results
    are equal to the expected.


    Expected
    --------
    Expected tags, ruler and lepidopteran are equal to the ones returned by
    binarization.binarization.
    """
    lepid_rgb = imread(IMAGE_RGB)
    tags_result, ruler_result, lepid_result = binarization.binarization(
        image_rgb=lepid_rgb,
        weights=WEIGHTS_BIN)

    tags_expected = img_as_bool(imread(TAGS_SEG))

    assert (tags_expected.all() == tags_result.all())

    ruler_expected = img_as_bool(imread(RULER_SEG))

    assert (ruler_expected.all() == ruler_result.all())

    lepid_expected = img_as_bool(imread(LEPID_SEG))

    assert (lepid_expected.all() == lepid_result.all())


def test_return_bbox_largest_region(fake_lepid_layout):
    """Testing function binarization.return_bbox_largest_region.

    Summary
    -------
    We pass a binary input image with a region and compare the resulting
    bounding box from binarization.return_bbox_largest_region with the expected
    result.

    Expected
    --------
    Resulting and expected bounding boxes are equal.
    """
    lepid, _ = fake_lepid_layout
    lepid = (lepid == 3)  # getting only the lepidopteran

    bbox_result = binarization.return_bbox_largest_region(lepid)

    bbox_expected = (50, 50, 181, 221)

    assert bbox_result == bbox_expected


def test_return_largest_region(fake_lepid_layout):
    """Testing function binarization.return_largest_region.

    Summary
    -------
    We pass a binary input image with three regions and check if the resulting
    image from binarization.return_largest_region contains only the largest
    one.

    Expected
    --------
    Resulting image has only the largest region.
    """
    lepid, _ = fake_lepid_layout
    img_result = binarization.return_largest_region(lepid)

    img_expect = (lepid == 3)  # getting only the lepidopteran

    assert (img_expect.all() == img_result.all())
