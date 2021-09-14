import numpy as np
import pytest

from mothra import binarization
from skimage import draw

# defining labels for classes in the processed images.
TAGS_LABEL = 1


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


def test_missing_tags(fake_lepid_no_tags):
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
