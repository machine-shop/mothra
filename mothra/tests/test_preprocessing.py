from mothra import preprocessing
from skimage.io import imread

PATH_TEST_FILES = 'mothra/tests/test_files'
TEST_IMAGE_0DEG =  f'{PATH_TEST_FILES}/test_input/BMNHE_1105737_angle0.JPG'
TEST_IMAGE_90DEG =  f'{PATH_TEST_FILES}/test_input/BMNHE_1105737_angle90.JPG'


def test_auto_rotate():
    """Checks if tilted image is rotated correctly.

    Summary
    -------
    We pass a test image and a tilted version of it, and check if
    misc.auto_rotate fixes the orientation of the tilted one correctly.

    Expected
    --------
    image_0deg and image_90deg are equal.
    """
    image_0deg = imread(TEST_IMAGE_0DEG)
    image_90deg = imread(TEST_IMAGE_90DEG)
    image_90deg = preprocessing.auto_rotate(image_90deg, TEST_IMAGE_90DEG)

    assert (image_0deg.all() == image_90deg.all())


def test_read_angle():
    """Checks if orientation is extracted correctly from EXIF data.

    Summary
    -------
    We provide an input image with known angle and compare its angle
    read by `misc.read_angle`.

    Expected
    --------
    Orientation for the input image is 6, (right, top);
    misc.read_angle should return angle equals 90 deg.
    """
    angle = preprocessing.read_angle(TEST_IMAGE_90DEG)

    assert angle == 90
