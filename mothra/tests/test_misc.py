from glob import glob
from mothra import misc
from skimage.io import imread


PATH_TEST_FILES = 'mothra/tests/test_files'
TEST_INPUT_FILE = f'{PATH_TEST_FILES}/input_file.txt'
TEST_INPUT_IMAGES = glob(f'{PATH_TEST_FILES}/test_input/*.JPG')
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
    image_90deg = misc.auto_rotate(image_90deg, TEST_IMAGE_90DEG)

    assert (image_0deg.all() == image_90deg.all())


def test_process_paths_in_input():
    """Checks if files from a folder within an input text file are correctly
    read.

    Summary
    -------
    We provide the path for a text file containing the path for a folder, and
    check if the filenames are read by pipeline._process_paths_in_input.

    Expected
    --------
    TEST_INPUT_IMAGES and image_paths should contain the same filenames.
    """
    image_paths = misc.process_paths_in_input(TEST_INPUT_FILE)

    assert image_paths.sort() == TEST_INPUT_IMAGES.sort()


def test_read_filenames_in_folder():
    """Check if filenames in folder are read properly.

    Summary
    -------
    We pass a folder containing a number of known input images, and
    check if misc._read_filenames_in_folders reads their filenames
    properly.

    Expected
    --------
    result_fnames and TEST_INPUT_IMAGES contain the same filenames.
    """
    test_folder = f'{PATH_TEST_FILES}/test_input/'
    result_fnames = misc._read_filenames_in_folder(test_folder)

    assert result_fnames.sort() == TEST_INPUT_IMAGES.sort()


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
    angle = misc.read_angle(TEST_IMAGE_90DEG)

    assert angle == 90


def test_read_paths_in_file():
    """Checks if files from a folder within an input text file are correctly
    read.

    Summary
    -------
    We provide the path for a text file containing the path for a folder, and
    check if the filenames are read by misc._read_paths_in_file.

    Expected
    --------
    TEST_INPUT_IMAGES and image_paths should contain the same filenames.
    """
    image_paths = misc._read_paths_in_file(TEST_INPUT_FILE)

    assert image_paths.sort() == TEST_INPUT_IMAGES.sort()
