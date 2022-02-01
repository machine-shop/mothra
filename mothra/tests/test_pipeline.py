import subprocess
import os
import shutil
import pytest

import pipeline

from csv import reader
from glob import glob
from pathlib import Path
from skimage.io import imread
from skimage.util import img_as_float


PATH_TEST_FILES = 'mothra/tests/test_files'
TEST_CSV_FILE = f'{PATH_TEST_FILES}/test_file.csv'
TEST_INPUT_FILE = f'{PATH_TEST_FILES}/input_file.txt'
TEST_INPUT_IMAGES = glob(f'{PATH_TEST_FILES}/test_input/*.JPG')
TEST_TILTED_IMAGE =  f'{PATH_TEST_FILES}/test_input/BMNHE_1105737_17193_6eec94847b4939c6d117429d59829aac7a9fadf9.JPG'
TIMEOUT_TIME = 180


@pytest.mark.timeout(TIMEOUT_TIME)
def test_pipeline_main():

    test_input_dir = f'{PATH_TEST_FILES}/test_input/'
    test_output_dir = f'{PATH_TEST_FILES}/test_output/'
    test_command = [
        'python', 'pipeline.py', '-p',
        '-i', test_input_dir,
        '-o', test_output_dir,
        '-csv', test_output_dir + 'test_results.csv'
    ]

    # testing is done by calling the pipeline.py file with the test_command
    subprocess.check_call(test_command)

    files_in_test_output = os.listdir(test_output_dir)
    output_image, output_csv = '', ''

    for f in files_in_test_output:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            output_image = f
        if f.lower().endswith('.csv'):
            output_csv = files_in_test_output

    # some more testing functions can go here,
    # for example checking the contents

    # removing the test output directory from our pipeline run
    shutil.rmtree(test_output_dir)

    # assert the two outputs exist
    assert(output_image and output_csv)


def test_create_layout():
    """Checks if axes for regular and detailed plots were created
    properly.

    Summary
    -------
    We pass different stages and plot levels to pipeline.create_layout,
    and check the resulting axes.

    Expected
    --------
    - pipeline.create_layout(1, 0) should not return axes.
    - pipeline.create_layout(3, 2) should return all seven axes
    (ax_main, ax_bin, ax_poi, ax_structure, ax_signal, ax_fourier, ax_tags).
    - pipeline.create_layout(3, 1) should return a list with three axes and
    four None.
    """
    axes = pipeline.create_layout(1, 0)
    assert axes is None
    axes = pipeline.create_layout(3, 2)
    for ax in axes:
        assert ax
    axes = pipeline.create_layout(3, 1)
    for ax in axes[:3]:
        assert ax
    for ax in axes[3:]:
        assert ax is None


def test_initialize_csv_file():
    """Checks if CSV file is initialized properly, and if its filename
    is correct.

    Summary
    -------
    We pass a filename to pipeline.initialize_csv_file, check if the
    file is created in disk, and if the name is the same as expected.

    Expected
    --------
    result_csv is a file on disk, and its name is equal to expected_csv.
    """
    csv_fname = TEST_CSV_FILE
    result_csv = pipeline.initialize_csv_file(csv_fname)

    assert result_csv.is_file()

    # deleting file, else next round of tests will have a different filename
    result_csv.unlink()
    expected_csv = Path(TEST_CSV_FILE)

    assert expected_csv == result_csv


def test_read_orientation():
    """Checks if orientation is extracted correctly from EXIF data.

    Summary
    -------
    We provide an input image with known angle and compare its angle
    read by pipeline.read_orientation.

    Expected
    --------
    Orientation for the input image is 6, (right, top);
    pipeline.read_orientation should return angle equals 90 deg.
    """
    angle = pipeline.read_orientation(TEST_TILTED_IMAGE)

    assert angle == 90


def test_check_aux_file():
    """Checks if filename is updated correctly if file already exists
    in disk.

    Summary
    -------
    We provide the path of a file that exists in disk, and check
    if result_fname was updated correctly (a number was added to
    the end of the filename).

    Expected
    --------
    "result.csv.test" already exists in disk; pipeline._check_aux_file
    should return "result.csv_1.test".
    """
    expected_fname = Path('result.csv_1.test')

    filename = Path(f'{PATH_TEST_FILES}/result.csv.test')
    result_fname = pipeline._check_aux_file(filename)

    assert result_fname == expected_fname


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
    image_paths = pipeline._process_paths_in_input(TEST_INPUT_FILE)

    assert image_paths.sort() == TEST_INPUT_IMAGES.sort()


def test_read_paths_in_file():
    """Checks if files from a folder within an input text file are correctly
    read.

    Summary
    -------
    We provide the path for a text file containing the path for a folder, and
    check if the filenames are read by pipeline._read_paths_in_file.

    Expected
    --------
    TEST_INPUT_IMAGES and image_paths should contain the same filenames.
    """
    image_paths = pipeline._read_paths_in_file(TEST_INPUT_FILE)

    assert image_paths.sort() == TEST_INPUT_IMAGES.sort()


def test_write_csv_data():
    """ Check if data is written properly in the CSV file.

    Summary
    -------
    We create a CSV file with test data and check if the
    function pipeline._write_csv_data writes the data
    properly to the file.

    Expected
    --------
    Rows in expected_lines and csv_fname are equal.
    """
    csv_fname = f'{PATH_TEST_FILES}/test.csv'
    image_name = 'test_image'
    dist_mm = {
        "dist_l": 'test_l',
        "dist_r": 'test_r',
        "dist_l_center": 'test_dcl',
        "dist_r_center": 'test_dcr',
        "dist_span": 'test_span',
        "dist_shoulder": 'test_shoulder'
    }
    position = 'test_pos'
    gender = 'test_gender'
    probabilities = '['testprob_1', 'testprob_2', 'testprob_3']'

    expected_lines = [
        ['image_id', 'left_wing (mm)', 'right_wing (mm)',
         'left_wing_center (mm)', 'right_wing_center (mm)',
         'wing_span (mm)', 'wing_shoulder (mm)', 'position',
         'gender', 'probabilities'],
        ['test_image', 'test_l', 'test_r', 'test_dcl', 'test_dcr',
        'test_span', 'test_shoulder', 'test_pos', 'test_gender',
        ['testprob_1', 'testprob_2', 'testprob_3']]
        ]

    result_csv = pipeline.initialize_csv_file(csv_fname)
    with open(csv_fname, 'a') as csv:
        pipeline._write_csv_data(csv, image_name, dist_mm, position, gender,
                                 probabilities)

    with open(csv_fname, newline='') as csv:
        for idx, row in enumerate(reader(csv)):
            assert expected_lines[idx] == row

    # deleting file, else next round of tests will have more rows
    result_csv.unlink()


def test_read_filenames_in_folder():
    """Check if filenames in folder are read properly.

    Summary
    -------
    We pass a folder containing a number of known input images, and
    check if pipeline._read_filenames_in_folders reads their filenames
    properly.

    Expected
    --------
    result_fnames and TEST_INPUT_IMAGES contain the same filenames.
    """
    test_folder = f'{PATH_TEST_FILES}/test_input/'
    result_fnames = pipeline._read_filenames_in_folder(test_folder)

    assert result_fnames.sort() == TEST_INPUT_IMAGES.sort()
