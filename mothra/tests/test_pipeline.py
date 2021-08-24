import subprocess
import os
import shutil
import pytest

import pipeline

from pathlib import Path
from skimage.io import imread
from skimage.util import img_as_float


TIMEOUT_TIME = 180


@pytest.mark.timeout(TIMEOUT_TIME)
def test_pipeline_main():

    test_input_dir = 'mothra/tests/test_files/test_input/'
    test_output_dir = 'mothra/tests/test_files/test_output/'
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


def test_read_orientation():
    """Checks if orientation is extracted correctly from EXIF data.

    Summary
    -------
    We provide an input image with known angle and compare its angle
    read by pipeline.read_orientation().

    Expected
    --------
    Orientation for the input image is 6, (right, top);
    pipeline.read_orientation() should return angle equals 90 deg.
    """
    tilted_path = 'mothra/tests/test_files/test_input/BMNHE_1105737_17193_6eec94847b4939c6d117429d59829aac7a9fadf9.JPG'
    angle = pipeline.read_orientation(tilted_path)

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
    "result.csv.test" already exists in disk; pipeline._check_aux_file()
    should return "result.csv_1.test".
    """
    expected_fname = Path('result.csv_1.test')

    filename = Path('mothra/tests/test_files/result.csv.test')
    result_fname = pipeline._check_aux_file(filename)

    assert result_fname == expected_fname
