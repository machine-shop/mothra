import subprocess
import os
import shutil
import pytest

import pipeline


TIMEOUT_TIME = 60


@pytest.mark.timeout(TIMEOUT_TIME)
def test_pipeline_main():

    test_input_dir = 'butterfly/tests/test_files/test_input/'
    test_output_dir = 'butterfly/tests/test_files/test_output/'
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
