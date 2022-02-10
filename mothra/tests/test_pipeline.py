import subprocess
import os
import shutil
import pytest

from glob import glob


PATH_TEST_FILES = 'mothra/tests/test_files'
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
