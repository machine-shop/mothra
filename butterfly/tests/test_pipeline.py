import subprocess
import os

import pytest


@pytest.mark.timeout(60)
def test_main():

    test_input_dir = "butterfly/tests/test_files/test_input/"
    test_output_dir = "butterfly/tests/test_files/test_output/"
    test_command = "python pipeline.py -p -i " + test_input_dir \
                    + " -o " + test_output_dir \
                    + " -s measurements -csv " + test_output_dir \
                    + "test_results.csv"

    # testing is done by calling the pipeline.py file with the test_command
    subprocess.call(test_command.split())

    files_in_test_output = os.listdir(test_output_dir)
    output_image, output_csv = "", ""

    for file in files_in_test_output:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            output_image = file
        if file.lower().endswith('.csv'):
            output_csv = files_in_test_output

    # some more testing functions can go here,
    # for example checking the contents

    # removing the output files from our pipeline run
    for file in files_in_test_output:
        os.remove(test_output_dir + file)

    # assert the two outputs exist
    assert(output_image and output_csv)
