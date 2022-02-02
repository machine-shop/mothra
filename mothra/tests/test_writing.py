from csv import reader
from mothra import writing
from pathlib import Path


PATH_TEST_FILES = 'mothra/tests/test_files'
TEST_CSV_FILE = f'{PATH_TEST_FILES}/test_file.csv'


def test_initialize_csv_file():
    """Checks if CSV file is initialized properly, and if its filename
    is correct.

    Summary
    -------
    We pass a filename to writing.initialize_csv_file, check if the
    file is created in disk, and if the name is the same as expected.

    Expected
    --------
    result_csv is a file on disk, and its name is equal to expected_csv.
    """
    csv_fname = TEST_CSV_FILE
    result_csv = writing.initialize_csv_file(csv_fname)

    assert result_csv.is_file()

    # deleting file, else next round of tests will have a different filename
    result_csv.unlink()
    expected_csv = Path(TEST_CSV_FILE)

    assert expected_csv == result_csv


def test_write_csv_data():
    """ Check if data is written properly in the CSV file.

    Summary
    -------
    We create a CSV file with test data and check if the
    function writing._write_csv_data writes the data
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
    probabilities = ['test_prob_ud', 'test_prob_f', 'test_prob_m']

    expected_lines = [
        ['image_id', 'left_wing (mm)', 'right_wing (mm)',
         'left_wing_center (mm)', 'right_wing_center (mm)',
         'wing_span (mm)', 'wing_shoulder (mm)', 'position',
         'gender', 'prob_upside_down', 'prob_female', 'prob_male'],
        ['test_image', 'test_l', 'test_r', 'test_dcl', 'test_dcr',
         'test_span', 'test_shoulder', 'test_pos', 'test_gender',
         'test_prob_ud', 'test_prob_f', 'test_prob_m']
        ]

    result_csv = writing.initialize_csv_file(csv_fname)
    with open(csv_fname, 'a') as csv:
        writing.write_csv_data(csv, image_name, dist_mm, position, gender,
                               probabilities)

    with open(csv_fname, newline='') as csv:
        for idx, row in enumerate(reader(csv)):
            print(f'* row: {row}')
            assert expected_lines[idx] == row

    # deleting file, else next round of tests will have more rows
    result_csv.unlink()


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
    "result.csv.test" already exists in disk; writing._check_aux_file
    should return "result.csv_1.test".
    """
    expected_fname = Path('result.csv_1.test')

    filename = Path(f'{PATH_TEST_FILES}/result.csv.test')
    result_fname = writing._check_aux_file(filename)

    assert result_fname == expected_fname
