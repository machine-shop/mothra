from csv import writer
from pathlib import Path


def initialize_csv_file(csv_fname):
    """Sets up a CSV file to store the measurement results.

    Parameters
    ----------
    csv_fname : str or pathlib.Path
        The filename of the CSV file.

    Returns
    -------
    None
    """
    csv_fname = Path(csv_fname)
    # renaming csv file if it exists on disk already.
    csv_fname = _check_aux_file(csv_fname)

    # setting up the data columns that will be in the file.
    DATA_COLS = ['image_id',
                 'left_wing (mm)',
                 'right_wing (mm)',
                 'left_wing_center (mm)',
                 'right_wing_center (mm)',
                 'wing_span (mm)',
                 'wing_shoulder (mm)',
                 'position',
                 'gender',
                 'prob_upside_down',
                 'prob_female',
                 'prob_male']

    with open(csv_fname, 'w') as csv_file:
        write_to_file = writer(csv_file)
        write_to_file.writerow(DATA_COLS)
    return csv_fname


def write_csv_data(csv_file, image_name, dist_mm, position, gender,
                   probabilities):
    """Helper function. Writes data on the CSV input file."""
    write_to_file = writer(csv_file)

    # Separating probabilities into their own variables,
    # according to the order defined at the network
    prob_upside_down, prob_female, prob_male = probabilities

    write_to_file.writerow([image_name,
                            dist_mm["dist_l"],
                            dist_mm["dist_r"],
                            dist_mm["dist_l_center"],
                            dist_mm["dist_r_center"],
                            dist_mm["dist_span"],
                            dist_mm["dist_shoulder"],
                            position,
                            gender,
                            prob_upside_down,
                            prob_female,
                            prob_male])


def _check_aux_file(filename):
    """Helper function. Checks if filename exists; if yes, adds a number to
    it."""
    while filename.is_file():
        try:
            name, number = filename.stem.split('_')
            number = int(number) + 1
            filename = Path(f"{name}_{number}{filename.suffix}")
        except ValueError:
            filename = Path(f"{filename.stem}_1{filename.suffix}")
    return filename
