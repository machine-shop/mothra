#!/bin/env python

import os
import argparse
import csv
from butterfly import (ruler_detection, tracing, measurement, binarization,
                       identification)
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import rotate
from exif import Image
from pathlib import Path

WSPACE_SUBPLOTS = 0.7
SUPPORTED_IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.tiff', '.tif')
SUPPORTED_TEXT_EXT = ('.txt', '.text')

"""
Example :
    $ python pipeline_argparse.py --stage tracing --plot -dpi 400
"""


def create_layout(n_stages, plot_level):
    """Creates Axes to plot figures

    Parameters
    ----------
    n_stages : int
        length of pipeline process
    plot_level : int
        0 : no plotting
        1 : regular plots
        2 : detailed plots

    Returns
    -------
    axes : list of Axes
    """
    if plot_level == 0:
        return None

    elif plot_level == 1:
        ncols = n_stages
        fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(12, 5))
        if n_stages == 1:
            ax = [ax]
        ax_list = []
        for ax in ax:
            ax_list.append(ax)
        return ax_list + [None] * (7 - n_stages)

    elif plot_level == 2:
        shape = (3, 3)
        ax_main = plt.subplot2grid(shape, (0, 0))
        ax_structure = plt.subplot2grid(shape, (0, 1))
        ax_signal = plt.subplot2grid(shape, (1, 0), colspan=2)
        ax_fourier = plt.subplot2grid(shape, (2, 0), colspan=2)

        ax_tags = plt.subplot2grid(shape, (0, 2))
        ax_bin = plt.subplot2grid(shape, (1, 2))
        ax_poi = plt.subplot2grid(shape, (2, 2))
        plt.tight_layout()
        if n_stages == 1:
            return [ax_main, None, None, ax_structure, ax_signal, ax_fourier,
                    None]
        elif n_stages == 2:
            return [ax_main, ax_bin, None, ax_structure, ax_signal, ax_fourier,
                    ax_tags]
        elif n_stages == 3:
            return [ax_main, ax_bin, ax_poi, ax_structure, ax_signal,
                    ax_fourier, ax_tags]


def read_orientation(image_path):
    """Read orientation from image on path, according to EXIF data.

    Parameters
    ----------
    image_path : str
        Path of the input image.

    Returns
    -------
    angle : int or None
        Current orientation of the image in degrees, or None if EXIF data
        cannot be read.
    """
    metadata = Image(image_path)

    try:
        if metadata.has_exif:
            orientation = metadata.orientation.value
            # checking possible orientations for images.
            angles = {1: 0,  # (top, left)
                      6: 90,  # (right, top)
                      3: 180,  # (bottom, right)
                      8: 270}  # (left, bottom)
            return angles.get(orientation, 0)
        else:
            print(f'Cannot evaluate orientation for {image_path}.')
            return None
    except ValueError:  # ... is not a valid TiffByteOrder
        print(f'Cannot evaluate orientation for {image_path}.')
        return None


def separate_binary_elements(img_binary):
    """Separate the elements constutient of the binary input image.

    Parameters
    ----------
    img_binary : (M, N) ndarray
        Binary input image.

    Returns
    -------
    bin_lepidoptera : (M, N) ndarray
        Image contaning the Lepidoptera.
    bin_ruler : (M, N) ndarray
        Image contaning the ruler.
    bin_tags : (M, N) ndarray
        Image contaning the tags.
    """

    return bin_lepidoptera, bin_ruler, bin_tags


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


def _process_paths_in_input(input_name):
    """Helper function. Process the input argument and returns the images
    in path."""
    try:
        if os.path.isfile(input_name):
            # if input is a text file, reads paths listed in it.
            if input_name.lower().endswith(SUPPORTED_TEXT_EXT):
                image_paths = _read_paths_in_file(input_name)
            # if input is an image, add it to image_paths.
            elif input_name.lower().endswith(SUPPORTED_IMAGE_EXT):
                image_paths = [input_name]
        elif(os.path.isdir(input_name)):
            image_paths = _read_filenames_in_folder(input_name)
    except:
        print(f"Type of input not understood. Please enter path for single\
                image, folder or text file containing paths.")
        raise
    return image_paths


def _read_paths_in_file(input_name):
    """Helper function. Reads image paths in input file."""
    image_paths = []
    with open(input_name) as txt_file:
        for item in txt_file:
            try:
                item = item.strip()
                if os.path.isdir(item):
                    aux_paths = _read_filenames_in_folder(item)
                elif os.path.isfile(item) and item.lower().endswith(SUPPORTED_IMAGE_EXT):
                    aux_paths = [item]
            except FileNotFoundError:
                continue
            image_paths.extend(aux_paths)

    return list(set(image_paths))  # remove duplicated entries from list


def _read_filenames_in_folder(folder):
    """Helper function. Reads filenames in folder and appends them into a
    list."""
    image_paths = []
    for path, _, items in os.walk(folder):
        for item in items:
            item = os.path.join(path, item)
            if not item.lower().endswith(SUPPORTED_IMAGE_EXT):
                continue
            image_paths.append(item)

    return image_paths


def main():
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description='Script to automate butterfly wings measurment')
    # Add arguments
    # Plotting
    parser.add_argument('-p', '--plot',
                        action='store_true',
                        help='If entered images are plotted to the output\
                        folder')

    parser.add_argument('-pp', '--detailed_plot',
                        action='store_true',
                        help='If entered detailed images are plotted to the\
                        output folder')
    # Input path
    parser.add_argument('-i', '--input',
                        type=str,
                        help='Input path for single image, folder or text\
                        file (extension txt) containing paths',
                        required=False,
                        default='raw_images')
    # Output path
    parser.add_argument('-o', '--output_folder',
                        type=str,
                        help='Output path for raw image',
                        required=False,
                        default='outputs')
    # Stage
    parser.add_argument('-s', '--stage',
                        type=str,
                        help="Stage name: 'ruler_detection', 'binarization',\
                        'measurements'",
                        required=False,
                        default='measurements')
    # Dots per inch
    parser.add_argument('-dpi',
                        type=int,
                        help='Dots per inch of the saved figures',
                        default=300)

    # CSV output path
    parser.add_argument('-csv', '--path_csv',
                        type=str,
                        help='Path of the resulting csv file',
                        default='results.csv')

    # U-nets
    parser.add_argument('-u', '--unet',
                        action='store_true',
                        help='Use U-nets in binarization step')
    args = parser.parse_args()

    # Initialization
    if os.path.exists(args.output_folder):
        oldList = os.listdir(args.output_folder)
        for oldFile in oldList:
            os.remove(args.output_folder+"/"+oldFile)
    else:
        os.mkdir(args.output_folder)

    stages = ['ruler_detection', 'binarization', 'measurements']

    if args.stage not in stages:
        print("ERROR : Stage can only be 'ruler_detection', 'binarization',\
               or 'measurements'")
        return 0

    plot_level = 0
    if args.plot:
        plot_level = 1
    if args.detailed_plot:
        plot_level = 2

    # Initializing the csv file
    if args.stage == 'measurements':
        # renaming csv file if it exists on disk already.
        csv_fname = Path(args.path_csv)
        csv_fname = _check_aux_file(csv_fname)

        with open(csv_fname, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['image_id', 'left_wing (mm)', 'right_wing (mm)',
                'left_wing_center (mm)', 'right_wing_center (mm)', 'wing_span (mm)',
                'wing_shoulder (mm)', 'position', 'gender'])

    stage_idx = stages.index(args.stage)
    pipeline_process = stages[:stage_idx + 1]

    # reading and processing input path.
    input_name = args.input
    image_paths = _process_paths_in_input(input_name)

    n = len(image_paths)

    for i, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path)
        print(f'Image {i+1}/{n} : {image_name}')

        image_rgb = imread(image_path)

        # check image orientation and untilt it, if necessary.
        angle = read_orientation(image_path)
        if angle not in (None, 0):  # angle == 0 does not need untilting
            image_rgb = rotate(image_rgb, angle=angle, resize=True)

        axes = create_layout(len(pipeline_process), plot_level)

        # binarizing ruler, tags and Lepidoptera.
        img_binary = binarization.main(image_rgb, top_ruler, args.unet, axes)

        # separating elements in the input image.
        bin_lepidoptera, bin_ruler, bin_tags = _sep_binary_elements(img_binary)

        for step in pipeline_process:
            if step == 'ruler_detection':
                T_space, top_ruler = ruler_detection.main(image_rgb, bin_ruler, axes)

            # elif step == 'binarization':


            elif step == 'measurements':
                points_interest = tracing.main(img_binary, axes)
                dist_pix, dist_mm = measurement.main(points_interest, T_space,
                                                     axes)
                # measuring position and gender
                position, gender = identification.main(image_rgb, top_ruler)

                with open(csv_fname, 'a') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([image_name,
                                     dist_mm["dist_l"],
                                     dist_mm["dist_r"],
                                     dist_mm["dist_l_center"],
                                     dist_mm["dist_r_center"],
                                     dist_mm["dist_span"],
                                     dist_mm["dist_shoulder"],
                                     position,
                                     gender])

        if plot_level > 0:
            output_path = os.path.normpath(os.path.join(args.output_folder, image_name))
            dpi = args.dpi
            if plot_level == 2:
                dpi = int(1.5 * args.dpi)
            plt.savefig(output_path, dpi=dpi)
            plt.close()


if __name__ == "__main__":
    main()
