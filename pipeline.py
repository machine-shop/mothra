#!/bin/env python

import numpy as np
import os
import argparse
import csv
from mothra import (ruler_detection, tracing, measurement, binarization,
                    identification)
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import rotate
from skimage.util import img_as_ubyte
from exif import Image
from pathlib import Path
from fastai.vision.augment import RandTransform
from fastai.vision.core import PILImage
from fastcore.basics import store_attr

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
                 'gender']

    with open(csv_fname, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(DATA_COLS)
    return csv_fname


# required by fastai while predicting:
class AlbumentationsTransform(RandTransform):
    """A handler for multiple transforms from the package `albumentations`.
    Required by fastai."""
    split_idx,order=None,2
    def __init__(self, train_aug, valid_aug): store_attr()

    def before_call(self, b, split_idx):
        self.idx = split_idx

    def encodes(self, img: PILImage):
        if self.idx == 0:
            aug_img = self.train_aug(image=np.array(img))['image']
        else:
            aug_img = self.valid_aug(image=np.array(img))['image']
        return PILImage.create(aug_img)


# required by fastai while predicting:
def label_func(image):
    """Function used to label images while training. Required by fastai."""
    return path/"labels"/f"{image.stem}{LABEL_EXT}"


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
    image_paths = []
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
    image_paths, aux_paths = [], []
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


def _write_csv_data(csv_file, image_name, dist_mm, position, gender):
    """Helper function. Writes data on the CSV input file."""
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
    return None


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
        description='This is mothra, a software to automate different\
        measurements on images of Lepidopterae.')
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
                        default='input_images')

    # Output path
    parser.add_argument('-o', '--output_folder',
                        type=str,
                        help='Output path for raw image',
                        required=False,
                        default='outputs')
    # Stage
    parser.add_argument('-s', '--stage',
                        type=str,
                        help="Stage name: 'binarization', 'ruler_detection',\
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
                        default='outputs/results.csv')

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
        print((f"* mothra expects stage to be 'ruler_detection', "
               f"binarization', or 'measurements'. Received '{args.stage}'"))
        return None

    plot_level = 0
    if args.plot:
        plot_level = 1
    if args.detailed_plot:
        plot_level = 2

    # initializing the csv file.
    if args.stage == 'measurements':
        initialize_csv_file(csv_fname=args.path_csv)

    stage_idx = stages.index(args.stage)
    pipeline_process = stages[:stage_idx + 1]

    # reading and processing input path.
    input_name = args.input
    image_paths = _process_paths_in_input(input_name)

    number_of_images = len(image_paths)

    for i, image_path in enumerate(image_paths):
        try:
            # creating axes layout for plotting.
            axes = create_layout(len(pipeline_process), plot_level)

            image_name = os.path.basename(image_path)
            print(f'Image {i+1}/{number_of_images} : {image_name}')

            image_rgb = imread(image_path)

            # check image orientation and untilt it, if necessary.
            angle = read_orientation(image_path)

            if angle not in (None, 0):  # angle == 0 does not need untilting
                image_rgb = img_as_ubyte(rotate(image_rgb, angle=angle, resize=True))

            for step in pipeline_process:
                # first, binarize the input image and return its components.
                _, ruler_bin, lepidop_bin = binarization.main(image_rgb, axes)

                if step == 'ruler_detection':
                    T_space, top_ruler = ruler_detection.main(image_rgb, ruler_bin, axes)

                elif step == 'binarization':
                    # already binarized in the beginning. Moving on...
                    pass

                elif step == 'measurements':
                    points_interest = tracing.main(lepidop_bin, axes)
                    _, dist_mm = measurement.main(points_interest,
                                                  T_space,
                                                  axes)
                    # measuring position and gender
                    position, gender = identification.main(image_rgb)

                    with open(args.path_csv, 'a') as csv_file:
                        _write_csv_data(csv_file, image_name, dist_mm, position,
                                        gender)

            if plot_level > 0:
                output_path = os.path.normpath(
                    os.path.join(args.output_folder, image_name)
                    )
                dpi = args.dpi
                if plot_level == 2:
                    dpi = int(1.5 * args.dpi)
                plt.savefig(output_path, dpi=dpi)
                plt.close()
        except Exception as exc:
            print(f"* Sorry, could not process {image_path}. More details:\n {exc}")
            continue


if __name__ == "__main__":
    main()
