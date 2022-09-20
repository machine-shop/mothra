import argparse
import numpy as np
import os
import pathlib

from exif import Image
from fastai.vision.augment import RandTransform
from fastai.vision.core import PILImage
from sys import platform

SUPPORTED_IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.tiff', '.tif')
SUPPORTED_TEXT_EXT = ('.txt', '.text')

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


def _generate_parser():
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

    # Enable auto-rotation
    parser.add_argument('-ar', '--auto_rotate',
                        action='store_true',
                        help='Enable automatic rotation of input images\
                        based on EXIF tag')

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

    # Disable cache
    parser.add_argument('--cache',
                        action='store_true',
                        help='Enable computation cache (useful when developing algorithms)')

    args = parser.parse_args()

    return args


def initialize_path(output_folder):
    if os.path.exists(output_folder):
        oldList = os.listdir(output_folder)
        for oldFile in oldList:
            os.remove(output_folder+"/"+oldFile)
    else:
        os.mkdir(output_folder)
    return None


# required by fastai while predicting:
def label_func(image):
    """Function used to label images while training. Required by fastai."""
    return path/"labels"/f"{image.stem}{LABEL_EXT}"


def process_paths_in_input(input_name):
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


def _set_platform_path():
    """Helper function. Checks if the operating system is Windows-based, and adapts
    the path accordingly."""
    if platform.startswith('win'):
        pathlib.PosixPath = pathlib.WindowsPath
    return None
