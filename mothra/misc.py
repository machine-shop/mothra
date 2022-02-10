import numpy as np
import os
from exif import Image
from fastai.vision.augment import RandTransform
from fastai.vision.core import PILImage


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
