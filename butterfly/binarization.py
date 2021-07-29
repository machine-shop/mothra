import numpy as np
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage import color
from skimage.exposure import rescale_intensity
from skimage.morphology import binary_erosion, binary_dilation, selem
from skimage.transform import rescale
from skimage.util import img_as_bool, img_as_float32, img_as_ubyte
from joblib import Memory
from fastai.vision.core import PILImage, to_image
from fastai.vision.learner import load_learner
from pathlib import Path
from torch import from_numpy
from butterfly import connection, ruler_detection

location = './cachedir'
memory = Memory(location, verbose=0)

# Height of extra margin to make sure all of the ruler is cropped out in find_tags_edge.
# Percent of height of the image
RULER_CROP_MARGIN = 0.025

# Testing weights for segmentation of four classes (background, tags, ruler,
# Lepidoptera).
WEIGHTS_SEGMENTATION = './models/segmentation_test-4classes.pkl'


def _convert_image_to_tensor(image):
    """Auxiliary function. Receives an RGB image and convert it to be processed
    by fastai."""
    #image = img_as_float32(np.transpose(image, axes=(2, 0, 1)))
    tensor = to_image(img_as_float32(image))
    #tensor = PILImage.create(from_numpy(image))

    return tensor


def _rescale_image(image_refer, image_to_rescale):
    """Helper function. Rescale image back to original size, according to
    reference."""
    scale_ratio = np.asarray(image_refer.shape[:2]) / np.asarray(image_to_rescale.shape)
    return rescale(image=image_to_rescale, scale=scale_ratio)


def find_tags_edge(tags_bin, top_ruler, axes=None):
    """Find the edge between the tag area on the right and the butterfly area
    and returns the corresponding x coordinate of that vertical line

    Parameters
    ----------
    tags_bin : (M, N) ndarray
        Binary image contaning the tags.
    top_ruler : int
        Y-coordinate of the top of the ruler.
    axes : obj
        If any, detected tags will be plotted on it.

    Returns
    -------
    first_tag_edge : int
        X coordinate of the vertical line separating the tags area from the
        butterfly area.
    """
    # Make sure ruler is cropped out with some extra margin.
    tags_bin = tags_bin[:top_ruler - int(RULER_CROP_MARGIN * tags_bin.shape[0])]

    # Calculate regionprops
    tags_regions = regionprops(label(tags_bin))

    # checking where the first tag starts.
    tags_leftedges = [region.bbox[1] for region in tags_regions]
    first_tag_edge = np.min(tags_leftedges)

    if axes and axes[6]:
        halfway = tags_bin.shape[1] // 2
        axes[6].imshow(tags_bin[:, halfway:])
        axes[6].axvline(x=first_tag_edge-halfway,
                        color='c',
                        linestyle='dashed')
        axes[6].set_title('Tags detection')

    return first_tag_edge


def binarization(image_rgb, weights=WEIGHTS_SEGMENTATION):
    """Extract the shape of the elements in an input image using the U-net
    deep learning architecture.

    Parameters
    ----------
    image_rgb : (M, N, 3) ndarray
        Input RGB image of Lepidoptera, with ruler and tags.

    Returns
    -------
    image_bin : (M, N) ndarray
        Input image binarized.
    """
    if isinstance(weights, str):
        weights = Path(weights)

    connection.download_weights(weights)
    # parameters here were defined when training the U-net.
    learner = load_learner(fname=weights)

    print('Processing U-net...')
    _, _, classes = learner.predict(image_rgb)
    _, tags_bin, ruler_bin, lepidop_bin = classes[:4]

    # rescale the predicted images back up and binarize them.
    tags_bin = img_as_bool(_rescale_image(image_rgb, tags_bin))
    ruler_bin = img_as_bool(_rescale_image(image_rgb, ruler_bin))
    lepidop_bin = img_as_bool(_rescale_image(image_rgb, lepidop_bin))

    return tags_bin, ruler_bin, lepidop_bin


def return_largest_region(image_bin):
    """Returns the largest region in the input image.

    Parameters
    ----------
    image_bin : (M, N) ndarray
        A binary image.

    Returns
    -------
    image_bin : (M, N) ndarray
        The input binary image containing only the largest region.
    """
    props = regionprops(label(image_bin))

    # largest_reg will receive the largest region label and its area.
    largest_reg = [0, 0]
    for prop in props:
            if prop.area > largest_reg[1]:
                largest_reg = [prop.label, prop.area]

    image_bin[label(image_bin) != largest_reg[0]] = 0

    return img_as_bool(image_bin)


@memory.cache(ignore=['axes'])
def main(image_rgb, axes=None):
    """Binarizes and crops the Lepidoptera in image_rgb.

    Parameters
    ----------
    image_rgb : 3D array
        RGB image of the entire picture
    top_ruler: integer
        Y-coordinate of the height of the ruler top edge as
        found by ruler_detection.py
    axes : obj
        If any, the binarization result will be plotted on it.

    Returns
    -------
    image_bin : 2D array
        Binarized version of image_rgb.
    """
    # binarizing the input image and separating its elements.
    tags_bin, ruler_bin, lepidop_bin = binarization(image_rgb, weights=WEIGHTS_SEGMENTATION)

    # if the binary image has more than one region, returns the largest one.
    lepidop_bin = return_largest_region(lepidop_bin)

    # detecting where the ruler starts.
    _, top_ruler = ruler_detection.main(image_rgb, ruler_bin, axes)

    # detecting where the tags start.
    first_tag_edge = find_tags_edge(tags_bin, top_ruler, axes)

    # cropping the Lepidoptera.
    lepidop_bin = lepidop_bin[:top_ruler, :first_tag_edge]

    if axes and axes[1]:
        axes[1].imshow(lepidop_bin)
        axes[1].set_title('Binarized Lepidoptera')
    if axes and axes[3]:
        axes[3].axvline(x=first_tag_edge, color='c', linestyle='dashed')

    return lepidop_bin, ruler_bin, tags_bin
