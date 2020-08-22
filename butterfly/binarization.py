import numpy as np
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
import skimage.color as color
from skimage.exposure import rescale_intensity
from skimage.morphology import binary_erosion, binary_dilation, selem
from skimage.transform import rescale
from skimage.util import img_as_bool, img_as_ubyte
import cv2 as cv
from joblib import Memory
from fastai.vision import load_learner, open_image
from pathlib import Path
from skimage.io import imsave
from . import connection

location = './cachedir'
memory = Memory(location, verbose=0)

# Height of a square region used to determine the minimum region size in find_tags_edge.
# Percent of height of the image.
MIN_REGION_HEIGHT = 0.05

# Height of extra margin to make sure all of the ruler is cropped out in find_tags_edge.
# Percent of height of the image
RULER_CROP_MARGIN = 0.025

# Distance from right-hand edge of the image in which we consider regions to be tags.
# Used in find_tags_edge. Percent of width of the image
REGION_CUTOFF = 1/3

# Size of disk used in dilation to get connect butterfly regions
# Used during getting butterfly bounding box in grabcut
DILATION_SIZE = 10

# Image downsize percentage used to improve grabcut speed
GRABCUT_RESCALE_FACTOR = 0.25

# Number of iterations used in grabcut
GRABCUT_ITERATIONS = 10


def find_tags_edge(image_rgb, top_ruler, axes=None):
    """Find the edge between the tag area on the right and the butterfly area
    and returns the corresponding x coordinate of that vertical line

    Arguments
    ---------
    image_rgb : (M, N, 3) ndarray
        Full RGB image input image
    top_ruler : int
        Y-coordinate of the top of the ruler

    Returns
    -------
    label_edge : int
        x coordinate of the vertical line separating the tags area from the
        butterfly area
    """

    # Make sure ruler is cropped out with some extra margin
    image_rgb = image_rgb[:top_ruler - int(RULER_CROP_MARGIN*image_rgb.shape[0])]

    # Binarize the image with rgb2hsv to highlight the butterfly
    img_hsv = color.rgb2hsv(image_rgb)[:, :, 1]
    img_hsv_rescaled = rescale_intensity(img_hsv, out_range=(0, 255))
    img_hsv_thresh = threshold_otsu(img_hsv_rescaled)
    img_bfly_bin = img_hsv_rescaled > img_hsv_thresh

    # Fill holes and erode the butterfly to get clean butterfly region
    img_bfly_bin_filled = ndi.binary_fill_holes(img_bfly_bin)
    img_bfly_bin_filled_eroded = binary_erosion(img_bfly_bin_filled)


    # Binarize the image with otsu to highlight the labels/ruler
    img_gray = image_rgb[:, :, 0]
    img_otsu_thresh = threshold_otsu(img_gray, nbins=60)
    img_tags_bin = img_gray > img_otsu_thresh

    # Fill holes and erode tags to get clean regions
    img_tags_filled = ndi.binary_fill_holes(img_tags_bin)
    img_tags_filled_eroded = binary_erosion(img_tags_filled)


    # Combine clean butterfly and tags images
    max_img = np.max([img_bfly_bin_filled_eroded, img_tags_filled_eroded], axis=0)

    # Calculate regionprops
    max_img_markers, max_img_labels = ndi.label(max_img)
    max_img_regions = regionprops(max_img_markers)


    # For all notable regions (large, and and in the right third of the image), get their distance to top left corner (0, 0)
    smallest_area = (MIN_REGION_HEIGHT * max_img.shape[0]) ** 2
    max_img_focus_regions = [r for r in max_img_regions if r.area>smallest_area]
    max_img_region_disttocorner = [np.linalg.norm(r.centroid) for r in max_img_focus_regions]

    # Using those, find the ruler and butterfly and ignore them. The remaining regions are tags
    bfly_region = max_img_focus_regions[np.argsort(max_img_region_disttocorner)[0]]
    max_img_focus_regions.remove(bfly_region)

    # To remove ambiguity what is a tag, only look at the right REGION_CUTOFF percent of the image for tags
    cutoff = (1-REGION_CUTOFF) * max_img.shape[1]
    max_img_focus_cutoff_regions = [r for r in max_img_focus_regions if r.centroid[1]>cutoff]

    # From the remaining regions find their leftmost edge
    max_img_leftedges = [r.bbox[1] for r in max_img_focus_cutoff_regions] + [max_img.shape[1]]
    # Binary erosion causes a pixel to be eroded away from the tag edge
    label_edge = np.min(max_img_leftedges) - 1


    if axes and axes[6]:
        halfway = img_tags_filled_eroded.shape[1]//2
        axes[6].imshow(img_tags_filled_eroded[:, halfway:])
        axes[6].axvline(x=label_edge-halfway, color='c', linestyle='dashed')
        axes[6].set_title('Tags detection')

    return label_edge


def grabcut_binarization(bfly_rgb, bfly_bin):
    """Extract shape of the butterfly using OpenCV's grabcut. Greatly improves
    binarization of blue-colored butterflies.

    Arguments
    ---------
    bfly_rgb : (M, N, 3) ndarray
        Input RGB image of butterfly (ruler and tags cropped out)
    bfly_bin : (M, N) ndarray
        Binarizaed image of butterfly (ruler and tags cropped out) Expected
        to be binarized saturation channel of bfly_rgb.

    Returns
    -------
    bfly_grabcut_bin : (M, N) ndarray
        Resulting binarized image of butterfly after segmentation by grabcut.
    """

    # Dilation of image to capture butterfly region
    selem_arr = selem.disk(DILATION_SIZE)
    bfly_bin_dilated = binary_dilation(bfly_bin, selem_arr)
    bfly_bin_dilated_markers, _ = ndi.label(bfly_bin_dilated, ndi.generate_binary_structure(2, 1))
    bfly_bin_dilated_regions = regionprops(bfly_bin_dilated_markers)
    bfly_bin_dilated_regions_sorted = sorted(bfly_bin_dilated_regions, key=lambda r: r.area, reverse=True)
    bfly_region = bfly_bin_dilated_regions_sorted[0]

    # Downscale image to improve grabcut speed
    bfly_rgb_rescale = rescale(bfly_rgb, GRABCUT_RESCALE_FACTOR,
                               multichannel=True)
    bfly_rgb_rescale = img_as_ubyte(bfly_rgb_rescale)

    # Determine grabcut highlight region using butterfly region (after rescaling)
    padding = 0
    rect = (int(GRABCUT_RESCALE_FACTOR*bfly_region.bbox[1]-padding),
            int(GRABCUT_RESCALE_FACTOR*bfly_region.bbox[0]-padding),
            int(GRABCUT_RESCALE_FACTOR*bfly_region.bbox[3]+padding),
            int(GRABCUT_RESCALE_FACTOR*bfly_region.bbox[2]+padding))

    # Grabcut
    mask = np.zeros(bfly_rgb_rescale.shape[:2], np.uint8)
    bgd_model = np.zeros((1,65), np.float64)
    fgd_model = np.zeros((1,65), np.float64)

    cv.grabCut(bfly_rgb_rescale, mask, rect,
        bgd_model, fgd_model, GRABCUT_ITERATIONS, cv.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    bfly_grabcut_rescale = bfly_rgb_rescale*mask2[:, :, np.newaxis]

    # Rescale the image back up and get binary of result
    bfly_grabcut = rescale(bfly_grabcut_rescale, bfly_rgb.shape[0]/bfly_rgb_rescale.shape[0])
    bfly_grabcut_bin = np.max(bfly_grabcut, axis=2)>0

    return bfly_grabcut_bin


def unet_binarization(bfly_rgb):
    """
    """
    WEIGHTS = Path('./butterfly/misc/unet_butterfly.pkl')
    # check if weights is in its folder. If not, download it.
    if not WEIGHTS.is_file():
        print(f'{WEIGHTS} not in the path. Downloading...')
        connection.fetch_data(filename=WEIGHTS)
    # file exists: check if we have the last version; download if not.
    else:
        if connection.has_internet():
            local_hash = connection.read_hash_local()
            online_hash = connection.read_hash_online()
            if local_hash != online_hash:
                print('New training data available. Downloading...')
                connection.fetch_data(filename=WEIGHTS)

    learner = load_learner(path=WEIGHTS.parent, file=WEIGHTS.name)

    # parameters here were defined when training the U-net.
    print('Processing U-net...')
    aux_fname = Path('.bfly_aux.png')
    imsave(fname=aux_fname, arr=bfly_rgb, check_contrast=False)
    bfly_aux = open_image(aux_fname)

    _, pred_classes, _ = learner.predict(bfly_aux)

    # rescale the image back up.
    scale_ratio = np.asarray(bfly_rgb.shape[:2]) / np.asarray(
        pred_classes[0].shape)
    bfly_unet_bin = rescale(image=pred_classes[0].numpy().astype('float'),
                            scale=scale_ratio)

    # removing auxiliary file.
    Path.unlink(aux_fname)

    return bfly_unet_bin


@memory.cache(ignore=['axes'])
def main(image_rgb, top_ruler, grabcut=False, unet=False, axes=None):
    """Binarizes and crops properly image_rgb

    Arguments
    ---------
    image_rgb : 3D array
        RGB image of the entire picture
    top_ruler: integer
        Y-coordinate of the height of the ruler top edge as
        found by ruler_detection.py
    ax : obj
        If any, the result of the binarization and cropping
        will be plotted on it

    Returns
    -------
    bfly_bin : 2D array
        Binarized and cropped version of imge_rgb
    """

    label_edge = find_tags_edge(image_rgb, top_ruler, axes)

    bfly_rgb = image_rgb[:top_ruler, :label_edge]

    if unet:
        bfly_bin = unet_binarization(bfly_rgb)
    else:
        bfly_hsv = color.rgb2hsv(bfly_rgb)[:, :, 1]
        rescaled = rescale_intensity(bfly_hsv, out_range=(0, 255))
        thresh_hsv = threshold_otsu(rescaled)
        bfly_bin = rescaled > thresh_hsv
        if grabcut:
            bfly_bin = grabcut_binarization(bfly_rgb, bfly_bin)

    # if the binary image has more than one region, returns the largest one.
    bfly_bin = return_largest_region(bfly_bin)

    if axes and axes[1]:
        axes[1].imshow(bfly_bin)
        axes[1].set_title('Binarized butterfly')
    if axes and axes[3]:
        axes[3].axvline(x=label_edge, color='c', linestyle='dashed')

    return bfly_bin


def return_largest_region(img_bin):
    """Returns the largest region in the input image.

    Parameters
    ----------
    img_bin : (M, N) ndarray
        A binary image.

    Returns
    -------
    img_bin : (M, N) ndarray
        The input binary image containing only the largest region.
    """
    props = regionprops(label(img_bin))

    # largest_reg will receive the largest region label and its area.
    largest_reg = [0, 0]
    for prop in props:
            if prop.area > largest_reg[1]:
                largest_reg = [prop.label, prop.area]

    img_bin[label(img_bin) != largest_reg[0]] = 0

    return img_as_bool(img_bin)
