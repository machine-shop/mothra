import numpy as np
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from skimage.measure import regionprops
import skimage.color as color
from skimage.exposure import rescale_intensity


def find_tags_edge(binary, top_ruler):
    """Find the edge between the tag area on the right and the butterfly area
    and returns the corresponding x coordinate of that vertical line

    Arguments
    ---------
    binary : 2D array
        Binarized image of the entire RGB image
    top_ruler : int
        Y-coordinate of the top of the ruler

    Returns
    -------
    crop_right : int
        x coordinate of the vertical line separating the tags area from the
        butterfly area
    """

    focus = binary[:top_ruler, int(binary.shape[1] * 0.5):]
    markers = ndi.label(focus,
                        structure=ndi.generate_binary_structure(2, 1))[0]
    regions = regionprops(markers)
    areas = [region.area for region in regions]
    area_min = 0.01*binary.shape[0]*binary.shape[1]
    filtered_regions = []
    for i, area in enumerate(areas):
        if area > area_min:
            filtered_regions.append(regions[i])
    left_pixels = [np.min(region.coords[:, 1]) for region in filtered_regions]
    crop_right = int(0.5*binary.shape[1] + np.min(left_pixels))

    return crop_right


def main(image_rgb, top_ruler, ax=None):
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

    image_gray = image_rgb[:, :, 0]
    thresh_rgb = threshold_otsu(image_gray, nbins=60)
    binary = image_gray > thresh_rgb

    label_edge = find_tags_edge(binary, top_ruler)

    bfly_rgb = image_rgb[:top_ruler, :label_edge]
    bfly_hsv = color.rgb2hsv(bfly_rgb)[:, :, 1]
    rescaled = rescale_intensity(bfly_hsv, out_range=(0, 255))
    thresh_hsv = threshold_otsu(rescaled)
    bfly_bin = rescaled > thresh_hsv

    if ax:
        ax.set_title('Binary')
        ax.imshow(bfly_bin)

    return bfly_bin
