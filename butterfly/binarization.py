import numpy as np
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from skimage.measure import regionprops
import skimage.color as color
from skimage.exposure import rescale_intensity
from skimage.morphology import binary_erosion
from joblib import Memory
location = './cachedir'
memory = Memory(location, verbose=0)

MIN_REGION_HEIGHT = 0.05
RULER_CROP_MARGIN = 0.025
REGION_CUTOFF = 2/5

def find_tags_edge(image_rgb, top_ruler, axes=None):
    """Find the edge between the tag area on the right and the butterfly area
    and returns the corresponding x coordinate of that vertical line

    Arguments
    ---------
    image_rgb : color image
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


@memory.cache()
def main(image_rgb, top_ruler, axes=None):
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
    bfly_hsv = color.rgb2hsv(bfly_rgb)[:, :, 1]
    rescaled = rescale_intensity(bfly_hsv, out_range=(0, 255))
    thresh_hsv = threshold_otsu(rescaled)
    bfly_bin = rescaled > thresh_hsv

    if axes and axes[1]:
        axes[1].imshow(bfly_bin)
        axes[1].set_title('Binarized butterfly')
    if axes and axes[3]:
        axes[3].axvline(x=label_edge, color='c', linestyle='dashed')

    return bfly_bin
