import logging
import numpy as np
from statsmodels.tsa.stattools import acf
from vision.ruler_detection.hough_space import hough_transform, hspace_features
from vision.ruler_detection.find_ruler import find_ruler, best_angles
import peakutils
from skimage.morphology import skeletonize
from scipy.ndimage.filters import gaussian_filter1d
from vision.image_functions import threshold, remove_large_components


logging.basicConfig(filename='ruler.log',
                    filemode='w',
                    level=logging.DEBUG,
                    format='%(levelname)s %(message)s')


def find_grid(hspace_angle, max_separation):
    """Returns the separation between graduations of the ruler.

    Args:
        hspace_angle: Bins outputted from :py:meth:`hough_transform`, but for only a single angle.
        max_separation: Maximum size of the *largest* graduation.

    Returns:
        int: Separation between graduations in pixels

    """

    autocorrelation = acf(hspace_angle, nlags=max_separation, unbiased=False)

    smooth = gaussian_filter1d(autocorrelation, 1)
    peaks = peakutils.indexes(smooth, thres=0.25)

    return np.mean(np.diff(np.insert(peaks[:4], 0, 0)))


def ruler_scale_factor(image, distance):
    """Returns the scale factor to convert from image coordinates to real world coordinates

    Args:
        image: BGR image of shape n x m x 3.
        distance: The real world size of the smallest graduation spacing
    Returns:
        float: Unitless scale factor from image coordinates to real world coordinates.

    """

    height, width = image.shape[:2]
    image, mask = find_ruler(image)
    binary_image = mask * threshold(image, mask)

    if binary_image[mask].mean() > 0.5:
        binary_image[mask] = ~binary_image[mask]
    remove_large_components(binary_image, max(height, width))
    edges = skeletonize(binary_image)
    hspace, angles, distances = hough_transform(edges)
    features = hspace_features(hspace, splits=16)
    angle_index = best_angles(np.array(features))

    max_graduation_size = int(max(image.shape))
    line_separation_pixels = find_grid(hspace[:, angle_index], max_graduation_size)

    logging.info('Line separation: {:.3f}'.format(line_separation_pixels))
    return line_separation_pixels, distance / line_separation_pixels
