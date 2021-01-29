import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops
from joblib import Memory
from scipy.ndimage.morphology import binary_dilation


location = './cachedir'
memory = Memory(location, verbose=0)


def remove_antenna(half_binary):
    """Remove antenna if connected to the wing

    Arguments
    ---------
    half_binary : 2D array
        binary image of left/right wing

    Returns
    -------
    without_antenna : 2D array
        binary image, same shape as input without antenna (if it touches the
        wing)
    """

    markers, _ = ndi.label(1-half_binary, ndi.generate_binary_structure(2, 1))
    regions = regionprops(markers)

    areas = np.array([r.area for r in regions])
    idx_sorted = 1 + np.argsort(-areas)[:2]

    try:
        dilated_bg = binary_dilation(markers == idx_sorted[0], iterations=35)
        dilated_hole = binary_dilation(markers == idx_sorted[1], iterations=35)
        intersection = np.minimum(dilated_bg, dilated_hole)
        without_antenna = np.copy(half_binary)
        without_antenna[intersection] = 0
    except IndexError:
        return half_binary

    return without_antenna


def detect_outer_pix(half_binary, center):
    """Relative (r, c) coordinates of outer pixel (wing's tip)

    Arguments
    ---------
    half_binary : 2D array
        Binary image of left/right wing.
    center : tuple
        Centroid of the lepidopteran.

    Returns
    -------
    outer_pix : 1D array
        relative coordinates of the outer pixel (r, c)
    """
    markers, _ = ndi.label(half_binary,
                           ndi.generate_binary_structure(2, 1))
    regions = regionprops(markers)
    areas = [r.area for r in regions]
    idx_max = np.argmax(areas)

    coords = regions[idx_max].coords
    distances = np.linalg.norm(coords - center, axis=-1)
    idx_outer_pix = np.argmax(distances)
    outer_pix = coords[idx_outer_pix]

    return outer_pix


def detect_inner_pix(half_binary, outer_pix, side):
    """Relative (r, c) coordinates of the inner pixel (between wing and body)

    Arguments
    ---------
    half_binary : 2D array
        binary image of left/right wing
    outer_pix : 1D array
        (r, c) coordinates (relative) of the outer pixel
    side : str
        left ('l') or right ('r') wing

    Returns
    -------
    inner_pix : 2D array
        relative coordinates of the inner pixel (r, c)
    """
    lower_bound = int(half_binary.shape[0]*0.75)

    if side == 'l':
        focus = half_binary[:lower_bound, outer_pix[1]:]
    else:
        focus = half_binary[:lower_bound, :outer_pix[1]]

    focus_inv = 1 - focus

    markers, _ = ndi.label(focus_inv, ndi.generate_binary_structure(2, 1))
    regions = regionprops(markers)
    areas = [r.area for r in regions]
    idx_max = np.argmax(areas)
    coords = regions[idx_max].coords
    y_max = np.max(coords[:, 0])
    mask = (coords[:, 0] == y_max)
    selection = coords[mask]
    if side == 'l':
        idx = np.argmax(selection[:, 1])
    else:
        idx = np.argmin(selection[:, 1])

    outer_pix = selection[idx]

    return outer_pix


def split_picture(binary):
    """Calculate the middle of the butterfly.

    Parameters
    ----------
    binary : ndarray of bool
        Binary butterfly mask.

    Returns
    -------
    midpoint : int
        Horizontal coordinate for middle of butterfly.

    Notes
    -----
    Currently, this is calculated by finding the center
    of gravity of the butterfly.
    """

    means = np.mean(binary, 0)
    normalized = means / np.sum(means)
    sum_values = 0
    for i, value in enumerate(normalized):
        sum_values += i * value
    return int(sum_values)


@memory.cache(ignore=['axes'])
def main(binary, axes=None):
    """Find and returns the coordinates of the 4 points of interest

    Arguments
    ---------
    binary : 2D array
        Binarized and cropped version of the butterfly
    ax : obj
        If any is provided, POI, smoothed wings boundaries and binary will
        be plotted on it

    Returns
    -------
    points_interest : dictionary 
        Dictionary containing the points of interest in the form [y, x],
        keyed with "outer_pix_l", "inner_pix_l", "outer_pix_r", "inner_pix_r", 
        "body_center"
    """
    binary = binary_dilation(binary, iterations=2)

    # Split the butterfly
    middle = split_picture(binary)

    binary_left = binary[:, :middle]
    binary_right = binary[:, middle:]

    # Centroid of central column
    middle_arr = binary[:, middle]
    middle_y = int(np.mean(np.argwhere(middle_arr)))
    body_center = (middle_y, middle)

    # Left wing
    without_antenna_l = remove_antenna(binary_left)
    outer_pix_l = detect_outer_pix(without_antenna_l, body_center)
    inner_pix_l = detect_inner_pix(without_antenna_l, outer_pix_l, 'l')
    inner_pix_l = inner_pix_l + np.array([0, outer_pix_l[1]])

    # Right wing
    body_center_r = (middle_y, 0)  # to calculate outer_pix_r correctly
    without_antenna_r = remove_antenna(binary_right)
    outer_pix_r = detect_outer_pix(without_antenna_r, body_center_r)
    inner_pix_r = detect_inner_pix(without_antenna_r, outer_pix_r, 'r')
    inner_pix_r = inner_pix_r + np.array([0, middle])
    outer_pix_r = outer_pix_r + np.array([0, middle])

    points_interest = {
        "outer_pix_l": outer_pix_l,
        "inner_pix_l": inner_pix_l,
        "outer_pix_r": outer_pix_r,
        "inner_pix_r": inner_pix_r,
        "body_center": body_center
    }

    # Reconstruct binary image without antennae
    without_antennae = np.concatenate((without_antenna_l, without_antenna_r),
                                      axis=1)
    if axes and axes[2]:
        axes[2].set_title('Points of interest')
        axes[2].imshow(without_antennae)
        axes[2].axvline(middle, color='m', linestyle='dashed')
        markersize = 10
        if axes[3]:
            markersize = 2
        points_interest_arr = np.array(list(points_interest.values()))
        axes[2].scatter(points_interest_arr[:, 1], points_interest_arr[:, 0],
                        color='r', s=markersize)

    return points_interest
