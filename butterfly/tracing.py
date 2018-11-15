import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops


def moore_neighborhood(current, backtrack):  # y, x
    operations = np.array([[-1, 0], [-1, 1], [0, 1], [1, 1],
                           [1, 0], [1, -1], [0, -1], [-1, -1]])
    neighbors = (current + operations).astype(int)

    for i, point in enumerate(neighbors):
        if np.all(point == backtrack):
            # we return the sorted neighborhood
            return np.concatenate((neighbors[i:], neighbors[:i]))
    return 0


def boundary_tracing(region):
    # creating the binary image
    coords = region.coords
    maxs = np.amax(coords, axis=0)
    binary = np.zeros((maxs[0] + 2, maxs[1] + 2))
    x = coords[:, 1]
    y = coords[:, 0]
    binary[tuple([y, x])] = 1

    # initilization
    start = [y[0], x[0]]  # starting point is the most upper left point
    if (binary[start[0] + 1, start[1]] == 0 and
            binary[start[0] + 1, start[1] - 1] == 0):
        backtrack_start = [start[0] + 1, start[1]]
    else:
        backtrack_start = [start[0], start[1] - 1]

    current = start
    backtrack = backtrack_start
    boundary = []
    counter = 0
    while True:
        neighbors_current = moore_neighborhood(current, backtrack)
        y = neighbors_current[:, 0]
        x = neighbors_current[:, 1]
        idx = np.argmax(binary[tuple([y, x])])
        boundary.append(current)
        backtrack = neighbors_current[idx - 1]
        current = neighbors_current[idx]
        counter += 1

        if (np.all(current == start) and np.all(backtrack == backtrack_start)):
            # print('iterations :', counter)
            break
    return np.array(boundary)


def symetric_list(n):
    output = []
    for i in range(n):
        if i % 2 == 0:
            output.append(-i / 2)
        else:
            output.append((i + 1) / 2)

    return np.array(output).astype(int)


def fourier_descriptors(boundary, n_descriptors=15):
    """List of Fourier descriptors with the first [n_descriptors] non-zeros.

    Arguments
    ---------
    boundary : 2D array
        Each row represents a different point of the boundary [y, x]
    n_descriptors : int
        Number of first descriptors we don't want to filter

    Returns
    -------
    descriptors : 1D array
        array of same length as boundary.shape[0]
    """
    y = boundary[:, 0]
    x = boundary[:, 1]
    complex_boundary = x + y * 1j

    descriptors = np.fft.fft(complex_boundary)
    descriptors[1+n_descriptors//2: 1+(-n_descriptors//2)] = 0
    return descriptors


def inv_fourier(descriptors):
    """Recreate a smooth boundary from the filtered descriptors.

    Arguments
    ---------
    descriptors : 1D array
        filtered descriptors (the discarded coefficients are zeros)

    Returns
    -------
    smoothed_y : 1D array
        y coordinates of smoothed boundary points
    smoothed_x : 1D array
        x coordinates of smoothed boundary points
    """
    inv = np.fft.ifft(descriptors)
    smoothed_y = np.imag(inv)
    smoothed_x = np.real(inv)

    return smoothed_y, smoothed_x


def detect_points_interest(smooth_boundary, side, width_cropped):
    """Calculate the outer and inner points of interest of the wing

    Arguments
    ---------
    smooth_boundary : 2D numpy array [[y1, x1], [y2, x2] .. ]
        Clockwise coordinates of the smoothed boundary, starting
        with the highest pixel.
    side : str 'l' or 'r'
        'left' or 'right' wing
    width_cropped : int
        width og the cropped butterfly in pixels

    Returns
    -------
    outer_pixel : 1D numpy array [y, x]
        Coordinates of the boundary closest point to the top right/left corner
    inner_pixel : 1D numpy array [y, x]
        Coordinates of the top of the junction between body and wing
    """
    if side == 'r':
        corner = np.array([0, width_cropped])
    else:
        corner = np.array([0, 0])

    # we look for the boundary closest point to the top right/ left corner
    # as our starting point
    distances = np.linalg.norm(smooth_boundary - corner, axis=1)
    idx_start = np.argmin(distances)
    outer_pixel = smooth_boundary[idx_start]
    if side == 'r':
        ordered_boundary = np.concatenate((smooth_boundary[idx_start + 1:],
                                           smooth_boundary[:idx_start + 1]),
                                          axis=0)
        ordered_boundary = np.flip(ordered_boundary, axis=0)
    else:
        ordered_boundary = np.concatenate((smooth_boundary[idx_start:],
                                           smooth_boundary[:idx_start]),
                                          axis=0)

    current_idx = max(1, int(len(smooth_boundary)*0.05))
    step = max(1, int(len(smooth_boundary)*0.001))

    iterations = 1
    if side == 'r':
        coeff = 1
    else:
        coeff = -1
    while True:
        current_pixel = ordered_boundary[current_idx]
        next_idx = current_idx + step
        next_pixel = ordered_boundary[next_idx]
        if (current_pixel[0] > next_pixel[0]
                or coeff * current_pixel[1] < coeff * next_pixel[1]):
            print('iterations :', iterations)
            inner_pixel = next_pixel
            break
        iterations += 1
        current_idx = next_idx
    return outer_pixel, inner_pixel


def split_picture(closed):
    """Calculate the middle of the butterfly.

    Parameters
    ----------
    closed : ndarray of bool
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

    means = np.mean(closed, 0)
    normalized = means / np.sum(means)
    sum_values = 0
    for i, value in enumerate(normalized):
        sum_values += i * value
    return int(sum_values)


def main(binary, ax):
    """Find and retunrs the coordinates of the 4 points of interest

    Arguments
    ---------
    binary : 2D array
        Binarized and and cropped version of the butterfly
    ax : obj
        If any is provided, POI, smoothed wings boundaries and binary will
        be plotted on it

    Returns
    -------
    points_interest : 2D array
        array of coordinates of the points of interest in the form [y, x]
        [outer_pix_l, inner_pix_l, outer_pix_r, inner_pix_r]

    Notes
    -----
    This stage can be split in 4 different parts :
    1. Split the butterfly to separate the two wings
    2. Find the boundaries of each wing
    3. Smooth those boundaries with Fourier transformation
    4. Detect the points of interest from the smoothed boundaries

    """

    # 1. Split the butterfly
    half = split_picture(binary)
    # print(half)

    divided = np.copy(binary)
    divided[:, half:half + 5] = 0

    # Detecting the wing regions
    markers_divided, _ = ndi.label(divided,
                                   ndi.generate_binary_structure(2, 1))
    regions = regionprops(markers_divided)
    areas = [region.area for region in regions]

    idx_1 = np.argmax(areas)
    coords1 = regions[idx_1].coords
    areas[idx_1] = 0
    idx_2 = np.argmax(areas)
    coords2 = regions[idx_2].coords

    # Determining which one is left or right
    if np.min(coords1[:, 1]) < np.min(coords2[:, 1]):
        region_l, region_r = regions[idx_1], regions[idx_2]
    else:
        region_l, region_r = regions[idx_2], regions[idx_1]

    # 2. Find boundaries
    boundary_l = boundary_tracing(region_l)
    boundary_r = boundary_tracing(region_r)

    # 3. Smooth boundaries
    descriptors_l = fourier_descriptors(boundary_l, 45)
    descriptors_r = fourier_descriptors(boundary_r, 45)
    smoothed_y_l, smoothed_x_l = inv_fourier(descriptors_l)
    smoothed_y_r, smoothed_x_r = inv_fourier(descriptors_r)
    smoothed_l = np.concatenate((smoothed_y_l.reshape(-1, 1),
                                 smoothed_x_l.reshape(-1, 1)),
                                axis=1)
    smoothed_r = np.concatenate((smoothed_y_r.reshape(-1, 1),
                                 smoothed_x_r.reshape(-1, 1)),
                                axis=1)

    # 4. Detecting points of interest
    outer_pix_l, inner_pix_l = detect_points_interest(smoothed_l,
                                                      'l', binary.shape[1])
    outer_pix_r, inner_pix_r = detect_points_interest(smoothed_r,
                                                      'r', binary.shape[1])

    points_interest = np.array([outer_pix_l, inner_pix_l,
                                outer_pix_r, inner_pix_r])

    if ax:
        ax.set_title('Tracing')
        ax.imshow(divided)
        ax.scatter(smoothed_x_l, smoothed_y_l, color='white', s=4)
        ax.scatter(smoothed_x_r, smoothed_y_r, color='cyan', s=4)
        ax.scatter(points_interest[:, 1],
                   points_interest[:, 0], color='r', s=10)

    return points_interest
