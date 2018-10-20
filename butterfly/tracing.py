import numpy as np
from scipy import ndimage as ndi
from cmath import exp, polar, pi
from skimage.measure import regionprops
# import matplotlib.pyplot as plt

def moore_neighborhood(current, backtrack): #y, x
    operations = np.array([[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]])
    neighbors = (current + operations).astype(int)

    for i, point in enumerate(neighbors):
        if np.all(point==backtrack):
            return np.concatenate((neighbors[i:], neighbors[:i])) # we return the sorted neighborhood
    return 0


def boundary_tracing(region):

    #creating the binary image
    coords = region.coords
    maxs = np.amax(coords, axis=0)
    binary = np.zeros((maxs[0] +2, maxs[1] +2))
    x = coords[:, 1]
    y = coords[:, 0]
    binary[tuple([y, x])] = 1


    #initilization
    start = [y[0], x[0]] # starting point is the most upper left point
    if binary[start[0] +1, start[1]]==0 and binary[start[0] +1, start[1]-1]==0:
        backtrack_start = [start[0] +1, start[1]]
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
        backtrack = neighbors_current[idx -1]
        current = neighbors_current[idx]
        counter += 1


        if (np.all(current==start) and np.all(backtrack==backtrack_start)):
            # print('iterations :', counter)
            break
    return np.array(boundary)

def symetric_list(n):
    output = []
    for i in range(n):
        if i%2==0:
            output.append(-i/2)
        else:
            output.append((i+1)/2)

    return np.array(output).astype(int)


def fourier_descriptors(boundary, n_descriptors=15):
    y = boundary[:, 0]
    x = boundary[:, 1]
    complex_boundary = x + y*1j
    n = len(boundary)
    descriptors = []
    k_values = symetric_list(n_descriptors)
    for p in range(n_descriptors):
        sum_c = 0
        k = k_values[p]
        for i in range(n):
            sum_c += complex_boundary[i] * exp(-2*pi*1j*(i+1)*k/n)
        descriptors.append(round((sum_c/n).real, 3) + round((sum_c/n).imag, 3)*1j)
    return descriptors



def normalize_descriptors(descriptors):
    mod_c1 = polar(descriptors[1])[0]
    return [round(polar(descriptor)[0]/mod_c1, 4) for descriptor in descriptors[2:]]


def inv_fourier(descriptors, n_points = 1000):
    k_values = symetric_list(len(descriptors))
    x = []
    y = []
    for i in range(n_points):
        z = 0
        for p in range(len(descriptors)):
            k = k_values[p]
            z += descriptors[p]*exp((2*pi*1j*k*i)/n_points)
        z = int(z.real) + int(z.imag)*1j
        x.append(z.real)
        y.append(z.imag)

    x = np.array(x).astype(int)
    y = np.array(y).astype(int)

    return y, x

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
        corner  = np.array([0, width_cropped])
    else:
        corner = np.array([0,0])

    # we look for the boundary closest point to the top right/ left corner
    # as our starting point
    distances = np.linalg.norm(smooth_boundary - corner, axis=1)
    idx_start = np.argmin(distances)
    outer_pixel = smooth_boundary[idx_start]
    if side == 'r':
        ordered_boundary = np.concatenate((smooth_boundary[idx_start+1:],
                                           smooth_boundary[:idx_start+1]),
                                          axis=0)
        ordered_boundary = np.flip(ordered_boundary, axis=0)
    else:
        ordered_boundary = np.concatenate((smooth_boundary[idx_start:],
                                           smooth_boundary[:idx_start]),
                                           axis=0)

    current_idx = 50
    step = 1
    iterations = 1
    if side=='r':
        coeff = 1
    else : 
        coeff = -1
    while True:
        current_pixel = ordered_boundary[current_idx]
        next_idx = current_idx + step
        next_pixel = ordered_boundary[next_idx]
        if current_pixel[0] > next_pixel[0] or coeff*current_pixel[1] < coeff*next_pixel[1]:
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
    normalized = means/np.sum(means)
    sum_values = 0
    for i, value in enumerate(normalized):
        sum_values += i*value
    return int(sum_values)


def main(binary, ax):
    """This function needs documentation

    """
    half = split_picture(binary)
    print(half)

    divided = np.copy(binary)
    divided[:, half:half+5] = 0

    # dilated = bfly_bin
    # for i in range(10):
    #     eroded = binary_erosion(dilated, iterations = 7)
    #     dilated = binary_dilation(eroded, iterations=7)

    # Splitting the image in two
    # divided = np.copy(dilated)
    # half = int(divided.shape[1]/2)
    # divided[:, half] = 0

    # Detecting the wing regions
    markers_divided, _ = ndi.label(divided,
                           structure=ndi.generate_binary_structure(2,1))
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

    # Smoothed boundaries
    boundary_l = boundary_tracing(region_l)
    boundary_r = boundary_tracing(region_r)
    descriptors_l = fourier_descriptors(boundary_l, 45)
    descriptors_r = fourier_descriptors(boundary_r, 45)
    smoothed_y_l, smoothed_x_l = inv_fourier(descriptors_l, 1500)
    smoothed_y_r, smoothed_x_r = inv_fourier(descriptors_r, 1500)
    smoothed_l = np.concatenate((smoothed_y_l.reshape(-1, 1),
                                 smoothed_x_l.reshape(-1, 1)),
                                 axis=1)
    smoothed_r = np.concatenate((smoothed_y_r.reshape(-1, 1),
                                 smoothed_x_r.reshape(-1, 1)),
                                 axis=1)


    # Detecting points of interest
    outer_pix_l, inner_pix_l = detect_points_interest(smoothed_l, 'l', binary.shape[1])
    outer_pix_r, inner_pix_r = detect_points_interest(smoothed_r, 'r', binary.shape[1])


    # Points of interest
    # coords_l = region_l.coords
    # coords_r = region_r.coords

    # idx_out_l = np.argmin(coords_l[:, 0])
    # pix_out_l = list(coords_l[idx_out_l])
    # pix_in_l = [smoothed_y_l[idx_in_l], smoothed_x_l[idx_in_l]]

    # idx_out_r = np.argmin(coords_r[:, 0])
    # pix_out_r = list(coords_r[idx_out_r])
    # pix_in_r = [smoothed_y_r[idx_in_r], smoothed_x_r[idx_in_r]]

    points_interest = np.array([outer_pix_l, inner_pix_l, outer_pix_r, inner_pix_r])
    print(points_interest)

    if ax :
        ax.set_title('Tracing')
        ax.imshow(divided)
        ax.scatter(smoothed_x_l, smoothed_y_l, color='b')
        ax.scatter(smoothed_x_r, smoothed_y_r, color='g')
        ax.scatter(points_interest[:, 1], points_interest[:, 0], color='r')


    return points_interest 


