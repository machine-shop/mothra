import numpy as np
from scipy.sparse.csgraph import connected_components
from vision.ruler_detection.hough_space import grid_hspace_features
from skimage.feature import canny
from skimage.measure import regionprops


def crop_boolean_array(arr):
    B = np.argwhere(arr)
    ystart, xstart = B.min(0)
    ystop, xstop = B.max(0) + 1
    return slice(ystart, ystop), slice(xstart, xstop)


def find_edges(image):
    image_single_channel = image[:, :, 1]
    return canny(image_single_channel, sigma=3)


def best_angles(hspace_entropy):
    angle_scores = np.nanmax(hspace_entropy, axis=-1)
    angle_indices = np.where(np.isnan(np.min(angle_scores, axis=-1)), -1, np.argmin(angle_scores, axis=-1))
    return angle_indices


def find_ruler(image):
    binary_image = find_edges(image) * 255
    hough_spaces, grid_sum = grid_hspace_features(binary_image, grid=16)

    angle_indices = best_angles(hough_spaces)

    labels = merge_cells(angle_indices)

    label_props = regionprops(labels)
    sizes = [prop.filled_area * prop.eccentricity for prop in label_props]
    order = np.argsort(sizes)[::-1]

    height, width = binary_image.shape
    grid_height = np.ceil(height / labels.shape[0])
    grid_width = np.ceil(width / labels.shape[1])
    label_image_size = labels.repeat(grid_height, axis=0).repeat(grid_width, axis=1)
    mask = label_image_size[:height, :width] == label_props[order[0]].label

    crop = crop_boolean_array(mask)
    return image[crop], mask[crop]


def merge_cells(angle_indices):
    def connection(index_a, index_b):
        if (index_a == 0 or index_a == 2) and (index_b == 0 or index_b == 2):
            angle_difference = min((index_a - index_b) % 180, (index_b - index_a) % 180)
            return abs(angle_difference) <= 0
        else:
            return False

    grid_size = len(angle_indices)
    num_grid_elements = grid_size * grid_size

    graph = np.zeros((num_grid_elements, num_grid_elements))
    for i in range(grid_size):
        for j in range(grid_size):
            index = np.ravel_multi_index((i, j), (grid_size, grid_size))

            if i > 0 and connection(angle_indices[i, j], angle_indices[i - 1, j]):
                graph[index, np.ravel_multi_index((i - 1, j), (grid_size, grid_size))] = 1

            if i < (grid_size - 1) and connection(angle_indices[i, j], angle_indices[i + 1, j]):
                graph[index, np.ravel_multi_index((i + 1, j), (grid_size, grid_size))] = 1

            if j > 0 and connection(angle_indices[i, j], angle_indices[i, j - 1]):
                graph[index, np.ravel_multi_index((i, j - 1), (grid_size, grid_size))] = 1

            if j < (grid_size - 1) and connection(angle_indices[i, j], angle_indices[i, j + 1]):
                graph[index, np.ravel_multi_index((i, j + 1), (grid_size, grid_size))] = 1

    n_components, labels = connected_components(graph, directed=False, return_labels=True)
    return (labels + 1).reshape(grid_size, grid_size)
