import numpy as np
from skimage.transform import hough_line


def average_local_entropy(arr, window_size=2):
    """Calculate the average entropy computed with a sliding window.

    Note:
        Assumes all elements of array are positive

    """
    if np.min(arr) < 0:
        raise ValueError("all elements of array must be posiive")
    arr = 1.0 * arr / np.sum(arr)
    log_arr = np.where(arr <= 0, 0, np.log(arr))

    kernel = np.ones(2 * window_size + 1)
    local_entropy_unnormalized = np.convolve(-arr * log_arr, kernel, mode='valid')
    local_sum = np.convolve(arr, kernel, mode='valid')
    local_entropy = np.where(local_sum == 0, 0, (local_entropy_unnormalized / local_sum) + np.log(local_sum))

    return np.sum(local_entropy) / arr.size


def hspace_angle_score(distance_bins):
    non_zero_indices = np.nonzero(distance_bins)[0]
    if non_zero_indices.size > 10:
        non_zero_distance_bins = distance_bins[non_zero_indices[0]:non_zero_indices[-1]]
        return average_local_entropy(non_zero_distance_bins)
    else:
        return np.nan


def hspace_angle_scale(distance_bins, splits=2):
    if splits <= 1:
        return [hspace_angle_score(distance_bins)]
    else:
        bins_split = np.array_split(distance_bins, 2)
        downscaled_scores = [score for bins in bins_split for score in hspace_angle_scale(bins, splits / 2)]
        current_level_score = hspace_angle_score(distance_bins)
        return downscaled_scores + [current_level_score]


def hspace_features(hspace, splits=2):
    num_angles = hspace.shape[1]
    return [hspace_angle_scale(hspace[:, i], splits) for i in range(num_angles)]


def hough_transform(binary_image, theta=np.linspace(0, np.pi / 2, 3, endpoint=True)):
    """Compute a Hough Transform on a binary image to detect straight lines

    Args:
        binary_image: 2D image, where 0 is off and 255 is on.

    Returns:
        (ndarray, array, array): Bins, angles, distances
                 Values of the bins after the Hough Transform, where the value at (i, j)
                 is the number of 'votes' for a straight line with distance[i] perpendicular to the origin
                 and at angle[j]. Also returns the corresponding array of angles and the corresponding array
                 of distances.

    """
    hspace, angles, distances = hough_line(binary_image, theta)
    return hspace.astype(np.float32), angles, distances


def best_angle(features, feature_range):
    """Return the angle most likely to represent a ruler's graduation, given the bins resulting from a
    Hough Transform.

    Args:
        features: Feature vector describing the bins returned after the Hough Transform.
        feature_range: Range of the values of the features, given feature vectors for all candidate rulers.

    Returns:
        (int, float): The best angle index and its score, for the given features.

    Note:
        The returned angle index refers to an element in an angles array, and not the actual angle value.

    """
    num_features = len(features)
    for i in range(num_features):
        features[i] = (features[i] - feature_range[i][0]) / (feature_range[i][1] - feature_range[i][0])

    spread = features[0] - features[1]
    spread[(features[0] == 0) & (features[1] == 0)] = np.min(spread) - 1

    spread_global = np.zeros_like(spread)
    num_angles = spread.size
    weight = np.arange(num_angles) * np.arange(num_angles)[::-1]
    weight = weight.astype(np.float32) / np.max(weight)
    total_weight = np.sum(weight)
    for i in range(num_angles):
        current_weight = np.roll(weight, i)
        spread_global[i] = spread[i] - np.sum(spread * current_weight) / total_weight

    return np.argmax(spread_global), np.max(spread_global)


def grid_hspace_features(binary_image, grid=8, theta=np.linspace(0, np.pi / 2, 3, endpoint=True)):
    levels = 4
    num_angles = theta.size
    grid_local_entropy = np.zeros((grid, grid, num_angles, np.power(2, levels) - 1))

    height, width = binary_image.shape
    grid_height = int(np.ceil(height / grid))
    grid_width = int(np.ceil(width / grid))

    grid_sum_edges = np.zeros((grid, grid))

    for i in range(grid):
        for j in range(grid):
            grid_i = slice(i * grid_height, (i + 1) * grid_height)
            grid_j = slice(j * grid_width, (j + 1) * grid_width)
            hough_space = hough_transform(binary_image[grid_i, grid_j], theta=theta)
            features = np.array(hspace_features(hough_space[0], splits=np.power(2, levels - 1)))
            grid_local_entropy[i, j, :, :] = features
            grid_sum_edges[i, j] = np.sum(binary_image[grid_i, grid_j])

    return grid_local_entropy, grid_sum_edges
