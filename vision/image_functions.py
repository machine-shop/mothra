from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops


def threshold(image, mask=None):
    """Convert a full color image to a binary image

    Args:
        image (ndarray): BGR image of shape n x m x 3.
        mask (ndarray): binary image. Calculates threshold value based only on pixels within the mask.
                        However, all pixels are included in the output image.

    Returns:
        ndarray: Boolean array of shape n x m.

    """
    if len(image.shape) == 3:
        image = image[:, :, 1]

    if mask is not None:
        image_vals = image[mask > 0]
    else:
        image_vals = image

    return image > threshold_otsu(image_vals.reshape(-1, 1))


def remove_large_components(binary_image, threshold_size=0):
    if threshold_size == 0:
        threshold_size = max(binary_image.shape)

    labels = label(binary_image)
    components = regionprops(labels)

    for component in components:
        if component.filled_area > threshold_size:
            binary_image[labels == component.label] = False
