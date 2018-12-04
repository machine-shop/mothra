from skimage.filters import threshold_otsu
from skimage.measure import regionprops
import numpy as np
from scipy import ndimage as ndi
from joblib import Memory

location = './cachedir'
memory = Memory(location, verbose=0)

RULER_TOP = 0.7
RULER_LEFT = 0.2
RULER_RIGHT = 0.4
FIRST_INDEX_THRESHOLD = 0.9
HEIGHT_FOCUS = 400
LINE_WIDTH = 40


def grayscale(img):
    ''' Returns a grayscale version of the image.

    Parameters
    ----------
    img : array
        array that represents the image

    Returns
    -------
    binary : array
        array that represents the binarized image
    '''
    image_gray = img[:, :, 0]
    thresh = threshold_otsu(image_gray, nbins=60)
    binary = image_gray > thresh
    return binary


def binarize_rect(up_rectangle, binary):
    '''Returns binary rectangle of segment of ruler were interested in

        Parameters
        ----------
        up_rectangle : integer
            This is the height of the rectangle we are fetching.
        binary : array
            array that represents the binarized image

        Returns
        -------
        rectangle_binary : array
            array that represents just the rectangle area of the image we want
        '''
    left_rectangle = int(binary.shape[1] * RULER_LEFT)
    right_rectangle = int(binary.shape[1] * RULER_RIGHT)

    rectangle = np.zeros((binary.shape[0], binary.shape[1]))
    rectangle[up_rectangle:, left_rectangle: right_rectangle] = 1

    rectangle_binary = binary[up_rectangle:, left_rectangle: right_rectangle]
    return rectangle_binary


def fourier(signal):
    '''Performs a fourier transform to find the frequency and t space

    Parameters
    ----------
    signal : 1D array
        array representing the value of the ticks in space

    Returns
    -------
    t_space : float
        distance in pixels between two ticks (.5 mm)
    '''
    fourier = np.fft.rfft(signal)
    mod = np.abs(fourier)
    mod[0] = 0  # we discard the first coeff
    freq = np.fft.rfftfreq(len(signal))


    # Normalization
    mod = mod / np.max(mod)

    # Choose frequence
    f_space = freq[mod > 0.6][0]
    T_space = 1 / f_space

    return T_space


@memory.cache()
def main(img):
    '''Finds the distance between ticks

    Parameters
    ----------
    img : array
        array representing the image
    ax : array
        array of Axes that show subplots

    Returns
    -------
    t_space : float
        distance between two ticks (.5 mm)
    '''
    binary = grayscale(img)

    up_rectangle = int(binary.shape[0] * RULER_TOP)
    rectangle_binary = binarize_rect(up_rectangle, binary)
    markers, nb_labels = ndi.label(rectangle_binary,
                                   ndi.generate_binary_structure(2, 1))

    regions = regionprops(markers)
    areas = [region.area for region in regions]

    idx_max = np.argmax(areas)
    coords = regions[idx_max].coords
    offset = np.min(coords[:, 0])
    top_ruler = up_rectangle + offset

    # Focusing on the ticks
    up_focus = up_rectangle + offset + 60
    left_focus = int(binary.shape[1] * 0.1)
    right_focus = int(binary.shape[1] * 0.9)
    focus = ~binary[up_focus: up_focus + HEIGHT_FOCUS, left_focus: right_focus]

    sums = np.sum(focus, axis=0) / float(HEIGHT_FOCUS)

    first_index = np.argmax(sums > FIRST_INDEX_THRESHOLD)

    t_space = abs(fourier(sums))

    x_single = [left_focus + first_index, left_focus + first_index +
                t_space]
    y = np.array([up_focus, up_focus])
    x_mult = [left_focus + first_index, left_focus + first_index +
              t_space * 10]
    return t_space, top_ruler, [x_single, y, x_mult, img]
