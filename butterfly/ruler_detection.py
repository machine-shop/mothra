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
    thresh = threshold_otsu(img)
    binary = img > thresh
    return binary[:, :, 0]


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

def remove_numbers(focus):
    '''
    Needs docs and test coverage
    '''
    focus_numbers_markers, focus_numbers_nb_labels = ndi.label(focus, ndi.generate_binary_structure(2, 1))
    focus_numbers_regions = regionprops(focus_numbers_markers)
    focus_numbers_region_areas = [region.filled_area for region in focus_numbers_regions]
    focus_numbers_avg_area = np.mean(focus_numbers_region_areas)
    
    focus_numbers_filled = np.copy(focus)
    for region in focus_numbers_regions:
        if region.eccentricity < 0.99 and region.filled_area > focus_numbers_avg_area:
            min_row, min_col, max_row, max_col = region.bbox
            focus_numbers_filled[min_row:max_row, min_col:max_col] = 0
    
    return focus_numbers_filled


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

    highest_mod_indices = np.flip(np.argsort(mod))
    highest_freqs = freq[highest_mod_indices]
    
    last_freq = highest_freqs[0]
    epsilon = 0.05 * freq[-1]
    for f in highest_freqs:
        if abs(last_freq - f) < epsilon:
            continue
        if np.abs(f - 2 * last_freq) < epsilon:
            break
        if np.abs(last_freq - 2 * f) < epsilon:
            last_freq = f
            break
    f_space = last_freq
    T_space = 1 / f_space

    return 2 * T_space


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
    up_focus = up_rectangle + offset
    left_focus = int(binary.shape[1] * 0.1)
    right_focus = int(binary.shape[1] * 0.9)
    focus = ~binary[up_focus: , left_focus: right_focus]

    # Removing the numbers in the ruler to denoise the fourier transform analysis
    focus_numbers_filled = remove_numbers(focus)

    sums = np.sum(focus_numbers_filled, axis=0) / float(HEIGHT_FOCUS)

    first_index = np.argmax(sums > FIRST_INDEX_THRESHOLD)

    t_space = abs(fourier(sums))

    x_single = [left_focus + first_index, left_focus + first_index +
                t_space]
    y = np.array([up_focus, up_focus])
    x_mult = [left_focus + first_index, left_focus + first_index +
              t_space * 10]
    return t_space, top_ruler, [x_single, y, x_mult, img]
