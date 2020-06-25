from skimage.filters import threshold_otsu
from skimage.measure import regionprops
import numpy as np
from scipy import ndimage as ndi
from joblib import Memory
import matplotlib.patches as patches

location = './cachedir'
memory = Memory(location, verbose=0)

RULER_TOP = 0.7
RULER_LEFT = 0.2
RULER_RIGHT = 0.4
FIRST_INDEX_THRESHOLD = 0.9
LINE_WIDTH = 40


def binarize(img):
    ''' Returns a binarized version of the image.

    Parameters
    ----------
    img : array
        array that represents the image

    Returns
    -------
    binary : array
        array that represents the binarized image
    '''
    gray = color.rgb2gray(img)
    thresh = threshold_otsu(gray)
    binary = gray > thresh
    return binary


def binarize_rect(up_rectangle, binary, axes=None):
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

    rectangle_binary = binary[up_rectangle:, left_rectangle: right_rectangle]
    if axes and axes[3]:
        rect = patches.Rectangle((left_rectangle, up_rectangle),
                                 right_rectangle - left_rectangle,
                                 binary.shape[0] - up_rectangle,
                                 linewidth=1, edgecolor='g', facecolor='none')
        axes[3].add_patch(rect)

    return rectangle_binary

def remove_numbers(focus):
    ''' Returns a ruler image but with the numbers stripped away, to improve ruler
    fourier transform analysis

    Parameters
    ----------
    focus : 2D array
        Binary image of the ruler

    Returns
    -------
    focus_nummbers_filled : 2D array
        Binary image of the ruler without numbers
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


def fourier(signal, axes=None):
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
    
    # thresholding the signal so the fourier transform results better correlate to 
    # frequency and not amplitude of the signal
    signal_thresholded = signal > 0

    fourier = np.fft.rfft(signal_thresholded)
    mod = np.abs(fourier)
    mod[0:10] = 0  # we discard the first several coeffs
    freq = np.fft.rfftfreq(len(signal_thresholded))
    
    f_space = freq[np.argmax(mod)]
    T_space = 1 / f_space

    if axes and axes[4]:
        axes[4].plot(signal, linewidth=0.5)
        axes[5].axvline(x=f_space, color='r', linestyle='dotted', linewidth=1)
        axes[5].plot(freq, mod, linewidth=0.5)

    return T_space


@memory.cache(ignore=['axes'])
def main(img, axes=None):
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
    binary = binarize(img)
    if axes and axes[0]:
        axes[0].set_title('Final output')
        axes[0].imshow(img)
        if axes[3]:
            axes[3].set_title('Image structure')
            axes[4].set_title('Ruler signal')
            axes[5].set_title('Fourier transform of ruler signal')
            axes[3].imshow(img)

    # Detecting top ruler
    up_rectangle = int(binary.shape[0] * RULER_TOP)
    rectangle_binary = binarize_rect(up_rectangle, binary, axes)
    markers, nb_labels = ndi.label(rectangle_binary,
                                   ndi.generate_binary_structure(2, 1))

    regions = regionprops(markers)
    areas = [region.area for region in regions]

    idx_max = np.argmax(areas)
    coords = regions[idx_max].coords
    offset = np.min(coords[:, 0])
    top_ruler = up_rectangle + offset

    # Focusing on the ruler
    up_focus = up_rectangle + offset
    focus = ~binary[up_focus:]

    # Removing the numbers in the ruler to denoise the fourier transform analysis
    focus_numbers_filled = remove_numbers(focus)

    # Cropping the center of the ruler to improve detection
    up_trim = int(0.1*focus_numbers_filled.shape[0])
    down_trim = int(0.75*focus_numbers_filled.shape[0])
    left_focus = int(0.1*focus_numbers_filled.shape[1])
    right_focus = int(0.9*focus_numbers_filled.shape[1])
    focus_numbers_filled = focus_numbers_filled[up_trim:down_trim, left_focus:right_focus]

    means = np.mean(focus_numbers_filled, axis=0)
    first_index = np.argmax(means > FIRST_INDEX_THRESHOLD * means.max())

    # Fourier transform analysis to give us the pixels between the 1mm ticks
    sums = np.sum(focus_numbers_filled, axis=0)
    t_space = 2 * fourier(sums, axes)

    x_single = [left_focus + first_index, left_focus + first_index +
                t_space]
    y = np.array([up_focus, up_focus])
    x_mult = [left_focus + first_index, left_focus + first_index +
              t_space * 10]

    # Plotting
    if axes and axes[0]:
        axes[0].fill_between(x_single, y, y + LINE_WIDTH, color='red', linewidth=0)
        axes[0].fill_between(x_mult, y - LINE_WIDTH, y, color='blue', linewidth=0)

    if axes and axes[3]:
        rect = patches.Rectangle((left_focus, up_focus+up_trim),
                                 right_focus - left_focus,
                                 down_trim,
                                 linewidth=1, edgecolor='r', facecolor='none')
        axes[3].axhline(y=top_ruler, color='b', linestyle='dashed')
        axes[3].add_patch(rect)

    return t_space, top_ruler
