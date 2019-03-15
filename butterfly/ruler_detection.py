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
HEIGHT_FOCUS_RATIO = 0.116
OFFSET_FOCUS_RATIO = 0.017
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
    image_gray = img[:, :, 0]
    thresh = threshold_otsu(image_gray, nbins=60)
    binary = image_gray > thresh
    return binary


def binarize_rect(up_rectangle, binary, axes):
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
    if axes[3]:
        rect = patches.Rectangle((left_rectangle, up_rectangle),
                                 right_rectangle - left_rectangle,
                                 binary.shape[0] - up_rectangle,
                                 linewidth=1, edgecolor='g', facecolor='none')
        axes[3].add_patch(rect)

    return rectangle_binary


def fourier(signal, axes):
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

    if axes[4]:
        axes[4].plot(signal, linewidth=0.5)
        axes[5].axvline(x=f_space, color='r', linestyle='dotted', linewidth=1)
        axes[5].plot(freq, mod, linewidth=0.5)

    return T_space


@memory.cache()
def main(img, axes):
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
    if axes[0]:
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

    # Focusing on the ticks
    up_focus = int(top_ruler + OFFSET_FOCUS_RATIO * binary.shape[0])
    left_focus = int(binary.shape[1] * 0.1)
    right_focus = int(binary.shape[1] * 0.9)
    height_focus = int(HEIGHT_FOCUS_RATIO * binary.shape[0])
    focus = ~binary[up_focus: up_focus + height_focus, left_focus: right_focus]

    means = np.mean(focus, axis=0)

    first_index = np.argmax(means > FIRST_INDEX_THRESHOLD)

    t_space = abs(fourier(means, axes))

    x_single = [left_focus + first_index, left_focus + first_index +
                t_space]
    y = np.array([up_focus, up_focus])
    x_mult = [left_focus + first_index, left_focus + first_index +
              t_space * 10]

    # Plotting
    if axes[0]:
        axes[0].fill_between(x_single, y, y + LINE_WIDTH, color='red')
        axes[0].fill_between(x_mult, y - LINE_WIDTH, y, color='blue')

    if axes[3]:
        rect = patches.Rectangle((left_focus, up_focus),
                                 right_focus - left_focus,
                                 height_focus,
                                 linewidth=1, edgecolor='r', facecolor='none')
        axes[3].axhline(y=top_ruler, color='b', linestyle='dashed')
        axes[3].add_patch(rect)

    return t_space, top_ruler
