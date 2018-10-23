from skimage.filters import threshold_otsu
from skimage.measure import regionprops
import numpy as np
from scipy import ndimage as ndi
import cmath

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
    thresh = threshold_otsu(image_gray, nbins = 60)
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


def fourier(sums):
    '''Performs a fourier transform to find the frequency and t space

    Parameters
    ----------
    sums : array
        array representing the sums (SUMS OF WHAT)???

    Returns
    -------
    t_space : float
        distance in pixels between two ticks (.5 mm)
    '''
    fourier = np.fft.fft(sums)
    mod = [cmath.polar(el)[0] for el in fourier]
    freq = np.fft.fftfreq(len(sums))

    idx_max = np.argmax(mod[1:]) + 1
    f_space = freq[idx_max] # nb patterns per pixel
    t_space = 1/f_space
    return t_space

def main(img, ax):
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
    markers, nb_labels = ndi.label(rectangle_binary, structure=ndi.generate_binary_structure(2,1))

    regions = regionprops(markers)
    areas = [region.area for region in regions]

    idx_max = np.argmax(areas)
    coords = regions[idx_max].coords
    offset = np.min(coords[:, 0])

    # Focusing on the ticks
    up_focus = up_rectangle + offset + 60
    left_focus = int(binary.shape[1]*0.1)
    right_focus = int(binary.shape[1]*0.9)
    focus = ~binary[up_focus: up_focus + HEIGHT_FOCUS, left_focus: right_focus]

    sums = np.sum(focus, axis=0)/float(HEIGHT_FOCUS)

    first_index = np.argmax(sums > FIRST_INDEX_THRESHOLD)

    t_space = abs(fourier(sums))

    # fig, ax = plt.subplots(figsize=(200, 50))
    if ax:
        ax.imshow(img)

        x_single = [left_focus + first_index, left_focus + first_index + t_space]
        y = np.array([up_focus, up_focus])
        ax.fill_between(x_single, y, y+LINE_WIDTH, color='red')

        x_mult = [left_focus + first_index, left_focus + first_index + t_space*10]
        ax.fill_between(x_mult, y-LINE_WIDTH, y, color='blue')
    return t_space

# if __name__ == '__main__':
#     name = "BMNHE_502320.JPG"
#     image_name = "./pictures/"+name
#     img = imread(image_name)
#     t_space, ax = main(img)
#     print "T: ", t_space
#     plt.savefig("./output/"+name)
#     plt.close()
