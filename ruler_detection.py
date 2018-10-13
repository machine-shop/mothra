from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
import numpy as np
from scipy import ndimage as ndi
import cmath
import matplotlib.pyplot as plt

RULER_TOP = 0.7
RULER_LEFT = 0.2
RULER_RIGHT = 0.4
FIRST_INDEX_THRESHOLD = 0.9

"""
    Converts image to binary
"""
def grayscale(img):
    image_gray = img[:, :, 0]
    thresh = threshold_otsu(image_gray, nbins = 60)
    binary = image_gray > thresh
    return binary

"""
    Returns binary rectangle of segment of ruler were interested in
"""
def binarize_rect(up_rectangle, binary):
    left_rectangle = int(binary.shape[1] * RULER_LEFT)
    right_rectangle = int(binary.shape[1] * RULER_RIGHT)

    rectangle = np.zeros((binary.shape[0], binary.shape[1]))
    rectangle[up_rectangle:, left_rectangle: right_rectangle] = 1

    rectangle_binary = binary[up_rectangle:, left_rectangle: right_rectangle]
    return rectangle_binary

"""
    Performs a fourier transform to find the frequency and t space
"""
def fourier(sums):
    fourier = np.fft.fft(sums)
    mod = [cmath.polar(el)[0] for el in fourier]
    freq = np.fft.fftfreq(len(sums))

    idx_max = np.argmax(mod[1:]) + 1
    f_space = freq[idx_max] # nb patterns per pixel
    t_space = 1/f_space
    return t_space

def main(img, ax):
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
    height_focus = 200
    focus = ~binary[up_focus: up_focus + height_focus, left_focus: right_focus]

    sums = np.sum(focus, axis=0)/float(height_focus)

    first_index = np.argmax(sums > FIRST_INDEX_THRESHOLD)

    t_space = fourier(sums)

    ax[0].imshow(img)
    ax[0].plot([left_focus + first_index, left_focus + first_index + t_space], [up_focus, up_focus],  color='red', linewidth=20, markersize=12)
    ax[0].plot([left_focus + first_index, left_focus + first_index + t_space*10], [up_focus-30, up_focus-30],  color='blue', linewidth=20, markersize=12)
    return t_space

if __name__ == '__main__':
    name = "BMNHE_500606.JPG"
    image_name = "./pictures/"+name
    img = imread(image_name)
    fig, ax = plt.subplots(ncols = 2, figsize=(200, 50))
    plt.suptitle(image_name)
    t_space = main(img, ax)
    print "T: ", t_space
    plt.savefig("./output/"+name)
    plt.close()
