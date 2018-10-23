import numpy as np
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from skimage.measure import regionprops
import skimage.color as color
from skimage.exposure import rescale_intensity



def find_ruler_edge(binary):
    ''' Returns the pixel
    binary : array
        array that represents the binarized image


    :return:
    '''

    up_rectangle = int(binary.shape[0]*0.7)
    left_rectangle = int(binary.shape[1]*0.2)
    right_rectangle = int(binary.shape[1]*0.4)

    rectangle = np.zeros((binary.shape[0], binary.shape[1]))
    rectangle[up_rectangle:, left_rectangle: right_rectangle] = 1

    rectangle_binary = binary[up_rectangle:, left_rectangle: right_rectangle]
    markers, nb_labels = ndi.label(rectangle_binary, structure=ndi.generate_binary_structure(2,1))

    regions = regionprops(markers)
    areas = [region.area for region in regions]

    idx_max = np.argmax(areas)
    coords = regions[idx_max].coords
    offset = np.min(coords[:, 0])

    return up_rectangle + offset


def find_label_edge(binary, focus):
	
	markers = ndi.label(focus, structure=ndi.generate_binary_structure(2,1))[0]
	regions = regionprops(markers)
	areas = [region.area for region in regions]
	area_min = 0.01*binary.shape[0]*binary.shape[1]
	filtered_regions = []
	for i, area in enumerate(areas):
	    if area > area_min:
	        filtered_regions.append(regions[i])       
	left_pixels = [np.min(region.coords[:, 1]) for region in filtered_regions]
	crop_right = int(0.5*binary.shape[1] + np.min(left_pixels))

	return crop_right


def main(image_rgb, ax):

	# find top of ruler
    image_gray = image_rgb[:, :, 0]
    thresh = threshold_otsu(image_gray, nbins = 60)
    binary = image_gray > thresh

    ruler_edge = find_ruler_edge(binary)

    focus = binary[:ruler_edge, int(binary.shape[1]* 0.5):]

    label_edge = find_label_edge(binary, focus)

    bfly_rgb = image_rgb[:ruler_edge, :label_edge]
    bfly_hsv = color.rgb2hsv(bfly_rgb)[:, :, 1]
    rescaled = rescale_intensity(bfly_hsv, out_range=(0, 255))
    tresh = threshold_otsu(rescaled)
    bfly_bin = rescaled > thresh

    # fig, ax = plt.subplots()
    if ax:
        ax.set_title('binary')
        ax.imshow(bfly_bin)

    return bfly_bin


# image_rgb = imread('../data/Measured_images_Data_H.comma/BMNHE_502326.JPG')
# result = main(image_rgb)
# plt.imshow(result)
# plt.show(block=True)
