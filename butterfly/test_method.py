import numpy as np
import click
from skimage.io import imread
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.filters import threshold_otsu
import skimage.color as color
from scipy import ndimage as ndi
from skimage.measure import regionprops
import cmath
import os
from skimage.exposure import rescale_intensity
from cmath import exp, polar, pi
from skimage.morphology.selem import disk
from scipy.ndimage.morphology import binary_erosion, binary_dilation, binary_closing
import shutil


def moore_neighborhood(current, backtrack): #y, x
    operations = np.array([[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]])
    neighbors = (current + operations).astype(int)

    for i, point in enumerate(neighbors):
        if np.all(point==backtrack):
            return np.concatenate((neighbors[i:], neighbors[:i])) # we return the sorted neighborhood
    return 0


def boundary_tracing(region):

    #creating the binary image
    coords = region.coords
    maxs = np.amax(coords, axis=0)
    binary = np.zeros((maxs[0] +2, maxs[1] +2))
    x = coords[:, 1]
    y = coords[:, 0]
    binary[tuple([y, x])] = 1


    #initilization
    start = [y[0], x[0]] # starting point is the most upper left point
    if binary[start[0] +1, start[1]]==0 and binary[start[0] +1, start[1]-1]==0:
        backtrack_start = [start[0] +1, start[1]]
    else:
        backtrack_start = [start[0], start[1] - 1]

    current = start
    backtrack = backtrack_start
    boundary = []
    counter = 0
    while True:
        neighbors_current = moore_neighborhood(current, backtrack)
        y = neighbors_current[:, 0]
        x = neighbors_current[:, 1]
        idx = np.argmax(binary[tuple([y, x])])
        boundary.append(current)
        backtrack = neighbors_current[idx -1]
        current = neighbors_current[idx]
        counter += 1


        if (np.all(current==start) and np.all(backtrack==backtrack_start)):
            print('iterations :', counter)
            break
    return np.array(boundary)

def symetric_list(n):
    output = []
    for i in range(n):
        if i%2==0:
            output.append(-i/2)
        else:
            output.append((i+1)/2)
    return np.array(output).astype(int)


def fourier_descriptors(boundary, n_descriptors=15):
    y = boundary[:, 0]
    x = boundary[:, 1]
    complex_boundary = x + y*1j
    n = len(boundary)
    descriptors = []
    k_values = symetric_list(n_descriptors)
    for p in range(n_descriptors):
        sum_c = 0
        k = k_values[p]
        for i in range(n):
            sum_c += complex_boundary[i] * exp(-2*pi*1j*(i+1)*k/n)
        descriptors.append(round((sum_c/n).real, 3) + round((sum_c/n).imag, 3)*1j)
    return descriptors



def normalize_descriptors(descriptors):
    mod_c1 = polar(descriptors[1])[0]
    return [round(polar(descriptor)[0]/mod_c1, 4) for descriptor in descriptors[2:]]


def inv_fourier(descriptors, n_points = 1000):
    k_values = symetric_list(len(descriptors))
    x = []
    y = []
    for i in range(n_points):
        z = 0
        for p in range(len(descriptors)):
            k = k_values[p]
            z += descriptors[p]*exp((2*pi*1j*k*i)/n_points)
        z = int(z.real) + int(z.imag)*1j
        x.append(z.real)
        y.append(z.imag)

    x = np.array(x).astype(int)
    y = np.array(y).astype(int)

    return y, x

def detect_top_junction(smooth_boundary_y, side):
    if side=='r':
        coeff = -1
    elif side == 'l':
        coeff = 1
    start_idx = 50
    current_idx = coeff*start_idx
    step = 1
    iterations = 1
    while True:
        current_pixel_y = smooth_boundary_y[current_idx]
        next_idx = current_idx + coeff*step
        next_pixel_y = smooth_boundary_y[next_idx]
        if current_pixel_y > next_pixel_y:
            print('iterations :', iterations)
            return next_idx
        iterations += 1
        current_idx = next_idx

def split_picture(closed):
	means = np.mean(closed, 0)
	diff = np.diff(means, 5)
	thresholded = diff > 0.1
	left_margin = np.argmax(thresholded)
	right_margin = np.argmax(np.flip(thresholded, 0))
	return int((len(thresholded) - right_margin -left_margin)/2)


@click.command()
@click.option('--input', default='pictures/', help='Input directory of images')
@click.option('--output', default='output_figures/', help='Output directory of images')
@click.option('--stage', default=3, help='Stage of processing to run to')
def main(input, output, stage):
    if os.path.exists(output):
        shutil.rmtree(output)
    os.mkdir(output)
    pictures = os.listdir(input)
    n = len(pictures)

    for i, image_name in enumerate(pictures):


        # Plotting
        fig, ax = plt.subplots(ncols = 4, figsize=(200, 50))
        plt.suptitle(image_name)

        print('image %i/%i' %(i + 1, n))
        # Opening the picture
        image_rgb = imread(input + image_name)

        ax[0].set_title('original image')
        ax[0].imshow(image_rgb)

        if stage >= 1:
            # Gray image
            image_gray = image_rgb[:, :, 0]
            thresh = threshold_otsu(image_gray, nbins = 60)
            binary = image_gray > thresh

            # Top of the ruler
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

            # Focusing on the ticks
            up_focus = up_rectangle + offset + 60
            left_focus = int(binary.shape[1]*0.1)
            right_focus = int(binary.shape[1]*0.9)
            height_focus = 200
            focus = ~binary[up_focus: up_focus + height_focus, left_focus: right_focus]
            sums = np.sum(focus, axis=0)/height_focus


            sums = np.sum(focus, axis=0)/float(height_focus)

            first_index = np.argmax(sums > 0.9)
            print(first_index)

            x = range(len(sums))


            # Fourier transformation
            fourier = np.fft.fft(sums)
            mod = [cmath.polar(el)[0] for el in fourier]
            freq = np.fft.fftfreq(len(sums))

            idx_max = np.argmax(mod[1:]) + 1
            f_space = freq[idx_max] # nb patterns per pixel
            T_space = 1/f_space
            print("TSPACE: " , T_space)
            ax[0].plot([left_focus + first_index - 18, left_focus + first_index + T_space - 18], [up_focus, up_focus],  color='red', linewidth=20, markersize=12)
            ax[0].plot([left_focus + first_index - 18, left_focus + first_index + T_space*10 - 18], [up_focus-30, up_focus-30],  color='blue', linewidth=20, markersize=12)

            # Butterfly binarization
            bfly_rgb = image_rgb[:up_rectangle - 60, :3500] # /!\ Magic number here
            bfly_hed = color.rgb2hsv(bfly_rgb)[:, :, 1]
            rescaled = rescale_intensity(bfly_hed, out_range=(0, 255))
            tresh = threshold_otsu(rescaled)
            bfly_bin = rescaled > thresh

            ax[1].set_title('binary')
            ax[1].imshow(bfly_bin)

            if stage >= 2:
                # Trying to remove antennas - TODO: this part kinda screws with it
                closed = binary_closing(bfly_bin, disk(3))
                half = split_picture(closed)

                divided = np.copy(closed)
                divided[:, half:half+5] = 0

                # dilated = bfly_bin
                # for i in range(10):
                #     eroded = binary_erosion(dilated, iterations = 7)
                #     dilated = binary_dilation(eroded, iterations=7)

                # Splitting the image in two
                # divided = np.copy(dilated)
                # half = int(divided.shape[1]/2)
                # divided[:, half] = 0

                # Detecting the wing regions
                markers_divided, _ = ndi.label(divided,
                                       structure=ndi.generate_binary_structure(2,1))
                regions = regionprops(markers_divided)
                areas = [region.area for region in regions]

                idx_1 = np.argmax(areas)
                coords1 = regions[idx_1].coords
                areas[idx_1] = 0
                idx_2 = np.argmax(areas)
                coords2 = regions[idx_2].coords

                # Determining which one is left or right
                if np.min(coords1[:, 1]) < np.min(coords2[:, 1]):
                    region_l, region_r = regions[idx_1], regions[idx_2]
                else:
                    region_l, region_r = regions[idx_2], regions[idx_1]

                # Smoothed boundaries
                boundary_l = boundary_tracing(region_l)
                boundary_r = boundary_tracing(region_r)
                descriptors_l = fourier_descriptors(boundary_l, 45)
                descriptors_r = fourier_descriptors(boundary_r, 45)
                smoothed_y_l, smoothed_x_l = inv_fourier(descriptors_l, 1500)
                smoothed_y_r, smoothed_x_r = inv_fourier(descriptors_r, 1500)

                # Detecting top of the junctions
                idx_in_l = detect_top_junction(smoothed_y_l, 'l')
                idx_in_r = detect_top_junction(smoothed_y_r, 'r')

                # Points of interest
                coords_l = region_l.coords
                coords_r = region_r.coords

                idx_out_l = np.argmin(coords_l[:, 0])
                pix_out_l = list(coords_l[idx_out_l])
                pix_in_l = [smoothed_y_l[idx_in_l], smoothed_x_l[idx_in_l]]

                idx_out_r = np.argmin(coords_r[:, 0])
                pix_out_r = list(coords_r[idx_out_r])
                pix_in_r = [smoothed_y_r[idx_in_r], smoothed_x_r[idx_in_r]]

                ax[2].set_title('eroded and Fourier filter')
                ax[2].imshow(divided)
                ax[2].scatter(smoothed_x_l, smoothed_y_l, color='b')
                ax[2].scatter(smoothed_x_r, smoothed_y_r, color='g')
                points_interest = np.array([pix_out_l, pix_in_l, pix_out_r, pix_in_r])
                print(points_interest)
                ax[2].scatter(points_interest[:, 1], points_interest[:, 0], color='r')

                if stage >= 3:

                    # Computing distances in pixels
                    dist_r_pix = np.sqrt((pix_out_r[0]-pix_in_r[0])**2 + (pix_out_r[1]-pix_in_r[1])**2)
                    dist_l_pix = np.sqrt((pix_out_l[0]-pix_in_l[0])**2 + (pix_out_l[1]-pix_in_l[1])**2)

                    # Converting to millimeters
                    dist_l_mm = dist_l_pix/(2*T_space)
                    dist_r_mm = dist_r_pix/(2*T_space)

                    ax[3].set_title('final image')
                    ax[3].imshow(image_rgb)
                    ax[3].plot([pix_out_l[1], pix_in_l[1]], [pix_out_l[0], pix_in_l[0]], color='r')
                    ax[3].plot([pix_out_r[1], pix_in_r[1]], [pix_out_r[0], pix_in_r[0]], color='r')
                    ax[3].text(int((pix_out_l[1] + pix_in_l[1])/2) +50,
                            int((pix_out_l[0] + pix_in_l[0])/2) - 50,
                            'dist_left = ' + str(round(dist_l_mm, 2)) +' mm',
                            color='r')
                    ax[3].text(int((pix_out_r[1] + pix_in_r[1])/2) +50,
                            int((pix_out_r[0] + pix_in_r[0])/2) + 50,
                            'dist_right = ' + str(round(dist_r_mm, 2)) +' mm',
                            color='r')

        plt.savefig(output + image_name)
        plt.close()

if __name__ == "__main__":
    main()
