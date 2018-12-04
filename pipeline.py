import os
import argparse
import csv
from butterfly import (ruler_detection, tracing, measurement, binarization)
import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np


LINE_WIDTH = 40


"""
Example :
    $ python pipeline_argparse.py --stage tracing --plot -dpi 400
"""


def main():
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description='Script to automate butterfly wings measurment')
    # Add arguments
    # Plotting
    parser.add_argument('-p', '--plot',
                        action='store_true',
                        help='If entered images are plotted to the output\
                        folder')

    # Input path
    parser.add_argument('-i', '--input',
                        type=str,
                        help='Input path for folder or single image',
                        required=False,
                        default='raw_images')

    # Output path
    parser.add_argument('-o', '--output_folder',
                        type=str,
                        help='Output path for raw image',
                        required=False,
                        default='outputs')
    # Stage
    parser.add_argument('-s', '--stage',
                        type=str,
                        help="Stage name: 'ruler_detection', 'binarization',\
                        'tracing', 'measurements",
                        required=True)
    # Dots per inch
    parser.add_argument('-dpi',
                        type=int,
                        help='Dots per inch of the saved figures',
                        default=300)
    parser.add_argument('-csv', '--path_csv',
                        type=str,
                        help='Path of the resulting csv file',
                        default='results.csv')

    args = parser.parse_args()

    # Initialization
    if os.path.exists(args.output_folder):
        oldList = os.listdir(args.output_folder)
        for oldFile in oldList:
            os.remove(args.output_folder+"/"+oldFile)
    else:
        os.mkdir(args.output_folder)

    stages = ['ruler_detection', 'binarization', 'tracing', 'measurements']

    if args.stage not in stages:
        print("ERROR : Stage can only be 'ruler_detection', 'binarization',\
              'tracing' or 'measurements'")
        return 0

    # Initializing the csv file
    if args.stage == 'measurements':
        if os.path.exists(args.path_csv):
            os.remove(args.path_csv)
        with open(args.path_csv, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['image_id', 'left_wing (mm)', 'right_wing (mm)'])

    raw_image_path = args.input

    stage_idx = stages.index(args.stage)
    pipeline_process = stages[:stage_idx + 1]

    raw_image_path = args.input
    if(os.path.isdir(raw_image_path)):
        image_names = os.listdir(raw_image_path)
    else:
        image_names = [""]
    n = len(image_names)

    
    for i, image_name in enumerate(image_names):

        # if input is not an image, continue
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        print(f'Image {i+1}/{n} : {image_name}')
        image_path = os.path.normpath(raw_image_path + '/' + image_name)
        image_rgb = imread(image_path)
        ax = [None, None, None]
        if args.plot:
            ncols = min(len(pipeline_process), 3)
            fig, ax = plt.subplots(ncols=ncols, figsize=(20, 5))

        for step in pipeline_process:
            if step == 'ruler_detection':
                ax0 = ax
                if len(pipeline_process) > 1:
                    ax0 = ax[0]
                T_space, top_ruler, plot_info = ruler_detection.main(image_rgb)
                if ax0:
                    x_single = plot_info[0]
                    y = plot_info[1]
                    x_mult = plot_info[2]
                    plt_img = plot_info[3]
                    ax0.imshow(plt_img)
                    ax0.fill_between(x_single, y, y + LINE_WIDTH, color='red')
                    ax0.fill_between(x_mult, y - LINE_WIDTH, y, color='blue')
            elif step == 'binarization':
                binary = binarization.main(image_rgb, top_ruler)
                if(ax[1]):
                    ax[1].set_title('Binary')
                    ax[1].imshow(binary)
            elif step == 'tracing':
                points_interest, plot_info = tracing.main(binary)
                if ax[2]:
                    without_antennae = plot_info[0]
                    middle = plot_info[1]
                    binary = plot_info[2]
                    ax[2].set_title('Tracing')
                    ax[2].imshow(without_antennae)
                    ax[2].plot([middle, middle], [0, binary.shape[0]-5])
                    ax[2].scatter(points_interest[:, 1],
                                  points_interest[:, 0], color='r', s=10)
            else:
                dst_pix, dst_mm, plot_info = measurement.main(points_interest,
                                                              T_space)
                if ax0 is not None:
                    pix_out_l = plot_info[0]
                    pix_out_r = plot_info[1]
                    pix_in_l = plot_info[2]
                    pix_in_r = plot_info[3]
                    dist_l_mm = plot_info[4]
                    dist_r_mm = plot_info[5]
                    ax0.set_title('final image')
                    ax0.plot([pix_out_l[1], pix_in_l[1]],
                             [pix_out_l[0], pix_in_l[0]], color='r')
                    ax0.plot([pix_out_r[1], pix_in_r[1]],
                             [pix_out_r[0], pix_in_r[0]], color='r')
                    ax0.text(int((pix_out_l[1] + pix_in_l[1]) / 2) + 50,
                             int((pix_out_l[0] + pix_in_l[0]) / 2) - 50,
                             'dist_left = ' + str(round(dist_l_mm, 2)) + ' mm',
                             color='r')
                    ax0.text(int((pix_out_r[1] + pix_in_r[1]) / 2) + 50,
                             int((pix_out_r[0] + pix_in_r[0]) / 2) + 50,
                             'dist_right = ' + str(round(dist_r_mm, 2))
                             + ' mm', color='r')
                print(f'left_wing : {dst_mm[0]} mm')
                print(f'right_wing : {dst_mm[1]} mm')

                with open(args.path_csv, 'a') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([image_name, dst_mm[0], dst_mm[1]])

        if args.plot:
            filename = args.output_folder + '/' + image_name
            output_path = os.path.normpath(filename)
            plt.savefig(output_path, dpi=args.dpi)
            plt.close()


if __name__ == "__main__":
    main()
