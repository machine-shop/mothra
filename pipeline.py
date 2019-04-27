import os
import argparse
import csv
from butterfly import (ruler_detection, tracing, measurement, binarization)
import matplotlib.pyplot as plt
from skimage.io import imread

WSPACE_SUBPLOTS = 0.7

"""
Example :
    $ python pipeline_argparse.py --stage tracing --plot -dpi 400
"""


def create_layout(n_stages, plot_level):
    """Creates Axes to plot figures

    Parameters
    ----------
    n_stages : int
        length of pipeline process
    plot_level : int
        0 : no plotting
        1 : regular plots
        2 : detailed plots

    Returns
    -------
    axes : list of Axes
    """
    if plot_level == 0:
        return None

    elif plot_level == 1:
        ncols = n_stages
        fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(12, 5))
        if n_stages == 1:
            ax = [ax]
        ax_list = []
        for ax in ax:
            ax_list.append(ax)
        return ax_list + [None] * (7 - n_stages)

    elif plot_level == 2:
        shape = (3, 3)
        ax_main = plt.subplot2grid(shape, (0, 0))
        ax_structure = plt.subplot2grid(shape, (0, 1))
        ax_signal = plt.subplot2grid(shape, (1, 0), colspan=2)
        ax_fourier = plt.subplot2grid(shape, (2, 0), colspan=2)

        ax_tags = plt.subplot2grid(shape, (0, 2))
        ax_bin = plt.subplot2grid(shape, (1, 2))
        ax_poi = plt.subplot2grid(shape, (2, 2))
        plt.tight_layout()
        if n_stages == 1:
            return [ax_main, None, None, ax_structure, ax_signal, ax_fourier,
                    None]
        elif n_stages == 2:
            return [ax_main, ax_bin, None, ax_structure, ax_signal, ax_fourier,
                    ax_tags]
        elif n_stages == 3:
            return [ax_main, ax_bin, ax_poi, ax_structure, ax_signal,
                    ax_fourier, ax_tags]


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

    parser.add_argument('-pp', '--detailed_plot',
                        action='store_true',
                        help='If entered detailed images are plotted to the\
                        output folder')

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
                        required=False,
                        default='measurements')
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

    stages = ['ruler_detection', 'binarization', 'measurements']

    if args.stage not in stages:
        print("ERROR : Stage can only be 'ruler_detection', 'binarization',\
               or 'measurements'")
        return 0

    plot_level = 0
    if args.plot:
        plot_level = 1
    if args.detailed_plot:
        plot_level = 2

    # Initializing the csv file
    if args.stage == 'measurements':
        if os.path.exists(args.path_csv):
            os.remove(args.path_csv)
        with open(args.path_csv, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['image_id', 'left_wing (mm)', 'right_wing (mm)',
                'left_wing_center (mm)', 'right_wing_center (mm)', 'wing_span (mm)'])

    stage_idx = stages.index(args.stage)
    pipeline_process = stages[:stage_idx + 1]

    raw_image_path = args.input
    if(os.path.isdir(raw_image_path)):
        image_names = os.listdir(raw_image_path)
        image_paths = []
        for image_name in image_names:
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            image_path = os.path.join(raw_image_path, image_name)
            image_paths.append(image_path)
    else:
        image_paths = [raw_image_path]
    n = len(image_paths)

    for i, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path)
        print(f'Image {i+1}/{n} : {image_name}')

        image_rgb = imread(image_path)
        axes = create_layout(len(pipeline_process), plot_level)

        for step in pipeline_process:
            if step == 'ruler_detection':
                T_space, top_ruler = ruler_detection.main(image_rgb, axes)

            elif step == 'binarization':
                binary = binarization.main(image_rgb, top_ruler, axes)

            elif step == 'measurements':
                points_interest = tracing.main(binary, axes)
                dist_pix, dist_mm = measurement.main(points_interest, T_space,
                                                   axes)

                with open(args.path_csv, 'a') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([image_name, dist_mm["dist_l"], dist_mm["dist_r"], 
                                    dist_mm["dist_l_center"], dist_mm["dist_r_center"], 
                                    dist_mm["dist_span"]])

        if plot_level > 0:
            output_path = os.path.normpath(os.path.join(args.output_folder, image_name))
            dpi = args.dpi
            if plot_level == 2:
                dpi = int(1.5 * args.dpi)
            plt.savefig(output_path, dpi=dpi)
            plt.close()


if __name__ == "__main__":
    main()
