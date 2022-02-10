#!/bin/env python

import os
import argparse
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import rotate
from skimage.util import img_as_ubyte
from mothra.misc import AlbumentationsTransform, label_func


WSPACE_SUBPLOTS = 0.7

"""
Example :
    $ python pipeline_argparse.py --stage tracing --plot -dpi 400
"""


def main():
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description='This is mothra, a software to automate different\
        measurements on images of Lepidopterae.')
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
                        help='Input path for single image, folder or text\
                        file (extension txt) containing paths',
                        required=False,
                        default='input_images')

    # Output path
    parser.add_argument('-o', '--output_folder',
                        type=str,
                        help='Output path for raw image',
                        required=False,
                        default='outputs')
    # Stage
    parser.add_argument('-s', '--stage',
                        type=str,
                        help="Stage name: 'binarization', 'ruler_detection',\
                        'measurements'",
                        required=False,
                        default='measurements')
    # Dots per inch
    parser.add_argument('-dpi',
                        type=int,
                        help='Dots per inch of the saved figures',
                        default=300)

    # CSV output path
    parser.add_argument('-csv', '--path_csv',
                        type=str,
                        help='Path of the resulting csv file',
                        default='outputs/results.csv')

    # Disable cache
    parser.add_argument('--cache',
                        action='store_true',
                        help='Enable computation cache (useful when developing algorithms)')

    args = parser.parse_args()

    stages = ['ruler_detection', 'binarization', 'measurements']

    if args.stage not in stages:
        print((f"* mothra expects stage to be 'ruler_detection', "
               f"binarization', or 'measurements'. Received '{args.stage}'"))
        return None

    plot_level = 0
    if args.plot:
        plot_level = 1
    if args.detailed_plot:
        plot_level = 2

    # Set up caching and import mothra modules
    if args.cache:
        from mothra import cache
        import joblib
        cache.memory = joblib.Memory('./cachedir', verbose=0)

    from mothra import (ruler_detection, tracing, measurement, binarization,
                        identification, misc, plotting, writing)

    # Initializing output folder
    misc.initialize_path(args.output_folder)

    stage_idx = stages.index(args.stage)
    pipeline_process = stages[:stage_idx + 1]

    # reading and processing input path.
    input_name = args.input
    image_paths = misc.process_paths_in_input(input_name)

    number_of_images = len(image_paths)

    # Initializing csv file
    if args.stage == 'measurements':
        writing.initialize_csv_file(csv_fname=args.path_csv)

    for i, image_path in enumerate(image_paths):
        try:
            # creating axes layout for plotting.
            axes = plotting.create_layout(len(pipeline_process), plot_level)

            image_name = os.path.basename(image_path)
            print(f'Image {i+1}/{number_of_images} : {image_name}')

            image_rgb = imread(image_path)

            # check image orientation and untilt it, if necessary.
            angle = misc.read_orientation(image_path)

            if angle not in (None, 0):  # angle == 0 does not need untilting
                image_rgb = img_as_ubyte(rotate(image_rgb, angle=angle, resize=True))

            for step in pipeline_process:
                # first, binarize the input image and return its components.
                _, ruler_bin, lepidop_bin = binarization.main(image_rgb, axes)

                if step == 'ruler_detection':
                    T_space, top_ruler = ruler_detection.main(image_rgb, ruler_bin, axes)

                elif step == 'binarization':
                    # already binarized in the beginning. Moving on...
                    pass

                elif step == 'measurements':
                    points_interest = tracing.main(lepidop_bin, axes)
                    _, dist_mm = measurement.main(points_interest,
                                                  T_space,
                                                  axes)
                    # measuring position and gender
                    position, gender, probabilities = identification.main(image_rgb)

                    with open(args.path_csv, 'a') as csv_file:
                        writing.write_csv_data(csv_file, image_name, dist_mm,
                                               position, gender,
                                               probabilities)

            if plot_level > 0:
                output_path = os.path.normpath(
                    os.path.join(args.output_folder, image_name)
                    )
                dpi = args.dpi
                if plot_level == 2:
                    dpi = int(1.5 * args.dpi)
                plt.savefig(output_path, dpi=dpi)
                plt.close()
        except Exception as exc:
            print(f"* Sorry, could not process {image_path}. More details:\n {exc}")
            continue


if __name__ == "__main__":
    main()
