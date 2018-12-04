import os
import argparse
import csv
from butterfly import (ruler_detection, tracing, measurement, binarization)
import matplotlib.pyplot as plt
from skimage.io import imread


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
                T_space, top_ruler = ruler_detection.main(image_rgb, ax0)
            elif step == 'binarization':
                binary = binarization.main(image_rgb, top_ruler, ax[1])
            elif step == 'tracing':
                points_interest = tracing.main(binary, ax[2])
            else:
                dst_pix, dst_mm = measurement.main(points_interest,
                                                   T_space,
                                                   ax[0])
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
