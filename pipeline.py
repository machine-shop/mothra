import os
import argparse
from butterfly import (ruler_detection, tracing, measurement, binarization)
import matplotlib.pyplot as plt
from skimage.io import imread
import shutil

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
                        help='If entered images are plotted to the output folder')

    # Input path
    parser.add_argument('-i', '--input',
                        type=str,
                        help='Input path for raw images folder or single image',
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

    args = parser.parse_args()

    # Initialization
    if os.path.exists(args.output_folder):
        oldList = os.listdir(args.output_folder)
        for oldFile in oldList:
            os.remove(args.output_folder+"/"+oldFile)
    else:
        os.mkdir(args.output_folder)

    stages = ['ruler_detection', 'binarization', 'tracing', 'measurements']

    if not args.stage in stages:
        print("ERROR : Stage can only be 'ruler_detection', 'binarization', 'tracing' or 'measurements'")
        return 0
    raw_image_path = args.input

    stage_idx = stages.index(args.stage)
    pipeline_process = stages[:stage_idx + 1]

    raw_image_path = args.input
    if(os.path.isdir(raw_image_path)):
        image_names = os.listdir(raw_image_path)
    else:
        image_names = [""]

    # For testing purpose, the pipeline is only applied to the first 10 images
    for image_name in image_names[:10]:
        print(raw_image_path + '/' + image_name)
        image_path = os.path.normpath(raw_image_path + '/' + image_name)
        image_rgb = imread(image_path)
        ax = [None, None, None]
        if args.plot:
            ncols = min(len(pipeline_process), 3)
            fig, ax = plt.subplots(ncols = ncols, figsize=(20, 5))

        for step in pipeline_process:
            if step == 'ruler_detection':
                ax0 = ax
                if len(pipeline_process) > 1:
                    ax0 = ax[0]
                T_space, top_ruler  = ruler_detection.main(image_rgb, ax0)
            elif step == 'binarization':  
                binary = binarization.main(image_rgb, top_ruler, ax[1]) 
            elif step == 'tracing':
                points_interest = tracing.main(binary, ax[2])
            else :
               dst_pix, dst_mm = measurement.main(points_interest, T_space, ax[0])

        if args.plot:
            output_path = os.path.normpath(args.output_folder + '/' + image_name)
            plt.savefig(output_path, dpi=args.dpi)
            plt.close()

if __name__ == "__main__":
    main()
