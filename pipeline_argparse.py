import os
import argparse
import ruler_detection
import matplotlib.pyplot as plt
import tracing
import measurement
import binarization
from skimage.io import imread
import shutil

"""
Docstring
"""
def main():
     # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description='Description here')
    # Add arguments
    # Plotting
    parser.add_argument('-p', '--plot', 
        action='store_true',
        help='If entered images are plotted to the output folder')
    # Input path
    parser.add_argument('-r', '--raw_image', 
        type=str, 
        help='The input path for raw images', 
        required=False, 
        default='raw_images')
    # Output path
    parser.add_argument('-o', '--output_folder', 
        type=str, 
        help='The input path for raw image', 
        required=False, 
        default='outputs')
    # Stage
    parser.add_argument('-s', '--stage', 
        type=int,
        help='Stage number', 
        required=True)
    # Dots per inch
    parser.add_argument('-dpi',  
        type=int,
        help='Dots per pixel of the saved figures', 
        default=300)

    args = parser.parse_args()

    # Initialization
    if os.path.exists(args.output_folder):
        shutil.rmtree(args.output_folder)
    os.mkdir(args.output_folder)


    if not args.stage in [1, 2, 3, 4]:
        print('ERROR : Stage can only be 1, 2, 3 or 4')
        return 0
    raw_image_path = args.raw_image
    
    stages = ['ruler_detection', 'binarization', 'tracing', 'measurement']
    pipeline_process = stages[:args.stage]

    image_names = os.listdir(raw_image_path)
    ax = [None, None, None]
    if args.plot:
        ncols = min(len(pipeline_process), 3)
        fig, ax = plt.subplots(ncols = ncols, figsize=(20, 5))

    # For testing purpose, the pipeline is only applied to the first 3 images
    for image_name in image_names[:3]:
        print(image_name)
        image_path = os.path.normpath(raw_image_path + '/' + image_name)
        image_rgb = imread(image_path)
        for step in pipeline_process:
            if step == 'ruler_detection': 
                ax0 = ax
                if len(pipeline_process) > 1:
                    ax0 = ax[0]
                T_space  = ruler_detection.main(image_rgb, ax0)
            elif step == 'binarization':  
                binary = binarization.main(image_rgb, ax[1]) 
            elif step == 'tracing':
                points_interest = tracing.main(binary, ax[2])
            else :
               dst_pix, dst_mm = measurement.main(points_interest, T_space, ax[0]) 

        if args.plot:
            fig.show()
            output_path = os.path.normpath(args.output_folder + '/' + image_name)
            plt.savefig(output_path, dpi=args.dpi)

if __name__ == "__main__":
    main()
