import click
import matplotlib.pyplot as plt
import os
import shutil
from skimage.io import imread

# Our modules
import ruler_detection
import binarization
import tracing
import measurement

@click.command()
@click.option('--input', default='pictures/', help='Input directory of images')
@click.option('--output', default='output_figures/', help='Output directory of images')
@click.option('--stage', default=3, help='Stage of processing to run to')
def main(input, output, stage):
    
    if os.path.exists(output):
        shutil.rmtree(output)
    os.mkdir(output) # runs into access denied every other run
    pictures = os.listdir(input)
    n = len(pictures)

    for i, image_name in enumerate(pictures):

        fig, ax = plt.subplots(ncols = 3)
        plt.suptitle(image_name)

        print('image %i/%i: %s' %(i + 1, n, image_name))
        # Opening the picture
        image_rgb = imread(input + image_name)

        # Start of pipeline
        pixels_per_tick, ruler_subplot = ruler_detection.main(image_rgb)
        ax[0] = ruler_subplot
        if stage >= 1:
            binarized_image, binarization_subplot = binarization.main(image_rgb)
            ax[1] = binarization_subplot
            if stage >= 2:
                wing_coordinates, tracing_subplot = tracing.main(binarized_image)
                ax[2] = tracing_subplot
                if stage >= 3:
                    results = measurement.main(ruler_subplot, wing_coordinates, pixels_per_tick)

        plt.savefig(output + image_name)
        plt.close()

if __name__ == "__main__":
    main()