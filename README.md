# Butterfly Wings

Analyzing images of butterflies and measuring their wing lengths. Using binarization techniques and calculating the resolution of ruler ticks, we read in images of butterflies and output the millimeter lengths of their wings.

![example output](example_result.JPG)

## Usage
```
$ python pipeline.py -p -i [input directory or image path] -o [output directory] -s [stage to run to] -csv [csv output file path]
```
The pipeline script combines four modules to analyze an image: ruler detection, binarization, tracing, and final measurement. These modules are located in `/butterfly`. Python module requirements are listed in `requirements.txt`.

Run the `pipeline.py` file with the arguments to read in raw images and output result images and `.csv` file with the measurements.

The results are cached in `cachedir` so that if the same methods are re-run with the same inputs, the computation will simply be retrieved from memory instead of being recomputed. Delete `cachedir` to remove the cache and to recompute all results. If the source files for any part of the pipeline are tweaked, then results will be recomputed automatically.

## Options
* -p, --plot : This flag is used to generate output images. This can be ommitted to not plot any images (Image outputs can be ommitted to improve runtime or save space.)
* -pp, --detailed_plot : Outputs detailed plots to help debugging. Included in the detailed plot are the various points of interest of the image marked in seperate plots, as well as the method we are using to measure the pixels per millimeter on the ruler (Again, can be ommitted to improve runtime and save space). Example detailed result: 
<p align="center">
    <img src="./example_result_detailed.JPG" width="400">
</p>

* -i, --input : A single image input or a directory of images to be analyzed. (Default is `raw_images`).
* -o, --output_folder : The output directory in which the result images will be outputted. (Default is `outputs`).
* -s, --stage : The stage which to run the pipeline until. Must be one of  `['ruler_detection', 'binarization', 'measurements']`. To run the entire pipeline, simply use `-s measurements`. (Running the pipeline and stopping at an earlier stage can be useful for debugging.)
* -csv, --path_csv :  Path of `.csv` file for the measurement results. (Default is `results.csv`).
* -dpi : Optional argument to specify resolution of the output image. (Default is `300`.)

## Example
Example data can be found at [github.com/machine-shop/butterfly-wings-data](https://github.com/machine-shop/butterfly-wings-data). For this example, clone the repository alongside `butterfly-wings`.
```
git clone https://github.com/machine-shop/butterfly-wings.git
git clone https://github.com/machine-shop/butterfly-wings-data.git
```
Resulting files:
```
/butterfly-wings
    ...
    pipeline.py
    ...
/butterfly-wings-data
    image1.jpg
    image2.jpg
    ...
```

Running this command
```
$ python pipeline.py -p -i ../butterfly-wings-data -o ../test_output -s measurements -csv ../test_output/results.csv
```
in `/butterfly-wings` will run the pipeline on the example data in `/butterfly-wings-data`. The file locations should look like this:
```
/butterfly-wings
    ...
    pipeline.py
    ...
/butterfly-wings-data
    image1.jpg
    image2.jpg
    ...
/test_output
    image1.jpg (result image for image1.jpg)
    image2.jpg
    ...
    results.csv
```

## Miscellaneous

Jupyter notebooks with prototypes for various methods are located in the `/notebooks` directory.

The testing suite can be run with `PYTHONPATH=. pytest` from `/butterfly-wings`.

## Accuracy
Run the `result_plotting.py` file to print out to generate a histogram of the difference for all actual - predicted measurements. (Not parametarized, so you will need to edit the script for the correct paths to compare result `.csv`'s). Default expected locations are:
```
/butterfly-wings/results.csv (generated measurements)
/butterfly-wings/actual.xlsx (original measurements spreadsheet)
```
