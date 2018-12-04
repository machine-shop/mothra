# Butterfly Wings

Analyzing images of butterflies and measuring their wing lengths. Using binarization techniques and calculating the resolution of ruler ticks, we read in images of butterflies and output the millimeter lengths of their wings.

![example output](example_result.JPG)

## Usage
```
$ python pipeline.py -p -i [input directory or image path] -o [output directory] -s [stage to run to] -csv [csv output file path]
```
The pipeline script combines four modules to analyze an image: ruler detection, binarization, tracing, and final measurement. Run the pipeline.py file with the arguments to read in raw images and output result images and `.csv` file with the measurements.

## Options :
* -s, --stage : The stage which to run the pipeline until. Must be one of  `['ruler_detection', 'binarization', 'tracing', 'measurements']`. To run the entire pipeline, simply use `-s measurement`. (Running the pipeline and stopping at an earlier stage can be useful for debugging.)
* -p, --plot : This flag is used to generate output images. (Image outputs can be ommitted to improve runtime or save space.)
* -i, --input : A single image input or a directory of images to be analyzed.
* -o, --output_folder : The output directory in which the result images will be outputted.
* -dpi : The resolution of the output image. (Default is `300`.)
* -csv, --path_csv :  Path of `.csv` file for the measurement results.

Example:
```
$ python pipeline.py -p -i ../data/test_pictures -o ../data/test_output -s measurements -csv ../data/test_output/results.csv
```
## Accuracy: 
Run the ```result_plotting.py``` file to print out to generate a histogram of the difference for all actual - predicted measurements. 
