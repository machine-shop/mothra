# Butterflies-wings

The pipeline takes either a single image file or a folder of images and runs the pipeline up to the specified stage on the given images. Can output all images into an output folder as well as all measurements into a CSV file. 
## Usage

```
$ python pipeline.py -s <stage> [OPTION]
```
## Options :
* -s, --stage : string in ['ruler_detection', 'binarization', 'tracing', 'measurements']
* -p, --plot : if entered, figures are plot to the output folder
* -i, --input : path of the input folder or file (default: 'raw_images')
* -o, --output_folder : path of the output folder (default: 'outputs')
* -dpi : Dots per inch (default: 300)
* -csv, --path_csv :  path of CSV file that contains results of measurements 

Example:
```
$ python pipeline.py -i test_pictures -o out -p -s measurements
```
## Accuracy: 
Run the ```result_plotting.py``` file to print out to generate a histogram of the difference for all actual - predicted measurements. 
