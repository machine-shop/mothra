# Butterfly Wings

Analyzing images of butterflies and measuring their wing lengths. Using binarization techniques and calculating the resolution of ruler ticks, we read in images of butterflies and output the millimeter lengths of their wings.

![alt text](example_result.jpg?raw=true "Example measurement output")

## Usage

```
$ python pipeline.py -p -i [input directory or image path] -o [output directory] -s [stage to run to] -csv [csv output file path]
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
$ python pipeline.py -p -i ../data/test_pictures -o ../data/test_output -s measurements -csv ../data/test_output/results.csv
```
## Accuracy: 
Run the ```result_plotting.py``` file to print out to generate a histogram of the difference for all actual - predicted measurements. 
