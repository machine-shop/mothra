# Butterflies-wings

The pipeline is set to process the 10 first images of the input folder, just to test it quickly. 
## Usage

```
$ python pipeline.py -s <stage> [OPTION]
```
## Description :
* -s, --stage : string in ['ruler_detection', 'binarization', 'tracing', 'measurements']
* -p, --plot : if entered, figures are plot to the output folder
* -r, --raw_image : path of the input folder (default: 'raw_images')
* -o, --output_folder : path of the output folder (default: 'outputs')
* -dpi : Dots per inch (default: 300)

