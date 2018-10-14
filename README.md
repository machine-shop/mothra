# Butterflies-wings

The pipeline is set to process the 3 first images of the input folder, just to test it quickly. 
## Usage

```
$ python pipeline_argparse.py -s <stage> [OPTION]
```
## Description :
* -s, --stage : integer in [1, 2, 3, 4]
* -p, --plot : if entered, figures are plot to the output folder
* -r, --raw_image : path of the input folder (default: 'raw_images')
* -o, --output_folder : path of the output folder (default: 'outputs')
* -dpi : Dots per inch (default: 300)

