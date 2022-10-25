<p align="center">
    <img src="./mothra_logo-text.PNG" width="300">
</p>

Mothra analyzes images of lepidopterans — mainly butterflies and moths — and measures their wing lengths. Using binarization techniques and calculating the resolution of ruler ticks, we read in images of lepidopterans and output the millimeter lengths of their wings.

![example output](examples/result-plot-BMNHE_500607.JPG)

## Usage

```
$ python pipeline.py -p -i [input directory or image path] -o [output directory] -csv [csv output file path]
```

The pipeline script combines four modules to analyze an image: ruler detection, binarization, tracing, and final measurement. These modules are located in `/mothra`. Python module requirements are listed in `requirements.txt`.

Run the `pipeline.py` file with the arguments to read input images, and output result images and a `.csv` file with the measurements.

The results are cached in `cachedir` so that if the same methods are re-run with the same inputs, the computation will simply be retrieved from memory instead of being recomputed. Delete `cachedir` to remove the cache and to recompute all results. If the source files for any part of the pipeline are tweaked, then results will be recomputed automatically.

## Parameters

The following parameters can be used as input arguments for `pipeline.py`:

- `-p`, `--plot` : This flag is used to generate output images. When this option is ommited the pipeline do not plot images, thus improving runtime and saving space.
- `-pp`, `--detailed_plot` : Outputs detailed plots to help debugging. Included in the detailed plot are the various points of interest of the image marked in seperate plots, as well as the method we are using to measure the pixels per millimeter on the ruler. This option can also be ommitted to improve runtime and save space. An example of the `-pp` option follows:

<p align="center">
    <img src="./examples/result-detailed_plot-BMNHE_500607.JPG" width="400">
</p>

- `-i`, `--input` : A single image input or a directory of images to be analyzed. (Default is `input_images`).
- `-o`, `--output_folder` : The output directory in which the result images will be outputted. (Default is `outputs`).
- `-s`, `--stage` : The stage which to run the pipeline until. Options are `'ruler_detection'`, `'binarization'`, and `'measurements'`. Default is `measurement` (running to completion). Running the pipeline and stopping at an earlier stage can be useful for debugging.
- `-ar`, `--auto_rotate` : Enable automatic rotation of input images, according to the information in the EXIF tag.
- `-csv`, `--path_csv` : Path of `.csv` file for the measurement results. (Default is `results.csv`).
- `-dpi` : Optional argument to specify resolution of the output image. (Default is `300`.)

## Measurement results

The file with measurement results (default: `results.csv`), returned by this pipeline, contains the following measurements:

- `image_id`, the filename of the processed image.
- `left_wing (mm)`, distance from the tip of the left wing to the left shoulder.
- `right_wing (mm)`, distance from the tip of the right wing to the right shoulder.
- `left_wing_center (mm)`, distance from the left wing to the center.
- `right_wing_center (mm)`, distance from the right wing to the center.
- `wing_span (mm)`, distance from the tip of the left wing to the tip of the right wing.
- `wing_shoulder (mm)`, distance from the left shoulder to the right shoulder.
- `position`, position of the lepidopteran being `right-side_up` or `upside_down`.
- `gender`, gender of the lepidopteran being `female` or `male`, returned if `position` is `right-side_up`.
- `prob_upside_down`, the probability of the prediction returned by the network to be `upside_down`.
- `prob_female`, the probability of the prediction returned by the network to be `female`.
- `prob_male`, the probability of the prediction returned by the network to be `male`.

## Example

Example data can be found at [github.com/machine-shop/mothra-data](https://github.com/machine-shop/mothra-data). For this example, clone the repository alongside the folder `mothra`.

```
git clone https://github.com/machine-shop/mothra.git
git clone https://github.com/machine-shop/mothra-data.git
```

Resulting files:

```
/mothra
    ...
    pipeline.py
    ...
/mothra-data
    image1.jpg
    image2.jpg
    ...
```

Running the command

```
$ python pipeline.py -p -i ../mothra-data -o ../test_output -csv ../test_output/results.csv
```

in `./mothra` will run the pipeline on the example data in the folder `/mothra-data`. The file locations should look like this:

```
/mothra
    ...
    pipeline.py
    ...
/mothra-data
    image1.jpg
    image2.jpg
    ...
/test_output
    image1.jpg
    image2.jpg
    ...
    results.csv
```

`/test_output` contains the resulting images, i.e. `/test_output/image1.jpg` contains the results of `/mothra-data/image1.jpg`, `/test_output/image2.jpg` has the results of `/mothra-data/image2.jpg`, and so on.

For a larger repository of test data (~15 GB), please check [Wilson et al. (2020)](https://doi.org/10.5281/zenodo.3732132).

## Miscellaneous

The testing suite can be run with `PYTHONPATH=. pytest` from `/mothra`.

# Result Plotting

`result_plotting.py` is a script that generates a histogram of differences between actual measurements and predicted measurements. This is useful for debugging and evaluating accuracy. This can be used in isolation from the main pipeline, and simply takes in the predicted `results.csv` from the pipeline and either an `.xlsx` file or `.csv` file with actual measurements.

## Usage

```
python result_plotting.py -a "h_comma_wing_lengths.xlsx" -n "full name" -l "Left" -r "Right"
```

Apart from outputting a plot of the differences, it can also output a `comparison.csv` with all differences between predicted and actual measurements, and/or a `outliers.csv` with only measurement differences from outliers. It can also copy outlier images to a `outliers/` folder, for easier debugging by rerunning the pipeline on these outlier images.

## Parameters

The following parameters can be used as input arguments for `result_plotting.py`:

- `-a`, `--actual` : File path of either Excel `.xlsx` file or `.csv` file containing the actual measurements.
- `-n`, `--name` : Name of column in `actual` file that contains the image names for each measurement.
- `-l`, `--left` : Name of column in `actual` file that contains the name of left wing measurements.
- `-r`, `--right` : Name of column in `actual` file that contains the name of right wing measurements.
- `-p`, `--predicted` : File path of `.csv` predictions outputted by this pipeline, `results.csv` by default.
- `-c`, `--comparison` : If specified, will output a `comparison.csv` file containing all measurements and the differences.
- `-o`, `--outliers` : If specified, will output a `outliers.csv` file containing only measurements that are deemed outliers.
- `-sd`, `--sd` : By default, the standard deviation to determine an outlier is ± 2 standard deviations away from the average measurement. If specified, you can use something else.
- `-co`, `--copy_outliers` : Specify a folder where the outlier images are from, and copy any outlier images to a `outliers/` folder in the current directory.

## Example

Example files:

```
/"Measured_images_Data_H.comma"
    BMNHE_1354218.JPG
    ...
    ...
/mothra
    result_plotting.py
    h_comma_wing_lengths.xlsx
    results.csv
```

`h_comma_wing_lengths.xlsx` has these columns:

| full name         | ... | Right | Left   |     |
| ----------------- | --- | ----- | ------ | --- |
| BMNHE_1354218.JPG | ... | 15.11 | 15.288 |     |
| ...               | ... | ...   | ...    |     |
|                   |     |       |        |     |

Thus, we will specify `-n "full name"`, `-l "Left"`, `-r "Right"`.

We know what `results.csv` looks like (since it is output by the pipeline).

We also want to output both the `comparison.csv` and `outliers.csv` table, so we will specify `-c` and `-o`, and we also want to copy outliers from the images folder `Measured_images_Data_H.comma`, so we will do `-co "../Measured_images_Data_H.comma"`. We also want to define an outlier as a measurement ± 1.75 standard deviations from the actual measurement, so we well specify `-sd 1.75`. This way, the usage is:

```
python result_plotting.py -a "h_comma_wing_lengths.xlsx" -n "full name" -l "Left" -r "Right" -c -o -co "../Measured_images_Data_H.comma" -sd 1.75
```

File results:

```
/mothra
    result_plotting.py
    h_comma_wing_lengths.xlsx
    results.csv
    ...
    result_plot.png
    comparison.csv
    outliers.csv
    /outliers
        BMNHE_1354218.JPG (an outlier image)
        ...
```

## Citing mothra

If you find mothra useful, please cite:

[Wilson, R. J. et al. (2022). Applying computer vision to digitised natural history collections for climate change research: Temperature-size responses in British butterflies. _Methods in Ecology and Evolution_, 1–13](https://doi.org/10.1111/2041-210X.13844).

Mothra's logo was designed by [Rogério Maroja](https://rogeriomaroja.myportfolio.com/mothra), and is available under the [CC-BY license](https://creativecommons.org/licenses/by/2.0/).
