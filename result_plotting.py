import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='Script to plot differences between actual lepidopteran measurements versus predicted measurements.')
parser.add_argument('-a', '--actual',
                    type=str,
                    help='File name with actual measurements',
                    required=True)
parser.add_argument('-n', '--name',
                    type=str,
                    help='Name of column containing file name of each measured image',
                    required=True)
parser.add_argument('-l', '--left',
                    type=str,
                    help='Name of column containing left wing measurements in actual measurements file',
                    required=True)
parser.add_argument('-r', '--right',
                    type=str,
                    help='Name of column containing right wing measurements in actual measurements file',
                    required=True)
parser.add_argument('-p', '--predicted',
                    type=str,
                    help='File name with predicted measurement results (assumed to be a csv produced by our pipeline)',
                    default='results.csv')
parser.add_argument('-c', '--comparison',
                    action='store_true',
                    help='Produce an "comparison.csv" file containing all measurements and the differences')
parser.add_argument('-o', '--outliers',
                    action='store_true',
                    help='Produce an "outliers.csv" file containing only measurements that are deemed outliers')
parser.add_argument('-sd', '--sd',
                    type=float,
                    help="The number of SD's that are used as a theshold for classifying an outlier",
                    default=2)
parser.add_argument('-co', '--copy_outliers',
                    type=str,
                    help='Create a folder "outliers" and copy images from the specified path that correspond to outliers')
args = parser.parse_args()


# Reading in both actual measurements file, and select desired columns
actual_file, actual_ext = os.path.splitext(args.actual)
actual = pd.DataFrame()
if actual_ext.lower() == ".xlsx":
    actual = pd.read_excel(args.actual)
elif actual_ext.lower() == ".csv":
    actual = pd.read_csv(args.actual)
actual = actual[[args.name, args.left, args.right]]
actual.rename(columns={args.name: "image_id", args.left: "actual_left", args.right: "actual_right"}, inplace=True)


# Reading in predicted results from csv
predicted = pd.read_csv(args.predicted)
predicted.rename(columns={"left_wing (mm)": "predicted_left", "right_wing (mm)": "predicted_right"}, inplace=True)


# Merging them together and creating new columns for the difference
both = pd.merge(actual, predicted, on="image_id", how='inner')
both['left_diff'] = both['predicted_left'] - both['actual_left']
both['right_diff'] = both['predicted_right'] - both['actual_right']

all_diffs = both['right_diff'].append(both['left_diff'])
mean = np.mean(all_diffs)
sd = np.std(all_diffs)

both['left_SD'] = (both['left_diff'] - mean)/sd
both['right_SD'] = (both['right_diff'] - mean)/sd
both['is_outlier'] = (abs(both['left_SD'])>args.sd) | (abs(both['right_SD'])>args.sd)


# Print statistics about differences
lower = mean - args.sd * sd
upper = mean + args.sd * sd
num_outlier_measurements = len(all_diffs[(all_diffs < lower) | (all_diffs > upper)])
num_outlier_images = np.count_nonzero(both['is_outlier'])
print("DIFFERENCE STATISTICS")
print(f"    Mean Differences: {mean}")
print(f"    Differences SD: {sd}.")
print(f"    Lower Bound (-{args.sd} SD) of Differences: {lower}")
print(f"    Upper Bound (+{args.sd} SD) of Differences: {upper}")
print(f"    Number of outlying measurements: {num_outlier_measurements}")
print(f"    Number of images with outlying measurements: {num_outlier_images}")
print("")


# Plot histogram of differences
all_diffs_nonoutlier = all_diffs[(all_diffs >= lower) & (all_diffs <= upper)]
fig, ax = plt.subplots(figsize=(10, 5))
ax = all_diffs_nonoutlier.hist(bins='auto')


# Saving the plot
filename = 'result_plot.png'
output_path = os.path.normpath(filename)
plt.xlabel('Difference between (predicted - actual) in mm')
start, end = ax.get_xlim()
plt.ylabel('Number of images')
plt.title('Error in predicted measurements')
plt.savefig(output_path)
plt.close()
print(f"Saved plot of differences to {filename}")


# Printing either full comparison csv or outliers csv
both['SD_sum'] = abs(both['left_SD']) + abs(both['left_SD'])
both.sort_values('SD_sum', ascending=False, inplace=True)
both.drop('SD_sum', axis=1, inplace=True)
both.sort_values('is_outlier', ascending=False, inplace=True, kind='mergesort')

if args.comparison:
    comparison_filename = 'comparison.csv'
    outlier_col_str = both["is_outlier"].replace({True:"TRUE", False:""})
    both_outlier_col_str = both.copy()
    both_outlier_col_str["is_outlier"] = outlier_col_str
    both_outlier_col_str.to_csv(comparison_filename)
    print(f"Saved all differences to {comparison_filename}")

if args.outliers:
    outliers_filename = 'outliers.csv'
    both_outliers_only = both[both['is_outlier']].copy()
    both_outliers_only.drop('is_outlier', axis=1, inplace=True)
    both_outliers_only.to_csv(outliers_filename)
    print(f'Saved {num_outlier_images} rows to {outliers_filename}')

# Fetching outlier images
if args.copy_outliers:
    outliers_folder = 'outliers/'
    if os.path.exists(outliers_folder):
        oldList = os.listdir(outliers_folder)
        for oldFile in oldList:
            os.remove(os.path.join(outliers_folder, oldFile))
    else:
        os.mkdir(outliers_folder)

    image_list = both[both['is_outlier']]['image_id']
    print(f'Copying {len(image_list)} outlier images to {outliers_folder} ...', end="")

    for image_name in image_list:
        image_path = os.path.join(args.copy_outliers, image_name)
        shutil.copy(image_path, outliers_folder)

    print("done")
