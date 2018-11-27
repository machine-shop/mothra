import pandas as pd
import os
import matplotlib.pyplot as plt


THRESH = 9
OUTLIERS = 2

# Reads in the files and parses out only the useful columns
actual = pd.read_excel('actual.xlsx')
predicted = pd.read_csv('predicted.csv')
actual = actual[['full name', 'Right', 'Left']]

# Takes all rows that need to be doubled and doubles them into a new dataframe
mod_pred = predicted
doubles = mod_pred.loc[(mod_pred['left_wing (mm)'] < THRESH) &
                       (mod_pred['right_wing (mm)'] < THRESH)]
doubles['left_wing (mm)'] = doubles['left_wing (mm)'] * 2
doubles['right_wing (mm)'] = doubles['right_wing (mm)'] * 2

# Keeping only the previous rows where either value doesn't need to be doubled
mod_pred = mod_pred[(mod_pred['left_wing (mm)'] > THRESH) |
                    (mod_pred['right_wing (mm)'] > THRESH)]
new_pred = pd.concat([mod_pred, doubles])

# Merging them together and creating new columns for the difference between predicted an
og = pd.merge(actual, mod_pred, left_on = 'full name', right_on = 'image_id').drop(['image_id'], axis=1)
both = og
both['left_diff'] = both['Left'] - both['left_wing (mm)']
both['right_diff'] = both['Right'] - both['right_wing (mm)']
all_diffs = both['right_diff'].append(both['left_diff'])

# Outliers
outliers = all_diffs[(all_diffs < -OUTLIERS) | (all_diffs > OUTLIERS)]
all_diffs = all_diffs[(all_diffs > -OUTLIERS) & (all_diffs < OUTLIERS)]

fig, ax = plt.subplots(figsize=(5, 5))
ax = all_diffs.hist()

filename = 'result_plot.png'
output_path = os.path.normpath(filename)
plt.xlabel('Difference between (actual - predicted) in mm')
plt.ylabel('Number of samples')
plt.title('Error in predicted length')
plt.savefig(output_path)
plt.close()
