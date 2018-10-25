import sys
sys.path.append('../butterfly/')

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import binarization

# constructing fake picture
binary = np.zeros((300, 400))
binary[230:] = 1 # ruler
binary[30:100, 250:380] = 1 # tag 1
binary[120:200, 260:370] = 1 # tag 2

# "butterfly"
bfly_upper, bfly_lower = 50, 180
bfly_height = bfly_lower - bfly_upper
bfly_left, bfly_right = 50, 220
bfly_width = bfly_right - bfly_left
for row in range(bfly_upper, bfly_lower):
    for col in range(bfly_left, bfly_right):
        bfly_row, bfly_col = row-bfly_upper, col-bfly_left
        if ((bfly_width/2) / bfly_height * bfly_row) < bfly_col < (bfly_width - (bfly_width/2) / bfly_height * bfly_row):
            binary[row, col] = 1

# plt.imshow(binary)
# plt.show(block=True)

def test_find_tags_edge():
    result = binarization.find_tags_edge(binary, 230)
    assert(250 <= result <= 260) # assert the tags edge is in a proper place

def test_main():
    picture_2d = binary.astype(np.uint8)
    picture_3d = np.dstack((picture_2d, 1/2 * picture_2d, 1/4 * picture_2d)) # fake RGB image
    result = binarization.main(picture_3d, 230)
    y_where, x_where = np.where(result)
    x_where_min, x_where_max = np.min(x_where), np.max(x_where)
    y_where_min, y_where_max = np.min(y_where), np.max(y_where)
    assert((bfly_height - 5 <= y_where_max-y_where_min <= bfly_height + 5) and (bfly_width - 5 <= x_where_max-x_where_min <= bfly_width + 5) ) # assert the "butterfly" is included in the cropped and binarized result

