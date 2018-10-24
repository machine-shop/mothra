import sys
sys.path.append('../butterfly/')
import numpy as np
import tracing
# import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage.measure import regionprops

def test_moore_neighborhood():
    array = np.array([[10, 9],
                      [9, 9], 
                      [9, 10], 
                      [9, 11], 
                      [10, 11],
                      [11, 11], 
                      [11, 10],
                      [11, 9]])
    assert np.all(tracing.moore_neighborhood([10, 10], [10, 9])==array)
fake_butterfly = np.zeros((100, 200))
fake_butterfly[25:50, 25:50] = 1 # left wing
fake_butterfly[25:50, 150:175] = 1 # right wing
fake_butterfly[50:90, 40:160] = 1 # body
fake_butterfly[25:50, 80:120] = 1 # head

fake_shape = np.zeros((5, 5))
fake_shape[1:, 4] = 1
fake_shape[2:4, 2] = 1
fake_shape[[1, 4], 3] = 1

def test_split():
    assert tracing.split_picture(fake_butterfly) in [99, 100]

def test_boundary_tracing():
    bound1 = np.array([[1, 3], [1, 4], [2, 4], [3, 4], [4, 4],
                       [4, 3], [3, 4], [2, 4]])
    bound2 = np.array([[1, 3], [1, 4], [2, 4], [3, 4], [4, 4],
                        [4, 3], [3, 2], [2, 2]])
    labels1, _  = ndi.label(fake_shape, structure=ndi.generate_binary_structure(2, 1))
    labels2, _  = ndi.label(fake_shape, structure=ndi.generate_binary_structure(2, 2))

    regions1 = regionprops(labels1)
    regions2 = regionprops(labels2)

    boundary1 = tracing.boundary_tracing(regions1[0])
    boundary2 = tracing.boundary_tracing(regions2[0])

    assert np.all(boundary1==bound1)
    assert np.all(boundary2==bound2)



def test_detect_points_interest():
    
    middle = tracing.split_picture(fake_butterfly)
    fake_butterfly[:, middle] = 0
    labels, _  = ndi.label(fake_butterfly, structure=ndi.generate_binary_structure(2, 2))
    regions = regionprops(labels)
    boundary_l = tracing.boundary_tracing(regions[0])
    boundary_r = tracing.boundary_tracing(regions[1])
    
    test_poi_l = np.array([[25, 25], [49, 80]])
    test_poi_r = np.array([[25, 174], [49, 119]])
    poi_l = tracing.detect_points_interest(boundary_l, 'l', 200)
    poi_r = tracing.detect_points_interest(boundary_r, 'r', 200)

    assert np.all(poi_r[0] == test_poi_r[0])
    assert np.all(poi_l[0] == test_poi_l[0])
    assert np.all(poi_r[1] == test_poi_r[1])
    assert np.all(poi_l[1] == test_poi_l[1])










