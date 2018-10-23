import sys
sys.path.append('../butterfly/')
import numpy as np
import tracing
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage.measure import regionprops


fake_butterfly = np.zeros((100, 200))
fake_butterfly[25:50, 25:50] = 1
fake_butterfly[25:50, -50:-25] = 1
fake_butterfly[50:90, 40:-40] = 1



fake_shape = np.zeros((10, 10))
fake_shape[2:8, 3:7] = 1
fake_shape[4:7, 4] = 0
fake_shape[4, 2] = 1
fake_shape[4, 3] = 0

def test_split():
    assert tracing.split_picture(fake_butterfly)==100
    
# def test_boundary_tracing():
    # labels = ndi.label(fake_butterfly)
    # plt.imshow(labels)
    # plt.show()
    # return 0

labels, _  = ndi.label(fake_shape, structure=ndi.generate_binary_structure(2, 1))
regions = regionprops(labels)
boundary = tracing.boundary_tracing(regions[0])
x = boundary[:, 1]
y = boundary[:, 0]
print(boundary)
plt.imshow(fake_shape)
plt.scatter(x, y)
plt.show()


# plt.imshow(fake_butterfly)
# plt.show()



