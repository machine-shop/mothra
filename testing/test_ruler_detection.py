import sys
sys.path.append("./../")
import ruler_detection as rd
import numpy as np
from numpy import testing as nt
import matplotlib.pyplot as plt


def test_grayscale():
    input = np.array([[[1, 1, 1],
              [2, 2, 2]],
             [[1, 1, 1],
              [2, 2, 2]],
             [[1, 1, 1],
              [1, 1, 1]]])
    print(input.shape)
    output = np.array([[False, True], [False, True], [False, False]])
    nt.assert_equal(rd.grayscale(input), output)


def test_fourier():
    seven_zeroes = np.zeros(7)
    seven_t = rd.fourier(seven_zeroes)
    nt.assert_equal(seven_t, 7)

    seven_zero_ones = np.array([0, 1, 0, 1, 0, 1, 0])
    seven_alt = rd.fourier(seven_zero_ones)
    nt.assert_equal(seven_alt, 7.0/3.0)

    arr256 = np.arange(256)
    x256 = rd.fourier(arr256)
    nt.assert_equal(x256, -256)

    arr100 = np.arange(5,100,3)
    x100 = rd.fourier(arr100)
    nt.assert_equal(x100, -32)

def test_main_ruler_detection():
    arr = np.loadtxt('data.txt', dtype=int)
    input_data = arr.reshape((815, 5184, 3))
    zeroes = np.full((2641, 5184, 3), 255.0)
    data = np.concatenate((zeroes, input_data), axis=0)
    t_space = rd.main(data)
    nt.assert_almost_equal(t_space, 94.25, decimal=0)
