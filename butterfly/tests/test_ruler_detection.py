from butterfly import ruler_detection as rd
import numpy as np
from numpy import testing as nt


RULER_TOP = 50
RULER_LEFT = 0.2
RULER_RIGHT = 0.4


def test_grayscale():
    input = np.array([[[1, 1, 1],
                      [2, 2, 2]],
                     [[1, 1, 1],
                      [2, 2, 2]],
                     [[1, 1, 1],
                      [1, 1, 1]]])
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
    nt.assert_equal(x256, 256)

    arr100 = np.arange(5, 100, 3)
    x100 = rd.fourier(arr100)
    nt.assert_equal(x100, 32)


def test_binary_rect():
    data = np.full((500, 500, 3), 0.0)
    data[450:500, :, :] += 0.6
    expected = np.full((50, 100, 3), 0.6)
    nt.assert_equal(rd.binarize_rect(450, data), expected)


def test_main_ruler_detection():
    data = np.ones((500, 500, 3))
    data[350:500, 0:500:10, :] = 0
    t_space, top_space = rd.main(data, None)
    nt.assert_almost_equal(t_space, 10, decimal=0)
