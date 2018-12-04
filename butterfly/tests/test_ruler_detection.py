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

    T_BIG = 8
    T_SMALL = T_BIG // 2

    signal = np.random.randint(0, 7, 500)
    signal[::T_SMALL] = 10
    signal[::T_BIG] = 20
    T_space = rd.fourier(signal)
    nt.assert_almost_equal(T_space, T_SMALL, decimal=0)

    signal = np.random.randint(0, 7, 500)
    signal[::T_BIG] = 20
    T_space = rd.fourier(signal)
    nt.assert_almost_equal(T_space, T_BIG, decimal=0)


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
