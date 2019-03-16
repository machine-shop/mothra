import matplotlib.pyplot as plt
from butterfly import measurement


def test_t_space_unchanged():
    # tests with simple t_space = 1 that everything remains the same.
    pixel_points = [(0, 0), (5, 0), (0, 0), (0, 4)]
    dst_pix, dst_mm = measurement.main(pixel_points, 1)
    assert (dst_pix, dst_mm) == ((5, 4), (5, 4))


def test_different_t_space():
    pixel_points = [(0, 0), (5, 0), (0, 0), (0, 4)]
    dst_pix, dst_mm = measurement.main(pixel_points, 2)
    assert (dst_pix, dst_mm) == ((5, 4), (2.5, 2))


def test_with_ax():
    fig, ax = plt.subplots(figsize=(20, 5))
    pixel_points = [(0, 0), (5, 0), (0, 0), (0, 4)]
    axes = [ax] + [None] * 6
    dst_pix, dst_mm = measurement.main(pixel_points, 2)
    assert (dst_pix, dst_mm) == ((5, 4), (2.5, 2))
