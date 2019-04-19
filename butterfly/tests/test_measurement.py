import matplotlib.pyplot as plt
from butterfly import measurement

import pytest


@pytest.fixture(scope="module")
def fake_points_of_interest():
    pixel_points = [(0, -4), (0, -1), (0, 5), (0, 1), (0, 0)]
    true_results_pix = (3, 4, 4, 5, 9)
    true_results_mm = (1.5, 2, 2, 2.5, 4.5)
    return pixel_points, true_results_pix, true_results_mm


def test_t_space_unchanged(fake_points_of_interest):
    # tests with simple t_space = 1 that everything remains the same.
    pixel_points, true_results_pix, true_results_mm = fake_points_of_interest
    dst_pix, dst_mm = measurement.main(pixel_points, 1)
    assert (dst_pix, dst_mm) == (true_results_pix, true_results_pix)


def test_different_t_space(fake_points_of_interest):
    pixel_points, true_results_pix, true_results_mm = fake_points_of_interest
    dst_pix, dst_mm = measurement.main(pixel_points, 2)
    assert (dst_pix, dst_mm) == (true_results_pix, true_results_mm)


def test_with_ax(fake_points_of_interest):
    fig, ax = plt.subplots(figsize=(20, 5))
    pixel_points, true_results_pix, true_results_mm = fake_points_of_interest
    axes = [ax] + [None] * 6
    dst_pix, dst_mm = measurement.main(pixel_points, 2)
    assert (dst_pix, dst_mm) == (true_results_pix, true_results_mm)
