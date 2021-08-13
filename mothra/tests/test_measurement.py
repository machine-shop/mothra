import matplotlib.pyplot as plt
from mothra import measurement

import pytest


@pytest.fixture(scope="module")
def fake_points_of_interest():
    points_interest = {
        "outer_pix_l": (0, -4),
        "inner_pix_l": (0, -1),
        "outer_pix_r": (0, 5),
        "inner_pix_r": (0, 1),
        "body_center": (0, 0)
    }
    dist_pix = {
        "dist_l": 3,
        "dist_r": 4,
        "dist_l_center": 4,
        "dist_r_center": 5,
        "dist_span": 9,
        "dist_shoulder": 2
    }
    dist_mm = {
        "dist_l": 1.5,
        "dist_r": 2,
        "dist_l_center": 2,
        "dist_r_center": 2.5,
        "dist_span": 4.5,
        "dist_shoulder": 1
    }
    return points_interest, dist_pix, dist_mm


def test_t_space_unchanged(fake_points_of_interest):
    # tests with simple t_space = 1 that everything remains the same.
    points_interest, true_results_pix, true_results_mm = fake_points_of_interest
    dist_pix, dist_mm = measurement.main(points_interest, 1)
    assert dist_pix == true_results_pix
    assert dist_mm == true_results_pix


def test_different_t_space(fake_points_of_interest):
    points_interest, true_results_pix, true_results_mm = fake_points_of_interest
    dist_pix, dist_mm = measurement.main(points_interest, 2)
    assert dist_pix == true_results_pix
    assert dist_mm == true_results_mm


def test_with_ax(fake_points_of_interest):
    fig, ax = plt.subplots(figsize=(20, 5))
    points_interest, true_results_pix, true_results_mm = fake_points_of_interest
    axes = [ax] + [None] * 6
    dist_pix, dist_mm = measurement.main(points_interest, 2)
    assert dist_pix == true_results_pix
    assert dist_mm == true_results_mm
