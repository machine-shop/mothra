import numpy as np
from joblib import Memory

from .cache import memory


@memory.cache(ignore=['axes'])
def main(points_interest, T_space, axes=None):
    ''' Calculates the length and draws the lines for length
    of the lepidopteran wings.

    Parameters
    ----------
    ax: array
        the array containing the 3 intermediary Axes.
    points_interest: dictionary
        dictionary containing the points of interest in the form [y, x],
        keyed with "outer_pix_l", "inner_pix_l", "outer_pix_r", "inner_pix_r",
        "body_center"
    T_space: float
        number of pixels between 2 ticks.

    Returns
    -------
    ax: ax
        an ax object
    dst_pix: dictionary
        dictionary containing measurements in pixels, keyed with
        "dist_l", "dist_r", "dist_l_center",
        "dist_r_center", "dist_span", "dist_shoulder"
    dst_mm: tuple
        dictionary containing measurements in mm, keyed with
        the same keys as dst_pix

    '''

    # Extract points of interest
    pix_out_l, pix_out_r = np.array(points_interest["outer_pix_l"]), np.array(points_interest["outer_pix_r"])
    pix_in_l, pix_in_r = np.array(points_interest["inner_pix_l"]), np.array(points_interest["inner_pix_r"])
    body_center = np.array(points_interest["body_center"])

    # Distance measurements between points of interest
    dist_r_pix = np.linalg.norm(pix_out_r - pix_in_r)
    dist_l_pix = np.linalg.norm(pix_out_l - pix_in_l)
    dist_r_center_pix = np.linalg.norm(pix_out_r - body_center)
    dist_l_center_pix = np.linalg.norm(pix_out_l - body_center)
    dist_span_pix = np.linalg.norm(pix_out_l - pix_out_r)
    dist_shoulder_pix = np.linalg.norm(pix_in_l - pix_in_r)

    # Converting to millimeters
    dist_l_mm = round(dist_l_pix / T_space, 2)
    dist_r_mm = round(dist_r_pix / T_space, 2)
    dist_l_center_mm = round(dist_l_center_pix / T_space, 2)
    dist_r_center_mm = round(dist_r_center_pix / T_space, 2)
    dist_span_mm = round(dist_span_pix / T_space, 2)
    dist_shoulder_mm = round(dist_shoulder_pix / T_space, 2)

    # Do we want to round these?
    dist_l_pix = round(dist_l_pix, 2)
    dist_r_pix = round(dist_r_pix, 2)
    dist_l_center_pix = round(dist_l_center_pix, 2)
    dist_r_center_pix = round(dist_r_center_pix, 2)
    dist_span_pix = round(dist_span_pix, 2)
    dist_shoulder_pix = round(dist_shoulder_pix, 2)

    dist_pix = {
        "dist_l": dist_l_pix,
        "dist_r": dist_r_pix,
        "dist_l_center": dist_l_center_pix,
        "dist_r_center": dist_r_center_pix,
        "dist_span": dist_span_pix,
        "dist_shoulder": dist_shoulder_pix
    }
    dist_mm = {
        "dist_l": dist_l_mm,
        "dist_r": dist_r_mm,
        "dist_l_center": dist_l_center_mm,
        "dist_r_center": dist_r_center_mm,
        "dist_span": dist_span_mm,
        "dist_shoulder": dist_shoulder_mm
    }

    if axes and axes[0]:
        textsize = 5
        if axes[3]:
            textsize = 3
        axes[0].plot([pix_out_l[1], pix_in_l[1]],
                     [pix_out_l[0], pix_in_l[0]], color='r')
        axes[0].plot([pix_out_r[1], pix_in_r[1]],
                     [pix_out_r[0], pix_in_r[0]], color='r')
        axes[0].text(int((pix_out_l[1] + pix_in_l[1]) / 2) + 50,
                     int((pix_out_l[0] + pix_in_l[0]) / 2) - 50,
                     'left_wing = ' + str(round(dist_l_mm, 2)) + ' mm',
                     size=textsize,
                     color='r')
        axes[0].text(int((pix_out_r[1] + pix_in_r[1]) / 2) + 50,
                     int((pix_out_r[0] + pix_in_r[0]) / 2) + 50,
                     'right_wing = ' + str(round(dist_r_mm, 2))
                     + ' mm',
                     size=textsize, color='r')

        axes[0].plot([pix_out_l[1], body_center[1]],
                     [pix_out_l[0], body_center[0]], color='orange', linestyle='dotted')
        axes[0].plot([pix_out_r[1], body_center[1]],
                     [pix_out_r[0], body_center[0]], color='orange', linestyle='dotted')
        axes[0].text(int((pix_out_l[1] + body_center[1]) / 2) + 50,
                     int((pix_out_l[0] + body_center[0]) / 2) - 50,
                     'left_wing_center = ' + str(round(dist_l_center_mm, 2)) + ' mm',
                     size=textsize,
                     color='orange')
        axes[0].text(int((pix_out_r[1] + body_center[1]) / 2) + 50,
                     int((pix_out_r[0] + body_center[0]) / 2) + 50,
                     'right_wing_center = ' + str(round(dist_r_center_mm, 2))
                     + ' mm',
                     size=textsize, color='orange')

        axes[0].plot([pix_out_l[1], pix_out_r[1]],
                     [pix_out_l[0], pix_out_r[0]], color='orange', linestyle='dashed')
        axes[0].text(int((pix_out_l[1] + pix_out_r[1]) / 2) - 50,
                     int((pix_out_l[0] + pix_out_r[0]) / 2) - 50,
                     'wing_span = ' + str(round(dist_span_mm, 2))
                     + ' mm',
                     size=textsize, color='orange')

        axes[0].plot([pix_in_l[1], pix_in_r[1]],
                     [pix_in_l[0], pix_in_r[0]], color='orange', linestyle='dashed')
        axes[0].text(int((pix_in_l[1] + pix_in_r[1]) / 2) + 50,
                     int((pix_in_l[0] + pix_in_r[0]) / 2) + 50,
                     'wing_shoulder = ' + str(round(dist_shoulder_mm, 2))
                     + ' mm',
                     size=textsize, color='orange')

    print(f'Measurements:')
    print(f'* left_wing: {dist_mm["dist_l"]} mm')
    print(f'* right_wing: {dist_mm["dist_r"]} mm')
    print(f'* left_wing_center: {dist_mm["dist_l_center"]} mm')
    print(f'* right_wing_center: {dist_mm["dist_r_center"]} mm')
    print(f'* wing_span: {dist_mm["dist_span"]} mm')
    print(f'* wing_shoulder: {dist_mm["dist_shoulder"]} mm')

    return dist_pix, dist_mm
