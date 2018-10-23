import numpy as np

def main(points_interest, T_space, ax=None):
    ''' Calculates the length and draws the lines for length
    of the butterfly wings.

    Parameters
    ----------
    ax: array
        the array containing the 3 intermediary Axes.
    points_interest: array
        the array containing the four points of interest,
        each of which is a coordinate specifying the start/end
        point of the left/right wing.
    T_space: float
        number of pixels between 2 ticks.

    Returns
    -------
    ax: array
        the array containing the 3 intermediary Axes.
    dst_pix: tuple
        the tuple contains the distance of the left/right wing
        distance in pixels
    dst_mm: tuple
        the tuple contains the distance of the left/right wing
        distance in millimeters

    '''

    # do i need to take in an image if it's already in ax[0]? no
    # image = ax[0]

    pix_out_l, pix_in_l, pix_out_r, pix_in_r = points_interest
    dist_r_pix = np.sqrt((pix_out_r[0] -pix_in_r[0])**2 + (pix_out_r[1] -pix_in_r[1])**2)
    dist_l_pix = np.sqrt((pix_out_l[0] -pix_in_l[0])**2 + (pix_out_l[1] -pix_in_l[1])**2)

    # Converting to millimeters
    dist_l_mm = dist_l_pix /( 2 *T_space)
    dist_r_mm = dist_r_pix /( 2 *T_space)

    # Do we want to round these?
    dist_l_pix = round(dist_l_pix, 2)
    dist_r_pix = round(dist_r_pix, 2)
    dist_l_mm = round(dist_l_mm, 2)
    dist_r_mm = round(dist_r_mm, 2)


    dst_pix = (dist_l_pix, dist_r_pix)
    dst_mm = (dist_l_mm, dist_r_mm)
    if ax:
        ax.set_title('final image')
        # ax.imshow(image)
        ax.plot([pix_out_l[1], pix_in_l[1]], [pix_out_l[0], pix_in_l[0]], color='r')
        ax.plot([pix_out_r[1], pix_in_r[1]], [pix_out_r[0], pix_in_r[0]], color='r')
        ax.text(int((pix_out_l[1] + pix_in_l[1] ) /2) +50,
                   int((pix_out_l[0] + pix_in_l[0]) / 2) - 50,
                   'dist_left = ' + str(round(dist_l_mm, 2)) + ' mm',
                   color='r')
        ax.text(int((pix_out_r[1] + pix_in_r[1]) / 2) + 50,
                   int((pix_out_r[0] + pix_in_r[0]) / 2) + 50,
                   'dist_right = ' + str(round(dist_r_mm, 2)) + ' mm',
                   color='r')

    return dst_pix, dst_mm
