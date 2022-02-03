import matplotlib.pyplot as plt


def create_layout(n_stages, plot_level):
    """Creates Axes to plot figures

    Parameters
    ----------
    n_stages : int
        length of pipeline process
    plot_level : int
        0 : no plotting
        1 : regular plots
        2 : detailed plots

    Returns
    -------
    axes : list of Axes
    """
    if plot_level == 0:
        return None

    elif plot_level == 1:
        ncols = n_stages
        fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(12, 5))
        if n_stages == 1:
            ax = [ax]
        ax_list = []
        for ax in ax:
            ax_list.append(ax)
        return ax_list + [None] * (7 - n_stages)

    elif plot_level == 2:
        shape = (3, 3)
        ax_main = plt.subplot2grid(shape, (0, 0))
        ax_structure = plt.subplot2grid(shape, (0, 1))
        ax_signal = plt.subplot2grid(shape, (1, 0), colspan=2)
        ax_fourier = plt.subplot2grid(shape, (2, 0), colspan=2)

        ax_tags = plt.subplot2grid(shape, (0, 2))
        ax_bin = plt.subplot2grid(shape, (1, 2))
        ax_poi = plt.subplot2grid(shape, (2, 2))
        plt.tight_layout()
        if n_stages == 1:
            return [ax_main, None, None, ax_structure, ax_signal, ax_fourier,
                    None]
        elif n_stages == 2:
            return [ax_main, ax_bin, None, ax_structure, ax_signal, ax_fourier,
                    ax_tags]
        elif n_stages == 3:
            return [ax_main, ax_bin, ax_poi, ax_structure, ax_signal,
                    ax_fourier, ax_tags]
