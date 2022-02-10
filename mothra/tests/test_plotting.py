from mothra import plotting


def test_create_layout():
    """Checks if axes for regular and detailed plots were created
    properly.

    Summary
    -------
    We pass different stages and plot levels to plotting.create_layout,
    and check the resulting axes.

    Expected
    --------
    - plotting.create_layout(1, 0) should not return axes.
    - plotting.create_layout(3, 2) should return all seven axes
    (ax_main, ax_bin, ax_poi, ax_structure, ax_signal, ax_fourier, ax_tags).
    - plotting.create_layout(3, 1) should return a list with three axes and
    four None.
    """
    axes = plotting.create_layout(1, 0)
    assert axes is None
    axes = plotting.create_layout(3, 2)
    for ax in axes:
        assert ax
    axes = plotting.create_layout(3, 1)
    for ax in axes[:3]:
        assert ax
    for ax in axes[3:]:
        assert ax is None
