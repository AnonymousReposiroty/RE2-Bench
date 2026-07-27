import matplotlib._layoutgrid as mlayoutgrid

def make_layoutgrids(fig, layoutgrids, rect=(0, 0, 1, 1)):
    """
    Make the layoutgrid tree.

    (Sub)Figures get a layoutgrid so we can have figure margins.

    Gridspecs that are attached to Axes get a layoutgrid so Axes
    can have margins.
    """

    if layoutgrids is None:
        layoutgrids = dict()
        layoutgrids['hasgrids'] = False
    if not hasattr(fig, '_parent'):
        # top figure;  pass rect as parent to allow user-specified
        # margins
        layoutgrids[fig] = mlayoutgrid.LayoutGrid(parent=rect, name='figlb')
    else:
        # subfigure
        gs = fig._subplotspec.get_gridspec()
        # it is possible the gridspec containing this subfigure hasn't
        # been added to the tree yet:
        layoutgrids = make_layoutgrids_gs(layoutgrids, gs)
        # add the layoutgrid for the subfigure:
        parentlb = layoutgrids[gs]
        layoutgrids[fig] = mlayoutgrid.LayoutGrid(
            parent=parentlb,
            name='panellb',
            parent_inner=True,
            nrows=1, ncols=1,
            parent_pos=(fig._subplotspec.rowspan,
                        fig._subplotspec.colspan))
    # recursively do all subfigures in this figure...
    for sfig in fig.subfigs:
        layoutgrids = make_layoutgrids(sfig, layoutgrids)

    # for each Axes at the local level add its gridspec:
    for ax in fig._localaxes:
        gs = ax.get_gridspec()
        if gs is not None:
            layoutgrids = make_layoutgrids_gs(layoutgrids, gs)

    return layoutgrids
