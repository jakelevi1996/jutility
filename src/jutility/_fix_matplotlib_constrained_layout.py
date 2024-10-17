import matplotlib.gridspec
import matplotlib._constrained_layout
import matplotlib._layoutgrid as mlayoutgrid

def make_layoutgrids_gs(layoutgrids, gs):
    """
    Make the layoutgrid for a gridspec (and anything nested in the gridspec)
    """

    if gs in layoutgrids or gs.figure is None:
        return layoutgrids
    # in order to do constrained_layout there has to be at least *one*
    # gridspec in the tree:
    layoutgrids['hasgrids'] = True
    if not hasattr(gs, '_subplot_spec'):
        # normal gridspec
        parent = layoutgrids[gs.figure]
        layoutgrids[gs] = mlayoutgrid.LayoutGrid(
                parent=parent,
                parent_inner=True,
                name='gridspec',
                ncols=gs._ncols, nrows=gs._nrows,
                width_ratios=gs.get_width_ratios(),
                height_ratios=gs.get_height_ratios())
    else:
        # this is a gridspecfromsubplotspec:
        subplot_spec = gs._subplot_spec
        parentgs = subplot_spec.get_gridspec()
        # if a nested gridspec it is possible the parent is not in there yet:
        if parentgs not in layoutgrids:
            layoutgrids = make_layoutgrids_gs(layoutgrids, parentgs)
        subspeclb = layoutgrids[parentgs]
        # gridspecfromsubplotspec need an outer container:
        # get a unique representation:
        rep = (gs, 'top')
        if rep not in layoutgrids:
            layoutgrids[rep] = mlayoutgrid.LayoutGrid(
                parent=subspeclb,
                parent_inner=True,
                name='top',
                nrows=1, ncols=1,
                parent_pos=(subplot_spec.rowspan, subplot_spec.colspan))
        layoutgrids[gs] = mlayoutgrid.LayoutGrid(
                parent=layoutgrids[rep],
                name='gridspec',
                nrows=gs._nrows, ncols=gs._ncols,
                width_ratios=gs.get_width_ratios(),
                height_ratios=gs.get_height_ratios())
    return layoutgrids

matplotlib._constrained_layout.make_layoutgrids_gs = make_layoutgrids_gs
