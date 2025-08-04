from jutility import plotting, util

OUTPUT_DIR = util.get_test_output_dir("test_plotting/test_multiplot")

def test_unevenly_spaced_grids():
    rng = util.get_numpy_rng("test_unevenly_spaced_grids")
    cp = plotting.ColourPicker(4)

    mp = plotting.MultiPlot(
        *[
            plotting.MultiPlot(
                *[
                    plotting.Subplot(
                        plotting.Line(rng.random(10), c=cp.next()),
                        xticks=[],
                        yticks=[],
                    )
                    for _ in range(25)
                ],
                title="Split = %s" % s,
            )
            for s in "train test".split()
        ],
        space=0.2,
        width_ratios=[2, 1],
        figsize=[10, 5],
        title="test_unevenly_spaced_grids",
    )
    mp.save("test_unevenly_spaced_grids", OUTPUT_DIR)

def test_uneven_leaf_levels():
    rng = util.get_numpy_rng("test_uneven_leaf_levels")
    cp = plotting.ColourPicker(4)

    mp = plotting.MultiPlot(
        plotting.MultiPlot(
            plotting.MultiPlot(
                plotting.Subplot(plotting.Line(rng.random(10), c=cp.next())),
                plotting.Subplot(plotting.Line(rng.random(10), c=cp.next())),
                num_rows=1,
                width_ratios=[2, 3],
            ),
            plotting.Subplot(plotting.Line(rng.random(10), c=cp.next())),
            num_cols=1,
            height_ratios=[1, 2],
        ),
        plotting.Subplot(plotting.Line(rng.random(10), c=cp.next())),
        plotting.ColourBar(0, 10),
        width_ratios=[3, 2, 0.8],
        num_rows=1,
        pad=0.05,
        title="test_uneven_leaf_levels",
    )
    mp.save("test_uneven_leaf_levels", OUTPUT_DIR)

def test_nested_figure_legend():
    rng = util.get_numpy_rng("test_nested_figure_legend")
    cp = plotting.ColourPicker(4)

    def get_subplot(**kwargs):
        line = plotting.Line(rng.uniform(0, 1, 4), c=cp.next())
        return plotting.Subplot(line, **kwargs)

    mp = plotting.MultiPlot(
        plotting.MultiPlot(
            plotting.MultiPlot(
                get_subplot(),
                get_subplot(),
            ),
            get_subplot(),
            get_subplot(),
            legend=plotting.FigureLegend(
                plotting.Line(a=0.5, label="Semi-transparent"),
                plotting.Line(a=1, label="Opaque"),
            ),
        ),
        plotting.MultiPlot(
            plotting.MultiPlot(
                get_subplot(),
                get_subplot(),
                legend=plotting.FigureLegend(
                    plotting.Line(ls="--", label="Dashed"),
                    plotting.Line(ls="-", label="Solid"),
                ),
            ),
            get_subplot(),
            get_subplot(),
            legend=plotting.FigureLegend(
                *[
                    plotting.Line(lw=i, label=i)
                    for i in range(1, 11)
                ],
                loc="outside center right",
                num_rows=None,
                title="Line width",
            ),
        ),
        legend=plotting.FigureLegend(
            plotting.Line(c="r", label="red"),
            plotting.Line(c="g", label="green"),
            plotting.Line(c="b", label="blue"),
        ),
        figsize=[12, 6],
        title="test_nested_figure_legend",
    )
    mp.save("test_nested_figure_legend", OUTPUT_DIR)
