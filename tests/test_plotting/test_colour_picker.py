from jutility import plotting, util

OUTPUT_DIR = util.get_test_output_dir("test_plotting/test_colour_picker")

def test_hsv():
    rng = util.get_numpy_rng("test_hsv")
    cp = plotting.ColourPicker.hsv(4)

    plotting.plot(
        *[
            plotting.Line(rng.normal(0, 1, 10), c=c)
            for c in cp
        ],
        plotting.Legend.from_plottables(
            *cp.get_legend_lines(*[str(i) for i in range(len(cp))]),
        ),
        plot_name="test_hsv",
        dir_name=OUTPUT_DIR,
    )

def test_hsv_offset():
    rng = util.get_numpy_rng("test_hsv_offset")
    cp = plotting.ColourPicker.hsv(5, offset=0.05)

    plotting.plot(
        *[
            plotting.Line(rng.normal(0, 1, 10), c=c)
            for c in cp
        ],
        plotting.Legend.from_plottables(
            *cp.get_legend_lines(*[str(i) for i in range(len(cp))]),
        ),
        plot_name="test_hsv_offset",
        dir_name=OUTPUT_DIR,
    )

def test_cool():
    rng = util.get_numpy_rng("test_cool")
    cp = plotting.ColourPicker.cool(4)

    plotting.plot(
        *[
            plotting.Line(rng.normal(0, 1, 10), c=c)
            for c in cp
        ],
        plotting.Legend.from_plottables(
            *cp.get_legend_lines(*[str(i) for i in range(len(cp))]),
        ),
        plot_name="test_cool",
        dir_name=OUTPUT_DIR,
    )

def test_ibm():
    rng = util.get_numpy_rng("test_ibm")
    cp = plotting.ColourPicker.ibm()

    plotting.plot(
        *[
            plotting.Line(rng.normal(0, 1, 10), c=c)
            for c in cp
        ],
        plotting.Legend.from_plottables(
            *cp.get_legend_lines(*[str(i) for i in range(len(cp))]),
        ),
        plot_name="test_ibm",
        dir_name=OUTPUT_DIR,
    )

def test_get_legend_sweeps():
    cp = plotting.ColourPicker.hsv(50)
    mp = plotting.MultiPlot(
        plotting.LegendSubplot(
            *cp.get_legend_sweeps(*[str(i) for i in range(len(cp))]),
            ncols=5,
        ),
    )
    mp.save(
        plot_name="test_get_legend_sweeps",
        dir_name=OUTPUT_DIR,
    )
