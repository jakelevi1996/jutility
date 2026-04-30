from jutility import plotting, util

OUTPUT_DIR = util.get_test_output_dir("test_plotting/test_colour_picker")

def test_offset():
    rng = util.get_numpy_rng("test_offset")
    labels = "A BC DEF GHIJK LMNOP".split()

    ns = plotting.NoisySweep()
    for _ in range(5):
        x, y = rng.normal(), rng.normal()
        ns.update(labels[0], x, y)
        x, y = rng.normal(), rng.normal()
        ns.update(labels[1], x, y)
        x, y = rng.normal(), rng.normal()
        ns.update(labels[2], x, y)
        x, y = rng.normal(), rng.normal()
        ns.update(labels[3], x, y)
        x, y = rng.normal(), rng.normal()
        ns.update(labels[4], x, y)

    cp = plotting.ColourPicker(len(labels), offset=0.05)

    mp = plotting.MultiPlot(
        plotting.Subplot(
            *ns.plot(cp, labels),
            xlim=[-2, 2],
            ylim=[-2, 2],
        ),
        legend=plotting.FigureLegend(
            *cp.get_legend_lines(*labels),
            num_rows=None,
            loc="outside center right",
        ),
    )
    mp.save(
        "test_offset",
        dir_name=OUTPUT_DIR,
    )

def test_get_legend_lines():
    rng = util.get_numpy_rng("test_get_legend_lines")
    labels = "A BC DEF GHIJK LMNOP".split()

    ns = plotting.NoisySweep()
    for _ in range(5):
        x, y = rng.normal(), rng.normal()
        ns.update(labels[0], x, y)
        x, y = rng.normal(), rng.normal()
        ns.update(labels[1], x, y)
        x, y = rng.normal(), rng.normal()
        ns.update(labels[2], x, y)
        x, y = rng.normal(), rng.normal()
        ns.update(labels[3], x, y)
        x, y = rng.normal(), rng.normal()
        ns.update(labels[4], x, y)

    cp = plotting.ColourPicker(len(labels))

    mp = plotting.MultiPlot(
        plotting.Subplot(
            *ns.plot(cp, labels),
            xlim=[-2, 2],
            ylim=[-2, 2],
        ),
        legend=plotting.FigureLegend(
            *cp.get_legend_lines(*labels),
            num_rows=None,
            loc="outside center right",
        ),
    )
    mp.save(
        "test_get_legend_lines",
        dir_name=OUTPUT_DIR,
    )

def test_ibm():
    rng = util.get_numpy_rng("test_ibm")
    labels = "A BC DEF GHIJK LMNOP".split()

    cp = plotting.ColourPicker.ibm()
    lines = [
        plotting.Line(rng.normal(0, 1, 10), c=c)
        for c in cp
    ]

    plotting.plot(
        *lines,
        plotting.Legend.from_plottables(
            *cp.get_legend_lines(*labels),
        ),
        plot_name="test_ibm",
        dir_name=OUTPUT_DIR,
    )

def test_ibm_2_colour():
    rng = util.get_numpy_rng("test_ibm_2_colour")
    labels = "A BC DEF GHIJK LMNOP".split()

    cp = plotting.ColourPicker.ibm_2_colour()
    lines = [
        plotting.Line(rng.normal(0, 1, 10), c=c)
        for c in cp
    ]

    plotting.plot(
        *lines,
        plotting.Legend.from_plottables(
            *cp.get_legend_lines(*labels),
        ),
        plot_name="test_ibm_2_colour",
        dir_name=OUTPUT_DIR,
    )
