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
