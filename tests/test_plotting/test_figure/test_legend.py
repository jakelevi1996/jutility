from jutility import plotting, util

OUTPUT_DIR = util.get_test_output_dir("test_plotting/test_figure/test_legend")

def test_bottom_centre():
    rng = util.Seeder().get_rng("test_bottom_centre")

    cp = plotting.ColourPicker.hsv(8)
    lines = [
        plotting.Line(rng.normal(0, 1, 10), c=c, label=util.list_to_hex(c))
        for c in cp
    ]
    mp = plotting.MultiPlot(
        plotting.Subplot(*lines),
        legend=plotting.FigureLegend.bottom_centre(*lines, num_rows=2)
    )
    mp.save(
        plot_name="test_bottom_centre",
        dir_name=OUTPUT_DIR,
    )

def test_centre_right():
    rng = util.Seeder().get_rng("test_centre_right")

    cp = plotting.ColourPicker.hsv(8)
    lines = [
        plotting.Line(rng.normal(0, 1, 10), c=c, label=util.list_to_hex(c))
        for c in cp
    ]
    mp = plotting.MultiPlot(
        plotting.Subplot(*lines),
        legend=plotting.FigureLegend.centre_right(*lines)
    )
    mp.save(
        plot_name="test_centre_right",
        dir_name=OUTPUT_DIR,
    )
