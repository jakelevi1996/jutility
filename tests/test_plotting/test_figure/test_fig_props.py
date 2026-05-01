import pytest
from jutility import plotting, util

OUTPUT_DIR = util.get_test_output_dir(
    "test_plotting/test_figure/test_fig_props",
)

def test_invalid_args():
    rng = util.Seeder().get_rng("test_invalid_args")

    with pytest.raises(ValueError):
        mp = plotting.MultiPlot(
            plotting.Subplot(),
            invalid_arg=None,
        )
        mp.save()

    with pytest.raises(ValueError):
        mp = plotting.MultiPlot(
            plotting.Subplot(),
            plotting.MultiPlot(
                plotting.Subplot(),
                invalid_arg=None,
            ),
        )
        mp.save()

    with pytest.raises(ValueError):
        mp = plotting.MultiPlot(
            plotting.Subplot(),
            plotting.MultiPlot(
                plotting.Subplot(),
                fs=[3, 3],
            ),
        )
        mp.save()

    mp = plotting.MultiPlot(
        plotting.Subplot(),
        fs=[3, 3],
    )
    mp.save(
        "test_invalid_args_1",
        dir_name=OUTPUT_DIR,
    )

    mp = plotting.MultiPlot(
        plotting.Subplot(),
        plotting.MultiPlot(
            plotting.Subplot(),
        ),
        fs=[3, 3],
    )
    mp.save(
        "test_invalid_args_2",
        dir_name=OUTPUT_DIR,
    )
