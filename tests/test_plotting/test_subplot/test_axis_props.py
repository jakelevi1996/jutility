from jutility import plotting, util

OUTPUT_DIR = util.get_test_output_dir(
    "test_plotting/test_subplot/test_axis_props",
)

def test_axisproperties_abbreviations():
    xy = [
        [1,  0.94260 ],
        [1,  0.94280 ],
        [1,  0.94020 ],
        [2,  0.95320 ],
        [2,  0.95270 ],
        [2,  0.94930 ],
        [3,  0.95750 ],
        [3,  0.95900 ],
        [3,  0.95810 ],
        [4,  0.96130 ],
        [4,  0.96240 ],
        [4,  0.96160 ],
        [5,  0.96360 ],
        [5,  0.96420 ],
        [5,  0.96100 ],
    ]

    nd = plotting.NoisyData()
    for x, y in xy:
        nd.update(x, y)

    plotting.plot(
        nd.plot(),
        xt=list(range(1, 6)),
        title="Accuracy vs depth",
        tfs=20,
        plot_name="test_axisproperties_abbreviations",
        dir_name=OUTPUT_DIR,
    )
