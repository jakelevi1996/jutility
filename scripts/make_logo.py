import os
import numpy as np
from jutility import plotting

num_colours = 7
cp = plotting.ColourPicker(num_colours, cyclic=False)
x = np.linspace(-1, 7, 100)
line_list = [
    plotting.Line(
        x,
        ((1 + (i/10)) * np.sin(x + (i / num_colours))),
        c=cp(i),
    )
    for i in range(num_colours)
]
mp = plotting.MultiPlot(
    plotting.Subplot(
        *line_list,
        axis_properties=plotting.AxisProperties(axis_off=True, xlim=[-1, 7]),
    ),
    figure_properties=plotting.FigureProperties(
        figsize=[10, 4],
        colour="k",
        title="  ".join("jutility"),
        title_colour="w",
        title_font_size=40,
        top_space=0.2,
    ),
)
mp.save("logo_black", dir_name=os.path.join(os.getcwd(), "images"))
