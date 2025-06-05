import matplotlib

def set_latex_params(use_tex=True, use_times=False):
    latex_params_dict = {
        "font.family":          "serif",
        "font.serif":           ["Times" if use_times else "Computer Modern"],
        "text.usetex":          True,
        "text.latex.preamble":  "\\usepackage{amsmath}",
        "legend.edgecolor":     "k",
        "legend.fancybox":      False,
        "legend.framealpha":    1,
    }

    for key, value in latex_params_dict.items():
        if use_tex:
            matplotlib.rcParams[key] = value
        else:
            matplotlib.rcParams[key] = matplotlib.rcParamsDefault[key]
