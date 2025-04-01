import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
from jutility import plotting, util, time_sweep

def main():
    e_list: list[ExperimentBase] = [
        MplMathTextLegend(),
        MplPlainTextLegend(),
        MplNoLegend(),
        JutilityMathTextLegend(),
        JutilityPlainTextLegend(),
        JutilityNoLegend(),
    ]
    repeats = list(range(20))
    cp = plotting.ColourPicker(len(e_list))
    lines = []
    for e in e_list:
        e.setup(100)
        timer = util.Timer(e)
        for _ in repeats:
            with timer:
                e.run()

        t = list(timer)
        ls = "--" if isinstance(e, JutilityBase) else "-"
        line = plotting.Line(t, m="o", ls=ls, c=cp.next(), label=repr(e))
        lines.append(line)

    plotting.plot(
        *lines,
        plotting.Legend(),
        plot_name="Legend comparison",
        xlabel="Repeat",
        ylabel="Time (seconds)",
        xticks=repeats[::2],
        log_y=True,
    )

    time_sweep.time_sweep(
        *e_list,
        plot_name="Legend comparison (sweeping over line length)",
    )

class ExperimentBase(time_sweep.Experiment):
    def setup(self, n):
        rng = np.random.default_rng(n)
        self.x = np.linspace(0, 1, n)
        self.y = self.x + rng.normal(0, 0.1, n)

class MplBase(ExperimentBase):
    def run(self):
        figure = plt.figure(figsize=[6, 4])
        axis_list = figure.subplots(1, 1, squeeze=False).flatten().tolist()
        axis = axis_list[0]
        assert isinstance(axis, matplotlib.axes.Axes)
        handles = axis.plot(self.x, self.y, c="b")
        self.make_legend(handles, axis)
        figure.savefig("results/%r.png" % self)
        plt.close(figure)

    def make_legend(self, handles, axis: matplotlib.axes.Axes):
        raise NotImplementedError()

class MplNoLegend(MplBase):
    def make_legend(self, handles, axis: matplotlib.axes.Axes):
        return

class MplPlainTextLegend(MplBase):
    def make_legend(self, handles, axis: matplotlib.axes.Axes):
        axis.legend(handles=handles, labels=["y = x + \\epsilon"])

class MplMathTextLegend(MplBase):
    def make_legend(self, handles, axis: matplotlib.axes.Axes):
        axis.legend(handles=handles, labels=["$y = x + \\epsilon$"])

class JutilityBase(ExperimentBase):
    def run(self):
        line = plotting.Line(self.x, self.y, c="b")
        legend = self.get_legend([line.get_handle()])
        mp = plotting.MultiPlot(
            plotting.Subplot(line, *legend, grid=False),
            figsize=[6, 4],
            constrained_layout=False,
        )
        mp.save(repr(self), verbose=False)

    def get_legend(self, handles):
        raise NotImplementedError()

class JutilityNoLegend(JutilityBase):
    def get_legend(self, handles):
        return []

class JutilityPlainTextLegend(JutilityBase):
    def get_legend(self, handles):
        labels = ["y = x + \\epsilon"]
        return [plotting.Legend(handles=handles, labels=labels)]

class JutilityMathTextLegend(JutilityBase):
    def get_legend(self, handles):
        labels = ["$y = x + \\epsilon$"]
        return [plotting.Legend(handles=handles, labels=labels)]

if __name__ == "__main__":
    with util.Timer("main"):
        main()
