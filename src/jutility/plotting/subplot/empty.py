import matplotlib.axes
from jutility.plotting.subplot.subplot import Subplot

class Empty(Subplot):
    def plot_axis(self, axis: matplotlib.axes.Axes):
        axis.set_axis_off()
