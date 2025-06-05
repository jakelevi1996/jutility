import matplotlib.axes
from jutility.plotting.subplot.base import Subplot

class Empty(Subplot):
    def plot(self, axis: matplotlib.axes.Axes):
        axis.set_axis_off()
