import matplotlib.axes
from jutility import util
from jutility.plotting.properties import PropertyDict

class AxisProperties(PropertyDict):
    def apply(self, axis: matplotlib.axes.Axes):
        if self._has("xlabel"):
            axis.set_xlabel(self._get("xlabel"))
        if self._has("ylabel"):
            axis.set_ylabel(self._get("ylabel"))
        if self._get_default("axis_equal", False):
            axis.axis("equal")
        if self._get_default("axis_square", False):
            axis.axis("square")
        if self._get_default("axis_off", False):
            axis.set_axis_off()
        if self._get_default("log_x", False):
            axis.set_xscale("log")
        if self._get_default("log_y", False):
            axis.set_yscale("log")
        if self._get_default("symlog_x", False):
            axis.set_xscale("symlog")
        if self._get_default("symlog_y", False):
            axis.set_yscale("symlog")
        if self._get_default("grid", True):
            grid_x = self._get_default("grid_x", "both")
            grid_y = self._get_default("grid_y", "both")
            if grid_x:
                axis.grid(visible=True, which=grid_x, axis="x")
            if grid_y:
                axis.grid(visible=True, which=grid_y, axis="y")
        if self._has("xticks"):
            ticks  = self._get("xticks")
            labels = self._get_default("xticklabels", None)
            axis.set_xticks(ticks, labels)
        if self._has("yticks"):
            ticks  = self._get("yticks")
            labels = self._get_default("yticklabels", None)
            axis.set_yticks(ticks, labels)
        if self._has("xlim"):
            axis.set_xlim(self._get("xlim"))
        if self._has("ylim"):
            axis.set_ylim(self._get("ylim"))
        if self._get_default("rotate_xticklabels", False):
            for xtl in axis.get_xticklabels():
                xtl.set(rotation=-45, ha="left")
        title = self._get_default("title", None)
        if title is not None:
            colour_dict = dict()
            if self._has("title_colour"):
                colour_dict["color"] = self._get("title_colour")
            if self._get_default("wrap_title", True):
                title = util.wrap_string(title)
            axis.set_title(
                title,
                fontsize=self._get_default("title_font_size", 12),
                **colour_dict,
            )
        if self._has("colour"):
            axis.set_facecolor(self._get("colour"))
