import numpy as np
from jutility import util
from jutility.plotting.plottable import (
    Scatter,
    Line,
    FillBetween,
    PlottableGroup,
    VLine,
    HLine,
    AxLine,
)
from jutility.plotting.noisy.bounds import confidence_bounds

class NoisyData:
    def __init__(
        self,
        log_x:      bool=False,
        log_y:      bool=False,
        x_index:    bool=False,
        results:    (dict[float, list[float]] | None)=None,
    ):
        if results is None:
            results = dict()

        self._results_list_dict = results
        self._log_x = log_x
        self._log_y = log_y
        self._x_index = x_index
        self._x_index_list = []

    def update(self, x, y):
        if self._x_index:
            if x not in self._x_index_list:
                self._x_index_list.append(x)

            x = self._x_index_list.index(x)

        if x in self._results_list_dict:
            self._results_list_dict[x].append(y)
        else:
            self._results_list_dict[x] = [y]

    def get_all_data(self):
        all_results_pairs = [
            [x, y]
            for x, y_list in self._results_list_dict.items()
            for y in y_list
        ]
        n = len(all_results_pairs)
        xy_n2 = np.array(all_results_pairs).reshape(n, 2)
        x_n1, y_n1 = np.split(xy_n2, 2, axis=1)
        return x_n1.flatten(), y_n1.flatten()

    def get_statistics(self, n_sigma: float=1.0):
        x_list = self.get_x()
        y_list_list = [self._results_list_dict[x] for x in x_list]
        mean, ucb, lcb = confidence_bounds(
            y_list_list,
            n_sigma=n_sigma,
            log=self._log_y,
        )
        return np.array(x_list), mean, ucb, lcb

    def get_x(self) -> list[float]:
        return sorted(
            x
            for x, y_list in self._results_list_dict.items()
            if len(y_list) > 0
        )

    def get_mean(self, x: float) -> float:
        return np.mean(self._results_list_dict[x])

    def argmax(self):
        best_y_dict = {
            max(y_list): x
            for x, y_list in self._results_list_dict.items()
            if len(y_list) > 0
        }
        best_y = max(best_y_dict.keys())
        best_x = best_y_dict[best_y]
        best_repeat = self._results_list_dict[best_x].index(best_y)
        if self._x_index:
            best_x = self._x_index_list[best_x]

        return best_x, best_repeat, best_y

    def argmin(self):
        best_y_dict = {
            min(y_list): x
            for x, y_list in self._results_list_dict.items()
            if len(y_list) > 0
        }
        best_y = min(best_y_dict.keys())
        best_x = best_y_dict[best_y]
        best_repeat = self._results_list_dict[best_x].index(best_y)
        if self._x_index:
            best_x = self._x_index_list[best_x]

        return best_x, best_repeat, best_y

    def plot_best(
        self,
        maximise:   bool,
        c:          str="r",
        ls:         str="--",
        z:          int=40,
        label_fmt:  str="%s, %s",
    ) -> PlottableGroup:
        x, _, y = (self.argmax() if maximise else self.argmin())
        label = label_fmt % (x, y)
        return PlottableGroup(
            VLine(x, c=c, ls=ls, z=z),
            HLine(y, c=c, ls=ls, z=z),
            Scatter([], [], c=c, m="+", label=label),
        )

    def plot_value(
        self,
        y:          float,
        y_str:      (str | None)=None,
        c:          str="r",
        ls:         str="--",
        z:          int=40,
        label_fmt:  str="%s, %s",
    ) -> PlottableGroup:
        if y_str is None:
            y_str = str(y)

        x, _ = min(self.inverse(y))
        label = label_fmt % (x, y)
        return PlottableGroup(
            VLine(x, c=c, ls=ls, z=z),
            HLine(y, c=c, ls=ls, z=z),
            Scatter([], [], c=c, m="+", label=label),
        )

    def plot(
        self,
        c:          str="b",
        label:      (str | None)=None,
        n_sigma:    float=1.0,
        alpha_scat: float=0.5,
        alpha_line: float=1.0,
        alpha_fill: float=0.2,
    ):
        x, mean, ucb, lcb = self.get_statistics(n_sigma)
        return PlottableGroup(
            Scatter(*self.get_all_data(),   a=alpha_scat, z=20, color=c),
            Line(x, mean,                   a=alpha_line, z=30, color=c),
            FillBetween(x, lcb, ucb,        a=alpha_fill, z=10, color=c),
            label=label,
        )

    def get_xtick_kwargs(self):
        if self._x_index:
            ticks = list(range(len(self._x_index_list)))
            labels = self._x_index_list
        else:
            ticks = sorted(self._results_list_dict.keys())
            labels = ticks

        return {"xticks": ticks, "xticklabels": labels}

    def predict(self, x_pred: np.ndarray, eps=1e-5):
        x, y = self.get_all_data()
        if self._log_x:
            x_pred = np.log(x_pred)
            x = np.log(x)
        if self._log_y:
            y = np.log(y)

        xm = x.mean()
        ym = y.mean()
        xc = x - xm
        yc = y - ym

        w = np.sum(yc * xc) / (np.sum(xc * xc) + eps)
        b = ym - w * xm
        y_pred = w * x_pred + b

        if self._log_y:
            y_pred = np.exp(y_pred)

        return y_pred

    def predict_line(self, x0: float, x1: float, eps=1e-5, **line_kwargs):
        y0, y1 = self.predict(np.array([x0, x1]), eps)
        line_kwargs.setdefault("ls", "--")
        return AxLine([x0, y0], [x1, y1], **line_kwargs)

    def inverse(self, y: float) -> set[tuple[float, int]]:
        return set(
            (x, i)
            for x, y_list in self._results_list_dict.items()
            for i, yi in enumerate(y_list)
            if  yi == y
        )

    def __iter__(self):
        return (
            y
            for y_list in self._results_list_dict.values()
            for y in y_list
        )

    def __len__(self) -> int:
        return sum(
            len(y_list)
            for y_list in self._results_list_dict.values()
        )

    def __repr__(self):
        return util.format_type(type(self), results=self._results_list_dict)
