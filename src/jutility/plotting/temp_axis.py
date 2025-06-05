import matplotlib.pyplot as plt

class _TempAxis:
    def __init__(self):
        self._fig = None
        self._axis = None
        self._old_children = None

    def get_axis(self):
        if self._axis is None:
            self._fig = plt.figure()
            self._axis = self._fig.gca()
            self._old_children = set(self._axis.get_children())

        return self._axis

    def pop_artists(self):
        new_children = set(self._axis.get_children()) - self._old_children
        for a in new_children:
            a.remove()

        return new_children

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)

_temp_axis = _TempAxis()
