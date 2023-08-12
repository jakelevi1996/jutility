from jutility import util

class Indenter:
    def __init__(self, indent_str=None, initial_indent=0):
        if indent_str is None:
            indent_str = "".ljust(4)
        self._indent_str = indent_str
        self._num_indent = initial_indent

        self.new_block = util.CallbackContext(
            lambda: self._add_indent( 1),
            lambda: self._add_indent(-1),
        )

    def _add_indent(self, n):
        self._num_indent += n

    def __call__(self, s):
        prefix = self._indent_str * self._num_indent
        return prefix + str(s)
