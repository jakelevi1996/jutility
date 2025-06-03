from jutility import util

class Verbose:
    def __init__(self):
        self.reset()
        self.set_printer(util.Printer())

    def display_init(self, object_type: type, kwargs: dict):
        self._printer("cli: %s" % util.format_type(object_type, **kwargs))

    def display_retrieve(self, full_name: str):
        self._printer("cli: `%s` retrieved from cache" % full_name)

    def reset(self):
        self._verbosity = 0

    def set_printer(self, printer: util.Printer):
        self._printer = printer

    def __enter__(self):
        self._verbosity += 1

    def __exit__(self, *args):
        self._verbosity = max(self._verbosity - 1, 0)

    def __bool__(self):
        return self._verbosity > 0

verbose = Verbose()
