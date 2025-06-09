import datetime
import traceback
from jutility.util.print_util import Printer

class CallbackContext:
    def __init__(
        self,
        enter_callback=None,
        exit_callback=None,
        enter_return=None,
        suppress_exceptions=False,
    ):
        self._enter_callback        = enter_callback
        self._exit_callback         = exit_callback
        self._enter_return          = enter_return
        self._suppress_exceptions   = suppress_exceptions

    def __enter__(self):
        if self._enter_callback is not None:
            self._enter_callback()

        return self._enter_return

    def __exit__(self, *args):
        if self._exit_callback is not None:
            self._exit_callback()

        if self._suppress_exceptions:
            return True

class ExceptionContext:
    def __init__(self, suppress_exceptions=True, printer=None):
        if printer is None:
            printer = Printer()
        self._suppress_exceptions = suppress_exceptions
        self._print = printer

    def __enter__(self):
        return

    def __exit__(self, *args):
        if args[0] is not None:
            self._print("%s: An exception occured:" % datetime.datetime.now())
            self._print("".join(traceback.format_exception(*args)))
            if self._suppress_exceptions:
                self._print("Suppressing exception and continuing...")
                return True

class StoreDictContext:
    def __init__(self, input_dict: dict, *keys):
        self._input_dict = input_dict
        self._keys = keys
        self._stored_dict = dict()

    def __enter__(self):
        for k in self._keys:
            if k in self._input_dict:
                self._stored_dict[k] = self._input_dict.pop(k)

    def __exit__(self, *args):
        for k in self._keys:
            if k in self._stored_dict:
                self._input_dict[k] = self._stored_dict.pop(k)

        assert len(self._stored_dict) == 0
