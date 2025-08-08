from jutility import util

class PropertyDict:
    def __init__(self, **kwargs):
        self._kwargs = self._expand_abbreviations(kwargs)
        self._init_defaults()

    def _expand_abbreviations(self, kwargs: dict) -> dict:
        key_map = self._get_abbreviated_keys_dict()
        kwargs = {
            key_map.get(k, k): v
            for k, v in kwargs.items()
        }
        return kwargs

    def _get_abbreviated_keys_dict(self) -> dict:
        return dict()

    def _init_defaults(self):
        return

    def _has(self, name):
        return (name in self._kwargs)

    def _get(self, name):
        return self._kwargs.pop(name)

    def _get_default(self, name, default):
        return self._kwargs.pop(name, default)

    def _set(self, name, value):
        self._kwargs[name] = value

    def _set_default(self, name, default):
        return self._kwargs.setdefault(name, default)

    def check_unused_kwargs(self):
        if len(self._kwargs) > 0:
            raise ValueError("Invalid options: %s" % self._kwargs)

    def __repr__(self):
        return util.format_type(type(self), **self._kwargs)
