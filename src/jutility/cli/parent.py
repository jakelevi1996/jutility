from jutility import util
import jutility.cli.arg as arg

class _ArgParent:
    def _init_arg_parent(
        self,
        arg_list:   list["arg.Arg"],
        kwargs:     dict,
    ):
        self._arg_list  = arg_list
        self._kwargs    = kwargs

    def get_value_dict(self) -> dict:
        return self.register_values(dict(), False)

    def get_summary(self, replaces=None) -> str:
        return util.abbreviate_dictionary(
            input_dict=self.register_values(dict(), True),
            key_abbreviations=self.register_tags(dict(), ""),
            replaces=replaces,
        )

    def get_kwargs(self) -> dict:
        return {
            arg.name: arg.init_object()
            for arg in self._arg_list
            if  arg.is_kwarg
        }

    def register_names(
        self,
        arg_dict: dict[str, "arg.Arg"],
        prefix: str,
    ) -> dict[str, "arg.Arg"]:
        for arg in self._arg_list:
            arg.set_full_name(prefix + arg.name)
            arg.store_name(arg_dict)
            arg.register_names(arg_dict, arg.full_name + ".")

        return arg_dict

    def register_values(self, value_dict: dict, summarise: bool) -> dict:
        for arg in self._get_active_args():
            arg.store_value(value_dict, summarise)
            arg.register_values(value_dict, summarise)

        return value_dict

    def register_tags(
        self,
        tag_dict: dict[str, str],
        prefix: str,
    ) -> dict[str, str]:
        tagged_args = [arg for arg in self._get_active_args() if arg.tagged]
        for arg in tagged_args:
            if arg.tag is not None:
                tag_dict[arg.full_name] = self._make_tag(prefix, arg.tag)

        default_tag_dict = {
            arg.full_name: self._make_tag(prefix, arg.name)
            for arg in tagged_args
            if arg.tag is None
        }
        prefix_dict = util.get_unique_prefixes(
            input_list=default_tag_dict.values(),
            forbidden=set(tag_dict.values()),
            min_len=(len(prefix) + 1),
        )
        for full_name, full_tag in default_tag_dict.items():
            if not self._hide_tag(arg):
                tag_dict[full_name] = prefix_dict[full_tag]

        for arg in tagged_args:
            if self._hide_tag(arg):
                arg.register_tags(tag_dict, prefix)
            else:
                arg.register_tags(tag_dict, tag_dict[arg.full_name])

        return tag_dict

    def _make_tag(self, prefix: str, suffix: str) -> str:
        return prefix + suffix.lower().replace("_", "")

    def _hide_tag(self, arg: "arg.Arg") -> bool:
        return False

    def _get_active_args(self) -> list["arg.Arg"]:
        return self._arg_list

    def __repr__(self):
        return util.format_type(type(self))
