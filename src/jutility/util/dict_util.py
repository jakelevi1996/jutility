from jutility.util.str_util import clean_string

def format_dict(
    input_dict:     dict,
    item_fmt:       str="%s=%r",
    item_sep:       str=", ",
    prefix:         str="",
    suffix:         str="",
    key_order:      (list[str] | None)=None,
    ignore_keys:    (list[str] | None)=None,
) -> str:
    if key_order is None:
        key_order = sorted(input_dict.keys())
    if ignore_keys is None:
        ignore_keys = []

    items_str = item_sep.join(
        item_fmt % (k, input_dict[k])
        for k in key_order
        if  k not in ignore_keys
    )
    return prefix + items_str + suffix

def format_type(
    input_type:     type,
    *args,
    item_fmt:       str="%s=%r",
    key_order:      (list[str] | None)=None,
    **kwargs,
) -> str:
    prefix = input_type.__name__ + "("
    if len(args) > 0:
        prefix += ", ".join(repr(a) for a in args)
        if len(kwargs) > 0:
            prefix += ", "

    return format_dict(
        input_dict=kwargs,
        item_fmt=item_fmt,
        prefix=prefix,
        suffix=")",
        key_order=key_order,
    )

def abbreviate_dictionary(
    input_dict:         dict,
    key_abbreviations:  dict[str, str],
    replaces:           (dict[str, str] | None)=None,
):
    if replaces is None:
        replaces = {
            "_":        "",
            "FALSE":    "F",
            "TRUE":     "T",
            "NONE":     "N",
        }

    sorted_keys = sorted(
        sorted(set(input_dict.keys()) & set(key_abbreviations.keys())),
        key=lambda k: key_abbreviations[k],
    )
    pairs_list = [
        (key_abbreviations[k].lower() + str(input_dict[k]).upper())
        for k in sorted_keys
    ]
    s_clean = clean_string("".join(pairs_list))
    for k, v in replaces.items():
        s_clean = s_clean.replace(k, v)

    return s_clean
