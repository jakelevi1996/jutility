import os
import sys
from jutility.util.str_util import has_whitespace

def get_argv_str(basename_first_arg: bool=False) -> str:
    if basename_first_arg:
        arg_list = [os.path.basename(sys.argv[0]), *sys.argv[1:]]
    else:
        arg_list = sys.argv

    return " ".join(
        (("'%s'" % s) if has_whitespace(s) else s)
        for s in arg_list
    )

def get_program_command() -> str:
    return " ".join([sys.executable, get_argv_str()])
