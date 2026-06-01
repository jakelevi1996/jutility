import sys
from jutility.util.str_util import has_whitespace

def get_argv_str() -> str:
    return " ".join(
        (("'%s'" % s) if has_whitespace(s) else s)
        for s in sys.argv
    )

def get_program_command() -> str:
    return " ".join([sys.executable, get_argv_str()])
