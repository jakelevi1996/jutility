import sys
from jutility import util

OUTPUT_DIR = util.get_test_output_dir("test_util/test_sys_util")

def test_get_argv_str():
    printer = util.Printer("test_get_argv_str", OUTPUT_DIR)

    s = util.get_argv_str()
    printer(s)
    for si in sys.argv:
        assert si in s

    assert isinstance(s, str)

def test_get_program_command():
    printer = util.Printer("test_get_program_command", OUTPUT_DIR)

    s = util.get_program_command()
    printer(s)
    for si in sys.argv:
        assert si in s

    assert isinstance(s, str)
