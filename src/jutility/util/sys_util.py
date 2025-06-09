import sys

def get_argv_str() -> str:
    return " ".join(sys.argv)

def get_program_command() -> str:
    return " ".join([sys.executable, get_argv_str()])
