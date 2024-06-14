import os
from jutility import util

util.numpy_set_print_options()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))

def get_output_dir(*subdir_names):
    return os.path.join(CURRENT_DIR, "Outputs", *subdir_names)
