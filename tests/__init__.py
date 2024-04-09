import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
sys.path.insert(0, SOURCE_DIR)

from jutility import util
util.numpy_set_print_options()

def get_output_dir(*subdir_names):
    return os.path.join(CURRENT_DIR, "Outputs", *subdir_names)
