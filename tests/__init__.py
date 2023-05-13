import os
import sys
import numpy as np
from jutility import util

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(SOURCE_DIR)

util.numpy_set_print_options()
