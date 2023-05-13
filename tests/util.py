import os
import torch
import tests

def get_output_dir(*subdir_names):
    output_dir = os.path.join(tests.CURRENT_DIR, "Outputs", *subdir_names)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    return output_dir
