import os

def add_filename_suffix(full_path: str, suffix: str) -> str:
    dir_name, tail = os.path.split(full_path)
    root, ext = os.path.splitext(tail)
    new_tail = root + suffix + ext
    new_full_path = os.path.join(dir_name, new_tail)
    return new_full_path
