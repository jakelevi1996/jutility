import os
from jutility.util.save_load import load_image, save_image
import numpy as np

def trim_image(
    full_path:  str,
    pad:        int=0,
    suffix:     str="_trimmed",
) -> str:
    a = load_image(full_path)

    mask = np.std(a, axis=(1, 2)) > 0
    inds = np.arange(a.shape[0])
    keep = inds[mask]
    y_lo = max(np.min(keep) - pad, 0)
    y_hi = min(np.max(keep) + pad + 1, a.shape[0])

    mask = np.std(a, axis=(0, 2)) > 0
    inds = np.arange(a.shape[1])
    keep = inds[mask]
    x_lo = max(np.min(keep) - pad, 0)
    x_hi = min(np.max(keep) + pad + 1, a.shape[1])

    a = a[y_lo:y_hi, x_lo:x_hi]

    dir_name, base_name = os.path.split(full_path)
    root, _ = os.path.splitext(base_name)
    name = str(root) + str(suffix)

    return save_image(a, name, dir_name)
