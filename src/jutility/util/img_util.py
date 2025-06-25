import os
import numpy as np
import PIL.Image
from jutility.util.save_load import load_image, save_image

def trim_image(
    full_path:  str,
    force_udlr: (tuple[int, int, int, int] | None)=None,
    pad:        int=0,
    suffix:     str="_trimmed",
) -> str:
    if force_udlr is None:
        force_udlr = (0, 0, 0, 0)

    a = load_image(full_path)

    fu, fd, fl, fr = force_udlr
    h, w, _ = a.shape
    a = a[fu:(h - fd), fl:(w - fr)]

    mask = np.max(np.std(a, axis=1), axis=1) > 0
    inds = np.arange(a.shape[0])
    keep = inds[mask]
    y_lo = max(np.min(keep) - pad, 0)
    y_hi = min(np.max(keep) + pad + 1, a.shape[0])

    mask = np.max(np.std(a, axis=0), axis=1) > 0
    inds = np.arange(a.shape[1])
    keep = inds[mask]
    x_lo = max(np.min(keep) - pad, 0)
    x_hi = min(np.max(keep) + pad + 1, a.shape[1])

    a = a[y_lo:y_hi, x_lo:x_hi]

    dir_name, base_name = os.path.split(full_path)
    root, _ = os.path.splitext(base_name)
    name = str(root) + str(suffix)

    return save_image(a, name, dir_name)

def save_image_diff(
    full_path_1:    str,
    full_path_2:    str,
    output_name:    str="diff",
    dir_name:       (str | None)=None,
    normalise:      bool=True,
) -> str:
    if dir_name is None:
        dir_name = os.path.dirname(full_path_1)

    x_pil = PIL.Image.open(full_path_1)
    y_pil = PIL.Image.open(full_path_2)
    print("Input sizes = %s and %s" % (x_pil.size, y_pil.size))
    if y_pil.size != x_pil.size:
        print("Resizing %s to %s" % (y_pil.size, x_pil.size))
        y_pil = y_pil.resize(x_pil.size)

    x = np.array(x_pil, dtype=np.float64)
    y = np.array(y_pil, dtype=np.float64)
    z = np.uint8(np.abs(x - y))
    print("Min image difference = %s" % z.min())
    print("Max image difference = %s" % z.max())
    if normalise and (z.max() > 0):
        z = np.float64(z)
        z *= 255 / z.max()
        z = np.uint8(z)
    if (len(z.shape) == 3) and (z.shape[-1] == 4):
        z[:, :, 3] = 255

    return save_image(z, output_name, dir_name, verbose=True)
