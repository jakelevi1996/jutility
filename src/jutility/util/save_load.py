import os
import pickle
import json
import numpy as np
import PIL.Image
from jutility.util.str_util import clean_string, trim_string

CURRENT_DIR = os.path.abspath(os.getcwd())
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")

def get_full_path(
    filename,
    dir_name=None,
    file_ext=None,
    loading=False,
    verbose=True,
):
    if dir_name is None:
        dir_name = RESULTS_DIR
    if (not os.path.isdir(dir_name)) and (len(dir_name) > 0):
        os.makedirs(dir_name)

    filename = clean_string(filename)
    filename = trim_string(filename, 240 - len(os.path.abspath(dir_name)))

    if file_ext is not None:
        filename = "%s.%s" % (filename, file_ext)

    full_path = os.path.join(dir_name, filename)

    if verbose:
        action_str = "Loading from" if loading else "Saving in"
        print("%s \"%s\"" % (action_str, full_path))

    return full_path

def save_text(s, filename, dir_name=None, file_ext="txt", verbose=True):
    full_path = get_full_path(filename, dir_name, file_ext, verbose=verbose)
    with open(full_path, "w") as f:
        print(s, file=f)

    return full_path

def load_text(full_path):
    with open(full_path, "r") as f:
        s = f.read()

    return s

def save_pickle(data, filename, dir_name=None, verbose=True, **kwargs):
    full_path = get_full_path(filename, dir_name, "pkl", verbose=verbose)
    with open(full_path, "wb") as f:
        pickle.dump(data, f, **kwargs)

    return full_path

def load_pickle(full_path):
    with open(full_path, "rb") as f:
        data = pickle.load(f)

    return data

def save_json(data, filename, dir_name=None, verbose=True, **kwargs):
    full_path = get_full_path(filename, dir_name, "json", verbose=verbose)
    kwargs.setdefault("indent", 4)
    kwargs.setdefault(
        "default",
        (lambda x: x.tolist() if isinstance(x, np.ndarray) else None),
    )
    with open(full_path, "w") as f:
        json.dump(data, f, **kwargs)

    return full_path

def load_json(full_path):
    with open(full_path, "r") as f:
        data = json.load(f)

    return data

def save_image(
    image_uint8: np.ndarray,
    filename: str,
    dir_name: str=None,
    verbose: bool=True,
):
    if image_uint8.dtype != np.uint8:
        im_ge0 = image_uint8 - np.min(image_uint8)
        im_255 = im_ge0 * (255 / np.max(im_ge0))
        image_uint8 = np.uint8(im_255)

    shape = image_uint8.shape
    mode_dict_1 = {(3, 3): "RGB", (3, 4): "RGBA"}
    mode_dict_2 = {2: "L"}
    mode = mode_dict_1.get(
        (len(shape), shape[-1]),
        mode_dict_2.get(len(shape)),
    )

    pil_image = PIL.Image.fromarray(image_uint8, mode=mode)
    full_path = get_full_path(filename, dir_name, "png", verbose=verbose)
    pil_image.save(full_path)
    return full_path

def save_image_diff(
    full_path_1:    str,
    full_path_2:    str,
    output_name:    str="diff",
    dir_name:       (str | None)=None,
    normalise:      bool=True,
):
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

def load_image(full_path) -> np.ndarray:
    image_uint8 = np.array(PIL.Image.open(full_path))
    return image_uint8
