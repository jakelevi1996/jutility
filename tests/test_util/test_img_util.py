import numpy as np
from jutility import util

OUTPUT_DIR = util.get_test_output_dir("test_util/test_img_util")

def test_trim_image():
    a = np.zeros([60, 80, 3], dtype=np.uint8)

    x = 13
    y = 17
    w = 21
    h = 23
    g = [0, 123, 0]

    a[y:y+h, x:x+w] = g

    input_path = util.save_image(a, "test_trim_image", OUTPUT_DIR)

    output_path = util.trim_image(input_path)

    a2 = util.load_image(output_path)
    assert isinstance(a2, np.ndarray)
    assert a2.dtype == np.uint8
    assert a2.dtype != np.float64
    assert a2.shape == (h, w, 3)
    assert np.all(a2 == g)

    p = 5

    output_path = util.trim_image(input_path, pad=p, suffix="_t2")

    a2 = util.load_image(output_path)
    assert isinstance(a2, np.ndarray)
    assert a2.dtype == np.uint8
    assert a2.dtype != np.float64
    assert a2.shape == (h + 2*p, w + 2*p, 3)
    assert not np.all(a2 == g)

    output_path = util.trim_image(input_path, pad=1000, suffix="_t3")

    a2 = util.load_image(output_path)
    assert isinstance(a2, np.ndarray)
    assert a2.dtype == np.uint8
    assert a2.dtype != np.float64
    assert a2.shape == a.shape
    assert np.all(a2 == a)
    assert not np.all(a2 == g)

def test_trim_image_force():
    a = np.zeros([60, 80, 3], dtype=np.uint8)

    x = 13
    y = 17
    w = 21
    h = 23
    g = [0, 123, 0]

    fx = 2
    fy = 3

    a[y:y+h, x:x+w] = g
    a[-fy, x, 0] = 5
    a[y, -fx, 1] = 7

    input_path = util.save_image(a, "test_trim_image", OUTPUT_DIR)

    output_path = util.trim_image(input_path)
    a2 = util.load_image(output_path)
    assert a2.shape == (a.shape[0] - y - fy + 1, a.shape[1] - x - fx + 1, 3)

    output_path = util.trim_image(input_path, force_lrud=(0, 1, 0, 0))
    a2 = util.load_image(output_path)
    assert a2.shape == (a.shape[0] - y - fy + 1, a.shape[1] - x - fx + 1, 3)

    output_path = util.trim_image(input_path, force_lrud=(0, fx, 0, 0))
    a2 = util.load_image(output_path)
    assert a2.shape == (a.shape[0] - y - fy + 1, w, 3)

    output_path = util.trim_image(input_path, force_lrud=(0, 0, 0, 1))
    a2 = util.load_image(output_path)
    assert a2.shape == (a.shape[0] - y - fy + 1, a.shape[1] - x - fx + 1, 3)

    output_path = util.trim_image(input_path, force_lrud=(0, 0, 0, fy))
    a2 = util.load_image(output_path)
    assert a2.shape == (h, a.shape[1] - x - fx + 1, 3)

    output_path = util.trim_image(input_path, force_lrud=(0, fx, 0, fy))
    a2 = util.load_image(output_path)
    assert a2.shape == (h, w, 3)
