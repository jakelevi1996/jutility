import os
from jutility import util

def test_format_dict():
    dir_name = "tests/outputs/test_util/test_url_util/test_format_dict"
    path = dir_name + "/output.txt"
    if os.path.isfile(path):
        os.remove(path)
    if os.path.isdir(dir_name):
        os.removedirs(dir_name)

    assert not os.path.exists(path)
    assert not os.path.exists(dir_name)

    url = (
        "https://raw.githubusercontent.com/jakelevi1996/jutility/refs/heads"
        "/main/README.md"
    )
    s = util.load_or_download(path, url)

    assert isinstance(s, str)
    assert len(s) > 1e2
    assert len(s) < 1e6

    assert os.path.exists(path)
    assert os.path.exists(dir_name)
