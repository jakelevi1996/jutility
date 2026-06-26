import os
import urllib.request
from jutility.util.save_load import load_text

def load_or_download(
    path:   str,
    url:    str,
) -> str:
    if not os.path.isfile(path):
        print("\"%s\" not found, downloading from \"%s\"..." % (path, url))
        dir_name = os.path.dirname(path)
        if (not os.path.isdir(dir_name)) and (len(dir_name) > 0):
            os.makedirs(dir_name)

        urllib.request.urlretrieve(url, path)

    return load_text(path)
