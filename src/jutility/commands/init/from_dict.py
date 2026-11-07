import os
from jutility import util

def init_from_dict(path_contents_dict: dict[str, str]):
    for full_path, content in path_contents_dict.items():
        if not os.path.isfile(full_path):
            print("Saving in \"%s\"" % full_path)

            dir_name = os.path.dirname(full_path)
            if (not os.path.isdir(dir_name)) and (len(dir_name) > 0):
                os.makedirs(dir_name)

            with open(full_path, "w") as f:
                f.write(content)

        else:
            print("Skipping \"%s\" (already exists)" % full_path)

    util.run_subprocess(["code"] + list(path_contents_dict.keys()))
