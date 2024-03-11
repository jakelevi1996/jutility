import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "src"))

sys.path.insert(0, SOURCE_DIR)

import subprocess
from jutility import util

def main():
    package_info = util.load_text("setup.cfg")
    version = util.extract_substring(package_info, "version = ", "\n")
    wheel_name = "dist/jutility-%s-py3-none-any.whl" % version
    python_exe = sys.executable

    build_args = [python_exe, "-m", "build"]
    install_args = [
        python_exe,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        wheel_name,
    ]
    run_command(build_args)
    run_command(install_args)

def run_command(args):
    print("Running command `%s`..." % " ".join(args))
    subprocess.run(args, check=True)
    print("Finished command `%s`\n" % " ".join(args))

if __name__ == "__main__":
    with util.Timer("build_local.py"):
        main()
