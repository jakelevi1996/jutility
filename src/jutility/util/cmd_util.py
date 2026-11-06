import subprocess

def run_subprocess(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("> %s" % " ".join(args))
    return subprocess.run(args, **kwargs)
