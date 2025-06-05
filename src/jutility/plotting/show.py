import matplotlib.pyplot as plt
import PIL.Image
import IPython.display

def show_ipython(full_path: str, close_plt=True) -> IPython.display.Image:
    if close_plt:
        close_all()
    with open(full_path, "rb") as f:
        image_bytes = f.read()

    return IPython.display.Image(image_bytes)

def show_pil(full_path: str) -> PIL.Image.Image:
    im = PIL.Image.open(full_path)
    im.show()
    return im

def close_all():
    plt.close("all")
