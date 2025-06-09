import os
import numpy as np
import PIL.Image
from jutility import util
from jutility.plotting.multiplot import MultiPlot
from jutility.plotting.plot import plot

class Gif:
    def __init__(self, frame_list: (list[PIL.Image.Image] | None)=None):
        if frame_list is None:
            frame_list = []

        self._frame_list = frame_list

    def add_pil_image_frame(self, pil_image: PIL.Image.Image):
        self._frame_list.append(pil_image)

    def add_image_file_frame(self, full_path: str):
        self.add_pil_image_frame(PIL.Image.open(full_path))

    def add_rgb_bytes_frame(self, rgb_bytes, width, height):
        pil_image = PIL.Image.frombytes(
            mode="RGB",
            size=[width, height],
            data=rgb_bytes,
        )
        self.add_pil_image_frame(pil_image)

    def add_rgba_bytes_frame(self, rgba_bytes, width, height):
        pil_image = PIL.Image.frombytes(
            mode="RGBA",
            size=[width, height],
            data=rgba_bytes,
        )
        self.add_pil_image_frame(pil_image)

    def add_multiplot_frame(self, mp: MultiPlot):
        self.add_pil_image_frame(mp.get_pil_image())

    def add_plot_frame(self, *lines, save=False, **plot_kwargs):
        plot_kwargs.setdefault("save_close", False)
        mp = plot(*lines, **plot_kwargs)
        self.add_multiplot_frame(mp)
        if save:
            mp.save(plot_kwargs.get("plot_name"), plot_kwargs.get("dir_name"))
        mp.close()

    def add_rgb_array_frame(self, ndarray_hwc, vmin=0, vmax=1):
        util.check_type(ndarray_hwc, np.ndarray, "ndarray_hwc")
        if (ndarray_hwc.ndim != 3) or (ndarray_hwc.shape[2] != 3):
            raise ValueError(
                "Expected shape (H, W, C=3), but received shape %s"
                % ndarray_hwc.shape
            )

        ndarray_scaled = 255 * (ndarray_hwc - vmin) / (vmax - vmin)
        ndarray_clipped = np.clip(ndarray_scaled, 0, 255)
        ndarray_int8 = ndarray_clipped.astype(np.uint8)
        pil_image = PIL.Image.fromarray(ndarray_int8, mode="RGB")
        self.add_pil_image_frame(pil_image)

    def add_bw_array_frame(self, ndarray_hw, vmin=0, vmax=1):
        util.check_type(ndarray_hw, np.ndarray, "ndarray_hw")
        if ndarray_hw.ndim != 2:
            raise ValueError(
                "Expected shape (H, W), but received shape %s"
                % ndarray_hw.shape
            )

        ndarray_scaled = 255 * (ndarray_hw - vmin) / (vmax - vmin)
        ndarray_clipped = np.clip(ndarray_scaled, 0, 255)
        ndarray_int8 = ndarray_clipped.astype(np.uint8)
        pil_image = PIL.Image.fromarray(ndarray_int8, mode="L")
        self.add_pil_image_frame(pil_image)

    def add_rgb_array_sequence(self, ndarray_lhwc, vmin=0, vmax=1):
        util.check_type(ndarray_lhwc, np.ndarray, "ndarray_lhwc")
        if (ndarray_lhwc.ndim != 4) or (ndarray_lhwc.shape[3] != 3):
            raise ValueError(
                "Expected shape (L, H, W, C=3), but received shape %s"
                % ndarray_lhwc.shape
            )

        for i in range(ndarray_lhwc.shape[0]):
            self.add_rgb_array_frame(ndarray_lhwc[i], vmin, vmax)

    def add_bw_array_sequence(self, ndarray_lhw, vmin=0, vmax=1):
        util.check_type(ndarray_lhw, np.ndarray, "ndarray_lhw")
        if ndarray_lhw.ndim != 3:
            raise ValueError(
                "Expected shape (L, H, W), but received shape %s"
                % ndarray_lhw.shape
            )

        for i in range(ndarray_lhw.shape[0]):
            self.add_bw_array_frame(ndarray_lhw[i], vmin, vmax)

    def shuffle(self, rng: np.random.Generator=None):
        if rng is None:
            rng = np.random.default_rng()

        perm = rng.permutation(len(self._frame_list)).tolist()
        self._frame_list = [self._frame_list[i] for i in perm]

    def save(
        self,
        output_name=None,
        dir_name=None,
        frame_duration_ms=100,
        optimise=False,
        loop_forever=True,
        n_loops=1,
        verbose=True,
    ):
        if output_name is None:
            output_name = "output"

        self.full_path = util.get_full_path(
            output_name,
            dir_name,
            file_ext="gif",
            verbose=verbose,
        )

        if loop_forever:
            n_loops = 0

        self._frame_list[0].save(
            self.full_path,
            format="gif",
            save_all=True,
            append_images=self._frame_list[1:],
            duration=frame_duration_ms,
            optimise=optimise,
            loop=n_loops,
        )

        return self.full_path
