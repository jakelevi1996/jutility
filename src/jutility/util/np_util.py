import numpy as np

def numpy_set_print_options(
    precision:  int=3,
    threshold:  (int | float)=1e5,
    linewidth:  (int | float)=1e5,
    suppress:   bool=True,
):
    np.set_printoptions(
        precision=precision,
        threshold=int(threshold),
        linewidth=int(linewidth),
        suppress=suppress,
    )

def is_numeric(x):
    return any(isinstance(x, t) for t in [int, float, np.number])

def log_range(start, stop, num=50, unique_integers=False) -> np.ndarray:
    if unique_integers:
        max_num = int(np.ceil(abs(stop - start) + 1))
        for num_float in range(num, max_num + 1):
            x = log_range(start, stop, num_float, unique_integers=False)
            x = np.unique(np.int64(np.round(x)))
            if len(x) >= num:
                return x

        x = np.linspace(start, stop, max_num)
        x = np.unique(np.int64(np.round(x)))
        return x

    return np.exp(np.linspace(np.log(start), np.log(stop), num))
