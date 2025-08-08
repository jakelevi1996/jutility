import numpy as np

def confidence_bounds(
    data_list:  list,
    n_sigma:    float=1.0,
    log:        bool=False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if log:
        data_list = [np.log(x) for x in data_list]

    mean = np.array([np.mean(x) for x in data_list])
    std  = np.array([np.std( x) for x in data_list])
    ucb = mean + (n_sigma * std)
    lcb = mean - (n_sigma * std)

    if log:
        mean, ucb, lcb = np.exp([mean, ucb, lcb])

    return mean, ucb, lcb

def summarise(
    data_array: np.ndarray,
    split_dim:  int=0,
    num_split:  int=50,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_split = min(num_split, data_array.shape[split_dim])
    data_list = np.array_split(data_array, num_split, split_dim)
    return confidence_bounds(data_list, **kwargs)
