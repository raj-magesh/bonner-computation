import numpy as np


def permutation_test(samples, null_distribution, tail: str = "both"):
    n = len(null_distribution)
    if tail == "left":
        p_value = ((samples >= null_distribution).sum() + 1) / (n + 1)
    elif tail == "right":
        p_value = ((samples <= null_distribution).sum() + 1) / (n + 1)
    elif tail == "both":
        samples = np.abs(samples)
        null_distribution = np.abs(null_distribution)
        p_value = ((samples <= null_distribution).sum() + 1) / (n + 1)
    else:
        raise ValueError("tail must be `left`, `right` or `both`")
    return p_value
