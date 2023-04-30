import numpy as np
from math import floor


def get_bin(x, bins, idx_offset=0):
    """
    Based on scalar input x this functions calculates the index of the bin,
    to which it belongs.
    :param x: Scalar input. Assumption is that bins[0] <= x < bins[n]. If min(x) and/or max(x) are unknown
              bins[0] can be considered -inf and/or bins[n] can be considered +inf respectively
    :param bins: Array representing bins. Interval [bin[i], bin[i+1]),
              where bin[i] < bin[i+1], represent i-th bin
    :param idx_offset (Optional): Bin index offset, By default the first bin index is 0. If you want
              bin enumeration to start with a different index, specify it by setting idx_offset to that value
    :return: Bin index idx to which x belongs, i.e. bin[idx] <= x < bin[idx+1]
    """
    n = bins.shape[0] - 1
    if n <= 0:
        raise ValueError("The number of bins must be 1 or more")
    elif n == 1:
        return idx_offset
    elif n == 2:
        return idx_offset if x < bins[1] else idx_offset+1
    else:
        counts = np.zeros(n, dtype=int)
        offsets = np.zeros(n, dtype=int)
        u_step = (bins[n] - bins[0]) / n
        k_prev = 0
        for i in range(1, n):
            k = floor((bins[i] - bins[0]) / u_step)
            counts[k] += 1
            offsets[k_prev:k+1] = i-1
            k_prev = k+1
        offsets[k_prev:] = n-1

        ix = floor((x - bins[0]) / u_step)
        if counts[ix] == 0:
            return offsets[ix] + idx_offset
        else:
            offset = offsets[ix]
            sub_n = counts[ix] + 1
            sub_bins = np.zeros(sub_n + 1)
            sub_bins[0] = ix * u_step + bins[0]
            sub_bins[1:sub_n] = bins[offset+1:offset+sub_n]
            sub_bins[-1] = (ix + 1) * u_step + bins[0]
            return get_bin(x, sub_bins, offset) + idx_offset


if __name__ == "__main__":
    bins = np.asarray([0, 21, 25, 28, 44, 47, 57, 70])
    x = 24.2
    j = get_bin(x, bins)
    print(j)
