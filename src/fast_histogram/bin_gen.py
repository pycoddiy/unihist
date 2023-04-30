import numpy as np


def random_bins(nbins, width, min_width):
    """
    Generate random bins randomly
    :param nbins: Total number of bins
    :param width: Total width of all bins
    :param min_width: Minimal bin width
    :return: Bins
    """
    if min_width*nbins > width:
        raise ValueError("Expect min_width*nbins <= width")

    bins = np.empty(nbins+1)
    bins[1:] = np.random.uniform(0.0, width, nbins)
    bins.sort()
    bins[0] = 0.0
    bins[-1] = width

    widths = np.empty(nbins)
    for i in range(nbins):
        widths[i] = bins[i+1] - bins[i]

    for i in range(nbins-1):
        if widths[i] < min_width:
            w = widths[i]
            widths[i] = min_width
            dw = min_width - w
            widths[i+1] -= dw

    for i in range(nbins-1, 0, -1):
        if widths[i] < min_width:
            w = widths[i]
            widths[i] = min_width
            dw = min_width - w
            widths[i-1] -= dw

    return widths


def almost_uniform_bins(nbins, width, width_variation):
    """
    Generate random bins randomly
    :param nbins: Total number of bins
    :param width: Total width of all bins
    :param width_variation: Random variation from exactly uniform mesh
    :return: Bins
    """

    if width_variation > width/nbins:
        raise ValueError("Expect width_variation < width/nbins")

    bins = np.ones(nbins) * width/nbins
    u = np.random.uniform(-width_variation, width_variation, nbins-1)

    for i in range(nbins-1):
        bins[i] += u[i]
        bins[i+1] -= u[i]
    return bins


def worst_bins(nbins, width, min_width, left=True):
    """
    Generate worst case bin allocation
    :param nbins: Total number of bins
    :param width: Total width of all bins
    :param min_width: Minimal bin width
    :param left (default True): Allocate smallest bins on the left. On the right otherwise
    :return: Bins
    """
    if min_width*nbins > width:
        raise ValueError("Expect min_width*nbins <= width")

    bins = np.empty(nbins)
    s = 0.0
    if left:
        for i in range(nbins-1):
            bins[i] = min_width
            s += min_width
        bins[-1] = width - s
    else:
        for i in range(nbins-1):
            bins[-i-1] = min_width
            s += min_width
        bins[0] = width - s
    return bins


def construct_monotonic_bins(bin_widths, b0=0.0):
    """
    Construct monotonically increasing bin edges from array of bin widths
    :param bin_widths: Array of bin widths
    :param b0: (Optional) Leftmost bin coordinate
    :return: monotonic bins
    """
    nbins = bin_widths.shape[0]
    mbins = np.empty(nbins+1)
    mbins[0] = b0
    for i in range(nbins):
        mbins[i+1] = mbins[i] + bin_widths[i]
    return mbins


if __name__ == "__main__":
    bins = random_bins(4, 100.0, 1.0)
    print(bins, bins.sum())

    bins = worst_bins(4, 100.0, 1.0)
    print(bins, bins.sum())

    bins = almost_uniform_bins(4, 100.0, 10.0)
    print(bins, bins.sum())

    mbins = construct_monotonic_bins(bins)
    print(mbins)
