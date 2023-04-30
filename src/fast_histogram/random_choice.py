import numpy as np
#from hist_scalar import get_bin
from fast_histogram.hist_wrapper import BinarySearchHist_name, UnimMeshHist_name, NumPyHist_name


def random_choice_with_replacement(a, p, size=1):
    k = len(p)
    b = np.empty(k+1)
    b[0] = 0
    b[1:] = np.cumsum(p)

    x = np.random.random(size=size)
    indices = np.empty(shape=size, dtype=int)
    # for j in range(size):
    #     indices[j] = get_bin(x[j], b)
    # return [a[j] for j in indices]



if __name__ == "__main__":
    N = 2000

    population = ["a", "b", "c", "d"]
    probability = np.array([0.02, 0.03, 0.25, 0.7])
    np.random.seed(7777777)

    b = random_choice_with_replacement(population, probability, size=N)

    # print(b)

    unique, counts = np.unique(b, return_counts=True)
    freqs = dict(zip(unique, counts/N))
    print(freqs)
