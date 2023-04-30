import numpy as np
from hist_scalar import get_bin

N = 2000


def random_choice_with_replacement(a, p, size=1):
    k = len(p)
    b = np.empty(k+1)
    b[0] = 0
    for i in range(1, k+1):
        b[i] = b[i-1] + p[i-1]

    x = np.random.random(size=size)
    indices = np.empty(shape=size, dtype=int)
    for j in range(size):
        indices[j] = get_bin(x[j], b)
    return [a[j] for j in indices]


def random_choice_without_replacement(a, p, size=1):
    weights = p.copy()
    pos = range(len(a))
    indices = []
    remaining = size - len(indices)
    while remaining > 0:
        for i in random_choice_with_replacement(pos, weights, size=remaining):
            if p[i] > 0.0:
                p[i] = 0.0
                indices.append(i)
        remaining = size - len(indices)
    return [a[i] for i in indices]


if __name__ == "__main__":
    population = ["a", "b", "c", "d"]
    probability = np.array([0.02, 0.03, 0.25, 0.7])
    np.random.seed(7777777)

    b = random_choice_with_replacement(population, probability, size=N)

    print(b)

    unique, counts = np.unique(b, return_counts=True)
    freqs = dict(zip(unique, counts/N))
    print(freqs)

    b = random_choice_without_replacement(population, probability, size=len(population))
    print(b)
