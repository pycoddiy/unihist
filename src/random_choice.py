from fast_histogram.bin_gen import worst_bins, construct_monotonic_bins
import numpy as np
from fast_histogram.rng_runner import construct_default, construct_without_numpy, run_rng, print_results, print_implementations, get_all_implementations

N = 1024*1024*200

population = np.array([*range(100)])
probability = np.array([0.01]*100)
np.random.seed(7777777)

get_all_implementations(True)
print_implementations()
implementations = construct_default()
cpu_results = run_rng(population, N, probability, "cpu", implementations)
gpu_results = run_rng(population, N, probability, "gpu", implementations)

print()
print_results(list(cpu_results.items()) + list(gpu_results.items()), N)