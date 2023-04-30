from time import time
from fast_histogram.bin_gen import almost_uniform_bins, construct_monotonic_bins
import numpy as np
from fast_histogram.hist_runner import construct_default, construct_without_numpy, run_histograms, print_results, print_implementations


N = 1024*1000*100
N_BINS = [100, 1000]
WIDTH = 1000.0
WIDTH_VARIATION = [0.1, 0.01, 0.001]

print("======================================================")
print("ALMOST UNIFORM BIN DISTRIBUTION PERFORMANCE EVALUATION")
print("======================================================")

print_implementations()
implementations = construct_default()
np.random.seed(0)
x = np.random.uniform(0.0, WIDTH, N).astype("float32")

for nbins in N_BINS:
    for width_variation in WIDTH_VARIATION:
        np.random.seed(0)
        bins = almost_uniform_bins(nbins, WIDTH, width_variation)
        bins = construct_monotonic_bins(bins)

        try:
            cpu_results = run_histograms(x, bins, "cpu", implementations)
        except:
            print("Failed to run cpu implementations...")
            raise

        try:
            gpu_results =  run_histograms(x, bins, "gpu", implementations)
        except:
            print("Failed to run gpu implementations...")
            raise

        print()
        print("****************************************************************")
        print(f"n={N}\tnbins={nbins}\twidth={WIDTH}\twidth_variation={width_variation}")
        print("****************************************************************")

        print_results(list(cpu_results.items()) + list(gpu_results.items()), N)
        print()