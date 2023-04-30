from time import time
from fast_histogram.bin_gen import worst_bins, construct_monotonic_bins
import numpy as np
from fast_histogram.hist_runner import construct_default, construct_without_numpy, run_histograms, print_results, print_implementations


N = 1024*1000*100
N_BINS = [100, 1000]
WIDTH = 1000.0
MIN_WIDTH = [0.1, 0.01, 0.001]

print("==================================================")
print("WORST-CASE BIN DISTRIBUTION PERFORMANCE EVALUATION")
print("==================================================")

print_implementations()
implementations = construct_default()
np.random.seed(0)
x = np.random.uniform(0.0, WIDTH, N).astype("float32")

for nbins in N_BINS:
    for min_width in MIN_WIDTH:
        np.random.seed(0)
        bins = worst_bins(nbins, WIDTH, min_width)
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
        print(f"n={N}\tnbins={nbins}\twidth={WIDTH}\tmin_width={min_width}")
        print("****************************************************************")

        print_results(list(cpu_results.items()) + list(gpu_results.items()), N)
        print()