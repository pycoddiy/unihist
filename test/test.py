import numpy as np
import dpctl.tensor as dpt

import fast_histogram
from fast_histogram.hist_wrapper import UnimMeshHist_name, BinarySearchHist_name, NumPyHist_name
from fast_histogram.hist_runner import construct_default

a = np.random.ranf(1000).astype("float32")
a_device = dpt.asarray(a)

bins = np.asarray([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], dtype="float32")
bins_device = dpt.asarray(bins)

impls = construct_default()

bsh = impls[BinarySearchHist_name]
bsh.prepare(bins_device)
umh = impls[UnimMeshHist_name]
umh.prepare(bins_device)
nph = impls[NumPyHist_name]
nph.prepare(bins)

print("numpy:         ", nph(a))
print("binary search: ", bsh(a_device))
print("uniform mesh:  ", umh(a_device))
