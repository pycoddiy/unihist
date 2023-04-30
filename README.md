# Fast histogram computation for data-parallel devices
We introduce the fast histogram computing algorithm for non-uniform bins that avoids
bin-search and, instead, uses iterative method for bin indices correction.
Approximate bin indices are computed using parallel map operation, which makes the
algorithm suitable for implementation for data-parallel devices. We demonstrate superior
performance of the method using uniform and log-scale map functions, but the method is
easily extensible to other maps. It is also extensible to multi-dimensional histogram
computation. Finally, we demonstrate how the same technique can be applied for generalized
discrete random number generation. Our implementation leverages oneAPI DPC++ compiler to run
the code on both CPU and GPU devices.
**Keywords** — histogram, discrete random number generation, data-parallel computing, GPU,
SYCL, oneAPI, Data Parallel Extensions for Python, NumPy, Numba

## Algorithm Idea

Uniform bins
![Uniform bins](https://github.com/samaid/fast-histogram/blob/main/images/uniform_bins.png)

Non-uniform bins
![Non-uniform bins](https://github.com/samaid/fast-histogram/blob/main/images/nonuniform_bins.png)

Step 1: Mapping onto non-uniform bins
![Step 1: Mapping onto non-uniform bins](https://github.com/samaid/fast-histogram/blob/main/images/grid_step0.png)

Step 2: Mapping onto non-uniform bins recursion
![Step 2: Mapping onto non-uniform bins recursion](https://github.com/samaid/fast-histogram/blob/main/images/grid_step1.png)

## How to run

### Linux
```
conda create -n hist python=3.10 dpctl cmake ninja pybind11 dpcpp_linux-64 -c intel -c conda-forge
conda activate hist
git clone https://github.com/samaid/fast-histogram.git
cd .fast-histogram/src
python setup.py develop
cd ..
python test/test.py
```
