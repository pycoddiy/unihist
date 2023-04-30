import sys
from itertools import product
from time import time
from fast_histogram.rng_wrappers import create_rng_wrappers, BinarySearchRNG_name, UnimMeshRNG_name, NumPyRNG_name

implementations = None

def run_rng(a, size, p, device, implementations, dtype="float32"):

    results = dict()
    for key, impl in implementations.items():
        if impl.is_device_supported(device):
            print(f"{key} {device}...", end=' ', flush=True)
            a_dev = impl.from_numpy(a, a.dtype, device)
            p_dev = impl.from_numpy(p, dtype, device)
            impl.prepare(p_dev)
            impl(a_dev, size)

            total_time = 0
            time_list = []
            while total_time < 1:
                start = time()
                impl(a_dev, size)
                end = time()
                t = end - start
                total_time += t
                time_list.append(t)

            results[key + " (" + device + ")"] = min(time_list)

    return results


def get_device_implementations(device, verbose=False):
    """
    Returns available histogram implementations: numpy, cupy, binary search and uniform mesh as dictionary
    If conditions are not met (e.g. cupy is missing) implementation would be missing in results

    If verbose=True error message would be printed for unavailable implementations
    If device (cpu or gpu) is specified only implementations which supports specified device would be present in result.
    If device is None result would contain all available implementations

    Examples of usage:
        Impls = get_device_implementations(None)
        impls = construct_default(Impls)
    """
    global implementations
    if verbose or implementations is None:
        implementations = create_rng_wrappers(verbose)

    if device is None:
        return implementations

    return {name: impl for name, impl in implementations.items() if impl.is_device_supported(device)}


def get_all_implementations(verbose=False):
    """
    returns all available implementations for all devices
    """
    return get_device_implementations(None, verbose)


def filter_implementations(include=None, impls=None, exclude=None, verbose=False):
    """
    filters implementations by names with include or exclude parameters.
    Include and exclude parameters assumes exact name match.
    Examples of usage:
        filter_implementations(include=["UNIHIST", "Binary search"])
        filter_implementations(exclude="NumPy")
    """
    if impls is None:
        impls = get_all_implementations()

    if exclude is None:
        exclude = []

    if include is None:
        include = []

    if not isinstance(include, list):
        include = [include]

    if not isinstance(exclude, list):
        exclude = [exclude]

    return {key: value for key, value in impls.items()
            if (key in include or len(include) == 0) and (key not in exclude)}


def print_implementations(impls=None, device=None):
    if impls is None:
        impls = get_all_implementations()

    if device is None:
        device = ["cpu", "gpu"]

    impl_list = []
    for d in device:
        impl_list += [key + " " + d for key, value in impls.items() if value.is_device_supported(d)]

    print("Available implementations: " + ", ".join(impl_list))


def print_results(results, N):
    if isinstance(results, dict):
        results = list(results.items())

    method_width = max([len(key) for (key, value) in results])
    throughput_width = len("THROUGHPUT (Mpts/sec)")
    time_width = len("TIME(sec)")

    print(f"{'METHOD':{method_width}} | {'THROUGHPUT (Mpts/sec)':{throughput_width}} | TIME(sec)")
    print("-"*(method_width + 1) + "+" + '-'*(throughput_width + 2) + "+" + '-'*(time_width + 1))
    for key, value in results:
        print(f"{key:{method_width}} | {f'{N/value*1e-6:.0f}':{throughput_width}} | {value:.3f}")


def construct_default(impls=None):
    """
    Construct histograms from implementations with default parameters
    Example of usage:
        impls = construct_default()
        run_histograms(a, bins, "cpu", impls)
    """
    if impls is None:
        impls = get_all_implementations()

    return {key: impl() for key, impl in impls.items()}


def construct_full_and_pure_time(impls=None):
    """
    Constructs all available implementation with two versions of UNIHIST:
    one which includes uniform mesh building time into measurements ([full time]),
    and one which doesn't ([pure time])

    Uniform mesh building is unoptimised and on gpu it causes significant overhead

    Example of usage:
        impls = construct_full_and_pure_time()
        run_histograms(a, bins, "cpu", impls)
    """
    if impls is None:
        impls = get_all_implementations()

    result = dict()

    for key, impl in impls.items():
        if key != UnimMeshHist_name:
            result[key] = impl()
        else:
            result[key + " [full time]"] = impl(full_time=True)
            result[key + " [pure time]"] = impl(full_time=False)

    return result


def construct_full_and_pure_time_only():
    """
    Constructs only versions of UNIHIST:
    one which includes uniform mesh building time into measurements ([full time]),
    and one which doesn't ([pure time])
    Other implementations would not be present in result

    unifrom mesh building is unoptimised and on gpu it causes signifficant overhead

    Example of usage:
        impls = construct_full_and_pure_time_only()
        run_histograms(a, bins, "cpu", impls)
    """
    impls = filter_implementations(UnimMeshHist_name)

    return construct_full_and_pure_time(impls)


def construct_uniform_mesh_only(linear_size=None, work_per_item=None, local_size=None):
    """
    Constructs only versions of UNIHIST with non-default parameters linear_size, work_per_item and local_size.
    Different devices runs efficiently with different parameters.
    Parameters could be either lists or ints or None.
    Result would contain instances with all combinations of parameters.

    Example of usage:
        impls = construct_uniform_mesh_only()
        run_histograms(a, bins, "cpu", impls)
    """
    if linear_size is None:
        linear_size = [4]

    if work_per_item is None:
        work_per_item = [64]

    if local_size is None:
        local_size = [256]

    if not isinstance(linear_size, list):
        linear_size = [linear_size]

    if not isinstance(work_per_item, list):
        work_per_item = [work_per_item]

    if not isinstance(local_size, list):
        local_size = [local_size]

    impl = get_all_implementations()[UnimMeshHist_name]

    params = product(linear_size, work_per_item, local_size)

    result = dict()
    for lis, wpi, los in params:
        _impl = impl(lis, wpi, los)
        result[_impl.full_name()] = _impl

    return result


def construct_binary_search_only(work_per_item=None, local_size=None):
    """
    Constructs only versions of Binary search with non-default parameters work_per_item and local_size.
    Different devices runs efficiently with different parameters.
    Parameters could be either lists or ints or None.
    Result would contain instances with all combinations of parameters.

    Example of usage:
        impls = construct_binary_search_only()
        run_histograms(a, bins, "cpu", impls)
    """
    if work_per_item is None:
        work_per_item = [64]

    if local_size is None:
        local_size = [256]

    if not isinstance(work_per_item, list):
        work_per_item = [work_per_item]

    if not isinstance(local_size, list):
        local_size = [local_size]

    impl = get_all_implementations()[BinarySearchHist_name]

    params = product(work_per_item, local_size)

    result = dict()
    for wpi, los in params:
        _impl = impl(wpi, los)
        result[_impl.full_name()] = _impl

    return result


def construct_without_numpy(impls=None):
    """
    Constructs all available implementations woth default parameters but without NumPy implementation

    Example of usage:
        impls = construct_without_numpy()
        run_histograms(a, bins, "cpu", impls)
    """
    impls = filter_implementations(impls=impls, exclude=NumPyHist_name)
    return {key: impl() for key, impl in impls.items()}
