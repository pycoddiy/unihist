import numpy as np
from fast_histogram.hist_wrapper import create_hist_wrappers, BinarySearchHist_name, UnimMeshHist_name
import dpnp as dp

NumPyRNG_name = "NUMPYRNG"
BinarySearchRNG_name = "BINARYRNG"
UnimMeshRNG_name = "UNIHISTRNG"
CuPyRNG_name = "CUPYRNG"


class BaseRNG:
    name = "BaseRNG"
    @staticmethod
    def is_device_supported(device):
        return false

    @staticmethod
    def from_numpy(x, dtype, device):
        return np.asarray(x, dtype=dtype)

    @staticmethod
    def to_numpy(x):
        return x

    def prepare(self, p):
        self.p = p


    def full_name(self):
        return self.name


def create_rng_wrappers(verbose=False):
    hist_wrappers = create_hist_wrappers(verbose=False)
    class NumpyRng(BaseRNG):
        name = NumPyRNG_name
        @staticmethod
        def is_device_supported(device):
            return device == "cpu"

        def __call__(self, a, size):
            return np.random.choice(a, size, p=self.p)

    result = {NumpyRng.name: NumpyRng}
    try:
        import dpctl
        from dpctl import tensor as dpt
        from dpctl.enum_types import device_type
        from fast_histogram.sycl.sycl_histogram import binary_hist, uniform_mesh_features, make_uniform_mesh, uniform_mesh_hist

        class SyclRng(BaseRNG):
            def __init__(self, full_time=False):
                self.full_time = full_time

            @staticmethod
            def is_device_supported(device):
                if device == "gpu":
                    return len(dpctl.get_devices(device_type=device_type.gpu)) > 0

                if device == "cpu":
                    return len(dpctl.get_devices(device_type=device_type.cpu)) > 0

                return False

            @staticmethod
            def from_numpy(x, dtype, device):
                return dpt.asarray(x, device=device, dtype=dtype)

            @staticmethod
            def to_numpy(x):
                return dpt.asnumpy(x)

            def _prepare(self, p):
                p_host = self.to_numpy(p)
                p_sum = np.zeros_like(p, shape=p.shape[0] + 1)
                cs = np.cumsum(p_host).astype(p.dtype)
                p_sum[1:] = np.cumsum(p_host).astype(p.dtype)[:]
                p_sum = p_sum.astype(p.dtype)
                bins_device = self.from_numpy(p_sum, dtype=p.dtype, device=p.device)
                self.p = bins_device
                self.hist_impl.prepare(self.p)

            def prepare(self, p):
                self.p = p
                if not self.full_time:
                    self._prepare(p)

            def __call__(self, a, size):
                if self.full_time:
                    self._prepare(self.p)

                x = dp.random.random(size=size, device=a.device).astype(self.p.dtype)
                indices = self.hist_impl(x.get_array())
                i = dp.asarray(indices, dtype=int).get_array()
                return a[i]


        class BinarySearchRng(SyclRng):
            name = BinarySearchRNG_name

            def __init__(self, work_per_item=64, local_size=256, full_time=False):
                hist_class = hist_wrappers[BinarySearchHist_name]
                self.hist_impl = hist_class(work_per_item, local_size)

                self.work_per_item = work_per_item
                self.local_size = local_size
                self.full_time = full_time

                super().__init__(full_time)

            def full_name(self):
                return self.name + "[wpi=" + str(self.work_per_item) + ":los=" + str(self.local_size) + "]"

        class UniformMeshRng(SyclRng):
            name = UnimMeshRNG_name

            def __init__(self, linear_size=4, work_per_item=64, local_size=256, full_time=False):
                hist_class = hist_wrappers[UnimMeshHist_name]
                self.hist_impl = hist_class(linear_size, work_per_item, local_size, full_time)

                self.linear_size = linear_size
                self.work_per_item = work_per_item
                self.local_size = local_size
                self.full_time = full_time

                super().__init__(full_time)

            def full_name(self):
                return self.name + "[lis=" + str(self.linear_size) + \
                    ":wpi=" + str(self.work_per_item) + ":los=" + str(self.local_size) + "]"

        result[BinarySearchRng.name] = BinarySearchRng
        result[UniformMeshRng.name] = UniformMeshRng
    except Exception as e:
        if verbose:
            print("Failed to initialize SYCL-based histograms")
            print(e)

    try:
        import cupy as cp

        class CupyRNG(BaseRNG):
            name = CuPyRNG_name
            @staticmethod
            def is_device_supported(device):
                return device == "gpu"

            @staticmethod
            def from_numpy(x, dtype, device):
                return cp.asarray(x, dtype=dtype)

            @staticmethod
            def to_numpy(x):
                return cp.asnumpy(x)

            def __call__(self, a, size):
                return cp.random.choice(a, size, p=self.p)

        result[CupyRNG.name] = CupyRNG
    except Exception as e:
        if verbose:
            print("Failed to initialize cupy histogram")
            print(e)

    return result
