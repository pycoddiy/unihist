import numpy as np

NumPyHist_name = "NUMPYHIST"
BinarySearchHist_name = "BINARYHIST"
UnimMeshHist_name = "UNIHIST"
CuPyHist_name = "CUPYHIST"

class BaseHist:
    name = "BaseHist"
    @staticmethod
    def is_device_supported(device):
        return false

    @staticmethod
    def from_numpy(x, dtype, device):
        return np.asarray(x, dtype=dtype)

    @staticmethod
    def to_numpy(x):
        return x

    def prepare(self, bins):
        self.bins = bins

    def full_name(self):
        return self.name


def create_hist_wrappers(verbose=False):
    class NumpyHist(BaseHist):
        name = NumPyHist_name
        @staticmethod
        def is_device_supported(device):
            return device == "cpu"

        def __call__(self, a):
            return np.histogram(a, self.bins)[0]

    result = {NumpyHist.name: NumpyHist}
    try:
        import dpctl
        from dpctl import tensor as dpt
        from dpctl.enum_types import device_type
        from fast_histogram.sycl.sycl_histogram import binary_hist, uniform_mesh_features, make_uniform_mesh, uniform_mesh_hist

        class SyclHist(BaseHist):
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

        class BinarySearchHist(SyclHist):
            name = BinarySearchHist_name

            def __init__(self, work_per_item=64, local_size=256):
                self.work_per_item = work_per_item
                self.local_size = local_size

            def prepare(self, bins):
                self.bins = bins
                self.hist_size = bins.shape[0] - 1

            def __call__(self, a):
                hist = dpt.zeros(self.hist_size, dtype="int32", device=a.device)
                binary_hist(a, self.bins, hist, self.work_per_item, self.local_size)

                return hist

            def full_name(self):
                return self.name + "[wpi=" + str(self.work_per_item) + ":los=" + str(self.local_size) + "]"

        class UniformMeshHist(SyclHist):
            name = UnimMeshHist_name

            def __init__(self, linear_size=4, work_per_item=64, local_size=256, full_time=False):
                self.linear_size = linear_size
                self.work_per_item = work_per_item
                self.local_size = local_size
                self.full_time = full_time

            def _prepare(self, bins):
                mesh_count, offsets_size, aligned_bins_size = uniform_mesh_features(bins, self.linear_size)

                self.udiff = dpt.empty(mesh_count, dtype=bins.dtype, device=bins.device)
                self.nbins = dpt.empty(mesh_count, dtype="int32", device=bins.device)
                self.lower_bound = dpt.empty(mesh_count, dtype=bins.dtype, device=bins.device)
                self.offsets_start = dpt.empty(mesh_count, dtype="int32", device=bins.device)

                self.offsets = dpt.empty(offsets_size, dtype="int32", device=bins.device)
                self.next_mesh = dpt.empty(offsets_size, dtype="int32", device=bins.device)

                self.aligned_bins = dpt.empty(aligned_bins_size, dtype=bins.dtype, device=bins.device)

                # When UniformMesh is constructed it is represented as a tree,
                # where the root and nodes are UniformMesh and leaves are non-uniform bins
                #
                # UnifromMesh itself has the following fields:
                #   nbins - single value. number of leaves in this Mesh
                #   udiff - single value. interval size of this Mesh.
                #   For root mesh it would be equal to bins[-1] - bins[0]
                #   lower_bound - single value. begin of this Mesh interval. For root mesh it is equal to bins[0]
                #
                # In combination lower_bounds, nbins and udiff are used to calculate corresponding Uniform Bin for the
                # incoming vaue x as u_bin_id = floor((x - lower_bound)*nbins/udiff)
                #
                # Also UnifromMesh contains the following fields:
                #   offsets - array of ints of nbins size. It represent FIRST NON-uniform
                #             bin related to specific UNIFORM bin.
                #             all Non-uniform bins related Uniform bins should be
                #             in range offset[u_bin_id] + linear_size.
                #             If number of Non-uniform bins is greater than
                #             linear_size next level UniformMesh is created
                #   next_mesh - array of pointers to next-level UniformMesh or null
                #             if number of Non-uniform bins is less than linear_size.
                #             Array size is nbins.
                #
                # So UniformMesh consists of 3 single values and 2 arrays of size nbins.
                # In order to use it on GPU we need to flatten the tree into array(s).
                # It results in group of arrays:
                #    udiff - array of udiffs of all UniformMeshes.
                #            The size is total count of UnifromMeshes (mesh_count)
                #    nbins - array of nbins of all UniformMeshes.
                #            The size is total count of UnifromMeshes (mesh_count)
                #    lower_bound - array of lower_bound of all UniformMeshes.
                #            The size is total count of UnifromMeshes (mesh_count)
                #    offsets_start - array of ints.
                #            Specify position in ofssets from which offsets of the specific Mesh starts.
                #            The size is total count of UnifromMeshes (mesh_count)
                #
                #    offsets - array of all offsets of all UniformMeshes.
                #              Since it is flatten into 1d array support array offsets_start is needed
                #              to access offsets related to specific UniformMesh.
                #              The size is sum of all UnifomMesh nbins (offsets_size)
                #    next_mesh - array of ints. It represent index of UniformMesh data in arrays
                #              udiff, nbins, lower_bound and offsets_start.
                #              If zero - current uniform bin is a leaf and no UniformMesh existing for it
                #              The size is sum of all UnifomMesh nbins (offsets_size)
                #
                # aligned_bins - bins padded with extra linear_search values on the right to avoid boundaries check.
                make_uniform_mesh(bins,
                                self.linear_size,
                                self.udiff,
                                self.nbins,
                                self.lower_bound,
                                self.offsets_start,
                                self.offsets,
                                self.next_mesh,
                                self.aligned_bins)

            def prepare(self, bins):
                if not self.full_time:
                    self._prepare(bins)
                else:
                    self.bins = bins

                self.hist_size = bins.shape[0] - 1

            def __call__(self, a):
                if self.full_time:
                    self._prepare(self.bins)

                hist = dpt.zeros(self.hist_size, dtype="int32", device=a.device)
                uniform_mesh_hist(a,
                                self.aligned_bins,
                                hist,
                                self.udiff,
                                self.nbins,
                                self.lower_bound,
                                self.offsets_start,
                                self.offsets,
                                self.next_mesh,
                                self.linear_size,
                                self.work_per_item,
                                self.local_size)

                return hist

            def full_name(self):
                return self.name + "[lis=" + str(self.linear_size) + \
                    ":wpi=" + str(self.work_per_item) + ":los=" + str(self.local_size) + "]"

        result[BinarySearchHist.name] = BinarySearchHist
        result[UniformMeshHist.name] = UniformMeshHist
    except Exception as e:
        if verbose:
            print("Failed to initialize SYCL-based histograms")
            print(e)

    try:
        import cupy as cp

        class CupyHist(BaseHist):
            name = CuPyHist_name
            @staticmethod
            def is_device_supported(device):
                return device == "gpu"

            @staticmethod
            def from_numpy(x, dtype, device):
                return cp.asarray(x, dtype=dtype)

            @staticmethod
            def to_numpy(x):
                return cp.asnumpy(x)

            def __call__(self, a):
                null_stream = cp.cuda.Stream.null
                result = np.histogram(a, self.bins)[0]
                null_stream.synchronize()

                return result

        result[CupyHist.name] = CupyHist
    except Exception as e:
        if verbose:
            print("Failed to initialize cupy histogram")
            print(e)

    return result

