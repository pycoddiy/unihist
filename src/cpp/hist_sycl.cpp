#include <iostream>
#include <algorithm>
#include <chrono>
#include <memory>
#include <CL/sycl.hpp>
#include <array>
#include <stdexcept>
#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <deque>

namespace
{
template<class NumT, class DenT>
NumT DivUp(NumT numerator, DenT denominator)
{
    return (numerator + denominator - 1)/denominator;
}

template<class VT, class BT>
VT Multiple(VT value, BT base)
{
    return base*DivUp(value, base);
}

template<class DataType, int Dims>
using localMem = sycl::accessor<DataType, Dims, sycl::access::mode::read_write, sycl::access::target::local>;

constexpr sycl::specialization_id<int> const_work_per_item;
constexpr sycl::specialization_id<int> const_local_hist_count;

template<class FuncPtr, template<class, class> class Wrapper>
struct func_dispatcher
{
    std::array<FuncPtr, 8> table;

    func_dispatcher(sycl::queue)
    {
        auto double_support = false;

        table[get_id(UAR_FLOAT, UAR_INT)] = Wrapper<float, int32_t>::call;
        table[get_id(UAR_FLOAT, UAR_UINT)] = Wrapper<float, uint32_t>::call;
        table[get_id(UAR_FLOAT, UAR_LONG)] = Wrapper<float, int64_t>::call;
        table[get_id(UAR_FLOAT, UAR_ULONG)] = Wrapper<float, uint64_t>::call;

        if (double_support)
        {
            table[get_id(UAR_DOUBLE, UAR_INT)] = Wrapper<double, int32_t>::call;
            table[get_id(UAR_DOUBLE, UAR_UINT)] = Wrapper<double, uint32_t>::call;
            table[get_id(UAR_DOUBLE, UAR_LONG)] = Wrapper<double, int64_t>::call;
            table[get_id(UAR_DOUBLE, UAR_ULONG)] = Wrapper<double, uint64_t>::call;
        }
    }

    int get_id(int DataType, int HistType) const
    {
        int id = 0;
        if (HistType == UAR_INT)
            { /*do nothing*/ }
        else if (HistType == UAR_UINT)
            id = 1;
        else if (HistType == UAR_LONG or HistType == UAR_LONGLONG)
            id = 2;
        else if (HistType == UAR_ULONG or HistType == UAR_ULONGLONG)
            id = 3;
        else
            throw new std::runtime_error("Unsupported data type");


        if (DataType == UAR_FLOAT)
            { /*do nothing*/ }
        else if (HistType == UAR_DOUBLE)
            id += 4;
        else
            throw new std::runtime_error("Unsupported data type");

        return id;
    }

    FuncPtr operator()(int DataType, int HistType) const
    {
        return table[get_id(DataType, HistType)];
    }
};

}

namespace BinarySearchHistogram
{

template<class DataType, class HistType>
class histogram_kernel;

template<class DataType, class HistType = int64_t>
void histogram(sycl::queue queue, DataType* data, uint64_t data_size, DataType* bins, int bins_count, HistType* hist, uint32_t work_per_item, uint32_t local_size)
{
    auto global_size = Multiple(DivUp(data_size, work_per_item), local_size);
    int local_hist_count = std::max(std::min(int(std::ceil((float(4*local_size)/bins_count))), 4), 1);

    queue.submit([&](sycl::handler& cgh)
    {
        cgh.set_specialization_constant<const_work_per_item>(work_per_item);
        cgh.set_specialization_constant<const_local_hist_count>(local_hist_count);

        auto local_hist = localMem<HistType, 2>(sycl::range<2>(local_hist_count, bins_count), cgh);

        cgh.parallel_for<histogram_kernel<DataType, HistType>>(sycl::nd_range<1>(sycl::range(global_size), sycl::range(local_size)),[=](sycl::nd_item<1> item, sycl::kernel_handler h)
        {
            auto id = item.get_group_linear_id();
            auto lid = item.get_local_linear_id();

            auto minVal = bins[0];
            auto maxVal = bins[bins_count - 1];

            auto group = item.get_group();

            int LocalHistCount = h.get_specialization_constant<const_local_hist_count>();

            for (int i = lid; i < (bins_count - 1); i += local_size)
            {
                for (int lhc = 0; lhc < LocalHistCount; ++lhc)
                {
                    local_hist[lhc][i] = 0;
                }
            }
            sycl::group_barrier(group, sycl::memory_scope::work_group);

            int WorkPI = h.get_specialization_constant<const_work_per_item>();

            for (int i = 0; i < WorkPI; ++i)
            {
                auto data_idx = id*WorkPI*local_size + i*local_size + lid;
                auto d = data_idx < data_size ? data[data_idx] : maxVal;

                if (d >= minVal or d < maxVal)
                {
                    int bin = std::upper_bound(bins, bins + bins_count, d) - bins - 1;
                    int lhi = LocalHistCount == 1 ? 0 : lid%LocalHistCount;
                    sycl::atomic_ref<HistType, sycl::memory_order::relaxed, sycl::memory_scope::work_group> lh(local_hist[lhi][bin]);
                    ++lh;
                }
            }

            sycl::group_barrier(group, sycl::memory_scope::work_group);

            for (int i = lid; i < (bins_count - 1); i += local_size)
            {
                auto lh_value = local_hist[0][i];
                for (int lhc = 1; lhc < LocalHistCount; ++lhc)
                {
                    lh_value += local_hist[lhc][i];
                }
                if (lh_value > 0)
                {
                    sycl::atomic_ref<HistType, sycl::memory_order::relaxed, sycl::memory_scope::device> gh(hist[i]);
                    gh += lh_value;
                }
            }
        });
    }).wait();
}

using hist_func_t = void(*)(dpctl::tensor::usm_ndarray, dpctl::tensor::usm_ndarray, dpctl::tensor::usm_ndarray, int, int);

template<class DataType, class HistType>
struct call_wrapper
{
    static void call(dpctl::tensor::usm_ndarray data, dpctl::tensor::usm_ndarray bins, dpctl::tensor::usm_ndarray hist, int work_per_item, int local_size)
    {
        auto queue = data.get_queue();
        int data_size = data.get_shape(0);
        int bins_count = bins.get_shape(0);

        histogram(queue,
                data.get_data<DataType>(),
                data_size,
                bins.get_data<DataType>(),
                bins_count,
                hist.get_data<HistType>(),
                work_per_item,
                local_size);
    }
};

static std::unique_ptr<func_dispatcher<hist_func_t, call_wrapper>> dispatcher;

void py_histogram(dpctl::tensor::usm_ndarray data, dpctl::tensor::usm_ndarray bins, dpctl::tensor::usm_ndarray hist, int work_per_item, int local_size)
{
    if (dispatcher == nullptr)
    {
        auto queue = data.get_queue();
        dispatcher.reset(new func_dispatcher<hist_func_t, call_wrapper>(queue));
    }

    (*dispatcher)(data.get_typenum(), hist.get_typenum())(data, bins, hist, work_per_item, local_size);
}

}

namespace UniformMeshHistogram
{

constexpr sycl::specialization_id<int> const_linear_size;

template<class DataType>
struct UniformMeshLinearSearch
{
    DataType u_diff;
    int n_bins;
    DataType lower_bound;
    std::vector<int> offsets;
    std::vector<int> counts;
    std::vector<std::unique_ptr<UniformMeshLinearSearch<DataType>>> next_mesh;

    int Size = 0;

    int depth = 0;
    int total_size = 0;
    int uniform_mesh_count = 0;

    UniformMeshLinearSearch(const std::vector<DataType> bins, int LinearSize)
    {
        Size = LinearSize;
        n_bins = bins.size() - 1;
        offsets.resize(n_bins);
        counts.resize(n_bins);
        next_mesh.resize(n_bins);

        u_diff = bins.back() - bins.front();
        lower_bound = bins[0];

        uniform_mesh_count = 1;
        total_size = n_bins;

        int curr_nun_bin = 0;
        for (int i = 0; i < n_bins; ++i)
        {
            DataType curr_lo_bound = i*u_diff/n_bins;
            DataType curr_up_bound = (i+1)*u_diff/n_bins;

            int nun_bins_count = 0;
            offsets[i] = curr_nun_bin;

            while (((bins[curr_nun_bin + 1] - lower_bound) < curr_up_bound) and (curr_nun_bin != (n_bins - 1)))
            {
                ++curr_nun_bin;
                ++nun_bins_count;
            }

            counts[i] = nun_bins_count;

            if (nun_bins_count > (Size - 2))
            {
                int new_bins_start = offsets[i];
                int new_bins_end = new_bins_start + nun_bins_count + 1;
                new_bins_end += (bins[new_bins_end] < curr_up_bound);
                std::vector<DataType> new_bins(&bins[new_bins_start], &bins[new_bins_end]);
                new_bins.front() = curr_lo_bound + lower_bound;
                new_bins.back() =  curr_up_bound + lower_bound;

                next_mesh[i].reset(new UniformMeshLinearSearch<DataType>(new_bins, Size));

                depth = std::max(1 + next_mesh[i]->depth, depth);
                total_size += next_mesh[i]->total_size;
                uniform_mesh_count += next_mesh[i]->uniform_mesh_count;
            }
        }
    }

    int _get_non_uniform_bin(const DataType value, int start_offset)
    {
        int u_bin = std::floor((value - lower_bound)*n_bins/u_diff);
        u_bin = std::min(u_bin, n_bins - 1);
        int count = counts[u_bin];
        int offset = start_offset + offsets[u_bin];

        if (count > (Size - 2))
        {
            offset = next_mesh[u_bin]->_get_non_uniform_bin(value, offset);
        }

        return offset;
    }

    int get_non_uniform_bin(const DataType value, const std::vector<DataType>& aligned_bins)
    {
        auto offset = _get_non_uniform_bin(value, 0);

        int _offset = 0;
        for (int i = 0; i < Size; ++i)
            _offset += (value >= aligned_bins[offset + i]);

        return _offset + offset - 1;
    }

    std::vector<DataType> align_bins(const std::vector<DataType>& bins)
    {
        std::vector<DataType> _bins(bins.size() + Size, bins.back());
        std::copy_n(bins.begin(), bins.size(), _bins.begin());

        return _bins;
    }
};

template<class DataType, class HistType>
class histogram_kernel;

template<class DataType, class HistType = uint64_t>
void histogram(sycl::queue queue,
               DataType* bins,
               int bins_count,
               DataType* data,
               uint64_t data_size,
               HistType* hist,
               DataType* u_diff,
               int* n_bins,
               DataType* lower_bound,
               int* device_offsets_start,
               int* offsets,
               int* next_mesh,
               int mesh_count,
               int ofssets_count,
               int linear_size,
               int work_per_item,
               int local_size)
{
    auto global_size = Multiple(DivUp(data_size, work_per_item), local_size);
    int local_hist_count = std::max(std::min(int(std::ceil((float(4*local_size)/bins_count))), 4), 1);

    queue.submit([&](sycl::handler& cgh)
    {
        cgh.set_specialization_constant<const_work_per_item>(work_per_item);
        cgh.set_specialization_constant<const_linear_size>(linear_size);
        cgh.set_specialization_constant<const_local_hist_count>(local_hist_count);

        auto local_offset = localMem<int, 1>(sycl::range<1>(ofssets_count), cgh);
        auto local_hist = localMem<HistType, 2>(sycl::range<2>(local_hist_count, bins_count), cgh);

        cgh.parallel_for<histogram_kernel<DataType, HistType>>(sycl::nd_range<1>(sycl::range(global_size), sycl::range(local_size)),[=](sycl::nd_item<1> item, sycl::kernel_handler h)
        {
            auto id = item.get_group_linear_id();
            auto lid = item.get_local_linear_id();
            auto group = item.get_group();

            for (int i = lid; i < ofssets_count; i += local_size)
            {
                local_offset[i] = offsets[i];
            }

            int LocalHistCount = h.get_specialization_constant<const_local_hist_count>();
            for (int i = lid; i < (bins_count - 1); i += local_size)
            {
                for (int lhc = 0; lhc < LocalHistCount; ++lhc)
                {
                    local_hist[lhc][i] = 0;
                }
            }
            sycl::group_barrier(group, sycl::memory_scope::work_group);

            auto lower_bound_0 = lower_bound[0];
            auto n_bins_0 = n_bins[0];
            auto u_diff_0 = u_diff[0];

            auto minVal = bins[0];
            auto maxVal = bins[bins_count - 1];

            int WorkPI = h.get_specialization_constant<const_work_per_item>();
            int LinearSize = h.get_specialization_constant<const_linear_size>();

            for (int wi = 0; wi < WorkPI; ++wi)
            {
                auto data_idx = id*WorkPI*local_size + wi*local_size + lid;
                auto d = data_idx < data_size ? data[data_idx] : maxVal;

                if (d >= minVal and d < maxVal)
                {
                    int u_bin = std::floor((d - lower_bound_0)*n_bins_0/u_diff_0);
                    u_bin = std::min(u_bin, n_bins_0 - 1);
                    int offset = local_offset[u_bin];
                    int next_um = next_mesh[u_bin];

                    while (next_um > 0)
                    {
                        auto u_diff_ = u_diff[next_um];
                        auto n_bins_ = n_bins[next_um];
                        auto lower_bound_ = lower_bound[next_um];
                        auto device_offsets_start_ = device_offsets_start[next_um];

                        u_bin = std::floor((d - lower_bound_)*n_bins_/u_diff_);
                        u_bin = std::min(u_bin, n_bins_ - 1);
                        offset = offset + offsets[device_offsets_start_ + u_bin];
                        next_um = next_mesh[device_offsets_start_ + u_bin];
                    }

                    int _offset = 0;
                    for (int i = 0; i < LinearSize; ++i)
                        _offset += (d >= bins[offset + i]);

                    offset = offset + _offset - 1;

                    int lhi = LocalHistCount == 1 ? 0 : lid%LocalHistCount;
                    sycl::atomic_ref<HistType, sycl::memory_order::relaxed, sycl::memory_scope::work_group> lh(local_hist[lhi][offset]);
                    ++lh;
                }
            }

            sycl::group_barrier(group, sycl::memory_scope::work_group);

            for (int i = lid; i < (bins_count - 1); i += local_size)
            {
                auto lh_value = local_hist[0][i];
                for (int lhc = 1; lhc < LocalHistCount; ++lhc)
                {
                    lh_value += local_hist[lhc][i];
                }
                if (lh_value > 0)
                {
                    sycl::atomic_ref<HistType, sycl::memory_order::relaxed, sycl::memory_scope::device> gh(hist[i]);
                    gh += lh_value;
                }
            }
        });
    }).wait();
}

using hist_func_t = void(*)(dpctl::tensor::usm_ndarray,
                            dpctl::tensor::usm_ndarray,
                            dpctl::tensor::usm_ndarray,
                            dpctl::tensor::usm_ndarray,
                            dpctl::tensor::usm_ndarray,
                            dpctl::tensor::usm_ndarray,
                            dpctl::tensor::usm_ndarray,
                            dpctl::tensor::usm_ndarray,
                            dpctl::tensor::usm_ndarray,
                            int,
                            int,
                            int);

template<class DataType, class HistType>
struct call_wrapper
{
    static void call(dpctl::tensor::usm_ndarray data,
                     dpctl::tensor::usm_ndarray bins,
                     dpctl::tensor::usm_ndarray hist,
                     dpctl::tensor::usm_ndarray u_diff,
                     dpctl::tensor::usm_ndarray n_bins,
                     dpctl::tensor::usm_ndarray lower_bound,
                     dpctl::tensor::usm_ndarray device_offsets_start,
                     dpctl::tensor::usm_ndarray offsets,
                     dpctl::tensor::usm_ndarray next_mesh,
                     int linear_size,
                     int work_per_item,
                     int local_size)
    {
        auto queue = data.get_queue();
        int data_size = data.get_shape(0);
        int bins_count = bins.get_shape(0);
        int mesh_count = n_bins.get_shape(0);
        int offset_count = offsets.get_shape(0);

        histogram(queue,
                  bins.get_data<DataType>(),
                  bins_count,
                  data.get_data<DataType>(),
                  data_size,
                  hist.get_data<HistType>(),
                  u_diff.get_data<DataType>(),
                  n_bins.get_data<int>(),
                  lower_bound.get_data<DataType>(),
                  device_offsets_start.get_data<int>(),
                  offsets.get_data<int>(),
                  next_mesh.get_data<int>(),
                  mesh_count,
                  offset_count,
                  linear_size,
                  work_per_item,
                  local_size);
    }
};

static std::unique_ptr<func_dispatcher<hist_func_t, call_wrapper>> dispatcher;

void py_histogram(dpctl::tensor::usm_ndarray data,
                  dpctl::tensor::usm_ndarray bins,
                  dpctl::tensor::usm_ndarray hist,
                  dpctl::tensor::usm_ndarray u_diff,
                  dpctl::tensor::usm_ndarray n_bins,
                  dpctl::tensor::usm_ndarray lower_bound,
                  dpctl::tensor::usm_ndarray device_offsets_start,
                  dpctl::tensor::usm_ndarray offsets,
                  dpctl::tensor::usm_ndarray next_mesh,
                  int linear_size,
                  int work_per_item,
                  int local_size)
{
    if (dispatcher == nullptr)
    {
        auto queue = data.get_queue();
        dispatcher.reset(new func_dispatcher<hist_func_t, call_wrapper>(queue));
    }

    (*dispatcher)(data.get_typenum(), hist.get_typenum())(
        data,
        bins,
        hist,
        u_diff,
        n_bins,
        lower_bound,
        device_offsets_start,
        offsets,
        next_mesh,
        linear_size,
        work_per_item,
        local_size);
}

template<class DataType>
std::tuple<int, int, int> uniform_mesh_features(sycl::queue queue, DataType* bins, int bins_count, int LinearSize)
{
    std::vector<DataType> host_bins(bins_count);
    queue.copy(bins, host_bins.data(), bins_count).wait();
    auto um = UniformMeshLinearSearch(host_bins, LinearSize);

    return std::make_tuple(um.uniform_mesh_count, um.total_size, bins_count + LinearSize);
}

std::tuple<int, int, int> py_uniform_mesh_features(dpctl::tensor::usm_ndarray bins, int LinearSize)
{
    int um_count = 0;
    int offsets_count = 0;

    auto queue = bins.get_queue();
    auto bins_count = bins.get_shape(0);
    auto type = bins.get_typenum();

    if (type == UAR_FLOAT)
        return uniform_mesh_features(queue, bins.get_data<float>(), bins_count, LinearSize);
    else if (type == UAR_DOUBLE)
        return uniform_mesh_features(queue, bins.get_data<double>(), bins_count, LinearSize);
    else
        throw new std::runtime_error("Unsupported data type");
}

template<class DataType>
struct SyclUniformMeshLinearSearch
{
    std::vector<DataType> u_diff;
    std::vector<int> n_bins;
    std::vector<DataType> lower_bound;
    std::vector<int> offsets_start;

    std::vector<int> offsets;
    std::vector<int> next_mesh;

    DataType* device_u_diff;
    int* device_n_bins;
    DataType* device_lower_bound;
    int* device_offsets_start;

    int* device_offsets;
    int* device_next_mesh;

    void to_device(sycl::queue queue,
                   DataType* device_u_diff,
                   int* device_n_bins,
                   DataType* device_lower_bound,
                   int* device_offsets_start,
                   int* device_offsets,
                   int* device_next_mesh)
    {
        int um_count = u_diff.size();
        int total_size = offsets.size();

        queue.copy(u_diff.data(), device_u_diff, um_count).wait();
        queue.copy(n_bins.data(), device_n_bins, um_count).wait();
        queue.copy(lower_bound.data(), device_lower_bound, um_count).wait();
        queue.copy(offsets_start.data(), device_offsets_start, um_count).wait();

        queue.copy(offsets.data(), device_offsets, total_size).wait();
        queue.copy(next_mesh.data(), device_next_mesh, total_size).wait();
    }

    void walk_uniform_tree(UniformMeshLinearSearch<DataType>& um)
    {
        std::deque<struct UniformMeshLinearSearch<DataType>*> queue;
        queue.push_back(&um);

        while (queue.size() > 0)
        {
            auto curr_um = queue.front();
            queue.pop_front();

            u_diff.push_back(curr_um->u_diff);
            n_bins.push_back(curr_um->n_bins);
            lower_bound.push_back(curr_um->lower_bound);
            offsets_start.push_back(offsets.size());

            offsets.insert(offsets.end(), curr_um->offsets.begin(), curr_um->offsets.end());
            auto& _next_mesh = curr_um->next_mesh;

            for (auto&& nm : _next_mesh)
            {
                if (nm == nullptr)
                {
                    next_mesh.push_back(0);
                }
                else
                {
                    next_mesh.push_back(u_diff.size() + queue.size());
                    queue.push_back(nm.get());
                }
            }
        }
    }

    SyclUniformMeshLinearSearch(const std::vector<DataType>& host_bins, int LinearSize)
    {
        auto um = UniformMeshLinearSearch<DataType>(host_bins, LinearSize);
        auto um_count = um.uniform_mesh_count + 1;
        auto total_count = um.total_size;

        u_diff.reserve(um_count);
        n_bins.reserve(um_count);
        lower_bound.reserve(um_count);
        offsets_start.reserve(um_count);

        offsets.reserve(total_count);
        next_mesh.reserve(total_count);

        walk_uniform_tree(um);
    }

    std::vector<DataType> align_bins(const std::vector<DataType>& bins, int LinearSize)
    {
        std::vector<DataType> _bins(bins.size() + LinearSize, bins.back());
        std::copy_n(bins.begin(), bins.size(), _bins.begin());

        return _bins;
    }
};

template<class DataType>
void make_uniform_mesh(sycl::queue queue,
                       DataType* bins,
                       int bins_count,
                       int LinearSize,
                       DataType* u_diff,
                       int* n_bins,
                       DataType* lower_bound,
                       int* offsets_start,
                       int* offsets,
                       int* next_mesh,
                       DataType* aligned_bins)
{
    std::vector<DataType> host_bins(bins_count);
    queue.copy(bins, host_bins.data(), bins_count).wait();
    auto um = SyclUniformMeshLinearSearch<DataType>(host_bins, LinearSize);
    um.to_device(queue, u_diff, n_bins, lower_bound, offsets_start, offsets, next_mesh);
    auto host_aligned_bins = um.align_bins(host_bins, LinearSize);
    queue.copy(host_aligned_bins.data(), aligned_bins, host_aligned_bins.size()).wait();
}

void py_make_uniform_mesh(dpctl::tensor::usm_ndarray bins,
                          int LinearSize,
                          dpctl::tensor::usm_ndarray u_diff,
                          dpctl::tensor::usm_ndarray n_bins,
                          dpctl::tensor::usm_ndarray lower_bound,
                          dpctl::tensor::usm_ndarray offsets_start,
                          dpctl::tensor::usm_ndarray offsets,
                          dpctl::tensor::usm_ndarray next_mesh,
                          dpctl::tensor::usm_ndarray aligned_bins)
{
    auto queue = bins.get_queue();
    auto type = bins.get_typenum();
    auto bins_count = bins.get_shape(0);

    if (type == UAR_FLOAT)
        make_uniform_mesh(queue,
                          bins.get_data<float>(),
                          bins_count,
                          LinearSize,
                          u_diff.get_data<float>(),
                          n_bins.get_data<int>(),
                          lower_bound.get_data<float>(),
                          offsets_start.get_data<int>(),
                          offsets.get_data<int>(),
                          next_mesh.get_data<int>(),
                          aligned_bins.get_data<float>());
    else if (type == UAR_DOUBLE)
        make_uniform_mesh(queue,
                          bins.get_data<double>(),
                          bins_count,
                          LinearSize,
                          u_diff.get_data<double>(),
                          n_bins.get_data<int>(),
                          lower_bound.get_data<double>(),
                          offsets_start.get_data<int>(),
                          offsets.get_data<int>(),
                          next_mesh.get_data<int>(),
                          aligned_bins.get_data<double>());
    else
        throw new std::runtime_error("Unsupported data type");
}

}

namespace py = pybind11;

PYBIND11_MODULE(sycl_histogram, m) {
    import_dpctl();

    m.def("binary_hist", &BinarySearchHistogram::py_histogram,
          "Histogram binary search",
          py::arg("data"),
          py::arg("bins"),
          py::arg("hist"),
          py::arg("work_per_item"),
          py::arg("local_size"));

    m.def("uniform_mesh_features", &UniformMeshHistogram::py_uniform_mesh_features,
          "Unifrom mesh sizes",
          py::arg("bins"),
          py::arg("linear_size"));

    m.def("make_uniform_mesh", &UniformMeshHistogram::py_make_uniform_mesh,
          "Fills in uniform mesh data",
          py::arg("bins"),
          py::arg("linear_size"),
          py::arg("u_diff"),
          py::arg("n_bins"),
          py::arg("lower_bound"),
          py::arg("device_offsets_start"),
          py::arg("offsets"),
          py::arg("next_mesh"),
          py::arg("aligned_bins"));

    m.def("uniform_mesh_hist", &UniformMeshHistogram::py_histogram,
          "Unifrom mesh histogram",
          py::arg("data"),
          py::arg("bins"),
          py::arg("hist"),
          py::arg("u_diff"),
          py::arg("n_bins"),
          py::arg("lower_bound"),
          py::arg("device_offsets_start"),
          py::arg("offsets"),
          py::arg("next_mesh"),
          py::arg("linear_size"),
          py::arg("work_per_item"),
          py::arg("local_size"));
}
