#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <memory>
#include <tbb/tbb.h>
#include <algorithm>
#include <CL/sycl.hpp>

using DType = float;

int DivUp(int numerator, int denominator)
{
    return (numerator + denominator - 1)/denominator;
}

int Multiple(int value, int base)
{
    return base*DivUp(value, base);
}

template<class DataType, int Dims>
using localMem = sycl::accessor<DataType, Dims, sycl::access::mode::read_write, sycl::access::target::local>;

template<class T> using DeviceMem = std::unique_ptr<T[], std::function<void (T*)>>;

template<class T> DeviceMem<T> malloc_device(int64_t count, cl::sycl::queue& queue)
{
    return DeviceMem<T>(sycl::malloc_device<T>(count, queue), [queue](T* mem)
    {
        sycl::free(mem, queue);
    });
}

template<class DataType>
void print_hist(const std::vector<DataType>& data)
{
    for (auto&& d : data)
        std::cout << d << " ";

    std::cout << std::endl;
}

template<class DataType>
DataType rnd(DataType min_val, DataType max_val)
{
    return min_val + (max_val - min_val)*(static_cast<DataType>(rand()) / static_cast<DataType>(RAND_MAX));
}

template<class DataType>
DataType rnd(DataType max = 1)
{
    return rnd(0, max);
}

//Function to generate data
template<class DataType>
std::vector<DataType> generate_data(int n, DataType min_val = 0, DataType max_val = 1) {
    std::vector<DataType> data(n);
    for (int i = 0; i < n; i++) {
        data[i] = rnd(min_val, max_val);
    }
    return data;
}

//Function to generate histogram bucket bins
template<class DataType>
std::vector<DataType> generate_bins(int num_bins, DataType min_value = 0, DataType max_value = 1) {
    std::vector<DataType> bins(num_bins + 1);
    for (int i = 0; i <= num_bins; i++) {
        bins[i] = min_value + i*(max_value - min_value) / static_cast<DataType>(num_bins);
    }
    return bins;
}

template<class DataType>
std::vector<DataType> generate_log_bins(int num_bins, DataType min_value = -1, DataType max_value = 1) {
    auto bins = generate_bins(num_bins, std::exp(min_value), std::exp(max_value));
    // print_hist(bins);
    std::transform(bins.cbegin(), bins.cend(), bins.begin(), [](float v) { return std::log(v); } );

    return bins;
}

template<class DataType>
std::vector<DataType> generate_nu_bins(int num_bins, DataType min_value = 0, DataType max_value = 1) {
    std::vector<DataType> bins(num_bins + 1);
    DataType last_bin = min_value;
    for (int i = 0; i < num_bins; i++)
    {
        bins[i] = last_bin;
        DataType avgBin = (max_value - last_bin)/(num_bins - i);
        last_bin = last_bin + rnd(0.01f*avgBin, avgBin);
    }
    bins[num_bins - 1] = last_bin;
    bins[num_bins] = max_value;
    // for (int i = 0; i <= num_bins; i++) {
    //     bins[i] = min_value + i*(max_value - min_value) / static_cast<DataType>(num_bins);
    // }
    return bins;
}

template<class DataType>
auto custom_upper_bound(const DataType* first, const DataType* last, const DataType& value)
{
    const DataType* it;
    auto count = last - first;
    int step  = 0;

    bool stop = false;
    while ( (count > 0) && !stop)
    {
        it = first;
        step = count/2;
        it += step;

        if (!(value < *it))
        {
            first = ++it;
            count -= step + 1;
        }
        else
            count = step;

        // stop = (*first <= value) && (value < *(first + 1));
    }

    return first + stop;
}

//Function to calculate histogram using tbb parallel_for
struct HistogramCustomUBRef
{
    static constexpr char const * const name = "histogram custom reference implementation";

    template<class DataType>
    static std::vector<int> hist(const std::vector<DataType>& data, const std::vector<DataType>& bins) {
        std::vector<int> hist(bins.size() - 1);
        auto begin = bins.data();
        auto end = begin + bins.size();
        for (auto&& d : data)
        {
            if (d >= bins.front() or d < bins.back())
                hist[custom_upper_bound(begin, end, d) - begin - 1]++;
        }
        return hist;
    }
};

struct HistogramRef
{
    static constexpr char const * const name = "histogram reference implementation";

    template<class DataType>
    static std::vector<int> hist(const std::vector<DataType>& data, const std::vector<DataType>& bins) {
        std::vector<int> hist(bins.size() - 1);
        for (auto&& d : data)
        {
            if (d >= bins.front() or d < bins.back())
                hist[std::upper_bound(bins.begin(), bins.end(), d) - bins.begin() - 1]++;
        }
        return hist;
    }
};


struct ParallelSort
{
    static constexpr char const * const name = "just parallel sort for performance comparission";

    template<class DataType>
    static std::vector<int> hist(const std::vector<DataType>& data, const std::vector<DataType>& bins) {
        std::vector<int> hist(bins.size() - 1);

        auto copy = data;

        tbb::parallel_sort(copy.begin(), copy.end());

        return hist;
    }
};

struct HistogramTbbIncorrect
{
    static constexpr char const * const name = "histogram tbb incorrect implementation";

    template<class DataType>
    static std::vector<int> hist(const std::vector<DataType>& data, const std::vector<DataType>& bins) {
        std::vector<int> hist(bins.size() - 1);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, data.size()), [&](tbb::blocked_range<size_t> r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                if (data[i] >= bins.front() and data[i] < bins.back())
                    hist[std::upper_bound(bins.begin(), bins.end(), data[i]) - bins.begin() - 1]++;
            }
        });
        return hist;
    }
};

//Function to calculate histogram using tbb parallel_for
struct HistogramTbbCombinableBase
{
    static constexpr char const * const name = "histogram tbb combinable base implementation";

    template<class DataType>
    static std::vector<int> hist(const std::vector<DataType>& data, const std::vector<DataType>& bins) {
        auto hist_size = bins.size() - 1;
        std::vector<int> hist(hist_size);
        using LocalHist = tbb::combinable<std::vector<DataType>>;
        LocalHist local_hist = LocalHist([hist_size] ()
        {
            return std::vector<DataType>(hist_size);
        });

        tbb::parallel_for(tbb::blocked_range<size_t>(0, data.size()), [&](tbb::blocked_range<size_t> r) {
            auto& _hist = local_hist.local();
            for (size_t i = r.begin(); i != r.end(); ++i) {
                if (data[i] >= bins.front() and data[i] < bins.back())
                    _hist[std::upper_bound(bins.begin(), bins.end(), data[i]) - bins.begin() - 1]++;
            }
        });

        local_hist.combine_each([&hist](const std::vector<DataType>& lh)
        {
            for (int i = 0; i < hist.size(); ++i)
                hist[i] += lh[i];
        });

        return hist;
    }
};

template<class DataType>
struct UniformMesh
{
    DataType u_diff;
    int n_bins;
    DataType lower_bound;
    std::vector<int> offsets;
    std::vector<int> counts;
    std::vector<std::unique_ptr<UniformMesh>> next_mesh;

    int depth = 0;
    int total_size = 0;
    int uniform_mesh_count = 0;

    UniformMesh(const std::vector<DataType>& bins)
    {
        n_bins = bins.size() - 1;
        offsets.resize(n_bins);
        counts.resize(n_bins);
        next_mesh.resize(n_bins);

        u_diff = bins.back() - bins.front();
        lower_bound = bins[0];
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

            if (nun_bins_count > 1)
            {
                int new_bins_start = offsets[i];
                int new_bins_end = new_bins_start + nun_bins_count + 1;
                new_bins_end += (bins[new_bins_end] < curr_up_bound);
                std::vector<DataType> new_bins(&bins[new_bins_start], &bins[new_bins_end]);
                new_bins.front() = curr_lo_bound + lower_bound;
                new_bins.back() =  curr_up_bound + lower_bound;

                next_mesh[i].reset(new UniformMesh(new_bins));

                depth = std::max(1 + next_mesh[i]->depth, depth);
                total_size += next_mesh[i]->total_size;
                uniform_mesh_count += (1 + next_mesh[i]->uniform_mesh_count);
            }
        }
    }

    int _get_non_uniform_bin(const DataType value, int start_offset)
    {
        // std::cout << "_get_non_uniform_bin: " << value << " " << start_offset << " " << lower_bound << " " << n_bins << " " << u_diff << " " << ((value - lower_bound)*n_bins/u_diff);
        int u_bin = std::floor((value - lower_bound)*n_bins/u_diff);
        u_bin = std::min(u_bin, n_bins - 1);
        int count = counts[u_bin];
        int offset = start_offset + offsets[u_bin];

        // std::cout << " " << u_bin << " " << count << " " <<  offsets[u_bin] << " " << offset << std::endl;
        if (count > 1)
        {
            offset = next_mesh[u_bin]->_get_non_uniform_bin(value, offset);
        }

        return offset;
    }

    int get_non_uniform_bin(const DataType value, const std::vector<DataType>& bins)
    {
        auto offset = _get_non_uniform_bin(value, 0);
        // std::cout << value << " " << offset << " " << bins[offset] << " " << bins[offset + 1] << std::endl;
        offset = offset - (value < bins[offset]) + (value >= bins[offset + 1]);
        return offset + (value >= bins[offset + 1]);
    }
};

//Function to calculate histogram using tbb parallel_for
struct HistogramUniformMesh
{
    static constexpr char const * const name = "histogram Uniform Mesh implementation";

    template<class DataType>
    // static std::vector<int> hist(const std::vector<DataType>& data, const std::vector<DataType>& bins, std::vector<std::vector<DataType>>& db) {
    static std::vector<int> hist(const std::vector<DataType>& data, const std::vector<DataType>& bins) {
        std::vector<int> hist(bins.size() - 1);
        auto um = UniformMesh<DataType>(bins);

        // db.resize(bins.size() - 1);
        for (auto&& d : data)
        {
            if (d >= bins.front() and d < bins.back())
            {
                int b = um.get_non_uniform_bin(d, bins);
                hist[b]++;
                // db[b].push_back(d);
            }
        }
        return hist;
    }
};

//Function to calculate histogram using tbb parallel_for
struct HistogramTbbUniformMesh
{
    static constexpr char const * const name = "histogram tbb Uniform Mesh implementation";

    template<class DataType>
    static std::vector<int> hist(const std::vector<DataType>& data, const std::vector<DataType>& bins) {
        auto hist_size = bins.size() - 1;
        std::vector<int> hist(hist_size);
        using LocalHist = tbb::combinable<std::vector<DataType>>;
        LocalHist local_hist = LocalHist([hist_size] ()
        {
            return std::vector<DataType>(hist_size);
        });

        auto um = UniformMesh<DataType>(bins);

        tbb::parallel_for(tbb::blocked_range<size_t>(0, data.size()), [&](tbb::blocked_range<size_t> r) {
            auto& _hist = local_hist.local();
            for (size_t i = r.begin(); i != r.end(); ++i) {
                auto d = data[i];
                if (d >= bins.front() and d < bins.back())
                    _hist[um.get_non_uniform_bin(d, bins)]++;
            }
        });

        local_hist.combine_each([&hist](const std::vector<DataType>& lh)
        {
            for (int i = 0; i < hist.size(); ++i)
                hist[i] += lh[i];
        });

        return hist;
    }
};

template<int Size, class DataType>
struct UniformMeshLinearSearch
{
    DataType u_diff;
    int n_bins;
    DataType lower_bound;
    std::vector<int> offsets;
    std::vector<int> counts;
    std::vector<std::unique_ptr<UniformMeshLinearSearch<Size, DataType>>> next_mesh;

    int depth = 0;
    int total_size = 0;
    int uniform_mesh_count = 0;

    UniformMeshLinearSearch(const std::vector<DataType>& bins)
    {
        // std::cout << "UniformMesh" << std::endl;

        n_bins = bins.size() - 1;
        offsets.resize(n_bins);
        counts.resize(n_bins);
        next_mesh.resize(n_bins);

        u_diff = bins.back() - bins.front();
        lower_bound = bins[0];

        // std::vector<DataType> u_bins(n_bins);

        int curr_nun_bin = 0;
        for (int i = 0; i < n_bins; ++i)
        {
            DataType curr_lo_bound = i*u_diff/n_bins;
            DataType curr_up_bound = (i+1)*u_diff/n_bins;
            // u_bins[i] = lower_bound + curr_up_bound;
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

                next_mesh[i].reset(new UniformMeshLinearSearch<Size, DataType>(new_bins));

                depth = std::max(1 + next_mesh[i]->depth, depth);
                total_size += next_mesh[i]->total_size;
                uniform_mesh_count += (1 + next_mesh[i]->uniform_mesh_count);
            }
        }
    }

    int _get_non_uniform_bin(const DataType value, int start_offset)
    {
        // std::cout << "_get_non_uniform_bin: " << value << " " << start_offset << " " << lower_bound << " " << n_bins << " " << u_diff << " " << ((value - lower_bound)*n_bins/u_diff);
        int u_bin = std::floor((value - lower_bound)*n_bins/u_diff);
        u_bin = std::min(u_bin, n_bins - 1);
        int count = counts[u_bin];
        int offset = start_offset + offsets[u_bin];

        // std::cout << " " << u_bin << " " << count << " " <<  offsets[u_bin] << " " << offset << std::endl;
        if (count > (Size - 2))
        {
            offset = next_mesh[u_bin]->_get_non_uniform_bin(value, offset);
        }

        return offset;
    }

    int get_non_uniform_bin(const DataType value, const std::vector<DataType>& aligned_bins)
    {
        auto offset = _get_non_uniform_bin(value, 0);
        // std::cout << value << " " << offset << " " << bins[offset] << " " << bins[offset + 1] << std::endl;

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

//Function to calculate histogram using tbb parallel_for
template<int Size>
struct HistogramUniformMeshLinearSearch
{
    static constexpr char const * const name = "histogram Uniform Mesh with linear search implementation";

    template<class DataType>
    // static std::vector<int> hist(const std::vector<DataType>& data, const std::vector<DataType>& bins, std::vector<std::vector<DataType>>& db) {
    static std::vector<int> hist(const std::vector<DataType>& data, const std::vector<DataType>& bins) {
        std::vector<int> hist(bins.size() - 1);
        auto um = UniformMeshLinearSearch<Size, DataType>(bins);
        auto _bins = um.align_bins(bins);

        // db.resize(bins.size() - 1);
        for (auto&& d : data)
        {
            if (d >= _bins.front() and d < _bins.back())
            {
                int b = um.get_non_uniform_bin(d, _bins);
                hist[b]++;
                // db[b].push_back(d);
            }
        }
        return hist;
    }
};

//Function to calculate histogram using tbb parallel_for
template<int Size>
struct HistogramTbbUniformMeshLinearSearch
{
    static constexpr char const * const name = "histogram tbb Uniform Mesh with linear search implementation";

    template<class DataType>
    static std::vector<int> hist(const std::vector<DataType>& data, const std::vector<DataType>& bins) {
        auto hist_size = bins.size() - 1;
        std::vector<int> hist(hist_size);
        using LocalHist = tbb::combinable<std::vector<DataType>>;
        LocalHist local_hist = LocalHist([hist_size] ()
        {
            return std::vector<DataType>(hist_size);
        });

        auto um = UniformMeshLinearSearch<Size, DataType>(bins);
        auto _bins = um.align_bins(bins);

        tbb::parallel_for(tbb::blocked_range<size_t>(0, data.size()), [&](tbb::blocked_range<size_t> r) {
            auto& _hist = local_hist.local();
            for (size_t i = r.begin(); i != r.end(); ++i) {
                auto d = data[i];
                if (d >= _bins.front() and d < _bins.back())
                    _hist[um.get_non_uniform_bin(d, _bins)]++;
            }
        });

        local_hist.combine_each([&hist](const std::vector<DataType>& lh)
        {
            for (int i = 0; i < hist.size(); ++i)
                hist[i] += lh[i];
        });

        return hist;
    }
};

template<int Size, class DataType>
std::vector<DataType> reorder_bins(const std::vector<DataType>& bins, std::vector<int>& offsets, int& depth)
{
    auto nbins = bins.size() - 1;
    // depth = std::ceil(std::log(nbins)/std::log(Size + 1));
    // std::cout << depth << std::endl;
    depth = 4;

    std::vector<int> level_sizes(depth);
    int curr_level_bins = nbins;
    for (int i = 0; i < depth; ++i)
    {
        // std::cout << " " << curr_level_bins << std::endl;
        curr_level_bins = DivUp(curr_level_bins, Size + 1);
        level_sizes[i] = Size*curr_level_bins;
    }

    // print_hist(level_sizes);
    int total_size = std::accumulate(level_sizes.begin(), level_sizes.end(), 0);
    offsets.resize(depth);
    for (int i = 0; i < depth; ++i)
        offsets[i] = std::accumulate(level_sizes.end() - i, level_sizes.end(), 0);

    // print_hist(offsets);

    auto copy_bins = [](DataType* curr_level, DataType* next_level, DataType* source, int size)
    {
        int next_level_count = 0;
        int curr_level_count = 0;
        for (int i = 0; i < size; ++i)
        {
            if ((i + 1)%(Size + 1) > 0)
            {
                curr_level[curr_level_count++] = source[i];
            }
            else
            {
                next_level[next_level_count++] = source[i];
            }
        }

        return next_level_count;
    };

    std::vector<DataType> new_bins(total_size, bins.back());
    std::vector<DataType> tmp[2];

    tmp[0].resize(bins.size() - 2);
    std::copy_n(bins.begin() + 1, tmp[0].size(), tmp[0].begin());
    tmp[1].resize(bins.size() - 2, bins.back());
    int curr_level_size = tmp[0].size();

    for (int i = 0; i < depth; ++i)
    {
        curr_level_size = copy_bins(&new_bins[offsets[depth - i - 1]], tmp[1 - i&1].data(), tmp[i&1].data(), curr_level_size);
    }

    // print_hist(bins);
    // print_hist(new_bins);

    return new_bins;
}

template<int Size, class DataType>
int get_reordered_bin(DataType value, const std::vector<DataType>& reordered_bins, const std::vector<int>& offsets, int depth)
{
    auto get_bin = [](const DataType* bins, DataType value)
    {
        int offset = 0;
        for (int i = 0; i < Size; ++i)
        {
            offset += (value >= bins[i]);
        }

        return offset;
    };

    int prev_bins = 0;
    int result_bin = 0;
    for (int i = 0; i < depth; ++i)
    {
        result_bin = get_bin(&reordered_bins[offsets[i] + result_bin*Size], value) + (Size + 1)*result_bin;
    }

    // std::cout << "val " << value << " bin " << result_bin << std::endl;

    return result_bin;
}

template<int Size>
struct HistogramNNarySearchWithReorder
{
    static constexpr char const * const name = "histogram NNary search with reorder implementation";

    template<class DataType>
    // static std::vector<int> hist(const std::vector<DataType>& data, const std::vector<DataType>& bins, std::vector<std::vector<DataType>>& db) {
    static std::vector<int> hist(const std::vector<DataType>& data, const std::vector<DataType>& bins) {
        std::vector<int> hist(bins.size() - 1);
        std::vector<int> offsets;
        int depth = 0;

        auto new_bins = reorder_bins<Size>(bins, offsets, depth);
        auto minVal = bins.front();
        auto maxVal = bins.back();

        // db.resize(bins.size() - 1);
        for (auto&& d : data)
        {
            if (d >= minVal and d < maxVal)
            {
                int b = get_reordered_bin<Size>(d, new_bins, offsets, depth);
                hist[b]++;
                // db[b].push_back(d);
            }
        }
        return hist;
    }
};

template<int Size>
struct HistogramTBBNNarySearchWithReorder
{
    static constexpr char const * const name = "histogram NNary search with reorder tbb implementation";

    template<class DataType>
    // static std::vector<int> hist(const std::vector<DataType>& data, const std::vector<DataType>& bins, std::vector<std::vector<DataType>>& db) {
    static std::vector<int> hist(const std::vector<DataType>& data, const std::vector<DataType>& bins) {
        auto hist_size = bins.size() - 1;
        std::vector<int> hist(hist_size);
        using LocalHist = tbb::combinable<std::vector<DataType>>;
        LocalHist local_hist = LocalHist([hist_size] ()
        {
            return std::vector<DataType>(hist_size);
        });

        std::vector<int> offsets;
        int depth = 0;

        auto new_bins = reorder_bins<Size>(bins, offsets, depth);
        auto minVal = bins.front();
        auto maxVal = bins.back();

        tbb::parallel_for(tbb::blocked_range<size_t>(0, data.size()), [&](tbb::blocked_range<size_t> r) {
            auto& _hist = local_hist.local();
            for (size_t i = r.begin(); i != r.end(); ++i) {
                auto d = data[i];
                if (d >= minVal and d < maxVal)
                    _hist[get_reordered_bin<Size>(d, new_bins, offsets, depth)]++;
            }
        });

        local_hist.combine_each([&hist](const std::vector<DataType>& lh)
        {
            for (int i = 0; i < hist.size(); ++i)
                hist[i] += lh[i];
        });

        return hist;
    }
};

bool validate(const std::vector<int>& data, const std::vector<int>& ref)
{
    return data == ref;
}

template<class HistogramImplementation, class DataType>
bool measure_and_validate(const std::vector<DataType>& data, const std::vector<DataType>& bins, const std::vector<int>& ref)
{
    // std::vector<std::vector<DataType>> db;
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    // std::vector<int> hist = HistogramImplementation::hist(data, bins, db);
    std::vector<int> hist = HistogramImplementation::hist(data, bins);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    std::string validation_result = "[validation passed]";
    if (not validate(hist, ref))
        validation_result = "[validation failed]";

    std::chrono::duration<double, std::milli> time_span = t2 - t1;
    std::cout << HistogramImplementation::name << ": " << time_span.count() << " milliseconds. " << validation_result << std::endl;
    // print_hist(ref);
    // print_hist(hist);
    // for (int i = 0; i < db.size(); ++i)
    // {
    //     std::cout << std::fixed << std::setprecision(17) << "[" << bins[i] << "," << bins[i+1] << ") ";
    //     print_hist(db[i]);
    // }

    return validate(hist, ref);
}

template<class DataType>
struct SyclIncorrectUniformHist
{
    DataType step = 0;
    DataType minVal = 0;
    DataType maxVal = 0;
    std::string name = "sycl incorrect";

    SyclIncorrectUniformHist() {}

    void init(sycl::queue queue, DataType* bins, int bins_size)
    {
        std::vector<DataType> host_bins(bins_size);
        queue.copy(bins, host_bins.data(), bins_size).wait();

        minVal = host_bins.front();
        maxVal = host_bins.back();
        step = (maxVal - minVal)/bins_size;
    }

    void run(sycl::queue queue, DataType* bins, int bins_size, DataType* data, int data_size, int* hist)
    {
        auto _minVal = minVal;
        auto _maxVal = maxVal;
        auto _step = step;
        queue.parallel_for(sycl::range<1>(data_size),[=](sycl::id<1> idx)
        {
            int64_t id = idx.get(0);
            auto d = data[id];
            int bin = (d - _minVal)/_step;

            bin = std::max(std::min(bin, bins_size), 0);
            if (d >= _minVal or d < _maxVal)
            {
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> h(hist[bin]);
                ++h;
            }
            // hist[id] = bin;
        }).wait();
    }
};

template<class DataType>
struct SyclBasicHistImpl
{
    std::string name = "sycl basic implementation";
    SyclBasicHistImpl() {}

    void init(sycl::queue queue, DataType* bins, int bins_size) {}

    void run(sycl::queue queue, DataType* bins, int bins_size, DataType* data, int data_size, int* hist)
    {
        queue.parallel_for(sycl::range<1>(data_size),[=](sycl::id<1> idx)
        {
            int64_t id = idx.get(0);
            auto minVal = bins[0];
            auto maxVal = bins[bins_size - 1];

            auto d = data[id];
            int bin = 0;
            if (d >= minVal or d < maxVal)
            {
                bin = std::upper_bound(bins, bins + bins_size, d) - bins - 1;

                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> h(hist[bin]);
                ++h;
            }
            // hist[id] = bin;
        }).wait();
    }
};

template<class DataType>
struct SyclBasicCustomHistImpl
{
    std::string name = "sycl basic custom implementation";
    SyclBasicCustomHistImpl() {}

    void init(sycl::queue queue, DataType* bins, int bins_size) {}

    void run(sycl::queue queue, DataType* bins, int bins_size, DataType* data, int data_size, int* hist)
    {
        queue.parallel_for(sycl::range<1>(data_size),[=](sycl::id<1> idx)
        {
            int64_t id = idx.get(0);
            auto minVal = bins[0];
            auto maxVal = bins[bins_size - 1];

            auto d = data[id];
            int bin = 0;
            if (d >= minVal or d < maxVal)
            {
                bin = custom_upper_bound(bins, bins + bins_size, d) - bins - 1;

                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> h(hist[bin]);
                ++h;
            }
            // hist[id] = bin;
        }).wait();
    }
};

template<class DataType, class SBGR>
auto custom_sycl_upper_bound(SBGR& sbgr, const DataType* first, const DataType* last, const DataType& value)
{
    const DataType* it;
    auto count = last - first;
    int step  = 0;

    int stop = 0;
    while ( (count > 0) && !stop)
    {
        it = first;
        step = count/2;
        it += step;

        if (!(value < *it))
        {
            first = ++it;
            count -= step + 1;
        }
        else
            count = step;

        stop = (*first <= value) && (value < *(first + 1));
        stop = sycl::reduce_over_group(sbgr, stop, sycl::minimum<>());
    }

    return first + stop;
}

template<class DataType>
struct SyclBasicCustomWithStopHistImpl
{
    std::string name = "sycl basic custom with stop implementation";
    SyclBasicCustomWithStopHistImpl() {}

    void init(sycl::queue queue, DataType* bins, int bins_size) {}

    void run(sycl::queue queue, DataType* bins, int bins_size, DataType* data, int data_size, int* hist)
    {
        queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(data_size), sycl::range<1>(256)), [=](sycl::nd_item<1> item)
        {
            int64_t id = item.get_global_id(0);
            auto sbgr = item.get_sub_group();

            auto minVal = bins[0];
            auto maxVal = bins[bins_size - 1];

            auto d = data[id];
            int bin = 0;
            if (d >= minVal or d < maxVal)
            {
                bin = custom_sycl_upper_bound(sbgr, bins, bins + bins_size, d) - bins - 1;
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> h(hist[bin]);
                ++h;
            }
            // hist[id] = bin;
        }).wait();
    }
};

template<int WorkPI, class DataType>
struct SyclBasicHistLocalHistImpl
{
    std::string name = "sycl local hist basic implementation";
    SyclBasicHistLocalHistImpl() {}

    void init(sycl::queue queue, DataType* bins, int bins_size) {}

    void run(sycl::queue queue, DataType* bins, int bins_size, DataType* data, int data_size, int* hist)
    {
        queue.submit([&](sycl::handler& cgh)
        {
            int local_size = 256;
            auto local_hist = localMem<int, 1>(sycl::range<1>(bins_size - 1), cgh);
            cgh.parallel_for(sycl::nd_range<1>(sycl::range(data_size/WorkPI), sycl::range(local_size)),[=](sycl::nd_item<1> item)
            {
                int64_t id = item.get_group_linear_id();
                int64_t lid = item.get_local_linear_id();

                auto minVal = bins[0];
                auto maxVal = bins[bins_size - 1];

                auto group = item.get_group();

                for (int i = lid; i < (bins_size - 1); i += local_size)
                {
                    local_hist[i] = 0;
                }
                sycl::group_barrier(group, sycl::memory_scope::work_group);

                for (int i = 0; i < WorkPI; ++i)
                {
                    auto d = data[id*WorkPI*local_size + i*local_size + lid];
                    if (d >= minVal or d < maxVal)
                    {
                        int bin = std::upper_bound(bins, bins + bins_size, d) - bins - 1;
                        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group> lh(local_hist[bin]);
                        lh.fetch_add(1);
                    }
                }

                sycl::group_barrier(group, sycl::memory_scope::work_group);

                for (int i = lid; i < (bins_size - 1); i += local_size)
                {
                    auto lh_value = local_hist[i];
                    if (lh_value > 0)
                    {
                        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> h(hist[i]);
                        h.fetch_add(lh_value);
                    }
                }
                // hist[id] = bin;
            });
        }).wait();
    }
};

template<int Size, class DataType>
static int get_reordered_bin(DataType value, DataType* reordered_bins, int* offsets, int depth)
{
    auto get_bin = [](const DataType* bins, DataType value)
    {
        int offset = 0;
        for (int i = 0; i < Size; ++i)
        {
            offset += (value >= bins[i]);
        }

        return offset;
    };

    int prev_bins = 0;
    int result_bin = 0;
    for (int i = 0; i < depth; ++i)
    {
        result_bin = get_bin(&reordered_bins[offsets[i] + result_bin*Size], value) + (Size + 1)*result_bin;
    }

    return result_bin;
}

template<int Size, int WorkPI, class DataType>
struct SyclNNaryHistImpl
{
    std::string name = "sycl nnary initial implementation";
    DeviceMem<DataType> reordered_bins = nullptr;
    DeviceMem<int> offsets = nullptr;
    int depth = 0;
    DataType minValue = 0;
    DataType maxValue = 0;

    SyclNNaryHistImpl() {}

    void init(sycl::queue queue, DataType* bins, int bins_size)
    {
        std::vector<DataType> host_bins(bins_size);
        std::vector<int> host_offsets;

        queue.copy(bins, host_bins.data(), bins_size).wait();
        auto rbins = reorder_bins<Size>(host_bins, host_offsets, depth);

        reordered_bins = std::move(malloc_device<DataType>(int64_t(rbins.size()), queue));
        offsets = std::move(malloc_device<int>(depth, queue));

        queue.copy(rbins.data(), reordered_bins.get(), rbins.size()).wait();
        queue.copy(host_offsets.data(), offsets.get(), depth).wait();

        minValue = host_bins.front();
        maxValue = host_bins.back();
    }

    void run(sycl::queue queue, DataType* bins, int bins_size, DataType* data, int data_size, int* hist)
    {
        DataType* _rbins = reordered_bins.get();
        int* _offsets = offsets.get();
        int _depth = depth;
        DataType _minValue = minValue;
        DataType _maxValue = maxValue;

        queue.submit([&](sycl::handler& cgh)
        {
            int local_size = 256;
            auto local_hist = localMem<int, 1>(sycl::range<1>(bins_size - 1), cgh);
            cgh.parallel_for(sycl::nd_range<1>(sycl::range(data_size/WorkPI), sycl::range(local_size)),[=](sycl::nd_item<1> item)
            {
                int64_t id = item.get_group_linear_id();
                int64_t lid = item.get_local_linear_id();

                auto minVal = bins[0];
                auto maxVal = bins[bins_size - 1];

                auto group = item.get_group();

                for (int i = lid; i < (bins_size - 1); i += local_size)
                {
                    local_hist[i] = 0;
                }
                sycl::group_barrier(group, sycl::memory_scope::work_group);

                for (int i = 0; i < WorkPI; ++i)
                {
                    auto d = data[id*WorkPI*local_size + i*local_size + lid];
                    if (d >= minVal or d < maxVal)
                    {
                        int bin = get_reordered_bin<Size, DataType>(d, _rbins, _offsets, _depth);
                        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group> lh(local_hist[bin]);
                        lh.fetch_add(1);
                    }
                }

                sycl::group_barrier(group, sycl::memory_scope::work_group);

                for (int i = lid; i < (bins_size - 1); i += local_size)
                {
                    auto lh_value = local_hist[i];
                    if (lh_value > 0)
                    {
                        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> h(hist[i]);
                        h.fetch_add(lh_value);
                    }
                }
                // hist[id] = bin;
            });
        }).wait();
        // queue.parallel_for(sycl::range<1>(data_size),[=](sycl::id<1> idx)
        // {
        //     int64_t id = idx.get(0);

        //     auto d = data[id];
        //     int bin = 0;
        //     if (d >= _minValue or d < _maxValue)
        //     {
        //         bin = get_reordered_bin<Size, DataType>(d, _rbins, _offsets, _depth);

        //         sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> h(hist[bin]);
        //         ++h;
        //     }
        // }).wait();
    }
};

// template<int Size, class DataType>
// struct SyclNNaryHistImpl
// {
//     std::string name = "sycl nnary initial implementation";
//     DeviceMem<DataType> reordered_bins = nullptr;
//     DeviceMem<int> offsets = nullptr;
//     int depth = 0;
//     DataType minValue = 0;
//     DataType maxValue = 0;

//     SyclNNaryHistImpl() {}

//     void init(sycl::queue queue, DataType* bins, int bins_size)
//     {
//         std::vector<DataType> host_bins(bins_size);
//         std::vector<int> host_offsets;

//         queue.copy(bins, host_bins.data(), bins_size).wait();
//         auto rbins = reorder_bins<Size>(host_bins, host_offsets, depth);

//         reordered_bins = std::move(malloc_device<DataType>(int64_t(rbins.size()), queue));
//         offsets = std::move(malloc_device<int>(depth, queue));

//         queue.copy(rbins.data(), reordered_bins.get(), rbins.size()).wait();
//         queue.copy(host_offsets.data(), offsets.get(), depth).wait();

//         minValue = host_bins.front();
//         maxValue = host_bins.back();
//     }

//     void run(sycl::queue queue, DataType* bins, int bins_size, DataType* data, int data_size, int* hist)
//     {
//         DataType* _rbins = reordered_bins.get();
//         int* _offsets = offsets.get();
//         int _depth = depth;
//         DataType _minValue = minValue;
//         DataType _maxValue = maxValue;

//         queue.parallel_for(sycl::range<1>(data_size),[=](sycl::id<1> idx)
//         {
//             int64_t id = idx.get(0);

//             auto d = data[id];
//             int bin = 0;
//             if (d >= _minValue or d < _maxValue)
//             {
//                 bin = get_reordered_bin<Size, DataType>(d, _rbins, _offsets, _depth);

//                 sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> h(hist[bin]);
//                 ++h;
//             }
//         }).wait();
//     }
// };


template<int Size, class DataType>
struct SyclNNaryHistVectImpl
{
    std::string name = "sycl nnary vectorized implementation";
    DeviceMem<DataType> reordered_bins = nullptr;
    DeviceMem<int> offsets = nullptr;
    int depth = 0;
    DataType minValue = 0;
    DataType maxValue = 0;

    SyclNNaryHistVectImpl()
    {
        name = std::string("sycl nnary vectorized(") + std::to_string(Size) + ") implementation";
    }

    void init(sycl::queue queue, DataType* bins, int bins_size)
    {
        std::vector<DataType> host_bins(bins_size);
        std::vector<int> host_offsets;

        queue.copy(bins, host_bins.data(), bins_size).wait();
        auto rbins = reorder_bins<Size>(host_bins, host_offsets, depth);

        reordered_bins = std::move(malloc_device<DataType>(int64_t(rbins.size()), queue));
        offsets = std::move(malloc_device<int>(Multiple(depth, Size), queue));

        queue.copy(rbins.data(), reordered_bins.get(), rbins.size()).wait();
        queue.copy(host_offsets.data(), offsets.get(), depth).wait();

        minValue = host_bins.front();
        maxValue = host_bins.back();
    }

    void run(sycl::queue queue, DataType* bins, int bins_size, DataType* data, int data_size, int* hist)
    {
        DataType* _rbins = reordered_bins.get();
        int* _offsets = offsets.get();
        int _depth = depth;
        DataType _minValue = minValue;
        DataType _maxValue = maxValue;


        queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(data_size), sycl::range<1>(256)), [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(Size)]]
        {
            int64_t id = item.get_global_id(0);
            auto sbgr = item.get_sub_group();
            int64_t local_id_1 = sbgr.get_local_id();

            auto level0_bound = _rbins[local_id_1];
            int _local_offset = _offsets[local_id_1];

            auto d = data[id];

            for (int j = 0; j < Size; ++j)
            {
                auto _d = sycl::select_from_group(sbgr, d, j);
                if (_d >= _minValue or _d < _maxValue)
                {
                    int result_bin = (_d >= level0_bound);
                    result_bin = sycl::reduce_over_group(sbgr, result_bin, sycl::plus<>());

                    for (int i = 1; i < _depth; ++i)
                    {
                        auto _level_offset = sycl::select_from_group(sbgr, _local_offset, i);
                        auto bin_id = _level_offset + result_bin*Size;
                        auto level_bound = _rbins[bin_id + local_id_1];
                        int local_bin = (_d >= level_bound);
                        local_bin = sycl::reduce_over_group(sbgr, local_bin, sycl::plus<>());
                        result_bin = result_bin*(Size + 1) + local_bin;
                    }

                    if (sbgr.leader())
                    {
                        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> h(hist[result_bin]);
                        ++h;
                    }
                }
            }
        }).wait();
    }
};

constexpr sycl::specialization_id<int> const_depth_id;

template<int Size, class DataType>
struct SyclNNaryHistVectConstImpl
{
    std::string name = "sycl nnary vectorized with const implementation";
    DeviceMem<DataType> reordered_bins = nullptr;
    DeviceMem<int> offsets = nullptr;
    int depth = 0;
    DataType minValue = 0;
    DataType maxValue = 0;

    SyclNNaryHistVectConstImpl()
    {
        name = std::string("sycl nnary vectorized(") + std::to_string(Size) + ") with const implementation";
    }

    void init(sycl::queue queue, DataType* bins, int bins_size)
    {
        std::vector<DataType> host_bins(bins_size);
        std::vector<int> host_offsets;

        queue.copy(bins, host_bins.data(), bins_size).wait();
        auto rbins = reorder_bins<Size>(host_bins, host_offsets, depth);

        reordered_bins = std::move(malloc_device<DataType>(int64_t(rbins.size()), queue));
        offsets = std::move(malloc_device<int>(Multiple(depth, Size), queue));

        queue.copy(rbins.data(), reordered_bins.get(), rbins.size()).wait();
        queue.copy(host_offsets.data(), offsets.get(), depth).wait();

        minValue = host_bins.front();
        maxValue = host_bins.back();
    }

    void run(sycl::queue queue, DataType* bins, int bins_size, DataType* data, int data_size, int* hist)
    {
        DataType* _rbins = reordered_bins.get();
        int* _offsets = offsets.get();
        int _depth = depth;
        DataType _minValue = minValue;
        DataType _maxValue = maxValue;

        queue.submit([&](sycl::handler& cgh)
        {
            cgh.set_specialization_constant<const_depth_id>(depth);

            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(data_size), sycl::range<1>(256)), [=](sycl::nd_item<1> item, sycl::kernel_handler h) [[intel::reqd_sub_group_size(Size)]]
            {
                int64_t id = item.get_global_id(0);
                auto sbgr = item.get_sub_group();
                int64_t local_id_1 = sbgr.get_local_id();

                auto level0_bound = _rbins[local_id_1];
                int _local_offset = _offsets[local_id_1];

                auto d = data[id];
                int Depth = h.get_specialization_constant<const_depth_id>();

                for (int j = 0; j < Size; ++j)
                {
                    auto _d = sycl::select_from_group(sbgr, d, j);
                    if (_d >= _minValue or _d < _maxValue)
                    {
                        int result_bin = (_d >= level0_bound);
                        result_bin = sycl::reduce_over_group(sbgr, result_bin, sycl::plus<>());

                        for (int i = 1; i < Depth; ++i)
                        {
                            auto _level_offset = sycl::select_from_group(sbgr, _local_offset, i);
                            auto bin_id = _level_offset + result_bin*Size;
                            auto level_bound = _rbins[bin_id + local_id_1];
                            // auto level_bound = sbgr.load(_rbins + bin_id);
                            int local_bin = (_d >= level_bound);
                            local_bin = sycl::reduce_over_group(sbgr, local_bin, sycl::plus<>());
                            result_bin = result_bin*(Size + 1) + local_bin;
                        }

                        if (sbgr.leader())
                        {
                            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> h(hist[result_bin]);
                            ++h;
                        }
                    }
                }
            });
        }).wait();
    }
};

template<int Size, class DataType>
struct SyclUniformMeshLinearSearch
{
    std::vector<DataType> u_diff;
    std::vector<int> n_bins;
    std::vector<DataType> lower_bound;
    std::vector<int> offsets_start;

    std::vector<int> offsets;
    std::vector<int> next_mesh;

    DeviceMem<DataType> device_u_diff;
    DeviceMem<int> device_n_bins;
    DeviceMem<DataType> device_lower_bound;
    DeviceMem<int> device_offsets_start;

    DeviceMem<int> device_offsets;
    DeviceMem<int> device_next_mesh;

    void to_device(sycl::queue queue)
    {
        int um_count = u_diff.size();
        int total_size = offsets.size();

        // std::cout << "um_count " << um_count << std::endl;
        // std::cout << "total_size " << total_size << std::endl;

        device_u_diff = std::move(malloc_device<DataType>(um_count, queue));
        device_n_bins = std::move(malloc_device<int>(um_count, queue));
        device_lower_bound = std::move(malloc_device<DataType>(um_count, queue));
        device_offsets_start = std::move(malloc_device<int>(um_count, queue));

        device_offsets = std::move(malloc_device<int>(total_size, queue));
        device_next_mesh = std::move(malloc_device<int>(total_size, queue));

        queue.copy(u_diff.data(), device_u_diff.get(), um_count).wait();
        queue.copy(n_bins.data(), device_n_bins.get(), um_count).wait();
        queue.copy(lower_bound.data(), device_lower_bound.get(), um_count).wait();
        queue.copy(offsets_start.data(), device_offsets_start.get(), um_count).wait();

        queue.copy(offsets.data(), device_offsets.get(), total_size).wait();
        queue.copy(next_mesh.data(), device_next_mesh.get(), total_size).wait();
    }

    void walk_uniform_tree(UniformMeshLinearSearch<Size, DataType>& um)
    {
        std::deque<struct UniformMeshLinearSearch<Size, DataType>*> queue;
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

    SyclUniformMeshLinearSearch(const std::vector<DataType>& host_bins, int bins_size, sycl::queue queue)
    {
        auto um = UniformMeshLinearSearch<Size, DataType>(host_bins);
        auto um_count = um.uniform_mesh_count + 1;
        auto total_count = um.total_size;

        u_diff.reserve(um_count);
        n_bins.reserve(um_count);
        lower_bound.reserve(um_count);
        offsets_start.reserve(um_count);

        offsets.reserve(total_count);
        next_mesh.reserve(total_count);

        walk_uniform_tree(um);
        to_device(queue);
    }

    std::vector<DataType> align_bins(const std::vector<DataType>& bins)
    {
        std::vector<DataType> _bins(bins.size() + Size, bins.back());
        std::copy_n(bins.begin(), bins.size(), _bins.begin());

        return _bins;
    }
};

template<class DataType> using SyclUniformMesh = SyclUniformMeshLinearSearch<3, DataType>;

template<int Size, class DataType>
struct SyclHistogramUniformMeshLinearSearchImpl
{
    std::string name = "sycl uniform mesh implementation";
    std::unique_ptr<SyclUniformMeshLinearSearch<Size, DataType>> sum;

    DeviceMem<DataType> aligned_bins = nullptr;
    DataType minVal = 0;
    DataType maxVal = 0;

    SyclHistogramUniformMeshLinearSearchImpl() {}

    void init(sycl::queue queue, DataType* bins, int bins_size)
    {
        std::vector<DataType> host_bins(bins_size);
        queue.copy(bins, host_bins.data(), bins_size).wait();
        sum.reset(new SyclUniformMeshLinearSearch<Size, DataType>(host_bins, bins_size, queue));

        auto a_host_bins = sum->align_bins(host_bins);

        aligned_bins = std::move(malloc_device<DataType>(int(a_host_bins.size()), queue));
        queue.copy(a_host_bins.data(), aligned_bins.get(), a_host_bins.size()).wait();

        minVal = host_bins.front();
        maxVal = host_bins.back();
    }

    void run(sycl::queue queue, DataType* bins, int bins_size, DataType* data, int data_size, int* hist)
    {
        DataType* u_diff = sum->device_u_diff.get();
        int* n_bins = sum->device_n_bins.get();
        DataType* lower_bound = sum->device_lower_bound.get();
        int* device_offsets_start = sum->device_offsets_start.get();

        int* offsets = sum->device_offsets.get();
        int* next_mesh = sum->device_next_mesh.get();

        DataType* _bins = aligned_bins.get();

        auto _minVal = minVal;
        auto _maxVal = maxVal;

        // queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(data_size), sycl::range<1>(256)), [=](sycl::nd_item<1> item)
        queue.parallel_for(sycl::range<1>(data_size),[=](sycl::id<1> idx)
        {
            // int id = item.get(0);
            int64_t id = idx.get(0);

            auto d = data[id];

            auto u_diff_0 = u_diff[0];
            auto n_bins_0 = n_bins[0];
            auto lower_bound_0 = lower_bound[0];

            if (d >= _minVal and d < _maxVal)
            {
                int u_bin = std::floor((d - lower_bound_0)*n_bins_0/u_diff_0);
                u_bin = std::min(u_bin, n_bins_0 - 1);
                int offset = offsets[u_bin];
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
                for (int i = 0; i < Size; ++i)
                    _offset += (d >= _bins[offset + i]);

                offset = offset + _offset - 1;

                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> h(hist[offset]);
                ++h;
            }
        }).wait();
    }
};

template<class DataType> using SyclHistogramUniformMeshImpl = SyclHistogramUniformMeshLinearSearchImpl<3, DataType>;


template<int Size, int WorkPI, class DataType>
struct SyclHistogramUniformMeshLinearSearchLMImpl
{
    std::string name = "sycl uniform mesh implementation";
    std::unique_ptr<SyclUniformMeshLinearSearch<Size, DataType>> sum;

    DeviceMem<DataType> aligned_bins = nullptr;
    DataType minVal = 0;
    DataType maxVal = 0;

    DataType u_diff_0 = 0;
    int n_bins_0 = 0;
    DataType lower_bound_0 = 0;

    int total_size = 0;

    SyclHistogramUniformMeshLinearSearchLMImpl() {}

    void init(sycl::queue queue, DataType* bins, int bins_size)
    {
        std::vector<DataType> host_bins(bins_size);
        queue.copy(bins, host_bins.data(), bins_size).wait();
        sum.reset(new SyclUniformMeshLinearSearch<Size, DataType>(host_bins, bins_size, queue));

        auto a_host_bins = sum->align_bins(host_bins);

        aligned_bins = std::move(malloc_device<DataType>(int(a_host_bins.size()), queue));
        queue.copy(a_host_bins.data(), aligned_bins.get(), a_host_bins.size()).wait();

        minVal = host_bins.front();
        maxVal = host_bins.back();

        total_size = sum->offsets.size();

        u_diff_0 = sum->u_diff[0];
        n_bins_0 = sum->n_bins[0];
        lower_bound_0 = sum->lower_bound[0];
    }

    void run(sycl::queue queue, DataType* bins, int bins_size, DataType* data, int data_size, int* hist)
    {
        DataType* u_diff = sum->device_u_diff.get();
        int* n_bins = sum->device_n_bins.get();
        DataType* lower_bound = sum->device_lower_bound.get();
        int* device_offsets_start = sum->device_offsets_start.get();

        int* offsets = sum->device_offsets.get();
        int* next_mesh = sum->device_next_mesh.get();

        DataType* _bins = aligned_bins.get();

        auto _minVal = minVal;
        auto _maxVal = maxVal;

        auto u_diff_0_ = u_diff_0;
        auto n_bins_0_ = n_bins_0;
        auto lower_bound_0_ = lower_bound_0;

        auto local_size = 256;
        auto total_size_ = total_size;

        constexpr int work_per_item = 8;

        // queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(data_size), sycl::range<1>(256)), [=](sycl::nd_item<1> item)
        queue.submit([&](sycl::handler& cgh)
        {
            auto local_offset = localMem<int, 1>(sycl::range<1>(total_size), cgh);
            auto local_hist = localMem<int, 1>(sycl::range<1>(bins_size - 1), cgh);

            cgh.parallel_for(sycl::nd_range<1>(sycl::range(data_size/WorkPI), sycl::range(local_size)),[=](sycl::nd_item<1> item)
            {
                // int id = item.get(0);
                int64_t id = item.get_group_linear_id();
                int64_t lid = item.get_local_linear_id();
                auto group = item.get_group();

                for (int i = lid; i < total_size_; i += local_size)
                {
                    local_offset[i] = offsets[i];
                }

                for (int i = lid; i < (bins_size - 1); i += local_size)
                {
                    local_hist[i] = 0;
                }
                sycl::group_barrier(group, sycl::memory_scope::work_group);

                for (int wi = 0; wi < WorkPI; ++wi)
                {
                    auto d = data[id*WorkPI*local_size + wi*local_size + lid];

                    if (d >= _minVal and d < _maxVal)
                    {
                        int u_bin = std::floor((d - lower_bound_0_)*n_bins_0_/u_diff_0_);
                        u_bin = std::min(u_bin, n_bins_0_ - 1);
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
                        for (int i = 0; i < Size; ++i)
                            _offset += (d >= _bins[offset + i]);

                        offset = offset + _offset - 1;

                        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group> lh(local_hist[offset]);
                        ++lh;
                    }
                }

                sycl::group_barrier(group, sycl::memory_scope::work_group);

                for (int i = lid; i < (bins_size - 1); i += local_size)
                {
                    // sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::work_group> lh_value(local_hist[hist_id]);
                    auto lh_value = local_hist[i];
                    if (lh_value > 0)
                    {
                        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> h(hist[i]);
                        h += lh_value;
                    }
                }
            });
        }).wait();
    }
};


template<class Impl, class DataType>
bool measure_and_validate_sycl(sycl::queue& queue, const DeviceMem<DataType>& device_bins, int bins_size, const DeviceMem<DataType>& device_data, int data_size, const std::vector<int>& ref, Impl&& impl, bool measure_and_validate=true)
{
    int hist_size = bins_size - 1;
    auto device_output = malloc_device<int>(hist_size, queue);
    // int hist_size = data_size;
    // auto device_output = malloc_device<int>(data_size, queue);
    queue.fill<int>(device_output.get(), 0, hist_size).wait();

    impl.init(queue, device_bins.get(), bins_size);
    impl.run(queue, device_bins.get(), bins_size, device_data.get(), data_size, device_output.get());
    queue.fill<int>(device_output.get(), 0, hist_size).wait();
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    impl.run(queue, device_bins.get(), bins_size, device_data.get(), data_size, device_output.get());
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

    std::vector<int> hist(hist_size);
    if (measure_and_validate)
    {
        queue.copy(device_output.get(), hist.data(), hist_size).wait();
        std::string validation_result = "[validation passed]";
        if (not validate(hist, ref))
            validation_result = "[validation failed]";

        std::chrono::duration<double, std::milli> time_span = t2 - t1;
        std::cout << impl.name << ": " << time_span.count() << " milliseconds. " << validation_result << std::endl;

        // int total_count_1 = 0;
        // for (auto&& d : ref) total_count_1 += d;

        // int total_count_2 = 0;
        // for (auto&& d : hist) total_count_2 += d;

        // print_hist(ref);
        // std::cout << total_count_1 << std::endl;
        // print_hist(hist);
        // std::cout << total_count_2 << std::endl;
    }
    // print_hist(ref);
    // print_hist(hist);
    // for (int i = 0; i < db.size(); ++i)
    // {
    //     std::cout << std::fixed << std::setprecision(17) << "[" << bins[i] << "," << bins[i+1] << ") ";
    //     print_hist(db[i]);
    // }

    return validate(hist, ref);
}

template<class DataType>
bool run_sycl_impl(const std::vector<DataType>& data, const std::vector<DataType>& bins, const std::vector<int>& ref, cl::sycl::queue queue)
{
    // cl::sycl::queue queue(sycl::gpu_selector_v);
    int bins_size = bins.size();
    auto device_bins = malloc_device<DataType>(bins_size, queue);
    queue.copy(bins.data(), device_bins.get(), bins_size).wait();

    int data_size = data.size();
    auto device_data = malloc_device<DataType>(data_size, queue);
    queue.copy(data.data(), device_data.get(), data_size).wait();

    // measure_and_validate_sycl(queue, device_bins, bins_size, device_data, data_size, ref, SyclIncorrectUniformHist<DataType>());
    // measure_and_validate_sycl(queue, device_bins, bins_size, device_data, data_size, ref, SyclBasicHistImpl<DataType>());
    measure_and_validate_sycl(queue, device_bins, bins_size, device_data, data_size, ref, SyclBasicHistLocalHistImpl<64, DataType>());
    // measure_and_validate_sycl(queue, device_bins, bins_size, device_data, data_size, ref, SyclBasicCustomHistImpl<DataType>());
    // measure_and_validate_sycl(queue, device_bins, bins_size, device_data, data_size, ref, SyclBasicCustomWithStopHistImpl<DataType>());
    measure_and_validate_sycl(queue, device_bins, bins_size, device_data, data_size, ref, SyclNNaryHistImpl<4, 64, DataType>());
    // // measure_and_validate_sycl(queue, device_bins, bins_size, device_data, data_size, ref, SyclNNaryHistImpl<4, DataType>());
    // measure_and_validate_sycl(queue, device_bins, bins_size, device_data, data_size, ref, SyclNNaryHistVectImpl<8, DataType>());
    // measure_and_validate_sycl(queue, device_bins, bins_size, device_data, data_size, ref, SyclNNaryHistVectConstImpl<8, DataType>());
    // // measure_and_validate_sycl(queue, device_bins, bins_size, device_data, data_size, ref, SyclNNaryHistVectImpl<32, DataType>());
    // measure_and_validate_sycl(queue, device_bins, bins_size, device_data, data_size, ref, SyclHistogramUniformMeshImpl<DataType>());
    // measure_and_validate_sycl(queue, device_bins, bins_size, device_data, data_size, ref, SyclHistogramUniformMeshLinearSearchImpl<4, DataType>());
    // measure_and_validate_sycl(queue, device_bins, bins_size, device_data, data_size, ref, SyclHistogramUniformMeshLinearSearchImpl<8, DataType>());
    measure_and_validate_sycl(queue, device_bins, bins_size, device_data, data_size, ref, SyclHistogramUniformMeshLinearSearchLMImpl<4, 64, DataType>());
    // measure_and_validate_sycl(queue, device_bins, bins_size, device_data, data_size, ref, SyclHistogramUniformMeshLinearSearchLMImpl<4, 64, DataType>());

    return true;
}

void print_device_info(const sycl::device& device)
{
    using namespace sycl::info::device;

    std::cout << "Max work group size: " << device.get_info<max_work_group_size>() << std::endl;
}

int main() {
    //Generate data and bucket bins
    // int n = 20;
    // int num_bins = 13;

    // int n = 20000000;
    int n = 256*32*2300;
    int num_bins = 1000;
    // int n = 16;
    // int num_bins = 16;

    float min_value = -10;
    float max_value = 10;

    cl::sycl::queue cpu_queue(sycl::cpu_selector_v);
    cl::sycl::queue gpu_queue(sycl::gpu_selector_v);

    print_device_info(cpu_queue.get_device());
    print_device_info(gpu_queue.get_device());

    // srand(0);

    for (int i = 135995; i < 135996; ++i)
    // for (int i = 1375144; i < 1375145; ++i)
    // for (int i = 2144; i < 2145; ++i)
    {
        srand(i);
        // auto bins = generate_bins<dtype>(num_bins, min_value, max_value);
        auto bins = generate_nu_bins<DType>(num_bins, min_value, max_value);
        // print_hist(bins);
        // auto bins = generate_log_bins<dtype>(num_bins, min_value, max_value);
        // print_hist(bins);
        auto data = generate_data<DType>(n, min_value, max_value);

        auto ref = HistogramTbbCombinableBase::hist(data, bins);
        // auto ref = HistogramRef::hist(data, bins);
        // measure_and_validate<HistogramRef>(data, bins, ref);
        // measure_and_validate<HistogramCustomUBRef>(data, bins, ref);
        // measure_and_validate<HistogramTbbIncorrect>(data, bins, ref);
        // measure_and_validate<HistogramTbbCombinableBase>(data, bins, ref);
        // // measure_and_validate<ParallelSort>(data, bins, ref);
        // // if (not measure_and_validate<HistogramUniformMesh>(data, bins, ref))
        // // {
        // //     std::cout << "Failed on seed " << i << std::endl;
        // //     break;
        // // }
        // if (not measure_and_validate<HistogramTbbUniformMesh>(data, bins, ref))
        // {
        //     std::cout << "Failed on seed " << i << std::endl;
        //     break;
        // }
        // if (not measure_and_validate<HistogramUniformMeshLinearSearch<4>>(data, bins, ref))
        // {
        //     std::cout << "Failed on seed " << i << std::endl;
        //     break;
        // }
        // if (not measure_and_validate<HistogramTbbUniformMeshLinearSearch<8>>(data, bins, ref))
        // {
        //     std::cout << "Failed on seed " << i << std::endl;
        //     break;
        // }
        // if (not measure_and_validate<HistogramNNarySearchWithReorder<4>>(data, bins, ref))
        // {
        //     std::cout << "Failed on seed " << i << std::endl;
        //     break;
        // }
        // if (not measure_and_validate<HistogramTBBNNarySearchWithReorder<8>>(data, bins, ref))
        // {
        //     std::cout << "Failed on seed " << i << std::endl;
        //     break;
        // }

        // if (not run_sycl_impl(data, bins, ref, cpu_queue))
        // {
        //     std::cout << "Failed on seed " << i << std::endl;
        //     break;
        // }
        run_sycl_impl(data, bins, ref, cpu_queue);
        run_sycl_impl(data, bins, ref, gpu_queue);

        // auto begin = bins.data();
        // auto end = begin + bins.size();

        // for (int i = 0; i < n; ++i)
        // {
        //     auto d = data[i];
        //     auto std_ub = std::upper_bound(bins.begin(), bins.end(), d) - bins.begin() - 1;
        //     auto cust_ub = custom_upper_bound(begin, end, d) - begin - 1;

        //     if (std_ub != cust_ub)
        //     {
        //         std::cout << "Wrong result for " << d << " " << std_ub << " " << cust_ub << std::endl;
        //         // break;
        //     }
        // }
    }

    return 0;
}
