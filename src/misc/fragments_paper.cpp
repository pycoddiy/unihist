template<class DataType>
struct UniformMesh
{
    // Using interval length and bins count instead of step=u_diff/n_bins
    // to reduce computation errors.
    DataType u_diff;
    int n_bins;

    // the beginning of uniformed interval. The uniform bin index is calculated as
    // (x - lower_bound)*n_bins/u_diff
    DataType lower_bound;

    // Offsets and counts. Size is n_bins - 1
    std::vector<int> offsets;
    std::vector<int> counts;

    // Pointers to the next level UniformMesh. Size is n_bins - 1. nullptr if
    // corresponding count <= 1
    std::vector<std::unique_ptr<UniformMesh>> next_mesh;

    UniformMesh(const std::vector<DataType>& bins)
    {
        n_bins = bins.size() - 1;
        offsets.resize(n_bins);
        counts.resize(n_bins);
        next_mesh.resize(n_bins);

        u_diff = bins.back() - bins.front();
        lower_bound = bins[0];

        int curr_nun_bin = 0;
        for (int i = 0; i < n_bins; ++i)
        {
            // lower and upper bound of current uniform bin
            DataType curr_lo_bound = i*u_diff/n_bins;
            DataType curr_up_bound = (i+1)*u_diff/n_bins;

            int nun_bins_count = 0;
            offsets[i] = curr_nun_bin;

            // count how many non-uniform bins bound inside this uniform bin
            while (((bins[curr_nun_bin + 1] - lower_bound) < curr_up_bound) and (curr_nun_bin != (n_bins - 1)))
            {
                ++curr_nun_bin;
                ++nun_bins_count;
            }

            counts[i] = nun_bins_count;

            // if number of bounds count inside uniform bin is more than one create next level UniformMesh
            if (nun_bins_count > 1)
            {
                int new_bins_start = offsets[i];

                // count of non-uniform bins inside uniform bin is count + 1 or count + 2.
                // In most of cases it is count + 2
                // but in some rare cases when uniform bin bound and non-uniform bin matches
                // it is count + 1. We need to distinguish these two case to avoid creation of zero-sized bin.
                int new_bins_end = new_bins_start + nun_bins_count + 1;
                new_bins_end += (bins[new_bins_end] < curr_up_bound);
                std::vector<DataType> new_bins(&bins[new_bins_start], &bins[new_bins_end]);

                // Lower and upper bound of new UniformMesh
                new_bins.front() = curr_lo_bound + lower_bound;
                new_bins.back() =  curr_up_bound + lower_bound;
                next_mesh[i].reset(new UniformMesh(new_bins));
            }
        }
    }

    int _get_non_uniform_bin(const DataType value, int start_offset)
    {
        // calculate uniform bin index
        int u_bin = std::floor((value - lower_bound)*n_bins/u_diff);
        // due to calculation errors sometimes bin index could be greater than actual bins count
        // just restrict u_bin to be not greater than actual bins count.
        u_bin = std::min(u_bin, n_bins - 1);
        int count = counts[u_bin];
        int offset = start_offset + offsets[u_bin];


        // Actually we have 3 cases:
        // 1. Count == 0. This means we (almost) presicely predicted non-Uniform bin.
        //    We should just return resulting offset
        // 2. Count == 1. This means we need to pick from either [offset] or [offset + 1] bin.
        //    Need to do single border check
        // 3. Count > 2. We need to go to the next level Uniform Mesh.

        // But in order to reduce number of branches we could unify case 1 and 2. And check only for case 3.
        if (count > 1)
        {
            offset = next_mesh[u_bin]->_get_non_uniform_bin(value, offset);
        }

        return offset;
    }

    int get_non_uniform_bin(const DataType value, const std::vector<DataType>& bins)
    {
        auto offset = _get_non_uniform_bin(value, 0);
        // Due to computation errors we could miss +-1 bin. Just check bounds and correct, if needed.
        offset = offset - (value < bins[offset]) + (value >= bins[offset + 1]);
        // Since we have unified [count == 0] and [count == 1] case we probably need to move +1 bin
        // Checking border and move if needed.
        return offset + (value >= bins[offset + 1]);
    }
};

template<class DataType>
static std::vector<int> hist(const std::vector<DataType>& data,
                             const std::vector<DataType>& bins) {
    auto hist_size = bins.size() - 1;
    std::vector<int> hist(hist_size);

    // Per-thread local copy of histogram
    using LocalHist = tbb::combinable<std::vector<DataType>>;
    LocalHist local_hist = LocalHist([hist_size] ()
    {
        return std::vector<DataType>(hist_size);
    });

    auto um = UniformMesh<DataType>(bins);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, data.size()),
                     [&](tbb::blocked_range<size_t> r) {
        auto& _hist = local_hist.local();
        for (size_t i = r.begin(); i != r.end(); ++i) {
            auto d = data[i];
            // Verify that we are in bounds of bins
            if (d >= bins.front() and d < bins.back())
                _hist[um.get_non_uniform_bin(d, bins)]++;
        }
    });

    // Collect and sum our local histograms into resulting histogram
    local_hist.combine_each([&hist](const std::vector<DataType>& lh)
    {
        for (int i = 0; i < hist.size(); ++i)
            hist[i] += lh[i];
    });

    return hist;
}
