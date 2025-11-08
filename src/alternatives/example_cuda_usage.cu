/**
 * Example: Using VariablePhilox CUDA for large-scale sampling
 * 
 * This example demonstrates how to use the CUDA implementation
 * for high-performance bijective shuffling and sampling.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <chrono>
#include <algorithm>

// Include the CUDA kernel definitions (in a real project, this would be in a separate .cu file)
#define M0 UINT64_C(0xD2B74407B1CE6E93)

__device__ __forceinline__ uint32_t mulhilo(uint64_t m, uint32_t x, uint32_t& hi) {
    uint64_t product = m * static_cast<uint64_t>(x);
    hi = static_cast<uint32_t>(product >> 32);
    return static_cast<uint32_t>(product & 0xFFFFFFFF);
}

__global__ void variablePhiloxKernel(
    const uint64_t* input, uint64_t* output, uint64_t n,
    int left_side_bits, int right_side_bits,
    uint32_t left_side_mask, uint32_t right_side_mask,
    int num_rounds, const uint32_t* keys
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    uint64_t val = input[idx];
    uint32_t state[2];
    state[0] = static_cast<uint32_t>((val >> right_side_bits) & left_side_mask);
    state[1] = static_cast<uint32_t>(val & right_side_mask);
    
    for (int i = 0; i < num_rounds; i++) {
        uint32_t hi;
        uint32_t lo = mulhilo(M0, state[0], hi);
        int shift_amount = right_side_bits - left_side_bits;
        if (shift_amount >= 0) {
            lo = (lo << shift_amount) | (state[1] >> left_side_bits);
        } else {
            lo = (lo >> (-shift_amount)) | (state[1] << (-shift_amount));
        }
        state[0] = ((hi ^ keys[i]) ^ state[1]) & left_side_mask;
        state[1] = lo & right_side_mask;
    }
    
    output[idx] = (static_cast<uint64_t>(state[0]) << right_side_bits) | 
                   static_cast<uint64_t>(state[1]);
}

// Simple wrapper for the example
class SimplePhiloxCUDA {
private:
    int left_bits, right_bits, num_rounds;
    uint32_t left_mask, right_mask;
    uint32_t* d_keys;
    
public:
    SimplePhiloxCUDA(int lb, int rb, int rounds, uint64_t seed) 
        : left_bits(lb), right_bits(rb), num_rounds(rounds) {
        left_mask = (1U << lb) - 1;
        right_mask = (1U << rb) - 1;
        
        std::vector<uint32_t> h_keys(rounds);
        for (int i = 0; i < rounds; i++) {
            h_keys[i] = static_cast<uint32_t>((seed * 0x9E3779B97F4A7C15ULL + i) >> 32);
        }
        
        cudaMalloc(&d_keys, rounds * sizeof(uint32_t));
        cudaMemcpy(d_keys, h_keys.data(), rounds * sizeof(uint32_t), cudaMemcpyHostToDevice);
    }
    
    ~SimplePhiloxCUDA() { cudaFree(d_keys); }
    
    void apply(const uint64_t* input, uint64_t* output, uint64_t n) {
        uint64_t *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(uint64_t));
        cudaMalloc(&d_out, n * sizeof(uint64_t));
        
        cudaMemcpy(d_in, input, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        
        variablePhiloxKernel<<<numBlocks, blockSize>>>(
            d_in, d_out, n, left_bits, right_bits,
            left_mask, right_mask, num_rounds, d_keys
        );
        
        cudaMemcpy(output, d_out, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        cudaFree(d_in);
        cudaFree(d_out);
    }
};

int main() {
    std::cout << "VariablePhilox CUDA - Performance Example" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Check CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << std::endl;
    
    // Example 1: Process 1 million values
    std::cout << "Example 1: Large-scale bijection (1M values)" << std::endl;
    const uint64_t N = 1000000;
    
    // Prepare input
    std::vector<uint64_t> input(N);
    std::vector<uint64_t> output(N);
    for (uint64_t i = 0; i < N; i++) {
        input[i] = i;
    }
    
    // Create PhiloxCUDA with 20 bits total (10+10)
    SimplePhiloxCUDA philox(10, 10, 7, 12345);
    
    // Time the GPU computation
    auto start = std::chrono::high_resolution_clock::now();
    philox.apply(input.data(), output.data(), N);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Processed " << N << " values in " << duration.count() << " ms" << std::endl;
    std::cout << "Throughput: " << (N / 1000.0 / duration.count()) << " M values/sec" << std::endl;
    
    // Verify some results are in range
    uint64_t max_val = 1ULL << 20;  // 2^20 = 1048576
    bool all_valid = std::all_of(output.begin(), output.end(), 
                                  [max_val](uint64_t v) { return v < max_val; });
    std::cout << "All outputs in range [0, " << max_val << "): " 
              << (all_valid ? "YES" : "NO") << std::endl;
    
    // Show some sample outputs
    std::cout << "Sample outputs: ";
    for (int i = 0; i < 10; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << "..." << std::endl;
    
    // Example 2: Comparison with sequential range
    std::cout << "\nExample 2: Distribution analysis" << std::endl;
    
    const uint64_t SAMPLE_SIZE = 10000;
    std::vector<uint64_t> small_input(SAMPLE_SIZE);
    std::vector<uint64_t> small_output(SAMPLE_SIZE);
    
    for (uint64_t i = 0; i < SAMPLE_SIZE; i++) {
        small_input[i] = i;
    }
    
    SimplePhiloxCUDA small_philox(7, 7, 7, 999);  // 14 bits = 16384 values
    small_philox.apply(small_input.data(), small_output.data(), SAMPLE_SIZE);
    
    // Check distribution across quarters
    uint64_t quarters[4] = {0, 0, 0, 0};
    for (auto v : small_output) {
        int q = (v * 4) / (1 << 14);
        if (q < 4) quarters[q]++;
    }
    
    std::cout << "Distribution across 4 quarters:" << std::endl;
    for (int i = 0; i < 4; i++) {
        std::cout << "  Q" << i+1 << ": " << quarters[i] << " (" 
                  << (100.0 * quarters[i] / SAMPLE_SIZE) << "%)" << std::endl;
    }
    
    std::cout << "\nCUDA examples completed successfully!" << std::endl;
    
    return 0;
}
