#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <cmath>
#include <iostream>

/**
 * CUDA implementation of VariablePhilox: Bijective shuffle algorithm
 * 
 * This implementation is based on the algorithm described in bijective-shuffle.pdf.
 * VariablePhilox creates a bijective mapping from [0, 2^n) to [0, 2^n) where
 * n = left_side_bits + right_side_bits.
 * 
 * The CUDA version provides GPU-accelerated parallel computation of the bijection
 * for large-scale sampling and shuffling operations.
 */

// Philox multiplication constant (selected for good avalanche properties)
#define M0 UINT64_C(0xD2B74407B1CE6E93)

/**
 * CUDA device function: Multiply two values and return 64-bit result split into high and low 32 bits.
 * 
 * @param m Multiplication constant
 * @param x Input value
 * @param hi Output: upper 32 bits of product
 * @return Lower 32 bits of product
 */
__device__ __forceinline__ uint32_t mulhilo(uint64_t m, uint32_t x, uint32_t& hi) {
    uint64_t product = m * static_cast<uint64_t>(x);
    hi = static_cast<uint32_t>(product >> 32);
    return static_cast<uint32_t>(product & 0xFFFFFFFF);
}

/**
 * CUDA kernel: Apply VariablePhilox bijection to array of values
 * 
 * @param input Input array of values
 * @param output Output array for bijected values
 * @param n Number of elements to process
 * @param left_side_bits Number of bits for left side
 * @param right_side_bits Number of bits for right side
 * @param left_side_mask Mask for left side
 * @param right_side_mask Mask for right side
 * @param num_rounds Number of Philox rounds
 * @param keys Array of round keys
 */
__global__ void variablePhiloxKernel(
    const uint64_t* input,
    uint64_t* output,
    uint64_t n,
    int left_side_bits,
    int right_side_bits,
    uint32_t left_side_mask,
    uint32_t right_side_mask,
    int num_rounds,
    const uint32_t* keys
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    uint64_t val = input[idx];
    
    // Split val into left (high) and right (low) sides
    uint32_t state[2];
    state[0] = static_cast<uint32_t>((val >> right_side_bits) & left_side_mask);
    state[1] = static_cast<uint32_t>(val & right_side_mask);
    
    // Apply num_rounds rounds of Philox mixing
    for (int i = 0; i < num_rounds; i++) {
        uint32_t hi;
        uint32_t lo = mulhilo(M0, state[0], hi);
        
        // Shift and combine with state[1]
        int shift_amount = right_side_bits - left_side_bits;
        if (shift_amount >= 0) {
            lo = (lo << shift_amount) | (state[1] >> left_side_bits);
        } else {
            lo = (lo >> (-shift_amount)) | (state[1] << (-shift_amount));
        }
        
        // Update state with XOR mixing and masking
        state[0] = ((hi ^ keys[i]) ^ state[1]) & left_side_mask;
        state[1] = lo & right_side_mask;
    }
    
    // Combine the left and right sides together to get result
    output[idx] = (static_cast<uint64_t>(state[0]) << right_side_bits) | 
                   static_cast<uint64_t>(state[1]);
}

/**
 * CUDA kernel: Bijective shuffle with compaction
 * 
 * This kernel applies the bijection and compacts results that fall within [0, target_n)
 * 
 * @param indices Input indices to map
 * @param output Output array for valid samples
 * @param valid_count Output: count of valid samples found
 * @param n Number of input indices
 * @param target_n Maximum valid value (samples must be < target_n)
 * @param left_side_bits Number of bits for left side
 * @param right_side_bits Number of bits for right side
 * @param left_side_mask Mask for left side
 * @param right_side_mask Mask for right side
 * @param num_rounds Number of Philox rounds
 * @param keys Array of round keys
 */
__global__ void bijectiveShuffleKernel(
    const uint64_t* indices,
    uint64_t* output,
    uint64_t* valid_flags,
    uint64_t n,
    uint64_t target_n,
    int left_side_bits,
    int right_side_bits,
    uint32_t left_side_mask,
    uint32_t right_side_mask,
    int num_rounds,
    const uint32_t* keys
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    uint64_t val = indices[idx];
    
    // Split val into left (high) and right (low) sides
    uint32_t state[2];
    state[0] = static_cast<uint32_t>((val >> right_side_bits) & left_side_mask);
    state[1] = static_cast<uint32_t>(val & right_side_mask);
    
    // Apply num_rounds rounds of Philox mixing
    for (int i = 0; i < num_rounds; i++) {
        uint32_t hi;
        uint32_t lo = mulhilo(M0, state[0], hi);
        
        // Shift and combine with state[1]
        int shift_amount = right_side_bits - left_side_bits;
        if (shift_amount >= 0) {
            lo = (lo << shift_amount) | (state[1] >> left_side_bits);
        } else {
            lo = (lo >> (-shift_amount)) | (state[1] << (-shift_amount));
        }
        
        // Update state with XOR mixing and masking
        state[0] = ((hi ^ keys[i]) ^ state[1]) & left_side_mask;
        state[1] = lo & right_side_mask;
    }
    
    // Combine the left and right sides together to get result
    uint64_t result = (static_cast<uint64_t>(state[0]) << right_side_bits) | 
                      static_cast<uint64_t>(state[1]);
    
    // Mark as valid if within target range
    if (result < target_n) {
        valid_flags[idx] = 1;
        output[idx] = result;
    } else {
        valid_flags[idx] = 0;
    }
}

/**
 * C++ wrapper class for CUDA VariablePhilox
 */
class VariablePhiloxCUDA {
private:
    int left_side_bits;
    int right_side_bits;
    int num_rounds;
    uint32_t left_side_mask;
    uint32_t right_side_mask;
    uint32_t* d_keys;  // Device memory for keys
    std::vector<uint32_t> h_keys;  // Host memory for keys
    
    /**
     * Generate round keys for the Philox rounds.
     */
    void generate_keys(uint64_t seed) {
        h_keys.resize(num_rounds);
        // Simple key generation
        for (int i = 0; i < num_rounds; i++) {
            h_keys[i] = static_cast<uint32_t>((seed * 0x9E3779B97F4A7C15ULL + i) >> 32);
        }
        
        // Copy keys to device
        cudaMalloc(&d_keys, num_rounds * sizeof(uint32_t));
        cudaMemcpy(d_keys, h_keys.data(), num_rounds * sizeof(uint32_t), cudaMemcpyHostToDevice);
    }
    
public:
    /**
     * Constructor
     * 
     * @param left_bits Number of bits for left side (must be <= 32)
     * @param right_bits Number of bits for right side (must be <= 32)
     * @param rounds Number of mixing rounds (typically 7-10)
     * @param seed Seed for round key generation
     */
    VariablePhiloxCUDA(int left_bits = 32, int right_bits = 32, int rounds = 7, uint64_t seed = 0)
        : left_side_bits(left_bits), right_side_bits(right_bits), num_rounds(rounds), d_keys(nullptr) {
        
        // Compute masks for left and right sides
        left_side_mask = (1U << left_bits) - 1;
        right_side_mask = (1U << right_bits) - 1;
        
        // Generate round keys
        generate_keys(seed);
    }
    
    /**
     * Destructor - free device memory
     */
    ~VariablePhiloxCUDA() {
        if (d_keys) {
            cudaFree(d_keys);
        }
    }
    
    /**
     * Apply VariablePhilox to an array of values on GPU
     * 
     * @param input Host array of input values
     * @param output Host array for output values
     * @param n Number of elements
     */
    void apply(const uint64_t* input, uint64_t* output, uint64_t n) {
        // Allocate device memory
        uint64_t* d_input;
        uint64_t* d_output;
        cudaMalloc(&d_input, n * sizeof(uint64_t));
        cudaMalloc(&d_output, n * sizeof(uint64_t));
        
        // Copy input to device
        cudaMemcpy(d_input, input, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        // Launch kernel
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        
        variablePhiloxKernel<<<numBlocks, blockSize>>>(
            d_input, d_output, n,
            left_side_bits, right_side_bits,
            left_side_mask, right_side_mask,
            num_rounds, d_keys
        );
        
        // Copy output back to host
        cudaMemcpy(output, d_output, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);
    }
    
    /**
     * Generate k samples from range [0, target_n) using bijective shuffle
     * 
     * @param target_n Size of the range to sample from
     * @param k Number of samples desired
     * @return Vector of samples
     */
    std::vector<uint64_t> bijective_shuffle(uint64_t target_n, uint64_t k) {
        // We'll process in batches to find k valid samples
        uint64_t batch_size = std::max(k * 2, 1024ULL);  // Oversample to reduce iterations
        
        std::vector<uint64_t> samples;
        samples.reserve(k);
        
        uint64_t index = 0;
        
        while (samples.size() < k) {
            // Create batch of indices
            std::vector<uint64_t> h_indices(batch_size);
            for (uint64_t i = 0; i < batch_size; i++) {
                h_indices[i] = index + i;
            }
            
            // Allocate device memory
            uint64_t* d_indices;
            uint64_t* d_output;
            uint64_t* d_valid_flags;
            
            cudaMalloc(&d_indices, batch_size * sizeof(uint64_t));
            cudaMalloc(&d_output, batch_size * sizeof(uint64_t));
            cudaMalloc(&d_valid_flags, batch_size * sizeof(uint64_t));
            
            // Copy indices to device
            cudaMemcpy(d_indices, h_indices.data(), batch_size * sizeof(uint64_t), cudaMemcpyHostToDevice);
            
            // Launch kernel
            int blockSize = 256;
            int numBlocks = (batch_size + blockSize - 1) / blockSize;
            
            bijectiveShuffleKernel<<<numBlocks, blockSize>>>(
                d_indices, d_output, d_valid_flags, batch_size, target_n,
                left_side_bits, right_side_bits,
                left_side_mask, right_side_mask,
                num_rounds, d_keys
            );
            
            // Copy results back
            std::vector<uint64_t> h_output(batch_size);
            std::vector<uint64_t> h_valid_flags(batch_size);
            
            cudaMemcpy(h_output.data(), d_output, batch_size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_valid_flags.data(), d_valid_flags, batch_size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            
            // Collect valid samples
            for (uint64_t i = 0; i < batch_size && samples.size() < k; i++) {
                if (h_valid_flags[i]) {
                    samples.push_back(h_output[i]);
                }
            }
            
            // Free device memory
            cudaFree(d_indices);
            cudaFree(d_output);
            cudaFree(d_valid_flags);
            
            index += batch_size;
        }
        
        // Trim to exactly k samples
        samples.resize(k);
        return samples;
    }
};

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * Test/demo code
 */
int main() {
    std::cout << "Testing VariablePhilox CUDA:" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Check CUDA availability
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "Using CUDA device: " << deviceProp.name << std::endl;
    std::cout << std::endl;
    
    // Test 1: Small range (16 values = 4 bits)
    std::cout << "Test 1: 4-bit bijection (range [0, 16))" << std::endl;
    VariablePhiloxCUDA philox_4bit(2, 2, 7, 42);
    
    std::vector<uint64_t> inputs(16);
    std::vector<uint64_t> outputs(16);
    
    for (int i = 0; i < 16; i++) {
        inputs[i] = i;
    }
    
    philox_4bit.apply(inputs.data(), outputs.data(), 16);
    
    std::cout << "Inputs:  ";
    for (auto v : inputs) std::cout << v << " ";
    std::cout << "\nOutputs: ";
    for (auto v : outputs) std::cout << v << " ";
    std::cout << std::endl;
    
    // Test 2: Larger parallel computation
    std::cout << "\nTest 2: Large-scale parallel bijection (1024 values)" << std::endl;
    VariablePhiloxCUDA philox_10bit(5, 5, 7, 123);
    
    const uint64_t N = 1024;
    std::vector<uint64_t> large_inputs(N);
    std::vector<uint64_t> large_outputs(N);
    
    for (uint64_t i = 0; i < N; i++) {
        large_inputs[i] = i;
    }
    
    philox_10bit.apply(large_inputs.data(), large_outputs.data(), N);
    
    // Verify bijectivity
    std::vector<bool> seen(N, false);
    bool is_bijection = true;
    for (auto v : large_outputs) {
        if (v >= N || seen[v]) {
            is_bijection = false;
            break;
        }
        seen[v] = true;
    }
    
    std::cout << "Processed " << N << " values in parallel" << std::endl;
    std::cout << "Is bijection: " << (is_bijection ? "Yes" : "No") << std::endl;
    std::cout << "Sample outputs: ";
    for (int i = 0; i < 10; i++) {
        std::cout << large_outputs[i] << " ";
    }
    std::cout << "..." << std::endl;
    
    // Test 3: Bijective shuffle for sampling
    std::cout << "\nTest 3: Bijective shuffle sampling on GPU" << std::endl;
    uint64_t n = 100, k = 15;
    
    // Determine bit allocation
    int bits_needed = static_cast<int>(std::ceil(std::log2(n)));
    int left_bits = bits_needed / 2;
    int right_bits = bits_needed - left_bits;
    
    VariablePhiloxCUDA sampler(left_bits, right_bits, 7, 777);
    auto samples = sampler.bijective_shuffle(n, k);
    
    std::cout << "Sample " << k << " values from [0, " << n << "):" << std::endl;
    std::cout << "Samples: ";
    for (auto s : samples) {
        std::cout << s << " ";
    }
    std::cout << std::endl;
    
    // Verify uniqueness
    std::vector<bool> sample_seen(n, false);
    bool all_unique = true;
    for (auto s : samples) {
        if (s >= n || sample_seen[s]) {
            all_unique = false;
            break;
        }
        sample_seen[s] = true;
    }
    std::cout << "All unique: " << (all_unique ? "Yes" : "No") << std::endl;
    std::cout << "All in range: " << (std::all_of(samples.begin(), samples.end(), 
                                                   [n](uint64_t v) { return v < n; }) ? "Yes" : "No") << std::endl;
    
    return 0;
}
