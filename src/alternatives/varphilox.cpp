#include <cstdint>
#include <vector>
#include <cmath>
#include <iostream>

/**
 * VariablePhilox: Bijective shuffle algorithm based on the Philox random number generator.
 * 
 * This implementation is based on the algorithm described in https://arxiv.org/pdf/2106.06161.
 * VariablePhilox creates a bijective mapping from [0, 2^n) to [0, 2^n) where
 * n = left_side_bits + right_side_bits.
 * 
 * The algorithm uses a modified Philox construction that works with variable bit widths,
 * making it suitable for creating bijective shuffles of arbitrary ranges.
 */
class VariablePhilox {
private:
    int left_side_bits;
    int right_side_bits;
    int num_rounds;
    uint32_t left_side_mask;
    uint32_t right_side_mask;
    std::vector<uint32_t> key;
    
    // Philox multiplication constant
    static constexpr uint64_t M0 = UINT64_C(0xD2B74407B1CE6E93);
    
    /**
     * Multiply two values and return 64-bit result split into high and low 32 bits.
     * 
     * @param m Multiplication constant
     * @param x Input value
     * @param hi Output: upper 32 bits of product
     * @return Lower 32 bits of product
     */
    static inline uint32_t mulhilo(uint64_t m, uint32_t x, uint32_t& hi) {
        uint64_t product = m * static_cast<uint64_t>(x);
        hi = static_cast<uint32_t>(product >> 32);
        return static_cast<uint32_t>(product & 0xFFFFFFFF);
    }
    
    /**
     * Generate round keys for the Philox rounds.
     * Simple counter-based scheme for demonstration.
     */
    void generate_keys(uint64_t seed) {
        key.resize(num_rounds);
        // TODO: Replace with a proper PRNG
        for (int i = 0; i < num_rounds; i++) {
            key[i] = static_cast<uint32_t>((seed * 0x9E3779B97F4A7C15ULL + i) >> 32);
        }
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
    VariablePhilox(int left_bits = 32, int right_bits = 32, int rounds = 7, uint64_t seed = 7777777)
        : left_side_bits(left_bits), right_side_bits(right_bits), num_rounds(rounds) {
        
        // Compute masks for left and right sides
        left_side_mask = (1U << left_bits) - 1;
        right_side_mask = (1U << right_bits) - 1;
        
        // Generate round keys
        generate_keys(seed);
    }
    
    /**
     * Apply the VariablePhilox bijective function to a single value.
     * 
     * @param val Input value in range [0, 2^(left_side_bits + right_side_bits))
     * @return Bijectively mapped output value in same range as input
     */
    uint64_t operator()(const uint64_t val) const {
        // Split val into left (high) and right (low) sides
        uint32_t state[2] = { 
            static_cast<uint32_t>((val >> right_side_bits) & left_side_mask),
            static_cast<uint32_t>(val & right_side_mask)
        };

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
            state[0] = ((hi ^ key[i]) ^ state[1]) & left_side_mask;
            state[1] = lo & right_side_mask;
        }
        
        // Combine the left and right sides together to get result
        return (static_cast<uint64_t>(state[0]) << right_side_bits) | 
               static_cast<uint64_t>(state[1]);
    }
};

/**
 * Generate k samples from range [0, n) using bijective shuffle method.
 * 
 * This implements the shuffle compaction technique from the paper:
 * 1. Extend n to the next power of two: m = 2^ceil(log2(n))
 * 2. Use VariablePhilox to create a bijection on [0, m)
 * 3. Apply bijection to indices and filter results to keep only values < n
 * 4. Continue until k valid samples are obtained
 */
std::vector<uint64_t> bijective_shuffle(uint64_t n, uint64_t k, uint64_t seed = 0) {
    // Find the next power of two >= n
    int bits_needed = static_cast<int>(std::ceil(std::log2(n)));
    uint64_t m = 1ULL << bits_needed;
    
    // Split bits roughly evenly between left and right
    int left_bits = bits_needed / 2;
    int right_bits = bits_needed - left_bits;
    
    // Create VariablePhilox bijection
    VariablePhilox philox(left_bits, right_bits, 7, seed);
    
    std::vector<uint64_t> samples;
    samples.reserve(k);
    
    uint64_t index = 0;
    
    // Keep applying bijection until we have k valid samples
    while (samples.size() < k) {
        // Apply bijection to current index
        uint64_t mapped = philox(index);
        
        // Check if mapped value is in valid range [0, n)
        if (mapped < n) {
            samples.push_back(mapped);
        }
        
        index++;
        
        // Safety check
        if (index >= m && samples.empty()) {
            throw std::runtime_error("Failed to generate any valid samples");
        }
    }
    
    return samples;
}

// Test/demo code
int main() {
    std::cout << "Testing VariablePhilox:" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Test 1: Small range (16 values = 4 bits)
    std::cout << "\nTest 1: 4-bit bijection (range [0, 16))" << std::endl;
    VariablePhilox philox_4bit(2, 2, 7, 42);
    
    std::cout << "Inputs:  ";
    for (int i = 0; i < 16; i++) {
        std::cout << i << " ";
    }
    std::cout << "\nOutputs: ";
    for (int i = 0; i < 16; i++) {
        std::cout << philox_4bit(i) << " ";
    }
    std::cout << std::endl;
    
    // Test 2: Larger range
    std::cout << "\nTest 2: 8-bit bijection (range [0, 256))" << std::endl;
    VariablePhilox philox_8bit(4, 4, 7, 123);
    
    std::cout << "Sample mappings:" << std::endl;
    int test_vals[] = {0, 1, 2, 100, 200, 255};
    for (int val : test_vals) {
        std::cout << "  " << val << " -> " << philox_8bit(val) << std::endl;
    }
    
    // Test 3: Bijective shuffle for sampling
    std::cout << "\nTest 3: Bijective shuffle sampling" << std::endl;
    uint64_t n = 20, k = 9;
    auto samples = bijective_shuffle(n, k, 777);
    std::cout << "Sample " << k << " values from [0, " << n << "):" << std::endl;
    std::cout << "Samples: ";
    for (uint64_t s : samples) {
        std::cout << s << " ";
    }
    std::cout << std::endl;
    
    return 0;
}