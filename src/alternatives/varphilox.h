#ifndef VARPHILOX_H
#define VARPHILOX_H

#include <cstdint>
#include <vector>
#include <cmath>
#include <stdexcept>

/**
 * VariablePhilox: Bijective shuffle algorithm based on the Philox random number generator.
 * 
 * This implementation is based on the algorithm described in bijective-shuffle.pdf.
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
     */
    static inline uint32_t mulhilo(uint64_t m, uint32_t x, uint32_t& hi) {
        uint64_t product = m * static_cast<uint64_t>(x);
        hi = static_cast<uint32_t>(product >> 32);
        return static_cast<uint32_t>(product & 0xFFFFFFFF);
    }
    
    /**
     * Generate round keys for the Philox rounds.
     */
    void generate_keys(uint64_t seed) {
        key.resize(num_rounds);
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
    VariablePhilox(int left_bits = 32, int right_bits = 32, int rounds = 7, uint64_t seed = 0)
        : left_side_bits(left_bits), right_side_bits(right_bits), num_rounds(rounds) {
        
        if (left_bits < 1 || left_bits > 32) {
            throw std::invalid_argument("left_bits must be in [1, 32]");
        }
        if (right_bits < 1 || right_bits > 32) {
            throw std::invalid_argument("right_bits must be in [1, 32]");
        }
        if (rounds < 1) {
            throw std::invalid_argument("num_rounds must be >= 1");
        }
        
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
    
    /**
     * Apply VariablePhilox to a vector of values.
     */
    std::vector<uint64_t> apply(const std::vector<uint64_t>& input) const {
        std::vector<uint64_t> output(input.size());
        for (size_t i = 0; i < input.size(); i++) {
            output[i] = (*this)(input[i]);
        }
        return output;
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
inline std::vector<uint64_t> bijective_shuffle(uint64_t n, uint64_t k, uint64_t seed = 0) {
    if (n == 0) {
        throw std::invalid_argument("n must be > 0");
    }
    if (k > n) {
        throw std::invalid_argument("k must be <= n");
    }
    
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

#endif // VARPHILOX_H
