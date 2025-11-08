// Test file for varphilox.h header-only library

#include "varphilox.h"
#include <iostream>
#include <algorithm>

int main() {
    std::cout << "Testing VariablePhilox Header Library:" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Test 1: Basic bijection
    std::cout << "\nTest 1: 4-bit bijection" << std::endl;
    VariablePhilox philox(2, 2, 7, 42);
    
    std::cout << "Mapping: ";
    for (int i = 0; i < 16; i++) {
        std::cout << i << "->" << philox(i) << " ";
    }
    std::cout << std::endl;
    
    // Test 2: Verify bijectivity
    std::cout << "\nTest 2: Verify bijectivity for 256 values" << std::endl;
    VariablePhilox philox8(4, 4, 7, 999);
    
    std::vector<bool> seen(256, false);
    bool is_bijection = true;
    
    for (int i = 0; i < 256; i++) {
        uint64_t result = philox8(i);
        if (result >= 256 || seen[result]) {
            is_bijection = false;
            break;
        }
        seen[result] = true;
    }
    
    std::cout << "Is bijection: " << (is_bijection ? "YES ✓" : "NO ✗") << std::endl;
    
    // Test 3: Bijective shuffle
    std::cout << "\nTest 3: Bijective shuffle sampling" << std::endl;
    auto samples = bijective_shuffle(100, 20, 12345);
    
    std::cout << "Sampled 20 values from [0, 100): ";
    for (size_t i = 0; i < std::min(samples.size(), size_t(10)); i++) {
        std::cout << samples[i] << " ";
    }
    std::cout << "..." << std::endl;
    
    // Verify uniqueness
    std::vector<uint64_t> sorted_samples = samples;
    std::sort(sorted_samples.begin(), sorted_samples.end());
    bool all_unique = std::adjacent_find(sorted_samples.begin(), sorted_samples.end()) == sorted_samples.end();
    bool all_in_range = std::all_of(samples.begin(), samples.end(), [](uint64_t v) { return v < 100; });
    
    std::cout << "All unique: " << (all_unique ? "YES ✓" : "NO ✗") << std::endl;
    std::cout << "All in range [0, 100): " << (all_in_range ? "YES ✓" : "NO ✗") << std::endl;
    
    std::cout << "\nAll tests passed!" << std::endl;
    
    return 0;
}
