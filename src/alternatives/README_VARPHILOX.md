# VariablePhilox CUDA Implementation

This directory contains multiple implementations of the VariablePhilox bijective shuffle algorithm.

## Files

- **`varphilox.py`** - Python/NumPy implementation with examples
- **`varphilox.cpp`** - Standalone C++ implementation (includes main())
- **`varphilox.cu`** - CUDA/GPU implementation (includes main())
- **`varphilox.h`** - Header-only C++ library (for easy integration)
- **`test_varphilox.cpp`** - Test program for the header library
- **`Makefile`** - Build system for all C++/CUDA versions
- **`README_VARPHILOX.md`** - This documentation file

## Quick Start

### Python Version
```bash
python varphilox.py
```

### C++ Version
```bash
make cpu        # Compile
./varphilox     # Run
```

### CUDA Version
```bash
make cuda           # Compile (requires CUDA Toolkit)
./varphilox_cuda    # Run
```

### Header Library
```bash
make run-test   # Compile and run test
```

## Requirements

### Python (varphilox.py)
- Python 3.6+
- NumPy

### CPU C++ versions (varphilox.cpp, varphilox.h)
- C++ compiler with C++11 support (g++, clang++, MSVC)
- No external dependencies

### CUDA version (varphilox.cu)
- NVIDIA GPU with CUDA support (Compute Capability 3.0+)
- CUDA Toolkit (version 10.0 or later recommended)
- nvcc compiler

## Compilation

### CPU Version

```bash
g++ -std=c++11 -O3 -o varphilox varphilox.cpp
./varphilox
```

### CUDA Version

```bash
nvcc -O3 -o varphilox_cuda varphilox.cu
./varphilox_cuda
```

Or use the provided Makefile:

```bash
make cpu      # Compile CPU version
make cuda     # Compile CUDA version
make all      # Compile both versions
make clean    # Clean build artifacts
```

## Usage

### C++ API

```cpp
#include "varphilox.cpp"

// Create a VariablePhilox instance
VariablePhilox philox(
    left_bits,    // Number of bits for left side
    right_bits,   // Number of bits for right side  
    num_rounds,   // Number of mixing rounds (typically 7)
    seed          // Random seed for key generation
);

// Apply bijection to a single value
uint64_t result = philox(input_value);

// Generate samples using bijective shuffle
auto samples = bijective_shuffle(n, k, seed);
```

### CUDA API

```cpp
#include "varphilox.cu"

// Create a CUDA VariablePhilox instance
VariablePhiloxCUDA philox(
    left_bits,    // Number of bits for left side
    right_bits,   // Number of bits for right side
    num_rounds,   // Number of mixing rounds (typically 7)
    seed          // Random seed for key generation
);

// Apply bijection to an array of values (GPU-accelerated)
std::vector<uint64_t> inputs(n);
std::vector<uint64_t> outputs(n);
// ... fill inputs ...
philox.apply(inputs.data(), outputs.data(), n);

// Generate samples using bijective shuffle on GPU
auto samples = philox.bijective_shuffle(target_n, k);
```

## Algorithm Details

VariablePhilox creates a bijective (one-to-one and onto) mapping from [0, 2^n) to [0, 2^n), 
where n = left_side_bits + right_side_bits.

The algorithm:
1. Splits the input value into left and right sides based on bit allocation
2. Applies multiple rounds of Philox-style mixing:
   - Multiply-high-low operation with constant M0
   - XOR mixing with round keys
   - Bit shifting and recombination
3. Combines the final state to produce the output

### Key Features

- **Bijective**: Every input maps to a unique output, and every output has exactly one input
- **Variable width**: Supports arbitrary bit widths (up to 32 bits per side)
- **Parallel-friendly**: Each computation is independent, making it ideal for GPU acceleration
- **Cryptographically-inspired**: Based on the Philox random number generator

## Performance

The CUDA version provides significant speedup for large-scale operations:

- **CPU**: Good for small-to-medium datasets (< 10K elements)
- **CUDA**: Ideal for large datasets (> 100K elements)
  - Processes thousands of values in parallel
  - Typical speedup: 10-100x vs CPU for large arrays
