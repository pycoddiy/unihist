# Makefile for VariablePhilox implementations

CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++11 -O3 -Wall
NVCCFLAGS = -O3 --std=c++11

# Targets
CPU_TARGET = varphilox
CUDA_TARGET = varphilox_cuda
TEST_TARGET = test_varphilox

# Source files
CPU_SRC = varphilox.cpp
CUDA_SRC = varphilox.cu
TEST_SRC = test_varphilox.cpp
HEADER = varphilox.h

.PHONY: all cpu cuda test clean help

# Default target
all: cpu cuda

# CPU version
cpu: $(CPU_TARGET)

$(CPU_TARGET): $(CPU_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "CPU version compiled successfully!"
	@echo "Run with: ./$(CPU_TARGET)"

# CUDA version
cuda: $(CUDA_TARGET)

$(CUDA_TARGET): $(CUDA_SRC)
	@if command -v $(NVCC) >/dev/null 2>&1; then \
		$(NVCC) $(NVCCFLAGS) -o $@ $<; \
		echo "CUDA version compiled successfully!"; \
		echo "Run with: ./$(CUDA_TARGET)"; \
	else \
		echo "ERROR: nvcc compiler not found. Please install CUDA Toolkit."; \
		exit 1; \
	fi

# Test program using header library
test: $(TEST_TARGET)

$(TEST_TARGET): $(TEST_SRC) $(HEADER)
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "Test program compiled successfully!"
	@echo "Run with: ./$(TEST_TARGET)"

# Clean build artifacts
clean:
	rm -f $(CPU_TARGET) $(CUDA_TARGET) $(TEST_TARGET)
	@echo "Clean complete!"

# Test CPU version
test-cpu: cpu
	./$(CPU_TARGET)

# Test CUDA version
test-cuda: cuda
	./$(CUDA_TARGET)

# Run header library test
run-test: test
	./$(TEST_TARGET)

# Help
help:
	@echo "VariablePhilox Makefile"
	@echo "======================="
	@echo ""
	@echo "Available targets:"
	@echo "  make all        - Build both CPU and CUDA versions (default)"
	@echo "  make cpu        - Build CPU version only"
	@echo "  make cuda       - Build CUDA version only"
	@echo "  make test       - Build header library test program"
	@echo "  make test-cpu   - Build and run CPU version"
	@echo "  make test-cuda  - Build and run CUDA version"
	@echo "  make run-test   - Build and run header library test"
	@echo "  make clean      - Remove compiled binaries"
	@echo "  make help       - Show this help message"
	@echo ""
	@echo "Files:"
	@echo "  varphilox.cpp   - Standalone CPU implementation with main()"
	@echo "  varphilox.cu    - CUDA GPU implementation"
	@echo "  varphilox.h     - Header-only library (no main, for integration)"
	@echo "  test_varphilox.cpp - Test program for header library"
	@echo ""
	@echo "Requirements:"
	@echo "  CPU:  C++ compiler with C++11 support (g++, clang++)"
	@echo "  CUDA: NVIDIA CUDA Toolkit with nvcc compiler"
