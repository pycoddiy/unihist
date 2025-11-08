import numpy as np


class VariablePhilox:
    """
    VariablePhilox: A bijective shuffle algorithm based on the Philox random number generator.
    
    This implementation is based on the algorithm described in bijective-shuffle.pdf.
    VariablePhilox creates a bijective mapping from [0, 2^n) to [0, 2^n) where n = left_side_bits + right_side_bits.
    
    The algorithm uses a modified Philox construction that works with variable bit widths,
    making it suitable for creating bijective shuffles of arbitrary ranges by mapping them
    to the nearest power of two and filtering out-of-range values.
    
    Algorithm from Listing 1:
    - Uses constant M0 = 0xD2B74407B1CE6E93 (selected for good mixing properties)
    - Splits input into left and right sides based on bit allocation
    - Applies num_rounds rounds of Philox-style mixing using multiply-high-low operations
    - Each round: multiply, shift, XOR with round key, and recombine
    
    Parameters:
    -----------
    left_side_bits : int
        Number of bits for the left side of the state (default: 32)
    right_side_bits : int
        Number of bits for the right side of the state (default: 32)
    num_rounds : int
        Number of Philox rounds to apply (default: 7, more rounds = better mixing)
    seed : int
        Seed for generating round keys (default: 0)
    """
    
    # Philox multiplication constant (selected for good avalanche properties)
    M0 = np.uint64(0xD2B74407B1CE6E93)
    
    def __init__(self, left_side_bits=32, right_side_bits=32, num_rounds=7, seed=0):
        """
        Initialize VariablePhilox with specified bit widths and number of rounds.
        
        Parameters:
        -----------
        left_side_bits : int
            Number of bits for left side (must be <= 32)
        right_side_bits : int
            Number of bits for right side (must be <= 32)
        num_rounds : int
            Number of mixing rounds (typically 7-10)
        seed : int
            Seed for round key generation
        """
        assert 1 <= left_side_bits <= 32, "left_side_bits must be in [1, 32]"
        assert 1 <= right_side_bits <= 32, "right_side_bits must be in [1, 32]"
        assert num_rounds >= 1, "num_rounds must be >= 1"
        
        self.left_side_bits = left_side_bits
        self.right_side_bits = right_side_bits
        self.num_rounds = num_rounds
        
        # Compute masks for left and right sides
        self.left_side_mask = np.uint32((1 << left_side_bits) - 1)
        self.right_side_mask = np.uint32((1 << right_side_bits) - 1)
        
        # Generate round keys
        self.key = self._generate_keys(seed)
    
    def _generate_keys(self, seed):
        """
        Generate round keys for the Philox rounds.
        
        Uses a simple counter-based scheme where each round key is derived
        from the seed and round index.
        """
        np.random.seed(seed)
        # Generate random keys for each round
        keys = np.random.randint(0, 2**32, size=self.num_rounds, dtype=np.uint32)
        return keys
    
    @staticmethod
    def _mulhilo(m, x):
        """
        Multiply two 32-bit values and return 64-bit result split into high and low 32 bits.
        
        Parameters:
        -----------
        m : uint64
            Multiplication constant
        x : uint32
            Input value
            
        Returns:
        --------
        lo : uint32
            Lower 32 bits of the 64-bit product
        hi : uint32
            Upper 32 bits of the 64-bit product
        """
        # Perform 64-bit multiplication using Python integers to avoid overflow warnings
        product = int(m) * int(x)
        lo = np.uint32(product & 0xFFFFFFFF)
        hi = np.uint32((product >> 32) & 0xFFFFFFFF)
        return lo, hi
    
    def __call__(self, val):
        """
        Apply the VariablePhilox bijective function to a single value.
        
        Parameters:
        -----------
        val : int or uint64
            Input value in range [0, 2^(left_side_bits + right_side_bits))
            
        Returns:
        --------
        result : uint64
            Bijectively mapped output value in same range as input
        """
        val = np.uint64(val)
        
        # Split val into left (high) and right (low) sides
        state = np.zeros(2, dtype=np.uint32)
        state[0] = np.uint32((val >> self.right_side_bits) & self.left_side_mask)
        state[1] = np.uint32(val & self.right_side_mask)
        
        # Apply num_rounds rounds of Philox mixing
        for i in range(self.num_rounds):
            # Multiply state[0] by M0 and get high and low 32 bits
            lo, hi = self._mulhilo(self.M0, state[0])
            
            # Shift and combine with state[1]
            # Note: The shift operation creates mixing between left and right sides
            shift_amount = self.right_side_bits - self.left_side_bits
            if shift_amount >= 0:
                lo = np.uint32((lo << shift_amount) | (state[1] >> self.left_side_bits))
            else:
                lo = np.uint32((lo >> (-shift_amount)) | (state[1] << (-shift_amount)))
            
            # Update state with XOR mixing and masking
            state[0] = np.uint32(((hi ^ self.key[i]) ^ state[1]) & self.left_side_mask)
            state[1] = np.uint32(lo & self.right_side_mask)
        
        # Combine left and right sides to get final result
        result = (np.uint64(state[0]) << self.right_side_bits) | np.uint64(state[1])
        return result
    
    def apply_vectorized(self, vals):
        """
        Apply VariablePhilox to an array of values.
        
        Parameters:
        -----------
        vals : array-like
            Input values
            
        Returns:
        --------
        results : ndarray
            Bijectively mapped output values
        """
        vals = np.asarray(vals, dtype=np.uint64)
        results = np.zeros_like(vals, dtype=np.uint64)
        for i, val in enumerate(vals):
            results[i] = self(val)
        return results
    
    def inverse(self, val):
        """
        Compute the inverse of the VariablePhilox function.
        
        Since VariablePhilox is bijective, each output has a unique input.
        This finds it by running the Philox rounds in reverse.
        
        Note: This is a simplified implementation. A proper inverse would
        require inverting the Philox rounds, which is more complex.
        
        Parameters:
        -----------
        val : int or uint64
            Output value to invert
            
        Returns:
        --------
        result : uint64
            Original input that maps to val
        """
        # For now, use brute force search for small ranges
        # A proper implementation would invert the Philox structure
        raise NotImplementedError("Inverse function requires Philox round inversion")


def bijective_shuffle(n, k, seed=0):
    """
    Generate k samples from range [0, n) using bijective shuffle method.
    
    This implements the shuffle compaction technique from the paper:
    1. Extend n to the next power of two: m = 2^ceil(log2(n))
    2. Use VariablePhilox to create a bijection on [0, m)
    3. Apply bijection to indices [0, k) and filter results to keep only values < n
    4. Continue until k valid samples are obtained
    
    Parameters:
    -----------
    n : int
        Size of the range to sample from
    k : int
        Number of samples to generate
    seed : int
        Random seed
        
    Returns:
    --------
    samples : ndarray
        k unique samples from [0, n)
    """
    # Find the next power of two >= n
    bits_needed = int(np.ceil(np.log2(n)))
    m = 1 << bits_needed
    
    # Split bits roughly evenly between left and right
    left_bits = bits_needed // 2
    right_bits = bits_needed - left_bits
    
    # Create VariablePhilox bijection
    philox = VariablePhilox(left_side_bits=left_bits, 
                           right_side_bits=right_bits,
                           num_rounds=7,
                           seed=seed)
    
    samples = []
    index = 0
    
    # Keep applying bijection until we have k valid samples
    while len(samples) < k:
        # Apply bijection to current index
        mapped = philox(index)
        
        # Check if mapped value is in valid range [0, n)
        if mapped < n:
            samples.append(mapped)
        
        index += 1
        
        # Safety check: if we've checked m values, we should have found something
        if index >= m and len(samples) == 0:
            raise RuntimeError("Failed to generate any valid samples")
    
    return np.array(samples[:k], dtype=np.uint64)


if __name__ == "__main__":
    # Test VariablePhilox with simple examples
    print("Testing VariablePhilox:")
    print("=" * 60)
    
    # Test 1: Small range (16 values = 4 bits)
    print("\nTest 1: 4-bit bijection (range [0, 16))")
    philox_4bit = VariablePhilox(left_side_bits=2, right_side_bits=2, num_rounds=7, seed=42)
    
    inputs = np.arange(16, dtype=np.uint64)
    outputs = philox_4bit.apply_vectorized(inputs)
    
    print(f"Inputs:  {inputs}")
    print(f"Outputs: {outputs}")
    print(f"All unique: {len(np.unique(outputs)) == len(outputs)}")
    print(f"Same range: {np.all((outputs >= 0) & (outputs < 16))}")
    
    # Test 2: Larger range
    print("\nTest 2: 8-bit bijection (range [0, 256))")
    philox_8bit = VariablePhilox(left_side_bits=4, right_side_bits=4, num_rounds=7, seed=123)
    
    test_vals = [0, 1, 2, 100, 200, 255]
    print("Sample mappings:")
    for val in test_vals:
        mapped = philox_8bit(val)
        print(f"  {val:3d} -> {mapped:3d}")
    
    # Test 3: Bijective shuffle for sampling
    print("\nTest 3: Bijective shuffle sampling")
    n, k = 20, 9
    samples = bijective_shuffle(n, k, seed=777)
    print(f"Sample {k} values from [0, {n}):")
    print(f"Samples: {samples}")
    print(f"All unique: {len(np.unique(samples)) == len(samples)}")
    print(f"All in range: {np.all((samples >= 0) & (samples < n))}")
