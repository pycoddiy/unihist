import numpy as np

# Global constants
N = 20
K = 9


def fisher_sampling_1(n, k):
    """
    Consider the classical swapping, or Fisher-Yates, method for generating a simple ran-
    dom sample given below:
    
    Algorithm 1 Fisher Yates Sampler(n,k)
    x ← 1, ... , n
    for i = 0 → k-1 do
        r ← Uniform([n-i])
        Swap(x[n-i], x[r])
    end for
    return x[(n-k+1):n]

    In the classical swapping procedure, the last i elements in the array form a random
    sample without replacement after the ith iteration. The first n-i elements consist of
    all items that have not been sampled. By performing a swap, the algorithm moves an
    item from the remaining items that have not been sampled and adds it to the sampled
    items. Thus, each iteration exactly mimics the process of sequentially sampling without
    replacement:

        1. Sample index r from remaining items
        2. Remove x[r] from the remaining items, and move it to the sampled items
    
    This algorithm takes both O(n) time and space since it initializes an array of length
    n. This initialization step is wasteful when k ≪ n is small. In this case, at most 2k
    locations in the array can be affected by swaps. A majority of locations in the array
    contain no meaningful information and simply store x[i] = i
    """
    x = np.arange(n)
    for i in range(k):
        r = np.random.randint(0, n - i)
        x[n - i - 1], x[r] = x[r], x[n - i - 1]
    return x[n - k:]

s = fisher_sampling_1(N, K)
print(s)


def fisher_sampling_2(n, k):
    """
    Algorithm 2 Sparse Fisher-Yates Sampler(n,k)
    
    This algorithm uses a hash table to track only the swapped elements,
    avoiding the O(n) space initialization required by Algorithm 1.
    
    Algorithm 2 Sparse Fisher-Yates Sampler(n,k)
    H ← HashTable()
    x ← array(k)
    for i = 0 → k-1 do
        r ← Uniform([n-i])
        x[i] ← H.get(r, default= r)
        H[r] ← H.get(n-i, default= n-i)
        if r = n-i then Delete H[n-i]
    end for
    return x
    
    The hash table H stores only the elements that have been affected by swaps.
    If an element hasn't been swapped, H.get(i, default=i) returns i itself.
    This reduces space complexity to O(k) instead of O(n).
    """
    H = {}
    x = np.empty(k, dtype=int)
    for i in range(k):
        r = np.random.randint(0, n - i)
        x[i] = H.get(r, r)
        H[r] = H.get(n - i - 1, n - i - 1)
        if r == n - i - 1:
            del H[n - i - 1]
    return x

s = fisher_sampling_2(N, K)
print(s)


def fisher_sampling_3(n, k):
    """
    Algorithm 3 Membership Checking Sampler(n,k)
    
    This algorithm uses a hash table to track already sampled elements,
    rejecting duplicates through membership checking.
    
    Algorithm 3 Membership Checking Sampler(n,k)
    H ← HashTable()
    x ← array(k)
    for i = 0 → k-1 do
        repeat r ← Uniform([n]) until r ∉ keys(H)
        x[i] = r
        H[r] = 1
    end for
    return x
    
    This approach repeatedly samples from [0, n) until a new (unsampled) element
    is found. The hash table H tracks which elements have been sampled.
    Expected time is O(k) when k << n, but degrades as k approaches n.
    """
    H = {}
    x = np.empty(k, dtype=int)
    for i in range(k):
        while True:
            r = np.random.randint(0, n)
            if r not in H:
                break
        x[i] = r
        H[r] = 1
    return x

s = fisher_sampling_3(N, K)
print(s)


def fisher_sampling_4(n, k):
    """
    Algorithm 4 Pre-initialized Fisher-Yates with undo(x, n, k)
    
    This variation assumes x is initially [0, 1, 2, ..., n-1] (imaginary/implicit).
    Instead of physically creating the array, we use a hash table to track only the
    swapped positions. After sampling, we undo the swaps to restore the imaginary
    array to its original state (though this restoration is purely conceptual since
    we never modified a physical array).
    
    Mathematical Foundation:
    The Fisher-Yates shuffle can be represented as a product of transpositions where
    a permutation π is uniquely represented by:
        π = (1 r₁)(2 r₂)···(n rₙ)
    where rᵢ ≤ i. The Fisher-Yates shuffle is the left action of this random permutation
    on an array, applying transpositions from right to left.
    
    The classical swapping method truncates the permutation and only applies k 
    transpositions: ((n-k+1) rₙ₋ₖ₊₁)···(n rₙ). This takes advantage of a stability
    property - the n-k leftmost transpositions leave positions n-k+1 to n untouched.
    Since classical swapping only uses these k positions at the end for the sample,
    it does not need to apply the n-k remaining transpositions.
    
    Undoing the permutation simply requires applying the transpositions in reverse order:
    (n rₙ)···((n-k+1) rₙ₋ₖ₊₁). We store the values rₙ₋ₖ₊₁, ..., rₙ in array U and
    apply them in reverse.
    
    Algorithm 4 Pre-initialized Fisher-Yates with undo(x, n, k)
    U ← array(k)
    sample ← array(k)
    for i = 1 → k do
        r ← Uniform([n-i])
        Swap(x[n-i], x[r])
        U[i-1] ← r
        sample[i-1] ← x[n-i]
    end for
    for i = k → 1 do
        r ← U[i-1]
        Swap(x[n-i], x[r])
    end for
    return sample
    
    Implementation Note:
    Since x is initially [0, 1, ..., n-1], we use a hash table H to track swaps.
    H.get(i, i) returns the current value at position i (defaults to i if never swapped).
    
    Parameters:
    - n: Size of the population
    - k: Number of samples to draw (k <= n)
    
    Returns:
    - sample: Array of k elements sampled without replacement from [0, n)
    
    Time complexity: O(k)
    Space complexity: O(k) for H, U, and sample
    """
    H = {}
    U = np.empty(k, dtype=int)
    sample = np.empty(k, dtype=int)
    
    # Forward pass: sample with Fisher-Yates and record swaps
    # Applies transpositions ((n-k+1) r_{n-k+1})···(n r_n)
    for i in range(k):
        r = np.random.randint(0, n - i)
        # Swap x[n-i-1] and x[r] conceptually via hash table
        val_at_ni = H.get(n - i - 1, n - i - 1)
        val_at_r = H.get(r, r)
        H[n - i - 1] = val_at_r
        H[r] = val_at_ni
        U[i] = r
        sample[i] = val_at_r  # x[n-i-1] after swap
    
    # Backward pass: undo all swaps to restore x (conceptually)
    # Applies inverse transpositions (n r_n)···((n-k+1) r_{n-k+1})
    for i in range(k - 1, -1, -1):
        r = U[i]
        # Undo swap: swap x[n-i-1] and x[r] again
        val_at_ni = H.get(n - i - 1, n - i - 1)
        val_at_r = H.get(r, r)
        H[n - i - 1] = val_at_r
        H[r] = val_at_ni
    
    return sample

s = fisher_sampling_4(N, K)
print(s)
