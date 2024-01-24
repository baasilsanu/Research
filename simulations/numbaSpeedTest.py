import numpy as np
from numba import jit
import time

# Define the function with Numba's JIT decorator
@jit(nopython=True)
def sum_of_squares(arr):
    total = 0
    for i in arr:
        total += i ** 2
    return total

# Generate a large array of random numbers
data = np.random.rand(10000000)

# Time the execution
start_time = time.time()
result = sum_of_squares(data)
end_time = time.time()

print("Result:", result)
print("Time taken with Numba:", end_time - start_time, "seconds")
