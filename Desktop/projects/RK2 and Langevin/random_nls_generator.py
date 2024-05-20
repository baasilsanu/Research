import numpy as np

def calculate_epsilon(n0):
    """Calculate the epsilon value for a given N0 using the specified equation."""
    return 0.0039619 * n0 ** 0.5972656

def generate_values(n0_min, n0_max, num_values):
    """Generate random N0 values within a specified range and their corresponding epsilon values."""
    n0_values = np.random.uniform(low=n0_min, high=n0_max, size=num_values)
    epsilon_values = [calculate_epsilon(n0) for n0 in n0_values]
    return n0_values, epsilon_values

# Specify the range and number of values
n0_min = 1000  # minimum value of N0
n0_max = 1500  # maximum value of N0
num_values = 30  # number of values to generate

# Generate the values
n0_array, epsilon_array = generate_values(n0_min, n0_max, num_values)

# Print the results
# print("N0 Values:", list(n0_array))
# print("Epsilon Values:", epsilon_array)

for i in range(len(n0_array)):
    print("Epsilon Value:", epsilon_array[i], "N0 Square Value:", n0_array[i])
