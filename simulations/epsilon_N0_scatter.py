import os
import matplotlib.pyplot as plt
import numpy as np
import re

# Define the folder where the images are stored
folder_path = './processed_plots'

# Initialize lists to hold the epsilon and N0 values
epsilon_values = []
n0_values = []

# Regular expression pattern for parsing the values from the filenames
pattern = re.compile(r'epsilon_([0-9.]+)_N0_([0-9.]+)(?=_|\.)')

# Iterate over the files in the directory
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        # Use regular expression to find matches
        match = pattern.search(filename)
        if match:
            try:
                # Extract epsilon and N0 values and convert them to float
                epsilon = float(match.group(1))
                n0 = float(match.group(2))
                epsilon_values.append(epsilon)
                n0_values.append(n0)
            except ValueError as e:
                print(f"Could not convert values in {filename}: {e}")

# Create a single scatter plot with smaller dots
plt.figure(figsize=(8, 6))
plt.scatter(n0_values, epsilon_values, s=5)  # s controls the size of the dots

# Fit and plot the regression line
m, b = np.polyfit(n0_values, epsilon_values, 1)
plt.plot(np.linspace(min(n0_values), max(n0_values), 100), m * np.linspace(min(n0_values), max(n0_values), 100) + b, color='red')

plt.xlabel('N0')
plt.ylabel('Epsilon')
plt.title('Scatter plot of N0 vs Epsilon with Regression Line')
plt.xlim(min(n0_values), max(n0_values))
plt.ylim(min(epsilon_values), max(epsilon_values))
print("Epsilon values:",epsilon_values)
print("N0 values:", n0_values)
plt.grid(True)
plt.savefig('scatter_plot.png')
plt.show()
