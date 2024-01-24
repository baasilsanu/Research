import numpy as np
import matplotlib.pyplot as plt
import ast
import os


# Function to calculate the CDF of the given data
def calculate_cdf(data):
    # Sort the data in ascending order
    sorted_data = np.sort(data)

    # Calculate the CDF values
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    return sorted_data, cdf


# Function to plot the CDF
def plot_cdf(data):
    sorted_data, cdf = calculate_cdf(data)

    # Plot the CDF
    plt.figure(figsize=(8, 4))
    plt.plot(sorted_data, cdf, marker='.', linestyle='none')
    plt.xlabel('Data')
    plt.ylabel('CDF')
    plt.title('Cumulative Distribution Function')
    plt.grid(True)
    plt.show()


# Function to read data from a text file
def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        data_str = file.read()
        data = ast.literal_eval(data_str)
        return data


file_name = input("Enter your file name: ")  
full_file_name = f"generalized_method_analysis/{file_name}"
current_directory = os.getcwd()
file_path = os.path.join(current_directory, full_file_name)

U_history = read_data_from_file(file_path)
plot_cdf(U_history)
