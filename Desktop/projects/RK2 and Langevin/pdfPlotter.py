import numpy as np
import matplotlib.pyplot as plt
import ast
import os
import re
import diptest
from scipy.stats import skew, kurtosis
import numpy as np


# Function to read data from a text file
def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        data_str = file.read()
        data = ast.literal_eval(data_str)
        return data


# Function to plot and save the PDF using a histogram
def plot_and_save_pdf(data, file_name, save_directory, bins=60):
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=bins, density=True)
    plt.xlabel('Data')
    plt.ylabel('Density')
    plt.title('Probability Density Function (PDF)')
    plt.grid(True)

    # Extract epsilon and n0 values from the file name
    match = re.search(r'^U_History_plot_epsilon_([0-9.]+)_N0_([0-9.]+)\.txt$', file_name)
    if match:
        epsilon, n0 = match.groups()
        plot_file_name = f"plot_epsilon_{epsilon}_N0_{n0}.png"
    else:
        return  

    # Save the plot
    plot_path = os.path.join(save_directory, plot_file_name)
    plt.savefig(plot_path)
    # print(f"Saved to {save_directory} as {plot_file_name}")
    plt.close()

def calculate_bimodality_coefficient(data):
    skewness = skew(data)
    kurt = kurtosis(data, fisher = True)
    n = len(data)
    BC = (skewness**2 + 1) / (kurt + 3*((n - 1)**2)/((n - 2) * (n - 3)))
    return BC


if __name__ == "__main__":
    for i in (0.01, 0.01098541, 0.01206793, 0.01325711, 0.01456348, 0.01599859,
0.01757511, 0.01930698, 0.02120951, 0.02329952, 0.02559548, 0.02811769, 0.03088844,
0.03393222, 0.03727594, 0.04094915, 0.04498433, 0.04941713, 0.05428675, 0.05963623,
0.06551286, 0.07196857, 0.07906043, 0.08685114, 0.09540955, 0.10481131, 0.11513954,
0.12648552, 0.13894955, 0.1526418, 0.16768329, 0.184207, 0.20235896, 0.22229965,
0.24420531, 0.26826958, 0.29470517, 0.32374575, 0.35564803, 0.39069399, 0.42919343,
0.47148664, 0.51794747, 0.5689866, 0.62505519, 0.68664885, 0.75431201, 0.82864277,
0.91029818, 1.00):
        for j in (100.0, 1000.0):
            file_path = f'generalized_method_analysis/bimodality_test_data/epsilon_{i}_N0_{j}/U_History_plot_epsilon_{i}_N0_{j}.txt'
            current_directory = os.getcwd()
            full_file_path = os.path.join(current_directory, file_path)
            data = read_data_from_file(full_file_path)
            data = np.array(data)
            bimodality_coefficient = calculate_bimodality_coefficient(data)
            print(f"For epsilon {i}, N_0_Squared {j}")
            if bimodality_coefficient > .555:
                print("The Distribution is bimodal with coeff:", bimodality_coefficient)
            else:
                print("The Distribution is not bimodal with coeff:", bimodality_coefficient)
            dip = diptest.dipstat(data)
            print(f"Only the dip statistic is:", dip)
            dip, pval = diptest.diptest(data)
            print(f"The Dip is {dip}, the P-value is {pval}")
            num_positive = (data > 0).sum()
            num_negative = (data < 0).sum()
            print(f"The number of positive values are {num_positive} and the number of negative values are {num_negative}")
            print()
            save_directory = os.path.join(current_directory, "generalized_method_analysis/bimodality_plots")
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            plot_and_save_pdf(data, f"U_History_plot_epsilon_{i}_N0_{j}.txt", save_directory)


# how about checking for the number of positive values and negative values??





