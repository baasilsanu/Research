import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import os
import random

@jit(nopython=True)
def simulate_numba(num_steps, k_e_square, W_e, L_e_plus, W_plus, L_plus_e, eta_batch, dt, epsilon, r_m, k_plus_square):
    phi_e = np.array([0.0, 0.0])
    phi_plus = np.array([0.0, 0.0])
    U = 0.01
    
    phi_e_history = np.zeros((num_steps, 2))
    phi_plus_history = np.zeros((num_steps, 2))
    U_history = np.zeros(num_steps)
    R_vals = np.zeros(num_steps)
    switch_times = []

    for i in range(num_steps):
        eta = eta_batch[i]
        xi = np.array([2 * np.sqrt(2) * eta[0] / np.sqrt(k_e_square), 0.0])
        phi_e_dot = np.zeros(2)
        phi_plus_dot = np.zeros(2)

        phi_e_dot[0] = W_e[0, 0] * phi_e[0] + W_e[0, 1] * phi_e[1] + U * (
            L_e_plus[0, 0] * phi_plus[0] + L_e_plus[0, 1] * phi_plus[1]) + (np.sqrt(epsilon) * xi[0]) / np.sqrt(dt)
        phi_e_dot[1] = W_e[1, 0] * phi_e[0] + W_e[1, 1] * phi_e[1] + U * (
            L_e_plus[1, 0] * phi_plus[0] + L_e_plus[1, 1] * phi_plus[1]) + (np.sqrt(epsilon) * xi[1]) / np.sqrt(dt)

        phi_plus_dot[0] = W_plus[0, 0] * phi_plus[0] + W_plus[0, 1] * phi_plus[1] + U * (
            L_plus_e[0, 0] * phi_e[0] + L_plus_e[0, 1] * phi_e[1])
        phi_plus_dot[1] = W_plus[1, 0] * phi_plus[0] + W_plus[1, 1] * phi_plus[1] + U * (
            L_plus_e[1, 0] * phi_e[0] + L_plus_e[1, 1] * phi_e[1])

        phi_e[0] += phi_e_dot[0] * dt
        phi_e[1] += phi_e_dot[1] * dt
        phi_plus[0] += phi_plus_dot[0] * dt
        phi_plus[1] += phi_plus_dot[1] * dt

        R = 0.25 * k * (k_plus_square - k_e_square) * phi_e[0] * phi_plus[0]
        U_dot = R - r_m * U
        U += U_dot * dt

        phi_e_history[i, 0] = phi_e[0]
        phi_e_history[i, 1] = phi_e[1]
        phi_plus_history[i, 0] = phi_plus[0]
        phi_plus_history[i, 1] = phi_plus[1]
        U_history[i] = U
        R_vals[i] = R

        if i > 0:
            if U_history[i - 1] > 0 and U_history[i] < 0:
                switch_times.append(i)
    return phi_e_history, phi_plus_history, U_history, R_vals, switch_times

class Simulation:
    def __init__(self, epsilon, N_0_squared, r_m, k, m, m_u, dt, total_time):
        self.epsilon = epsilon
        self.N_0_squared = N_0_squared
        self.r_m = r_m
        self.k = k
        self.m = m
        self.m_u = m_u
        self.dt = dt
        self.total_time = total_time
        self.num_steps = int(total_time / dt)
        self.k_e_square = k**2 + m**2
        self.k_plus_square = k**2 + (m + m_u)**2

        self.W_e = np.array([[-1, (k / self.k_e_square)], [-k * N_0_squared, -1]])
        self.W_plus = np.array([[-1, -k / self.k_plus_square], [k * N_0_squared, -1]])
        self.L_e_plus = np.array([[(-k / (2 * self.k_e_square)) * (self.k_plus_square - m_u**2), 0],
                                  [0, k / 2]])
        self.L_plus_e = np.array([[(-k / (2 * self.k_plus_square)) * (m_u**2 - self.k_e_square), 0],
                                  [0, -k / 2]])

        self.phi_e_history = np.zeros((self.num_steps, 2))
        self.phi_plus_history = np.zeros((self.num_steps, 2))
        self.U_history = np.zeros(self.num_steps)
        self.R_vals = np.zeros(self.num_steps)
        self.eta_batch = self.generate_eta_batch()

    def generate_eta_batch(self):
        return np.random.normal(0, 1, size=(self.num_steps, 1))

    def simulate(self):
        self.phi_e_history, self.phi_plus_history, self.U_history, self.R_vals, self.switch_times = simulate_numba(
            self.num_steps, self.k_e_square, self.W_e, self.L_e_plus, self.W_plus, self.L_plus_e, self.eta_batch, self.dt, self.epsilon, self.r_m, self.k_plus_square)
        return len(self.switch_times), [t * self.dt for t in self.switch_times]

    def extract_reversal_data(self, window_size=5000):
        reversal_data = {}
        step_units = window_size
        last_reversal_index = -step_units  

        for switch_time in self.switch_times:
            index = switch_time
            if index - last_reversal_index < step_units:
                continue  
            

            if index >= step_units and index + step_units < self.num_steps:
                pre_reversal_positive = np.all(self.U_history[index - step_units:index] > 0)
                post_reversal_negative = np.all(self.U_history[index:index + step_units] < 0)
                
                if not pre_reversal_positive or not post_reversal_negative:
                    continue 
                
                reversal_data[f"reversal_at_{switch_time}"] = {
                    'phi_e': self.phi_e_history[index - step_units:index + step_units],
                    'phi_plus': self.phi_plus_history[index - step_units:index + step_units],
                    'U': self.U_history[index - step_units:index + step_units],
                    'R': self.R_vals[index - step_units:index + step_units],
                    'eta': self.eta_batch[index - step_units:index + step_units]
                }
            last_reversal_index = index
        return reversal_data


def average_arrays(*arrays):
    if not arrays:
        raise ValueError("No arrays provided for averaging.")
    
    np_arrays = [np.array(arr) for arr in arrays]
    array_lengths = [len(arr) for arr in np_arrays]

    if len(set(array_lengths)) != 1:
        raise ValueError("All input arrays must have the same length.")
    
    average_array = np.mean(np_arrays, axis=0)
    
    return average_array

def plot_composite_analysis(plotName, epsilon, N_0_squared, simulations, window_size=5000):
    phi_e_list = []
    phi_plus_list = []
    U_list = []
    R_list = []
    eta_list = []

    for sim in simulations:
        reversal_data = sim.extract_reversal_data(window_size)
        for key in reversal_data:
            phi_e_list.append(reversal_data[key]['phi_e'])
            phi_plus_list.append(reversal_data[key]['phi_plus'])
            U_list.append(reversal_data[key]['U'])
            R_list.append(reversal_data[key]['R'])
            eta_list.append(reversal_data[key]['eta'])

    average_phi_e = average_arrays(*phi_e_list)
    average_phi_plus = average_arrays(*phi_plus_list)
    average_U = average_arrays(*U_list)
    average_R = average_arrays(*R_list)
    average_eta = average_arrays(*eta_list)



    time_array = np.arange(-window_size, window_size) * simulations[0].dt

    fig, axs = plt.subplots(4, 2, figsize=(15, 20))


    for i in range(5):
        axs[0, 0].plot(time_array, phi_e_list[i][:, 0], linewidth=0.5, linestyle='--', color='gray')
    axs[0, 0].plot(time_array, average_phi_e[:, 0], label='Average', linewidth=1.5, color='blue')
    axs[0, 0].set_title(f'Average phi_e')
    axs[0, 0].grid()
    

    for i in range(5):
        axs[0, 1].plot(time_array, phi_e_list[i][:, 1], linewidth=0.5, linestyle='--', color='gray')
    axs[0, 1].plot(time_array, average_phi_e[:, 1], label='Average', linewidth=1.5, color='blue')
    axs[0, 1].set_title(f'Average b_e')
    axs[0, 1].grid()

    for i in range(5):
        axs[1, 0].plot(time_array, phi_plus_list[i][:, 0], linewidth=0.5, linestyle='--', color='gray')
    axs[1, 0].plot(time_array, average_phi_plus[:, 0], label='Average', linewidth=1.5, color='blue')
    axs[1, 0].set_title(f'Average phi_plus')
    axs[1, 0].grid()

    for i in range(5):
        axs[1, 1].plot(time_array, phi_plus_list[i][:, 1], linewidth=0.5, linestyle='--', color='gray')
    axs[1, 1].plot(time_array, average_phi_plus[:, 1], label='Average', linewidth=1.5, color='blue')
    axs[1, 1].set_title(f'Average b_plus')
    axs[1, 1].grid()

    for i in range(5):
        axs[2, 0].plot(time_array, U_list[i], linewidth=0.5, linestyle='--', color='gray')
    axs[2, 0].plot(time_array, average_U, label='Average', linewidth=1.5, color='blue')
    axs[2, 0].set_title(f'Average U')
    axs[2, 0].grid()

    for i in range(5):
        axs[2, 1].plot(time_array, R_list[i], linewidth=0.5, linestyle='--', color='gray')
    axs[2, 1].plot(time_array, average_R, label='Average', linewidth=1.5, color='blue')
    axs[2, 1].set_title(f'Average R')
    axs[2, 1].grid()

    for i in range(5):
        axs[3, 0].plot(time_array, eta_list[i], linewidth=0.5, linestyle='--', color='gray')
    axs[3, 0].plot(time_array, average_eta, label='Average', linewidth=1.5, color='blue')
    axs[3, 0].set_title(f'Average eta')
    axs[3, 0].grid()

    plt.suptitle(f'Composite Analysis for Eps: {epsilon} and N0Sq: {N_0_squared} (Total count: {len(phi_e_list)})', fontsize=16)
    plt.subplots_adjust(wspace=0.3, hspace=0.3, top=1.85)

    plt.tight_layout(pad = 3)
    output_dir = 'composite_analysis_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f"composite_analysis_for_eps_{epsilon}_and_N_0_square_{N_0_squared}_plot_{plotName}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.clf()
    print(f"Composite analysis plot saved to {filepath}. There are {len(phi_e_list)} simulations included.")

def split_list_randomly(l):
    random.shuffle(l)
    
    midpoint = len(l) // 2
    
    first_half = l[:midpoint]
    second_half = l[midpoint:]

    return first_half, second_half

if __name__ == "__main__":
    simulations = []
    compositeHalfList = []
    epsilon = .1526418
    N_0_squared = 596.36
    r_m = 0.1
    k = 2 * np.pi * 6
    m = 2 * np.pi * 3
    m_u = 2 * np.pi * 7
    dt = 0.001
    total_time = 200

    for i in range(450):
        print(f"Running iteration {i}")
        sim = Simulation(epsilon, N_0_squared, r_m, k, m, m_u, dt, total_time)
        sim.simulate()
        simulations.append(sim)

    firstHalf, secondHalf = split_list_randomly(simulations)
    plot_composite_analysis("totalPlot" ,epsilon, N_0_squared, simulations)
    plot_composite_analysis("firstHalf" ,epsilon, N_0_squared, firstHalf)
    plot_composite_analysis("secondHalf" ,epsilon, N_0_squared, secondHalf)


