import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from numba import jit
import argparse
import os
def initialize_matrices(N_0_squared, k, m, m_u):
    k_e_square = k**2 + m**2
    k_plus_square = k**2 + (m + m_u)**2
    W_e = np.array([[-1, (k / k_e_square)], [-k * N_0_squared, -1]])
    W_plus = np.array([[-1, -k / k_plus_square], [k * N_0_squared, -1]])
    L_e_plus = np.array([[(-k / (2 * k_e_square)) * (k_plus_square - m_u**2), 0],
                         [0, k / 2]])
    L_plus_e = np.array([[(-k / (2 * k_plus_square)) * (m_u**2 - k_e_square), 0],
                         [0, -k / 2]])
    return W_e, W_plus, L_e_plus, L_plus_e, k_e_square, k_plus_square

def generate_eta_batch(batch_size):
    return np.random.normal(0, 1, size=(batch_size, 1))

@jit(nopython=True)
def simulate(num_steps, dt, epsilon, r_m, W_e,k, k_e_square, k_plus_square, W_plus, L_e_plus, L_plus_e, eta_batch):
    phi_e_history = np.zeros((num_steps, 2))
    phi_plus_history = np.zeros((num_steps, 2))
    U_history = np.zeros(num_steps)
    R_vals = np.zeros(num_steps)

    phi_e = np.array([0.0, 0.0])  # Initial condition for phi_e
    phi_plus = np.array([0.0, 0.0])  
    U = 0.01  # Initial condition for U

    for i in range(num_steps):
        eta = eta_batch[i]
        xi = np.array([2 * np.sqrt(2) * eta[0] / np.sqrt(k_e_square), 0.0])
        phi_e_dot = np.zeros(2)
        phi_plus_dot = np.zeros(2)

        phi_e_dot[0] = W_e[0, 0] * phi_e[0] + W_e[0, 1] * phi_e[1] + U * (
                    L_e_plus[0, 0] * phi_plus[0] + L_e_plus[0, 1] * phi_plus[1]) + (np.sqrt(epsilon) * xi[0]) / np.sqrt(
            dt)
        phi_e_dot[1] = W_e[1, 0] * phi_e[0] + W_e[1, 1] * phi_e[1] + U * (
                    L_e_plus[1, 0] * phi_plus[0] + L_e_plus[1, 1] * phi_plus[1]) + (np.sqrt(epsilon) * xi[1]) / np.sqrt(
            dt)

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

    return phi_e_history, phi_plus_history, U_history, R_vals

@jit(nopython=True)
def calculate_sma(data, period):
    sma = np.zeros(len(data))
    for i in range(len(data)):
        if i < period:
            sma[i] = np.mean(data[:i+1])
        else:
            sma[i] = np.mean(data[i-period:i])
    return sma

@jit(nopython=True)
def calculate_ema(data, alpha):
    ema = np.zeros(len(data))
    ema[0] = data[0]  
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema

def plot_results(time_array, U_history, R_vals, epsilon, N_0_squared):
    average_U_hist = np.full(len(time_array), np.mean(U_history))
    average_R_hist = np.full(len(time_array), np.mean(R_vals))
    N = 20000  # Number of periods for SMA
    alpha = 2 / (N + 1)  # Smoothing factor for EMA

    sma_U_history = calculate_sma(U_history, N)
    ema_U_history = calculate_ema(U_history, alpha)


    fig, axs = plt.subplots(1, 2, figsize = (18, 8))

    axs[0].plot(time_array, U_history, color = 'blue')
    axs[0].plot(time_array, average_U_hist, linestyle = ":", color = 'red')
    axs[0].plot(time_array, sma_U_history, color='orange')
    axs[0].plot(time_array, ema_U_history, color='black')
    axs[0].grid()
    axs[0].set_title('U vs Time')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('U')


    axs[1].plot(time_array, R_vals)
    axs[1].plot(time_array, average_R_hist, linestyle = ':', color = 'red')
    axs[1].grid()
    axs[1].set_title('R vs Time')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('R')

    plt.tight_layout()
    output_dir = 'nls_model_generated_images_moving_averages'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f"plot_epsilon_{epsilon}_N0_{N_0_squared}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"Saved plot to {filepath}")
    plt.clf()

def run_simulation(params):
    epsilon, N_0_squared, r_m, k, m, m_u, dt, total_time = params
    num_steps = int(total_time / dt)

    W_e, W_plus, L_e_plus, L_plus_e, k_e_square, k_plus_square = initialize_matrices(N_0_squared, k, m, m_u)
    eta_batch = generate_eta_batch(num_steps)
    phi_e_history, phi_plus_history, U_history, R_vals = simulate(num_steps, dt, epsilon, r_m, W_e, k, k_e_square, k_plus_square, W_plus, L_e_plus, L_plus_e, eta_batch)

    return U_history, R_vals

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run simulations with varying epsilon and N_0_squared.')
    parser.add_argument('--epsilon', type=float, required=True, help='Epsilon value')
    parser.add_argument('--N_0_squared', type=float, required=True, help='N_0_squared value')

    args = parser.parse_args()

    # Fixed parameters
    r_m = 0.1
    k = 2 * np.pi * 6
    m = 2 * np.pi * 3
    m_u = 2 * np.pi * 7
    dt = 0.001
    total_time = 200

    # Prepare parameters for the simulation
    parameters = (args.epsilon, args.N_0_squared, r_m, k, m, m_u, dt, total_time)
    start_time = time.time()


    U_history, R_vals = run_simulation(parameters)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed Time: ", elapsed_time)
    print("U history: ", U_history)
    print("R Vals: ", R_vals)

    time_array = np.arange(0, parameters[7], parameters[6])
    plot_results(time_array, U_history, R_vals,args.epsilon, args.N_0_squared)
