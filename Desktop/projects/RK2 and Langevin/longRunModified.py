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
    phi_plus = np.array([0.0, 0.0])  # Initial condition for phi_plus
    U = 0.01  # Initial condition for U

    for i in range(num_steps):
        eta = eta_batch[i]
        # Update equations
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

        # Record history in pre-allocated arrays
        phi_e_history[i, 0] = phi_e[0]
        phi_e_history[i, 1] = phi_e[1]
        phi_plus_history[i, 0] = phi_plus[0]
        phi_plus_history[i, 1] = phi_plus[1]
        U_history[i] = U
        R_vals[i] = R

    return phi_e_history, phi_plus_history, U_history, R_vals

def plot_results(time_array, U_history, R_vals, epsilon, N_0_squared):
    average_U_hist = np.full(len(time_array), np.mean(U_history))
    average_R_hist = np.full(len(time_array), np.mean(R_vals))
    fig, axs = plt.subplots(1, 2, figsize = (18, 8))

    axs[0].plot(time_array, U_history)
    axs[0].plot(time_array, average_U_hist, linestyle = ":", color = 'red')
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
    output_dir = 'processed_plots/new_scatter_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f"plot_epsilon_{epsilon}_N0_{N_0_squared}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"Saved plot to {filepath}")
    plt.clf()

#it needs to know the previous state to see if a state switch happened
def countReversal(U_history, time_values, step_size):
    #instead of counting it in terms of duration, can I count it in terms of step sizes???????????
    reversal_details = []
    initial_state_changes = []
    cycles = [0]
    initial_state = is_greater_than_zero(U_history[0])
    duration = 0
    okayCondition = False
    for i in range(0, len(U_history)):
        current_state = is_greater_than_zero(U_history[i])
        time = float(time_values[i])
        duration += step_size
        if(time_values[i] % 10 == 0):
            print(time_values[i])
            print("Initial State:", initial_state)
            print("Current State:", current_state)
            print("Duration:", duration)
            print()

        # if duration >= 10:
        #     okayCondition = False

        if current_state == initial_state and okayCondition:
            okayCondition = False
            cycles.pop()
        

        if current_state != initial_state and duration >= 10:
            if len(cycles) == 1:
                cycles.append(time) 
                reversal_details.append((reverseState(current_state), cycles[-1] - cycles[-2], cycles[-1]))
                initial_state = current_state
                duration = 0
                okayCondition = True
            else:
                cycles.append(time) 
                reversal_details.append((reverseState(current_state), cycles[-1] - cycles[-2], cycles[-1] - 10))
                initial_state = current_state
                duration = 0
                okayCondition = True


    return reversal_details

def newCountReversal(U_History, time_values):
    returnList = []
    for i in range(1, len(U_History)):
        curr_state = is_greater_than_zero(U_History[i])
        prev_state = is_greater_than_zero(U_History[i - 1])
        if curr_state != prev_state:
            returnList.append(time_values[i])
    return returnList

def trimReversals(reversal_list):
    done = False
    i = 0
    j = 1
    while not done:
        if (reversal_list[j] - reversal_list[i]) < 10:
            reversal_list[i] = reversal_list[i] + (reversal_list[j] - reversal_list[i])
            reversal_list.pop(j)
        else:
            i += 1
            j += 1
        if j > (len(reversal_list) - 1):
            done = True
    return reversal_list


# def record_significant_durations(u_history, step_size):
#     significant_durations = []
#     positive = u_history[0] > 0
#     start_time = 0
#     current_duration = 0

#     for i in range(1, len(u_history)):
#         # If the sign is the same, increase the duration
#         if (u_history[i] > 0) == positive:
#             current_duration += step_size
#         else:
#             # If the sign changed, check if the current duration is significant
#             if current_duration >= 10:
#                 # Record the start time and duration
#                 significant_durations.append((start_time * step_size, current_duration))
#             # Reset the start time and duration
#             start_time = i
#             positive = not positive
#             current_duration = step_size

#     # Check for the last segment
#     if current_duration >= 10:
#         significant_durations.append((start_time * step_size, current_duration))

#     return significant_durations





def is_greater_than_zero(num):
    if num > 0:
        return 1
    else:
        return 0
def reverseState(num):
    if num == 1:
        return 0
    else:
        return 1
@jit(nopython = True)
def calculate_sma(data, period):
    sma = np.zeros(len(data))
    for i in range(len(data)):
        if i < period:
            sma[i] = np.mean(data[:i+1])
        else:
            sma[i] = np.mean(data[i-period:i])
    return sma


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

    # output_dir = f'generalized_method_analysis/bimodality_test_data/epsilon_{args.epsilon}_N0_{args.N_0_squared}'
    #
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # filename = f"U_History_plot_epsilon_{args.epsilon}_N0_{args.N_0_squared}.txt"
    # filepath = os.path.join(output_dir, filename)
    # with open(filepath, 'w') as file:
    #     file.write(str(list(U_history)))
    # print("Wrote U_History")
    #
    # filename = f"R_Vals_plot_epsilon_{args.epsilon}_N0_{args.N_0_squared}.txt"
    # filepath = os.path.join(output_dir, filename)
    # with open(filepath, 'w') as file:
    #     file.write(str(list(R_vals)))
    # print("Wrote R_vals")

    time_array = np.arange(0, parameters[7], parameters[6])
    # reversal_details = countReversal(U_history, time_array, .001)
    # reversal_details = record_significant_durations(U_history, dt)
    newCountReversal_arr = newCountReversal(U_history, time_array)
    newCountReversalTest = newCountReversal([-1, 0, 3, -3, 2], [0, 1, 2, 3, 4])
    reversal_details = newCountReversal(calculate_sma(U_history, 10000), time_array)
    duration_before_reversals = []
    for i in range(len(reversal_details)):
        if reversal_details[i] > 5:
            reversal_details[i] = reversal_details[i] - 5
    if len(reversal_details) != 0:
        duration_before_reversals.append(reversal_details[0])
        for i in range(len(reversal_details)):
            if i < len(reversal_details) - 2:
                duration_before_reversals.append(reversal_details[i + 1] - reversal_details[i])
            else:
                duration_before_reversals.append(total_time - reversal_details[i])
        

    
    switch_count = len(reversal_details)
    print("Elapsed Time: ", elapsed_time)
    if switch_count > 0:
        print()
    print("Reversal Count:", switch_count)
    print()
    print("Reversal times:")
    for i in reversal_details:
        print(i)
    print()
    print("Durations:")
    for i in duration_before_reversals:
        print(i)
    print()
    if len(duration_before_reversals) != 0:
        print("Average Duration:", np.mean(duration_before_reversals))
    
    # print("Trim Reversal Test: ", trimReversals([1, 2, 3, 4, 20, 23, 26]))
    plot_results(time_array, U_history, R_vals,args.epsilon, args.N_0_squared)
