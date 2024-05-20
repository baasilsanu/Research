import numpy as np
from numba import jit
import os
import pandas as pd

#NEEDS TIME AND REVERSAL DATA STORE FIXED

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
    def __init__(self, run_id,epsilon, N_0_squared, r_m, k, m, m_u, dt, total_time): #change here
        self.run_id = run_id
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

def saveFunction(run_id, epsilon, N_0_squared, U_history, reversal_data):

    output_dir = 'bigger_composite_analysis_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    u_data = {
        'run_id':[run_id] * len(U_history),
        'epsilon': [epsilon] * len(U_history),
        'N_0_squared': [N_0_squared] * len(U_history),
        'U_values': U_history
    }

    u_df = pd.DataFrame(u_data)
    u_df.to_csv(os.path.join(output_dir, 'U_values.csv'), mode='a', index=False, header=not os.path.exists(os.path.join(output_dir, 'U_values.csv')))

    reversal_list = []
    for key, value in reversal_data.items():
        reversal_list.append({
            'run_id': run_id,
            'epsilon': epsilon,
            'N_0_squared': N_0_squared,
            'reversal_time': key,
            'phi_e': value['phi_e'].tolist(),
            'phi_plus': value['phi_plus'].tolist(),
            'U': value['U'].tolist(),
            'R': value['R'].tolist(),
            'eta': value['eta'].tolist()
        })
    reversal_df = pd.DataFrame(reversal_list)
    reversal_df.to_csv(os.path.join(output_dir, 'reversal_data.csv'), mode='a', index=False, header=not os.path.exists(os.path.join(output_dir, 'reversal_data.csv')))


    


if __name__ == "__main__":
    r_m = 0.1
    k = 2 * np.pi * 6
    m = 2 * np.pi * 3
    m_u = 2 * np.pi * 7
    dt = 0.001
    total_time = 2000

    eps_combos = [0.07894281881282045, 0.14509802140045203, 0.24142071558123343, 0.3055431793036454]
    N_0_squared_combos = [149.82931246564826, 415.1406776514857, 973.6529697935448, 1444.3802061982885]

    run_id = 1
    for i in eps_combos:
        for j in N_0_squared_combos:
            epsilon = i
            N_0_squared = j
            print(f"Running iteration {run_id} for values epsilon {epsilon} and N0Sq: {N_0_squared}")
            sim = Simulation(run_id, epsilon, N_0_squared, r_m, k, m, m_u, dt, total_time)
            sim.simulate()
            reversal_data = sim.extract_reversal_data()
            saveFunction(run_id, epsilon, N_0_squared, sim.U_history, reversal_data)
            run_id += 1



