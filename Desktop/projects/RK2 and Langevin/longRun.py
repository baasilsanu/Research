#develop logic to detectect simulation crossovers.
#to do that, when U becomes -ve, add 1 to central counter
#I can just loop through the array right?
#yeah I can, thats so convenient.
#you wouldnt be able to get the time like that right now.
#because it only scans U_history
#i think you should run it for 100 times before classifying it

#how can I measure the time?
#time from one direction change to the next direction change
#should I distinguish between simulations that cause direction change?
#yes, lets do average direction change.
#so if I recognize the step number, it should help me with the direction change right?
#so direction change happenens. get step number.
#should i be counting it once the direction changes only?

#so, calcualte the negative and postivive values
#create new table
#add those values to the table.
#you can create a new column in the dataset in pandas as the test

###WRONG FIX DURATION
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from numba import jit
import argparse
import os
import psycopg2
from psycopg2 import OperationalError
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
    #add the reversal counter here, how about that.
    #so if u history changes from +ve to -ve, then +1, and then add the time it happened to another new list.
    #what about the index problem?What index problem? referring to i - 1??
    switch_count = 0
    switch_times = []

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

        if i > 0:
            if (U_history[i - 1] > 0 and U_history[i] < 0) or (U_history[i - 1] < 0 and U_history[i] >0):
                switch_count += 1
                switch_times.append(i)





    return phi_e_history, phi_plus_history, U_history, R_vals, switch_count, switch_times

    

def plot_results(time_array, U_history, R_vals, epsilon, N_0_squared, iterationCount):
    pass
    # average_U_hist = np.full(len(time_array), np.mean(U_history))
    # average_R_hist = np.full(len(time_array), np.mean(R_vals))
    # fig, axs = plt.subplots(1, 2, figsize = (18, 8))
    
    # axs[0].plot(time_array, U_history)
    # axs[0].plot(time_array, average_U_hist, linestyle = ":", color = 'red')
    # axs[0].grid()
    # axs[0].set_title('U vs Time')
    # axs[0].set_xlabel('Time')
    # axs[0].set_ylabel('U')
    
    
    # axs[1].plot(time_array, R_vals)
    # axs[1].plot(time_array, average_R_hist, linestyle = ':', color = 'red')
    # axs[1].grid()
    # axs[1].set_title('R vs Time')
    # axs[1].set_xlabel('Time')
    # axs[1].set_ylabel('R')
    
    # plt.tight_layout()
    # output_dir = 'processed_plots/new_scatter_data'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # filename = f"plot_epsilon_{epsilon}_N0_{N_0_squared}_IterationCount_{iterationCount}.png"
    # filepath = os.path.join(output_dir, filename)
    # plt.savefig(filepath)
    # print()
    # # print(f"Saved plot to {filepath}")
    # plt.clf()
    # print()

def insert_into_simulationsiter1(db_params, epsilon, n_zero_square, reversal_count, average_reversal_time):
    try:
        conn = psycopg2.connect(**db_params)
        print("Connected to database")
        cur = conn.cursor()
        query = '''
        INSERT INTO simulationsiter1 (epsilon, N_Zero_Square, reversal_count, average_reversal_time)
        VALUES (%s, %s, %s, %s);
        '''
        cur.execute(query, (epsilon, n_zero_square, reversal_count, average_reversal_time))
        conn.commit()

        cur.close()
        conn.close()
        print("Inserted")
    except OperationalError as e:
        print(f"An error occured: {e}")

    except Exception as e:
        print(f"Exception occured: {e}")

db_connection_params = {
    'dbname': 'simulationdatainit',
    'user': 'postgres',
    'password': 'Simulation2024',
    'host': 'localhost'
}

def insert_into_simulationsiter2(db_params, epsilon, n_zero_square, reversal_count, average_reversal_time, reversal_times, reversal_durations, positive_vals, negative_vals):
    try:
        conn = psycopg2.connect(**db_params)
        print("Connected to database")
        cur = conn.cursor()
        query = '''
        INSERT INTO simulationsiter2 (epsilon, N_Zero_Square, reversal_count, average_reversal_time, reversal_times, reversal_durations, positive_values, negative_values)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        '''
        cur.execute(query, (epsilon, n_zero_square, reversal_count, average_reversal_time, reversal_times, reversal_durations, positive_vals, negative_vals))
        conn.commit()

        cur.close()
        conn.close()
        print("Inserted")
    except OperationalError as e:
        print(f"An error occured: {e}")

    except Exception as e:
        print(f"Exception occured: {e}")

db_connection_params = {
    'dbname': 'simulationdatainit',
    'user': 'postgres',
    'password': 'Simulation2024',
    'host': 'localhost'
}

def insert_into_simulationsagg1(db_params, epsilon, n_zero_square, reversal_count, average_reversal_time, reversal_times, reversal_durations, positive_vals, negative_vals):
    try:
        conn = psycopg2.connect(**db_params)
        print("Connected to database")
        cur = conn.cursor()
        query = '''
        INSERT INTO simulationsagg1 (epsilon, N_Zero_Square, reversal_count, average_reversal_time, reversal_times, reversal_durations, positive_values, negative_values)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        '''
        cur.execute(query, (epsilon, n_zero_square, reversal_count, average_reversal_time, reversal_times, reversal_durations, positive_vals, negative_vals))
        conn.commit()

        cur.close()
        conn.close()
        print("Inserted")
    except OperationalError as e:
        print(f"An error occured: {e}")

    except Exception as e:
        print(f"Exception occured: {e}")

db_connection_params = {
    'dbname': 'simulationdatainit',
    'user': 'postgres',
    'password': 'Simulation2024',
    'host': 'localhost'
}

def insert_into_simulationsagg2(db_params, epsilon, n_zero_square, reversal_count, average_reversal_time, reversal_times, reversal_durations, positive_vals, negative_vals):
    try:
        conn = psycopg2.connect(**db_params)
        print("Connected to database")
        cur = conn.cursor()
        query = '''
        INSERT INTO simulationsagg2 (epsilon, N_Zero_Square, reversal_count, average_reversal_time, reversal_times, reversal_durations, positive_values, negative_values)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        '''
        cur.execute(query, (epsilon, n_zero_square, reversal_count, average_reversal_time, reversal_times, reversal_durations, positive_vals, negative_vals))
        conn.commit()

        cur.close()
        conn.close()
        print("Inserted")
    except OperationalError as e:
        print(f"An error occured: {e}")

    except Exception as e:
        print(f"Exception occured: {e}")

db_connection_params = {
    'dbname': 'simulationdatainit',
    'user': 'postgres',
    'password': 'Simulation2024',
    'host': 'localhost'
}

def insert_into_simulationsagg3(db_params, epsilon, n_zero_square, reversal_count, average_reversal_time, reversal_times, reversal_durations, positive_vals, negative_vals):
    try:
        conn = psycopg2.connect(**db_params)
        print("Connected to database")
        cur = conn.cursor()
        query = '''
        INSERT INTO simulationsagg3 (epsilon, N_Zero_Square, reversal_count, average_reversal_time, reversal_times, reversal_durations, positive_values, negative_values)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        '''
        cur.execute(query, (epsilon, n_zero_square, reversal_count, average_reversal_time, reversal_times, reversal_durations, positive_vals, negative_vals))
        conn.commit()

        cur.close()
        conn.close()
        print("Inserted")
    except OperationalError as e:
        print(f"An error occured: {e}")

    except Exception as e:
        print(f"Exception occured: {e}")

db_connection_params = {
    'dbname': 'simulationdatainit',
    'user': 'postgres',
    'password': 'Simulation2024',
    'host': 'localhost'
}

def insert_into_simulationsagg4(db_params, epsilon, n_zero_square, reversal_count, average_reversal_time, reversal_times, reversal_durations, positive_vals, negative_vals):
    try:
        conn = psycopg2.connect(**db_params)
        print("Connected to database")
        cur = conn.cursor()
        query = '''
        INSERT INTO simulationsagg4 (epsilon, N_Zero_Square, reversal_count, average_reversal_time, reversal_times, reversal_durations, positive_values, negative_values)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        '''
        cur.execute(query, (epsilon, n_zero_square, reversal_count, average_reversal_time, reversal_times, reversal_durations, positive_vals, negative_vals))
        conn.commit()

        cur.close()
        conn.close()
        print("Inserted")
    except OperationalError as e:
        print(f"An error occured: {e}")

    except Exception as e:
        print(f"Exception occured: {e}")

db_connection_params = {
    'dbname': 'simulationdatainit',
    'user': 'postgres',
    'password': 'Simulation2024',
    'host': 'localhost'
}

def insert_into_simulationdata1(db_params, epsilon, n_zero_square, reversal_count, average_reversal_time, positive_vals, negative_vals):
    try:
        conn = psycopg2.connect(**db_params)
        print("Connected to database")
        cur = conn.cursor()
        query = '''
        INSERT INTO simulationdata1 (epsilon, N_Zero_Square, reversal_count, average_reversal_time, positive_vals, negative_vals)
        VALUES (%s, %s, %s, %s, %s, %s);
        '''
        cur.execute(query, (epsilon, n_zero_square, reversal_count, average_reversal_time, positive_vals, negative_vals))
        conn.commit()

        cur.close()
        conn.close()
        print("Inserted")
    except OperationalError as e:
        print(f"An error occured: {e}")

    except Exception as e:
        print(f"Exception occured: {e}")


def run_simulation(params):
    epsilon, N_0_squared, r_m, k, m, m_u, dt, total_time = params
    num_steps = int(total_time / dt)

    W_e, W_plus, L_e_plus, L_plus_e, k_e_square, k_plus_square = initialize_matrices(N_0_squared, k, m, m_u)
    eta_batch = generate_eta_batch(num_steps)
    phi_e_history, phi_plus_history, U_history, R_vals, switch_count, switch_times = simulate(num_steps, dt, epsilon, r_m, W_e, k, k_e_square, k_plus_square, W_plus, L_e_plus, L_plus_e, eta_batch)

    return U_history, R_vals, switch_count, switch_times

@jit(nopython = True)
def calculate_sma(data, period):
    sma = np.zeros(len(data))
    for i in range(len(data)):
        if i < period:
            sma[i] = np.mean(data[:i+1])
        else:
            sma[i] = np.mean(data[i-period:i])
    return sma

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

def newCountReversal(U_History, time_values):
    returnList = []
    for i in range(1, len(U_History)):
        curr_state = is_greater_than_zero(U_History[i])
        prev_state = is_greater_than_zero(U_History[i - 1])
        if curr_state != prev_state:
            returnList.append(time_values[i])
    return returnList

def getTimesAndDurations(U_History, time_values, total_time):
    smaArray = calculate_sma(U_History, 10000)
    reversal_array = newCountReversal(smaArray, time_values)
    duration_before_reversals = []

    for i in range(len(reversal_array)):
        if reversal_array[i] > 5:
            reversal_array[i] = reversal_array[i] - 5

    if len(reversal_array) != 0:
        duration_before_reversals.append(reversal_array[0])  
        for i in range(1, len(reversal_array)):
            duration_before_reversals.append(reversal_array[i] - reversal_array[i - 1]) 

    return reversal_array, duration_before_reversals

def getTimesAndDurationsCheck(U_History, time_values):
    smaArray = calculate_sma(U_History, 1)
    reversal_array = newCountReversal(smaArray, time_array)
    duration_before_reversals = []
    for i in range(len(reversal_array)):
        if reversal_array[i] > 5:
            reversal_array[i] = reversal_array[i] - 5
    if len(reversal_array) != 0:
        duration_before_reversals.append(reversal_array[0])
        for i in range(len(reversal_array)):
            if i < len(reversal_array) - 2:
                duration_before_reversals.append(reversal_array[i + 1] - reversal_array[i])
            else:
                duration_before_reversals.append(total_time - reversal_array[i])  #this line needs to be fixed
    return reversal_array, duration_before_reversals



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run simulations with varying epsilon and N_0_squared.')
    parser.add_argument('--epsilon', type=float, required=True, help='Epsilon value')
    parser.add_argument('--N_0_squared', type=float, required=True, help='N_0_squared value')
    parser.add_argument('--Count', type = int, required=True, help='The iteration number')

    args = parser.parse_args()

    # Fixed parameters
    r_m = 0.1
    k = 2 * np.pi * 6
    m = 2 * np.pi * 3
    m_u = 2 * np.pi * 7
    dt = 0.001
    total_time = 2000

    # Prepare parameters for the simulation
    parameters = (args.epsilon, args.N_0_squared, r_m, k, m, m_u, dt, total_time)
    start_time = time.time()


    U_history, R_vals, switch_count, switch_times = run_simulation(parameters)


    end_time = time.time()
    elapsed_time = end_time - start_time

    switch_times = [i * dt for i in switch_times]
    switch_times.append(total_time)
    average_switch_times = []
    for i in range(len(switch_times) - 2):
        average_switch_times.append(switch_times[i + 1] - switch_times[i])
    average_switch_times.insert(0, switch_times[0])
    average_switch_times = np.array(average_switch_times)
    average_time = np.mean(average_switch_times)
    positive_vals = (U_history > 0).sum()
    negative_vals = (U_history < 0).sum()
    positive_vals = int(positive_vals)
    negative_vals = int(negative_vals)


    #hold on, so if a list has len 4, for i in range goes 0 1 2 3. so the function i + 1 will try

    # output_dir = 'generalized_method_analysis'
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
    reversal_array, duration_before_reversals = getTimesAndDurations(U_history, time_array)
    ##fix this
    insert_into_simulationsagg4(db_connection_params, parameters[0], parameters[1], len(reversal_array), np.mean(duration_before_reversals), reversal_array, duration_before_reversals, positive_vals, negative_vals)
    switch_count = len(reversal_array)
    print("Elapsed Time:", elapsed_time)
    if switch_count > 0:
        print()
    print("Reversal Count:", switch_count)
    print("Positive values:", positive_vals)
    print("Negative values:", negative_vals)
    if switch_count > 0:
        # print("Times of reversal:")
        # for i in reversal_array:
        #     print(i, end = " ")
        # print()
        # print("Duration before reversal:")
        # for i in duration_before_reversals:
        #     print(i, end = " ")
        # print()
        print("Average reversal time:", np.mean(duration_before_reversals))



    
    plot_results(time_array, U_history, R_vals,args.epsilon, args.N_0_squared, args.Count)
