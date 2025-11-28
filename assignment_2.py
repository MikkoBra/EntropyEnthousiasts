import csv
import numpy as np
import simpy
import logging
import matplotlib.pyplot as plt
import os
from math import erf
import random


## Random seeds
GLOBAL_SEED = 23
np.random.seed(GLOBAL_SEED)
GLOBAL_RNG = np.random.default_rng(GLOBAL_SEED)
random.seed(GLOBAL_SEED) 

## Logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger('AirportSim')

## Columns to extract from airport.csv
COLUMNS = np.array(["Year", "Month", "Europe Passengers", "Intercontinental Passengers", "Total Passengers"])


############## CSV EXTRACTION ##############

def extract_columns(columns=COLUMNS, csv_path="airport.csv", delimiter=","):
    """
    Extracts column data from a csv file into numpy arrays.

    Parameters
    ----------
    columns: Array of column names to extract
    csv_path: String representing the filepath to the csv file
    delimiter: String delimiter in the csv file
    
    Returns
    ----------
    Dictionary containing column names as keys, numpy data arrays as values.
    """
    # Extract data to dictionary
    extracted = {col: [] for col in columns}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader)
        col_indices = []
        for col in columns:
            if col not in header:
                raise ValueError(f"Column '{col}' not found in CSV.")
            col_indices.append(header.index(col))
        for row in reader:
            for name, index in zip(COLUMNS, col_indices):
                extracted[name].append(row[index])
    
    # Convert value arrays to numpy arrays
    for name in extracted:
        if name == "Month":
            extracted[name] = np.array(extracted[name], dtype=str)
        else:
            extracted[name] = np.array(extracted[name], dtype=float)
    
    return extracted


def filter_by_year(data, threshold=2019):
    """
    Filters a data dictionary by the values of the "Year" column.

    Parameters
    -----------
    data: Dictionary with column names as keys, data arrays as values
    threshold: Int of last year to include in kept data, later years
        are excluded
    
    Returns
    -----------
    Filtered data dictionary with only entries where Year <= threshold
    """
    years = data["Year"]
    kept_data_indices = [
        i for i, year in enumerate(years)
        if int(year) <= threshold
    ]
    filtered = {}
    for col, values in data.items():
        filtered[col] = [values[i] for i in kept_data_indices]
    return filtered


def filter_by_month(data, filter_month="September"):
    """
    Filters a data dictionary by the values of the "Month" column.

    Parameters
    -----------
    data: Dictionary with column names as keys, data arrays as values
    filter_month: String of month to include in kept data, other months
        are excluded
    
    Returns
    -----------
    Filtered data dictionary with only entries where Month = filter_month
    """
    months = data["Month"]
    kept_data_indices = [
        i for i, month in enumerate(months)
        if month == filter_month
    ]
    filtered = {}
    for col, values in data.items():
        filtered[col] = [values[i] for i in kept_data_indices]
    return filtered
    

######################## PART 1 ########################


## Calculate Arrival Rate (λ) in passengers per minute

def per_minute_arrival_rate(data):
    """
    Calculate the average arrival rate per minute (lambda) of passengers.

    Parameters
    -----------
    data: Dictionary with column names as keys, data arrays as values.
        Assumes column "Total Passengers" exists, and entries are per month.
    
    Returns
    -----------
    Arrival rate in passengers per minute
    """
    avg_monthly = np.mean(data["Total Passengers"])/50
    avg_daily = avg_monthly / 30
    avg_hourly = avg_daily / 16
    return avg_hourly / 60


data = extract_columns()
data = filter_by_year(data)
data = filter_by_month(data)

# lambda (rate per minute)
arrival_rate = per_minute_arrival_rate(data)
print("\n------------------- PART 1 -------------------\n")
print(f"Arrival rate per minute calculated from airport.csv: {arrival_rate:.2f}")


######################## PART 2 ########################


############## SIMULATOR ##############

class Simulator_Parameters:
    """
    Class to store parameters for the simulator.
    """
    def __init__(self, arrivals_lambda, expected_service_time, service_time_sigma, n_passengers, n_servers=1, halt_steady_state=False, warmup_passengers=1000, log_info=True):
        """
        Initializes the simulator with the arrival rate (number of passengers per minue), expected service time, and service time variance.
        """
        self.arrivals_lambda = arrivals_lambda
        self.expected_service_time = expected_service_time
        self.service_time_sigma = service_time_sigma
        self.n_passengers = n_passengers
        self.n_servers = n_servers
        self.halt_steady_state = halt_steady_state
        self.warmup_passengers = warmup_passengers
        self.log_info = log_info

class Passenger:
    """
    Passenger class for airport arrivals.
    """

    def __init__(self, env, passenger_id, server_time, servers, log_info=True):
        """
        Initializes the passenger with a unique ID.
        """
        self.passenger_id = passenger_id
        self.server_time = server_time
        self.servers = servers
        self.env = env
        self.in_queue = False
        self.completed = False
        self.log_info = log_info
    
    def run(self):
        """
        Defines the run behavior of the Passenger: enter queue, exit queue to be served, leave service.
        """
        start_time = self.env.now
        if self.log_info:
            log.debug(f"[{self.env.now:.1f}]: Passenger {self.passenger_id} arrives at airport and enteres queue")
        self.in_queue = True
        with self.servers.request() as req:
            yield req
            if self.log_info:
                log.debug(f"[{self.env.now:.1f}]: Passenger {self.passenger_id} exits queue and enters server")
            self.in_queue = False
            self.queue_time = self.env.now - start_time
            yield self.env.timeout(self.server_time)
            if self.log_info:
                log.debug(f"[{self.env.now:.1f}]: Passenger {self.passenger_id} leaves security")
            self.total_time = self.env.now - start_time
            self.completed = True

class AirportSimulator:
    """
    Simulator class for airport arrivals.
    """

    def __init__(self, params):
        """
        Initializes the simulator with the arrival rate (number of passengers per minue),
        expected service time, and service time variance.
        """
        self.expected_service_time = params.expected_service_time
        self.service_time_sigma = params.service_time_sigma
        self.arrival_rate = params.arrivals_lambda
        self.n_passengers = params.n_passengers
        self.n_servers = params.n_servers
        self.halt_steady_state = params.halt_steady_state
        self.env = simpy.Environment()
        self.servers = simpy.Resource(self.env, capacity=self.n_servers)
        self.warmup_passengers = params.warmup_passengers
        self.last_passenger = 0
        self.queue_snapshots = []
        self.completed = False
        self.log_info = params.log_info

    def count_queue(self):
        """
        Counts number of passengers waiting in queue.
        """
        in_queue = 0
        for p in self.passengers:
            if p.in_queue:
                in_queue += 1
        return in_queue

    def passenger_arrivals(self):
        """
        Defines arrival behavior of passengers and service time per passenger,
        before running the Passenger object. Includes termination logic.
        """
        self.passengers = []
        for i in range(self.n_passengers):
            next_arrival = np.random.exponential(1/self.arrival_rate)
            try:
                yield self.env.timeout(next_arrival)
            except simpy.Interrupt:
                self.completed = True
                log.info(f"[{self.env.now:.1f}]: Steady state reached after {self.last_passenger} passengers. {self.count_queue()} passengers in queue.")
                break
            if i == self.warmup_passengers and self.warmup_passengers > 0 and self.log_info:
                log.info(f"[{self.env.now:.1f}]: Warmup period complete.")
            service_time = np.random.normal(self.expected_service_time, self.service_time_sigma)
            while service_time < 0:
                service_time = np.random.normal(self.expected_service_time, self.service_time_sigma)
            passenger = Passenger(self.env, i+1, service_time, self.servers, log_info=self.log_info)
            self.env.process(passenger.run())
            self.passengers.append(passenger)
            self.last_passenger = i
        if len(self.passengers) == self.n_passengers:
            self.completed = True
            if self.log_info:
                log.info(f"[{self.env.now:.1f}]: All passengers have arrived. {self.count_queue()} still in queue.")
        if self.halt_steady_state:
            if self.sim_halter.is_alive:
                self.sim_halter.interrupt()

    def queue_stats(self):
        """
        Every minute, get the number of passengers in the queue and store the results (for plotting).
        """       
        while True:
            try:
                yield self.env.timeout(1)
            except simpy.Interrupt:
                break
            queue_size = self.count_queue()
            self.queue_snapshots.append(queue_size)
            if self.completed and queue_size == 0:
                break

    def steady_halter(self):
        """
        Simulation termination condition.
        Every 5 minutes, get the number of passengers in the queue. If that value is the same 5 times
        in a row and the warmup period is over, halt the simulation.
        """
        prev = np.zeros(5)
        i=0
        while True:
            try:
                yield self.env.timeout(5)
            except simpy.Interrupt:
                break
            if self.last_passenger < self.warmup_passengers:
                continue
            i = (i+1) % 5
            prev[i] = self.count_queue()
            log.debug(f"[{int(np.round(self.env.now))}]: In queue: {prev[i]}")
            if np.std(prev) == 0:
                if self.sim_arrivals.is_alive:
                    self.sim_arrivals.interrupt()
                break
            

    def start(self):
        """
        Startup logic of the simulation.
        """
        start_time = self.env.now
        self.sim_arrivals = self.env.process(self.passenger_arrivals())
        if self.halt_steady_state:
            self.sim_halter = self.env.process(self.steady_halter())
        self.sim_stats = self.env.process(self.queue_stats())
        self.env.run()
        if self.log_info:
            log.info(f"[{self.env.now:.1f}]: Simulation complete.")
        self.total_time = self.env.now - start_time
        self.n_passengers = self.last_passenger + 1
        if self.log_info:
            log.info(f"[{self.env.now:.1f}]: Simulated {self.n_passengers} passengers.")
        completed_passengers = [p for p in self.passengers[self.warmup_passengers:] if p.completed]
        self.n_completed = len(completed_passengers)
        self.queue_times = np.array([p.queue_time for p in completed_passengers])
        self.server_times = np.array([p.server_time for p in completed_passengers])
        self.total_times = np.array([p.total_time for p in completed_passengers])

    def print_results(self):
        print(f"Total time: {self.total_time:.1f} minutes")
        print(f"Average queue time: {np.mean(self.queue_times):.2f} minutes, sd: {np.std(self.queue_times):.2f} minutes")
        print(f"Average service time: {np.mean(self.server_times):.2f} minutes, sd: {np.std(self.server_times):.2f} minutes")
        print(f"Average total time: {np.mean(self.total_times):.2f} minutes, sd: {np.std(self.total_times):.2f} minutes")


############## THEORETICAL CALCULATIONS ##############

def mg1_theoretical_Wq(target_utilization, service_mean, service_sd):
    """
    Calculates the theoretical average waiting time in queue (Wq) for an M/G/1 queue.

    Parameters
    -----------
    target_utilization: Float representing the target utilization (ρ)
    service_mean: Float representing the mean service time (E[S])
    service_sd: Float representing the standard deviation of service time (σ[S])
    
    Returns
    -----------
    Theoretical average waiting time in queue (Wq)
    """
    ES2 = service_sd**2 + service_mean**2
    lambda_test = target_utilization / service_mean
    check_utilization(lambda_test, service_mean)
    Wq = (lambda_test * ES2) / (2 * (1 - target_utilization))
    return Wq, lambda_test


def check_utilization(arrival_rate, service_mean, n_servers=1):
    rho = (arrival_rate * service_mean)/n_servers
    if rho > 1: print(f"rho = {rho}, Chosen target utilization is not stable.")
    else: print(f"rho = {rho}, Chosen target utilization is stable.")


############## VALIDATION ##############

def multiple_runs(simulator_params, R=40, save_data=False, sim_label="chosen parameters"):
    """
    Code for running multiple simulations with the same parameters
    and returning statistics from those simulations.
    """
    simulator = AirportSimulator(simulator_params)
    ### Simulation Baseline
    rep_Wq = []
    print(f"Running {R} simulations for {sim_label}...")
    while len(rep_Wq) < R:
        simulator.start()
        if len(simulator.queue_times) > 0:
            rep_Wq.append(np.mean(simulator.queue_times))
        else:
            print(f"Warning: Replication attempt has no completed passengers. Retrying...")
    rep_Wq = np.array(rep_Wq)
    if save_data:
        data_folder = "a2_data"
        os.makedirs(data_folder, exist_ok=True)
        save_path = os.path.join(data_folder, "mean_queue_times.npy")
        np.save(save_path, rep_Wq)
        print(f"Replication means saved to {save_path}")

    avg_Wq = np.mean(rep_Wq)
    sd_Wq = np.std(rep_Wq, ddof=1)
    print(f"Simulated average waiting time in queue (Wq): {avg_Wq:.4f} minutes, sd: {sd_Wq:.4f} minutes")

    return rep_Wq, avg_Wq, sd_Wq


def one_sample_t_test(avg_Wq, theoretical_Wq, sd_Wq, R):
    """
    One-sample t-test implementation. Data assumptions:
    - Independent
    - Normally distributed
    """
    t_stat = (avg_Wq - theoretical_Wq) / (sd_Wq / np.sqrt(R))
    t_critical = 2.0227  # 95% confidence interval with R=40, df=39

    print(f"\nONE-SAMPLE T-TEST\nT-statistic: {t_stat:.4f}, T-critical: {t_critical}")
    if abs(t_stat) < t_critical:
        print("Fail to reject the null hypothesis: The simulation matches the theoretical model.")
    else:
        print("Reject the null hypothesis: The simulation does not match the theoretical model.")


def two_sample_t_test(mean1, mean2, sd1, sd2, R1, R2):
    """
    Two-sample t-test implementation. Data assumptions:
    - Independent
    - Normally distributed
    """
    pooled_variance = (
        ((R1 - 1) * sd1**2) + ((R2 - 1) * sd2**2)
    ) / (R1 + R2 - 2)
    pooled_sd = np.sqrt(pooled_variance)
    se = (pooled_sd * (1/R1 + 1/R2))**0.5

    t_stat = (mean1 - mean2) / se
    t_critical = 1.990  # 95% confidence interval with R=40 per group, df = 78
                        # Modify if R is different

    print(f"\nTWO-SAMPLE T-TEST\nT-statistic: {t_stat:.4f}, T-critical: {t_critical}")
    if abs(t_stat) < t_critical:
        print("Fail to reject the null hypothesis: The two samples have no significant difference.")
    else:
        print("Reject the null hypothesis: The two samples differ significantly.")


def wilcoxon(rep_Wq, theoretical_Wq):
    """
    Wilcoxon signed-rank test implementation. Data assumptions:
    - Symmetricaly distributed around the median
    """
    diffs = rep_Wq - theoretical_Wq
    diffs = diffs[diffs != 0]
    abs_diffs = np.abs(diffs)
    ranks = np.argsort(np.argsort(abs_diffs)) + 1

    W_pos = np.sum(ranks[diffs > 0])
    W_neg = np.sum(ranks[diffs < 0])
    W = min(W_pos, W_neg)

    print(f"\nWILCOXON SIGNED-RANK TEST\nTest statistic W: {W}")

    n = len(diffs)
    # null hypothesis: positive and negative diffs are equally likely
    expected_W = n * (n + 1) / 4
    std_W = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (W - expected_W) / std_W
    p_value = 2 * (1 - 0.5 * (1 + erf(abs(z) / np.sqrt(2))))

    print(f"Approximate p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Reject the null hypothesis: The simulation does not match the theoretical Wq.")
    else:
        print("Fail to reject the null hypothesis: The simulation matches the theoretical Wq.")


def one_sample_bootstrap_test(rep_Wq, theoretical_mean, N=10000):
    """
    One-sample bootstrap means test.
    """
    rng = GLOBAL_RNG
    n = len(rep_Wq)
    avg_Wq = np.mean(rep_Wq)
    # Create the null distribution without assuming symmetry
    shifted_data = rep_Wq - avg_Wq + theoretical_mean

    bootstrap_means = np.empty(N)
    for i in range(N):
        resample = rng.choice(shifted_data, size=n, replace=True)
        bootstrap_means[i] = np.mean(resample)
    
    p_val = np.mean(np.abs(bootstrap_means - theoretical_mean) >= np.abs(avg_Wq - theoretical_mean))

    print(f"\nONE-SAMPLE BOOTSTRAP TEST\nBootstrap test p-value: {p_val:.4f}")
    if p_val < 0.05:
        print("Reject the null hypothesis: The simulation does not match the theoretical Wq.")
    else:
        print("Fail to reject the null hypothesis: The simulation matches the theoretical Wq.")
    
    data_folder = "a2_data"
    os.makedirs(data_folder, exist_ok=True)
    save_path = os.path.join(data_folder, "bootstrap.npy")
    np.save(save_path, bootstrap_means)


def two_sample_bootstrap_test(sim_1, sim_2, N=10000):
    """
    Two-sample bootstrap means test.
    """
    rng = GLOBAL_RNG
    x = np.asarray(sim_1)
    y = np.asarray(sim_2)

    n1, n2 = len(x), len(y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    diff_obs = mean_x - mean_y
    pooled_mean = (mean_x * n1 + mean_y * n2) / (n1 + n2)

    x_shifted = x - mean_x + pooled_mean
    y_shifted = y - mean_y + pooled_mean

    boot_diffs = np.empty(N)
    for i in range(N):
        xb = rng.choice(x_shifted, size=n1, replace=True)
        yb = rng.choice(y_shifted, size=n2, replace=True)
        boot_diffs[i] = np.mean(xb) - np.mean(yb)

    p_val = np.mean(np.abs(boot_diffs) >= np.abs(diff_obs))
    print(f"\nTWO-SAMPLE BOOTSTRAP TEST\nBootstrap test p-value: {p_val:.4f}")
    if p_val > 0.05:
        print("Fail to reject the null hypothesis: The two samples have no significant difference.")
    else:
        print("Reject the null hypothesis: The two samples differ significantly.")
    
    data_folder = "a2_data"
    os.makedirs(data_folder, exist_ok=True)
    save_path = os.path.join(data_folder, "two_sample_bootstrap.npy")
    np.save(save_path, boot_diffs)


############## OUTPUT ##############

def theoretical_mean(target_utilization, service_mean, service_sd):
    theoretical_Wq, lambda_test = mg1_theoretical_Wq(target_utilization, service_mean, service_sd)
    print(f"\nTheoretical average waiting time in queue (Wq): {theoretical_Wq:.4f} minutes, test lambda: {lambda_test:.2f}\n")
    return theoretical_Wq, lambda_test


def compare_strategies(sim_1, sim_2, sim_1_label, sim_2_label):
    rep_Wq_1, avg_Wq_1, sd_Wq_1 = multiple_runs(sim_1, sim_label=sim_1_label)
    rep_Wq_2, avg_Wq_2, sd_Wq_2 = multiple_runs(sim_2, sim_label=sim_2_label)

    two_sample_t_test(avg_Wq_1, avg_Wq_2, sd_Wq_1, sd_Wq_2, R1=40, R2=40)
    two_sample_bootstrap_test(rep_Wq_1, rep_Wq_2)


print("\n------------------- PART 2 -------------------\n")
print("~~~~~~ Part 2A: Validating Our Simulator ~~~~~~\n")
target_utilization = 0.85
service_mean = 1
service_sd = 0.25
print("Baseline utilization:")
theoretical_Wq, lambda_test = theoretical_mean(target_utilization, service_mean, service_sd)
sim2a = Simulator_Parameters(arrivals_lambda=lambda_test, expected_service_time=1, service_time_sigma=0.25, n_passengers=100000, warmup_passengers=1000, halt_steady_state=True, n_servers=1, log_info=False)
rep_Wq, avg_Wq, sd_Wq = multiple_runs(sim2a, save_data=True)
one_sample_t_test(avg_Wq, theoretical_Wq, sd_Wq, R=40)
wilcoxon(rep_Wq, theoretical_Wq)
one_sample_bootstrap_test(rep_Wq, theoretical_Wq)


print("\n~~~~~~ Part 2B: Evaluating Interventions ~~~~~~\n")
print("~~~ Baseline vs Intervention A ~~~")
sim_baseline = Simulator_Parameters(arrivals_lambda=arrival_rate, expected_service_time=1, service_time_sigma=0.25, n_passengers=3000, warmup_passengers=1000, n_servers=1, log_info=False)
print("Baseline utilization:")
check_utilization(arrival_rate=arrival_rate, service_mean=1)
sim_intervention_a = Simulator_Parameters(arrivals_lambda=arrival_rate, expected_service_time=1, service_time_sigma=0.25, n_passengers=3000, warmup_passengers=1000, n_servers=2, log_info=False)
print("Intervention A utilization:")
check_utilization(arrival_rate=arrival_rate, service_mean=1, n_servers=2)
compare_strategies(sim_baseline, sim_intervention_a, sim_1_label="baseline parameters", sim_2_label="intervention A")


print("\n~~~ Baseline vs Intervention B ~~~")
sim_intervention_b = Simulator_Parameters(arrivals_lambda=arrival_rate, expected_service_time=1, service_time_sigma=0.1, n_passengers=3000, warmup_passengers=0, n_servers=1, log_info=False)
print("Intervention B utilization:")
check_utilization(arrival_rate=arrival_rate, service_mean=1)
compare_strategies(sim_baseline, sim_intervention_b, sim_1_label="baseline parameters", sim_2_label="intervention B")


############## BONUS INTERVENTION: STABLE RHO ##############

print("\n~~~ Bonus Intervention ~~~")
### Choose expected service time so that rho = 0.9
service_mean = 0.9/arrival_rate
print(f"Service expectation for utilization 0.9: {service_mean:.2f}")
sim_bonus_params = Simulator_Parameters(arrivals_lambda=arrival_rate, expected_service_time=service_mean, service_time_sigma=0.1, n_passengers=3000, warmup_passengers=0, n_servers=1, log_info=False)
check_utilization(arrival_rate=arrival_rate, service_mean=service_mean)
sim_bonus = AirportSimulator(sim_bonus_params)
sim_bonus.start()
sim_bonus.print_results()

print("\n~~~ Bonus Intervention vs Baseline ~~~")
sim_bonus.log_info = False
compare_strategies(sim_bonus_params, sim_baseline, sim_1_label="baseline parameters", sim_2_label="bonus parameters")

print("\n########################################## DISCLAIMER ##########################################\n"+
      "A random seed was set after recording the results video for submission.\n" +
      "As such, the exact values shown in the output of this code may be different.\n"+
      "The findings remain the same.\n"+
      "################################################################################################")
