import csv
import numpy as np
import simpy
import logging
import matplotlib.pyplot as plt
import os
from math import erf

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger('AirportSim')


COLUMNS = np.array(["Year", "Month", "Europe Passengers", "Intercontinental Passengers", "Total Passengers"])


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
    

# Part 1

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
print("------------------- PART 1 -------------------")
print(f"Arrival rate per minute calculated from airport.csv: {arrival_rate:.2f}")


# Part 2
print("\n------------------- PART 2 -------------------\n")

## Build a discrete-event simulator

class Passenger:
    """
    Passenger class for airport arrivals.
    """

    def __init__(self, env, passenger_id, server_time, servers):
        """
        Initializes the passenger with a unique ID.
        """
        self.passenger_id = passenger_id
        self.server_time = server_time
        self.servers = servers
        self.env = env
        self.in_queue = False
        self.completed = False
    
    def run(self):
        start_time = self.env.now
        log.debug(f"[{self.env.now:.1f}]: Passenger {self.passenger_id} arrives at airport and enteres queue")
        self.in_queue = True
        with self.servers.request() as req:
            yield req
            log.debug(f"[{self.env.now:.1f}]: Passenger {self.passenger_id} exits queue and enters server")
            self.in_queue = False
            self.queue_time = self.env.now - start_time
            yield self.env.timeout(self.server_time)
            log.debug(f"[{self.env.now:.1f}]: Passenger {self.passenger_id} leaves security")
            self.total_time = self.env.now - start_time
            self.completed = True

class AirportSimulator:
    """
    Simulator class for airport arrivals.
    """

    def __init__(self, arrivals_lambda, expected_service_time, service_time_sigma, n_passengers, n_servers=1, halt_steady_state=False, warmup_passengers=1000):
        """
        Initializes the simulator with the arrival rate (number of passengers per minue), expected service time, and service time variance.
        """
        self.expected_service_time = expected_service_time
        self.service_time_sigma = service_time_sigma
        self.arrival_rate = 1/arrivals_lambda # rate = 1 / arrivals per minute
        self.n_passengers = n_passengers
        self.n_servers = n_servers
        self.halt_steady_state = halt_steady_state
        self.env = simpy.Environment()
        self.servers = simpy.Resource(self.env, capacity=self.n_servers)
        self.warmup_passengers = warmup_passengers
        self.last_passenger = 0
        self.queue_snapshots = []
        self.completed = False

    def count_queue(self):
        in_queue = 0
        for p in self.passengers:
            if p.in_queue:
                in_queue += 1
        return in_queue

    def passenger_arrivals(self):
        self.passengers = []
        for i in range(self.n_passengers):
            next_arrival = np.random.exponential(self.arrival_rate)
            try:
                yield self.env.timeout(next_arrival)
            except simpy.Interrupt:
                log.info(f"[{self.env.now:.1f}]: Steady state reached after {self.last_passenger} passengers. {self.count_queue()} passengers in queue.")
                self.completed = True
                break
            if i == self.warmup_passengers and self.warmup_passengers > 0:
                log.info(f"[{self.env.now:.1f}]: Warmup period complete.")
            service_time = np.random.normal(self.expected_service_time, self.service_time_sigma)
            while service_time < 0:
                service_time = np.random.normal(self.expected_service_time, self.service_time_sigma)
            passenger = Passenger(self.env, i+1, service_time, self.servers)
            self.env.process(passenger.run())
            self.passengers.append(passenger)
            self.last_passenger = i
        if len(self.passengers) == self.n_passengers:
            log.info(f"[{self.env.now:.1f}]: All passengers have arrived. {self.count_queue()} still in queue.")
            self.completed = True
        if self.halt_steady_state:
            if self.sim_halter.is_alive:
                self.sim_halter.interrupt()

    def steady_halter(self):
        """
        Every 5 minutes, get the number of passengers in the queue. If that value is the same 5 times in a row and the warmup period is over, halt the simulation.
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

    def queue_stats(self):
        """
        Every minue, get the number of passengers in the queue and store the resutls (for plotting).
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

    def start(self):
        start_time = self.env.now
        self.sim_stats = self.env.process(self.queue_stats())
        self.sim_arrivals = self.env.process(self.passenger_arrivals())
        if self.halt_steady_state:
            self.sim_halter = self.env.process(self.steady_halter())
        self.env.run()
        log.info(f"[{self.env.now:.1f}]: Simulation complete.")
        self.total_time = self.env.now - start_time
        self.n_passengers = self.last_passenger + 1
        log.info(f"[{self.env.now:.1f}]: Simulated {self.n_passengers} passengers.")
        self.queue_times = np.zeros(self.n_passengers - self.warmup_passengers)
        self.server_times = np.zeros(self.n_passengers - self.warmup_passengers)
        self.total_times = np.zeros(self.n_passengers - self.warmup_passengers)
        for i, j in enumerate(range(self.warmup_passengers, self.n_passengers)):
            self.queue_times[i] = self.passengers[j].queue_time
            self.server_times[i] = self.passengers[j].server_time
            self.total_times[i] = self.passengers[j].total_time

    def print_results(self):
        print(f"Total time: {self.total_time:.1f} minutes")
        print(f"Average queue time: {np.mean(self.queue_times):.2f} minutes, sd: {np.std(self.queue_times):.2f} minutes")
        print(f"Average service time: {np.mean(self.server_times):.2f} minutes, sd: {np.std(self.server_times):.2f} minutes")
        print(f"Average total time: {np.mean(self.total_times):.2f} minutes, sd: {np.std(self.total_times):.2f} minutes")

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
    assert lambda_test * service_mean < 1, "Chosen target utilization not stable."
    Wq = (lambda_test * ES2) / (2 * (1 - target_utilization))
    return Wq


## Run sample simulation

sample_sim = AirportSimulator(arrivals_lambda=3, expected_service_time=2, service_time_sigma=1, n_passengers=1000, warmup_passengers=100, halt_steady_state=True, n_servers=6)
print("~~~ Sample Simulation ~~~")
sample_sim.start()
sample_sim.print_results()


### Vaidate simulator

def multiple_runs():
    ### Establish Stable Test Rate
    print("\n~~~ Part 2a: Establish Stable Test Rate ~~~")
    target_utilization = 0.85
    service_mean = 1
    service_sd = 0.25

    ### Simulation Baseline
    R = 40
    rep_Wq = np.zeros(R)
    sims = []
    for r in range(R):
        sim2a = AirportSimulator(arrivals_lambda=0.85, expected_service_time=1, service_time_sigma=0.25, n_passengers=100000, warmup_passengers=1000, halt_steady_state=True, n_servers=1)
        sim2a.start()
        rep_Wq[r] = np.mean(sim2a.total_times) # TODO: check if this is correct
        sims.append(sim2a)
    data_folder = "a2_data"
    os.makedirs(data_folder, exist_ok=True)
    save_path = os.path.join(data_folder, "rep_Wq.npy")
    np.save(save_path, rep_Wq)
    print(f"Replication means saved to {save_path}")

    avg_Wq = np.mean(rep_Wq)
    sd_Wq = np.std(rep_Wq, ddof=1)
    print(f"Simulated average waiting time in queue (Wq): {avg_Wq:.4f} minutes, sd: {sd_Wq:.4f} minutes")
    ### Theoretical Baseline
    theoretical_Wq = mg1_theoretical_Wq(target_utilization, service_mean, service_sd)
    print(f"Theoretical average waiting time in queue (Wq): {theoretical_Wq:.4f} minutes")

    return rep_Wq, avg_Wq, theoretical_Wq, sd_Wq, sims


def one_sample_t_test(avg_Wq, theoretical_Wq, sd_Wq, R):
    ### Hypothesis Test
    t_stat = (avg_Wq - theoretical_Wq) / (sd_Wq / np.sqrt(R))
    t_critical = 2.0227  # 95% confidence interval with R=40, df=39

    print(f"T-statistic: {t_stat:.4f}, T-critical: {t_critical}")
    if abs(t_stat) < t_critical:
        print("Fail to reject the null hypothesis: The simulation matches the theoretical model.")
    else:
        print("Reject the null hypothesis: The simulation does not match the theoretical model.")


def wilcoxon(rep_Wq, theoretical_Wq):
    diffs = rep_Wq - theoretical_Wq
    diffs = diffs[diffs != 0]
    abs_diffs = np.abs(diffs)
    ranks = np.argsort(np.argsort(abs_diffs)) + 1

    W_pos = np.sum(ranks[diffs > 0])
    W_neg = np.sum(ranks[diffs < 0])
    W = min(W_pos, W_neg)

    print(f"Wilcoxon signed-rank test statistic W: {W}")

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


rep_Wq, avg_Wq, theoretical_Wq, sd_Wq, sims = multiple_runs()
one_sample_t_test(avg_Wq, theoretical_Wq, sd_Wq, R=40)
wilcoxon(rep_Wq, theoretical_Wq)


## Evaluating Intervations

print("\n~~~ Baseline ~~~")
sim_baseline = AirportSimulator(arrivals_lambda=arrival_rate, expected_service_time=1, service_time_sigma=0.25, n_passengers=3000, warmup_passengers=0, n_servers=1)
sim_baseline.start()
sim_baseline.print_results()

### Intervention A

print("\n~~~ Intervention A ~~~")
sim_intervention_a = AirportSimulator(arrivals_lambda=arrival_rate, expected_service_time=1, service_time_sigma=0.25, n_passengers=3000, warmup_passengers=0, n_servers=2)
sim_intervention_a.start()    
sim_intervention_a.print_results()

### Intervention B

print("\n~~~ Intervention B ~~~")
sim_intervention_b = AirportSimulator(arrivals_lambda=arrival_rate, expected_service_time=1, service_time_sigma=0.1, n_passengers=3000, warmup_passengers=0, n_servers=1)
sim_intervention_b.start()    
sim_intervention_b.print_results()

### Bonus Intervention

print("\n~~~ Bonus Intervention ~~~")
sim_bonus = AirportSimulator(arrivals_lambda=arrival_rate, expected_service_time=0.5, service_time_sigma=0.25, n_passengers=3000, warmup_passengers=0, n_servers=2)
sim_bonus.start()
sim_bonus.print_results()

### Plotting

plt.figure(figsize=(10,10))
plt.plot(sim_baseline.queue_snapshots, label="Baseline")
plt.plot(sim_intervention_a.queue_snapshots, label="Intervention A")
plt.plot(sim_intervention_b.queue_snapshots, label="Intervention B")
plt.plot(sim_bonus.queue_snapshots, label="Bonus Intervention")
plt.legend()
plt.xlabel("Minutes")
plt.ylabel("Number of Passengers in Queue")
plt.title("Airport Queue Length Over Time")
plt.savefig("airport_queue_length.png")

plt.figure(figsize=(10,10))
plt.plot(sim_baseline.queue_times, label="Baseline")
plt.plot(sim_intervention_a.queue_times, label="Intervention A")
plt.plot(sim_intervention_b.queue_times, label="Intervention B")
plt.plot(sim_bonus.queue_times, label="Bonus Intervention")
plt.legend()
plt.xlabel("Passenger number")
plt.ylabel("Queue Time")
plt.title("Airport Queue Time Over Passengers")
plt.savefig("airport_queue_time.png")

plt.figure(figsize=(10,10))
for i, sim in enumerate(sims):
    plt.plot(sim.queue_times, 'b', alpha=0.5)
plt.title("Stable Test Rate")
plt.xlabel("Passenger number")
plt.ylabel("Queue Time")
plt.savefig("stable_test_rate.png")

plt.figure(figsize=(10,5))
plt.ylim(-1, 1)
plt.yticks([])
plt.plot(rep_Wq, np.zeros_like(rep_Wq), 'ko', label="Simulation Iteration")
plt.axvline(theoretical_Wq, color='g', linestyle='--', label="Theoretical (M/G/1)")
plt.axvline(avg_Wq, color='r', linestyle='--', label="Mean (M/G/1)")
ci = sd_Wq/np.sqrt(40) * 1.96
plt.fill_between([avg_Wq - ci, avg_Wq + ci], [-1, -1], [1, 1], alpha=0.1, color='b', label="95% Confidence Interval")
plt.legend()
plt.xlabel("Average waiting time in queue")
plt.title("Distribution of Average Waiting Time in Queue")
plt.savefig("average_waiting_time_in_queue.png")