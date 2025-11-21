import csv
import numpy as np
import simpy
import logging

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
    avg_monthly = np.mean(data["Total Passengers"])
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

    def __init__(self, env, passenger_id, lane_time, lanes):
        """
        Initializes the passenger with a unique ID.
        """
        self.passenger_id = passenger_id
        self.lane_time = lane_time
        self.lanes = lanes
        self.env = env
        self.in_queue = False
        self.completed = False
    
    def run(self):
        start_time = self.env.now
        log.debug(f"[{self.env.now:.1f}]: Passenger {self.passenger_id} arrives at airport and enteres queue")
        self.in_queue = True
        with self.lanes.request() as req:
            yield req
            log.debug(f"[{self.env.now:.1f}]: Passenger {self.passenger_id} exits queue and enters lane")
            self.in_queue = False
            self.queue_time = self.env.now - start_time
            yield self.env.timeout(self.lane_time)
            log.debug(f"[{self.env.now:.1f}]: Passenger {self.passenger_id} leaves security")
            self.total_time = self.env.now - start_time
            self.completed = True

class AirportSimulator:
    """
    Simulator class for airport arrivals.
    """

    def __init__(self, arrivals_lambda, expected_service_time, service_time_sigma, n_passengers, n_lanes=1, halt_steady_state=False, warmup_passengers=1000):
        """
        Initializes the simulator with the arrival rate (number of passengers per minue), expected service time, and service time variance.
        """
        self.expected_service_time = expected_service_time
        self.service_time_sigma = service_time_sigma
        self.arrival_rate = 1/arrivals_lambda # rate = 1 / arrivals per minute
        self.n_passengers = n_passengers
        self.n_lanes = n_lanes
        self.halt_steady_state = halt_steady_state
        self.env = simpy.Environment()
        self.lanes = simpy.Resource(self.env, capacity=self.n_lanes)
        self.warmup_passengers = warmup_passengers
        self.last_passenger = 0

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
                break
            if i == self.warmup_passengers and self.warmup_passengers > 0:
                log.info(f"[{self.env.now:.1f}]: Warmup period complete.")
            service_time = np.random.normal(self.expected_service_time, self.service_time_sigma)
            while service_time < 0:
                service_time = np.random.normal(self.expected_service_time, self.service_time_sigma)
            passenger = Passenger(self.env, i+1, service_time, self.lanes)
            self.env.process(passenger.run())
            self.passengers.append(passenger)
            self.last_passenger = i
        if len(self.passengers) == self.n_passengers:
            log.info(f"[{self.env.now:.1f}]: All passengers have arrived. {self.count_queue()} still in queue.")
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
            

    def start(self):
        start_time = self.env.now
        self.sim_arrivals = self.env.process(self.passenger_arrivals())
        if self.halt_steady_state:
            self.sim_halter = self.env.process(self.steady_halter())
        self.env.run()
        log.info(f"[{self.env.now:.1f}]: Simulation complete.")
        self.total_time = self.env.now - start_time
        self.n_passengers = self.last_passenger + 1
        log.info(f"[{self.env.now:.1f}]: Simulated {self.n_passengers} passengers.")
        self.queue_times = np.zeros(self.n_passengers - self.warmup_passengers)
        self.lane_times = np.zeros(self.n_passengers - self.warmup_passengers)
        self.total_times = np.zeros(self.n_passengers - self.warmup_passengers)
        for i, j in enumerate(range(self.warmup_passengers, self.n_passengers)):
            self.queue_times[i] = self.passengers[j].queue_time
            self.lane_times[i] = self.passengers[j].lane_time
            self.total_times[i] = self.passengers[j].total_time

    def print_results(self):
        print(f"Total time: {self.total_time:.1f} minutes")
        print(f"Average queue time: {np.mean(self.queue_times):.2f} minutes, sd: {np.std(self.queue_times):.2f} minutes")
        print(f"Average lane time: {np.mean(self.lane_times):.2f} minutes, sd: {np.std(self.lane_times):.2f} minutes")
        print(f"Average total time: {np.mean(self.total_times):.2f} minutes, sd: {np.std(self.total_times):.2f} minutes")


## Run sample simulation

sample_sim = AirportSimulator(arrivals_lambda=3, expected_service_time=2, service_time_sigma=1, n_passengers=1000, warmup_passengers=100, halt_steady_state=True, n_lanes=6)
print("~~~ Sample Simulation ~~~")
sample_sim.start()
sample_sim.print_results()

## Validate the simulator

print("\n~~~ Stable Test Rate ~~~")
sim_2a = AirportSimulator(arrivals_lambda=0.85, expected_service_time=1, service_time_sigma=0.25, n_passengers=100000, warmup_passengers=1000, halt_steady_state=True, n_lanes=1)
sim_2a.start()
sim_2a.print_results()

### Establish Stable Test Rate

### Theoretical Baseline

### Simulation Baseline

### Hypothesis Test

## Evaluating Intervations

print("\n~~~ Baseline ~~~")
sim_baseline = AirportSimulator(arrivals_lambda=arrival_rate, expected_service_time=1, service_time_sigma=0.25, n_passengers=3000, warmup_passengers=0, n_lanes=1)
sim_baseline.start()
sim_baseline.print_results()

### Intervention A

print("\n~~~ Intervention A ~~~")
sim_intervention_a = AirportSimulator(arrivals_lambda=arrival_rate, expected_service_time=1, service_time_sigma=0.25, n_passengers=3000, warmup_passengers=0, n_lanes=2)
sim_intervention_a.start()    
sim_intervention_a.print_results()

### Intervention B

print("\n~~~ Intervention B ~~~")
sim_intervention_b = AirportSimulator(arrivals_lambda=arrival_rate, expected_service_time=1, service_time_sigma=0.1, n_passengers=3000, warmup_passengers=0, n_lanes=1)
sim_intervention_b.start()    
sim_intervention_b.print_results()

### Bonus Intervention