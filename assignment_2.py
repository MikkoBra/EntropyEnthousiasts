import csv
import numpy as np

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
        extracted[name] = np.array(extracted[name])
    
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

# Part 1

## Calculate Arrival Rate (λ) in passengers per minute

# Part 2

## Build a discrete-event simulator

## Validate the simulator

### Establish Stable Test Rate

### Theoretical Baseline

### Simulation Baseline

### Hypothesis Test

## Evaluating Intervations

### Intervention A

### Intervention B

### Bonus Intervention