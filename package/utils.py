import numpy as np
from lifelines.statistics import logrank_test

# Function to calculate log-rank test statistic
def logrank_fitness(cluster_indices, time_data, status_data):
    # Extracting data for c1 and c2
    cluster1_time_data = time_data[cluster_indices == 1]
    cluster1_status_data = status_data[cluster_indices == 1]

    cluster2_time_data = time_data[cluster_indices == 2]
    cluster2_status_data = status_data[cluster_indices == 2]

    # log-rank test
    result = logrank_test(cluster1_time_data, cluster2_time_data, event_observed_A=cluster1_status_data, event_observed_B=cluster2_status_data)

    # Returning p-value from log-rank test
    return result.p_value

# Function to initialize population
def initialize_population(population_size, num_patients):
    return [np.random.randint(1, 3, num_patients) for _ in range(population_size)]  #labels -> 1,2

