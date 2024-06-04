import numpy as np
from lifelines.statistics import multivariate_logrank_test

# Function to calculate log-rank test statistic
def logrank_fitness(cluster_indices, time_data, status_data):
    result = multivariate_logrank_test(event_durations=time_data, groups=cluster_indices, event_observed=status_data).p_value
    return result

# Function to initialize population
def initialize_population(population_size, num_patients,num_clusters):
    population = [np.random.randint(1, num_clusters+1, num_patients) for _ in range(population_size)]  
    return population
