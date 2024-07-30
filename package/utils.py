import numpy as np
from lifelines.statistics import multivariate_logrank_test
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Function to calculate log-rank test statistic
def logrank_fitness(cluster_indices, time_data, status_data):
    result = multivariate_logrank_test(event_durations=time_data, groups=cluster_indices, event_observed=status_data).p_value
    return result

# Function to initialize population
def initialize_population(population_size, num_patients,num_clusters):
    population = [np.random.randint(1, num_clusters+1, num_patients) for _ in range(population_size)]  
    return population

def transform_omics_data(omics_data):
    transformed_omics = []

    for omics_type, data in omics_data.items():
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        pca = PCA()
        pca_data = pca.fit_transform(scaled_data)

        transformed_omics.append(pca_data)

    concatenated_data = np.hstack(transformed_omics)
    final_scaler = StandardScaler()
    scaled_concatenated_data = final_scaler.fit_transform(concatenated_data)

    final_pca = PCA()
    final_pca_data = final_pca.fit_transform(scaled_concatenated_data)
    return final_pca_data
