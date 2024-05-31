from .utils import *

def random_search(time_data, status_data,population_size=100):
  num_patients = len(time_data)
  population = initialize_population(population_size, num_patients)
  scores = [logrank_fitness(cluster_indices, time_data, status_data) for cluster_indices in population]
  return scores, population[scores.index(min(scores))]  # solution with best score
