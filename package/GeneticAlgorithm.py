import numpy as np
import random
import multiprocessing
from multiprocessing import Pool
import time
from .utils import *

# Function to perform selection based on fitness scores
def selection(population, fitness_scores, num_selected):
    return [population[i] for i in np.argsort(fitness_scores)[:num_selected]]

# Function to perform crossover
def crossover(parent1, parent2, crossover_type):
    if crossover_type == 'one-point':
        crossover_point = random.randint(0, len(parent1) - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    elif crossover_type == 'two-point':
        point1, point2 = sorted(random.sample(range(len(parent1)), 2))
        child1 = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
        child2 = np.concatenate((parent2[:point1], parent1[point1:point2], parent2[point2:]))
    elif crossover_type == 'uniform':
        mask = np.random.randint(0, 2, size=len(parent1), dtype=bool)
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
    else:
        raise ValueError("Invalid crossover type. Choose from 'one-point', 'two-point', or 'uniform'.")
    return child1, child2

# Function to perform mutation
def mutation(individual, mutation_rate, mutation_type):
    if mutation_type == 'flip-bit':
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] = 2 if individual[i] == 1 else 1  # Flip 2 to 1 or 1 to 2
    elif mutation_type == 'swap':
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                swap_idx = random.randint(0, len(individual) - 1)
                individual[i], individual[swap_idx] = individual[swap_idx], individual[i]
    else:
        raise ValueError("Invalid mutation type. Choose from 'flip-bit' or 'swap'.")
    return individual

# Function to generate new population
def new_population(selected_population,mutation_rate,crossover_type,mutation_type):
    parent1, parent2 = random.sample(selected_population, 2)
    child1, child2 = crossover(parent1, parent2,crossover_type)
    child1 = mutation(child1, mutation_rate,mutation_type)
    child2 = mutation(child2, mutation_rate,mutation_type)
    return child1,child2

# Main-Genetic Algorithm
def genetic_algorithm(time_data, status_data, population_size=100, num_generations=500, mutation_rate=0.01,eps=1e-4, max_consecutive_generations=5, selection_percentage = 0.25,
                      min_cluster_size=0.1,crossover_type='one-point',mutation_type='flip-bit'):
    """
    time_data : number of days/months until the specified event i.e., death
    status_data : Binary Censoring status
    population_size : number of candidate solutions to be generated, default 100
    num_generations : number of iterations, default 500
    mutation_rate : used to decide if the value in candidate solution needs to be flipped or not while mutation process, default 0.01
    eps : epsilon, a small threshold value, indicating that the algorithm terminates if changes are less than this value, default 1e-4
    max_consecutive_generations : the maximum number of consecutive iterations during which the changes must be smaller than epsilon for the algorithm to terminate, default 5
    selection_percentage : indicates the percentage of top candidate solutions to be selected to create new population, default 0.25 (25% of total population)
    min_cluster_size : minimum cluster size, default 0.1 (10% of number of patients)
    crossover_type : type of crossover, possible values : one-point, two-point, uniform ; default one-point
    mutation_type : type of mutation, possible values : flip-bit, swap ; default flip-bit
    """
    # np.random.seed(42)
    # random.seed(42)
    start_time = time.time()
    num_patients = len(time_data)
    min_size = int(num_patients * min_cluster_size)
    population = initialize_population(population_size, num_patients)
    consecutive_generations_without_improvement=0
    pool = multiprocessing.Pool()

    for generation in range(num_generations):
        print(generation)
        fitness_scores = [logrank_fitness(cluster_indices, time_data, status_data) for cluster_indices in population]
        selected_population = selection(population, fitness_scores, int(population_size * selection_percentage))

        with Pool() as pool:
        # We need to run new_population for population_size / 2 times as we get 2 children each time
          args = [(selected_population, mutation_rate,crossover_type,mutation_type) for _ in range(int(population_size // 2))]
          results = pool.starmap(new_population, args)
        # Flatten the list of tuples, and check if cluster size is greater than min_size specified
        offspring_population = [child for pair in results for child in pair if np.all(np.bincount(child)[1:] >= min_size)]


        # Track the best fitness score achieved so far
        current_best_fitness = min(fitness_scores)
        if generation == 0:
          best_fitness = max(fitness_scores)  # setting initial best_fitness score
        print(current_best_fitness)
        # Check for improvement
        if -np.log10(current_best_fitness)+np.log10(best_fitness) <  eps:
            consecutive_generations_without_improvement += 1
        else:
            best_fitness = current_best_fitness
            consecutive_generations_without_improvement = 0

        # Terminate if there's no improvement for a defined number of consecutive generations
        if consecutive_generations_without_improvement >= max_consecutive_generations:
            break
        if generation == num_generations-1:
          print("Algorithm did not converge")
        else:
          population = offspring_population

    best_solution = population[fitness_scores.index(current_best_fitness)]  # Get the corresponding solution
    end_time = time.time()
    print('Total time taken : ',end_time - start_time,' seconds')
    return best_solution, current_best_fitness
