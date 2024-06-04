import numpy as np
import random
import multiprocessing
from multiprocessing import Pool
import time
from .utils import logrank_fitness, initialize_population

class GeneticAlgorithm:
    def __init__(self, time_data, status_data,num_clusters=2,res='best',nres=None, population_size=100, num_generations=500, mutation_rate=0.01,
                 eps=1e-4, max_consecutive_generations=5, selection_percentage=0.25,
                 min_cluster_size=0.1, crossover_type='one-point', mutation_type='flip-bit'):

        """
        time_data : number of days/months until the specified event i.e., death
        status_data : Binary Censoring status
        num_clusters : number of clusters
        res = required result ; 'best' returns only single best solution, 'best_dist' returns number of best solutions mentioned in nres
        nres = number of best scores needed if res = 'best_distr'
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
        self.time_data = time_data
        self.status_data = status_data
        self.num_clusters = num_clusters
        self.res = res
        self.nres = nres
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.eps = eps
        self.max_consecutive_generations = max_consecutive_generations
        self.selection_percentage = selection_percentage
        self.min_cluster_size = min_cluster_size
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type

    def selection(self, population, fitness_scores, num_selected):
        population_sel = [population[i] for i in np.argsort(fitness_scores)[:num_selected]]
        return population_sel

    def crossover(self, parent1, parent2):
        if self.crossover_type == 'one-point':
            crossover_point = random.randint(0, len(parent1) - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        elif self.crossover_type == 'two-point':
            point1, point2 = sorted(random.sample(range(len(parent1)), 2))
            child1 = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
            child2 = np.concatenate((parent2[:point1], parent1[point1:point2], parent2[point2:]))
        elif self.crossover_type == 'uniform':
            mask1 = np.random.randint(1, self.num_clusters+1, size=len(parent1))
            mask2 = np.random.randint(1, self.num_clusters+1, size=len(parent2))
            child1 = np.where(mask1 == mask2, parent1, parent2)
            child2 = np.where(mask1 == mask2, parent2, parent1)
        else:
            raise ValueError("Invalid crossover type. Choose from 'one-point', 'two-point', or 'uniform'.")
        return child1, child2

    def mutation(self, individual):
        if self.mutation_type == 'flip-bit':
            for i in range(len(individual)):
                if random.random() < self.mutation_rate:
                    new_cluster = random.randint(1, self.num_clusters+1)  # Assuming clusters are labeled 1 to num_clusters
                    while new_cluster == individual[i]:
                        new_cluster = random.randint(1, self.num_clusters+1)
                    individual[i] = new_cluster
        elif self.mutation_type == 'swap':
            for i in range(len(individual)):
                if random.random() < self.mutation_rate:
                    swap_idx = random.randint(0, len(individual) - 1)
                    individual[i], individual[swap_idx] = individual[swap_idx], individual[i]
        else:
            raise ValueError("Invalid mutation type. Choose from 'flip-bit' or 'swap'.")
        return individual

    def new_population(self, selected_population):
        parent1, parent2 = random.sample(selected_population, 2)
        child1, child2 = self.crossover(parent1, parent2)
        child1 = self.mutation(child1)
        child2 = self.mutation(child2)
        return child1, child2

    def run(self):
        start_time = time.time()
        num_patients = len(self.time_data)
        min_size = int(num_patients * self.min_cluster_size)
        population = initialize_population(self.population_size, num_patients,self.num_clusters)
        consecutive_generations_without_improvement = 0
        pool = multiprocessing.Pool()

        if self.nres!=None and self.nres >self.population_size :
            raise ValueError("nres should be less than or equal to population_size")

        for generation in range(self.num_generations):
            print(generation)
            fitness_scores = [logrank_fitness(cluster_indices, self.time_data, self.status_data) for cluster_indices in population]
            selected_population = self.selection(population, fitness_scores, int(self.population_size * self.selection_percentage))

            with Pool() as pool:
                args = [(selected_population, self.mutation_rate, self.crossover_type, self.mutation_type) for _ in range(int(self.population_size // 2))]
                results = pool.starmap(self.new_population, args)
                offspring_population = [child for pair in results for child in pair if np.all(np.bincount(child)[1:] >= min_size)]

            current_best_fitness = min(fitness_scores)
            if generation == 0:
                best_fitness = max(fitness_scores)
            print(current_best_fitness)

            if -np.log10(current_best_fitness) + np.log10(best_fitness) < self.eps:
                consecutive_generations_without_improvement += 1
            else:
                best_fitness = current_best_fitness
                consecutive_generations_without_improvement = 0

            if consecutive_generations_without_improvement >= self.max_consecutive_generations:
                break
            if generation == self.num_generations - 1:
                print("Algorithm did not converge")
            else:
                population = offspring_population

        end_time = time.time()
        print('Total time taken : ', end_time - start_time, ' seconds')
        if self.res == 'best':
            best_solution = population[fitness_scores.index(current_best_fitness)]
            return best_solution, current_best_fitness
        elif self.res == 'best_dist':
            population_sel, scores_sel = [population[i] for i in np.argsort(fitness_scores)[:self.nres]], [fitness_scores[i] for i in np.argsort(fitness_scores)[:self.nres]]
            return population_sel, scores_sel
