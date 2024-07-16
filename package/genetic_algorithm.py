import numpy as np
import random
import multiprocessing
from multiprocessing import Pool
import time
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test
from .utils import logrank_fitness, initialize_population

class GeneticAlgorithm:
    def __init__(self, time_data, status_data,num_clusters=2,optimize="p_val",objective='single',res='best',nres=None, population_size=100, num_generations=500, mutation_rate=0.01,
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
        optimize : optimizes given metric -> if given p_val - algorithm optimizes p_values, if given log_p - algorithm optimizes -log10(p),
                if given statistic - algorithm optimizes test statistic
        objective : 'single' objective uses only logrank metrics, 'multiple' objective uses all three metrics
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
        self.objective = objective
        self.optimize = optimize

    def selection(self, population, fitness_scores, num_selected):
        population_sel = [population[i] for i in np.argsort(fitness_scores)[-num_selected:][::-1]]
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
                    alternatives = set(range(1, self.num_clusters+1))
                    alternatives.discard(individual[i])
                    individual[i] = random.choice(list(alternatives))
        elif self.mutation_type == 'swap':
            for i in range(len(individual)):
                if random.random() < self.mutation_rate:
                    swap_idx = random.randint(0, len(individual) - 1)
                    individual[i], individual[swap_idx] = individual[swap_idx], individual[i]
        elif self.mutation_type == 're-prob':  #repeated probabilistic mutation
            while random.random() < self.mutation_rate:
                i = random.randint(0, len(individual) - 1)
                alternatives = set(range(1, self.num_clusters+1))
                alternatives.discard(individual[i])
                individual[i] = random.choice(list(alternatives))
        else:
            raise ValueError("Invalid mutation type. Choose from 'flip-bit', 'swap' or 're-prob'.")
        return individual

    def new_population(self, selected_population):
        parent1, parent2 = random.sample(selected_population, 2)
        child1, child2 = self.crossover(parent1, parent2)
        child1 = self.mutation(child1)
        child2 = self.mutation(child2)
        return child1, child2

    def fitness_pval(self, solution):
        fitness1 = -(multivariate_logrank_test(event_durations=self.time_data, groups=solution, event_observed=self.status_data).p_value)
        fitness2 = -(pairwise_logrank_test(event_durations=self.time_data, groups=solution, event_observed=self.status_data,weightings="wilcoxon").p_value[0])
        fitness3 = -(pairwise_logrank_test(event_durations=self.time_data, groups=solution, event_observed=self.status_data,weightings="tarone-ware").p_value[0])
        if self.objective == 'multiple':
          return [fitness1,fitness2,fitness3]
        return fitness1

    def fitness_logp(self, solution):
        fitness1 = -np.log10(multivariate_logrank_test(event_durations=self.time_data, groups=solution, event_observed=self.status_data).p_value)
        fitness2 = -np.log10(pairwise_logrank_test(event_durations=self.time_data, groups=solution, event_observed=self.status_data,weightings="wilcoxon").p_value[0])
        fitness3 = -np.log10(pairwise_logrank_test(event_durations=self.time_data, groups=solution, event_observed=self.status_data,weightings="tarone-ware").p_value[0])
        if self.objective == 'multiple':
          return [fitness1,fitness2,fitness3]
        return fitness1

    def fitness_statistic(self, solution):
        fitness1 = multivariate_logrank_test(event_durations=self.time_data, groups=solution, event_observed=self.status_data).test_statistic
        fitness2 = pairwise_logrank_test(event_durations=self.time_data, groups=solution, event_observed=self.status_data,weightings="wilcoxon").test_statistic[0]
        fitness3 = pairwise_logrank_test(event_durations=self.time_data, groups=solution, event_observed=self.status_data,weightings="tarone-ware").test_statistic[0]
        if self.objective == 'multiple':
          return [fitness1,fitness2,fitness3]
        return fitness1

    def pareto_dominates(self, fitness1, fitness2):
        """ Returns True if fitness1 Pareto dominates fitness2. """
        return all(x >= y for x, y in zip(fitness1, fitness2)) and any(x > y for x, y in zip(fitness1, fitness2))

    def get_pareto_front(self, fitness_scores):
        """ Returns the Pareto front from the fitness scores. """
        pareto_front = []
        for i, fitness1 in enumerate(fitness_scores):
            if not any(self.pareto_dominates(fitness2, fitness1) for j, fitness2 in enumerate(fitness_scores) if i != j):
                pareto_front.append(i)
        return pareto_front

    def assign_pareto_ranks(self, fitness_scores):
        """ Assigns Pareto ranks to the entire fitness scores list. """
        fitness_scores = fitness_scores.copy()
        pareto_ranks = {}
        rank = 1
        while fitness_scores:
            pareto_front = self.get_pareto_front(fitness_scores)
            for i in pareto_front:
                pareto_ranks[i] = rank
            fitness_scores = [fitness_scores[i] for i in range(len(fitness_scores)) if i not in pareto_front]
            rank += 1
        return pareto_ranks
    
    def selection_multi(self, population, fitness_scores,num_selected):
        """ Selects the top 25% of the population based on Pareto ranks. """
        pareto_ranks = self.assign_pareto_ranks(fitness_scores)
        sorted_indices = sorted(pareto_ranks, key=lambda ind: pareto_ranks[ind])
        selected_indices = sorted_indices[:num_selected]
        population_sel = [population[i] for i in selected_indices]
        return population_sel

    def run(self):
        start_time = time.time()
        num_patients = len(self.time_data)
        min_size = int(num_patients * self.min_cluster_size)
        population = initialize_population(self.population_size, num_patients,self.num_clusters)
        consecutive_generations_without_improvement = 0
        pool = multiprocessing.Pool()

        if self.nres!=None and self.nres >self.population_size :
            raise ValueError("nres should be less than or equal to population_size")

        if self.optimize == 'p_val':
            fitness_function = self.fitness_pval
        elif self.optimize == 'lop_p':
            fitness_function = self.fitness_logp
        elif self.optimize == 'statistic':
            fitness_function = self.fitness_statistic
        else:
            raise ValueError("Unknown optimization type: {}".format(self.optimize))

        if self.objective=='single':
            selection_function = self.selection
        elif self.objective=='multiple':
            selection_function = self.selection_multi
        else:
            raise ValueError("Unknown optimization type: {}".format(self.objective))

        for generation in range(self.num_generations):
            print(generation)
            fitness_scores = [fitness_function(cluster_indices) for cluster_indices in population]
            selected_population = selection_function(population, fitness_scores, int(self.population_size * self.selection_percentage))

            with Pool() as pool:
                args = [(selected_population,) for _ in range(int(self.population_size // 2))]
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
