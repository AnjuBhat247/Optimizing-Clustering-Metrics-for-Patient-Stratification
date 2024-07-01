import numpy as np
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test
import pygad

class pygad_ga:
  def __init__(self,ga_args,clinical_data,res="best_dist",optimize="p_val",objective='single',cluster_min_size=0.1,epsilon=1e-4, max_consecutive_generations=5):
    """
    ga_args : all necessary arguments for pygad
    res : type of result required -> best_dist returns final population, best returns only best solution
    optimize : optimizes given metric -> if given p_val - algorithm optimizes p_values, if given log_p - algorithm optimizes -log10(p),
                if given statistic - algorithm optimizes test statistic
    data : patient data that must have columns Time and Status to calculate various metrics
    objective : 'single' objective uses only logrank metrics, 'multiple' objective uses all three metrics
    """
    global data, obj,min_size, previous_best_fitness, no_change_count, eps, no_significant_change_generations
    data = clinical_data
    obj = objective
    min_size = cluster_min_size
    previous_best_fitness = None
    no_change_count = 0
    eps = epsilon
    no_significant_change_generations = max_consecutive_generations
    self.ga_args = ga_args
    self.res = res
    self.optimize = optimize
    # self.data = data
    self.objective = objective

  def fitness_pval(ga_instance, solution, solution_idx):
    cluster_counts = np.bincount(np.array(solution, dtype=int))
    if any(count < min_size for count in cluster_counts[1:]):
      if obj == 'multiple':
        return [np.inf] * 3
      return np.inf
    fitness1 = -(multivariate_logrank_test(event_durations=data['Time'], groups=solution, event_observed=data['status']).p_value)
    fitness2 = -(pairwise_logrank_test(event_durations=data['Time'], groups=solution, event_observed=data['status'],weightings="wilcoxon").p_value[0])
    fitness3 = -(pairwise_logrank_test(event_durations=data['Time'], groups=solution, event_observed=data['status'],weightings="tarone-ware").p_value[0])
    if obj == 'multiple':
      return [fitness1,fitness2,fitness3]
    return fitness1


  def fitness_logp(ga_instance, solution,solution_idx):
    cluster_counts = np.bincount(solution)
    if any(count < min_size for count in cluster_counts[1:]):
      if obj == 'multiple':
        return [np.inf] * 3
      return np.inf
    fitness1 = -np.log10(multivariate_logrank_test(event_durations=data['Time'], groups=solution, event_observed=data['status']).p_value)
    fitness2 = -np.log10(pairwise_logrank_test(event_durations=data['Time'], groups=solution, event_observed=data['status'],weightings="wilcoxon").p_value[0])
    fitness3 = -np.log10(pairwise_logrank_test(event_durations=data['Time'], groups=solution, event_observed=data['status'],weightings="tarone-ware").p_value[0])
    if obj == 'multiple':
      return [fitness1,fitness2,fitness3]
    return fitness1

  def fitness_statistic(ga_instance, solution,solution_idx):
    cluster_counts = np.bincount(solution)
    if any(count < min_size for count in cluster_counts[1:]):  # Start from 1 to skip empty clusters (count of 0)
      if obj == 'multiple':
        return [np.inf] * 3  # Return large values or NaNs for multi-objective case to avoid selection
      return np.inf
    fitness1 = multivariate_logrank_test(event_durations=data['Time'], groups=solution, event_observed=data['status']).test_statistic
    fitness2 = pairwise_logrank_test(event_durations=data['Time'], groups=solution, event_observed=data['status'],weightings="wilcoxon").test_statistic[0]
    fitness3 = pairwise_logrank_test(event_durations=data['Time'], groups=solution, event_observed=data['status'],weightings="tarone-ware").test_statistic[0]
    if obj == 'multiple':
      return [fitness1,fitness2,fitness3]
    return fitness1

  def callback_generation_pval(ga_instance):
    current_best_fitness = ga_instance.best_solution()[1]

    if previous_best_fitness is not None:
      fitness_change = abs(-np.log10(-previous_best_fitness) + np.log10(-current_best_fitness))
      if fitness_change < eps:
        no_change_count += 1
      else:
        no_change_count = 0
    else:
      no_change_count = 0

    previous_best_fitness = current_best_fitness
    print(f"Generation: {ga_instance.generations_completed}, Best Fitness: {-current_best_fitness}")
    if no_change_count == no_significant_change_generations:
      print(f"Terminating at Generation: {ga_instance.generations_completed}")
      return "stop"

  def callback_generation_logp(ga_instance):
    current_best_fitness = ga_instance.best_solution()[1]

    if previous_best_fitness is not None:
      fitness_change = abs(previous_best_fitness - current_best_fitness)
      if fitness_change < eps:
        no_change_count += 1
      else:
        no_change_count = 0
    else:
      no_change_count = 0

    previous_best_fitness = current_best_fitness
    print(f"Generation: {ga_instance.generations_completed}, Best Fitness: {current_best_fitness}")
    if no_change_count == no_significant_change_generations:
      print(f"Terminating at Generation: {ga_instance.generations_completed}")
      return "stop"

  def callback_generation_statistic(ga_instance):
    current_best_fitness = ga_instance.best_solution()[1]

    if previous_best_fitness is not None:
      fitness_change = abs(previous_best_fitness - current_best_fitness)
      if fitness_change < eps:
        no_change_count += 1
      else:
        no_change_count = 0
    else:
      no_change_count = 0

    previous_best_fitness = current_best_fitness
    print(f"Generation: {ga_instance.generations_completed}, Best Fitness: {current_best_fitness}")
    if no_change_count == no_significant_change_generations:
      print(f"Terminating at Generation: {ga_instance.generations_completed}")
      return "stop"

  def run(self):
    if 'fitness_func' not in self.ga_args:
      if self.optimize == 'p_val':
        self.ga_args['fitness_func'] = pygad_ga.fitness_pval
      elif self.optimize == 'log_p':
        self.ga_args['fitness_func'] = pygad_ga.fitness_logp
      elif self.optimize == 'statistic':
        self.ga_args['fitness_func'] = pygad_ga.fitness_statistic

    if ('stop_criteria' in self.ga_args or 'on_generation' in self.ga_args) and self.objective == 'single':
      if self.optimize == 'p_val':
        self.ga_args['on_generation'] = pygad_ga.callback_generation_pval
      elif self.optimize == 'log_p':
        self.ga_args['on_generation'] = pygad_ga.callback_generation_logp
      elif self.optimize == 'statistic':
        self.ga_args['on_generation'] = pygad_ga.callback_generation_statistic


    ga_instance = pygad.GA(**self.ga_args)
    ga_instance.run()
    print(f"Terminating at Generation: {ga_instance.generations_completed}")
    if self.res=="best_dist":
      final_population, fitness_scores = ga_instance.population,ga_instance.last_generation_fitness
      return final_population, fitness_scores
    elif self.res=="best":
      best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
      if self.optimize=="p_val":
        return best_solution, -(best_solution_fitness)
      else:
        print(f"Terminating at Generation: {ga_instance.generations_completed}")
        return best_solution,best_solution_fitness
