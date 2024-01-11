#Ventilation Demand
    # type1:luorescent/compact fluorescent (CFL)
    # type2: Balanced ventilation systems
    # type3: Demand control ventilation technology + sensor
import csv
import json
import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga
import fittness_function as ft
import random

def sort_population_by_fitness(population):
    return sorted(population, key=apply_fitness_function)

def crossover(parent1, parent2):
    # Select a random index for crossover
    crossover_point = random.randint(0, len(parent1) - 1)

    # Perform crossover
    offspring = parent1[:crossover_point] + parent2[crossover_point:]

    return offspring


def mutate(individual, mutation_rate):
    lighting_demand_code = ['type1','type2','type3'] 
    r_wall_code = ['R15','R19','R30']
    window_size_code = [4836,4872,5636,5672,6436,6472]
    u_window_code = [5.8,3.7,1]
    air_freshing_code = (5,10) # cfm/person
    r_roof_code = ['R15','R50']
    plug_load_code = (2,5) #watts/person
    mutated = individual.copy()  # Create a copy of the parent list
    for i in range(len(mutated)):
        # Check if mutation should occur for this element
        if random.random() < mutation_rate:
            # Generate a new random value for mutation
            if i==0:
                mutated[i] = random.choice(lighting_demand_code)
            elif i==1:
                mutated[i] = random.choice(r_wall_code)
            elif i==2:
                mutated[i] = random.choice(window_size_code)
            elif i==3:
                mutated[i] = random.choice(u_window_code)
            elif i==4:
                mutated[i] = random.choice(r_wall_code)
    return mutated


def make_next_generation(previous_population):
    next_generation = []
    sorted_by_fitness_population = sort_population_by_fitness(previous_population)
    population_size = len(previous_population)
    fitness_sum = sum(apply_fitness_function(individual) for individual in population)

    for i in range(population_size):
        first_choice = choice_by_roulette(sorted_by_fitness_population, fitness_sum)
        second_choice = choice_by_roulette(sorted_by_fitness_population, fitness_sum)

        individual = crossover(first_choice, second_choice)
        individual = mutate(individual, 0.15)
        next_generation.append(individual)

    return next_generation

def apply_fitness_function(chromosome):
    r_wall_score = {'R15': 3,'R19':6,'R30':10}
    lighting_demand_score = {'type1': 3,'type2':6,'type3':10} 
    window_size_score = {'4836':10,'4872':8,'5636':6,'5672':4,'6436':2,'6472':1}
    u_window_score = {'5.8':3,'3.7':6,'1':9}
    air_freshing_score = {'5':10,'10':5} # cfm/person
    r_roof_score = {'R15':5,'R50':10}
    plug_load_score = {'2':10,'5':5} #watts/person
    genes = [lighting_demand_score[chromosome[0]],
             r_wall_score[chromosome[1]],
             window_size_score[str(chromosome[2])],
             u_window_score[str(chromosome[3])],
             r_wall_score[chromosome[4]]]
    return np.sum(genes)
def choice_by_roulette(sorted_population, fitness_sum):
    offset = 0
    normalized_fitness_sum = fitness_sum

    lowest_fitness = apply_fitness_function(sorted_population[0])
    # lowest_fitness = ft.calculate_fitness(data)
    if lowest_fitness < 0:
        offset = -lowest_fitness
        normalized_fitness_sum += offset * len(sorted_population)

    draw = random.uniform(0, 1)

    accumulated = 0
    for individual in sorted_population:
        fitness = apply_fitness_function(individual) + offset
        probability = fitness / normalized_fitness_sum
        accumulated += probability

        if draw <= accumulated:
            return individual
def save_to_csv(file_path, objects):
    # data = json.dumps(json_data)
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in objects:
            writer.writerow(row)
def generate_population(size):
    # lower_x_boundary, upper_x_boundary = x_boundaries
    # lower_y_boundary, upper_y_boundary = y_boundaries
    population = []
    r_wall_code = ['R15','R19','R30']
    lighting_demand_code = ['type1','type2','type3'] 
    window_size_code = [4836,4872,5636,5672,6436,6472]
    u_window_code = [5.8,3.7,1]
    air_freshing_code = (5,10) # cfm/person
    r_roof_code = ['R15','R50']
    plug_load_code = (2,5) #watts/person
    for i in range(size):
        chromosome = {
            "lighting_demand_156": random.choice(lighting_demand_code),
            "r_wall_156": random.choice(r_wall_code),
            "window_size_156": random.choice(window_size_code),
            "u_window_340":random.choice(u_window_code),
            "r_wall_neighbour_326": random.choice(r_wall_code)
        }
        chromosome_list = (list(chromosome.values()))
        # ft_value = apply_fitness_function(chromosome)
        # chromosome_list.extend([ft_value])
        # population.append(chromosome_list)
        population.append(chromosome_list)
       
    return population
if __name__ == "__main__":
    file_path = r'D:/Implementation/GA/last_population.csv'
    generations = 120
    population = generate_population(50)
    best_indivduals =[]
    
    i = 1
while True:
    print(f"🧬 GENERATION {i}")
    best_individual = []
    # for individual in population:
    #     print(individual, apply_fitness_function(individual))

    if i == generations:
        break
    i += 1
    population = make_next_generation(population)
    # best_individual = sort_population_by_fitness(population)[-1]
    # fit_value = apply_fitness_function(best_individual)
    # indivi_fit = best_individual + [fit_value]
    # best_indivduals.append(indivi_fit)
# best_individual = sort_population_by_fitness(population)[-1]
print("\n🔬 FINAL RESULT")
# print(best_individual, apply_fitness_function(best_individual))
for item in population:
    fit_value = apply_fitness_function(item)
    indivi_fit = item + [fit_value]
    best_indivduals.append(indivi_fit)
save_to_csv(file_path,best_indivduals)
    # print('test')  