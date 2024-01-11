import random
import math
import pandas as pd
import fittness_function as ft
from pathlib import Path 



def generate_population(size, x_boundaries, y_boundaries):
    lower_x_boundary, upper_x_boundary = x_boundaries
    lower_y_boundary, upper_y_boundary = y_boundaries

    population = []
    r_list = ['R18','R22','R25']
    for i in range(size):
        individual = {
            "x": random.uniform(lower_x_boundary, upper_x_boundary),
            "y": random.uniform(lower_y_boundary, upper_y_boundary),
            # "r": random.choice(r_list)
        }
        population.append(individual)
        
    return population

def apply_function(individual):
    x = individual["x"]
    y = individual["y"]
    return math.sin(math.sqrt(x ** 2 + y ** 2))

def choice_by_roulette(sorted_population, fitness_sum):
    offset = 0
    normalized_fitness_sum = fitness_sum

    lowest_fitness = apply_function(sorted_population[0])
    # lowest_fitness = ft.calculate_fitness(data)
    if lowest_fitness < 0:
        offset = -lowest_fitness
        normalized_fitness_sum += offset * len(sorted_population)

    draw = random.uniform(0, 1)

    accumulated = 0
    for individual in sorted_population:
        fitness = apply_function(individual) + offset
        probability = fitness / normalized_fitness_sum
        accumulated += probability

        if draw <= accumulated:
            return individual

def sort_population_by_fitness(population):
    return sorted(population, key=apply_function)

def crossover(individual_a, individual_b):
    xa = individual_a["x"]
    ya = individual_a["y"]

    xb = individual_b["x"]
    yb = individual_b["y"]

    return {"x": (xa + xb) / 2, "y": (ya + yb) / 2}

def mutate(individual):
    next_x = individual["x"] + random.uniform(-0.05, 0.05)
    next_y = individual["y"] + random.uniform(-0.05, 0.05)

    lower_boundary, upper_boundary = (-4, 4)

    # Guarantee we keep inside boundaries
    next_x = min(max(next_x, lower_boundary), upper_boundary)
    next_y = min(max(next_y, lower_boundary), upper_boundary)

    return {"x": next_x, "y": next_y}

def make_next_generation(previous_population):
    next_generation = []
    sorted_by_fitness_population = sort_population_by_fitness(previous_population)
    population_size = len(previous_population)
    fitness_sum = sum(apply_function(individual) for individual in population)

    for i in range(population_size):
        first_choice = choice_by_roulette(sorted_by_fitness_population, fitness_sum)
        second_choice = choice_by_roulette(sorted_by_fitness_population, fitness_sum)

        individual = crossover(first_choice, second_choice)
        individual = mutate(individual)
        next_generation.append(individual)

    return next_generation

if __name__ == "__main__":
    # https://hackernoon.com/genetic-algorithms-explained-a-python-implementation-sd4w374i
    generations = 500
    columns = ['generations','fitness']
   
    population = generate_population(size=10, x_boundaries=(-4, 4), y_boundaries=(-4, 4))
    filepath = Path('D:/Implementation/GA/generations.csv')
    
    import random
    for i in range(5):
        number_list = [111, 222, 333, 444, 555]
        # random item from list
        print(random.choice(number_list))
    
    i = 1
    solutions =[]
    # for generations in range(50, 500,50):
    while True:
        # print(f"🧬 GENERATION {i}")

        # for individual in population:
        #     print(individual)

        if i == generations:
            break

        i += 1

        # Make next generation...
        population = make_next_generation(population)
        
        best_individual = sort_population_by_fitness(population)[-1]
        print("\n🔬 FINAL RESULT")
        fitness_value = apply_function(best_individual)
        print(best_individual, fitness_value)
        solutions.append([i,fitness_value])

df = pd.DataFrame(solutions, columns = columns)
df.to_csv(filepath) 