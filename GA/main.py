import random
import matplotlib.pyplot as plt

num_cities = 25
city_positions = [(random.uniform(0, 100), random.uniform(0, 100)) for i in range(num_cities)]

pop_size = 100
num_generations = 100

mutation_rate = 0.01
num_elites = 5

def fitness(individual):
    total_distance = 0
    for i in range(num_cities - 1):
        city1 = individual[i]
        city2 = individual[i+1]
        distance = ((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)**0.5         # Vzdálenost mezi dvěma městy je eukleidovská vzdálenost
        total_distance += distance
    return 1/total_distance 

population = [random.sample(range(num_cities), num_cities) for _ in range(pop_size)] # vytvoření populace
population = [individual + [individual[0]] for individual in population] # přidání prvního města na konec jedince

for generation in range(num_generations):
    fitness_scores = [fitness([city_positions[i] for i in individual]) for individual in population] # vytvoření seznamu fitness hodnot jedinců
    
    population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)] # seřazení populace podle fitness hodnot
    
    elites = population[:num_elites]  # vybrání nejlepších jedinců
    
    new_population = elites 
    
    while len(new_population) < pop_size:
        parent1 = random.choices(population, weights=fitness_scores)[0] # náhodný výběr jedince z populace s pravděpodobností závislou na jeho fitness hodnotě
        parent2 = random.choices(population, weights=fitness_scores)[0] 
        
        # Provedení křížení
        crossover_point = random.randint(1, num_cities-1) 
        child = parent1[:crossover_point] + [city for city in parent2 if city not in parent1[:crossover_point]] # první čast od parent1 a druhá část od parent2(neduplikují se města)
        
        # Mutace jedince s pravděpodobností mutation_rate
        if random.random() < mutation_rate: # náhodné číslo od 0 do 1
            mutation_point1 = random.randint(0, num_cities-1) # náhodný index města
            mutation_point2 = random.randint(0, num_cities-1) # náhodný index města
            child[mutation_point1], child[mutation_point2] = child[mutation_point2], child[mutation_point1] # prohození měst na indexech mutation_point1 a mutation_point2
            #print(child)
        
        child.append(child[0])
        new_population.append(child)
    
    # Nastavení populace na novou populaci
    population = new_population
    

print("Nejlepsi jedinec:", population[0])

# Vykreslení pozic měst
x = [city[0] for city in city_positions]
y = [city[1] for city in city_positions]
plt.scatter(x, y)

for i in range(num_generations):
    best_individual = population[0]
    x = [city_positions[city][0] for city in best_individual]
    y = [city_positions[city][1] for city in best_individual]
    plt.plot(x, y, alpha=(i+1)/num_generations, color='red')

plt.show()
