import math
import random
import matplotlib.pyplot as plt


def objective1(r, h):  # minimalizovat boční plochu S
    s = math.sqrt(r * r + h * h)
    return math.pi * r * s

def objective2(r, h):  # minimalizovat celkovou plochu T
    s = math.sqrt(r * r + h * h)
    return math.pi * r * (r + s)

def generate_initial_solution(): # Funkce, která nám vygeneruje náhodné řešení
    r = random.uniform(0, 10)
    h = random.uniform(0, 20)
    return r, h



def index_locator(a,list): # Funkce, která nám vrátí index prvku a v listu
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1



def sort_by_values(list1, values): # Funkce, která nám seřadí list1 podle hodnot ve values
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_locator(min(values),values) in list1:
            sorted_list.append(index_locator(min(values),values))
        values[index_locator(min(values),values)] = math.inf
    return sorted_list

def crowding_distance(values1, values2, front): # Funkce, která nám vrátí crowding distance, což je vzdálenost mezi dvěma body
    distance = [0 for i in range(0,len(front))] # Vytvoření listu o velikosti fronty
    sorted1 = sort_by_values(front, values1[:]) # Seřazení fronty podle hodnot ve values1
    sorted2 = sort_by_values(front, values2[:]) # Seřazení fronty podle hodnot ve values2
    distance[0] = 9999999999999999
    distance[len(front) - 1] = 9999999999999999
    for k in range(1,len(front)-1): 
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance


def crossover(a, b):
    r = random.random()
    if r > 0.5:
        new_r = (a[0] + b[0]) / 2  # Crossover on 'r'
        new_h = (a[1] + b[1]) / 2  # Crossover on 'h'
        return mutation((new_r, new_h))
    else:
        new_r = (a[0] - b[0]) / 2  # Crossover on 'r'
        new_h = (a[1] - b[1]) / 2  # Crossover on 'h'
        return mutation((new_r, new_h))
    
    

def mutation(solution):
    r, h = solution
    mutated_r = random.uniform(max(0, r - 0.1), min(10, r + 0.1))  
    mutated_h = random.uniform(max(0, h - 0.1), min(20, h + 0.1))  
    return mutated_r, mutated_h

def non_dominated_sorting_algorithm(values1, values2):
    S=[[] for i in range(0,len(values1))] # v tomto listu budou indexy jedinců, kteří jsou lepší než p
    front = [[]]  
    n=[0 for i in range(0,len(values1))]  
    rank = [0 for i in range(0, len(values1))] 

    for p in range(0,len(values1)): 
        S[p]=[] 
        n[p]=0 
        for q in range(0, len(values1)): # Porovnání hodnot values1 a values2
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:  # Pokud q není v S[p], tak se přidá do S[p]
                    S[p].append(q) 
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1 # Počítání počtu jedinců, které jsou lepší než p
        if n[p]==0: # Pokud je počet jedinců, které jsou lepší než p roven 0, tak se přidá do fronty 0
            rank[p] = 0 
            if p not in front[0]: # Pokud p není v frontě 0, tak se přidá do fronty 0
                front[0].append(p) 
    i = 0
    while(front[i] != []): # Dokud front[i] není prázdný
        Q=[] 
        for p in front[i]: # Pro každého jedince v frontě[i]
            for q in S[p]: # Pro každého jedince v S[p]
                n[q] =n[q] - 1 # Počítání počtu jedinců, které jsou lepší než q
                if( n[q]==0): # Pokud je počet jedinců, které jsou lepší než q roven 0, tak se přidá do fronty i+1
                    rank[q]=i+1 # Nastavení ranku jedince q na i+1
                    if q not in Q: # Pokud q není v Q, tak se přidá do Q
                        Q.append(q) 
        i = i+1
        front.append(Q)
    del front[len(front)-1]
    return front


def nsga2(population, max_gen):
    gen_no = 0
    solution = [generate_initial_solution() for _ in range(population)]

    while gen_no < max_gen:
        objective1_values = [objective1(*sol) for sol in solution]
        objective2_values = [objective2(*sol) for sol in solution]
        non_dominated_sorted_solution = non_dominated_sorting_algorithm(objective1_values[:],objective2_values[:])
        print('Best Front for Generation:', gen_no)
        for values in non_dominated_sorted_solution[0]:
            print(round(objective1_values[values], 3), round(objective2_values[values], 3), end=" ")
        print("\n")
        crowding_distance_values=[]
        for i in range(0,len(non_dominated_sorted_solution)):
            crowding_distance_values.append(crowding_distance(objective1_values[:],objective2_values[:],non_dominated_sorted_solution[i][:]))
        solution2 = solution[:]
        
        while(len(solution2)!=2*population):
            a1 = random.randint(0,population-1)
            b1 = random.randint(0,population-1)
            solution2.append(crossover(solution[a1],solution[b1]))
        objective1_values2 = [objective1(*solution2[i]) for i in range(0, 2 * population)]
        objective2_values2 = [objective2(*solution2[i]) for i in range(0, 2 * population)]
        non_dominated_sorted_solution2 = non_dominated_sorting_algorithm(objective1_values2[:],objective2_values2[:])
        crowding_distance_values2=[]
        for i in range(0,len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(crowding_distance(objective1_values2[:],objective2_values2[:],non_dominated_sorted_solution2[i][:]))
        new_solution= []
        for i in range(0,len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2_1 = [index_locator(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
            front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
            front.reverse()
            for value in front:
                new_solution.append(value)
                if(len(new_solution)==population):
                    break
            if (len(new_solution) == population):
                break
        solution = [solution2[i] for i in new_solution]
        gen_no = gen_no + 1
    return [objective1_values, objective2_values]



# Vytvoření populace
population = 25
max_gen = 500
results = nsga2(population, max_gen)
plt.scatter(results[0], results[1])
plt.xlabel('Lateral Surface Area (S)')
plt.ylabel('Total Area (T)')
plt.title('Pareto Front for Cone Optimization')
plt.show()