import numpy as np
import core as core

class BlindSearch(core.Algorithm):
    def __init__(self, function, limits, iterations):
        super().__init__(function, limits, iterations)
        
    def Exec(self):
        for _ in range(self.iterations): #pro vsechny iterace
            x, y = np.random.uniform(self.limits[0], self.limits[1], size=2) #vyber random bodu
            result = {'x': x, 'y': y, 'value': self.function(core.Point(x, y))} #vysledek
            if self.bestResult is None or self.bestResult > result['value']: #pokud je vysledek lepsi nez aktualni nejlepsi
                self.bestResult = result['value'] #aktualizuj nejlepsi
                yield result #vrat vysledek


class HillClimbing(core.Algorithm):
    def __init__(self, function, limits, iterations):
        super().__init__(function, limits, iterations)
        
    def Exec(self):
        x, y = np.random.uniform(self.limits[0], self.limits[1]), np.random.uniform(self.limits[0], self.limits[1]) #vyber random bodu

        while True:
            current_value = self.function(core.Point(x, y)) #hodnota v bode
            neighbors = [(x + self.limits[0] / 100, y), (x - self.limits[0] / 100, y), (x, y + self.limits[1] / 100), (x, y - self.limits[1] / 100)] #sousedi

            best_neighbor = min(neighbors, key=lambda pos: self.function(core.Point(pos[0], pos[1]))) #nejlepsi soused

            if self.bestResult is None or self.function(core.Point(best_neighbor[0], best_neighbor[1])) < self.bestResult: #pokud je nejlepsi soused lepsi nez aktualni nejlepsi
                self.bestResult = self.function(core.Point(best_neighbor[0], best_neighbor[1])) #aktualizuj nejlepsi
                x, y = best_neighbor #aktualizuj bod
                result = { #vrat vysledek
                    'x': x,
                    'y': y,
                    'value': self.bestResult
                }
                yield result
            else:
                break
            
            
class SimulatedAnnealing(core.Algorithm):
    def __init__(self, function, limits, iterations):
        super().__init__(function, limits, iterations)
        
    def Exec(self):
        temp = 500
        lastPos = None
        lastResult = 0.0

        while True:
            x = y = 0.0
            if lastPos is None:
                x = y = np.random.uniform(self.limits[0], self.limits[1]) # vyber random bodu
            else:
                x = lastPos[0] + np.random.uniform(self.limits[0] / 100, self.limits[1] / 100) # vyber random bodu
                y = lastPos[1] + np.random.uniform(self.limits[0] / 100, self.limits[1] / 100) # vyber random bodu
            lastResult = value = self.function(core.Point(x, y))

            if self.bestResult is None or value < self.bestResult:
                self.bestResult = value
                lastPos = (x, y)
                result = {
                    'x': x,
                    'y': y,
                    'value': value
                }
                yield result

            else:
                while temp > 0.1:
                    x = y = np.random.uniform(self.limits[0], self.limits[1])
                    value = self.function(core.Point(x, y))
                    delta = value - lastResult
                    temp = temp * 0.99
                    if value < lastResult + np.exp(-delta / temp):
                        lastPos = (x, y)
                        if value < lastResult:
                            print('Found better, temp: ' + str(temp))
                        else:
                            print('Found worse, temp: ' + str(temp))
                        result = {
                            'x': x,
                            'y': y,
                            'value': value
                        }
                        yield result
                        break

            if temp <= 0.1:
                print('Ran out of temperature')
                break
            
            
            
class DifferentialEvolution(core.Algorithm):
    def __init__(self, function, limits, NP, F, G, CR):
        super().__init__(function, limits, 0)
        self.NP = NP # počet jedinců v populaci
        self.F = F # faktor mutace
        self.G = G  # počet generaci
        self.CR = CR # pravděpodobnost křížení
        
    def Exec(self):    
        # Inicializace populace
        population = [core.Point(np.random.uniform(self.limits[0], self.limits[1]), np.random.uniform(self.limits[0], self.limits[1])) for _ in range(self.NP)]
                
        for generation in range(self.G):
            new_population = []
            
            for i in range(self.NP):
                indices = np.random.choice(self.NP, size=3, replace=False) # Výběr 3 náhodných jedinců z populace
                a, b, c = [population[idx] for idx in indices] # Výběr jedinců z populace
                
                mutant_vector = core.Point(a.x + self.F * (b.x - c.x), a.y + self.F * (b.y - c.y)) # Mutace pomocí diferenciálního vektoru (rand/bin)
                
                # Křížení 
                trial_vector = core.Point(mutant_vector.x if np.random.random() < self.CR else population[i].x,
                                        mutant_vector.y if np.random.random() < self.CR else population[i].y) 
                
                original_cost = self.function(population[i]) # Hodnocení původního jedince
                trial_cost = self.function(trial_vector) # Hodnocení nového jedince
                
                if trial_cost < original_cost: # Pokud je nový jedinec lepší, nahraď původní jedinec novým
                    yield {'x': trial_vector.x, 'y': trial_vector.y, 'value': trial_cost}
                    new_population.append(trial_vector)
                else:
                    yield {'x': population[i].x, 'y': population[i].y, 'value': original_cost}
                    new_population.append(population[i]) # Pokud je původní jedinec lepší, ponech ho v populaci
            
            population = new_population  # Nahrazení staré populace novou
        
        best_individual = min(population, key=lambda x: self.function(x)) # Výběr nejlepšího jedince
        best_cost = self.function(best_individual) # Hodnocení nejlepšího jedince
        print(f"Nejlepší řešení x: {best_individual.x}, y: {best_individual.y} s hodnotou {best_cost}") 
        yield {'x': best_individual.x, 'y': best_individual.y, 'value': best_cost} 


 

class ParticleSwarmOptimization(core.Algorithm):
    def __init__(self, function, limits, population_size = 15, M_max=50, c1=2, c2=2, w_min=0.4, w_max=0.9):
        super().__init__(function, limits, 0)
        self.population_size = population_size # počet jedinců v populaci
        self.M_max = M_max # počet generaci
        self.c1 = c1 # koeficienty
        self.c2 = c2 # koeficienty
        self.w_min = w_min # setrvačnost
        self.w_max = w_max # setrvačnost
        
    def Exec(self):
        # Inicializace populace
        population = [core.Point(np.random.uniform(self.limits[0], self.limits[1]), np.random.uniform(self.limits[0], self.limits[1])) for _ in range(self.population_size)]
        velocities = [core.Point(np.random.uniform(-1, 1), np.random.uniform(0, 1)) for _ in range(self.population_size)] # Inicializace rychlostí
        personal_bests = population.copy()  # Inicializace osobních nejlepších řešení
        global_best = min(personal_bests, key=lambda x: self.function(x)) # Výběr nejlepšího jedince
        
        # Inicializace proměnných
        M = 0
        w = self.w_max
        
        while M < self.M_max:
            for i in range(self.population_size):
                # Výpočet nové rychlosti
                r1, r2 = np.random.uniform(0, 1, size=2)
                velocities[i].x = w * velocities[i].x + self.c1 * r1 * (personal_bests[i].x - population[i].x) + self.c2 * r2 * (global_best.x - population[i].x)
                velocities[i].y = w * velocities[i].y + self.c1 * r1 * (personal_bests[i].y - population[i].y) + self.c2 * r2 * (global_best.y - population[i].y)
                
                # Omezení rychlosti
                if velocities[i].x > self.limits[1]:
                    velocities[i].x = 1
                elif velocities[i].x < self.limits[0]:
                    velocities[i].x = -1
                    
                if velocities[i].y > self.limits[1]:
                    velocities[i].y = 1
                elif velocities[i].y < self.limits[0]:
                    velocities[i].y = -1
                
                # Výpočet nové pozice
                population[i].x += velocities[i].x
                population[i].y += velocities[i].y
                
                # Omezení pozice
                if population[i].x > self.limits[1]:
                    population[i].x = self.limits[1]
                elif population[i].x < self.limits[0]:
                    population[i].x = self.limits[0]
                    
                if population[i].y > self.limits[1]:
                    population[i].y = self.limits[1]
                elif population[i].y < self.limits[0]:
                    population[i].y = self.limits[0]
                
                # Aktualizace osobního nejlepšího řešení
                if self.function(population[i]) < self.function(personal_bests[i]):
                    personal_bests[i] = population[i]
                
                # Aktualizace globálního nejlepšího řešení
                if self.function(personal_bests[i]) < self.function(global_best):
                    global_best = personal_bests[i]
                
                # Výpočet nové hodnoty setrvačnosti
                w = self.w_max - (self.w_max - self.w_min) * M / self.M_max
                
                # Výpočet fitness hodnoty
                fitness = self.function(global_best)
                
                # Výstup
                yield {'x': global_best.x, 'y': global_best.y, 'value': fitness}
                
            M += 1
