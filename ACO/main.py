import numpy as np
import matplotlib.pyplot as plt

class AntColonyOptimization():
    def __init__(self, cities, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        self.cities = cities # souřadnice měst
        self.distances = self.create_matrix_distances() # matice vzdáleností mezi městy
        self.pheromone = np.ones(self.distances.shape) / len(self.distances) # matice feromonů
        self.all_inds = range(len(self.distances)) # indexy měst
        self.n_ants = n_ants # počet mravenců
        self.n_best = n_best # počet nejlepších cest
        self.n_iterations = n_iterations # počet iterací
        self.decay = decay # úbytek feromonů
        self.alpha = alpha # vliv feromonů
        self.beta = beta # vliv vzdálenosti
        self.fig, self.ax = plt.subplots() 
        self.lines = []

    def create_matrix_distances(self): # matice vzdáleností mezi městy
        matrix = np.zeros((len(self.cities), len(self.cities))) 
        for i in range(len(self.cities)):
            for j in range(len(self.cities)):
                if i != j:
                    matrix[i][j] = ((self.cities[i][0] - self.cities[j][0]) ** 2 +
                                    (self.cities[i][1] - self.cities[j][1]) ** 2) ** 0.5
                else:
                    matrix[i][j] = np.inf

        return matrix

    def run(self):
        shortest_path = None
        all_time_shortest_path = (None, np.inf) 
        for i in range(self.n_iterations): 
            all_paths = self.gen_all_paths() # všechny cesty
            shortest_path = min(all_paths, key=lambda x: x[1]) # nejkratší cesta
            if shortest_path[1] < all_time_shortest_path[1]: # pokud je nejkratší cesta kratší než nejkratší cesta všech iterací
                print(shortest_path)
                all_time_shortest_path = shortest_path
                self.ax.clear()
                self.plot_paths(all_time_shortest_path[0])
                plt.pause(1)

            self.pheromone *= (1 - self.decay)  # Úbytek feromonů
            self.spread_pheromone(all_paths, self.n_best, shortest_path=shortest_path) # Rozšíření feromonů

        plt.show()

    def spread_pheromone(self, all_paths, n_best, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1]) # Seřazení cest podle délky
        for path, dist in sorted_paths[:n_best]: # Pro nejlepší cesty
            for move in path: # Pro každý krok
                self.pheromone[move] += 1.0 / self.distances[move] # Přidání feromonů

    def gen_path_dist(self, path):
        total_dist = 0 # Celková vzdálenost
        for ele in path:  
            total_dist += self.distances[ele]
        return total_dist

    def gen_all_paths(self):
        all_paths = [] # Všechny cesty
        for i in range(self.n_ants): 
            path = self.gen_path(i) # Cesta mravence
            all_paths.append((path, self.gen_path_dist(path))) # Přidání cesty a její vzdálenosti
        return all_paths

    def gen_path(self, ant_index):
        path = []
        visited = set() # Navštívená města
        start = np.random.choice(self.all_inds) # Náhodný výběr města
        visited.add(start) # Přidání města do navštívených
        prev = start 
        for i in range(len(self.distances) - 1): # Pro všechna města
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited) # Výběr dalšího města
            path.append((prev, move)) # Přidání kroku do cesty
            prev = move # Nastavení předchozího města
            visited.add(move) # Přidání města do navštívených
        path.append((prev, start))  ## Přidání posledního kroku
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone) 
        pheromone[list(visited)] = 0 # Nastavení feromonů na 0 pro navštívená města
        row = pheromone ** self.alpha * (1.0 / dist) ** self.beta # Výpočet pravděpodobnosti
        norm_row = row / row.sum() # Normalizace pravděpodobnosti
        move = np.random.choice(self.all_inds, 1, p=norm_row)[0] # Výběr dalšího města
        return move

    def plot_paths(self, path):
        path_coords = np.array([self.cities[i] for i, _ in path]) 
        path_line, = self.ax.plot(path_coords[:, 0], path_coords[:, 1], marker='o', linestyle='-', linewidth=1, markersize=5) 
        
        # Connect the last point to the starting point to close the loop
        start_point = path_coords[0]
        path_line, = self.ax.plot([path_coords[-1, 0], start_point[0]], [path_coords[-1, 1], start_point[1]],
                                marker='o', linestyle='-', linewidth=1, markersize=5, color='blue')
        
        self.lines.append(path_line)


# Example usage:
cities = [(np.random.uniform(0, 10), np.random.uniform(0, 10)) for i in range(35)]
ant_colony = AntColonyOptimization(cities, n_ants=5, n_best=2, n_iterations=1000, decay=0.1, alpha=1, beta=2)
ant_colony.run()
print('Done')