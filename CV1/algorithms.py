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