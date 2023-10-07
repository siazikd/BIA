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