import numpy as np
import core as core

class BlindSearch(core.Algorithm):
    def __init__(self, function, limits, iterations):
        super().__init__(function, limits, iterations)
        
    def Exec(self):
        for _ in range(self.iterations):
            x, y = np.random.uniform(self.limits[0], self.limits[1], size=2)
            result = {'x': x, 'y': y, 'value': self.function(core.Point(x, y))}
            if self.bestResult is None or self.bestResult > result['value']:
                self.bestResult = result['value']
                yield result


class HillClimbing(core.Algorithm):
    def __init__(self, function, limits, iterations):
        super().__init__(function, limits, iterations)
        
    def Exec(self):
        stepSize = 0.005
        x, y = np.random.uniform(self.limits[0], self.limits[1]), np.random.uniform(self.limits[0], self.limits[1])

        while True:
            current_value = self.function(core.Point(x, y))
            neighbors = [(x + self.limits[0] / 100, y), (x - self.limits[0] / 100, y), (x, y + self.limits[1] / 100), (x, y - self.limits[1] / 100)]

            best_neighbor = min(neighbors, key=lambda pos: self.function(core.Point(pos[0], pos[1])))

            if self.bestResult is None or self.function(core.Point(best_neighbor[0], best_neighbor[1])) < self.bestResult:
                self.bestResult = self.function(core.Point(best_neighbor[0], best_neighbor[1]))
                x, y = best_neighbor
                result = {
                    'x': x,
                    'y': y,
                    'value': self.bestResult
                }
                yield result
            else:
                break