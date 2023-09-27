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