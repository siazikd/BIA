import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import functions as fn
import core as core
import algorithms as alg
import pandas as pd


class Main:
    def __init__(self, algorithm: core.Algorithm = None, animation: bool = False, heatmap: bool = False):
        self.algorithm = algorithm
        self.animation = animation
        self.heapmap = heatmap
        self.visualizer = core.Visualizer(animation=self.animation, heatmap=self.heapmap, algorithm=self.algorithm, results=[])

    def run(self):
        self.visualizer.Exec()
        #print('Best result: ', self.algorithm.bestResult)
        return self.algorithm.bestResult['value']
        
algs = [
    Main(
    algorithm=alg.TeachingLearning(
            function=fn.Sphere, 
            limits=(-5.12, 5.12),
            M_max=30,
            population_size=30,
    )
    ),
    Main(
        algorithm=alg.DifferentialEvolution(
            function=fn.Sphere, 
            limits=(-5.12, 5.12),
            NP=30,
            G=30
        )
    ),
    Main(
        algorithm=alg.ParticleSwarmOptimization(
            function=fn.Sphere, 
            limits=(-5.12, 5.12),
            population_size=30,
        )
    ),
    Main(
        algorithm=alg.SOMA(
            function=fn.Sphere, 
            limits=(-5.12, 5.12),
            population_size=30,
        )
    ),
    Main(
        algorithm=alg.Firefly(
            function=fn.Sphere, 
            limits=(-5.12, 5.12),
            population_size=30,
            M_max=30,
        )
    ),
]


functions = [
    fn.Sphere,
    fn.Rosenbrock,
    fn.Rastrigin,
    fn.Griewank,
    fn.Ackley,
    fn.Schwefel,
    fn.Levy,
    fn.Michalewicz,
    fn.Zakharov 
]



limits = [
    (-5.12, 5.12),
    (-2.048, 2.048),
    (-5.12, 5.12),
    (-600, 600),
    (-32.768, 32.768),
    (-500, 500),
    (-10, 10),
    (0, np.pi),
    (-5, 10)
]

np.random.seed(2)
for i in range(len(functions)):
    name = functions[i].__name__
    fn = functions[i]
    lm = limits[i]
    results = {}
    #print(f'Function: {name}')
    #print(f'Limits: {lm[0]} - {lm[1]}')
    for alg in algs:
        alg.algorithm.function = fn
        alg.algorithm.limits = lm
        results[alg.algorithm.getClassName()] = {}
        for i in range(1, 31):  # Pro experimenty 1 až 30
            results[alg.algorithm.getClassName()][f'Exp {i}'] = (alg.run()) 
            

    summary = {}
    for algorithm in results.keys():
        for i in range(1, 31):
            if f'Exp {i}' not in summary:
                summary[f'Exp {i}'] = {}
            summary[f'Exp {i}'][algorithm] = '{:.30f}'.format(results[algorithm][f'Exp {i}'])

    summary['Mean'] = {}
    summary['Std'] = {}
    for algorithm in results.keys():
        summary['Mean'][algorithm] = '{:.30f}'.format(np.mean(list(results[algorithm].values())))
        summary['Std'][algorithm] = '{:.30f}'.format(np.std(list(results[algorithm].values())))

    df = pd.DataFrame.from_dict(summary, orient='index')

    df.to_csv(f'{name}.csv', sep=';', decimal='.')
    
"""
Main(
    animation=True,
    algorithm=alg.TeachingLearning(
        function=fn.Sphere, 
        limits=(0, np.pi),
        population_size=30,
        M_max=30,
    )
).run()
"""
