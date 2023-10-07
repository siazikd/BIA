import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import functions as fn
import core as core
import algorithms as alg


class Main:
    def __init__(self, algorithm: core.Algorithm = None, animation: bool = False, heatmap: bool = False):
        self.algorithm = algorithm
        self.animation = animation
        self.heapmap = heatmap
        self.visualizer = core.Visualizer(animation=self.animation, heatmap=self.heapmap, algorithm=self.algorithm, results=[])

    def run(self):
        self.visualizer.Exec()
        print('Best result: %.3f' % self.algorithm.bestResult)
        pass


Main(
    heatmap=True,
    animation=False, 
    algorithm=alg.SimulatedAnnealing(
            function=fn.Ackley, 
            limits=(-10, 10),
            iterations=100
        )
    ).run()

