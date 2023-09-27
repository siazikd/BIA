import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import functions as fn
import core as core
import algorithms as alg


class Main:
    def __init__(self, algorithm: core.Algorithm = None, animation: bool = False):
        self.algorithm = algorithm
        self.animation = animation
        self.visualizer = core.Visualizer(algorithm=self.algorithm, results=[])

    def run(self):
        self.visualizer.Exec()
        print('Best result: %.3f' % self.algorithm.bestResult)
        pass


Main(
    animation=False, 
    algorithm=alg.BlindSearch(
            function=fn.Michalewicz, 
            limits=(0,3.14),
            iterations=1000
        )
    ).run()

