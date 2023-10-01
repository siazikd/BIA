import numpy as np
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class Algorithm:
    def __init__(self, function : callable, limits : tuple, iterations : int):
        self.function = function
        self.limits = limits
        self.iterations = iterations
        self.bestResult = None
        
    def Exec(self):
        pass
        
        
    def getFunctionName(self):
        if callable(self.function):
            return self.function.__name__
        else:
            return "N/A"
        
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        

class Visualizer:
    def __init__(self, algorithm: Algorithm, results: list):
        self.algorithm = algorithm
        self.results = results
    
    def __drawPlot(self):
        N = 100
        X = np.linspace(self.algorithm.limits[0], self.algorithm.limits[1], N) 
        Y = np.linspace(self.algorithm.limits[0], self.algorithm.limits[1], N)
        X, Y = np.meshgrid(X, Y) #vytvori matici bodu
        Z = self.algorithm.function(Point(X, Y)) #vyhodnoti funkci v bodech

        resX = [result['x'] for result in self.results] #vytvori pole x-ovych souradnic
        resY = [result['y'] for result in self.results] #vytvori pole y-ovych souradnic
        resZ = [result['value'] for result in self.results] #vytvori pole hodnot

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='jet', linewidth=0, antialiased=False, alpha=0.2 ) #vykresli plochu
        ax.scatter(resX, resY, resZ, c='r', marker='o', s=50, depthshade=False ) #vykresli body

        plt.ylabel(self.algorithm.getFunctionName()) #popisky os
        plt.title(self.algorithm.getFunctionName()) #titulek
        
        plt.show()


    def Exec(self):
        for result in self.algorithm.Exec(): #pro vsechny vysledky
            self.results.append(result) #pridej vysledek
            print('%11s:[%.2f, %.2f]: %.3f' % ( 
                self.algorithm.getFunctionName(), 
                result['x'], result['y'], result['value'])
        
            )
        self.__drawPlot() #vykresli graf
