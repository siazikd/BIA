import numpy as np
import core as core
import math as Math

def Ackley(point: core.Point) -> float:
    x = point.x
    y = point.y
    return -20 * np.exp(-0.2 * np.sqrt(x**2 + y**2)) - np.exp((np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)) / 2) + 20 + np.exp(1)

def Rastrigin(point: core.Point) -> float:
    x = point.x
    y = point.y
    return 10 * 2 + (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y))

def Rosenbrock(point: core.Point) -> float:
    x = point.x
    y = point.y
    return 100 * (y - x**2)**2 + (1 - x)**2

def Griewank(point: core.Point) -> float:
    x = point.x
    y = point.y
    return 1 + (x**2 + y**2) / 4000 - np.cos(x) * np.cos(y / np.sqrt(2))

def Schwefel(point: core.Point) -> float:
    x = point.x
    y = point.y
    return 418.9829 * 2 - (x * np.sin(np.sqrt(np.abs(x))) + y * np.sin(np.sqrt(np.abs(y))))

def Sphere(point: core.Point) -> float:
    x = point.x
    y = point.y
    return x**2 + y**2

def Levy(point: core.Point) -> float:
    x = point.x
    y = point.y
    w1 = 1 + (x - 1) / 4
    w2 = 1 + (y - 1) / 4
    term1 = (np.sin(np.pi * w1))**2
    term2 = (w1 - 1)**2 * (1 + (np.sin(2 * np.pi * w1))**2)
    term3 = (w2 - 1)**2 * (1 + (np.sin(2 * np.pi * w2))**2)
    return term1 + term2 + term3

def Michalewicz(point: core.Point, m: int = 20) -> float:
    x = point.x
    y = point.y
    result = -np.sin(x) * np.sin(x**2 / np.pi) ** m - np.sin(y) * np.sin(2 * y**2 / np.pi) ** m
    return result

def Zakharov(point: core.Point) -> float:
    x = point.x
    y = point.y
    term1 = x**2 + y**2
    term2 = 0.5 * (x**2 + y**2)**2
    term3 = 0.5 * (x**2 + y**2)**4
    return term1 + term2 + term3
