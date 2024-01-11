import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga
import fittness_function as ft


def f(X):
    return np.sum(X)

if __name__ == "__main__":
    
    var_boundary=np.array([[0,10]]*3)
    model=ga(function=f,dimension=3,variable_type='int',variable_boundaries=var_boundary)

    v = model.run() 
    
    convergence = model.report
    solu11tion= model.result 
    print('test')  