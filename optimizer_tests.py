#!/usr/bin/env python3

from autograd import numpy as np
from autograd import grad
from autograd import hessian
from autograd.misc.optimizers import adam, sgd
from autograd import scipy as sp
import autograd.numpy.random as npr
import pandas as pd
from scipy.optimize import minimize_scalar

from optimizers import *


#---------------------------------------------------------------------------------------------------
# Test each optimizer on simple functions
#---------------------------------------------------------------------------------------------------


# Define function 1 - 3 inputs
def test_function(inputs, t=1.):
    return np.power(inputs[0],3) + 4*np.exp(inputs[1]) + 10*np.power(inputs[2],4)  

# Define function 2 - 1 input with a global minimum at -1.5
def test_function2(x, t=1.):
    return x[0]**2 + 3*x[0] + 1

# Define function 3 - 2 inputs with a global minimum at (0,0)
def test_function3(inputs):
    return inputs[0]**2 + inputs[1]**2


### Function 1 ###

# Test steepest descent
solution = steepest_descent(test_function, np.array([1.,3.,5.]), min_step_size=10**(-8), max_iter=2000)
print(solution)

# Test Newton's method
solution = newton_method(test_function, np.array([1.,3.,5.]), min_step_size=10**(-8), max_iter=2000)
print(solution)

# Test BFGS
solution = BFGS(test_function, np.array([1.,3.,5.]), min_step_size=10**(-8), max_iter=2000)
print(solution)

# Test the conjugate gradient method
solution = conjugate_gradient(test_function, np.array([1.,3.,5.]), min_step_size=10**(-8), max_iter=2000)
print(solution)


### Function 2 ###

# Test steepest descent
solution = steepest_descent(test_function2, np.array([10.]), min_step_size=10**(-8), max_iter=2000)
print(solution)

# Test Newton's method
solution = newton_method(test_function2, np.array([10.]), min_step_size=10**(-8), max_iter=2000)
print(solution)

# Test BFGS
solution = BFGS(test_function2, np.array([10.]), min_step_size=10**(-8), max_iter=2000)
print(solution)

# Test the conjugate gradient method
solution = conjugate_gradient(test_function2, np.array([10.]), min_step_size=10**(-8), max_iter=2000)
print(solution)


### Function 3 ###

# Test steepest descent
solution = steepest_descent(test_function3, np.array([12., 5.]), min_step_size=10**(-8), max_iter=2000)
print(solution)

# Test Newton's method
solution = newton_method(test_function3, np.array([10., 5.]), min_step_size=10**(-8), max_iter=2000)
print(solution)

# Test BFGS
solution = BFGS(test_function3, np.array([10., 5.]), min_step_size=10**(-8), max_iter=2000)
print(solution)

# Test the conjugate gradient method
solution = conjugate_gradient(test_function3, np.array([10.,5.]), min_step_size=10**(-8), max_iter=2000)
print(solution)