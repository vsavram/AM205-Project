#!/usr/bin/env python3

from autograd import numpy as np
from autograd import grad
from autograd import hessian
from autograd.misc.optimizers import adam, sgd
from autograd import scipy as sp
import autograd.numpy.random as npr
import pandas as pd
from scipy.optimize import minimize_scalar

# Perform steepest descent and iterate until the step size becomes small enough
def steepest_descent(objective_function, initial_W, min_step_size=10**(-8), max_iter=2000):
    
    current_iteration = 0
    step_size = 1

    # Define the starting point (the initial values for the weights and bias terms)
    W = initial_W
    
    # Define the gradient function
    gradient = grad(objective_function)
    
    while np.abs(step_size) > min_step_size and current_iteration < max_iter:
        
        previous_W = W
        
        # Determine the gradient of the objective function using autograd
        W_grad = gradient(np.array(previous_W))
        
        # Determine the learning rate for the update using a line search
        alpha = minimize_scalar(lambda alpha: objective_function(previous_W - alpha*W_grad))
        alpha = alpha.x
        
        # Update the values for x and y
        W = W - alpha*W_grad
        
        # Determine the step size
        delta_W = W - previous_W
        step_size = np.linalg.norm(delta_W)
        
        current_iteration = current_iteration + 1
    
    return W


# Perform Newton's method and iterate until the step size becomes small enough
def newton_method(objective_function, initial_W, min_step_size=10**(-8), max_iter=2000):
    
    current_iteration = 0
    step_size = 1
    
    # Define the starting point and learning rate
    W = initial_W
    
    # Define the gradient function
    gradient = grad(objective_function)
    # Define the Hessian function
    hessian_function = hessian(objective_function)
    
    while np.abs(step_size) > min_step_size and current_iteration < max_iter:
        
        previous_W = W
        
        # Compute the gradient
        W_grad = gradient(previous_W)
        
        # Compute the Hessian
        W_hessian = hessian_function(np.array(previous_W))
        
        # Solve the system of equations for the step size
        step = np.linalg.solve(W_hessian, -W_grad)
        # Update the values for W
        W = previous_W + step
        
        # Determine the step size
        step_size = np.linalg.norm(step)
        
        current_iteration = current_iteration + 1
    
    return W


# Perform the BFGS algorithm and iterate until the step size becomes small enough
def BFGS(objective_function, initial_W, min_step_size=10**(-8), max_iter=2000):
    
    current_iteration = 0
    step_size = 1
    beta = np.eye(len(initial_W))
    
    # Define the starting point and learning rate
    W = initial_W
    
    # Define the gradient function
    gradient = grad(objective_function)
    
    while np.abs(step_size) > min_step_size and current_iteration < max_iter:
        
        previous_W = W
        
        # Compute the gradient
        W_grad = gradient(previous_W)
        
        # Solve the system of equations for the step size
        step = np.linalg.solve(beta, -W_grad)
        # Update the values for W
        W = previous_W + step
        
        # Determine the step size
        step_size = np.linalg.norm(step)
        
        # Compute the gradient for the updated values and compute delta gradient
        updated_gradient = gradient(np.array(W))
        delta_gradient = updated_gradient - W_grad
        
        # Compute delta beta
        delta_gradient = delta_gradient.reshape(len(delta_gradient),1)
        step = step.reshape(len(step),1)
        delta_beta = np.dot(delta_gradient,delta_gradient.T)/np.dot(delta_gradient.T,step) - np.dot(np.dot(beta,step), np.dot(step.T,beta))/np.dot(np.dot(step.T,beta), step)
        
        # Update the value for beta
        beta = beta + delta_beta
        
        current_iteration = current_iteration + 1
    
    return W


# Perform the conjugate gradient method and iterate until the step size becomes small enough
def conjugate_gradient(objective_function, initial_W, min_step_size=10**(-8), max_iter=2000):
    
    current_iteration = 0
    step_size = 1
    
    # Derive a function that computes the gradient of the objective function
    gradient = grad(objective_function)
    
    # Compute the gradient
    W_grad = gradient(np.array(initial_W)) 
    # Define the starting point
    W = -W_grad
    
    # Set the initial value for s
    s = -W_grad
    
    while np.abs(step_size) > min_step_size and current_iteration < max_iter:
        
        previous_W = W
        W_grad_previous = W_grad
        
        # Determine the learning rate for the update using a line search
        alpha = minimize_scalar(lambda alpha: objective_function(previous_W - alpha*s))
        alpha = alpha.x
        
        # Update the values for x and y
        W = W + alpha*s
        
        # Compute the new gradient
        W_grad = gradient(np.array(W)) 
        # Reshape the gradient array
        W_grad = W_grad.reshape(len(W_grad),1)
        
        # Compute beta
        beta = np.dot(W_grad.T,W_grad)/np.dot(W_grad_previous.T,W_grad_previous)
        beta = beta[0][0]
        
        # Update the value for s
        s = -W_grad.flatten() + beta*s
        s = s[0]
        
        # Determine the step size
        delta_W = W - previous_W
        step_size = np.linalg.norm(delta_W)
        
        current_iteration = current_iteration + 1
    
    return W




def test_function(inputs, t=1.):
    return np.power(inputs[0],3) + 4*np.exp(inputs[1]) + 10*np.power(inputs[2],4)  

def test_function2(x, t=1.):
    return x[0]**2 + 3*x[0] + 1

def test_function3(inputs):
    return -np.exp(-(inputs[0]**2 + inputs[1]**2))


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


# Test steepest descent
solution = steepest_descent(test_function2, np.array([10.]), min_step_size=10**(-8), max_iter=2000)
print(solution)

# Test Newton's method
solution = newton_method(test_function3, np.array([10., 5.]), min_step_size=10**(-8), max_iter=2000)
print(solution)

# Test BFGS
solution = BFGS(test_function2, np.array([10.]), min_step_size=10**(-8), max_iter=2000)
print(solution)

# Test the conjugate gradient method
solution = conjugate_gradient(test_function3, np.array([10.,5.]), min_step_size=10**(-8), max_iter=2000)
print(solution)
