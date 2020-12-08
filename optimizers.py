#!/usr/bin/env python3

from autograd import numpy as np
from autograd import grad
from autograd import hessian
from autograd.misc.optimizers import adam, sgd
from autograd import scipy as sp
import autograd.numpy.random as npr
import pandas as pd
import numpy

# Perform steepest descent and iterate until the step size becomes small enough
def steepest_descent(objective_function, initial_W, min_step_size=10**(-8), max_iter=2000):
    
    current_iteration = 0
    step_size = 1

    # Define the starting point (the initial values for the weights and bias terms)
    W = initial_W
    
    while np.abs(step_size) > min_step_size and current_iteration < max_iter:
        
        previous_W = W
        
        # Determine the gradient of the objective function using autograd
        gradient = grad(objective_function)
        W_grad = gradient(previous_W)
        
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
def newton_method(objective_function, initial_W, min_step_size=10**(-8), max_iter):
    
    current_iteration = 0
    step_size = 1
    
    # Define the starting point and learning rate
    W = initial_W
    
    while np.abs(step_size) > min_step_size and current_iteration < max_iter:
        
        previous_W = W
        
        # Compute the gradient
        gradient = grad(objective_function)
        W_grad = gradient(previous_W, 1)
        
        # Compute the Hessian
        hessian_function = hessian(objective_function)
        W_hessian = hessian_function(previous_W, 1)
        
        # Solve the system of equations for the step size
        step = np.linalg.solve(W_hessian, -W_grad)
        # Update the values for W
        W = previous_W + step
        
        # Determine the step size
        step_size = np.linalg.norm(step)
        
        current_iteration = current_iteration + 1
    
    return W


# Perform the BFGS algorithm and iterate until the step size becomes small enough
def BFGS(objective_function, initial_W, min_step_size=10**(-8), max_iter):
    
    current_iteration = 0
    step_size = 1
    beta = np.eye(2)
    
    # Define the starting point and learning rate
    W = initial_W
    
    # Define the gradient function
    gradient = grad(objective_function)
    
    while np.abs(step_size) > min_step_size and current_iteration < max_iter:
        
        previous_W = W
        
        # Compute the gradient
        W_grad = gradient(previous_W, 1)
        
        # Solve the system of equations for the step size
        step = np.linalg.solve(beta, -W_grad)
        # Update the values for W
        W = previous_W + step
        
        # Determine the step size
        step_size = np.linalg.norm(step)
        
        # Compute the gradient for the updated values and compute delta gradient
        updated_gradient = gradient(previous_W, 1)
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
def conjugate_gradient(objective_function, initial_W, min_step_size=10**(-8), max_iter):
    
    current_iteration = 0
    step_size = 1
    
    # Derive a function that computes the gradient of the objective function
    gradient = grad(objective_function)
    
    # Compute the gradient
    W_grad = gradient(initial_W, 1) 
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
        W_grad = gradient(initial_W, 1) 
        # Reshape the gradient array
        W_grad = W_grad.reshape(len(delta_gradient),1)
        
        # Compute beta
        beta = np.dot(W_grad.T,W_grad)/np.dot(W_grad_previous.T,W_grad_previous)
        
        # Update the value for s
        s = -W_grad.flatten() + beta*s
        
        # Determine the step size
        delta_W = W - previous_W
        step_size = np.linalg.norm(delta_W)
        
        current_iteration = current_iteration + 1
    
    return W


def test_function(inputs):
    return np.power(inputs[0],3) + 4*np.exp(inputs[1]) + 10*np.power(inputs[2],4)

hessian_function = hessian(test_function)
W_hessian = hessian_function((10.,2.,1.))