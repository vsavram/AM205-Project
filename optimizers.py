#!/usr/bin/env python3
# primary author: victor

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
    
    # Compute the initial MSE
    mse_value = objective_function(W)
    # Initialize lists used to store the MSE and weights after each iteration
    objective_trace,weight_trace = [mse_value],[W.flatten()]
    
    while np.abs(step_size) > min_step_size and current_iteration < max_iter:
        
        previous_W = W
        
        # Determine the gradient of the objective function using autograd
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
    
        # Compute the MSE
        mse_value = objective_function(W)
        # Append the MSE and the updated weights to the respective lists
        objective_trace.append(mse_value)
        weight_trace.append(W.flatten())
                           
    return W,np.array(weight_trace),np.array(objective_trace)


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
    
    # Compute the initial MSE
    mse_value = objective_function(W)
    # Initialize lists used to store the MSE and weights after each iteration
    objective_trace,weight_trace = [mse_value],[W.flatten()]
    
    while np.abs(step_size) > min_step_size and current_iteration < max_iter:
        
        previous_W = W
                
        # Compute the gradient
        W_grad = gradient(previous_W)
                
        # Compute the Hessian
        W_hessian = hessian_function(previous_W)
        # Reshape the hessian
        proper_size = int(np.sqrt(len(W_hessian.flatten())))
        W_hessian = W_hessian.reshape(proper_size,proper_size)
        
        # Solve the system of equations for the step size
        step = np.linalg.solve(W_hessian, -W_grad.flatten())
        # Update the values for W
        W = previous_W + step
        
        # Determine the step size
        step_size = np.linalg.norm(step.flatten())
        
        current_iteration = current_iteration + 1
        
        # Compute the MSE
        mse_value = objective_function(W)
        # Append the MSE and the updated weights to the respective lists
        objective_trace.append(mse_value)
        weight_trace.append(W.flatten())
    
    return W,np.array(weight_trace),np.array(objective_trace)


# Perform the BFGS algorithm and iterate until the step size becomes small enough
def BFGS(objective_function, initial_W, min_step_size=10**(-8), max_iter=2000):
    
    current_iteration = 0
    step_size = 1
    beta = np.eye(len(initial_W.flatten()))
    
    # Define the starting point and learning rate
    W = initial_W
    
    # Define the gradient function
    gradient = grad(objective_function)
    
    # Compute the initial MSE
    mse_value = objective_function(W)
    # Initialize lists used to store the MSE and weights after each iteration
    objective_trace,weight_trace = [mse_value],[W.flatten()]
    
    while np.abs(step_size) > min_step_size and current_iteration < max_iter:
        
        previous_W = W
        
        # Compute the gradient
        W_grad = gradient(previous_W)
        
        # Solve the system of equations for the step size
        step = np.linalg.solve(beta, -W_grad.flatten())
        # Update the values for W
        W = previous_W + step
        
        # Determine the step size
        step_size = np.linalg.norm(step)
        
        # Compute the gradient for the updated values and compute delta gradient
        updated_gradient = gradient(np.array(W))
        delta_gradient = updated_gradient - W_grad
        
        # Compute delta beta
        delta_gradient = delta_gradient.reshape(len(delta_gradient.flatten()),1)
        step = step.reshape(len(step.flatten()),1)
        delta_beta = np.dot(delta_gradient,delta_gradient.T)/np.dot(delta_gradient.T,step) - np.dot(np.dot(beta,step), np.dot(step.T,beta))/np.dot(np.dot(step.T,beta), step)
        
        # Update the value for beta
        beta = beta + delta_beta
        
        current_iteration = current_iteration + 1
        
        # Compute the MSE
        mse_value = objective_function(W)
        # Append the MSE and the updated weights to the respective lists
        objective_trace.append(mse_value)
        weight_trace.append(W.flatten())
    
    return W,np.array(weight_trace),np.array(objective_trace)


# Perform the conjugate gradient method and iterate until the step size becomes small enough
def conjugate_gradient(objective_function, initial_W, min_step_size=10**(-8), max_iter=2000):
    
    current_iteration = 0
    step_size = 1
    
    # Derive a function that computes the gradient of the objective function
    gradient = grad(objective_function)
    
    # Compute the gradient
    W_grad = gradient(initial_W)
    # Define the starting point
    W = -W_grad

    # Set the initial value for s
    s = -W_grad
    
    # Compute the initial MSE
    mse_value = objective_function(initial_W)
    # Initialize lists used to store the MSE and weights after each iteration
    objective_trace,weight_trace = [mse_value],[initial_W.flatten()]
    
    while np.abs(step_size) > min_step_size and current_iteration < max_iter:
        
        previous_W = W
        W_grad_previous = W_grad
        
        # Determine the learning rate for the update using a line search
        alpha = minimize_scalar(lambda alpha: objective_function(previous_W + alpha*s))
        alpha = alpha.x
        
        # Update the values for x and y
        W = W + alpha*s
        
        # Compute the new gradient
        W_grad = gradient(W)
        # Reshape the gradient arrays
        W_grad_previous = W_grad_previous.reshape(len(W_grad_previous.flatten()),1)
        W_grad_reshaped = W_grad.reshape(len(W_grad.flatten()),1)
        
        # Compute beta
        beta = np.dot(W_grad_reshaped.T,W_grad_reshaped)/np.dot(W_grad_previous.T,W_grad_previous)
        beta = beta[0][0]
        
        # Update the value for s
        s = -W_grad + beta*s
        
        # Determine the step size
        delta_W = W - previous_W
        step_size = np.linalg.norm(delta_W.flatten())
        
        current_iteration = current_iteration + 1
    
        # Compute the MSE
        mse_value = objective_function(W)
        # Append the MSE and the updated weights to the respective lists
        objective_trace.append(mse_value)
        weight_trace.append(W.flatten())
        
    return W,np.array(weight_trace),np.array(objective_trace)
