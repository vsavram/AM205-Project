# primary author: wei wei pan, modified by Michael and Elaine and Victor

from autograd import numpy as np
from autograd import grad
from autograd.misc.optimizers import adam, sgd
from autograd import scipy as sp
import autograd.numpy.random as npr
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import sys
    
class Feedforward:
    def __init__(self, architecture, random=None, weights=None, objective_function=None):
        self.params = {'H': architecture['width'],
                       'L': architecture['hidden_layers'],
                       'D_in': architecture['input_dim'],
                       'D_out': architecture['output_dim'],
                       'activation_type': architecture['activation_fn_type'],
                       'activation_params': architecture['activation_fn_params']}

        self.D = (  (architecture['input_dim'] * architecture['width'] + architecture['width'])
                  + (architecture['output_dim'] * architecture['width'] + architecture['output_dim'])
                  + (architecture['hidden_layers'] - 1) * (architecture['width']**2 + architecture['width'])
                 )

        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(0)

        self.h = architecture['activation_fn']

        if weights is None:
            self.weights = self.random.normal(0, 1, size=(1, self.D))
        else:
            self.weights = weights

        self.objective_trace = np.empty((1, 1))
        self.weight_trace = np.empty((1, self.D))
            
        if objective_function:
            self.make_objective = objective_function
        else:
            self.make_objective = self.default_make_objective    
        


    def forward(self, weights, x, final_layer_out=False):
        ''' Forward pass given weights and input '''
        H = self.params['H']
        D_in = self.params['D_in']
        D_out = self.params['D_out']

        assert weights.shape[1] == self.D

        if len(x.shape) == 2:
            assert x.shape[0] == D_in
            x = x.reshape((1, D_in, -1))
        else:
            assert x.shape[1] == D_in

        weights = weights.T

        #input to first hidden layer
        W = weights[:H * D_in].T.reshape((-1, H, D_in))
        b = weights[H * D_in:H * D_in + H].T.reshape((-1, H, 1))
        input = self.h(np.matmul(W, x) + b)
        index = H * D_in + H

        assert input.shape[1] == H

        #additional hidden layers
        for _ in range(self.params['L'] - 1):
            before = index
            W = weights[index:index + H * H].T.reshape((-1, H, H))
            index += H * H
            b = weights[index:index + H].T.reshape((-1, H, 1))
            index += H
            output = np.matmul(W, input) + b
            input = self.h(output) #apply activation func

            assert input.shape[1] == H

        #output layer #THIS IS JUST FOR FINAL LAYER, INPUT STORES VALUES OF THE LAST HIDDEN LAYER
        W = weights[index:index + H * D_out].T.reshape((-1, D_out, H))
        b = weights[index + H * D_out:].T.reshape((-1, D_out, 1))
        output = np.matmul(W, input) + b
        assert output.shape[1] == D_out

        #last hidden layer
        final_layer = np.array(input,copy = True)

        self.current_output = output

        if final_layer_out:
            return final_layer
        else:
            return output
    
    """Default objective Neural network function"""
    def default_make_objective(self, x_train, y_train, reg_param):

        def objective(W, t=1.):
            squared_error = np.linalg.norm(y_train - self.forward(W, x_train), axis=1)**2
            if reg_param is None:
                sum_error = np.sum(squared_error)
                return sum_error
            else:
                mean_error = np.mean(squared_error) + reg_param * np.linalg.norm(W)
                return mean_error

        return objective, grad(objective)
    
    """
    - the params dictionary specificies a bunch of optimization parameters
    - the reg_param parameter regularizes the objection function
    """
    def fit(self, x_train, y_train, params, reg_param=None):

        assert x_train.shape[0] == self.params['D_in']

        # this assert statement is no longer true (aux functions). rip
        #assert y_train.shape[0] == self.params['D_out']

        ### make objective function for training

        self.objective, self.gradient = self.make_objective(x_train, y_train, reg_param)

        ### set up optimization
        step_size = 0.01
        max_iteration = 5000
        check_point = 100
        weights_init = self.weights.reshape((1, -1))
        mass = None
        optimizer = 'adam' # DEFAULT. CHANGE IN PARAMS
        opt_gradient = self.gradient # DEFAULT. CHANGE IN PARAMS
        random_restarts = 5

        if 'step_size' in params.keys():
            step_size = params['step_size']
        if 'max_iteration' in params.keys():
            max_iteration = params['max_iteration']
        if 'check_point' in params.keys():
            self.check_point = params['check_point']
        if 'init' in params.keys():
            weights_init = params['init']
        if 'call_back' in params.keys():
            call_back = params['call_back']
        if 'mass' in params.keys():
            mass = params['mass']
        if 'optimizer' in params.keys():
            optimizer = params['optimizer']
        if  'opt_gradient' in params.keys():
            gradient = params['opt_gradient']
        if 'random_restarts' in params.keys():
            random_restarts = params['random_restarts']

        def call_back(weights, iteration, g):
            ''' Actions per optimization step '''
            objective = self.objective(weights, iteration)
            self.objective_trace = np.vstack((self.objective_trace, objective))
            self.weight_trace = np.vstack((self.weight_trace, weights))
            if iteration % check_point == 0:
                print("\r Iteration {} lower bound {}; gradient mag: {}".format(iteration, objective, np.linalg.norm(self.gradient(weights, iteration))),end ="")

        ### train with random restarts
        optimal_obj = 1e16
        optimal_weights = self.weights

        for i in range(random_restarts):

            # The default optimizer is Adam
            # This is where we will use numerical methods for finding the minimum of the objective function
            if optimizer == 'adam':
                adam(opt_gradient, weights_init, step_size=step_size, num_iters=max_iteration, callback=call_back)
                local_opt = np.min(self.objective_trace[-100:])
                if local_opt < optimal_obj:
                    opt_index = np.argmin(self.objective_trace[-100:])
                    self.weights = self.weight_trace[-100:][opt_index].reshape((1, -1))
                    self.objective_trace = self.objective_trace[1:]
                    self.weight_trace = self.weight_trace[1:]
            else:
                optimal_weights,weight_trace,objective_trace = optimizer(self.objective, 
                                                                         weights_init, 
                                                                         min_step_size=10**(-8),
                                                                         max_iter=max_iteration)
                local_opt = self.objective(optimal_weights, 1.)
                if local_opt < optimal_obj:
                    self.weights = optimal_weights.reshape((1, -1))
                    self.objective_trace = objective_trace
                    self.weight_trace = weight_trace

            weights_init = self.random.normal(0, 1, size=(1, self.D))
    