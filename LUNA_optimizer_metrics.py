#!/usr/bin/env python3
# primary author: victor

# Import standard libraries
from autograd import numpy as np
from autograd import grad
from autograd.misc.optimizers import adam, sgd
from autograd import scipy as sp
import autograd.numpy.random as npr
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import sys
import timeit
import time

# import our libraries
import bayes_helpers as bh
from utils import generate_data
from utils import run_toy_nn
from utils import default_gen_func
from feed_forward import Feedforward
from nlm import NLM
from luna import LUNA
import bayes_helpers as bh
from config import *
from optimizers import *


# Generate the training and test sets
x_train, y_train, x_test = generate_data(random_seed)

training_sample_size = 100

######################
### Model Params
######################
prior_variance = 1 # chosen in the paper. declared "reasonable". who decided this
y_noise_variance = 9 # needs to match what the dataset itself is

######################
### Feed Forward Params
######################

####  activation function ####
activation_fn_type = 'relu'
activation_fn = lambda x: np.maximum(np.zeros(x.shape), x)

random_seed = np.random.RandomState(0)

real_max_iteration = 5000

########################
#### LUNA SPECIFIC  ####
########################

luna_architecture = {'width': 50,
            'hidden_layers': 2,
            'input_dim': 1,
            'output_dim': 50, #number of auxiliary functions, # note, in NLM, below, we change to 1
            'activation_fn_type': activation_fn_type,
            'activation_fn_params': 'rate=1',
            'activation_fn': activation_fn}

regularization_param_luna = 1e-1 # in the paper they searched over 1e-3,...,1e3 and chose 1e-1 for regularization

similarity_param = 1e0 # in the paper they searched over 1e-3,...,1e3 and chose 1e0 for similarity

########################
#### NLM SPECIFIC  ####
########################

nlm_architecture = luna_architecture.copy()
nlm_architecture["output_dim"] = 1
regularization_param_nlm = 8.37 #they chose this in the paper, what a beautifully specific number


# Define a function that sets the architecture for the NN
def set_architecture(width, hidden_layers, input_dim, output_dim, activation_fn_type, activation_fn):
    architecture = {'width': width,
                    'hidden_layers': hidden_layers,
                    'input_dim': input_dim,
                    'output_dim': output_dim,
                    'activation_fn_type': activation_fn_type,
                    'activation_fn_params': 'rate=1',
                    'activation_fn': activation_fn}
    return architecture 


# Define a function that produces the optimal weights and objective trace for each optimizer
def train_NLM(x_train, y_train, x_test, optimizer_list):
    
    # Initialize a list used to store the optimal weights, objective traces, and number of iterations
    weights_list,mse_list,num_iterations_list,elapsed_time_list,predictions = [],[],[],[],[]
    
    # Iterate over each optimizer
    for optimizer_function in optimizer_list:

        # Set the parameters
        params = {'max_iteration': 500, 
                  'random_restarts': 1,
                  'optimizer': optimizer_function}

        # Initialize the NLM
        optimizer_nlm = NLM(prior_var, y_var, regularization_param_nlm, architecture, random_state = random_seed)

        # Fit the NLM to the training set (compute the elapsed time)
        start_time = timeit.default_timer()
        optimizer_nlm.train(x_train, y_train, params)
        elapsed_time = timeit.default_timer() - start_time
        
        # Pull the optimal weights and objective trace
        weights = optimizer_nlm.ff.weights
        mse = optimizer_nlm.ff.mse
        num_iter = optimizer_nlm.ff.num_iter
        
        # Compute the predictions
        pred = optimizer_nlm.predict(x_test, prior=False)
        
        weights_list.append(weights)
        mse_list.append(mse)
        num_iterations_list.append(num_iter)
        elapsed_time_list.append(elapsed_time)
        predictions.append(pred)
        
    return weights_list, mse_list, num_iterations_list, elapsed_time_list, predictions


def visualize_LUNA(x_train,y_train,x_test,posterior_predictive_samples,title_text, multi=None, ylim = [-150,150],gen_func=default_gen_func):
    
    if multi ==None:
        fig, ax = plt.subplots(1, 1, figsize=(10,10))
    
    else:
        ax = multi
        
    # Compute the 97.5 th percentile of the posterior predictive predictions
    pp_upper = np.percentile(posterior_predictive_samples, 97.5, axis=0)

    # Compute the 2.5 th percentile of the posterior predictive predictions
    pp_lower = np.percentile(posterior_predictive_samples, 2.5, axis=0)

    # Compute the 50 th percentile of the posterior predictive predictions
    pp_mean = np.mean(posterior_predictive_samples, axis=0)

    ax.plot(x_test, pp_mean, color='blue', label = 'mean prediction') # visualize the mean of the posterior predictive
    ax.fill_between(x_test, pp_upper, pp_lower, color='blue', alpha=0.3, label='95% Conf. Interval') # visualize the 95% posterior predictive interval
    ax.scatter(x_train, y_train, color='red', label='training data') # visualize the training data
    
    ax.plot(x_test,[gen_func(x) for x in x_test], color = 'black', label = 'true function')
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    ax.set_title(title_text, fontsize=22)
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', labelsize=16)

    
    if multi == None:
        ax.legend(fontsize=14)
        return fig
    else:
        return ax
 
# Generate data in order to evaluate the model and compute the log likelihood
x_valid, y_valid, x_test_not_used = generate_data(valid_seed)


#---------------------------------------------------------------------------------------------------
# LUNA 100 iterations
#---------------------------------------------------------------------------------------------------
    
#### STEEPEST DESCENT ####
opt_params = {'step_size':1e-3,
          'max_iteration':100,
          'random_restarts':1,
          'optimizer': steepest_descent}

# Train LUNA
t0 = time.time()
luna_steepest = LUNA(prior_variance, y_noise_variance, regularization_param_luna, similarity_param, luna_architecture, random_seed)
luna_steepest.train(x_train, y_train, opt_params)
print(f"time: {np.round(time.time() - t0, 3)} seconds")
steepest_posterior_predictions, steepest_posterior_predictive_samples = luna_steepest.predict(x_test)
# Create a plot for steepest descent
steepest_plot = visualize_LUNA(x_train,y_train,x_test.flatten(),steepest_posterior_predictive_samples,'Steepest Descent (100 iterations)', multi=None, ylim = [-150,150],gen_func=default_gen_func)
steepest_plot.savefig("./LUNA_performance_data/Steepest_LUNA_plot_100iter.png")
# Compute the log likelihood
steepest_logl_100 = luna_steepest.get_log_l(x_train,y_train,x_valid,y_valid)

#### BFGS ####
opt_params = {'step_size':1e-3,
          'max_iteration':100,
          'random_restarts':1,
          'optimizer': 'scipy_BFGS'}

# Train LUNA
t0 = time.time()
luna_BFGS = LUNA(prior_variance, y_noise_variance, regularization_param_luna, similarity_param, luna_architecture, random_seed)
luna_BFGS.train(x_train, y_train, opt_params)
print(f"time: {np.round(time.time() - t0, 3)} seconds")
BFGS_posterior_predictions, BFGS_posterior_predictive_samples = luna_BFGS.predict(x_test)
# Create a plot for BFGS
BFGS_plot = visualize_LUNA(x_train,y_train,x_test.flatten(),BFGS_posterior_predictive_samples,'BFGS (100 iterations)', multi=None, ylim = [-150,150],gen_func=default_gen_func)
BFGS_plot.savefig("./LUNA_performance_data/BFGS_LUNA_plot_100iter.png")
# Compute the log likelihood
BFGS_logl_100 = luna_BFGS.get_log_l(x_train,y_train,x_valid,y_valid)

#### CONJUGATE GRADIENT ####
opt_params = {'step_size':1e-3,
          'max_iteration':100,
          'random_restarts':1,
          'optimizer': 'scipy_CG'}

# Train LUNA
t0 = time.time()
luna_CG = LUNA(prior_variance, y_noise_variance, regularization_param_luna, similarity_param, luna_architecture, random_seed)
luna_CG.train(x_train, y_train, opt_params)
print(f"time: {np.round(time.time() - t0, 3)} seconds")
CG_posterior_predictions, CG_posterior_predictive_samples = luna_CG.predict(x_test)
# Create a plot for conjugate gradient
CG_plot = visualize_LUNA(x_train,y_train,x_test.flatten(),CG_posterior_predictive_samples,'Conjugate Gradient (100 iterations)', multi=None, ylim = [-150,150],gen_func=default_gen_func)
CG_plot.savefig("./LUNA_performance_data/CG_LUNA_plot_100iter.png")
# Compute the log likelihood
CG_logl_100 = luna_CG.get_log_l(x_train,y_train,x_valid,y_valid)

#---------------------------------------------------------------------------------------------------
# LUNA 1000 iterations
#---------------------------------------------------------------------------------------------------
  
#### STEEPEST DESCENT ####
opt_params = {'step_size':1e-3,
          'max_iteration':1000,
          'random_restarts':1,
          'optimizer': steepest_descent}

# Train LUNA
t0 = time.time()
luna_steepest_1000 = LUNA(prior_variance, y_noise_variance, regularization_param_luna, similarity_param, luna_architecture, random_seed)
luna_steepest_1000.train(x_train, y_train, opt_params)
print(f"time: {np.round(time.time() - t0, 3)} seconds")
steepest_posterior_predictions_1000, steepest_posterior_predictive_samples_1000 = luna_steepest_1000.predict(x_test)
# Create a plot for steepest descent
steepest_plot = visualize_LUNA(x_train,y_train,x_test.flatten(),steepest_posterior_predictive_samples_1000,'Steepest Descent (1000 iterations)', multi=None, ylim = [-150,150],gen_func=default_gen_func)
steepest_plot.savefig("./LUNA_performance_data/Steepest_LUNA_plot_1000iter.png")
# Compute the log likelihood
steepest_logl_1000 = luna_steepest_1000.get_log_l(x_train,y_train,x_valid,y_valid)

#### BFGS ####
opt_params = {'step_size':1e-3,
          'max_iteration':1000,
          'random_restarts':1,
          'optimizer': 'scipy_BFGS'}

# Train LUNA
t0 = time.time()
luna_BFGS_1000 = LUNA(prior_variance, y_noise_variance, regularization_param_luna, similarity_param, luna_architecture, random_seed)
luna_BFGS_1000.train(x_train, y_train, opt_params)
print(f"time: {np.round(time.time() - t0, 3)} seconds")
BFGS_posterior_predictions_1000, BFGS_posterior_predictive_samples_1000 = luna_BFGS_1000.predict(x_test)
# Create a plot for BFGS
BFGS_plot = visualize_LUNA(x_train,y_train,x_test.flatten(),BFGS_posterior_predictive_samples_1000,'BFGS (1000 iterations)', multi=None, ylim = [-150,150],gen_func=default_gen_func)
BFGS_plot.savefig("./LUNA_performance_data/BFGS_LUNA_plot_1000iter.png")
# Compute the log likelihood
BFGS_logl_1000 = luna_BFGS_1000.get_log_l(x_train,y_train,x_valid,y_valid)


#### CONJUGATE GRADIENT ####
opt_params = {'step_size':1e-3,
          'max_iteration':1000,
          'random_restarts':1,
          'optimizer': 'scipy_CG'}

# Train LUNA
t0 = time.time()
luna_CG_1000 = LUNA(prior_variance, y_noise_variance, regularization_param_luna, similarity_param, luna_architecture, random_seed)
luna_CG_1000.train(x_train, y_train, opt_params)
print(f"time: {np.round(time.time() - t0, 3)} seconds")
CG_posterior_predictions_1000, CG_posterior_predictive_samples_1000 = luna_CG_1000.predict(x_test)
# Create a plot for conjugate gradient
CG_plot = visualize_LUNA(x_train,y_train,x_test.flatten(),CG_posterior_predictive_samples_1000,'Conjugate Gradient (1000 iterations)', multi=None, ylim = [-150,150],gen_func=default_gen_func)
CG_plot.savefig("./LUNA_performance_data/CG_LUNA_plot_1000iter.png")
# Compute the log likelihood
CG_logl_1000 = luna_CG_1000.get_log_l(x_train,y_train,x_valid,y_valid)


posterior_predictions, posterior_predictive_samples = model.predict(x_test)
bh.viz_pp_samples(x_train, y_train,x_test.flatten(),posterior_predictive_samples,name)
    
# Randomly generate a training set of 50 data points
#number_of_points = 50
x_train, y_train, x_test = generate_data(training_seed)

# Define a function that sets the architecture for the NN
def set_architecture(width, hidden_layers, input_dim, output_dim, activation_fn_type, activation_fn):
    architecture = {'width': width,
                    'hidden_layers': hidden_layers,
                    'input_dim': input_dim,
                    'output_dim': output_dim,
                    'activation_fn_type': activation_fn_type,
                    'activation_fn_params': 'rate=1',
                    'activation_fn': activation_fn}
    return architecture 


# Define a function that produces the optimal weights and objective trace for each optimizer
def train_NLM(x_train, y_train, x_test, optimizer_list):
    
    # Initialize a list used to store the optimal weights, objective traces, and number of iterations
    weights_list,mse_list,num_iterations_list,elapsed_time_list,predictions = [],[],[],[],[]
    
    # Iterate over each optimizer
    for optimizer_function in optimizer_list:

        # Set the parameters
        params = {'max_iteration': 500, 
                  'random_restarts': 1,
                  'optimizer': optimizer_function}

        # Initialize the NLM
        optimizer_nlm = NLM(prior_var, y_var, regularization_param_nlm, architecture, random_state = random_seed)

        # Fit the NLM to the training set (compute the elapsed time)
        start_time = timeit.default_timer()
        optimizer_nlm.train(x_train, y_train, params)
        elapsed_time = timeit.default_timer() - start_time
        
        # Pull the optimal weights and objective trace
        weights = optimizer_nlm.ff.weights
        mse = optimizer_nlm.ff.mse
        num_iter = optimizer_nlm.ff.num_iter
        
        # Compute the predictions
        pred = optimizer_nlm.predict(x_test, prior=False)
        
        weights_list.append(weights)
        mse_list.append(mse)
        num_iterations_list.append(num_iter)
        elapsed_time_list.append(elapsed_time)
        predictions.append(pred)
        
    return weights_list, mse_list, num_iterations_list, elapsed_time_list, predictions


#---------------------------------------------------------------------------------------------------
# Create performance metrics for a width of 50
#---------------------------------------------------------------------------------------------------

# Define the relu activation function
activation_fn_type = 'relu'
activation_fn = lambda x: np.maximum(np.zeros(x.shape), x)
# Define the neural network model parameters
width = 20
hidden_layers = 2
input_dim = 1
output_dim = 1
# Initially set the architecture
architecture = set_architecture(width, hidden_layers, input_dim, output_dim, activation_fn_type, activation_fn)

# Set the paramters
prior_var = 1.0
y_var = 9.0
regularization_param_nlm = 8.37

# Define the list of optimizers to use
optimizer_list = [steepest_descent, 'scipy_BFGS', 'scipy_CG']

# Initialize the NLM
nlm = NLM(prior_var, y_var, regularization_param_nlm, architecture, random_state = random_seed)
    
# Train the NLM using each of the optimizers
optimal_weights_list,mse_list,num_iterations_list,elapsed_time_list,predictions = train_NLM(x_train, y_train, x_test, optimizer_list)

# Compute the predictions
steepest_pred = np.mean(predictions[0][1], axis=0)
BFGS_pred = np.mean(predictions[1][1], axis=0)
conjugate_pred = np.mean(predictions[2][1], axis=0)

# Pull the final train MSE for each optimizer
#mse_list = [x[-1] for x in objective_trace_list]
# Determine the test set predictions
steepest_pred = nlm.ff.forward(optimal_weights_list[0].reshape(1,-1), x_test, final_layer_out=False)
BFGS_pred = nlm.ff.forward(optimal_weights_list[1].reshape(1,-1), x_test, final_layer_out=False)
conjugate_pred = nlm.ff.forward(optimal_weights_list[2].reshape(1,-1), x_test, final_layer_out=False)

# Create a dataframe that stores the performance metrics for each optimizer
performance_df = pd.DataFrame({'optimizer': ['Steepest Descent', 'BFGS', 'Conjugate Gradient'],
                               'MSE': mse_list, 
                               'number of iterations': num_iterations_list, 
                               'elapsed time': elapsed_time_list})
performance_df.to_csv("./NLM_performance_data/optimizer_standard_nlm_performance.csv")

# Create a plot of the training set and NN predictions
fig, ax = plt.subplots(figsize = (8,6))
ax.scatter(x_train[0], y_train[0], c='k', label='true data')
ax.plot(x_test[0], steepest_pred.flatten(), c='b', linewidth = 2, label='steepest predictions')
ax.plot(x_test[0], BFGS_pred.flatten(), c='r', linewidth = 2, label='BFGS predictions')
ax.plot(x_test[0], conjugate_pred.flatten(), c='g', linewidth = 2, label='conjugate predictions')
ax.set_xlabel('x', fontsize = 20)
ax.set_ylabel('y', fontsize = 20)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.title('NLM Data Fitting', fontsize = 22)
ax.legend(fontsize=16)
plt.savefig("./NLM_performance_data/NLM_mean_fit.png")