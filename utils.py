#primary author: Michael and elaine

import matplotlib.pyplot as plt
import numpy as np
import config
import random

#function relating x and y
default_gen_func = lambda x: (x**3)

def generate_data(random_seed,
                  number_of_points=config.training_sample_size,
                noise_variance=config.y_noise_variance,
                region_size = 2,
                gap_size=4,
                boundary_size = 1,
                f = default_gen_func):
    '''
    Function for generating toy regression data


    Generates data from the function y = f(x) + epsilon = x**3 + epsilon with epsilon~N(0, config.y_noise_variance) by default
    with a gap (default size = 1) in the middle of the train set and a boundary (default size is 2) at the left & right extremes for the test set

    '''
    random.seed(random_seed)
    x_train_min = - gap_size/2 - region_size
    x_train_max = gap_size/2 + region_size
    #training x
    x_train = np.hstack((np.linspace(x_train_min, -gap_size/2, number_of_points//2), np.linspace(gap_size/2, x_train_max, number_of_points//2)))
    

    #y is equal to f(x) plus gaussian noise
    y_train = f(x_train) + np.random.normal(0, noise_variance**0.5, number_of_points)

    x_test = np.linspace(x_train_min - boundary_size, x_train_max + boundary_size, number_of_points)
    return x_train.reshape(1, -1), y_train.reshape(1, -1), x_test.reshape(1, -1)


def run_toy_nn(nn_model,architecture,params,random,x_train,y_train,x_test):
    #instantiate a Feedforward neural network object
    nn = nn_model(architecture, random=random)

    #fit my neural network to minimize MSE on the given data
    nn.fit(x_train, y_train, params)

    #predict on the test x-values
    y_test_pred = nn.forward(nn.weights, x_test)
    print(x_test.flatten().reshape(-1,1).shape)

    #visualize the function learned by the neural network
    plt.scatter(x_train.flatten(), y_train.flatten(), color='black', label='data')
    plt.plot(x_test.flatten(), y_test_pred.flatten(), color='red', label='learned neural network function')
    plt.legend(loc='best')
    plt.show()
