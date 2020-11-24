import matplotlib.pyplot as plt
import numpy as np

def generate_data(number_of_points=10, noise_variance=0.1):
    '''Function for generating toy regression data'''
    x_train_min = -2
    x_train_max = 2
    #training x
    x_train = np.hstack((np.linspace(x_train_min, -0.5, number_of_points), np.linspace(0.5, x_train_max, number_of_points)))
    #function relating x and y
    f = lambda x: (x**3)/2
    #y is equal to f(x) plus gaussian noise
    y_train = f(x_train) + np.random.normal(0, noise_variance**0.5, 2 * number_of_points)
    #x_test = np.array(list(set(list(np.hstack((np.linspace(x_train_min-2, x_train_max+2, 200), x_train))))))
    #x_test = np.sort(x_test)
    x_test = np.linspace(x_train_min-2, x_train_max+2, 200)
    return x_train, y_train, x_test

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