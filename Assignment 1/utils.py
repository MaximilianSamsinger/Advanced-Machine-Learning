'''
sHelper file fore Ridge_Regression
'''

import numpy as np
import random

def sample_function(f, samplesize, mean = 0, sigma = 1):
    ''' We generate pairs two vectors x and f(x) + normalerror'''
    
    x = np.linspace(0,1,samplesize) #This choice is arbitary 
    function = f(x)
    error = np.random.normal(mean, sigma , samplesize)
    return x, function + error

def polynomial(X, coefficients):
    polynomial = np.zeros(len(X))
    for k in range(len(coefficients)):
        polynomial += coefficients[k]*X**k # Numerically not ideal
    return polynomial

def RMSerror(x,y, coefficients):
    return np.linalg.norm(y-polynomial(x,coefficients))/np.sqrt(len(x))
    

def linear_regression(x,y, degree = 30, regularizer = 1e-4):
    ''' Interpolate using a polynomial regression function '''
    A = np.zeros((len(x),degree + 1))
    for k in range(len(x)):
        A[k] = x[k] ** np.arange(degree + 1)
    
    ''' Solving a linear system, since computing the matrix inverse is 
    numerically unstable and evil in general '''
    coefficients = np.linalg.solve(A.T@A + regularizer*np.identity(degree + 1)
                                    , A.T@y)
    return coefficients


def partition(x, num_subsets):
    ''' 
    Returns a partition of numbers (indices) from 0 to len(x)-1 .
    These serve as indices for partitioning x.
    The indices get at first split into buckets and then equally 
    (but randomly per bucket) distributed to the subsets.
    '''
    subsets = [[] for k in range(num_subsets)]
    len_subset = len(x)//num_subsets
    numbers = list(range(len(x)))
    for k in range(len_subset):
        # Choose the kth bucket
        # A bucket ranges from shift to shift + len_subset
        shift = k*num_subsets 
        bucket = numbers[shift:shift+len_subset] 
        # Shuffle bucket
        numbers[shift:shift+len_subset] = random.sample(bucket,len(bucket))
    for i in range(len(x)):
        subsets[i%num_subsets] += [numbers[i]]
    return subsets

def optimal_coefficients(train_data, test_data, lambdas, degree = 30):
    x_train, y_train = train_data
    x_test, y_test = test_data
    
    train_errors = np.zeros(len(lambdas))
    test_errors = np.zeros(len(lambdas))
    
    ''' We first choose different values (lambdas) for regularization and then
    look for the optimal one (find lowest error)'''
    
    for k in range(len(lambdas)):
        coefficients = linear_regression(x_train, y_train, degree,
                                         regularizer = lambdas[k])
        train_errors[k] = RMSerror(x_train, y_train, coefficients)
        test_errors[k] = RMSerror(x_test, y_test, coefficients)
        
    optimal_lambda = lambdas[np.argmin(test_errors)]
    optimal_coeff = linear_regression(x_train, y_train, degree, optimal_lambda)
    
    return optimal_coeff, optimal_lambda, (train_errors, test_errors)