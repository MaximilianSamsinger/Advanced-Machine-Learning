import numpy as np
import matplotlib.pyplot as plt
from utils import sample_function, polynomial, RMSerror, linear_regression, \
    partition

def cross_validation(x, y, partitions, regularizer, degree = 30):
    ''' 
    Note: For each partition we get different optimal lambda! 
    We probably want to change that by changing optimal_coefficients()
    '''
    averaged_coeff = 0
    averaged_training_error = 0
    averaged_test_error = 0
    
    for count, test_partition in enumerate(partitions):
        ''' Create training partition by combining all non-test partitions '''
        train_partition = []
        for k in range(len(partitions)):
            if k != count:
                train_partition += partitions[k]
        x_train, y_train = x[train_partition], y[train_partition]
        x_test, y_test = x[test_partition], y[test_partition]
        
        coefficients = linear_regression(x_train, y_train, degree,
                                         regularizer = regularizer)
        train_error = RMSerror(x_train, y_train, coefficients)
        test_errors = RMSerror(x_test, y_test, coefficients)
        
        averaged_coeff += coefficients
        averaged_training_error += train_error
        averaged_test_error += test_errors
    
    averaged_coeff /= len(partitions)
    averaged_training_error /= len(partitions)
    averaged_test_error /= len(partitions)
    return averaged_coeff, averaged_training_error, averaged_test_error

def cross_validate_for_all_lambdas(x, y, partitions, lambdas, degree):
    train_errors = np.zeros(len(lambdas))
    test_errors = np.zeros(len(lambdas))
    coefficients = np.zeros((len(lambdas), degree + 1))
    for k, regularizer in enumerate(lambdas):
        coefficients[k], train_errors[k], test_errors[k] = cross_validation(
                x, y, partitions, regularizer, degree)
    return coefficients, train_errors, test_errors

'''
Define Parameters here
'''
N = 30  # Number of points to be sampled
M = 20  # Maximal interpolation degree
mean, sigma = 0, 1e-1 # Mean and standard deviation of the error term
K = 4 # Number of partitions

loglambdas = np.linspace(-37,-1,300)
lambdas = np.exp(loglambdas)


def f(x):
    return np.sin(2*np.pi*x)

x, y  = sample_function(f, N, mean, sigma)


partitions = partition(x,K)
coefficients, train_errors, test_errors = cross_validate_for_all_lambdas(
        x, y, partitions, lambdas, degree = M)

overfit_coeff = coefficients[0]
optimal_coeff = coefficients[np.argmin(test_errors)]
underfit_coeff = coefficients[-1]


X = np.linspace(0,1,500)

plt.figure(figsize=(19, 9))
plt.subplot(121)
plt.plot(x,y,'o', label='data')
plt.plot(X, polynomial(X, overfit_coeff), label='overfit, lambda = 0')
plt.plot(X, f(X), '-.', label='ground truth')
plt.plot(X, polynomial(X, optimal_coeff), label='optimal fit, log lamda = ' 
         + str(round(loglambdas[np.argmin(test_errors)],2)))
plt.plot(X, polynomial(X, underfit_coeff), label='underfit, log lamda = ' 
         + str(round(loglambdas[-1],2)))
plt.title('Interpolation without, with optimal and with too much regularization' 
          + '\ntest error= ' + str(round(RMSerror(x,y, overfit_coeff),6)))
plt.axis((-0.05, 1.05, -1.3, 1.3))
plt.legend()

plt.subplot(122)
plt.plot(loglambdas, train_errors, 'b', label='training')
plt.plot(loglambdas, test_errors, 'r', label='test')
plt.title('Error vs log lambda')
plt.xlabel('log lambda')
plt.ylabel('E_RMS')
plt.legend()
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, 0, 0.6))

'''
plt.figure(figsize=(19, 9))
plt.subplot(131)
plt.plot(x,y,'o', label='data')
plt.plot(X, polynomial(X, overfit_coeff), 'r', label='interpolation polynomial')
plt.plot(X, f(X), 'g--', label='ground truth')
plt.title('Interpolation without regularization' 
          + '\ntest error= ' + str(round(RMSerror(x,y, overfit_coeff),4)))
plt.legend()

plt.subplot(132)
plt.plot(x,y,'o', label='data')
plt.plot(X, polynomial(X, optimal_coeff), 'r', label='interpolation polynomial')
plt.plot(X, f(X), 'g--', label='ground truth')
plt.title('Interpolation with (optimal) regularization' 
          + '\ntest error= ' + str(round(RMSerror(x,y, optimal_coeff),4)))
plt.legend()

plt.subplot(133)
plt.plot(x,y,'o', label='data')
plt.plot(X, polynomial(X, underfit_coeff), 'r', label='interpolation polynomial')
plt.plot(X, f(X), 'g--', label='ground truth')
plt.title('Interpolation with too much regularization' 
          + '\ntest error= ' + str(round(RMSerror(x,y, underfit_coeff),4)))
plt.legend()

plt.figure()
plt.plot(loglambdas, train_errors, 'b', label='training')
plt.plot(loglambdas, test_errors, 'r', label='test')
plt.title('Error vs log lambda')
plt.xlabel('log lambda')
plt.ylabel('E_RMS')
plt.legend()
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, 0, 0.6))
'''