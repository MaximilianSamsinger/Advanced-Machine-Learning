import numpy as np
import matplotlib.pyplot as plt
from utils import sample_function, polynomial, RMSerror, linear_regression, \
    partition

def cross_validation(x, y, partition, regularizer, degree = 30):
    ''' 
    For each element in partition we split  the data (x, y)
    into a training partition and a test partition.
    
    Then we perform linear regression for the given regularization parameter 
    to calculate the optimal coefficients for the current training set
    and test set.
    
    Finally we average all coefficients and errors.
    
    Remark: degree is the maximal degree of our interpolation polynomial.  
    '''
    averaged_coeff = 0
    averaged_training_error = 0
    averaged_test_error = 0
    
    for count, test_subset in enumerate(partition):
        ''' Create training set by combining all non-test sets '''
        train_subset = []
        for k in range(len(partition)):
            if k != count:
                train_subset += partition[k]
        x_train, y_train = x[train_subset], y[train_subset]
        x_test, y_test = x[test_subset], y[test_subset]
        
        coefficients = linear_regression(x_train, y_train, degree,
                                         regularizer = regularizer)
        train_error = RMSerror(x_train, y_train, coefficients)
        test_errors = RMSerror(x_test, y_test, coefficients)
        
        averaged_coeff += coefficients
        averaged_training_error += train_error
        averaged_test_error += test_errors
    
    averaged_coeff /= len(partition)
    averaged_training_error /= len(partition)
    averaged_test_error /= len(partition)
    return averaged_coeff, averaged_training_error, averaged_test_error

def cross_validate_for_all_lambdas(x, y, partition, lambdas, degree):
    '''
    Same as cross_validation. However, we accept multiple regularization 
    parameters and call them lambdas. 
    
    This function returns a coefficient matrix which returns the coefficients
    coefficients[k] for each lambdas[k].
    '''
    
    train_errors = np.zeros(len(lambdas))
    test_errors = np.zeros(len(lambdas))
    coefficients = np.zeros((len(lambdas), degree + 1))
    for k, regularizer in enumerate(lambdas):
        coefficients[k], train_errors[k], test_errors[k] = cross_validation(
                x, y, partition, regularizer, degree)
    return coefficients, train_errors, test_errors

'''
Define Parameters here
'''
N = 30  # Number of points to be sampled
M = 20  # Maximal interpolation degree
mean, sigma = 0, 1e-1 # Mean and standard deviation of the error term
K = 5 # Partition our data into K subsets. 

loglambdas = np.linspace(-37,-1,300)
lambdas = np.exp(loglambdas)


def f(x):
    return np.sin(2*np.pi*x)

x, y  = sample_function(f, N, mean, sigma)


partition = partition(x,K)
coefficients, train_errors, test_errors = cross_validate_for_all_lambdas(
        x, y, partition, lambdas, degree = M)

overfit_coeff = coefficients[0]
optimal_coeff = coefficients[np.argmin(test_errors)]
underfit_coeff = coefficients[-1]

''' 
Plotting begins here 
'''
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
plt.title('Interpolation with variing regularization (lambda)')
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