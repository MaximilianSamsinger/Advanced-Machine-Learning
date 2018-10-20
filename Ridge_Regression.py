import numpy as np
import matplotlib.pyplot as plt
from utils import sample_function, polynomial, RMSerror, linear_regression, \
    partition, optimal_coefficients

np.random.seed(1) # For reproducibility

def cross_validation(x, y, partitions, lambdas, degree = 30):
    averaged_coeff = 0
    averaged_test_error = 0
    for count, test_partition in enumerate(partitions):
        ''' Create training partition by combining all non-test partitions '''
        train_partition = []
        for k in range(len(partitions)):
            if k != count:
                train_partition += partitions[k]
        train_data = (x[train_partition], y[train_partition])
        test_data = (x[test_partition], y[test_partition])
        
        optimal_coeff, optimal_lambda, errors = optimal_coefficients(
                train_data, test_data, lambdas, degree)
        
        averaged_coeff += optimal_coeff
        averaged_test_error += min(errors[1])
        
    averaged_coeff /= len(partitions)
    averaged_test_error /= len(partitions)
    return averaged_coeff, averaged_test_error


'''
Define Parameters here
'''
N = 12  # Number of points to be sampled
M = 9  # Maximal interpolation degree
mean, sigma = 0, 1e-1 # Mean and standard deviation of the error term
K = 4 # Number of partitions


def f(x):
    return np.sin(2*np.pi*x)

x, y  = sample_function(f, N, mean, sigma)
_, y_test = sample_function(f, N, mean, sigma)


loglambdas = np.linspace(-35,-1,300)
lambdas = np.exp(loglambdas)

optimal_coeff, optimal_lambda, errors = optimal_coefficients(
        (x,y), (x,y_test), lambdas, degree = M)

train_errors, test_errors = errors

partitions = partition(x,K)
cv_coeff, cv_test_error = cross_validation(
        x, y, partitions, lambdas, degree = 30)

X = np.linspace(0,1,500)

plt.figure(figsize=(19, 9))
plt.subplot(131)
plt.plot(x,y,'o', label='training data')
plt.plot(x,y_test,'o', label='test data')
plt.plot(X, polynomial(X, linear_regression(x,y, M, 0)), 'r', label='interpolation polynomial')
plt.plot(X, f(X), 'g--', label='ground truth')
plt.title('Interpolation polynomial without regularization' 
          + '\ntest error= ' + str(round(RMSerror(x,y_test, linear_regression(x,y, M, 0)),4)))
plt.legend()

plt.subplot(132)
plt.plot(x,y,'o', label='training data')
plt.plot(x,y_test,'o', label='test data')
plt.plot(X, polynomial(X, optimal_coeff), 'r', label='interpolation polynomial')
plt.plot(X, f(X), 'g--', label='ground truth')
plt.title('Interpolation polynomial with (optimal) regularization' 
          + '\ntest error= ' + str(round(RMSerror(x,y_test, optimal_coeff),4)))
plt.legend()

plt.subplot(133)
plt.plot(x,y,'o', label='data')
plt.plot(X, polynomial(X, cv_coeff), 'r', label='interpolation polynomial')
plt.plot(X, f(X), 'g--', label='ground truth')
plt.title('Cross validated interpolation polynomial with (optimal) regularization' 
          + '\ntest error= ' + str(round(cv_test_error,4)))
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