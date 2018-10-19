import numpy as np
import matplotlib.pyplot as plt


#np.random.seed(1337) # For reproducibility

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


'''
Define Parameters here
'''
N = 50  # Number of points to be sampled
M = 20  # Maximal interpolation degree
mean, sigma = 0, 1e-1 # Mean and standard deviation of the error term

def f(x):
    return np.sin(2*np.pi*x)

x, y  = sample_function(f, N, mean, sigma)
_, y_test = sample_function(f, N, mean, sigma)

loglambdas = np.linspace(-30,-1,300)
lambdas = np.exp(loglambdas)

training_errors = np.zeros(len(lambdas))
test_errors = np.zeros(len(lambdas))
for k in range(len(lambdas)):
    coefficients = linear_regression(x,y, regularizer = lambdas[k])
    training_errors[k] = RMSerror(x,y, coefficients)
    test_errors[k] = RMSerror(x,y_test, coefficients)

X = np.linspace(0,1,500)
optimal_training_lambda = lambdas[np.argmin(training_errors)]
optimal_training_coeff = linear_regression(x,y, M, optimal_training_lambda)
optimal_test_lambda = lambdas[np.argmin(test_errors)]
optimal_test_coeff = linear_regression(x,y, M, optimal_test_lambda)

plt.figure(figsize=(20, 10))
plt.subplot(131)
plt.plot(x,y,'o', label='training data')
plt.plot(X, polynomial(X, optimal_training_coeff), 'r', label='interpolation polynomial')
plt.plot(X, f(X), 'g--', label='ground truth')
plt.title('Interpolation polynomial minimizes training error= '
          + str(round(RMSerror(x,y, optimal_training_coeff),4)))
plt.legend()

plt.subplot(132)
plt.plot(x,y,'o', label='training data')
plt.plot(X, polynomial(X, optimal_test_coeff), 'r', label='interpolation polynomial')
plt.plot(X, f(X), 'g--', label='ground truth')
plt.title('Interpolation polynomial minimizes test error= ' 
          + str(round(RMSerror(x,y_test, optimal_test_coeff),4)))
plt.legend()
   
plt.subplot(133)
plt.plot(loglambdas, training_errors, 'b', label='training')
plt.plot(loglambdas, test_errors, 'r', label='test')
plt.title('Error vs log lambda')
plt.xlabel('log lambda')
plt.ylabel('E_RMS')
plt.legend()
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, 0, 0.6))