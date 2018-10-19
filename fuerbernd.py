import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,1,5)
y = np.linspace(0,1,5) + 0.1

def polynomial(x, coefficients):
    solution = 0
    for k in range(len(coefficients)):
        solution += coefficients[k]*x**k
        
    return solution

coefficients = np.array([0.1,-0.01,-0.001])

#plt.plot(x,y,'o')

X = np.linspace(-3,3,100)
plt.plot(X, polynomial(X,coefficients))