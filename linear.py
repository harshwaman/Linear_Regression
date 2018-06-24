import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('abc.txt', delimiter=',')

X = np.c_[np.ones(data.shape[0]),data[:,0]]
y = np.c_[data[:,1]]

plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')

def computeCost(X, y, theta=[[0], [0]]):
    m = y.size
    J = 0

    h = X.dot(theta)
    #print(h);
    J = 1 / (2 * m) * np.sum(np.square(h - y))

    return (J)

#print(computeCost(X,y))

def gradientDescent(X, y, theta=[[0], [0]], alpha=0.01, num_iters=1500):
    m = y.size
    J_history = np.zeros(num_iters)
    print(J_history)
    for iter in np.arange(num_iters):
        h = X.dot(theta)
        theta = theta - alpha * (1 / m) * (X.T.dot(h - y))
        J_history[iter] = computeCost(X, y, theta)
        #print( J_history[iter])
    return (theta, J_history)


# theta for minimized cost J
theta, Cost_J = gradientDescent(X, y)
print('theta: ', theta.ravel())
print(Cost_J)

xx = np.arange(5,23)
yy = theta[0]+theta[1]*xx


# Plot gradient descent
plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
plt.plot(xx,yy, label='Linear regression (Gradient descent)')
plt.show()
