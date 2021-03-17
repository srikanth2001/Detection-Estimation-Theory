
import numpy as np
import matplotlib.pyplot as plt

mu0, mu1, sigma0, sigma1 = 3, -1, 1, 1
N = int(1e6)

def computepx(x):
    return np.exp(-abs(x))/2

def computeGaussian(x, mean, var):
    return (1/np.sqrt(2 * np.pi * var)) * np.exp((-(x - mean) ** 2)/(2 * var))

def p0(y1, y2):
    return computeGaussian(y1, mu0, sigma0 ** 2) * computeGaussian(y2, mu1, sigma1 ** 2)

def p1(y1, y2):
    return computepx(y1) * computepx(y2)

def L(Y):
    y1, y2 = Y[0], Y[1]
    return p1(y1, y2)/p0(y1, y2)

V = []
pi = np.arange(0.1, 1, 0.1)
for pi0 in pi:
    choice = np.array(np.random.rand(N) < pi0, dtype = np.int64) # Generate a choice of indices
    Y = np.zeros((N, 2))
    cost  = 0
    tau = pi0/(1 - pi0)
    for i in range(N):
        if choice[i]: # Choose Y according to H0
            Y[i] = [np.random.normal(mu0, sigma0), np.random.normal(mu1, sigma1)]
            if L(Y[i]) >= tau:
                cost += 1
        else: # Choose Y according to H1
            Y[i] = [np.random.laplace(0, 1), np.random.laplace(0, 1)]
            if L(Y[i]) < tau:
                cost += 1
    
    V.append(cost / N) # Average cost

plt.plot(pi, V)
plt.xlabel("$\pi_0$")
plt.ylabel("$V(\pi_0)$")
plt.grid()
# plt.savefig("./V.png")
plt.show()