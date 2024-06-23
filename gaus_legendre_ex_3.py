import numpy as np
from scipy.special import legendre


def ztransform(x):
    '''
    x: original variable, going from [0,1]
    return: back transformed variable, going from [a,b]
    '''
    return (1+x) / (1-x)

def legendre_dev(n,x):
    '''
    n: order of legendre polynomial
    x: position of evaluation
    returns: derivative of legendre polynomial of order n, at position x
    '''
    return ((n+1)*x*legendre(n)(x) - (n+1)*legendre(n+1)(x))/(1 - x**2)


def get_weight(n,x):
    '''
    n: order of legendre polynomial
    x: poition of legendre polynomial
    returns: associated weight
    '''
    return 2 / (1 - x**2) / legendre_dev(n,x)**2

def get_zeros_legendre(n, zi, max_it: int = 20):
    eps = 1e-12
    zj = zi + 1
    for it in range(max_it):
        zj = zi - legendre(n)(zi) / legendre_dev(n, zi)
        if np.abs((zj - zi)) < eps:
            break
        zi = zj
    return zj


def gauss_legendre(f, n: int = 101):
    weights = np.zeros(n)
    zeros_x = np.zeros(n)
    for i in range(n):
        initial_guess = np.cos(np.pi * (i - 0.75) / (n + 0.5))
        xi = get_zeros_legendre(n, zeros_x[i])
        zeros_x[i] = xi
        weights[i] = get_weight(n, xi)

    zeros_z = ztransform(zeros_x)
    list = [2 * (2 / (1-x)**2) * w * f(z) for w, z, x in zip(weights, zeros_z, zeros_x)]
    S = np.sum(list)
    return S


