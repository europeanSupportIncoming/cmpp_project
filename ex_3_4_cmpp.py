import numpy as np
from gaus_legendre_ex_3 import gauss_legendre
from scipy.integrate import quad
import time

#at x = 0: 1/x * I1(x) ~ 1/2

# exercise 3
def get_next_I(I_nu_minus_1, I_nu, x, nu: int = 1):
    '''

    :param I_nu_minus_1: previous Bessel function
    :param I_nu: current Bessel function
    :param x: input value, p^2/omega^2
    :param nu: order of current Bessel function
    :return: Bessel function of order nu + 1
    '''
    if x == 0:
        I_new = I_nu_minus_1(x) - nu
    else:
        I_new = I_nu_minus_1(x) - 2 * nu / x * I_nu(x)
    return I_new


def I0(x):
    '''
    :param x: input value, p^2/omega^2
    :return: exp(-x)*x^1/2*I_0, a rescaled modified Bessel function of the zeroth kind
    '''
    t = x / 3.75
    if abs(x) <= 3.75:
        return np.exp(-x) *  (
                    1 + 3.5156229 * t ** 2 + 3.0899424 * t ** 4 + 1.2067492 * t ** 6 + 0.2659732 * t ** 8 + 0.0045813 * t ** 12)
    else:
        return 1 / np.sqrt(x) * (
                    0.3984228 + 0.01328592 / t + 0.00225319 / t ** 2 - 0.00157565 / t ** 3 + 0.00916281 / t ** 4 - 0.02057706 / t ** 5 + 0.02635537 / t ** 6 - 0.01647633 / t ** 7 + 0.00392377 / t ** 8)


def I1(x):
    '''
    x: input value, p^2/omega^2
    returns: exp(-x)*x^1/2*I_1, a rescaled modified Bessel function of the first kind
    '''
    t = x / 3.75
    if abs(x) <= 3.75:
        # ToDo: find common rescaling for all functions
        # extra factor of x^-1 here
        return np.exp(-x) * x * (
                    1 / 2 + 0.87890594 * t ** 2 + 0.51498869 * t ** 4 + 0.15084934 * t ** 6 + 0.02658733 * t ** 8 + + 0.00301532 * t ** 10 + 0.00032411 * t ** 12)
    else:
        return 1 / np.sqrt(x) * (
                    0.39894228 - 0.03988024 / t - 0.00362018 / t ** 2 + 0.00163801 / t ** 3 - 0.01031 / t ** 4 + 0.02282967 / t ** 5 - 0.02895312 / t ** 6 + 0.01787654 / t ** 7 - 0.00420059 / t ** 8)


def L2(x):
    return get_next_I(I0, I1, x)

#update using modified bessel functions (with factor exp(-|x|)
def update_A(A, B, x, D, omega):
    '''
    :param x: input value, p^2/omega^2
    :param D: interaction strength, static input parameter
    :param omega: sets interaction scale, static input parameter
    :return: updated quark propagator A(x)
    '''
    if x == 0:
        integrand = lambda y: y * A / (y * A ** 2 + B ** 2 / omega ** 2) * 2 / np.exp(
            (np.sqrt(x) + np.sqrt(y)) ** 2) * ((-2 / 3) * y * 1 * I0(2 * np.sqrt(x * y)) + y**2 * 1 - 4 / 3 * y * 1 * L2(2 * np.sqrt(x * y)))
    else:
        integrand = lambda y: y * A / (y * A ** 2 + B ** 2 / omega ** 2) * 2 / np.exp(
        (np.sqrt(x) + np.sqrt(y)) ** 2) * ((-2 / 3) * y * 1 * I0(2 * np.sqrt(x * y)) + (1 + y / x) * np.sqrt(x * y) * 1 * I1(
        2 * np.sqrt(x * y)) - 4 / 3 * y * 1 * L2(2 * np.sqrt(x * y)))
    integral = gauss_legendre(integrand, n=101)
    return 1 + D * omega ** 2 * integral


def update_B(A, B, x, D, omega, m):
    '''

    :param x: input value, p^2/omega^2
    :param D: interaction strength, static input parameter
    :param omega: sets interaction scale, static input parameter
    :return: updated quark propagator B(x)
    '''
    integrand = lambda y: y * B / (y * A ** 2 + B ** 2 / omega ** 2) * 2 / np.exp(
        (np.sqrt(x) + np.sqrt(y)) ** 2) * (
                                      (x + y) * 1 * I0(2 * np.sqrt(x * y)) - 2 * np.sqrt(x * y) * 1 * I1(2 * np.sqrt(x * y)))
    integral = gauss_legendre(integrand, n=101)
    return m + D * omega ** 2 * integral


# exercise 4

# given values
omega = 0.5
D = 16
masses = [0, 0.005, 0.115]
eta = 1 / 2


# get parabola values from imaginary axis
def get_Im_value(re_val: float = 0.0, eta: float = 0.0, M: float = 0.0):
    '''
    :param im_vals: imaginary part of p^2
    :param eta: input value
    :param M: input value
    :return: real part of p^2 corresponding to parabola
    '''
    return re_val ** 2 / 2 - M ** 2 * eta ** 2


precision = 1e-6

# set up container for M values, to get parabolas
M_vals = []

#I0(0) = 1, I1(0) = 0 analytically
# M_vals = -0.1*np.array(masses)
# get M values
t0 = time.time()
for m in masses:
    abserr = 1
    #some initial guesses for A and B
    A = 1.0
    B = 1.0
    while abserr >= precision:
        A_new = update_A(A, B, 0, D, omega)
        B_new = update_B(A, B, 0, D, omega, m)
        abserr = (A_new + B_new - A - B) / 2
        A = A_new
        B = B_new
    M_vals.append(B / 10)

print(M_vals, time.time() - t0)


A_vals_final_container = []
B_vals_final_container = []

t0 = time.time()

for M in M_vals:
    #set up real and imaginary values of the graph
    re_p_min = - eta ** 2 * M ** 2
    re_p_vals = np.linspace(2 * re_p_min, -2 * re_p_min, 10)
    im_p_vals = get_Im_value(re_p_vals, eta=eta, M=M)
    p_vals = np.array([re + im*1j for re, im in zip(re_p_vals, im_p_vals)])
    a_vals = []
    b_vals = []
    for p in p_vals:
        t1 = time.time()
        abserr = 1
        # some initial guesses for A and B
        A = 1.0
        B = 1.0
        while abserr >= precision:
            A_new = update_A(A, B, p, D, omega)
            B_new = update_B(A, B, p, D, omega, m)
            abserr = np.abs(A_new + B_new - A - B) / 2
            A = A_new
            B = B_new
            print(abserr, time.time() - t0)

        a_vals.append(A)
        b_vals.append(B)
        t2 = time.time()
        print(t2- t1)
    print('full stack', t2 - t0)
    A_vals_final_container.append(a_vals)
    B_vals_final_container.append(b_vals)

print(A_vals_final_container, B_vals_final_container)



#(-2/(x-1)**2)