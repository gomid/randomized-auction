from scipy.stats import uniform, norm, gamma, chi2, expon
import numpy as np


def chi2_bids(n):
    return chi2.rvs(4, size=n)


def uniform_bids(n):
    return uniform.rvs(size=n, loc=0, scale=1)


def normal_bids(n):
    b = norm.rvs(size=n, loc=1, scale=1)
    b = b - np.amin(b)
    return b


def equal_revenue_bids(n):
    return [1 for i in range(n-1)] + [n]


def exponential_bids(n):
    return expon.rvs(size=n)


def gamma_bids(n):
    a = 1.99
    return gamma.rvs(a, size=n)

