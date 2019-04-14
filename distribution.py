from scipy.stats import rv_continuous
from scipy.stats import uniform, norm, gamma, chi2, lognorm, beta, expon
import numpy as np


class equal_revenue(rv_continuous):
    "equal revenue"
    def _pdf(self, x, *args):
        rep = 1 / x
        return rep**2


def chi2_bids(n):
    return chi2.rvs(4, size=n)


def uniform_bids(n):
    return uniform.rvs(size=n, loc=0, scale=1)


def normal_bids(n):
    b = norm.rvs(size=n, loc=1, scale=1)
    b = b - np.amin(b)
    return b


def equal_revenue_bids(n):
    dis = equal_revenue(a=1)
    return dis.rvs(size=n)


def equal_revenue_bids_with_noise(n):
    dis = equal_revenue(a=1)
    return dis.rvs(size=n) + normal_bids(n)


def exponential_bids(n):
    return expon.rvs(size=n)


def gamma_bids(n):
    a = 1.99
    return gamma.rvs(a, size=n)

