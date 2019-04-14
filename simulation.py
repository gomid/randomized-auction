import random
import time
import math
from distribution import *


def simulate_distributions(distributions):
    """
    Simulate multiple distributions in parallel.
    """
    from joblib import Parallel, delayed
    Parallel(n_jobs=4)(delayed(simulate)(distribution) for distribution in distributions)


def simulate(distribution):
    """
    Sample [10, 40, 70, 100, 400, 700, 1000, 4000, 7000, 10000] data points from the given distribution,
    then evaluate DOT, DSOT, and SCS on these data points.
    """
    num_bids = [pow(10, i) * j for i in range(1, 4) for j in range(1, 10, 3)] + [10000]
    start = time.time()
    with open("{}.txt".format(distribution.__name__), 'w') as f:
        for n in num_bids:
            dsot, scs, dot = evaluate(distribution, n, iterations=20)
            f.write("{},{},{},{}\n".format(n, dsot, scs, dot))
    print("Time: ", time.time() - start)
    return


def simulate_parallel(distribution):
    """
    Sample [10, 40, 70, 100, 400, 700, 1000, 4000, 7000, 10000] data points from the given distribution,
    then evaluate DOT, DSOT, and SCS on these data points.
    Execution is in parallel.
    """
    from joblib import Parallel, delayed
    num_bids = [pow(10, i)*j for i in range(1, 4) for j in range(1, 10, 3)] + [10000]
    Parallel(n_jobs=4)(delayed(evaluate)(distribution, n, 10) for n in num_bids)


def evaluate(dis, n, iterations):
    dsot = 0
    scs = 0
    dot = 0
    for i in range(iterations):
        b = sorted(dis(n), reverse=True)
        _, F = opt(b)
        dsot += DSOT(b) / F
        scs += SCS(b) / F
        dot += DOT(b) / F
    dsot, scs, dot = dsot / iterations, scs / iterations, dot / iterations
    print(n, dsot, scs, dot)
    return dsot, scs, dot


def opt(b):
    # b = sorted(b, reverse=True)
    revenues = np.array([(i+1) * x for i, x in enumerate(b)])
    try:
        idx = np.argmax(revenues)
        optimal = np.amax(revenues)
        return b[idx], optimal
    except ValueError:
        return np.inf, 0


def DOT(b):
    revenue = 0
    for i, x in enumerate(b):
        b_ = np.delete(b, i)
        price, _ = opt(b_)
        if x >= price:
            revenue += price
    return revenue


def DSOT(b, dualPrice = False):
    x, y = partition(b)
    x_price, _ = opt(y)
    y_price, _ = opt(x)
    x_revenue = sum([x_price for val in x if val >= x_price])
    y_revenue = sum([y_price for val in y if val >= y_price])

    if dualPrice:
        return x_revenue + y_revenue
    else:
        return max(x_revenue, y_revenue)


def SCS(b, dualPrice = False):
    x, y = partition(b)
    x_price, x_cost = opt(y)
    y_price, y_cost = opt(x)

    # x = sorted(x, reverse=True)
    # y = sorted(y, reverse=True)

    x_revenue = 0
    for i, bid in enumerate(x):
        if (i + 1) * bid >= x_cost:
            x_revenue = x_cost
            break

    y_revenue = 0
    for i, bid in enumerate(y):
        if (i + 1) * bid >= y_cost:
            y_revenue = y_cost
            break

    if dualPrice or math.isclose(x_price, y_price, rel_tol=1e-6):
        return x_revenue + y_revenue
    else:
        return max(x_revenue, y_revenue)


def partition(b):
    x, y = [], []
    for bid in b:
        if random.random() < 0.5:
            x.append(bid)
        else:
            y.append(bid)
    return x, y


def main():
    distributions = [equal_revenue_bids, normal_bids, uniform_bids]
    # simulate_parallel(equal_revenue_bids)
    simulate_distributions(distributions)
    return


if __name__ == '__main__':
    main()
