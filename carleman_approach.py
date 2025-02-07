import numpy as np
from math import factorial
from numpy import polynomial as Polynomial
import matplotlib.pyplot as plt
import time
import random
from polynomial_utils import compose, compose_layers, l2_norm, l2_coefficient_norm, plot_polynomials


def carleman(j: int, k: int, poly: Polynomial):
    """
    Given a polynomial, return the elements of the Carleman matrix at the jth
    column and kth row.
    """

    # Take jth power of the polynomial
    poly_j = poly**j

    # Take the kth derivative of the jth power of the polynomial
    poly_j_k = poly_j.deriv(k)

    # Evaluate the kth derivative of the jth power of the polynomial at 0
    return 1/factorial(k)*poly_j_k(0)


def carleman_matrix(poly: Polynomial, n: int, m: int = 0):
    """
    Given a polynomial, return the Carleman matrix of the polynomial up to the
    nth row and mth column. If m is not provided, the Carleman matrix will be
    square.
    """
    if m == 0:
        m = n

    # Initialize the Carleman matrix
    carleman_matrix = np.zeros((n, m))

    # Fill the Carleman matrix
    for i in range(n):
        poly_j = poly**i
        carleman_matrix[i, :len(poly_j.coef)] = poly_j.coef[:m]  # marginally faster than a for loop
        # for j in range(m):
        #     carleman_matrix[i, j] = poly_j.coef[j] if j < len(poly_j.coef) else 0
        #     # carleman_matrix[i, j] = carleman(i, j, poly)

    return carleman_matrix


def carleman_solver(h, g, target_poly: Polynomial, iteration: int = 10, size: int = 10, w=None, verbose=False):
    """
    Given a target polynomial, find a polynomial that approximates the target
    polynomial using the Carleman matrix up to the nth row and mth column. If m
    is not provided, the Carleman matrix will be square.
    """
    target_carleman = carleman_matrix(target_poly, size, 4)
    if verbose:
        print(f"g: {g}")
        print(f"h: {h}")
    for i in range(iteration):

        # We only need the first 10 columns of the Carleman matrix, one for
        # each coeff of the target polynomial, and the first 4 rows, one for
        # each coeff of the g polynomial we try to find. We transpose the
        # matrix so that the linalg.solve function can solve the system of
        # equations.
        m_h = carleman_matrix(h, 4, 10).T

        # Solve the system of equations to find the coefficients of the g
        # polynomial. We use lstsq since the system is overdetermined

        if w is None:
            g = Polynomial.Polynomial(np.linalg.lstsq(
                m_h, target_poly.coef, rcond=None)[0])
        else:
            m_hw = np.sqrt(w[:, np.newaxis]) * m_h
            t_w = target_poly.coef * np.sqrt(w)

            g = Polynomial.Polynomial(np.linalg.lstsq(
                m_hw, t_w, rcond=None)[0])

        # g = Polynomial.Polynomial(np.linalg.inv(m_h.T @ m_h) @ m_h.T @ target_poly.coef)

        # composed = compose_layers([h, g])
        # plot_polynomials(composed, target_poly, i)

        # Compute the pseudo-inverse of the Carleman matrix of g
        m_g = carleman_matrix(g, size)
        m_g_inv = np.linalg.pinv(m_g)

        h = Polynomial.Polynomial([m_g_inv[1] @ target_carleman[:, j]
                                   for j in range(4)])

        if verbose:
            print(f"g: {g}")
            print(f"h: {h}")

            composed = compose_layers([h, g])
            plot_polynomials(composed, target_poly, i, linspace_range=(0, 1))

    if verbose:
        print()
    return h, g


# if __name__ == "__main__":
#     # target_poly = Polynomial.Polynomial(np.random.uniform(-2.5, 2.5, 10))
#     p1 = Polynomial.Polynomial(np.random.uniform(-1.5, 1.5, 4))
#     p2 = Polynomial.Polynomial(np.random.uniform(-1.5, 1.5, 4))
#     target_poly = compose(p1, p2)

#     start_time = time.time()

#     """h, g = genetic_alg(target_poly, population_size=100,
#             generations=1000, mutation_rate=0.1)"""

#     # for attempt in range(1000):
#     #     attempt_time = time.time()
#     #     h0 = Polynomial.Polynomial(np.random.uniform(-0.5, 0.5, 4))
#     #     g0 = Polynomial.Polynomial(np.random.uniform(-0.5, 0.5, 4))
#     #     h, g = carleman_solver(h0, g0, target_poly, 10)

#     #     composed = compose_layers([h, g])
#     #     error = l2_coefficient_norm(composed, target_poly)
#     #     print(f"Attempt {attempt} | Error: {error:.4f} | Time: {
#     #           time.time() - attempt_time:.4f}")
#     #     if error < 1e-4:
#     #         print(h0)
#     #         print(g0)
#     #         print(target_poly)
#     #         break

#     attempt_time = time.time()
#     h0 = Polynomial.Polynomial(np.random.uniform(-0.5, 0.5, 4))
#     g0 = Polynomial.Polynomial(np.random.uniform(-0.5, 0.5, 4))
#     h, g = carleman_solver(h0, g0, target_poly, 100)

#     composed = compose_layers([h, g])
#     error = l2_coefficient_norm(composed, target_poly)
#     print(f"Error: {error:.4f} | Time: {time.time() - attempt_time:.4f}")

#     # print(f"Time: {time.time() - start_time:.4f}")

#     # Print the L2 norm between the target polynomial and the composed
#     # polynomial
#     composed = compose_layers([h, g])
#     print(f"L2 Coefficient Norm: {l2_coefficient_norm(
#         composed, target_poly):.4f} | L2 Norm: {l2_norm(composed, target_poly):.4f}")
#     print(f"Target: {target_poly.coef}")
#     print(f"Composed: {composed.coef}")
#     plot_polynomials(composed, target_poly, 10)

if __name__ == "__main__":
    stats_10 = []
    stats_10_time = []
    stats_100 = []
    stats_100_time = []
    start = time.time()
    for i in range(10000):
        # width = 1e-6  # BEST VALUE FOUND SO FAR
        width = 1.5
        p1 = Polynomial.Polynomial(np.random.uniform(-width, width, 4))
        p2 = Polynomial.Polynomial(np.random.uniform(-width, width, 4))
        # p1.coef[0] = 0
        # p2.coef[0] = 0
        target_poly = compose(p1, p2)
        # target_poly = Polynomial.Polynomial(np.random.uniform(-5, 5, 10))

        h0 = Polynomial.Polynomial(np.random.uniform(-0.5, 0.5, 4))
        g0 = Polynomial.Polynomial(np.random.uniform(-0.5, 0.5, 4))
        # h0.coef[0] = 0
        # g0.coef[0] = 0

        # factor = 10**6
        # small_target_poly = target_poly / factor

        s1 = time.time()
        h, g = carleman_solver(h0, g0, target_poly, 10, verbose=False)
        stats_10_time.append(time.time() - s1)

        # g = g * factor

        composed = compose_layers([h, g]) # * factor
        error = l2_coefficient_norm(composed, target_poly)
        # print(f"Attempt {i} | Error: {error:.4f}")
        stats_10.append(error)

        # s2 = time.time()
        # h, g = carleman_solver(h0, g0, target_poly, 10)
        # stats_100_time.append(time.time() - s2)
        # composed = compose_layers([h, g])
        # error = l2_coefficient_norm(composed, target_poly)
        # stats_100.append(error)
        # print(f"Attempt {i} | Error: {error:.4f}")
        # print()
        if i % 100 == 0:
            print(i, end='\r')

    print(f"Time: {time.time() - start:.4f}")

    # Calculate statistics for stats_10
    mean_10 = np.mean(stats_10)
    median_10 = np.median(stats_10)
    variance_10 = np.var(stats_10)

    # Calculate statistics for stats_100
    mean_100 = np.mean(stats_100)
    median_100 = np.median(stats_100)
    variance_100 = np.var(stats_100)

    # Print statistics
    print(f"stats_10 - Mean: {mean_10}, Median: {median_10}, Variance: {variance_10}, Time: {np.mean(stats_10_time)}")
    print(f"stats_100 - Mean: {mean_100}, Median: {median_100}, Variance: {variance_100}, Time: {np.mean(stats_100_time)}")

    # Plot histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Determine the common bin edges for both histograms
    all_stats = np.concatenate((stats_10, stats_100))
    bins = np.logspace(np.log10(min(all_stats)), np.log10(max(all_stats)), 50)

    # Plot stats_10 histogram
    axes[0].hist(stats_10, bins=bins)
    axes[0].set_xscale('log')
    axes[0].set_title('stats_10')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')

    # Plot stats_100 histogram
    axes[1].hist(stats_100, bins=bins)
    axes[1].set_xscale('log')
    axes[1].set_title('stats_100')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')

    # Ensure the vertical axis is the same for both graphs
    max_freq = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(0, max_freq)
    axes[1].set_ylim(0, max_freq)

    plt.tight_layout()
    plt.show()
