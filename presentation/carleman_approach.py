import numpy as np
from math import factorial
from numpy import polynomial as Polynomial
import matplotlib.pyplot as plt
import time
import random
from polynomial_utils import compose, compose_layers, l2_norm, l2_coefficient_norm, plot_polynomials


"""def compose(p1, p2):
    # Compose two polynomials (p1(p2))

    coef1 = p1.coef
    coef2 = p2.coef

    max_degree = (len(coef1) - 1) * (len(coef2) - 1)
    result = np.zeros(max_degree + 1)

    # Compute powers of p2 efficiently
    power = np.ones(1)  # p2^0 = 1
    for i, c in enumerate(coef1):
        if c != 0:  # Skip zero coefficients
            result[:len(power)] += c * power
        if i < len(coef1) - 1:  # Don't compute unnecessary power
            power = np.convolve(power, coef2)

    # Trim trailing zeros and create a Polynomial object
    return Polynomial.Polynomial(np.trim_zeros(result, 'b'))


def compose_layers(layers):
    # Compose a list of polynomials in order, where each poly is applied in
    # order (meaning given [p1, p2, p3], the output is p3(p2(p1(x))))
    r = layers[0]
    for i in range(1, len(layers)):
        r = compose(layers[i], r)
    return r


def plot_polynomials(comp, target, iteration):
    x_vals = np.linspace(0, 1, 200)
    y_comp = comp(x_vals)
    y_target = target(x_vals)

    plt.clf()
    plt.plot(x_vals, y_comp, label="Composed Polynomial", color='blue')
    plt.plot(x_vals, y_target, label="Target Polynomial",
             color='red', linestyle='--')
    plt.title(f"Iteration {iteration}")
    plt.legend()
    plt.show()
    # plt.pause(0.5)


def l2_norm(p1, p2):
    # Extract coefficients as NumPy arrays
    c1 = p1.coef
    c2 = p2.coef

    # Ensure both polynomials have the same degree
    max_degree = max(len(c1), len(c2))
    c1 = np.pad(c1, (0, max_degree - len(c1)))
    c2 = np.pad(c2, (0, max_degree - len(c2)))

    # Compute the difference of coefficients
    diff = c1 - c2

    # Compute the first part of the sum
    i = np.arange(max_degree)
    r1 = np.sum(diff**2 / (2*i + 1))

    # Compute the second part of the sum
    i, j = np.meshgrid(i, i)
    mask = i > j
    r2 = 2 * np.sum(diff[i[mask]] * diff[j[mask]] / (i[mask] + j[mask] + 1))

    return (r1 + r2)**0.5


def l2_coefficient_norm(p1, p2):
    # Extract coefficients as NumPy arrays
    c1 = p1.coef
    c2 = p2.coef

    # Ensure both polynomials have the same degree
    max_degree = max(len(c1), len(c2))
    c1 = np.pad(c1, (0, max_degree - len(c1)))
    c2 = np.pad(c2, (0, max_degree - len(c2)))

    # Compute the difference of coefficients
    diff = c1 - c2

    return np.linalg.norm(diff, 2)"""


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
        for j in range(m):
            carleman_matrix[i, j] = carleman(i, j, poly)

    return carleman_matrix


def carleman_solver(h, g, target_poly: Polynomial, iteration: int = 10):
    """
    Given a target polynomial, find a polynomial that approximates the target
    polynomial using the Carleman matrix up to the nth row and mth column. If m
    is not provided, the Carleman matrix will be square.
    """
    target_carleman = carleman_matrix(target_poly, 10, 4)
    for i in range(iteration):

        # We only need the first 10 columns of the Carleman matrix, one for
        # each coeff of the target polynomial, and the first 4 rows, one for
        # each coeff of the g polynomial we try to find. We transpose the
        # matrix so that the linalg.solve function can solve the system of
        # equations.
        m_h = carleman_matrix(h, 4, 10).T

        # Solve the system of equations to find the coefficients of the g
        # polynomial. We use lstsq since the system is overdetermined
        g = Polynomial.Polynomial(np.linalg.lstsq(
            m_h, target_poly.coef, rcond=None)[0])

        # Compute the pseudo-inverse of the Carleman matrix of g
        m_g = carleman_matrix(g, 10)
        m_g_inv = np.linalg.pinv(m_g)

        h = Polynomial.Polynomial([m_g_inv[1] @ target_carleman[:, j]
                                   for j in range(4)])

    return h, g


if __name__ == "__main__":
    # target_poly = Polynomial.Polynomial(np.random.uniform(-2.5, 2.5, 10))
    p1 = Polynomial.Polynomial(np.random.uniform(-1.5, 1.5, 4))
    p2 = Polynomial.Polynomial(np.random.uniform(-1.5, 1.5, 4))
    target_poly = compose(p1, p2)

    start_time = time.time()

    """h, g = genetic_alg(target_poly, population_size=100,
            generations=1000, mutation_rate=0.1)"""

    for attempt in range(1000):
        attempt_time = time.time()
        h = Polynomial.Polynomial(np.random.uniform(-0.5, 0.5, 4))
        g = Polynomial.Polynomial(np.random.uniform(-0.5, 0.5, 4))
        h, g = carleman_solver(h, g, target_poly, 10)

        composed = compose_layers([h, g])
        error = l2_coefficient_norm(composed, target_poly)
        print(f"Attempt {attempt} | Error: {error:.4f} | Time: "
              f"{time.time() - attempt_time:.4f}")
        if error < 1e-4:
            break

    print(f"Time: {time.time() - start_time:.4f}")

    # Print the L2 norm between the target polynomial and the composed
    # polynomial
    composed = compose_layers([h, g])
    print(f"L2 Coefficient Norm: {l2_coefficient_norm(composed, target_poly):.4f}"
         f" | L2 Norm: {l2_norm(composed, target_poly):.4f}")
    print(f"Target: {target_poly.coef}")
    print(f"Composed: {composed.coef}")
    plot_polynomials(composed, target_poly, 10)
