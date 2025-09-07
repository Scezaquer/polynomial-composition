import numpy as np
import matplotlib.pyplot as plt
from numpy import polynomial as Polynomial
from math import factorial


def compose(p1, p2, dtype=np.float64):
    """
    Return the composition of two polynomials, p1(p2).
    """

    coef1 = p1.coef
    coef2 = p2.coef

    max_degree = (len(coef1) - 1) * (len(coef2) - 1)
    result = np.zeros(max_degree + 1, dtype=dtype)

    # Compute powers of p2 efficiently
    power = np.ones(1, dtype=dtype)  # p2^0 = 1
    for i, c in enumerate(coef1):
        if c != 0:  # Skip zero coefficients
            result[:len(power)] += c * power
        if i < len(coef1) - 1:  # Don't compute unnecessary power
            power = np.convolve(power, coef2)

    # Trim trailing zeros and create a Polynomial object
    return Polynomial.Polynomial(result)


def compose_layers(layers):
    # Compose a list of polynomials in order, where each poly is applied in
    # order (meaning given [p1, p2, p3], the output is p3(p2(p1(x))))
    r = layers[0]
    for i in range(1, len(layers)):
        r = compose(layers[i], r)
    return r


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

    return r1 + r2


def plot_polynomials(comp, target, iteration, linspace_range=(0, 1)):
    x_vals = np.linspace(linspace_range[0], linspace_range[1], 200)
    y_comp = comp(x_vals)
    y_target = target(x_vals)

    plt.clf()
    plt.plot(x_vals, y_comp, label="Composed Polynomial", color='blue')
    plt.plot(x_vals, y_target, label="Target Polynomial",
             color='red', linestyle='--')
    plt.title(f"Iteration {iteration}")
    plt.legend()
    #plt.show()
    plt.pause(0.5)


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

    return np.linalg.norm(diff, 2)


def l_inf_norm(p1, p2):
    pts = np.linspace(0, 1, 2000)
    return np.max(np.abs(p1(pts) - p2(pts)))


def l_inf_coefficient_norm(p1, p2):
    return np.max(np.abs(p1.coef - p2.coef))


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