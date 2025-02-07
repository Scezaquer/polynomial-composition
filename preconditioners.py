import numpy as np
from math import factorial
from numpy import polynomial as Polynomial
from numpy.linalg import lstsq, inv
import matplotlib.pyplot as plt
from polynomial_utils import compose, compose_layers, l2_norm, l2_coefficient_norm, plot_polynomials
from tqdm import tqdm


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
        for j in range(m):
            carleman_matrix[i, j] = poly_j.coef[j] if j < len(
                poly_j.coef) else 0

    return carleman_matrix


def carleman_solver(h, g, target_poly: Polynomial, iteration: int = 10,
                    size: int = 10, preconditioner='none', verbose=False):
    """
    Given a target polynomial, find a polynomial that approximates the target
    polynomial using the Carleman matrix up to the nth row and mth column.

    Args:
        h: Initial guess for polynomial h.
        g: Initial guess for polynomial g.
        target_poly: Target polynomial.
        iteration: Number of iterations.
        size: Size of the Carleman matrix.
        preconditioner: Type of preconditioner to use ('none', 'diagonal', 'jacobi').
        verbose: Whether to print intermediate results.

    Returns:
        h: Updated polynomial h.
        g: Updated polynomial g.
    """
    target_carleman = carleman_matrix(target_poly, size, 4)
    if verbose:
        print(f"g: {g}")
        print(f"h: {h}")

    for i in range(iteration):
        m_h = carleman_matrix(h, 4, 10).T

        # Apply preconditioner
        g = Polynomial.Polynomial(np.linalg.lstsq(
            m_h, target_poly.coef, rcond=None)[0])

        m_g = carleman_matrix(g, size)
        m_g_inv = np.linalg.pinv(m_g)

        h = Polynomial.Polynomial([m_g_inv[1] @ target_carleman[:, j]
                                  for j in range(4)])

        if verbose:
            print(f"g: {g}")
            print(f"h: {h}")

    return h, g


# Test different preconditioners
num_samples = 200
errors = {'none': [], 'diagonal': []}

for _ in tqdm(range(num_samples)):
    h0 = Polynomial.Polynomial(np.random.uniform(-0.5, 0.5, 4))
    g0 = Polynomial.Polynomial(np.random.uniform(-0.5, 0.5, 4))

    p1 = Polynomial.Polynomial(np.random.uniform(-1.5, 1.5, 4))
    p2 = Polynomial.Polynomial(np.random.uniform(-1.5, 1.5, 4))
    target_poly = compose(p1, p2)

    for preconditioner in errors.keys():
        h, g = carleman_solver(
            h0, g0, target_poly, iteration=2, size=10,
            preconditioner=preconditioner)
        error = l2_coefficient_norm(compose(h, g), target_poly)
        errors[preconditioner].append(error)

plt.figure(figsize=(15, 10))

for i, (preconditioner, error_list) in enumerate(errors.items(), 1):
    plt.subplot(3, 1, i)
    plt.hist(error_list, bins=np.logspace(np.log10(min(error_list)),
             np.log10(max(error_list)), 50), alpha=0.75)
    plt.xscale('log')
    plt.xlabel('L2 Norm Error')
    plt.ylabel('Frequency')
    plt.title(f'Preconditioner: {preconditioner}')

plt.tight_layout()
plt.show()
