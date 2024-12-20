import numpy as np
from numpy import polynomial as P
import matplotlib.pyplot as plt


def compose(p1, p2):
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
    return P.Polynomial(np.trim_zeros(result, 'b'))


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
    #plt.show()
    plt.pause(0.05)


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