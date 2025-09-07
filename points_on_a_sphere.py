import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
from math import factorial, comb
from numpy import polynomial as Polynomial
import matplotlib.pyplot as plt
import time
import random
from polynomial_utils import compose, compose_layers, l2_norm, l2_coefficient_norm, plot_polynomials, carleman, carleman_matrix, l_inf_coefficient_norm
from tqdm import tqdm
import traceback
import multiprocessing as mp
from functools import partial


# g_ratio = (1 + np.sqrt(5)) / 2

# n = 1000

# arr = [(i / g_ratio, i/n) for i in range(n)]
# # arr = [(2*np.pi*i, np.arccos(1 - 2*j)) for i, j in arr]


# # Convert the points to 3D coordinates on a sphere
# x, y, z = [], [], []
# for a, b in arr:
#     # Convert from spherical to cartesian coordinates

#     theta = 2 * np.pi * a
#     phi = np.arccos(1 - 2 * b)

#     x_i = np.cos(theta) * np.sin(phi)
#     y_i = np.sin(theta) * np.sin(phi)
#     z_i = np.cos(phi)

#     x.append(x_i)
#     y.append(y_i)
#     z.append(z_i)

# # Create 3D plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, c=range(n), cmap='viridis')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Points on a Sphere')
# plt.tight_layout()
# plt.show()

# Go here for explanation
# https://math.stackexchange.com/questions/3291489/can-the-fibonacci-lattice-be-extended-to-dimensions-higher-than-3
# TODO: implement full Fibonacci lattice but the guy's explanation is not clear

def polar_to_cartesian(polar):
    # polar = (r, theta_1, theta_2, ..., theta_n)

    # Convert polar coordinates to Cartesian coordinates
    r = polar[0]
    theta = polar[1:]
    n = len(theta)

    # Initialize the first coordinate
    cartesian = [r * np.cos(theta[0])]

    # Calculate all but the last coordinate
    for i in range(n-1, 0, -1):
        r *= np.sin(theta[i])
        cartesian.append(r * np.cos(theta[i-1]))

    # Calculate the last coordinate
    cartesian.append(r * np.sin(theta[0]))

    return cartesian


print(polar_to_cartesian((1, np.pi/4, np.pi/4, np.pi/4)))



# For now just generate random points on n-sphere
def random_points_on_sphere(n, d):
    # Generate n random points on a d-dimensional sphere
    points = np.random.randn(n, d)
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]

    # Cast all point on bottom half to top half
    #points[:, 0] = np.abs(points[:, 0])
    return points






###############################

def solve_for_h(g, target_poly):

    h_deg = target_poly.degree() / g.degree()
    if h_deg != int(h_deg):
        raise ValueError(
            "Degree of target poly should be a multiple of g degree")
    h_deg = int(h_deg)

    # Find shift variable d
    h0_poly = g - target_poly.coef[0]
    roots = h0_poly.roots()
    real_roots = [root.real for root in roots if np.isreal(root)]
    if len(real_roots) == 0:
        raise ValueError("No real roots found")

    # d = min(real_roots, key=abs)

    solutions = []
    for d in real_roots:

        # Shift g
        shifted_g0 = g(d) - d
        shifted_gl = []
        for l in range(1, g.coef.size):
            shifted_gl.append(
                sum([g.coef[j] * d**(j-l) * comb(j, l) for j in range(l, g.coef.size)]))
        # satisfies g(x+d)-d = shifted_g(x)
        shifted_g = Polynomial.Polynomial([shifted_g0] + shifted_gl)

        h = Polynomial.Polynomial([0])
        for i in range(h_deg):
            h_powers = [h ** j for j in range(i+2)]
            mh_row = np.array(
                [h_powers[j].coef[i+1] if h_powers[j].coef.size > i+1 else 0 for j in range(2, i+2)])
            hi = (target_poly.coef[i+1] - shifted_g.coef[2:2+min(i, len(mh_row))] @ mh_row[:min(
                len(mh_row), len(shifted_g.coef[2:2+min(i, len(mh_row))]))]) / shifted_g.coef[1]
            h = Polynomial.Polynomial(list(h.coef) + [hi])

        h = h + d
        solutions.append(h)
    # Is there a better way to choose the best h?

    h = min(solutions, key=lambda h: l2_coefficient_norm(
        compose(g, h), target_poly))
    return h


def carleman_upper_triangular_solver(h, g, target_poly: Polynomial, iteration: int = 10, size: int = 10, w=None, verbose=False):
    """
    Given a target polynomial, find a polynomial that approximates the target
    polynomial using the Carleman matrix up to the nth row and mth column. If m
    is not provided, the Carleman matrix will be square.
    """

    m, n, q = g.coef.size, h.coef.size, target_poly.coef.size

    if verbose:
        print(f"g: {g}")
        print(f"h: {h}")
    for i in range(iteration):

        # We only need the first 10 columns of the Carleman matrix, one for
        # each coeff of the target polynomial, and the first 4 rows, one for
        # each coeff of the g polynomial we try to find. We transpose the
        # matrix so that the linalg.solve function can solve the system of
        # equations.
        m_h = carleman_matrix(h, m, q).T

        # Solve the system of equations to find the coefficients of the g
        # polynomial. We use lstsq since the system is overdetermined

        g = Polynomial.Polynomial(np.linalg.lstsq(
            m_h, target_poly.coef, rcond=None)[0])

        # if verbose:
        #     composed = compose_layers([h, g])
        #     plot_polynomials(composed, target_poly, i+0.5)

        h = solve_for_h(g, target_poly)

        if verbose:
            print(f"g: {g}")
            print(f"h: {h}")

            composed = compose(g, h)
            plot_polynomials(composed, target_poly, i, linspace_range=(0, 1))

    if verbose:
        print()
    return h, g


def new_poly(width, m=3, n=3):
    p1 = Polynomial.Polynomial(np.random.uniform(-width, width, m+1))
    p2 = Polynomial.Polynomial(np.random.uniform(-width, width, n+1))
    for i in range(n//2):
        p2.coef[i+1] = 0

    target_poly = compose(p1, p2)

    return p1, p2, target_poly


DEG = 11
NUM_SAMPLES = 1000
ATTEMPT_PER_SAMPLE = 10


def process_poly(target_poly):
    try:
        # Generate a random polynomial
        g0 = Polynomial.Polynomial(np.zeros(DEG+1))
        best_error = float('inf')
        tries = 0

        points = random_points_on_sphere(ATTEMPT_PER_SAMPLE, DEG+1)
        for j in range(ATTEMPT_PER_SAMPLE):
            tries += 1
            # Generate random polynomials for h and g
            # h0 = Polynomial.Polynomial(np.random.uniform(-0.5, 0.5, 4))
            h0 = Polynomial.Polynomial(points[j])

            # Solve for h and g using the Carleman upper triangular solver
            h, g = carleman_upper_triangular_solver(
                h0, g0, target_poly, iteration=10, size=10)

            # Calculate and store the error
            error = l_inf_coefficient_norm(compose(g, h), target_poly)

            if error < best_error:
                best_error = error

            if j == 0:
                first_attempt = error

            if error < 1e-14:
                continue
                break
        return best_error, tries, first_attempt

    except Exception as e:
        # Skip failed attempts but keep track
        print(f"Skipping sample due to error: {e}")
        return None, None, None


if __name__ == "__main__":
    # Set the random seed for reproducibility
    np.random.seed(42)

    # Initialize an array to store errors
    errors = []

    # Use multiprocessing to speed up the experiment

    # Create a function that generates a random polynomial and processes it
    targets = []
    for i in range(NUM_SAMPLES):
        p1, p2, target_poly = new_poly(1, DEG, DEG)
        targets.append(target_poly)

    # Run the experiment NUM_SAMPLES times using a process pool
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(process_poly, targets),
            total=NUM_SAMPLES,
            desc="Computing polynomial decompositions"
        ))

    failed_attempts = sum(1 for result in results if result[0] is None)
    filtered_results = [result for result in results if result[0] is not None]
    errors, tries, first_attempts = zip(*filtered_results)
    # Filter out None values (failed attempts)
    # errors = [e for (e, t) in results if e is not None]

    # Print some statistics
    errors = np.array(errors)
    first_attempts = np.array(first_attempts)
    print(f"Completed {len(errors)}/{NUM_SAMPLES} successful decompositions")
    print(f"Mean error: {np.mean(errors)}")
    print(f"Median error: {np.median(errors)}")
    print(f"Min error: {np.min(errors)}")
    print(f"Max error: {np.max(errors)}")
    print(f"Failed attempts: {failed_attempts}")
    print(f"Mean tries per sample: {np.mean(tries)}")
    print(f"Median tries per sample: {np.median(tries)}")
    print("\nFirst attempt statistics:")
    print(f"Mean first attempt error: {np.mean(first_attempts)}")
    print(f"Median first attempt error: {np.median(first_attempts)}")
    print(f"Min first attempt error: {np.min(first_attempts)}")
    print(f"Max first attempt error: {np.max(first_attempts)}")

    # Calculate the bin range to include both errors and first attempts
    min_value = min(np.min(errors), np.min(first_attempts))
    max_value = max(np.max(errors), np.max(first_attempts))
    bins = np.logspace(np.log10(min_value), np.log10(max_value), 50)

    # Plot the histograms of errors and first attempts on a log scale
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # First attempts histogram
    ax1.hist(first_attempts, bins=bins)
    ax1.set_xscale('log')
    ax1.set_xlabel('First Attempt Error (L2 norm)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of First Attempt Errors')
    ax1.grid(True, alpha=0.3)

    # Final errors histogram
    ax2.hist(errors, bins=bins)
    ax2.set_xscale('log')
    ax2.set_xlabel('Final Error (L2 norm)')
    ax2.set_title('Distribution of best of 10 Errors')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
