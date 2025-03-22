import numpy as np
from math import factorial, comb
from numpy import polynomial as Polynomial
import matplotlib.pyplot as plt
import time
import random
from polynomial_utils import compose, compose_layers, l2_norm, l2_coefficient_norm, plot_polynomials, carleman, carleman_matrix, l_inf_coefficient_norm
from tqdm import tqdm
import traceback
from multiprocessing import Pool, cpu_count


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

        # turns out this doesn't use q[0] so we can just use target_poly instead of shifted_q
        # h1 = target_poly.coef[1] / shifted_g.coef[1]
        # h2 = (target_poly.coef[2] - shifted_g.coef[2] * h1**2) / shifted_g.coef[1]
        # h3 = (target_poly.coef[3] - shifted_g.coef[3] * h1**3 - shifted_g.coef[2] * 2 * h1 * h2) / shifted_g.coef[1]
        # h = Polynomial.Polynomial([d, h1, h2, h3])

        solutions.append(h)
    # Is there a better way to choose the best h?

    # print(np.argmin(np.abs(real_roots)), np.argmin([l2_coefficient_norm(compose(g, h), target_poly) for h in solutions]))
    # print(real_roots)
    # print([l2_coefficient_norm(compose(g, h), target_poly) for h in solutions])
    # print()
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


PIXEL_WIDTH = 1000
PIXEL_HEIGHT = 1000
ITER = 2
WIDTH = 15

if __name__ == "__main__":
    np.random.seed(2)  # for (3, 3), 5 is a good seed where x and y are h1 and h2

    start = time.time()

    deg_h, deg_g = (5, 5)  # (deg_h, deg_g)

    x = np.linspace(-WIDTH, WIDTH, PIXEL_WIDTH)
    y = np.linspace(-WIDTH, WIDTH, PIXEL_HEIGHT)

    p1, p2, target_poly = new_poly(1.5, deg_h, deg_g)

    # Create image array (white by default)
    image = np.ones((PIXEL_HEIGHT, PIXEL_WIDTH))

    # Function to process a single pixel
    def process_pixel(args):
        i, j, i_idx, j_idx = args
        p2_copy = Polynomial.Polynomial(p2.coef.copy())
        p2_copy.coef[3] = i
        p2_copy.coef[4] = j

        # target_poly = compose(p1, p2_copy)

        try:
            h, g = carleman_upper_triangular_solver(
                p2_copy, p1, target_poly, ITER, verbose=False)

            composed = compose(g, h)
            error = l2_coefficient_norm(composed, target_poly)

            return (j_idx, i_idx, error)
            # If error is small enough (converged), return black pixel coords
            if error <= 1e-6:
                return (j_idx, i_idx, 0)  # Black pixel for convergence
            return None
        except Exception:
            return None  # White pixel for non-convergence

    # Generate all tasks
    tasks = []
    for i_idx, i in enumerate(x):
        for j_idx, j in enumerate(y):
            tasks.append((i, j, i_idx, j_idx))

    # Use multiprocessing to distribute the tasks
    print(f"Processing {len(tasks)} pixels using {cpu_count()} cores...")
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_pixel, tasks), total=len(tasks)))
    
    # Update the image based on results
    for result in results:
        if result is not None:
            j_idx, i_idx, value = result
            image[j_idx, i_idx] = value

    # Plot the image
    plt.figure(figsize=(10, 10))
    # Add a small constant to avoid log(0) errors and apply log transform
    log_image = np.log1p(image)  # np.log1p = log(1+x)
    plt.imshow(log_image, cmap='viridis', extent=[-WIDTH, WIDTH, -WIDTH, WIDTH], norm=plt.cm.colors.LogNorm())
    plt.colorbar(label='Log Error (lower is better)')
    plt.title('Convergence Map (Log Scale)')
    plt.xlabel('p2.coef[1]')
    plt.ylabel('p2.coef[2]')
    plt.savefig('convergence_map.png', dpi=300)
    plt.show()

    print(f"Runtime: {time.time() - start:.2f} seconds")
