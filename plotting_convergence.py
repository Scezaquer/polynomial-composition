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


PIXEL_WIDTH = 100
PIXEL_HEIGHT = 100
DEPTH = 9
ITER = 2
WIDTH = 15

if __name__ == "__main__":
    np.random.seed(5)

    start = time.time()

    deg_h, deg_g = (5, 5)  # (deg_h, deg_g)

    x = np.linspace(-WIDTH, WIDTH, PIXEL_WIDTH)
    y = np.linspace(-WIDTH, WIDTH, PIXEL_HEIGHT)
    if DEPTH != 1:
        z = np.linspace(-WIDTH, WIDTH, DEPTH)
    else:
        z = [0]

    p1, p2, target_poly = new_poly(1.5, deg_h, deg_g)

    # Create image array (white by default)
    image = np.ones((DEPTH, PIXEL_HEIGHT, PIXEL_WIDTH))

    # Function to process a single pixel
    def process_pixel(args):
        i, j, k, i_idx, j_idx, k_idx = args
        p2_copy = Polynomial.Polynomial(p2.coef.copy())
        p2_copy.coef[2] = k
        p2_copy.coef[3] = i
        p2_copy.coef[4] = j

        # target_poly = compose(p1, p2_copy)

        try:
            h, g = carleman_upper_triangular_solver(
                p2_copy, p1, target_poly, ITER, verbose=False)

            composed = compose(g, h)
            error = l2_coefficient_norm(composed, target_poly)

            return (k_idx, j_idx, i_idx, error)
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
            for k_idx, k in enumerate(z):
                tasks.append((i, j, k, i_idx, j_idx, k_idx))

    # Use multiprocessing to distribute the tasks
    print(f"Processing {len(tasks)} pixels using {cpu_count()} cores...")
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_pixel, tasks), total=len(tasks)))

    # Update the image based on results
    for result in results:
        if result is not None:
            k_idx, j_idx, i_idx, value = result
            image[k_idx, j_idx, i_idx] = value

    # Calculate grid dimensions for the subplots
    grid_rows = int(np.ceil(np.sqrt(DEPTH)))
    grid_cols = int(np.ceil(DEPTH / grid_rows))

    # Create figure with more compact layout
    fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(
        grid_cols * 3, grid_rows * 3), squeeze=False)
    fig.suptitle(
        'Convergence Maps (Log Scale) for Different Depths', fontsize=14)

    # Find the global min/max for consistent color scaling
    all_values = []
    for k_idx in range(DEPTH):
        # Add a small constant to avoid log(0) errors
        log_image = np.log1p(image[k_idx])
        all_values.append(log_image)

    all_values = np.concatenate([img.flatten() for img in all_values])
    vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)

    # Create a single normalization for all plots
    norm = plt.cm.colors.LogNorm(vmin=vmin, vmax=vmax)

    # Create a new figure with more space
    plt.close(fig)  # Close the previous figure
    fig, axs = plt.subplots(grid_rows, grid_cols,
                            # Increased figure size with more space for colorbar
                            figsize=(grid_cols * 4 + 1, grid_rows * 3.5),
                            squeeze=False,
                            constrained_layout=True)  # Use constrained_layout instead of tight_layout

    fig.suptitle('Convergence Maps (Log Scale) for Different Depths',
                 fontsize=16, y=0.98)  # Move title up

    for k_idx in range(DEPTH):
        # Calculate the grid position
        row = k_idx // grid_cols
        col = k_idx % grid_cols

        # Apply log transform
        log_image = np.log1p(image[k_idx])

        # Plot the slice with shared normalization
        im = axs[row, col].imshow(log_image, cmap='viridis',
                                  extent=[-WIDTH, WIDTH, -WIDTH, WIDTH],
                                  norm=norm)

        # Add a red point at the original polynomial's coefficients
        axs[row, col].plot(p2.coef[3], p2.coef[4], 'ro', markersize=5,
                           label='Original p2 coefficients')

        axs[row, col].set_title(
            f'Depth {k_idx}: p2.coef[2]={z[k_idx]:.2f}', fontsize=12)
        axs[row, col].set_xlabel('p2.coef[3]', fontsize=10)

        # Add y-label to the leftmost plots
        if col == 0:
            axs[row, col].set_ylabel('p2.coef[4]', fontsize=10)

        # Make tick labels smaller but not too small
        axs[row, col].tick_params(labelsize=8)

    # Hide any unused subplots
    for idx in range(DEPTH, grid_rows * grid_cols):
        row = idx // grid_cols
        col = idx % grid_cols
        axs[row, col].set_visible(False)

    # Create colorbar with more space and better positioning
    # Adjust the constrained_layout_pads if needed
    plt.tight_layout()
    
    # Add colorbar with specified position and size
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # Moved further right
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Log Error (lower is better)', fontsize=12)

    plt.savefig('convergence_map_depths.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Runtime: {time.time() - start:.2f} seconds")
