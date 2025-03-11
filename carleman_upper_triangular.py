import numpy as np
from math import factorial, comb
from numpy import polynomial as Polynomial
import matplotlib.pyplot as plt
import time
import random
from polynomial_utils import compose, compose_layers, l2_norm, l2_coefficient_norm, plot_polynomials, carleman, carleman_matrix
from tqdm import tqdm
import traceback


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

def solve_for_h(g, target_poly):
    # Find shift variable d
    h0_poly = g - target_poly.coef[0]
    roots = h0_poly.roots()
    real_roots = [root.real for root in roots] # if np.isreal(root)]
    if len(real_roots) == 0:
        raise ValueError("No real roots found")
    
    solutions = []
    for d in real_roots:
    #d = min(real_roots, key=abs).real

        # Shift g and target_poly
        shifted_q = target_poly - d
        shifted_g0 = g(d) - d
        shifted_gl = []
        for l in range(1, g.coef.size):
            shifted_gl.append(sum([g.coef[j] * d**(j-l) * comb(j, l) for j in range(l, g.coef.size)]))
        shifted_g = Polynomial.Polynomial([shifted_g0] + shifted_gl)  # satisfies g(x+d)-d = shifted_g(x)

        # h1 = target_poly.coef[1] / g.coef[1]
        # h2 = (target_poly.coef[2] - g.coef[2] * h1**2) / g.coef[1]
        # h3 = (target_poly.coef[3] - g.coef[3] * h1**3 - g.coef[2] * 2 * h1 * h2) / g.coef[1]

        # turns out this doesn't use q[0] so we can just use target_poly instead of shifted_q
        h1 = target_poly.coef[1] / shifted_g.coef[1]
        h2 = (target_poly.coef[2] - shifted_g.coef[2] * h1**2) / shifted_g.coef[1]
        h3 = (target_poly.coef[3] - shifted_g.coef[3] * h1**3 - shifted_g.coef[2] * 2 * h1 * h2) / shifted_g.coef[1]
        h = Polynomial.Polynomial([d, h1, h2, h3])
        #print(h.coef)
        solutions.append(h)
    # Is there a better way to choose the best h?
    h = min(solutions, key=lambda h: l2_coefficient_norm(compose(g, h), target_poly))
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


def new_poly(width, m=4, n=4):
    p1 = Polynomial.Polynomial(np.random.uniform(-width, width, m+1))
    p2 = Polynomial.Polynomial(np.random.uniform(-width, width, n+1))
    p2.coef[1] = 0 # np.random.uniform(-0.1, 0.1)
    # p1 = Polynomial.Polynomial.fromroots(np.random.uniform(-width, width, 3))
    # p2 = Polynomial.Polynomial.fromroots(np.random.uniform(-width, width, 3))
    target_poly = compose(p1, p2)
    return p1, p2, target_poly

def poly_n_roots(n):
    if n not in [3, 5, 7, 9]:
        raise ValueError("n must be 3, 5, 7, or 9")

    roots = []
    if n == 9:
        roots = np.random.uniform(-2, 2, 3)
    else:
        while n > 3:
            root = np.random.uniform(-2, 2)
            if root not in roots:
                roots.append(root)
                n -= 3
        while n > 0:
            root = np.random.uniform(-4, -2) if np.random.uniform() < 0.5 else np.random.uniform(2, 4)
            if root not in roots:
                roots.append(root)
                n -= 1

    p1 = Polynomial.Polynomial.fromroots(roots)

    p2 = Polynomial.Polynomial([0, -3, 0, 1])
    target_poly = compose(p1, p2)
    return p1, p2, target_poly

if __name__ == "__main__":
    stats_10 = []
    stats_10_time = []
    stats_100 = []
    stats_100_time = []
    start = time.time()
    
    deg_g, deg_h = 3, 3
    
    converged = [0 for i in range(deg_g*deg_h+1)]
    did_not_converge = [0 for i in range(deg_g*deg_h+1)]
    
    converged_real_roots = []
    did_not_converge_real_roots = []
    
    converged_all_roots = []
    did_not_converge_all_roots = []

    p1_coeffs_converged = []
    p2_coeffs_converged = []
    target_coeffs_converged = []
    p1_coeffs_not_converged = []
    p2_coeffs_not_converged = []
    target_coeffs_not_converged = []

    for i in tqdm(range(1000)):
        width = 1.5

        nbr_real_roots = 0
        smallest_root = 0
        #while abs(smallest_root) < 0.75:
        p1, p2, target_poly = new_poly(width, deg_h, deg_g)
        roots = target_poly.roots()
        real_roots = [np.real(root) for root in roots if np.isreal(root)]
        nbr_real_roots = len(real_roots)
        smallest_root = min(real_roots, key=abs)

        # p1, p2, target_poly = poly_n_roots([3, 5, 7, 9][i % 4])
        # target_poly = Polynomial.Polynomial.fromroots(np.random.uniform(-1.5, 1.5, 9))
        # p1 = Polynomial.Polynomial.fromroots(np.random.uniform(-1.5, 1.5, 3))
        # p2 = Polynomial.Polynomial.fromroots(np.random.uniform(-1.5, 1.5, 3))
        # target_poly = compose(p1, p2)

        h0 = Polynomial.Polynomial(np.random.uniform(-0.5, 0.5, deg_h+1))
        g0 = Polynomial.Polynomial(np.random.uniform(-0.5, 0.5, deg_g+1))

        s1 = time.time()
        try:
            h, g = carleman_upper_triangular_solver(h0, g0, target_poly, 2, verbose=False)
            #h, g = carleman_upper_triangular_solver(p2, p1, target_poly, 100, verbose=False)
        except Exception:
            print(traceback.format_exc())
            continue
        stats_10_time.append(time.time() - s1)

        composed = compose(g, h)  # * factor
        error = l2_coefficient_norm(composed, target_poly)

        # print(f"Attempt {i} | Error: {error:.4f}")
        stats_10.append(error)

        # roots = target_poly.roots()
        # nbr_real_roots = len([root for root in roots if np.isreal(root)])
        # Save coefficients for analysis
        if error > 1e-6:
            did_not_converge[nbr_real_roots] += 1
            did_not_converge_real_roots.extend(real_roots)
            did_not_converge_all_roots.extend(roots)
            p1_coeffs_not_converged.append(p1.coef)
            p2_coeffs_not_converged.append(p2.coef)
            target_coeffs_not_converged.append(target_poly.coef)
        else:
            converged[nbr_real_roots] += 1
            converged_real_roots.extend(real_roots)
            converged_all_roots.extend(roots)
            p1_coeffs_converged.append(p1.coef)
            p2_coeffs_converged.append(p2.coef)
            target_coeffs_converged.append(target_poly.coef)

        # s2 = time.time()
        # h, g = carleman_solver(h0, g0, target_poly, 100)
        # stats_100_time.append(time.time() - s2)
        # composed = compose_layers([h, g])
        # error = l2_coefficient_norm(composed, target_poly)
        # stats_100.append(error)

        # if i % 100 == 0:
        #     print(i, end='\r')

    # Remove outliers before plotting
    def remove_outliers(data, m=2):
        return data[abs(data - np.mean(data)) < m * np.std(data)]

    did_not_converge_real_roots = np.array(did_not_converge_real_roots)
    converged_real_roots = np.array(converged_real_roots)

    # did_not_converge_real_roots = remove_outliers(did_not_converge_real_roots, 1)
    # converged_real_roots = remove_outliers(converged_real_roots, 1)

    # Plot histograms of real roots for converged and did not converge cases
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot did_not_converge_real_roots histogram
    axes[0].hist(did_not_converge_real_roots, bins=50, color='red', alpha=0.7)
    axes[0].set_title('Did Not Converge Real Roots')
    axes[0].set_xlabel('Real Root Value')
    axes[0].set_ylabel('Frequency')

    # Plot converged_real_roots histogram
    axes[1].hist(converged_real_roots, bins=50, color='green', alpha=0.7)
    axes[1].set_title('Converged Real Roots')
    axes[1].set_xlabel('Real Root Value')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()
    
    # Convert all roots to numpy arrays for easier manipulation
    did_not_converge_all_roots = np.array(did_not_converge_all_roots)
    converged_all_roots = np.array(converged_all_roots)

    # Create a 2D density plot for complex roots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot did_not_converge_all_roots
    axes[0].set_title('Did Not Converge - Complex Roots')
    axes[0].set_xlabel('Real Part')
    axes[0].set_ylabel('Imaginary Part')
    if len(did_not_converge_all_roots) > 0:
        axes[0].scatter(did_not_converge_all_roots.real, did_not_converge_all_roots.imag, 
                        c='red', alpha=0.5, s=20)
        # Use hexbin for density visualization
        hb = axes[0].hexbin(did_not_converge_all_roots.real, did_not_converge_all_roots.imag, 
                            gridsize=30, cmap='Reds', alpha=0.7, extent=[-4, 4, -4, 4])
        plt.colorbar(hb, ax=axes[0], label='Count')
    
    # Set limits for the first plot
    axes[0].set_xlim(-4, 4)
    axes[0].set_ylim(-4, 4)

    # Plot converged_all_roots
    axes[1].set_title('Converged - Complex Roots')
    axes[1].set_xlabel('Real Part')
    axes[1].set_ylabel('Imaginary Part')
    if len(converged_all_roots) > 0:
        axes[1].scatter(converged_all_roots.real, converged_all_roots.imag, 
                        c='green', alpha=0.5, s=20)
        # Use hexbin for density visualization
        hb = axes[1].hexbin(converged_all_roots.real, converged_all_roots.imag, 
                            gridsize=30, cmap='Greens', alpha=0.7, extent=[-4, 4, -4, 4])
        plt.colorbar(hb, ax=axes[1], label='Count')
    
    # Set limits for the second plot
    axes[1].set_xlim(-4, 4)
    axes[1].set_ylim(-4, 4)

    for ax in axes:
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Time: {time.time() - start:.4f}")

    # Convert coefficient lists to numpy arrays for easier analysis
    p1_coeffs_converged = np.array(p1_coeffs_converged)
    p2_coeffs_converged = np.array(p2_coeffs_converged)
    target_coeffs_converged = np.array(target_coeffs_converged)
    p1_coeffs_not_converged = np.array(p1_coeffs_not_converged)
    p2_coeffs_not_converged = np.array(p2_coeffs_not_converged)
    target_coeffs_not_converged = np.array(target_coeffs_not_converged)

    # Create a figure with 12 subplots (one for each coefficient)
    fig, axes = plt.subplots(2, 4, figsize=(20, 15))
    fig.suptitle('Coefficient Distributions', fontsize=16)

    # Labels for the rows
    poly_names = ['p1', 'p2']
    
    # Plot each coefficient as a separate subplot
    for row, (poly_name, converged_data, not_converged_data) in enumerate([
        ('p1', p1_coeffs_converged, p1_coeffs_not_converged),
        ('p2', p2_coeffs_converged, p2_coeffs_not_converged)
    ]):
        for col in range(4):
            ax = axes[row, col]
            
            # Only plot if we have data
            if len(converged_data) > 0:
                ax.hist(converged_data[:, col], bins=20, alpha=0.7, 
                        label='Converged', color='green', density=True)
            
            if len(not_converged_data) > 0:
                ax.hist(not_converged_data[:, col], bins=20, alpha=0.5,
                        label='Not Converged', color='red', density=True)
            
            # Add labels and title
            ax.set_xlabel(f'Value')
            ax.set_ylabel('Density')
            ax.set_title(f'{poly_name} coefficient {col}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics as text if data exists
            if len(converged_data) > 0:
                conv_mean = np.mean(converged_data[:, col])
                conv_std = np.std(converged_data[:, col])
                ax.text(0.05, 0.95, f'Conv: μ={conv_mean:.2f}, σ={conv_std:.2f}', 
                        transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', color='green')
            
            if len(not_converged_data) > 0:
                not_conv_mean = np.mean(not_converged_data[:, col])
                not_conv_std = np.std(not_converged_data[:, col])
                ax.text(0.05, 0.85, f'Not: μ={not_conv_mean:.2f}, σ={not_conv_std:.2f}', 
                        transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', color='red')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for the suptitle
    plt.show()
    
    # Plot the target polynomial coefficient distributions
    fig, axes = plt.subplots(2, 5, figsize=(20, 5))
    fig.suptitle('Target Polynomial Coefficient Distributions', fontsize=16)

    for col in range(10):
        row, col_idx = divmod(col, 5)
        ax = axes[row, col_idx]
        
        # Plot histograms if we have data
        if len(target_coeffs_converged) > 0:
            ax.hist(target_coeffs_converged[:, col], bins=20, alpha=0.7, 
                    label='Converged', color='green', density=True)
        
        if len(target_coeffs_not_converged) > 0:
            ax.hist(target_coeffs_not_converged[:, col], bins=20, alpha=0.5,
                    label='Not Converged', color='red', density=True)
        
        # Add labels and title
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title(f'Target coefficient {col}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics as text if data exists
        if len(target_coeffs_converged) > 0:
            conv_mean = np.mean(target_coeffs_converged[:, col])
            conv_std = np.std(target_coeffs_converged[:, col])
            ax.text(0.05, 0.95, f'Conv: μ={conv_mean:.2f}, σ={conv_std:.2f}', 
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', color='green')
        
        if len(target_coeffs_not_converged) > 0:
            not_conv_mean = np.mean(target_coeffs_not_converged[:, col])
            not_conv_std = np.std(target_coeffs_not_converged[:, col])
            ax.text(0.05, 0.85, f'Not: μ={not_conv_mean:.2f}, σ={not_conv_std:.2f}', 
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', color='red')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

    print("Converged:")
    print(converged)
    print("Did not converge:")
    print(did_not_converge)
    print("% Converged:")
    print([converged[i] / (converged[i] + did_not_converge[i]) if (converged[i] + did_not_converge[i]) != 0 else np.nan for i in range(10)])

    # Calculate statistics for stats_10
    mean_10 = np.mean(stats_10)
    median_10 = np.median(stats_10)
    variance_10 = np.var(stats_10)

    # Calculate statistics for stats_100
    mean_100 = np.mean(stats_100)
    median_100 = np.median(stats_100)
    variance_100 = np.var(stats_100)

    # Print statistics
    print(f"stats_upper_triangularization - Mean: {mean_10}, Median: {median_10}, Variance: {variance_10}, Time: {np.mean(stats_10_time)}")
    print(f"stats_moore_penrose_pinv - Mean: {mean_100}, Median: {median_100}, Variance: {variance_100}, Time: {np.mean(stats_100_time)}")

    # Plot histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Determine the common bin edges for both histograms
    all_stats = np.concatenate((stats_10, stats_100))
    bins = np.logspace(np.log10(min(all_stats)), np.log10(max(all_stats)), 50)

    # Plot stats_10 histogram
    axes[0].hist(stats_10, bins=bins)
    axes[0].set_xscale('log')
    axes[0].set_title('upper_triangularization')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')

    # Plot stats_100 histogram
    axes[1].hist(stats_100, bins=bins)
    axes[1].set_xscale('log')
    axes[1].set_title('moore_penrose_pinv')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')

    # Ensure the vertical axis is the same for both graphs
    max_freq = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(0, max_freq)
    axes[1].set_ylim(0, max_freq)

    plt.tight_layout()
    plt.show()
