import numpy as np
from numpy import polynomial as Polynomial
from math import floor

import matplotlib.pyplot as plt

def max_coeff_distribution(power, num_samples, ranges):
    max_coeffs = []
    for r in ranges:
        coeffs = []
        for _ in range(num_samples):
            a = Polynomial.Polynomial(np.random.uniform(-r, r, 4))
            max_coeff = max(abs((a**power).coef))
            coeffs.append(max_coeff)
        max_coeffs.append(coeffs)
    return max_coeffs

power = 100
num_samples = 1000
ranges = np.linspace(1, 0.1, 10)

max_coeffs = max_coeff_distribution(power, num_samples, ranges)
plt.figure(figsize=(10, 6))
for i, r in enumerate(ranges):
    plt.hist(max_coeffs[i], bins=np.logspace(np.log10(min(max_coeffs[i])), np.log10(max(max_coeffs[i])), 50), alpha=0.5, label=f'Range: {-r} to {r}')
plt.xscale('log')
plt.xlabel('Max Coefficient')
plt.ylabel('Frequency')
plt.legend()
plt.title(f'Distribution of Max Coefficients for Polynomial Power {power}')
plt.show()

x_values = np.logspace(-2, 0, 100)  # Using logspace instead of linspace
degrees = [2, 3, 4, 5, 10, 20, 30, 40, 50, 100]
plt.figure(figsize=(10, 6))

for degree in degrees:
    max_coeffs_x = []
    for x in x_values:
        a = Polynomial.Polynomial([x for _ in range(degree)])
        # a.coef[floor(degree/2)] = 1.1  # Add a small perturbation to the coefficients
        max_coeff = max(abs((a**100).coef))
        max_coeffs_x.append(max_coeff)
    plt.plot(x_values, max_coeffs_x, marker='o', label=f'Degree: {degree}')

plt.xscale('log')  # Set x axis to log scale
plt.yscale('log')
plt.xlabel('epsilon value')
plt.ylabel('Max Coefficient')
plt.title('Max Coefficient of Polynomial a**100 where a is a polynomials where all coefficients are epsilon')
plt.legend()
plt.grid(True)
plt.show()