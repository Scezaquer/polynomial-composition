import random
import numpy as np
from numpy import polynomial as Polynomial
from polynomial_utils import compose_layers, l2_norm, plot_polynomials, l2_coefficient_norm, compose
from main import gradient_descent
from carleman_approach import carleman_solver
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt


def random_polynomial(degree):
    return Polynomial.Polynomial(np.random.uniform(-1.5, 1.5, degree + 1))


def mutate(poly, rate, error):
    coef = poly.coef
    mutation_range = max(2 * (1 - np.exp(-error)), 0.02)  # Decrease mutation range as error decreases
    for i in range(len(coef)):
        if random.random() < rate:
            coef[i] += np.random.uniform(-mutation_range, mutation_range)
    return Polynomial.Polynomial(coef)


def crossover(p1, p2):
    coef1 = p1.coef
    coef2 = p2.coef
    new_coef = np.array([random.choice([c1, c2])
                        for c1, c2 in zip(coef1, coef2)])
    return Polynomial.Polynomial(new_coef)


def fitness(polynomials, target_poly):
    composed = compose_layers(polynomials)
    return l2_norm(composed, target_poly)


def genetic_alg(target_poly: Polynomial, num_polynomials=2, degrees=None, population_size=100, generations=1000, mutation_rate=0.1, verbose=False):
    if degrees is None:
        degrees = [3] * num_polynomials

    population = [[random_polynomial(degree) for degree in degrees]
                  for _ in range(population_size)]

    for generation in range(generations):
        population = sorted(population, key=lambda x: fitness(x, target_poly))
        best_error = fitness(population[0], target_poly)
        if best_error < 1e-4:
            break

        new_population = population[:10]  # Elitism: carry over the top 10
        while len(new_population) < population_size:
            parents = random.sample(population[:50], 2)
            children = []
            for i in range(num_polynomials):
                child = crossover(parents[0][i], parents[1][i])
                child = mutate(child, mutation_rate, best_error)
                children.append(child)
            new_population.append(children)

        population = new_population
        if verbose:
            print(f"Generation {generation} | Best Error: {best_error:.4f}")

    return population[0]


def compare_methods(num_experiments=200, num_polynomials=3, degrees=[3, 3, 3], population_size=100, generations=100, mutation_rate=0.1, max_iter=10000):
    errors_gd = []
    errors_ga_gd = []

    for _ in tqdm(range(num_experiments)):
        target_poly = Polynomial.Polynomial(np.random.uniform(-2.5, 2.5, 28))

        # Gradient Descent alone
        initial_polys = [random_polynomial(degree) for degree in degrees]
        _, losses_gd = gradient_descent(target_poly, initial_polys, max_iter=max_iter+generations, verbose=False, plot=False)
        errors_gd.append(losses_gd[-1])

        # Genetic Algorithm + Gradient Descent
        polys_ga = genetic_alg(target_poly, num_polynomials=num_polynomials, degrees=degrees, population_size=population_size, generations=generations, mutation_rate=mutation_rate)
        _, losses_ga_gd = gradient_descent(target_poly, polys_ga, max_iter=max_iter, verbose=False, plot=False)
        errors_ga_gd.append(losses_ga_gd[-1])

    # Plotting the distributions
    plt.figure(figsize=(12, 6))

    bins = np.logspace(np.log10(min(min(errors_gd), min(errors_ga_gd))), np.log10(max(max(errors_gd), max(errors_ga_gd))), 20)

    plt.subplot(1, 2, 1)
    plt.hist(errors_gd, bins=bins, alpha=0.5, label='Gradient Descent Alone')
    plt.xscale('log')
    plt.xlabel('Final Error (log scale)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Gradient Descent Alone')

    plt.subplot(1, 2, 2)
    plt.hist(errors_ga_gd, bins=bins, alpha=0.5, label='Genetic Algorithm + Gradient Descent')
    plt.xscale('log')
    plt.xlabel('Final Error (log scale)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Genetic Algorithm + Gradient Descent')

    plt.tight_layout()
    #plt.show()

    # Printing statistics
    print("Gradient Descent Alone:")
    print(f"Mean: {np.mean(errors_gd):.4f}")
    print(f"Median: {np.median(errors_gd):.4f}")
    print(f"Std: {np.std(errors_gd):.4f}")

    print("\nGenetic Algorithm + Gradient Descent:")
    print(f"Mean: {np.mean(errors_ga_gd):.4f}")
    print(f"Median: {np.median(errors_ga_gd):.4f}")
    print(f"Std: {np.std(errors_ga_gd):.4f}")
    
    # Save results to JSON
    results = {
        "errors_gd": errors_gd,
        "errors_ga_gd": errors_ga_gd,
        "statistics": {
            "gd": {
                "mean": np.mean(errors_gd),
                "median": np.median(errors_gd),
                "std": np.std(errors_gd)
            },
            "ga_gd": {
                "mean": np.mean(errors_ga_gd),
                "median": np.median(errors_ga_gd),
                "std": np.std(errors_ga_gd)
            }
        }
    }

    os.makedirs('experiment_results', exist_ok=True)
    with open('experiment_results/genetic_alg_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Save the plot
    plt.savefig('experiment_results/genetic_alg_comparison_plot.png')


compare_methods()