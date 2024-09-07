import json
import numpy as np
import matplotlib.pyplot as plt
from numpy import polynomial as P
from main import gradient_descent, compose_layers
from typing import List, Tuple
import os
import time


def run_experiment(experiment_name: str, num_trials: int, **kwargs) ->List[Tuple[List[P.Polynomial], List[float], int]]:
    results = []
    start_time = time.time()
    print(f"Starting experiment: {experiment_name}")

    for trial in range(num_trials):
        seed = np.random.randint(0, 1000000)
        np.random.seed(seed)
        print(f"  Running trial {trial + 1}/{num_trials}...")
        layers, losses = gradient_descent(**kwargs, seed=seed)
        results.append((layers, losses, seed))

    # Save results to JSON
    json_results = {
        "experiment_name": experiment_name,
        "num_trials": num_trials,
        "parameters": kwargs,
        "results": [
            {
                "seed": seed,
                "losses": losses,
                "final_loss": losses[-1],
                "layers": [layer.coef.tolist() for layer in layers]
            }
            for layers, losses, seed in results
        ]
    }

    if 'target' in json_results['parameters'] and json_results['parameters']['target'] is not None:
        json_results['parameters']['target'] = json_results['parameters']['target'].coef.tolist()

    os.makedirs("experiment_results", exist_ok=True)
    with open(f"experiment_results/{experiment_name}.json", "w") as f:
        json.dump(json_results, f, indent=2)

    end_time = time.time()
    print(f"Experiment {experiment_name} completed in {end_time - start_time:.2f} seconds")

    return results


def plot_experiment_results(experiment_name: str, results: List[Tuple[List[P.Polynomial], List[float], int]]):
    plt.figure(figsize=(12, 8))

    # Plot individual trials
    for i, (_, losses, _) in enumerate(results):
        plt.plot(losses, alpha=0.3, color='blue')

    # Calculate and plot average with shaded region for spread
    all_losses = [losses for _, losses, _ in results]
    max_len = max(len(losses) for losses in all_losses)
    padded_losses = [np.pad(losses, (0, max_len - len(losses)), mode='constant', constant_values=losses[-1]) for losses in all_losses]
    losses_array = np.array(padded_losses)

    mean_losses = np.mean(losses_array, axis=0)

    plt.plot(mean_losses, color='red', label='Average')

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(f'{experiment_name} - Loss over Iterations')
    plt.legend()
    plt.yscale('log')
    plt.savefig(f"experiment_results/{experiment_name}_plot.png")
    plt.close()


def test_adam_vs_sgd():
    adam_results = run_experiment("adam", 10, use_adam=True, max_iter=10000, verbose=False, plot=False)
    sgd_results = run_experiment("sgd", 10, use_adam=False, max_iter=10000, verbose=False, plot=False)

    plot_experiment_results("adam", adam_results)
    plot_experiment_results("sgd", sgd_results)


def test_multiple_initializations():
    target = P.Polynomial(np.random.randn(28))
    results = run_experiment("multiple_initializations", 10, target=target, max_iter=10000, verbose=False, plot=False)
    plot_experiment_results("multiple_initializations", results)


def test_polynomial_degree_limits():
    degrees = [3, 9, 27, 81]
    for i, deg in enumerate(degrees):
        results = run_experiment(f"degree_limit_{deg}", 5, random_target_poly_deg=deg, random_initialization_deg=[3]*(i+1), max_iter=10000, verbose=False, plot=False)
        plot_experiment_results(f"degree_limit_{deg}", results)


def test_component_count():
    configurations = [
        ([4, 4], 16),
        ([2, 2, 2, 2], 16),
        ([8, 8], 64),
        ([4, 4, 4], 64),
        ([2, 2, 2, 2, 2, 2], 64)
    ]
    for init_deg, target_deg in configurations:
        results = run_experiment(f"component_count_{len(init_deg)}_{target_deg}", 10, random_initialization_deg=init_deg, random_target_poly_deg=target_deg, max_iter=10000, verbose=False, plot=False)
        plot_experiment_results(f"component_count_{len(init_deg)}_{target_deg}", results)


def test_known_composition():
    np.random.seed(0)

    def create_known_target():
        p1 = P.Polynomial(np.random.randn(4))
        p2 = P.Polynomial(np.random.randn(4))
        p3 = P.Polynomial(np.random.randn(4))
        return compose_layers([p1, p2, p3])

    target = create_known_target()
    results = run_experiment("known_composition", 10, target=target, max_iter=10000, verbose=False, plot=False)
    plot_experiment_results("known_composition", results)


def test_hyperparameters():
    # Test batch size
    batch_sizes = [10, 50, 100, 200, 500]
    for bs in batch_sizes:
        results = run_experiment(f"batch_size_{bs}", 5, batch_size=bs, max_iter=10000, verbose=False, plot=False)
        plot_experiment_results(f"batch_size_{bs}", results)

    # Test learning rate
    learning_rates = [0.001, 0.002, 0.005, 0.01, 0.05]
    for lr in learning_rates:
        results = run_experiment(f"learning_rate_{lr}", 5, lr=lr, max_iter=10000, verbose=False, plot=False)
        plot_experiment_results(f"learning_rate_{lr}", results)

    # Test Adam parameters
    adam_betas = [(0.9, 0.999), (0.95, 0.999), (0.99, 0.999), (0.9, 0.9999)]
    for beta1, beta2 in adam_betas:
        results = run_experiment(f"adam_betas_{beta1}_{beta2}", 5, use_adam=True, beta1=beta1, beta2=beta2, max_iter=10000, verbose=False, plot=False)
        plot_experiment_results(f"adam_betas_{beta1}_{beta2}", results)

    # Test use_scale_lr
    for use_scale in [True, False]:
        results = run_experiment(f"use_scale_lr_{use_scale}", 10, use_scale_lr=use_scale, max_iter=10000, verbose=False, plot=False)
        plot_experiment_results(f"use_scale_lr_{use_scale}", results)


if __name__ == "__main__":
    test_adam_vs_sgd()
    test_multiple_initializations()
    test_polynomial_degree_limits()
    test_component_count()
    test_known_composition()
    test_hyperparameters()
