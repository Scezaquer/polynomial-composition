from numpy import polynomial as P
from numpy import random
import numpy as np
import matplotlib.pyplot as plt


def compose(p1, p2):
    # Compute the composition of p1 and p2 ( =p1(p2) )
    r = P.Polynomial([0])
    for i in range(len(p1)):
        r += p1.coef[i]*p2**i
    return r


def compose_layers(layers):
    # Compute the composition of all the layers
    r = layers[0]
    for i in range(1, len(layers)):
        r = compose(layers[i], r)
    return r


def l2_norm(p1, p2):
    # Compute the l2 norm of the difference between p1 and p2
    r = 0
    for i in range(len(p1)):
        r += (p1.coef[i] - p2.coef[i])**2/(2*i+1)
        for j in range(i):
            r += 2*(p1.coef[i] - p2.coef[i])*(p1.coef[j] - p2.coef[j])/(i+j+1)
    return r


def plot_polynomials(comp, target, iteration):
    # Set up x-axis points for plotting
    x_vals = np.linspace(0, 1, 200)
    y_comp = comp(x_vals)
    y_target = target(x_vals)

    plt.clf()
    plt.plot(x_vals, y_comp, label="Composed Polynomial", color='blue')
    plt.plot(x_vals, y_target, label="Target Polynomial", color='red', linestyle='--')
    plt.title(f"Iteration {iteration}")
    plt.legend()
    plt.pause(0.05)  # Pause to update the plot


def main():
    # Control variables
    seed = 0            # Set the seed for reproducibility
    max_iter = 1500     # Maximum number of iterations
    batch_size = 100    # Number of points to sample for the forward pass
    stop_loss = 1e-10   # Stop when the l2 norm of the difference between the target polynomial and the composed polynomial is less than this value
    lr = 0.001          # Learning rate
    random_target_poly = True     # If True, the target polynomial will be randomly initialized
    random_initialization = True  # If True, the initial polynomials will be randomly initialized
    random_target_poly_deg = 9   # Degree of the target polynomial if random_target_poly is True
    random_initialization_deg = [3, 3]  # Degree of the initial polynomials if random_initialization is True

    random.seed(seed)

    # target = P.Polynomial([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # p1 = P.Polynomial([1, -3, 2, 3])
    # p2 = P.Polynomial([-2, 1, 1.5, 4])

    prod = 1
    for i in random_initialization_deg:
        prod *= i
    if prod != random_target_poly_deg:
        raise Warning(
            "The product of the degrees of the initial polynomials "
            "is not equal to the degree of the target polynomial")

    if random_target_poly:
        target = P.Polynomial(random.rand(random_target_poly_deg+1)*5-2.5)

    if random_initialization:
        layers = [P.Polynomial(random.rand(i+1)*5-2.5) for i in random_initialization_deg]

    print("layers:")
    for layer in layers:
        print(layer)
    print("target:")
    print(target)

    # Compute the l2 norm of the difference between the target polynomial and the composition of p1 and p2
    loss = l2_norm(target, compose_layers(layers))
    print(loss)
    iteration = 0

    # Set up the plot
    plt.ion()  # Turn on interactive mode for live updates
    plt.figure()

    while (loss > stop_loss) and (iteration < max_iter):
        # Compute the gradient of the loss function
        grad = np.array([[0. for _ in layer] for layer in layers])
        poly_derivatives = [layer.deriv() for layer in layers]
        forward_pass_pts = random.rand(batch_size)

        for pt in forward_pass_pts:
            activations = [pt]
            for layer in layers:
                activations.append(layer(activations[-1]))
            activations_derivatives = []
            for i, layer in enumerate(poly_derivatives):
                activations_derivatives.append(layer(activations[i]))

            # Backprop
            dloss = activations[-1] - target(pt)
            for i in range(len(layers)-1, -1, -1):
                for j in range(len(layers[i])):
                    grad[i, j] += dloss*activations[i]**j

                dloss *= activations_derivatives[i]

        # Normalize the gradient
        grad = grad/batch_size

        # Update the parameters
        for i, layer in enumerate(layers):
            layers[i] -= lr*grad[i]

        # Compute the l2 norm of the difference between the target polynomial and the composition of p1 and p2
        loss = l2_norm(target, compose_layers(layers))

        # Update the plot every 100 iterations
        if iteration % 100 == 0:
            print(f"{iteration}, {loss}")
            plot_polynomials(compose_layers(layers), target, iteration)

        iteration += 1

    print(f"{iteration}, {loss}")
    print(compose_layers(layers).__str__().replace("·", ""))
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the final plot visible


if __name__ == '__main__':
    main()
