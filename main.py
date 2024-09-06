from numpy import polynomial as P
from numpy import random
import numpy as np
import matplotlib.pyplot as plt


def compose(p1, p2):
    r = P.Polynomial([0])
    for i in range(len(p1)):
        r += p1.coef[i]*p2**i
    return r


def compose_layers(layers):
    r = layers[0]
    for i in range(1, len(layers)):
        r = compose(layers[i], r)
    return r


def l2_norm(p1, p2):
    r = 0
    for i in range(len(p1)):
        r += (p1.coef[i] - p2.coef[i])**2/(2*i+1)
        for j in range(i):
            r += 2*(p1.coef[i] - p2.coef[i])*(p1.coef[j] - p2.coef[j])/(i+j+1)
    return r


def plot_polynomials(comp, target, iteration):
    x_vals = np.linspace(0, 1, 200)
    y_comp = comp(x_vals)
    y_target = target(x_vals)

    plt.clf()
    plt.plot(x_vals, y_comp, label="Composed Polynomial", color='blue')
    plt.plot(x_vals, y_target, label="Target Polynomial", color='red', linestyle='--')
    plt.title(f"Iteration {iteration}")
    plt.legend()
    plt.pause(0.05)


def main():
    # Control variables
    seed = 0
    max_iter = 10000
    batch_size = 100
    stop_loss = 1e-10
    lr = 0.05
    random_target_poly = True
    random_initialization = True
    random_target_poly_deg = 9
    random_initialization_deg = [3, 3]
    use_adam = True  # New control variable to activate/deactivate Adam optimizer
    beta1 = 0.9  # Adam parameter
    beta2 = 0.999  # Adam parameter
    epsilon = 1e-8  # Adam parameter

    random.seed(seed)

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

    loss = l2_norm(target, compose_layers(layers))
    print(loss)
    iteration = 0

    plt.ion()
    plt.figure()

    # Initialize Adam parameters
    if use_adam:
        m = [np.zeros_like(layer.coef) for layer in layers]
        v = [np.zeros_like(layer.coef) for layer in layers]

    while (loss > stop_loss) and (iteration < max_iter):
        grad = np.array([np.zeros_like(layer.coef) for layer in layers])
        poly_derivatives = [layer.deriv() for layer in layers]
        forward_pass_pts = random.rand(batch_size)

        for pt in forward_pass_pts:
            activations = [pt]
            for layer in layers:
                activations.append(layer(activations[-1]))
            activations_derivatives = []
            for i, layer in enumerate(poly_derivatives):
                activations_derivatives.append(layer(activations[i]))

            dloss = activations[-1] - target(pt)
            for i in range(len(layers)-1, -1, -1):
                for j in range(len(layers[i])):
                    grad[i][j] += dloss*activations[i]**j

                dloss *= activations_derivatives[i]

        grad = grad/batch_size

        if use_adam:
            for i in range(len(layers)):
                m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
                v[i] = beta2 * v[i] + (1 - beta2) * (grad[i] ** 2)
                m_hat = m[i] / (1 - beta1 ** (iteration + 1))
                v_hat = v[i] / (1 - beta2 ** (iteration + 1))
                layers[i].coef -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        else:
            for i, layer in enumerate(layers):
                layers[i].coef -= lr * grad[i]

        loss = l2_norm(target, compose_layers(layers))

        if iteration % 100 == 0:
            print(f"{iteration}, {loss}")
            plot_polynomials(compose_layers(layers), target, iteration)

        iteration += 1

    print(f"{iteration}, {loss}")
    print("Final polynomials:")
    for layer in layers:
        print(layer)
    print()
    print(compose_layers(layers).__str__().replace("·", ""))
    plot_polynomials(compose_layers(layers), target, iteration)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()