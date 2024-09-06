from numpy import polynomial as P
from numpy import random
import numpy as np


def compose(p1, p2):
    # Compute the composition of p1 and p2 ( =p1(p2) )
    r = P.Polynomial([0])
    for i in range(len(p1)):
        r += p1.coef[i]*p2**i
    return r


def l2_norm(p1, p2):
    # Compute the l2 norm of the difference between p1 and p2
    r = 0
    for i in range(len(p1)):
        r += (p1.coef[i] - p2.coef[i])**2/(2*i+1)

        for j in range(i):
            r += 2*(p1.coef[i] - p2.coef[i])*(p1.coef[j] - p2.coef[j])/(i+j+1)

    return r


def main():
    # Control variables
    seed = 0
    max_iter = 1500
    batch_size = 100
    stop_loss = 1e-10
    lr = 0.001

    # We try to build a polynomial of degree 9 by composition of 2 polynomials
    # of degree 3
    random.seed(seed)

    target = P.Polynomial([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    p1 = P.Polynomial([1, -3, 2, 3])
    p2 = P.Polynomial([-2, 1, 1.5, 4])

    layers = [p1, p2]

    print(p1)
    print(p2)
    print(target)

    # Compute the l2 norm of the difference between the target polynomial and
    # the composition of p1 and p2
    loss = l2_norm(target, compose(p2, p1))
    print(loss)

    iteration = 0
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

        # Compute the l2 norm of the difference between the target polynomial and the
        # composition of p1 and p2
        loss = l2_norm(target, compose(layers[1], layers[0]))
        if iteration % 100 == 0:
            print(f"{iteration}, {loss}")
        iteration += 1

    print(f"{iteration}, {loss}")
    print(compose(layers[1], layers[0]).__str__().replace("·", ""))


if __name__ == '__main__':
    main()
