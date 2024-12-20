from numpy import polynomial as P
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Any
import time
from polynomial_utils import compose_layers, plot_polynomials, l_inf_norm


def gradient_descent(
        target: Optional[P.Polynomial] = None,
        layers: Optional[List[P.Polynomial]] = None,
        random_target_poly_deg: int = 27,
        random_initialization_deg: List[int] = [3, 3, 3],
        seed: Optional[int] = None,
        max_iter: int = 10000,
        batch_size: int = 2000,
        stop_loss: float = 1e-10,
        lr: float = 0.001,
        use_adam: bool = True,
        beta1: float = 0.99,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        verbose: bool = True,
        plot: bool = True,
        use_scale_lr: bool = False,
        print_frequency: int = 250) -> Tuple[List[P.Polynomial], List[Any]]:
    """
    Perform gradient descent to approximate a target polynomial using a composition of smaller polynomials.

    This function uses stochastic gradient descent (SGD) or Adam optimization to find a composition of
    polynomials that closely approximates a target polynomial. It can work with a given target polynomial
    or generate a random one.

    Args:
        target (Optional[numpy.polynomial.Polynomial]): The target polynomial to approximate. If None, a random polynomial is generated.
        layers (Optional[List[numpy.polynomial.Polynomial]]): Initial list of polynomials to compose. If None, random polynomials are generated.
        random_target_poly_deg (int): Degree of the random target polynomial if target is None.
        random_initialization_deg (List[int]): Degrees of the initial random polynomials if layers is None.
        seed (Optional[int]): Seed for the random number generator. If None, no seed is set.
        max_iter (int): Maximum number of iterations for the optimization process.
        batch_size (int): Number of points used for each iteration in stochastic gradient descent.
        stop_loss (float): Stopping criterion; the optimization stops when the loss is below this value.
        lr (float): Learning rate for the gradient descent steps.
        use_adam (bool): If True, use Adam optimization; if False, use standard SGD.
        beta1 (float): Adam optimizer parameter for the first moment estimate.
        beta2 (float): Adam optimizer parameter for the second moment estimate.
        epsilon (float): Small value added to denominator in Adam update to improve numerical stability.
        verbose (bool): If True, print progress information during optimization.
        plot (bool): If True, plot the approximation progress during optimization.
        use_scale_lr (bool): If True, scale the learning rate by the degree of the polynomial at each layer. This can help stabilize the optimization process.
        print_frequency (int): Frequency at which to print progress information.

    Returns:
        Tuple[List[numpy.polynomial.Polynomial], List[Any]]: The list of polynomials whose composition approximates the target polynomial, and the list of losses at each iteration.

    Raises:
        Warning: If the product of the degrees of the initial polynomials does not match the degree of the target polynomial.

    Note:
        The function modifies the input layers in-place if provided, or the generated random layers otherwise.
        The final approximation can be obtained by composing the resulting layers after the function call.
    """

    init_timer = 0.
    forward_pass_timer = 0.
    backprop_timer = 0.
    adam_timer = 0.
    io_timer = 0.
    main_loop_timer = 0.
    loss_timer = 0.
    loop_init_timer = 0.

    st = time.process_time()

    if seed is not None:
        random.seed(seed)

    # Warning in case the product of the degrees of the initial polynomials
    # does not match the degree of the target polynomial
    prod = 1
    lrs = [lr]
    if layers is not None:
        for layer in layers[::-1]:
            prod *= layer.degree()
            lrs.append(lrs[-1]/layer.degree())
    else:
        for i in random_initialization_deg:
            prod *= i
            lrs.append(lrs[-1]/i)
    lrs = lrs[:-1][::-1]

    if (target is None and prod != random_target_poly_deg) or (target is not None and prod != target.degree()):
        raise Warning(
            "The product of the degrees of the initial polynomials "
            f"({'x'.join(map(str, random_initialization_deg))}={prod}) "
            "is not equal to the degree of the target polynomial "
            f"({random_target_poly_deg}).")

    # Initialize target and layers
    if target is None:
        target = P.Polynomial(random.rand(random_target_poly_deg+1)*5-2.5)

    if layers is None:
        layers = [P.Polynomial(random.rand(i+1)-0.5)
                  for i in random_initialization_deg]
        # NOTE: Starting with weights between -1 and 1 immensely improves
        # performances compared to a larger interval. This is probably due to
        # weights blowing up when composing polynomials, which gradient descent
        # has a hard time correcting

    forward_pass_pts = np.linspace(0, 1, batch_size)

    losses = []
    loss = 1e10
    # loss = l2_norm(target, compose_layers(layers))
    # loss = np.max(np.abs(target(forward_pass_pts) - compose_layers(layers)(forward_pass_pts)))
    # losses.append(loss)

    if verbose:
        print("layers:")
        for layer in layers:
            print(layer)
        print("target:")
        print(target)

    iteration = 0

    if plot:
        plt.ion()
        plt.figure()

    # Initialize Adam parameters
    if use_adam:
        m = [np.zeros_like(layer.coef) for layer in layers]
        v = [np.zeros_like(layer.coef) for layer in layers]

    largest_layer = max(layers, key=lambda x: x.degree())

    init_timer += time.process_time() - st

    st1 = time.process_time()
    grad = np.zeros((len(layers), len(largest_layer.coef)))

    while (loss > stop_loss) and (iteration < max_iter):
        st6 = time.process_time()
        # Initialize gradient
        grad.fill(0)

        # Compute the derivative of each layer wrt. its input
        poly_derivatives = [layer.deriv() for layer in layers]

        # Pick random points for the forward pass
        # forward_pass_pts = random.rand(batch_size)

        loop_init_timer += time.process_time() - st6

        st2 = time.process_time()

        # Compute the activations of each layer
        activations = np.zeros((batch_size, len(layers)+1))
        activations[:, 0] = forward_pass_pts
        for i, layer in enumerate(layers):
            activations[:, i+1] = layer(activations[:, i])

        # Compute the derivative of each layer at the activations
        activations_derivatives = np.zeros((batch_size, len(layers)))
        for i, layer in enumerate(poly_derivatives):
            activations_derivatives[:, i] = layer(activations[:, i])

        forward_pass_timer += time.process_time() - st2

        st3 = time.process_time()
        # Backward pass
        dloss = activations[:, -1] - target(forward_pass_pts)

        loss = max(np.abs(dloss))
        losses.append(loss)

        for i in range(len(layers)-1, -1, -1):
            for j in range(len(layers[i])):
                grad[i, j] += np.sum(dloss*activations[:, i]**j)

            if i > 0:
                dloss *= activations_derivatives[:, i]
        backprop_timer += time.process_time() - st3

        # Normalize the gradient
        grad = grad/batch_size

        st4 = time.process_time()
        # Update the weights
        if use_adam:
            for i in range(len(layers)):
                m[i] = beta1 * m[i] + (1 - beta1) * grad[i][:len(m[i])]
                v[i] = beta2 * v[i] + (1 - beta2) * (grad[i][:len(v[i])] ** 2)
                m_hat = m[i] / (1 - beta1 ** (iteration + 1))
                v_hat = v[i] / (1 - beta2 ** (iteration + 1))
                layers[i].coef -= m_hat / \
                    (np.sqrt(v_hat) + epsilon) * \
                    (lrs[i] if use_scale_lr else lr)
        else:
            for i, layer in enumerate(layers):
                layers[i].coef -= grad[i] * (lrs[i] if use_scale_lr else lr)

        adam_timer += time.process_time() - st4

        """st7 = time.process_time()
        #loss = l2_norm(target, compose_layers(layers))
        loss = np.max(np.abs(target(forward_pass_pts) - compose_layers(layers)(forward_pass_pts)))
        loss_timer += time.process_time() - st7
        losses.append(loss)"""

        st5 = time.process_time()
        if iteration % print_frequency == 0:
            if verbose:
                print(f"{iteration}, {loss}")
            if plot:
                plot_polynomials(compose_layers(layers), target, iteration)
        io_timer += time.process_time() - st5
        iteration += 1

    main_loop_timer += time.process_time() - st1
    
    loss = l_inf_norm(target, compose_layers(layers)) # np.max(np.abs(target(forward_pass_pts) - compose_layers(layers)(forward_pass_pts)))
    losses.append(loss)

    if verbose:
        print(f"{iteration}, {loss}")
        print("Final polynomials:")
        for layer in layers:
            print(layer)
        print()
        print(compose_layers(layers).__str__().replace("·", ""))

    if plot:
        plot_polynomials(compose_layers(layers), target, iteration)
        plt.ioff()
        plt.show()

    print(f"Initialization time: {init_timer:.2f}s")
    print(f"Loop init time: {loop_init_timer:.2f}s")
    print(f"Forward pass time: {forward_pass_timer:.2f}s")
    print(f"Backward pass time: {backprop_timer:.2f}s")
    print(f"Adam time: {adam_timer:.2f}s")
    print(f"Loss computation time: {loss_timer:.2f}s")
    print(f"I/O time: {io_timer:.2f}s")
    print(f"Main loop time: {main_loop_timer:.2f}s")

    return layers, losses


def main():
    # Control variables
    target = None       # Target polynomial
    layers = None       # Initial polynomials
    random_target_poly_deg = 27     # Degree of the target polynomial
    random_initialization_deg = [3, 3, 3]  # Degrees of the initial polynomials
    seed = 0            # Seed for random number generator
    max_iter = 10000    # Maximum number of iterations
    batch_size = 2000  # Number of points used for each iteration
    stop_loss = 1e-10   # Stop when loss is below this value
    lr = 0.001          # Learning rate
    use_adam = False  # activate/deactivate Adam optimizer
    beta1 = 0.99    # Adam parameter
    beta2 = 0.999   # Adam parameter
    epsilon = 1e-8  # Adam parameter
    verbose = True  # Print progress
    plot = False     # Plot progress
    use_scale_lr = False  # Scale learning rate by degree of polynomial at each layer
    print_frequency = 250  # Frequency at which to print progress

    """target = P.Polynomial(np.ones(5))
    layers = [P.Polynomial([-0.1, 0.1, 0.2]),
              P.Polynomial([-0.1, 0.1, 0.2])]"""

    gradient_descent(
        target=target,
        layers=layers,
        random_target_poly_deg=random_target_poly_deg,
        random_initialization_deg=random_initialization_deg,
        seed=seed,
        max_iter=max_iter,
        batch_size=batch_size,
        stop_loss=stop_loss,
        lr=lr,
        use_adam=use_adam,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        verbose=verbose,
        plot=plot,
        use_scale_lr=use_scale_lr,
        print_frequency=print_frequency
    )


if __name__ == '__main__':
    use_adam = True
    max_iter = 25000
    seed = 473753

    gradient_descent(use_adam=use_adam, max_iter=max_iter, seed=seed)

    # main()
