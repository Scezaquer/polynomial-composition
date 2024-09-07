from main import gradient_descent
from numpy import polynomial as P


def random_poly():
    # Generating a random polynomial to approximate
    gradient_descent(
        random_target_poly_deg=27,
        random_initialization_deg=[3, 3, 3],
        seed=0,
        max_iter=25000)


def known_poly():
    # Approximating a known polynomial of degree 9 with the composition of
    # two polynomials of degree 3
    p = P.Polynomial([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    gradient_descent(target=p, random_initialization_deg=[3, 3], seed=0)


if __name__ == '__main__':
    random_poly()