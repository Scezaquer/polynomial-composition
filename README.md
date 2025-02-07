# polynomial-composition

- TODO: Add comments
- TODO: Add requirements.txt
- TODO: Add README.md
- TODO: Add LICENSE
- TODO: Test with vs without adam
- TODO: Test multiple initializations for same target
- TODO: Test different degrees for target and initializations
- TODO: List pros and cons of method: can't guarantee every step is an improvement, slow
- TODO: Test limits in terms of polynomial degree/number of components
- TODO: Test if it's better to have more or less components
- TODO: Is there a specific initialization strategy that is strong?
- TODO: Optimize
- TODO: Test when the target is a composition of known polynomials
- TODO: Make sure just rescaling works to extend the interval to anything
- TODO: Find edgecases
- TODO: What happens when changing batch size? lr? dynamic batch size over time?
best adam params? decaying lr? Changing batch size based on degree? Change lr
based on layer (earliest layers have smallest lr since changes propagate more)?
- TODO: Is linspace better than random points??


Since composing polynomials is like taking the convolution of the vectors containing the coefficients, can we look at the problem from the perspective of trying to invert a convolution?

Checkout:
- Carleman matrices
- Faa di bruno's formula
- Levenberg–Marquardt algorithm
- polynomial regression
- deconvolution
- Chebyshev Polynomials
- Iterated Maps
- Ritt's Theorem
- Genetic algorithms
- Symbolic-Numeric Hybrid Methods

https://www.scirp.org/pdf/alamt_2022011815280635.pdf