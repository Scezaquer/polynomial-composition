def horner(coeffs):
    def f(x):
        result = 0
        for coeff in coeffs:
            result = result * x + coeff
        return result
    return f

def canonical(coeffs):
    def f(x):
        result = 0
        for i, coeff in enumerate(coeffs[::-1]):
            result += coeff * x ** i
        return result
    return f

def test():
    coeffs = [1, 2, 3, 7, 2, 4, 5]
    f1 = horner(coeffs)
    f2 = canonical(coeffs)
    for x in range(10):
        assert f1(x) == f2(x)
    print('All tests passed')
    
if __name__ == "__main__":
    test()