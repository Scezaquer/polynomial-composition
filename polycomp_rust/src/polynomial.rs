#[derive(Clone)]
pub struct Polynomial {
    degree: Option<usize>,  // degree of the polynomial. None for 0 polynomial
    coefficients: Vec<f64>, // coefficients[0] is the constant term
}

impl Polynomial {
    pub fn new(coefficients: Vec<f64>) -> Polynomial {
        let mut poly = Polynomial {
            degree: None,
            coefficients,
        };
        poly.update_degree();
        poly
    }

    pub fn update_degree(&mut self) {
        self.degree = self.compute_degree();
    }

    pub fn compute_degree(&self) -> Option<usize> {
        for (i, &coeff) in self.coefficients.iter().rev().enumerate() {
            if coeff != 0.0 {
                return Some(self.coefficients.len() - 1 - i);
            }
        }
        None // It's the zero polynomial
    }

    pub fn is_zero(&self) -> bool {
        self.degree.is_none()
    }

    /// Trim the polynomial by removing leading zeros
    pub fn trim(&self) -> Polynomial {
        if self.is_zero() {
            return self.clone();
        }

        let mut result = Polynomial::new(vec![]);
        let mut i = self.degree.unwrap();
        while i > 0 && self.coefficients[i] == 0.0 {
            i -= 1;
        }
        for j in 0..i + 1 {
            result.coefficients.push(self.coefficients[j]);
        }
        result.update_degree();
        return result;
    }

    /// Horner's method. Returns p(x)
    pub fn eval(&self, x: f64) -> f64 {
        if self.is_zero() {
            return 0.0;
        }

        let degree = self.degree.unwrap();
        let mut result = self.coefficients[degree];
        for i in (0..degree).rev() {
            result = result * x + self.coefficients[i];
        }
        return result;
    }

    /// Add two polynomials
    pub fn add(&self, other: &Polynomial) -> Polynomial {
        // If one of the polynomials is 0, return the other
        if self.is_zero() {
            return other.clone();
        }
        if other.is_zero() {
            return self.clone();
        }

        let self_degree = self.degree.unwrap();
        let other_degree = other.degree.unwrap();

        let mut result = Polynomial::new(vec![]);
        let mut i = 0;
        let mut j = 0;
        while i < self_degree + 1 && j < other_degree + 1 {
            result
                .coefficients
                .push(self.coefficients[i] + other.coefficients[j]);
            i += 1;
            j += 1;
        }
        while i < self_degree + 1 {
            result.coefficients.push(self.coefficients[i]);
            i += 1;
        }
        while j < other_degree + 1 {
            result.coefficients.push(other.coefficients[j]);
            j += 1;
        }

        result.update_degree();
        return result;
    }

    pub fn scale(&self, scalar: f64) -> Polynomial {
        let mut result = Polynomial::new(self.coefficients.iter().map(|&c| c * scalar).collect());
        result.update_degree();
        return result;
    }

    /// Compose two polynomials. Returns self(other(x))
    /// I think it might be better to chain eval calls? TBD
    pub fn compose(&self, other: &Polynomial) -> Polynomial {
        // If either polynomial is the zero polynomial, return zero
        if self.is_zero() {
            return Polynomial::new(vec![]);
        }
        if other.is_zero() {
            return Polynomial::new(vec![self.coefficients[0]]);
        }

        // Safe to assume both have degrees, we can proceed with the actual composition
        let self_degree = self.degree.unwrap();
        let other_degree = other.degree.unwrap();

        let mut result = Polynomial::new(vec![0.0; self_degree * other_degree + 1]);

        // Efficiently compute powers of the `other` polynomial
        let mut power = Polynomial::new(vec![1.0]);
        for (i, &c) in self.coefficients.iter().enumerate() {
            if c != 0.0 {
                result = result.add(&power.scale(c));
            }
            if i < self_degree {
                power = power.convolve(&other);
            }
        }
        result.update_degree();
        result
    }

    pub fn convolve(&self, other: &Polynomial) -> Polynomial {
        // If either polynomial is zero, return a zero polynomial
        if self.is_zero() || other.is_zero() {
            return Polynomial::new(vec![]);
        }

        // Safe unwrap since both degrees are Some at this point
        let self_degree = self.degree.unwrap();
        let other_degree = other.degree.unwrap();

        // Initialize result polynomial with zeros
        let mut result = Polynomial::new(vec![0.0; self_degree + other_degree + 1]);

        // Perform the convolution (polynomial multiplication)
        for i in 0..=self_degree {
            for j in 0..=other_degree {
                result.coefficients[i + j] += self.coefficients[i] * other.coefficients[j];
            }
        }

        // Update the degree of the resulting polynomial
        result.update_degree(); // This will adjust the degree correctly
        result
    }

    pub fn to_string(&self) -> String {
        if self.is_zero() {
            return "0".to_string();
        }
        let mut result = String::new();
        for i in 0..self.degree.unwrap() + 1 {
            if &self.coefficients[i] == &0.0 {
                continue;
            }

            if &self.coefficients[i] < &0.0 {
                result.push_str(" - ");
            } else if i > 0 {
                result.push_str(" + ");
            }

            result.push_str(&(self.coefficients[i]).abs().to_string());
            result.push_str("x^");
            result.push_str(&i.to_string());
        }
        return result;
    }
}
