use ndarray::{Array1, Array2, s};
use rand::prelude::*;
use std::time::Instant;
use std::ops::AddAssign;

#[derive(Clone, Debug)]
pub struct Polynomial {
    coefficients: Array1<f64>,
}

impl Polynomial {
    pub fn new(coefficients: Vec<f64>) -> Self {
        let mut poly = Self {
            coefficients: Array1::from(coefficients),
        };
        poly.update_degree();
        poly
    }

    pub fn get_coefficients(&self) -> &Array1<f64> {
        &self.coefficients
    }

    pub fn random_polynomial(degree: usize, rng: &mut StdRng, range_start: f64, range_end: f64) -> Self {
        let coefficients: Vec<f64> = (0..=degree)
            .map(|_| rng.gen_range(range_start..range_end))
            .collect();
        Self::new(coefficients)
    }

    pub fn update_degree(&mut self) {
        let last_non_zero = self.coefficients.iter().rposition(|&x| x != 0.0);
        if let Some(idx) = last_non_zero {
            self.coefficients = self.coefficients.slice(s![..=idx]).to_owned();
        } else {
            self.coefficients = Array1::zeros(1);
        }
    }

    pub fn is_zero(&self) -> bool {
        self.coefficients.len() == 1 && self.coefficients[0] == 0.0
    }

    pub fn eval(&self, x: f64) -> f64 {
        self.coefficients.iter().rev().fold(0.0, |acc, &c| acc * x + c)
    }

    pub fn add(&self, other: &Polynomial) -> Polynomial {
        let max_len = self.coefficients.len().max(other.coefficients.len());
        let mut result = Array1::zeros(max_len);
        result.slice_mut(s![..self.coefficients.len()]).add_assign(&self.coefficients);
        result.slice_mut(s![..other.coefficients.len()]).add_assign(&other.coefficients);
        Polynomial::new(result.to_vec())
    }

    pub fn scale(&self, scalar: f64) -> Polynomial {
        Polynomial::new((self.coefficients.to_owned() * scalar).to_vec())
    }

    pub fn compose(&self, other: &Polynomial) -> Polynomial {
        if self.is_zero() {
            return Polynomial::new(vec![0.0]);
        }
        if other.is_zero() {
            return Polynomial::new(vec![self.coefficients[0]]);
        }

        let mut result = Array1::zeros(self.coefficients.len() * other.coefficients.len());
        let mut power = Array1::from_vec(vec![1.0]);

        for (i, &c) in self.coefficients.iter().enumerate() {
            if c != 0.0 {
                result.slice_mut(s![..power.len()]).add_assign(&(&power.view() * c));
            }
            if i < self.coefficients.len() - 1 {
                power = convolve(&power, &other.coefficients);
            }
        }

        Polynomial::new(result.to_vec())
    }

    pub fn derivative(&self) -> Polynomial {
        if self.is_zero() {
            return Polynomial::new(vec![]);
        }

        let derivative_coeffs = self.coefficients.slice(s![1..])
            .iter()
            .enumerate()
            .map(|(i, &c)| c * (i + 1) as f64)
            .collect();

        Polynomial::new(derivative_coeffs)
    }
}

fn convolve(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
    let mut result = Array1::zeros(a.len() + b.len() - 1);
    for (i, &ai) in a.iter().enumerate() {
        result.slice_mut(s![i..i+b.len()]).add_assign(&(b * ai));
    }
    result
}

pub fn compose_vec(polys: &[Polynomial]) -> Polynomial {
    polys.iter().fold(Polynomial::new(vec![0.0, 1.0]), |acc, poly| acc.compose(poly))
}

pub fn l2_norm(p1: &Polynomial, p2: &Polynomial) -> f64 {
    let diff = p1.add(&p2.scale(-1.0));
    if diff.is_zero() {
        return 0.0;
    }

    let coeff = diff.get_coefficients();
    let n = coeff.len();
    let i = Array1::range(0.0, n as f64, 1.0);
    
    let r1 = (&coeff.view() * &coeff.view() / (2.0 * &i + 1.0)).sum();
    
    let i_2d = i.broadcast((n, n)).unwrap().to_owned();
    let j_2d = i_2d.t().to_owned();
    let mask = (&i_2d > &j_2d).mapv(|&x| if x { 1.0 } else { 0.0 });
    
    let coeff_2d = coeff.broadcast((n, n)).unwrap().to_owned();
    let r2 = 2.0 * ((&mask * &coeff_2d * &coeff_2d.t()) / (i_2d + j_2d + 1.0)).sum();
    */
    r1 + r2
}

#[derive(Clone, Debug)]
pub struct GradientDescentConfig {
    pub target: Option<Polynomial>,
    pub layers: Option<Vec<Polynomial>>,
    pub random_target_poly_deg: usize,
    pub random_init_deg: Vec<usize>,
    pub rng: StdRng,
    pub max_iter: usize,
    pub batch_size: usize,
    pub stop_loss: f64,
    pub learning_rate: f64,
    pub use_adam: bool,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub verbose: bool,
    pub plot: bool,
    pub use_scale_lr: bool,
    pub print_frequency: usize,
}

impl Default for GradientDescentConfig {
    fn default() -> Self {
        GradientDescentConfig {
            target: None,
            layers: None,
            random_target_poly_deg: 27,
            random_init_deg: vec![3, 3, 3],
            rng: StdRng::from_entropy(),
            max_iter: 10000,
            batch_size: 2000,
            stop_loss: 1e-10,
            learning_rate: 0.001,
            use_adam: false,
            beta1: 0.99,
            beta2: 0.999,
            epsilon: 1e-8,
            verbose: true,
            plot: true,
            use_scale_lr: false,
            print_frequency: 250,
        }
    }
}

pub fn gradient_descent(config: &mut GradientDescentConfig) -> Vec<Polynomial> {
    let start = Instant::now();

    let mut layers = config.layers.clone().unwrap_or_else(|| {
        config.random_init_deg.iter()
            .map(|&deg| Polynomial::random_polynomial(deg, &mut config.rng, -0.5, 0.5))
            .collect()
    });

    let target = config.target.clone().unwrap_or_else(|| {
        Polynomial::random_polynomial(config.random_target_poly_deg, &mut config.rng, -2.5, 2.5)
    });

    let mut loss = l2_norm(&target, &compose_vec(&layers));
    let mut iteration = 0;

    let mut m: Vec<Array1<f64>> = Vec::new();
    let mut v: Vec<Array1<f64>> = Vec::new();
    if config.use_adam {
        m = layers.iter().map(|layer| Array1::zeros(layer.get_coefficients().len())).collect();
        v = layers.iter().map(|layer| Array1::zeros(layer.get_coefficients().len())).collect();
    }

    let mut grad: Vec<Array1<f64>> = layers.iter()
        .map(|layer| Array1::zeros(layer.get_coefficients().len()))
        .collect();

    while loss > config.stop_loss && iteration < config.max_iter {
        for g in grad.iter_mut() {
            g.fill(0.0);
        }

        let poly_derivatives: Vec<Polynomial> = layers.iter().map(|poly| poly.derivative()).collect();

        let forward_pass_pts = Array1::linspace(0.0, 1.0, config.batch_size);

        let mut activations = Array2::zeros((config.batch_size, layers.len() + 1));
        activations.column_mut(0).assign(&forward_pass_pts);

        for (i, layer) in layers.iter().enumerate() {
            let col = activations.column(i).to_owned();
            activations.column_mut(i + 1).assign(&col.map(|&x| layer.eval(x)));
        }

        let mut activation_derivatives = Array2::zeros((config.batch_size, layers.len()));
        for (i, layer) in poly_derivatives.iter().enumerate() {
            let col = activations.column(i);
            activation_derivatives.column_mut(i).assign(&col.map(|&x| layer.eval(x)));
        }

        let mut dloss = activations.column(layers.len()).to_owned() - forward_pass_pts.map(|&x| target.eval(x));

        for i in (0..layers.len()).rev() {
            for j in 0..layers[i].get_coefficients().len() {
                grad[i][j] = (&dloss * &activations.column(i).map(|&x| x.powi(j as i32))).sum();
            }
            if i > 0 {
                dloss *= &activation_derivatives.column(i);
            }
        }

        for g in grad.iter_mut() {
            *g /= config.batch_size as f64;
        }

        if config.use_adam {
            for i in 0..layers.len() {
                m[i] = &m[i] * config.beta1 + &grad[i] * (1.0 - config.beta1);
                v[i] = &v[i] * config.beta2 + &(&grad[i] * &grad[i]) * (1.0 - config.beta2);
                
                let m_hat = &m[i] / (1.0 - config.beta1.powi(iteration as i32 + 1));
                let v_hat = &v[i] / (1.0 - config.beta2.powi(iteration as i32 + 1));
                
                let lr = if config.use_scale_lr {
                    config.learning_rate * (i + 1) as f64
                } else {
                    config.learning_rate
                };

                let update = &m_hat / (&v_hat.map(|x: &f64| x.sqrt()) + config.epsilon) * (-lr);
                layers[i] = Polynomial::new((layers[i].get_coefficients() + &update).to_vec());
            }
        } else {
            for i in 0..layers.len() {
                let lr = if config.use_scale_lr {
                    config.learning_rate * (i + 1) as f64
                } else {
                    config.learning_rate
                };
                layers[i] = Polynomial::new((layers[i].get_coefficients() - &(&grad[i] * lr)).to_vec());
            }
        }

        loss = l2_norm(&target, &compose_vec(&layers));

        if config.verbose && iteration % config.print_frequency == 0 {
            println!("Iteration: {}, Loss: {}", iteration, loss);
        }
        iteration += 1;
    }

    if config.verbose {
        println!("Final loss: {}", loss);
        println!("Final polynomials:");
        for layer in &layers {
            println!("{:?}", layer);
        }
        println!();
        println!("{:?}", compose_vec(&layers));
    }

    println!("Total time: {:?}", start.elapsed());

    layers
}

fn main() {
    let mut config = GradientDescentConfig::default();

    let target = Polynomial::new(vec![1.0; 28]);
    config.target = Some(target);

    let layers = vec![
        Polynomial::new(vec![-0.1, 0.1, 0.2, 0.3]),
        Polynomial::new(vec![-0.1, 0.1, 0.2, 0.3]),
        Polynomial::new(vec![-0.1, 0.1, 0.2, 0.3]),
    ];
    config.layers = Some(layers);

    gradient_descent(&mut config);
}