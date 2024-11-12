mod polynomial;

use crate::polynomial::Polynomial;
use rand::prelude::*;
use std::time::Instant;

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

fn linspace(start: f64, end: f64, num: usize) -> Vec<f64> {
    let step = (end - start) / (num as f64 - 1.0);
    (0..num).map(|i| start + i as f64 * step).collect()
}

fn gradient_descent(config: &mut GradientDescentConfig) -> Vec<Polynomial> {
    // Timers to profile the code
    let mut init_timer = 0.;
    let mut forward_pass_timer = 0.;
    let mut backprop_timer = 0.;
    let mut adam_timer = 0.;
    let mut io_timer = 0.;
    let mut main_loop_timer = 0.;
    let mut loss_timer = 0.;
    let mut loop_init_timer = 0.;

    let start = Instant::now();

    let mut layers;
    if let Some(layer_vec) = &config.layers {
        layers = layer_vec.clone();
    } else {
        layers = config
            .random_init_deg
            .iter()
            .map(|&deg| Polynomial::random_polynomial(deg, &mut config.rng, -0.5, 0.5))
            .collect::<Vec<Polynomial>>();
    }

    // Initialize the target polynomial
    let target;
    if let Some(target_poly) = &config.target {
        target = target_poly.clone();
    } else {
        target = Polynomial::random_polynomial(config.random_target_poly_deg, &mut config.rng, -2.5, 2.5);
    }

    // Initialize the losses vector
    let mut losses = vec![0.0; config.max_iter];
    let mut loss = target.l2_norm(&Polynomial::compose_vec(layers.clone()));
    losses[0] = loss;

    // Print the initial conditions
    if config.verbose {
        println!(" Layers:");
        for layer in layers.iter() {
            println!("{}", layer.to_string());
        }
        println!("Target: {}", target.to_string());
        println!("Initial loss: {}", loss);
    }

    let mut iteration = 0;

    // Initialize the Adam optimizer
    let mut m;
    let mut v;
    if config.use_adam {
        m = layers
            .iter()
            .map(|_| Polynomial::new(vec![0.0]))
            .collect::<Vec<Polynomial>>();
        v = layers
            .iter()
            .map(|_| Polynomial::new(vec![0.0]))
            .collect::<Vec<Polynomial>>();
    } else {
          m = vec![];
          v = vec![];
    }

    // Create the gradient, which has the same shape as the layers and is initialized to 0
    let mut grad = layers
        .iter()
        .map(|poly| vec![0.0; poly.get_degree().unwrap() + 1])
        .collect::<Vec<Vec<f64>>>();

    init_timer += start.elapsed().as_secs_f64();

    // Main loop
    while (loss > config.stop_loss) && (iteration < config.max_iter) {
        let start2 = Instant::now();

        // Initialize the gradient to 0
        for row in grad.iter_mut() {
            row.fill(0.0); // More efficient than manually iterating if you're resetting large rows
        }

        // Compute the polynomial derivatives
        let poly_derivatives = layers
            .iter()
            .map(|poly| poly.derivative())
            .collect::<Vec<Polynomial>>();

        let forward_pass_pts = linspace(0.0, 1.0, config.batch_size);
        /*
        // Pick random points for the forward pass
        let forward_pass_pts = (0..config.batch_size)
            .map(|_| config.rng.gen_range(0.0..1.0))
            .collect::<Vec<f64>>();
        */

        loop_init_timer += start2.elapsed().as_secs_f64();

        let start3 = Instant::now();

        // Compute the activations of each layer
        let mut activations = vec![vec![0.0; layers.len() + 1]; config.batch_size];
        for (i, row) in activations.iter_mut().enumerate() {
            row[0] = forward_pass_pts[i];
        }

        for (i, layer) in layers.iter().enumerate() {
            for j in 0..config.batch_size {
                activations[j][i+1] = layer.eval(activations[j][i]);
            }
        }

        // Compute the derivative of each layer at the activations
        let mut activation_derivatives = vec![vec![0.0; layers.len()]; config.batch_size];
        for (i, layer) in poly_derivatives.iter().enumerate() {
            for j in 0..config.batch_size {
                activation_derivatives[j][i] = layer.eval(activations[j][i]);
            }
        }

        forward_pass_timer += start3.elapsed().as_secs_f64();

        let start4 = Instant::now();

        // Backpropagation
        let mut dloss = activations
            .iter()
            .enumerate()
            .map(|(i, a)| (a.last().unwrap() - target.eval(forward_pass_pts[i])))
            .collect::<Vec<f64>>();

        for i in (0..layers.len()).rev() {
            for j in 0..config.batch_size {
                grad[i] = grad[i]
                    .iter()
                    .enumerate()
                    .map(|(k, val)| val + dloss[j] * activations[j][i].powi(k as i32))
                    .collect::<Vec<f64>>();
                dloss[j] = dloss[j] * activation_derivatives[j][i];
            }
        }

        // Normalize the gradient
        grad = grad
            .iter()
            .map(|row| {
                row.iter()
                    .map(|val| val / config.batch_size as f64)
                    .collect()
            })
            .collect();

        backprop_timer += start4.elapsed().as_secs_f64();

        let start5 = Instant::now();

        if config.use_adam {
            // Adam optimizer
            for i in 0..layers.len() {
                m[i] = m[i].scale(config.beta1).add(&Polynomial::new(
                    grad[i]
                        .iter()
                        .map(|val| val * (1.0 - config.beta1))
                        .collect(),
                ));
                v[i] = v[i].scale(config.beta2).add(&Polynomial::new(
                    grad[i]
                        .iter()
                        .map(|val| val * (1.0 - config.beta2))
                        .collect(),
                ));

                let m_hat = m[i].scale(1.0 / (1.0 - config.beta1.powi(iteration as i32 + 1)));
                let v_hat = v[i].scale(1.0 / (1.0 - config.beta2.powi(iteration as i32 + 1)));

                let lr = if config.use_scale_lr {
                    config.learning_rate * (i + 1) as f64
                } else {
                    config.learning_rate
                };

                let v_hat_sqrt = Polynomial::new(v_hat.get_coefficients().iter().map(|&x| x.sqrt()).collect());
                let denom = v_hat_sqrt.add(&Polynomial::new(vec![config.epsilon]));
                let update = Polynomial::new(
                    m_hat
                        .get_coefficients()
                        .iter()
                        .zip(denom.get_coefficients().iter())
                        .map(|(&m, &d)| -m / d * lr)
                        .collect(),
                );
                layers[i] = layers[i].add(&update);
            }
        } else {
            // Vanilla gradient descent
            for i in 0..layers.len() {
                layers[i] =
                    layers[i].add(&Polynomial::new(grad[i].clone()).scale(-1.0 * config.learning_rate));
            }
        }

        adam_timer += start5.elapsed().as_secs_f64();

        let start6 = Instant::now();

        loss = target.l2_norm(&Polynomial::compose_vec(layers.clone()));
        losses[iteration] = loss;
        loss_timer += start6.elapsed().as_secs_f64();

        let start7 = Instant::now();
        if config.verbose && iteration % config.print_frequency == 0 {
            println!("Iteration: {}, Loss: {}", iteration, loss);
        }
        iteration += 1;
        io_timer += start7.elapsed().as_secs_f64();
    }

    main_loop_timer += start.elapsed().as_secs_f64();

    if config.verbose {
        println!("Final loss: {}", loss);
        
        println!("Final polynomials:");
        for layer in layers.iter() {
            println!("{}", layer.to_string());
        }
        println!();
        println!("{}", Polynomial::compose_vec(layers.clone()).to_string());
    }

    println!("Init time: {}", init_timer);
    println!("Loop init time: {}", loop_init_timer);
    println!("Forward pass time: {}", forward_pass_timer);
    println!("Backprop time: {}", backprop_timer);
    println!("Adam time: {}", adam_timer);
    println!("Loss time: {}", loss_timer);
    println!("IO time: {}", io_timer);
    println!("Main loop time: {}", main_loop_timer);

    return layers;
}

fn main() {
    let mut config = GradientDescentConfig::default();

    let target = Polynomial::new(vec![1.0; 5]);
    config.target = Some(target);

    let layers = vec![
        Polynomial::new(vec![-0.1, 0.1, 0.2]),
        Polynomial::new(vec![-0.1, 0.1, 0.2]),
    ];
    config.layers = Some(layers);

    //config.print_frequency = 1;

    gradient_descent(&mut config);
}
