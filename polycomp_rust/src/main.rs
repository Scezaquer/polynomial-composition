mod polynomial;

use crate::polynomial::Polynomial;

fn main() {
    let poly = Polynomial::new(vec![1.0, 2.0, 3.0]);
    let poly2 = Polynomial::new(vec![1.0, 2.0, 0.0, -6.0]);
    let poly3 = poly.add(&poly2);

    println!("{}", poly3.to_string());

    let poly4 = poly.compose(&poly2);
    println!("{}", poly4.to_string());

    let poly5 = poly.convolve(&poly2);
    println!("{}", poly5.to_string());

    println!("{}", poly.eval(-32.7));

    let poly6 = Polynomial::new(vec![0.0]);
    println!("{}", poly6.trim().to_string());
}
