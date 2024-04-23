#![allow(dead_code)]

use na::{matrix, SMatrix, SVector, Scalar};
use nalgebra as na;
use num_traits::{real::Real, One, Zero};
use rand::distributions::Standard;
use rand::prelude::Distribution;
use std::ops::{AddAssign, MulAssign};

struct Layer<T, const SIZE: usize, const BATCHES: usize> {
    weights: SMatrix<T, SIZE, BATCHES>,
    biases: SVector<T, SIZE>,
}

impl<T, const SIZE: usize, const BATCHES: usize> Default for Layer<T, SIZE, BATCHES>
where
    Standard: Distribution<SMatrix<T, SIZE, BATCHES>>,
    T: Real + Scalar + Zero + AddAssign + MulAssign,
{
    fn default() -> Self {
        let mut s = Self {
            weights: rand::random() * T::from(0.02).unwrap(),
            biases: Zero::zero(),
        };
        s.weights.add_scalar_mut(T::from(-0.01).unwrap());
        s
    }
}

impl<T, const SIZE: usize, const BATCHES: usize> Layer<T, SIZE, BATCHES>
where
    T: Scalar + Zero + One + AddAssign + MulAssign,
{
    fn new(weights: SMatrix<T, SIZE, BATCHES>, biases: SVector<T, SIZE>) -> Self {
        Self { weights, biases }
    }

    fn forward(
        &self,
        inputs: SMatrix<T, SIZE, BATCHES>,
        activation_function: impl FnMut(&mut T),
    ) -> SSquareMatrix<T, SIZE> {
        let mut outputs = inputs * self.weights.transpose();

        outputs
            .column_iter_mut()
            .zip(self.biases.iter())
            .for_each(|(mut column, &ref coefficient)| column.add_scalar_mut(coefficient.clone()));

        outputs.apply_into(activation_function)
    }
}

type SSquareMatrix<T, const N: usize> = SMatrix<T, N, N>;

fn step<T>(x: &mut T)
where
    T: Zero + One + PartialOrd,
{
    if *x > T::zero() {
        *x = T::one();
    } else {
        *x = T::zero();
    }
}

fn linear<T>(x: &mut T) {
    _ = x;
}

fn sigmoid<T>(x: &mut T)
where
    T: Real,
{
    *x = T::one() / (T::one() + (-*x).exp());
}

fn relu<T>(x: &mut T)
where
    T: Real + Zero,
{
    *x = x.max(T::zero());
}

fn main() {
    let layer1 = Layer::default();
    let layer2 = Layer::default();
    let layer3 = Layer::default();

    let outputs = layer1.forward(
        matrix![
        1.0, 2.0,  3.0,  2.5 ;
        2.0, 5.0, -1.0,  2.0 ;
       -1.5, 2.7,  3.3, -0.8],
        relu,
    );
    let outputs2 = layer2.forward(outputs, relu);
    let outputs3 = layer3.forward(outputs2, relu);

    println!("{}", outputs3);
}
