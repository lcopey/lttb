use lttb;
use ndarray::{stack, Array, Axis};
use num_traits::Float;
fn main() {
    let x = Array::linspace(-3., 3., 2000);
    let y = x.map(|v| (-v.powi(2)).exp());
    let value = stack![Axis(1), x, y];
    println!("{:}", lttb::auto_lttb(value.view()));
}
