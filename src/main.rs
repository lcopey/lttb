use std::cmp::min;

use ndarray::{self, s, stack, Array, Array2, Axis};

fn lttb(value: Array2<f64>, n: usize) -> Array2<f64> {
    let len = value.len_of(Axis(0));
    let step = (len - 2) as f64 / (n - 2) as f64;
    let mut previous = value.index_axis(Axis(0), 0);

    let mut indices = Vec::<usize>::with_capacity(n);
    indices.push(0);

    for bucket in 0..n - 2 {
        let current_start_idx = (bucket as f64 * step + 1.).floor() as usize;
        let current_end_idx = ((bucket as f64 + 1.) * step + 1.).floor() as usize;

        let next_start_idx = current_end_idx;
        let next_end_idx = min(len, ((bucket as f64 + 2.) * step + 1.).floor() as usize);

        let avg_next = value
            .slice(s![next_start_idx..next_end_idx, ..])
            .mean_axis(Axis(0))
            .unwrap();

        let (best, _) = value
            .slice(s![current_start_idx..current_end_idx, ..])
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(iteration, v)| {
                let xa = previous[0];
                let ya = previous[1];
                let xb = v[0];
                let yb = v[1];
                let xc = avg_next[0];
                let yc = avg_next[1];
                let area = 0.5 * ((xa - xc) * (yb - ya) - (xa - xb) * (yc - ya)).abs();
                (iteration, area)
            })
            .max_by(|(_, area1), (_, area2)| area1.partial_cmp(area2).unwrap())
            .unwrap();
        let best = current_start_idx + best;
        indices.push(best);
        previous = value.index_axis(Axis(0), best);
    }
    indices.push(len - 1);
    value.select(Axis(0), &indices)
}

fn main() {
    let x = Array::linspace(-3., 3., 100);
    let y = x.map(|v: &f64| (-v * v).exp());
    let data = stack![Axis(1), x, y];
    println!("{:?}", lttb(data, 5));
}
