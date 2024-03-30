use std::cmp::min;

use ndarray::{self, s, Array2, Axis};
use num_traits::{Float, FromPrimitive};
use pyo3::prelude::*;
use pyo3::types::PySequence;

fn lttb_from_array<F>(value: Array2<F>, n: usize) -> Array2<F>
where
    F: Float + FromPrimitive,
{
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
                let area = ((xa - xc) * (yb - ya) - (xa - xb) * (yc - ya)).abs();
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

fn lttb_from_list<F>(value: Vec<Vec<F>>, n: usize) -> Vec<Vec<F>>
where
    F: Float + FromPrimitive,
{
    let value =
        Array2::from_shape_vec((value.len(), 2), value.into_iter().flatten().collect()).unwrap();
    lttb_from_array(value, n)
        .axis_iter(Axis(0))
        .map(|row| row.iter().cloned().collect())
        .collect()
}

#[pyfunction]
#[pyo3(name="lttb_from_list")]
fn py_lttb_from_list(value: &Bound<'_, PySequence>, n: usize) -> PyResult<Vec<Vec<f64>>> {
    let value: Vec<Vec<f64>> = value.extract().unwrap();
    Ok(lttb_from_list(value, n))
}

#[pymodule]
fn lttb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_lttb_from_list, m)?)?;

    Ok(())
}
