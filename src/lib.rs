use std::cmp::min;
use std::fmt::Display;

use ndarray::{s, Array, Array2, ArrayView2, Axis};
use num_traits::{Float, FromPrimitive};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PySequence;

pub fn lttb_from_array<F>(value: ArrayView2<F>, n: usize) -> Array2<F>
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
            .expect("Something went wrong when calculating next average. Please check that the array have the right dimensions.");

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
            .max_by(|(_, area1), (_, area2)| {
                area1
                    .partial_cmp(area2)
                    .expect("Something went wrong when comparing area")
            })
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
    lttb_from_array(value.view(), n)
        .axis_iter(Axis(0))
        .map(|row| row.iter().cloned().collect())
        .collect()
}

fn trapz<F>(value: &ArrayView2<F>) -> F
where
    F: Float + std::iter::Sum + Display,
{
    if value.len_of(Axis(0)) < 2 {
        return F::from(0.).unwrap();
    }
    value
        .slice(s![0..value.len_of(Axis(0)) - 1, ..])
        .axis_iter(Axis(0))
        .enumerate()
        .map(|(iteration, row)| {
            let next = value.index_axis(Axis(0), iteration + 1);
            (next[0] - row[0]) * F::from(0.5).unwrap() * (row[1] + next[1])
        })
        .sum()
}

pub fn auto_lttb<F>(value: ArrayView2<F>) -> Array2<F>
where
    F: Float + std::iter::Sum + FromPrimitive + Display,
{
    let integral = trapz(&value);
    let len = value.len_of(Axis(0));
    let power = (len as f64).log10().floor() as usize;
    let points = Array::linspace(-(power as f64), 0., power * 4)
        .map(|v| (f64::powf(10., *v) * (len as f64)).round() as usize);

    let mut i = 0;
    while i < points.len() - 1 {
        let sampled = lttb_from_array(value.view(), *points.get(i).unwrap());
        let loss = ((trapz(&sampled.view()) - integral) / integral).abs();
        if loss < F::from_f64(1e-4).unwrap() {
            return sampled;
        }
        i += 1;
    }
    value.to_owned()
}

#[pyfunction]
#[pyo3(name = "lttb_from_list")]
fn py_lttb_from_list(value: &PySequence, n: usize) -> PyResult<Vec<Vec<f64>>> {
    let value: Vec<Vec<f64>> = value.extract().unwrap();
    Ok(lttb_from_list(value, n))
}

#[pyfunction]
#[pyo3(name = "lttb_from_array")]
fn py_lttb_from_array<'py>(
    py: Python<'py>,
    value: PyReadonlyArray2<f64>,
    n: usize,
) -> PyResult<&'py PyArray2<f64>> {
    let result = lttb_from_array(value.as_array(), n).into_pyarray(py);
    Ok(result)
}

#[pyfunction]
#[pyo3(name = "auto_lttb")]
fn py_auto_lttb<'py>(
    py: Python<'py>,
    value: PyReadonlyArray2<f64>,
) -> PyResult<&'py PyArray2<f64>> {
    Ok(auto_lttb(value.as_array()).into_pyarray(py))
}

#[pymodule]
fn lttb(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_lttb_from_list, m)?)?;
    m.add_function(wrap_pyfunction!(py_lttb_from_array, m)?)?;
    m.add_function(wrap_pyfunction!(py_auto_lttb, m)?)?;

    Ok(())
}
