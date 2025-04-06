use ndarray::Array2;
use ndarray::array;

/// Naive matrix multiplication for 2D arrays.
/// # Examples
pub fn matmul(x1: Array2<f64>, x2: Array2<f64>) -> Array2<f64> {
    assert_eq!(
        x1.ncols(),
        x2.nrows(),
        "Incompatible matrix dimensions {:?}, {:?}",
        x1.dim(),
        x2.dim()
    );
    let (m, n) = x1.dim();
    let p = x2.ncols();
    let mut res: Array2<f64> = Array2::zeros((x1.nrows(), x2.ncols()));
    for i in 0..m {
        for j in 0..p {
            let mut sum: f64 = 0.0;
            for k in 0..n {
                sum += x1[(i, k)] * x2[(k, j)];
            }
            res[(i, j)] = sum;
        }
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_square() {
        // (2,2) @ (2,2) = (2,2)
        let x1 = array![[1.0, 1.0], [2.0, 2.0]];
        let x2 = array![[1.0, 2.0], [1.0, 2.0]];
        let result = matmul(x1, x2);
        assert_eq!(result, array![[2.0, 4.0], [4.0, 8.0]])
    }

    #[test]
    fn test_matmul_rectangle() {
        // (3,2) @ (2,3) = (3,3)
        let x1 = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let x2 = array![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]];
        let result = matmul(x1, x2);
        assert_eq!(
            result,
            array![[2.0, 4.0, 6.0], [4.0, 8.0, 12.0], [6.0, 12.0, 18.0]]
        )
    }

    #[test]
    #[should_panic(expected = "Incompatible matrix dimensions (2, 1), (2, 1)")]
    fn test_requires_matching_inner_dims() {
        let x1 = array![[1.0,], [1.0,]]; // (2,1)
        let x2 = array![[4.0], [8.0]];
        matmul(x1, x2);
    }
}
