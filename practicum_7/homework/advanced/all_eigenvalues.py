from collections import defaultdict
from dataclasses import dataclass
import os
import yaml
import time


import numpy as np
import scipy.io
import scipy.linalg

from src.common import NDArrayFloat
from src.linalg import get_numpy_eigenvalues


@dataclass
class Performance:
    time1: float = 0.0
    relative_error: float = 0.0

def qr(A: np.array):
    rows, cols = A.shape
    Q = A.copy()
    R = np.zeros((cols, cols))

    for i in range(cols):
        # Нормализация i-го столбца
        norm = np.linalg.norm(Q[:, i])
        R[i, i] = norm
        Q[:, i] = Q[:, i] / norm

        for j in range(i + 1, cols):
            # Проекция j-го столбца на i-й столбец
            proj = np.dot(Q[:, i], Q[:, j])
            R[i, j] = proj
            Q[:, j] = Q[:, j] - proj * Q[:, i]

    return Q, R


def get_all_eigenvalues(A: NDArrayFloat) -> NDArrayFloat:
    max_iter = 100
    A_k = A.copy()

    for k in range(max_iter):
        Q, R = qr(A_k)
        A_k = R @ Q

    return np.array(np.diag(A_k))

def run_test_cases(
        path_to_homework: str, path_to_matrices: str
) -> dict[str, Performance]:
    matrix_filenames = []
    performance_by_matrix = defaultdict(Performance)
    with open("matrices.yaml", "r") as f:
        matrix_filenames = yaml.safe_load(f)

    matrices_dir = "C:\_IT\SPbU-Algos\practicum_7\matrices"


    for i, matrix_filename in enumerate(matrix_filenames):
        print(f"Processing matrix {i+1} out of {len(matrix_filenames)}")

        full_matrix_path = matrices_dir + "/" + matrix_filename

        A = scipy.io.mmread(full_matrix_path).todense().A
        perf = performance_by_matrix[matrix_filename]
        t1 = time.time()
        eigvals = get_all_eigenvalues(A)
        t2 = time.time()
        eigvals_exact = get_numpy_eigenvalues(A)
        perf.time1 += t2 - t1
        eigvals_exact.sort()
        eigvals.sort()
        perf.relative_error = np.median(
            np.abs(eigvals_exact - eigvals) / np.abs(eigvals_exact)
        )
    return performance_by_matrix


if __name__ == "__main__":
    path_to_homework = "practicum_7/homework/advanced"
    path_to_matrices = "matrices"
    performance_by_matrix = run_test_cases(
        path_to_homework=path_to_homework,
        path_to_matrices=path_to_matrices,
    )

    print("\nResult summary:")
    for filename, perf in performance_by_matrix.items():
        print(
            f"Matrix: {filename}. "
            f"Average time: {perf.time1:.2e} seconds. "
            f"Relative error: {perf.relative_error:.2e}"
        )


"""
Result summary:
Matrix: bp__1000.mtx.gz. Average time: 2.81e+02 seconds. Relative error: 7.88e-01
Matrix: e05r0100.mtx.gz. Average time: 1.70e+01 seconds. Relative error: 7.82e-02
Matrix: fs_541_1.mtx.gz. Average time: 1.02e+02 seconds. Relative error: 4.09e-04
Matrix: fs_680_1.mtx.gz. Average time: 1.71e+02 seconds. Relative error: 1.39e-08
Matrix: gre_1107.mtx.gz. Average time: 6.30e+02 seconds. Relative error: 7.27e-01
Matrix: hor__131.mtx.gz. Average time: 6.34e+01 seconds. Relative error: 3.66e-01
Matrix: impcol_c.mtx.gz. Average time: 5.54e+00 seconds. Relative error: 6.65e-01
Matrix: impcol_d.mtx.gz. Average time: 6.02e+01 seconds. Relative error: 8.21e-01
Matrix: impcol_e.mtx.gz. Average time: 1.63e+01 seconds. Relative error: 7.61e-01
Matrix: jpwh_991.mtx.gz. Average time: 4.88e+02 seconds. Relative error: 2.09e-03
Matrix: lns__511.mtx.gz. Average time: 9.81e+01 seconds. Relative error: 9.92e-01
Matrix: mahindas.mtx.gz. Average time: 8.51e+02 seconds. Relative error: 8.74e-01
Matrix: mcca.mtx.gz. Average time: 9.48e+00 seconds. Relative error: 9.35e-06
Matrix: mcfe.mtx.gz. Average time: 2.58e+02 seconds. Relative error: 8.02e-02
Matrix: nos5.mtx.gz. Average time: 7.49e+01 seconds. Relative error: 4.30e-03
Matrix: orsirr_1.mtx.gz. Average time: 5.14e+02 seconds. Relative error: 3.99e-02
"""