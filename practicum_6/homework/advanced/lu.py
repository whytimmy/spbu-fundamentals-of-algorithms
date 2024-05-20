from collections import defaultdict
from dataclasses import dataclass
import os
import time
import yaml


import numpy as np
from numpy.typing import NDArray
import scipy.io
import scipy.linalg

from src.linalg import get_scipy_solution

@dataclass
class Performance:
    time1: float = 0.0
    relative_error: float = 0.0
def lu(A: NDArray, permute: bool) -> tuple[NDArray, NDArray, NDArray]:

    n = len(A)
    U = np.copy(A)
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n - 1):
        if permute:
            max_index = np.argmax(abs(U[k:, k])) + k
            if max_index != k:
                U[[k, max_index]] = U[[max_index, k]]
                P[[k, max_index]] = P[[max_index, k]]
                if k != 0:
                    L[[k, max_index], :k] = L[[max_index, k], :k]

        for s in range(k + 1, n):
            L[s][k] = U[s][k] / U[k, k]
            U[s] -= U[k] * U[s][k] / U[k, k]

    return L, U, P
def solve(L: NDArray, U: NDArray, P: NDArray, b: NDArray) -> NDArray:

    n = U.shape[0]
    x, y = np.zeros(n), np.zeros(n)
    b_rearranged = P.dot(b)

    for i in range(n):
        y[i] = b_rearranged[i] - L[i, :i].dot(y[:i])

    for i in range(n-1, -1, -1):
        x[i] = (y[i] - U[i, i+1:].dot(x[i+1:])) / U[i, i]

    return x
def run_test_cases(n_runs: int, path_to_homework: str) -> dict[str, Performance]:
    matrix_filenames = []
    performance_by_matrix = defaultdict(Performance)
    with open("matrices.yaml", "r") as f:
        matrix_filenames = yaml.safe_load(f)

    matrices_dir = "..."

    for i, matrix_filename in enumerate(matrix_filenames):
        print(f"Processing matrix {i+1} out of {len(matrix_filenames)}")

        full_matrix_path = matrices_dir + "/" + matrix_filename
        A = (
            scipy.io.mmread(full_matrix_path)
            .todense()
            .A
        )
        b = np.ones((A.shape[0],))
        perf = performance_by_matrix[matrix_filename]
        for j in range(n_runs):
            t1 = time.time()
            L, U, P = lu(A, permute=True)
            t2 = time.time()
            perf.time1 += t2 - t1
            if j == 0:  # first run => compute solution
                x = solve(L, U, P, b)
                x_exact = get_scipy_solution(A, b)
                perf.relative_error = np.linalg.norm(x - x_exact) / np.linalg.norm(
                    x_exact
                )
    return performance_by_matrix


if __name__ == "__main__":
    n_runs = 10
    path_to_homework = "..."
    performance_by_matrix = run_test_cases(
        n_runs=n_runs, path_to_homework=path_to_homework
    )

    print("\nResult summary:")
    for filename, perf in performance_by_matrix.items():
        print(
            f"Matrix: {filename}. "
            f"Average time: {perf.time1 / n_runs:.2e} seconds. "
            f"Relative error: {perf.relative_error:.2e}"
        )

"""
Result summary:
Matrix: add20.mtx.gz. Average time: 2.35e+01 seconds. Relative error: 6.54e-14
Matrix: bcsstk14.mtx.gz. Average time: 1.19e+01 seconds. Relative error: 2.59e-17
Matrix: bcsstk28.mtx.gz. Average time: 1.02e+02 seconds. Relative error: 1.10e-11
Matrix: bp__1000.mtx.gz. Average time: 2.02e+00 seconds. Relative error: 7.75e-15
Matrix: e05r0100.mtx.gz. Average time: 1.38e-01 seconds. Relative error: 2.03e-14
Matrix: fs_541_1.mtx.gz. Average time: 8.02e-01 seconds. Relative error: 2.28e-16
Matrix: fs_680_1.mtx.gz. Average time: 1.32e+00 seconds. Relative error: 4.63e-16
Matrix: gemat11.mtx.gz. Average time: 1.33e+02 seconds. Relative error: 1.72e-12
Matrix: gre_1107.mtx.gz. Average time: 3.80e+00 seconds. Relative error: 1.26e-12
Matrix: hor__131.mtx.gz. Average time: 4.53e-01 seconds. Relative error: 1.33e-15
Matrix: impcol_c.mtx.gz. Average time: 4.53e-02 seconds. Relative error: 8.60e-16
Matrix: impcol_d.mtx.gz. Average time: 4.43e-01 seconds. Relative error: 7.74e-15
Matrix: impcol_e.mtx.gz. Average time: 1.25e-01 seconds. Relative error: 1.19e-15
Matrix: jpwh_991.mtx.gz. Average time: 2.92e+00 seconds. Relative error: 3.51e-16
Matrix: lns__511.mtx.gz. Average time: 7.02e-01 seconds. Relative error: 3.63e-12
Matrix: mahindas.mtx.gz. Average time: 5.04e+00 seconds. Relative error: 1.37e-16
Matrix: mcca.mtx.gz. Average time: 7.97e-02 seconds. Relative error: 2.65e-11
Matrix: mcfe.mtx.gz. Average time: 1.66e+00 seconds. Relative error: 3.40e-16
Matrix: nnc1374.mtx.gz. Average time: 6.18e+00 seconds. Relative error: 1.57e-11
Matrix: nos5.mtx.gz. Average time: 5.38e-01 seconds. Relative error: 1.42e-14
Matrix: orani678.mtx.gz. Average time: 3.51e+01 seconds. Relative error: 1.79e-15
Matrix: orsirr_1.mtx.gz. Average time: 3.24e+00 seconds. Relative error: 2.12e-14
Matrix: west2021.mtx.gz. Average time: 1.53e+01 seconds. Relative error: 2.01e-12
"""