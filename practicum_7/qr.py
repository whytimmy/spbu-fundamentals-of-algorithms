import numpy as np
import matplotlib.pyplot as plt

from src.common import NDArrayFloat


def qr(A: NDArrayFloat) -> tuple[NDArrayFloat, NDArrayFloat]:
    n = A.shape[0]
    W = A.copy()
    Q = np.zeros_like(A)
    R = np.zeros_like(A)

    for j in range(n):
        w_j_norm = np.linalg.norm(W[:, j])
        Q[:, j] = W[:, j] / w_j_norm #W[:, j] == w_j^j
        for i in range(j):
            R[i, j] = A[:, j] @ Q[:, i]

        a_j_norm = np.linalg.norm(A[:, j])
        R[j, j] = np.sqrt(a_j_norm**2 - np.sum(R[:j, j] ** 2))
        for k in range(j + 1, n):
            prod = W[:, k] @ Q[:, j]
            W[:, k] -= prod * Q[:, j]

    return Q, R

def get_eigenvalues_via_qr(A: NDArrayFloat, n_iters: int) -> NDArrayFloat:
    A_k = A.copy()

    for k in range(n_iters):
        Q, R = qr(A_k)
        A_k = R @ Q
    return np.diag(A_k)

def householder_tridiagonalization(A: NDArrayFloat) -> NDArrayFloat:
    A_k = A.copy()
    n = A.shape[0]
    for i in range(n - 2):
        x = np.zeros(n)
        x[i + 1:] = A_k[i + 1:, i]
        y = np.zeros_like(x)
        y[i + 1] = -1 * sign(x[i + 1]) * np.linalg.norm(x)
        u = (x - y) / np.linalg.norm(x - y)
        H = np.eye(n) - 2 * np.outer(u, u)
        A_k = H @ A_k @ H

    return A_k

def sign(x):
    return 1 if x > 0 else -1


if __name__ == "__main__":
    A = np.array(
        [
            [4.0, 1.0, -1.0, 2.0],
            [1.0, 4.0, 1.0, -1.0],
            [-1.0, 1.0, 4.0, 1.0],
            [2.0, -1.0, 1.0, 1.0],
        ]
    )
    #Q, R = qr(A)
    #print(Q @ R)
    #eigvals = get_eigenvalues_via_qr(A, n_iters=20)
    #print(eigvals)

    A_tri = householder_tridiagonalization(A)
    print(A_tri)
    eigvals_tri = get_eigenvalues_via_qr(A_tri, n_iters=20)



#%%
