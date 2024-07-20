import numpy as np
import qutip as qt
import scipy.optimize as op
import matplotlib.pyplot as plt
from xmlrpc.client import MAXINT
import torch
from jax import grad
import os


# Initialize Parameters
L = 5
tole = 1e-4

"""
Tool Functions
"""


def initialize(n_, T_):
    # Initialize parameters with different (n,T) values
    global n, T, rho_size, a, b, I, Zero, cons, bonds
    n, T = n_, T_
    rho_size = 2**n
    a = 1 - 1 / (T * rho_size)
    b = 1 / (T * rho_size)
    I = np.eye(rho_size, dtype="complex_")
    Zero = np.zeros((rho_size, rho_size), dtype="complex_")
    cons = (
        {"type": "ineq", "fun": constr_norm},
        {"type": "ineq", "fun": constr_non_neg},
    )
    # Bounds
    bonds = bonds = op.Bounds(np.array([0] * rho_size), np.array([1] * rho_size))


def x_tilde(x):
    # mix rho
    # Input: (complex) x in Cn, 2-D array of shape (rho_size, rho_size)
    # Output: (complex) x_tilde in Cn, 2-D array of shape (rho_size, rho_size)
    return a * x + b * I


def compose_Matrix(eigenvalue, eVector):
    # Compose matrix from eigenvalues and eigenvectors
    evecList = np.array(np.split(eVector, eVector.shape[1], axis=1))
    eVec1 = np.einsum("k,kij -> kij", eigenvalue, evecList)
    eVec_dag = np.conjugate(np.transpose(eVector))
    eVec2 = np.array(np.split(eVec_dag, eVec_dag.shape[0], axis=0)).reshape(
        rho_size, 1, rho_size
    )
    x_new = np.einsum("kil, klj -> lij", eVec1, eVec2).reshape(rho_size, rho_size)
    return x_new


def gram_schmidt(vv):
    # Gram_Schmidt Orthogonalization, credit to "legendongary" on GitHub
    # Input: torch tensor with vectors in column
    # Output: torch tensor with othogonal vectors in colum
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(0)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[:, 0] = vv[:, 0].clone()
    for k in range(1, nk):
        vk = vv[k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:, j].clone()
            uk = uk + projection(uj, vk)
        uu[:, k] = vk - uk
    for k in range(nk):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()
    return uu


def generate_E():
    # Generate POVM operator randomly using qutip
    # Input: Nothing
    # Output: (complex matrix) Hermitian matrix with egienvalues in [0,1]
    egienvalue = np.random.random((rho_size,))
    Q = np.random.random((rho_size, rho_size)) + 1.0j * np.random.random(
        (rho_size, rho_size)
    )
    MQ = torch.from_numpy(Q)
    Q_vec = gram_schmidt(MQ).numpy()
    E = compose_Matrix(egienvalue, Q_vec)
    return E


def R(x):
    eigenvalue = np.linalg.eigvals(x).real
    # print("eigen=", eigenvalue)
    logeigen = np.log(eigenvalue)
    # print(logeigen)
    inter = eigenvalue * logeigen
    # print("inter=", inter)
    result = np.sum(inter)
    return result


def nabla_R(x):
    # Nabla_R(x)
    # Input: (complex) density matrix x
    # Output: (real matrix converted from complex) Gradient matrix
    eigenvalue, eVector = np.linalg.eig(x)
    eigenvalue = np.log(eigenvalue) + 1
    x_new = compose_Matrix(eigenvalue, eVector)
    return x_new


def evolve_rho(rho_last, Ht):
    # evovle rho according to rho = e^(Ht)*rho*e^(-Ht)
    eigenvalue, eVector = np.linalg.eig(Ht)
    eigenvalue = eigenvalue.real
    eigenCos = np.cos(eigenvalue)
    eigenSin = np.sin(eigenvalue)
    U = compose_Matrix(eigenCos, eVector) + compose_Matrix(eigenSin, eVector) * 1.0j
    rho_new = U @ rho_last @ np.conj(U.T)
    return rho_new


def l(zt, bt):
    # Loss function
    # Input: zt_x= trace(Et xt), bt_rho=trace(Et rho_t)
    # Output: (complex) L2 loss
    # P.S. The imagine component of trace is not exactly 0.0j due to computational accuracy limit.
    result = (bt - zt) ** 2 + 0.0j
    return result


def equation(B):
    # Update y_{t+1}
    # Input: (complex) matrix B
    # Output: (complex) density matrix y
    eigenvalue, eVector = np.linalg.eig(B)
    eigen_Y = np.exp(eigenvalue - 1)
    Y = compose_Matrix(eigen_Y, eVector)
    return Y


def min_func(Eigen, mu):
    # Minimize function in projection
    Eigen1 = a * Eigen + b
    if np.min(Eigen1) <= 0:
        print(np.min(Eigen1))
        return 10000000000
    Eigen2 = np.log(Eigen1)
    result = -a * np.sum(Eigen * np.log(mu)) + np.sum(Eigen1 * Eigen2)
    return result


def constr_norm(Eigen):
    # Constraint: 1-norm = 1
    return tole - np.abs(np.linalg.norm(Eigen, ord=1) - 1)


def constr_non_neg(Eigen):
    # Constraint: non-negtive
    return np.min(Eigen) + tole


def OMD_expert(x_last, Eigen_last, eta, Et, bt_rho):
    # instance of expert in Algorithm2(OMD)
    zt_x = np.trace(Et @ x_last)
    subD_l = grad(l, holomorphic=True)
    Nabla_t = subD_l(zt_x, bt_rho) * Et
    y_new = equation(nabla_R(x_last) - eta * Nabla_t)
    mu, vec = np.linalg.eig(y_new)
    mu = mu.real
    Eigen_solve = op.minimize(min_func, Eigen_last, mu, constraints=cons, tol=tole)
    if Eigen_solve.success == False:
        print(Eigen_solve)
    Eigen_new = Eigen_solve.x
    x_new_revtile = compose_Matrix(Eigen_new, vec)
    x_new = x_tilde(x_new_revtile)
    return x_new, Eigen_new


def dynamic_regret(H, alpha):
    k_max = int(np.log(T) + 1)
    # Initialization
    dynamic_Regret = 0
    P = 0
    W = np.ones((k_max,)) / k_max
    X_k = np.array([I / rho_size] * k_max, dtype="complex_")
    Eigen_k = np.zeros((k_max, rho_size))
    S_eta = np.zeros((k_max,))
    for k in range(k_max):
        S_eta[k] = 0.5 / (2 ** (k + 1))
    # Generate rho_0 randomly
    rho_last = qt.rand_dm(rho_size).full()
    rho_tilde = x_tilde(rho_last)
    x_pred = I / rho_size + 0.0j
    for t in range(T):
        Et = generate_E()
        # Predict rho, receive loss
        zt_x = np.trace(Et @ x_pred)
        bt_rho = np.trace(Et @ rho_tilde)
        lt = l(zt_x, bt_rho)
        dynamic_Regret += lt.real
        loss = np.zeros((k_max))
        for k in range(k_max):
            loss[k] = l(np.trace(Et @ X_k[k]), bt_rho).real
        # update x_pred
        loss = np.exp(-alpha * loss)
        W = W * loss
        Ht = (t + 1) / 500 * H
        rho_new = evolve_rho(rho_last, Ht)
        rho_tilde = x_tilde(rho_new)
        P += np.linalg.norm(rho_new - rho_last, ord="nuc")
        for k in range(k_max):
            X_k[k], Eigen_k[k] = OMD_expert(X_k[k], Eigen_k[k], S_eta[k], Et, bt_rho)
        x_pred = np.einsum("k,kij->ij", W, X_k) / np.sum(W)
        rho_last = rho_new
    return dynamic_Regret, P


def single_exper(n_, T_, N_):
    # Repeat experiment randomly for N_ times and take the upper bound
    initialize(n_, T_)
    D_regret = 0
    Path = 0
    for j in range(N_):
        H = qt.rand_herm(rho_size).full()
        regret_T, P = dynamic_regret(H, 1)
        if regret_T > D_regret:
            # might be changed !!!!
            D_regret = regret_T
            Path = P
    return D_regret, Path


def experiment(point_num, step_T, qubit_n, repeat_N, mar, col):
    # repeat experiment repeat_N times and plot subfigure(b)
    T_scale = point_num * step_T
    R_aver = np.zeros((point_num + 1,))
    PATHL = np.zeros((point_num + 1,))
    for i in range(1, point_num + 1):
        t = i * step_T
        R_aver[i], PATHL[i] = single_exper(n_=qubit_n, T_=t, N_=repeat_N)
        if PATHL[i] <= 1:
            Divide = np.sqrt((t + 1) * (n + np.log(t + 1)))
        else:
            Divide = np.sqrt((t + 1) * (n + np.log(t + 1)) * PATHL[i])
        R_aver[i] = R_aver[i] / Divide
    x_axis = np.arange(1 + T_scale)
    plt.plot(
        x_axis[step_T::step_T],
        R_aver[1:],
        lw=4,
        color=col,
        label="n=" + str(qubit_n) + ",P=" + str(int(PATHL[point_num] + 1)),
    )


def visual():
    if not os.path.exists("./Figure4"):
        os.makedirs("./Figure4")
    fig = plt.figure(num=1, figsize=(7, 5))
    plt.xlabel("time T")
    plt.ylabel(r"ratio term  $C=regret_T / \sqrt{T(n+logT)P}$")
    # repeat_N = 100 for subfigure(b)
    experiment(40, 5, 3, 10, "d", "forestgreen")
    experiment(40, 5, 2, 10, "o", "royalblue")
    experiment(40, 5, 1, 10, "^", "coral")
    plt.legend()
    plt.savefig("./Figure4/subfig_b.pdf")
    plt.show()


visual()
