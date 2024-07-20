import numpy as np
import math
import qutip as qt
import scipy.optimize as op
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
from xmlrpc.client import MAXINT
import torch
from jax import grad
import os


"""
All computation based on numpy. Only convert to qutip for utilization of functions
"""


"""
Tool Functions
"""


def initialize(n_, T_):
    global n, T, rho_size, I, Zero, eta, Indic, S, COL_NUM, cons, bonds, L, tole, k_max, S_eta

    L = 3
    tole = 1e-5

    n, T = n_, T_
    rho_size = 2**n
    I = np.eye(rho_size, dtype="complex_")
    Zero = np.zeros((rho_size, rho_size), dtype="complex_")

    k_max = 9
    S_eta = np.zeros((k_max,))
    for k in range(k_max):
        S_eta[k] = 1 - (k + 1) * 0.1

    COL_NUM = 2 ** (math.floor(np.log2(T)) + 1)
    Indic = np.zeros((T + 1, T + 1, COL_NUM))
    for t in range(1, T + 1):
        for i in range(1, t + 1):
            for j in range(t, COL_NUM):
                Indic[t][i][j] = 1
    S = np.ones((T + 1, T + 1, COL_NUM))
    for t in range(1, T + 1):
        for i in range(1, t + 1):
            for j in range(t, COL_NUM):
                S[t][i][j] += np.sum(Indic[:t, i, j])
    cons = (
        {"type": "ineq", "fun": constr_norm},
        {"type": "ineq", "fun": constr_non_neg},
    )
    bonds = op.Bounds(np.array([0] * rho_size), np.array([1] * rho_size))


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
    # Gram_Schmidt Orthogonalization
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


def subD_l(zt_x, bt):
    return 2 * (zt_x - bt)


def Active(t):
    # Find out all awake experts at t
    # Input: time t
    # Output: List of time intervals (in tuple) covering t
    J_list = []
    J_num = math.floor(np.log2(t)) + 1
    for k in range(J_num):
        step = 2**k
        i = math.floor(t / step)
        J_list.append((i * step, (i + 1) * step - 1))
    return J_list


def beta(t, i, j, g):
    sum = np.sum(Indic[:t, i, j] * g[:t, i, j])
    return sum / S[t, i, j]


def min_func(Eigen, mu, eta):
    # minimize function in projection
    if np.min(Eigen) <= 0:
        # print(np.min(Eigen))
        return 10000000000
    Eigen2 = np.log(Eigen)
    result = eta * np.sum(Eigen * mu) + np.sum(Eigen * Eigen2)
    return result * 100


def constr_norm(Eigen):
    return tole - np.abs(np.linalg.norm(Eigen, ord=1) - 1)


def constr_non_neg(Eigen):
    return np.min(Eigen) + tole


def RFTL(start_time, t, rho_list, E_list, eta):
    # Black-box algorithm RFTL in Algorithm3
    # Input: when expert becomes awake and when goes to sleep. The accurate value for state rho and POVM operator E
    # Output: prediction and regret list of this expert
    x_pred = I / rho_size
    Eigen_last = np.random.random((rho_size,)) / rho_size + tole
    Regret = 0
    Regret_list = []
    sigma_Nablas = np.zeros((rho_size, rho_size), dtype="complex_")
    for temp_t in range(start_time, t + 1):
        rho = rho_list[temp_t]
        E = E_list[temp_t]
        # Compute loss
        zt = np.trace(E @ x_pred)
        bt = np.trace(E @ rho)
        Regret += l(zt, bt).real
        Regret_list.append(Regret)
        Nabla_t = subD_l(zt, bt) * E
        sigma_Nablas += Nabla_t
        mu, vec = np.linalg.eig(sigma_Nablas)
        mu = mu.real
        Eigen_solve = op.minimize(
            min_func,
            Eigen_last,
            (mu, eta),
            constraints=cons,
            tol=tole,
            options={"maxiter": 500},
        )
        if Eigen_solve.success == False:
            print(Eigen_solve)
        Eigen_new = Eigen_solve.x
        x_pred = compose_Matrix(Eigen_new, vec)
    return x_pred, Regret_list


def CBCE(rho_list, E_list):
    # Meta algorithm CBCE in Algorithm3
    # Input: the accurate value for state rho and POVM operator E
    # Output:
    # Initialization
    Adaptive_regret = 0
    # w[t][i][j] for w_t on J=[i,j]
    w = np.zeros((T + 1, T + 1, COL_NUM))
    g = np.zeros((T + 1, T + 1, COL_NUM))
    p = np.zeros((T + 1, T + 1, COL_NUM))
    Regret_list = []
    for t in range(1, T + 1):
        J_list = Active(t)
        J_num = len(J_list)
        pi = np.zeros((J_num,))
        for k in range(J_num):
            J1 = J_list[k][0]
            pi[k] = 1 / ((J1**2) * (1 + np.floor(np.log2(J1))))
        pi = pi / np.sum(pi)
        x_expert = np.zeros((J_num, rho_size, rho_size), dtype="complex_")
        x_t = np.zeros((rho_size, rho_size), dtype="complex_")
        for k in range(J_num):
            i = J_list[k][0]
            j = J_list[k][1]
            w[t, i, j] = beta(t, i, j, g) * (
                1 + np.sum(Indic[1:t, i, j] * g[1:t, i, j] * w[1:t, i, j])
            )
            p[t, i, j] = pi[k] * Indic[t, i, j] * max(w[t, i, j], 0)
            eta = np.sqrt((np.log(2) * n) / (2 * (t + 1 - i) * (L**2)))
            x_expert[k], useless_regret = RFTL(i, t, rho_list, E_list, eta)
        p_norm = np.sum(p[t, 1:, 1:])
        if p_norm > 0:
            p[t, 1:, 1:] = p[t, 1:, 1:] / p_norm
        else:
            for k in range(J_num):
                i = J_list[k][0]
                j = J_list[k][1]
                p[t, i, j] = pi[k]
        for k in range(J_num):
            i = J_list[k][0]
            j = J_list[k][1]
            x_t += x_expert[k] * p[t, i, j]
        zt_x = np.trace(E_list[t] @ x_t)
        bt_rho = np.trace(E_list[t] @ rho_list[t])
        ft = l(zt_x, bt_rho).real
        Adaptive_regret += ft
        Regret_list.append(Adaptive_regret)
        print(Adaptive_regret)
        for k in range(J_num):
            i = J_list[k][0]
            j = J_list[k][1]
            zt_xk = np.trace(E_list[t] @ x_expert[k])
            if w[t, i, j] > 0:
                g[t, i, j] = ft - l(zt_xk, bt_rho).real
            else:
                g[t, i, j] = max(ft - l(zt_xk, bt_rho).real, 0)
    return Regret_list


def single_exper(K):
    Shift_num = np.sort(np.random.choice(np.arange(2, T + 1), size=(K,), replace=False))
    print("Shift_num=", Shift_num)
    rho_list = []
    E_list = []
    E_list.append(generate_E())
    rho = qt.rand_dm(rho_size).full()
    rho_list.append(rho)
    H = qt.rand_herm(rho_size).full()
    Pa = 0
    P_list = []
    P_list.append(Pa)
    iter = 0
    for t in range(1, T + 1):
        E_list.append(generate_E())
        if iter < K and t == Shift_num[iter]:
            iter += 1
            rho = qt.rand_dm(rho_size).full()
        else:
            Ht = (t + 1) / 50 * H
            rho = evolve_rho(rho, Ht)
        rho_list.append(rho)
        Pa += np.linalg.norm(rho - rho_list[t - 1], ord="nuc")
        P_list.append(Pa)

    Regret_cbce = CBCE(rho_list, E_list)
    # useless_x, Regret_rftl = RFTL(1, T, rho_list, E_list)
    Regret_rftl = []
    for k in range(k_max):
        useless_x, temp_Regret_rftl = RFTL(1, T, rho_list, E_list, S_eta[k])
        Regret_rftl.append(temp_Regret_rftl)

    return Regret_cbce, Regret_rftl, P_list


def experiment(n_, T_, K, N):
    initialize(n_, T_)
    Rc_aver = [0] * T
    Rc_max = [-1] * T
    # Rr_aver = [0] * T
    Rr_max = [[-1] * T for k in range(k_max)]
    for temp in range(N):
        R_cbce, R_rftl, Path = single_exper(K)
        print("path=", Path)
        # print("RFTL=", R_rftl)
        # print("Rrmax=", Rr_max)
        for t in range(T):
            Rc_max[t] = max(Rc_max[t], R_cbce[t])
            Rc_aver[t] += R_cbce[t] / N
        for k in range(k_max):
            for t in range(T):
                Rr_max[k][t] = max(Rr_max[k][t], R_rftl[k][t])
    x_axis = np.arange(T + 1)
    plt.plot(x_axis[1:], Rc_max, label="CBCE")
    # plt.plot(x_axis[1:], Rr_max, label="RFTL")
    # plt.plot(x_axis[1:], Rr_max[0], label=S_eta[0])
    for k in range(k_max):
        plt.plot(x_axis[1:], Rr_max[k], label=round(S_eta[k], 1))
    plt.legend()
    plt.xlabel("time T")
    plt.ylabel(r"regret $R$")
    plt.savefig(f"./Figure2/alg3_n{n_}_k{K}.pdf")
    plt.show()


experiment(3, 50, 4, 10)
# single_exper()
