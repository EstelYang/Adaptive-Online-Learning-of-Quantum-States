from copy import deepcopy
import math
import numpy as np
import qutip as qt
import scipy.optimize as op
import matplotlib.pyplot as plt
import torch
import os


"""
Initialize Parameters
"""
# Lipschitz coefficient
L = 3
# accuracy tolerance
tole = 1e-4


"""
Tool Functions
"""


def initialize(n_, T_):
    global n, T, rho_size, I, Zero, eta, Indic, S, COL_NUM, cons, bonds
    n, T = n_, T_
    rho_size = 2**n
    I = np.eye(rho_size)
    Zero = np.zeros((rho_size, rho_size)) + 0.0j
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


def l(zt, bt):
    # Loss function
    # Input: zt_x= trace(Et xt), bt_rho=trace(Et rho_t)
    # Output: (complex) L2 loss
    # P.S. The imagine component of trace is not exactly 0.0j due to computational accuracy limit.
    result = (bt - zt) ** 2 + 0.0j
    return result


def subD_l(zt_x, bt):
    # Subdifference for l
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


"""
minimize function and constraints & bounds
"""


def min_func(Eigen, mu, eta):
    # minimize function in projection
    if np.min(Eigen) <= 0:
        # print(np.min(Eigen))
        return 10000000000
    Eigen2 = np.log(Eigen)
    result = eta * np.sum(Eigen * mu) + np.sum(Eigen * Eigen2)
    return result


def constr_norm(Eigen):
    # 1-norm = 1
    return tole - np.abs(np.linalg.norm(Eigen, ord=1) - 1)


def constr_non_neg(Eigen):
    return np.min(Eigen)


def RFTL(start_time, t, rho_list, E_list):
    # Black-box algorithm RFTL in Algorithm3
    # Input: when expert becomes awake and when goes to sleep. The accurate value for state rho and POVM operator E
    # Output: prediction and regret list of this expert
    x_pred = I / rho_size
    Eigen_last = np.random.random((rho_size,)) / rho_size + tole
    Regret = 0
    Regret_list = []
    sigma_Nablas = np.zeros((rho_size, rho_size), dtype="complex_")
    eta = np.sqrt((np.log(2) * n) / (2 * (t + 1 - start_time) * (L**2)))
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
            min_func, Eigen_last, (mu, eta), constraints=cons, tol=tole
        )
        if Eigen_solve.success == False:
            print(Eigen_solve)
        Eigen_new = Eigen_solve.x
        x_pred = compose_Matrix(Eigen_new, vec)
    return x_pred, Regret_list


def CBCE(rho_list, E_list, QAR):
    # Meta algorithm CBCE in Algorithm3
    # Input: the accurate value for state rho and POVM operator E
    # Output:
    # Initialization
    Adaptive_regret = 0
    w = np.zeros((T + 1, T + 1, COL_NUM))
    g = np.zeros((T + 1, T + 1, COL_NUM))
    p = np.zeros((T + 1, T + 1, COL_NUM))
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
            x_expert[k], useless_regret = RFTL(i, t, rho_list, E_list)
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
        QAR[t] = Adaptive_regret
        print(Adaptive_regret)
        for k in range(J_num):
            i = J_list[k][0]
            j = J_list[k][1]
            zt_xk = np.trace(E_list[t] @ x_expert[k])
            if w[t, i, j] > 0:
                g[t, i, j] = ft - l(zt_xk, bt_rho).real
            else:
                g[t, i, j] = max(ft - l(zt_xk, bt_rho).real, 0)


def single_exper(K, QAR, ratio_C):
    # generate K timestamp when rho shifts
    Shift_num = np.sort(np.random.choice(np.arange(2, T + 1), size=(K,), replace=False))
    print("Shift_num=", Shift_num)
    rho_list = []
    E_list = []
    E_list.append(generate_E())
    rho = qt.rand_dm(rho_size).full()
    rho_list.append(rho)
    iter = 0
    for t in range(1, T + 1):
        E_list.append(generate_E())
        if iter < K and t == Shift_num[iter]:
            iter += 1
            rho = qt.rand_dm(rho_size).full()
        rho_list.append(rho)
    # run Algorithm 3
    CBCE(rho_list, E_list, QAR)
    for k in range(K):
        cur_T = Shift_num[k] - 1
        ratio_C[k] = max(
            ratio_C[k],
            QAR[cur_T] / np.sqrt((k + 1) * n * cur_T * np.log(cur_T + 1, dtype=float)),
        )


# code for subfigure(a)
def experiment_average(n_, T_, k, N, col, mark):
    initialize(n_, T_)
    QAR = np.zeros((T + 1,))
    ratio_C = np.zeros((k,))
    for j in range(N):
        single_exper(k, QAR, ratio_C)
    for t in range(1, T + 1):
        QAR[t] = QAR[t] / t
    x_axis = np.arange(1, T + 1, 1)
    plt.plot(
        x_axis, QAR[1:], color=col, marker=mark, label="k=" + str(k) + ", n=" + str(n_)
    )


def visual_average(Tv, Nv):
    if not os.path.exists("./Figure3"):
        os.makedirs("./Figure3")
    fig = plt.figure(num=1, figsize=(7, 5))
    plt.xlabel("time T")
    plt.ylabel(r"average regret  $R_{aver}$")
    experiment_average(n_=2, T_=Tv, k=9, N=Nv, col="royalblue", mark="o")
    experiment_average(n_=3, T_=Tv, k=4, N=Nv, col="coral", mark="^")
    experiment_average(n_=4, T_=Tv, k=1, N=Nv, col="forestgreen", mark="d")
    plt.legend()
    plt.savefig("./Figure3/subfig_a.pdf")
    plt.show()


# code for subfigure(b)
def experiment_ratio(n_, T_, k, N, col):
    initialize(n_, T_)
    QAR = np.zeros((T + 1,))
    ratio_C = np.zeros((k,))
    for j in range(N):
        single_exper(k, QAR, ratio_C)
    x_axis = np.arange(1, k + 1, 1)
    plt.bar(x_axis, ratio_C, color=col, tick_label=x_axis)
    print(ratio_C)
    # plt.scatter(x_axis, QAR[1:], color=col, label='k='+str(k)+',N='+str(N))


def visual_ratio():
    if not os.path.exists("./Figure3"):
        os.makedirs("./Figure3")
    fig = plt.figure(num=1, figsize=(7, 5))
    plt.xlabel("Shift interval k")
    plt.ylabel(r"ratio term  $C=regret_T / \sqrt{knTlog(T)}$")
    # n = 2, T = 200, k = 10, N = 50 for subfigure(b)
    experiment_ratio(n_=2, T_=200, k=10, N=5, col="royalblue")
    # plt.legend()
    plt.savefig("./Figure3/subfig_b.pdf")
    plt.show()


# T = 50, N = 100 for subfigure(a)
# visual_average(50, 5)

# subfigure(b)
visual_ratio()
