import numpy as np
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
# Parameters
n = 3
rho_size = 2**n
T = 50
a = 1 - 1 / T
b = 1 / (T * rho_size)
I = np.eye(rho_size, dtype="complex_")
Zero = np.zeros((rho_size, rho_size), dtype="complex_")
L = 3

tole = 1e-4

# k_max = int(np.log(T)) + 2
k_max = 9
print(k_max)
# k_fac = 0.95
S_eta = np.zeros((k_max,))
for k in range(k_max):
    S_eta[k] = 1 - (k + 1) * 0.1
    # S_eta[k] = 3 / 2 ** (k+1)


"""
Tool Functions
"""


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


def subD_l(zt_x, bt):
    return 2 * (zt_x - bt)


def min_func_path(Eigen, mu):
    # Minimize function in projection
    Eigen1 = a * Eigen + b
    if np.min(Eigen1) <= 0:
        # print(np.min(Eigen1))
        return 10000000000
    Eigen2 = np.log(Eigen1)
    result = -a * np.sum(Eigen * np.log(mu)) + np.sum(Eigen1 * Eigen2)
    return result


def min_func_rftl(Eigen, mu, eta):
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


cons = ({"type": "ineq", "fun": constr_norm}, {"type": "ineq", "fun": constr_non_neg})
bonds = op.Bounds(np.array([0] * rho_size), np.array([1] * rho_size))


def RFTL(rho_list, E_list, eta):
    x_pred = I / rho_size
    Eigen_last = np.random.random((rho_size,)) / rho_size + tole
    Regret = 0
    Regret_list = []
    sigma_Nablas = np.zeros((rho_size, rho_size), dtype="complex_")
    # eta = 0.5
    # eta = np.sqrt((np.log(2)*n)/(2 * T * (L ** 2)))
    for temp_t in range(T):
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
            min_func_rftl,
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
    return Regret_list


def OMD(x_last, Eigen_last, eta, Et, bt_rho):
    # instance of expert in Algorithm2(OMD)
    zt_x = np.trace(Et @ x_last)
    subD_l = grad(l, holomorphic=True)
    Nabla_t = subD_l(zt_x, bt_rho) * Et
    y_new = equation(nabla_R(x_last) - eta * Nabla_t)
    mu, vec = np.linalg.eig(y_new)
    mu = mu.real
    Eigen_solve = op.minimize(min_func_path, Eigen_last, mu, constraints=cons, tol=tole)
    if Eigen_solve.success == False:
        print(Eigen_solve)
    Eigen_new = Eigen_solve.x
    x_new_revtile = compose_Matrix(Eigen_new, vec)
    x_new = x_tilde(x_new_revtile)
    return x_new, Eigen_new


def DYMR(rho_list, E_list):
    alpha = 1
    # Initialization
    dynamic_Regret = 0
    Regret_list = []
    W = np.ones((k_max,)) / k_max
    X_k = np.array([I / rho_size] * k_max, dtype="complex_")
    Eigen_k = np.zeros((k_max, rho_size))
    x_pred = I / rho_size + 0.0j
    for t in range(T):
        Et = E_list[t]
        zt_x = np.trace(Et @ x_pred)
        bt_rho = np.trace(Et @ x_tilde(rho_list[t]))
        dynamic_Regret += l(zt_x, bt_rho).real
        Regret_list.append(dynamic_Regret)
        loss = np.zeros((k_max,))
        for k in range(k_max):
            loss[k] = l(np.trace(Et @ X_k[k]), bt_rho).real
        loss = np.exp(-alpha * loss)
        W = W * loss
        for k in range(k_max):
            X_k[k], Eigen_k[k] = OMD(X_k[k], Eigen_k[k], S_eta[k], Et, bt_rho)
        # print("newX_k=", X_k)
        # print("Eigen_k", Eigen_k)
        x_pred = np.einsum("k,kij->ij", W, X_k) / np.sum(W)
    return Regret_list


def single_exper():
    rho_list = []
    E_list = []
    E_list.append(generate_E())
    rho = qt.rand_dm(rho_size).full()
    rho_list.append(rho)
    H = qt.rand_herm(rho_size).full()
    Pa = 0
    P_list = []
    P_list.append(Pa)
    for t in range(1, T + 1):
        E_list.append(generate_E())
        Ht = (t + 1) * H
        # Ht = (t + 1) / 50 * H
        rho = evolve_rho(rho, Ht)
        rho_list.append(rho)
        Pa += np.linalg.norm(rho - rho_list[t - 1], ord="nuc")
        P_list.append(Pa)

    Regret_dymr = DYMR(rho_list, E_list)
    Regret_rftl = []
    for k in range(k_max):
        Regret_rftl.append(RFTL(rho_list, E_list, S_eta[k]))

    return Regret_dymr, Regret_rftl, P_list


def experiment(N):
    if not os.path.exists("./Figure2"):
        os.makedirs("./Figure2")
    Rd_aver = [0] * T
    Rd_max = [-1] * T
    Rr_max = [[-1] * T for k in range(k_max)]
    for temp in range(N):
        R_dymr, R_rftl, Path = single_exper()
        print("path=", Path)
        # print("RFTL regret=", R_rftl)
        for t in range(T):
            Rd_max[t] = max(Rd_max[t], R_dymr[t])
            Rd_aver[t] += R_dymr[t] / N
        for k in range(k_max):
            for t in range(T):
                Rr_max[k][t] = max(Rr_max[k][t], R_rftl[k][t])
    x_axis = np.arange(T + 1)
    # tab10 = mcolor.TABLEAU_COLORS
    # color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.plot(x_axis[1:], Rd_max, label="DOMD", zorder=10)
    for k in range(k_max):
        plt.plot(x_axis[1:], Rr_max[k], label=round(S_eta[k], 1))
    # plt.plot(x_axis[1:], Rr_max[0], label=S_eta[0])
    # for k in range(k_max):
    #     plt.plot(x_axis[1:], Rr_max[k], label="1/"+str(2 ** k))

    plt.legend()
    plt.xlabel("time T")
    plt.ylabel(r"regret $R$")
    plt.savefig(f"./Figure2/alg1_n{n}.pdf")
    plt.show()


experiment(10)
# single_exper()
