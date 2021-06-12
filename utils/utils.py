import pandas as pd
import torch
import numpy as np


class MnistQs():
    def __init__(self, csv_path):
        self.df = pd.read_pickle(csv_path)
        self.u_CE = self.df["u_CE"]
        self.u_MFPS = self.df["u_MFPS"]
        self.u_MFLS = self.df["u_MFLS"]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return self.u_CE.iloc[idx], self.u_MFPS.iloc[idx], self.u_MFLS.iloc[idx]

    def get_u_CE(self):
        self.u_CE = torch.zeros(2000, 10)

        for idx, u in enumerate(self.df.u_CE):
            self.u_CE[idx, :] = torch.tensor(u)

        return self.u_CE

    def get_u_MFPS(self):
        self.u_MFPS = torch.zeros(2000, 10)

        for idx, u in enumerate(self.df.u_MFPS):
            self.u_MFPS[idx, :] = torch.tensor(u)

        return self.u_MFPS

    def get_u_MFLS(self):
        self.u_MFLS = torch.zeros(2000, 10)

        for idx, u in enumerate(self.df.u_MFLS):
            self.u_MFLS[idx, :] = torch.tensor(u)

        return self.u_MFLS


# Define support function used to convert label to one-hot encoded tensor
def convert_labels(labels, num_class):
    target = torch.zeros([len(labels), num_class], dtype=torch.float32)
    for i, l in enumerate(labels):
        target[i][l] = 1.0
    return target


def create_MNIST_mask(labels):
    labels1 = labels[labels <= 4].unique()
    labels2 = labels[labels >= 5].unique()
    M = np.zeros([len(labels.unique()), 2])
    M[:len(labels1), 0] = 1
    M[len(labels2):, 1] = 1
    return M


def get_PZ_matrices_csv(M, z1, z2, p1, p2):
    Z = np.zeros(M.shape)
    Z[:5, 0] = z1.cpu().data.numpy()
    Z[5:, 1] = z2.cpu().data.numpy()

    P = np.zeros(M.shape)
    P[:5, 0] = p1
    P[5:, 1] = p2

    return P, Z


def grad_j(dict_probs_t1, dict_probs_t2, u):
    grad_j = np.random.rand(10)
    for i, u_i in enumerate(u):
        dui = 0
        if i in dict_probs_t1.keys():
            dui = dui - dict_probs_t1[i]
            e = np.exp(u_i) / np.sum(np.exp(list(dict_probs_t1.values())))
            dui = dui + np.sum(np.array(list(dict_probs_t1.values())) * e)
        if i in dict_probs_t2.keys():
            dui = dui - dict_probs_t2[i]
            e = np.exp(u_i) / np.sum(np.exp(list(dict_probs_t2.values())))
            dui = dui + np.sum(np.array(list(dict_probs_t2.values())) * e)
        grad_j[i] = dui
    return grad_j


def ce_method1_csv(p1, p2):
    iters = 3000
    dict_probs_t1 = {idx: p1[idx] for idx in range(5)}
    dict_probs_t2 = {idx: p2[idx - 5] for idx in range(5, 10)}
    u = np.random.rand(10)

    for it in range(iters):
        u = u - 0.1 * grad_j(dict_probs_t1, dict_probs_t2, u)
    return u


# MATRIX FACTORIZATION IN LOGIT SPACE
def mf_logit_space(M, Z):
    lambd = 0.01
    L, N = M.shape

    v = np.ones(N)
    u = np.ones(L)
    c = np.ones(N)
    iters = 0

    # Initialize c
    for i in range(N):
        c[i] = np.sum(M[:, i] * Z[:, i]) / np.sum(M[:, i])

    # Run until convergence
    while iters < 3000:
        for j in range(L):
            # First for loop
            u[j] = np.sum(M[j, :] * (Z[j, :] - c) * v) / (lambd + np.sum(M[j, :] * np.power(v, 2)))

        for i in range(N):
            v[i] = np.sum(M[:, i] * (Z[:, i]) * u) / (lambd + np.sum(M[:, i] * np.power(u, 2)))
            v[i] = max(0, v[i])

            c[i] = np.sum(M[:, i] * (Z[:, i] - u * v[i])) / np.sum(M[:, i])

        iters += 1

    return u, v, c


# MATRIX FACTORIZATION IN PROBABILITY SPACE
def mf_prob_space(M, P):
    # Parameter initialization
    L, N = M.shape
    v = np.ones(N)
    u = np.ones(L)
    iters = 0

    # Run until convergence
    while iters < 3000:
        for j in range(L):
            # First for loop
            u[j] = np.sum(M[j, :] * P[j, :] * v) / np.sum(M[j, :] * np.power(v, 2))
            u[j] = max(0, u[j])

            u = u / np.sum(u)

        for i in range(N):
            v[i] = np.sum(M[:, i] * P[:, i] * u) / np.sum(M[:, i] * np.power(u, 2))
            v[i] = max(0, v[i])

        iters += 1

    return u, v
