import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import elementwise_grad as egrad, grad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import sys
import os
import time


class pde_solver_neural_network:
    def __init__(self, X, diff_eq, trial, epochs, nodes, eta, activation="relu"):
        self.X = X
        self.diffeq = diff_eq
        self.trial = trial

        self.nodes = np.array([self.X.shape[1], *nodes, 1])
        self.initialize_weights(activation)

        self.epochs = epochs
        self.eta = eta

        self.train()

    def initialize_weights(self, activation):
        P = [None] * (len(self.nodes) - 1)
        for i in range(1, len(self.nodes)):
            n = self.nodes[i - 1]
            m = self.nodes[i]
            P[i - 1] = npr.normal(scale=np.sqrt(2 / n), size=(n + 1, m))  #  +1 for bias
            P[i - 1][-1, :] = 0.01  # bias initialisering
        self.P = P
        print(P[0].shape)

    def feed_forward(self, P):
        prev = self.X
        # prev = np.concatenate((X, np.ones()))
        for l in range(len(self.nodes) - 1):
            # prev = np.c_[prev, np.ones((prev.shape[0] ,1))]
            prev = np.concatenate((prev, np.ones((prev.shape[0] ,1))), axis=1)

            z = prev @ P[l]
            a = self.ReLU(z)
            prev = a

        self.output = z

    def train(self):
        grad_cost_func = egrad(self.cost_func)
        pbar = tqdm(range(self.epochs))
        for i in pbar:
            gradients = grad_cost_func(self.P)
            # print(len(gradients))
            for l in range(len(self.nodes) - 1):
                # self.P[l] = self.optimizer(self.P[l], gradients[l])
                self.P[l] = self.P[l] - self.eta * gradients[l]

    def ReLU(self, x):
        return np.where(x > 0, x, 0)

    def cost_func(self, P):
        F = self.diffeq(*self.X.T, self.trial, self(P))
        return F ** 2 / len(F)

    def __call__(self, P=None):
        if P is None:
            P = self.P
        self.feed_forward(P)
        return self.output[-1]


def trial_func(x, t, N):
    # sin(pi x) * (1 + t * N(x, t))
    return np.sin(np.pi * x) * (1 + t * N)

def diffeq(x, t, u, N):
    lhs = egrad(egrad(u, 0), 0)(x, t, N)  # 2nd derivative over x
    rhs = egrad(u, 1)(x, t, N)  # first derivative over t
    return lhs - rhs


if __name__ == "__main__":
    np.random.seed(2021)
    T = 1
    # dx = 1 / 100
    # dt = dx * dx * 0.5
    # Nx = int(1 // dx + 1)
    # Nt = int(T // dt + 1)
    Nx = 101
    Nt = 10001
    x = np.linspace(0, 1, Nx)
    t = np.linspace(0, T, Nt)

    x, t = np.meshgrid(x, t)
    X = np.c_[x.reshape(-1, 1), t.reshape(-1, 1)]
    epochs = 360
    print(X.shape)
    Solver = pde_solver_neural_network(X, diffeq, trial_func, epochs, [10, 10], 0.001)
    solution = trial_func(x, t, Solver())

    print(t.shape, x.shape, solution.shape)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.plot_surface(x, t, solution, antialiased=False)
    plt.show()

    # x = np.linspace(0, 1, 1001)
    # u = trial_func(x, 0, 0)
    # plt.plot(x, u)
    # plt.show()