from autograd.differential_operators import elementwise_grad
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import elementwise_grad as egrad, grad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import sys
import os
import time


class PDE_solver_neural_network:
    def __init__(self, X, diff_eq, trial, epochs, nodes, eta, activation="relu"):
        self.X = X
        self.diffeq = diff_eq
        self.trial = trial

        self.nodes = np.array([self.X.shape[1], *nodes, 1])
        self.initialize_weights(activation)

        self.epochs = epochs
        self.eta = eta
        print(f"Initial cost: {self.cost_func(self.P)}")
        self.train()
        print(f"Final cost: {self.cost_func(self.P)}")

    def initialize_weights(self, activation):
        P = [None] * (len(self.nodes) - 1)
        for i in range(1, len(self.nodes)):
            n = self.nodes[i - 1]
            m = self.nodes[i]
            P[i - 1] = npr.normal(scale=np.sqrt(2 / n), size=(n + 1, m))  #  +1 for bias
            # P[i - 1] = npr.normal(size=(n + 1, m))  #  +1 for bias
            P[i - 1][-1, :] = 0.01  # bias initialisering
        self.P = P

    def feed_forward(self, x=None, P=None):
        if x is None:
            prev = self.X
        else:
            prev = x
        # prev = np.concatenate((X, np.ones()))
        for l in range(len(self.nodes) - 1):
            # prev = np.c_[prev, np.ones((prev.shape[0] ,1))]
            prev = np.concatenate((prev, np.ones((prev.shape[0] ,1))), axis=1)
            # print(prev.shape, P[l].shape)
            z = prev @ P[l]
            a = self.sigmoid(z)
            prev = a

        self.output = z
        # print(self.output, self.output.shape)
        # input()

    def train(self):
        grad_cost_func = grad(self.cost_func, 0)
        pbar = tqdm(range(self.epochs), desc=f"{self.cost_func(self.P):.4f}")
        for i in pbar:
            gradients = grad_cost_func(self.P)
            # print(gradients)
            # input()
            for l in range(len(self.nodes) - 1):
                # self.P[l] = self.optimizer(self.P[l], gradients[l])
                self.P[l] = self.P[l] - self.eta * gradients[l]
            pbar.set_description(f"{self.cost_func(self.P):.4f}")

    def ReLU(self, x):
        return np.where(x > 0, x, 0.01 * x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def cost_func(self, P):
        F = self.diffeq(self.X, self.trial, self, P)
        return np.sum(F ** 2) / len(F)

    def __call__(self, x=None, P=None):
        if P is None:
            P = self.P
        if isinstance(x, list):
            print(x)
        self.feed_forward(x, P)
        return self.output

    # def cost_func(self, P):

    #     g_t = self.trial(self.X, self, P)
    #     dg_t = egrad(self.trial, 0)(self.X, self, P)
    #     func = self.diffeq(*self.X.T, g_t)

    #     err = (dg_t - func) ** 2
    #     return np.sum(err) / np.size(err)

def trial_func(x, N=None, P=None):
    return 10 + x * N(x, P)

# def g(x, u):
    # return -2 * u

# def diff(x, u, N):
#     lhs = egrad(u, 0)(x, N)
#     rhs = -2 * u(x, N)

def g(x, u, N, P):
    # gt = u(x, N, P)
    lhs = egrad(u, 0)(x, N, P)
    rhs = -2 * u(x, N, P)
    return lhs - rhs

def g_analytic(x, gamma = 2, g0 = 10):
    return g0*np.exp(-gamma*x)

def exp_decay():
    np.random.seed(15)
    x = np.linspace(0, 1, 10).reshape(-1, 1)
    nodes = [10,10]
    epochs = 1000
    eta = 0.001


    Solver = PDE_solver_neural_network(x, g, trial_func, epochs, nodes, eta)
    solution = trial_func(x, Solver)
    plt.plot(x, g_analytic(x))
    plt.plot(x, solution)
    plt.show()

if __name__ == "__main__":
    exp_decay()