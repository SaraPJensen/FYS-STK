import autograd.numpy as np
from autograd import elementwise_grad as ele_grad
import matplotlib.pyplot as plt
from PDE_solver_NN import PDE_solver_NN_base


class Exp_Decay(PDE_solver_NN_base):
    def lhs(self, X, P):
        return ele_grad(self.trial, 0)(X, P)
    def rhs(self, X, P):
        return -2 * self.trial(X, P)

    def trial(self, x, P):
        return 10 + x * self(x, P)


def g_analytic(x, gamma = 2, g0 = 10):
    return g0*np.exp(-gamma*x)


def main():
    x = np.linspace(0, 1, 101 * 101).reshape(-1, 1)
    nodes = [10, 10]
    epochs = 600
    eta = 0.001

    Solver = Exp_Decay(x, epochs, nodes, eta)
    solution = Solver.get_solution(x)
    plt.plot(x, g_analytic(x))
    plt.plot(x, solution)
    plt.show()

if __name__ == "__main__":
    main()
