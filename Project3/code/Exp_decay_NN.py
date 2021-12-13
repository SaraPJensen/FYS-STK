import autograd.numpy as np
from autograd import elementwise_grad as ele_grad
import matplotlib.pyplot as plt
from PDE_solver_NN import PDE_solver_NN_base


class Exp_Decay(PDE_solver_NN_base):
    def lhs(self, X, P):
        return ele_grad(self.trial, 0)(X, P)
    def rhs(self, X, P):
        return -2 * self.trial(X, P)

    def trial(self, X, P):
        return 10 + X * self(X, P)


def g_analytic(x, gamma = 2, g0 = 10):
    return g0*np.exp(-gamma*x)


def main():
    x = np.linspace(0, 1, 101 * 101).reshape(-1, 1)

    Solver = Exp_Decay(nodes=[1, 10, 10, 1],
                       activation="relu",
                       alpha=0,
                       epochs=600,
                       eta0=0.001,
                       lmb=0,
                       gamma=0.8,
                       )
    Solver.train(x)

    # Solver.trial(*x.T, x, Solver.P)
    solution = Solver.get_solution()
    plt.plot(x, g_analytic(x))
    plt.plot(x, solution)
    plt.show()

if __name__ == "__main__":
    main()
