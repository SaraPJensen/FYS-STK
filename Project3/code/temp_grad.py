from diffeq import PDE_solver_neural_network
from autograd import elementwise_grad as egrad
import autograd.numpy as np
import matplotlib.pyplot as plt



def trial_func(x, t, N, P=None):
    # sin(pi x) * (1 + t * N(x, t))
    return np.sin(np.pi * x) * (1 + t * N(X, P))

def diffeq(X, u, N, P):
    # x, t = X.T
    lhs = egrad(egrad(u, 0), 0)(X, N, P)  # 2nd derivative over x
    rhs = egrad(u, 1)(X, N, P)  # first derivative over t
    return lhs - rhs


if __name__ == "__main__":
    np.random.seed(2021)
    T = 1
    # dx = 1 / 100
    # dt = dx * dx * 0.5
    # Nx = int(1 // dx + 1)
    # Nt = int(T // dt + 1)
    Nx = 101
    Nt = 101
    x = np.linspace(0, 1, Nx)
    t = np.linspace(0, T, Nt)

    x, t = np.meshgrid(x, t)
    X = np.c_[x.reshape(-1, 1), t.reshape(-1, 1)]
    epochs = 360
    print(X.shape)
    Solver = PDE_solver_neural_network(X, diffeq, trial_func, epochs, [10, 10], 0.001)
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