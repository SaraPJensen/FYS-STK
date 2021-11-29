import autograd.numpy as np
from autograd import elementwise_grad as ele_grad
import plotly.graph_objects as go
from PDE_solver_NN import PDE_solver_NN_base



class Diffusion(PDE_solver_NN_base):
    def lhs(self, X, P):
        x, t = X.T
        return ele_grad(ele_grad(self.trial, 0), 0)(x, t, X, P)

    def rhs(self, X, P):
        x, t = X.T
        return ele_grad(self.trial, 1)(x, t, X, P)

    def trial(self, x, t, X, P):
        return np.sin(np.pi * x[:, None]) * (1 + t[:, None] * self(X, P))

    def eta(self, epoch):
        return self.eta0 * (1 - 0 * epoch / self.epochs)


def main():
    x = np.linspace(0, 1, 11)
    t = np.linspace(0, 3, 101)
    x, t = np.meshgrid(x, t)
    X = np.concatenate((x.reshape(-1, 1), t.reshape(-1, 1)), axis=1)
    nodes = [40, 20, 20, 20, 10]
    epochs = 1000
    eta0 = 0.0001

    Solver = Diffusion(X, epochs, nodes, eta0, load=False, name="Beatrice", seed=2021)
    solution = Solver.get_solution()  # trial function solution
    solution = solution.reshape(x.shape)
    raw_solution = Solver(X, Solver.P)  # the way the net looks like without trial function
    raw_solution = raw_solution.reshape(x.shape)

    fig = go.Figure(data=[go.Scatter(x=Solver.t[10:], y=Solver.history[10:], mode="lines")])
    fig.show()

    fig = go.Figure(data=[go.Surface(x=x, y=t, z=solution)])
    fig.show()

    fig = go.Figure(data=[go.Surface(x=x, y=t, z=raw_solution)])
    fig.show()

if __name__ == "__main__":
    main()

