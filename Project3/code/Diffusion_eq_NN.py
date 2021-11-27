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
        return self.eta0 # * (1.3 - epoch / self.epochs)


def main():
    x = np.linspace(0, 1, 11)
    t = np.linspace(0, 1.5, 101)
    x, t = np.meshgrid(x, t)
    X = np.concatenate((x.reshape(-1, 1), t.reshape(-1, 1)), axis=1)
    nodes = [20, 20]
    epochs = 50000
    eta0 = 0.0001

    Solver = Diffusion(X, epochs, nodes, eta0, load=False, name="Alice")
    solution = Solver.get_solution(*X.T)
    solution = solution.reshape(x.shape)

    fig = go.Figure(data=[go.Surface(x=x, y=t, z=solution)])
    fig.show()

if __name__ == "__main__":
    main()
