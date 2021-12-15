import autograd.numpy as np
from autograd import elementwise_grad as ele_grad
import plotly.graph_objects as go
from PDE_solver_NN import PDE_solver_NN_base


class Diffusion(PDE_solver_NN_base):
    def diff_eq(self, X, P):
        x, t = X.T
        x = x[:, None]
        t = t[:, None]
        lhs = ele_grad(ele_grad(self.trial, 0), 0)(x, t, P)
        rhs = ele_grad(self.trial, 1)(x, t, P)
        return lhs - rhs
        
    def trial(self, x, t, P):
        X = np.concatenate((x, t), axis=1)
        return np.sin(np.pi * x) * (1 + t * self(X, P))

    def eta(self, epoch):
        e = epoch / self.epochs
        p = 0.3 * e**4 - e**3 + 0.6 * e + 0.9
        return p * self.eta0


def main():
    dx = 1e-1
    dt = 0.5 * dx ** 2
    T = 1.5
    nx = int(1 / dx)
    nt = int(T / dt)
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, T, nt)
    x, t = np.meshgrid(x, t)
    
    X = np.concatenate((x.reshape(-1, 1), t.reshape(-1, 1)), axis=1)

    nodes = [[10,10,10,10], [10, 20, 20, 10], [50, 50], [20, 40, 20]]

    for node in nodes:
        Solver = Diffusion(
                    nodes=[2,] + node + [1,], 
                    activation="tanh",
                    alpha=0,
                    epochs=500, 
                    eta0=0.002,
                    lmb=0, 
                    gamma=0.95, 
                    load=False, 
                    name=None, 
                    seed=2021,
                    )
        print(Solver.nodes)
        Solver.train(X)
        print("\n"*2)

    solution = Solver.get_solution()  # trial function solution
    solution = solution.reshape(x.shape)

    fig = go.Figure(data=[go.Scatter(x=Solver.t[10:], y=Solver.history["cost"][10:], mode="lines")])
    fig.show()

    fig = go.Figure(data=[go.Surface(x=x, y=t, z=solution)])
    fig.show()

if __name__ == "__main__":
    main()

