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

    # def eta(self, epoch):
    #     if epoch > 50:
    #         r = self.history["cost"][epoch - 3] / self.history["cost"][epoch - 1]
    #         self.history["r"][epoch] = r
    #         if r > 1:
    #             eta = self.eta0 * r ** (30 * (1 - epoch / self.epochs))
    #             self.history["eta"][epoch] = eta
    #             return eta

    #         self.history["eta"][epoch] = self.eta0
    #         return self.eta0
    #     self.history["eta"][epoch] = self.eta0
        # return self.eta0

    def eta(self, epoch):
        eta = self.eta0
        self.history["eta"][epoch] = eta
        return eta



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

    # Solver = Diffusion(
    #                    nodes=[2, 35, 20, 10, 1], 
    #                    activation="abs",
    #                    alpha=-1,
    #                    epochs=500, 
    #                    eta0=0.00008,
    #                    lmb=0, 
    #                    gamma=0.95, 
    #                    load=False, 
    #                    name=None, 
    #                    seed=2021,
    #                    )
    Solver = Diffusion(
                       nodes=[2, 30, 20, 1], 
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
    Solver.train(X)

    solution = Solver.get_solution()  # trial function solution
    solution = solution.reshape(x.shape)

    fig = go.Figure(data=[go.Scatter(x=Solver.t[10:], y=Solver.history["cost"][10:], mode="lines")])
    fig.show()
    fig = go.Figure(data=[go.Scatter(x=Solver.t[10:], y=Solver.history["eta"][10:], mode="lines")])
    fig.show()

    # fig = go.Figure(data=[go.Surface(x=x, y=t, z=solution)])
    # fig.show()

    # fig = go.Figure(data=[go.Surface(x=x, y=t * T, z=raw_solution)])
    # fig.show()

if __name__ == "__main__":
    main()

