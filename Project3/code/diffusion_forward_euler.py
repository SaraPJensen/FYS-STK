import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm


class Diffusion_1D:
    def __init__(self, x1, t1, dx, r=0.5):

        self.r = r
        self.dx = dx
        self.dt = self.r * self.dx ** 2


        self.nt = int(t1 / self.dt)
        self.nx = int(x1 / self.dx)

        self.x = np.linspace(0, x1, self.nx)
        self.t = np.linspace(0, t1, self.nt)

        self.u = np.zeros((self.nt, self.nx))
        self.u[0] = np.sin(np.pi * self.x)  # initial condition

    def solve(self):
        pbar = tqdm(range(1, self.nt))
        for i in pbar:
            self.advance(i - 1)
        return self.u

    def advance(self, i):
        self.u[i + 1, 1:-1] = self.r * (self.u[i , 2:] - 2 * self.u[i, 1:-1] + self.u[i, :-2]) + self.u[i, 1:-1]

    def mesh(self):
        return np.meshgrid(self.x, self.t)


def main():
    T = 1
    X = 1
    dx = 1e-2

    Solver = Diffusion_1D(X, T, dx)
    u = Solver.solve()
    x, t = Solver.mesh()

    fig = go.Figure(data=[go.Surface(x=x, y=t, z=u)])
    fig.show()

if __name__ == "__main__":
    main()
