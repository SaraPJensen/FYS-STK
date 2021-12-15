import numpy as np
from FE_diffusion_eq import Diffusion_1D
from NN_diffusion_eq import Diffusion
import plotly.graph_objects as go


def analytic(x, t):
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)


def main():
    T = 1.5
    x0 = 1
    dx1 = 1e-1
    dx2 = 1e-2

    Euler1 = Diffusion_1D(x0, T, dx1)
    Euler2 = Diffusion_1D(x0, T, dx2)
    x1, t1 = Euler1.mesh()
    x2, t2 = Euler2.mesh()
    X = np.concatenate((x1.reshape(-1, 1), t1.reshape(-1, 1)), axis=1)

    times1 = Euler1.t
    times2 = Euler2.t

    epoch = 5000

    NN = Diffusion(nodes=[2, 10, 20, 20, 10, 1],
                   activation="tanh",
                   alpha=0,
                   epochs=epoch,
                   eta0=0.002,
                   lmb=0,
                   gamma=0.95,
                   seed=2021,
                   load=False,
                   name=f"Deeeeeep_e{epoch}",
                   )
    NN.train(X)
    # NN.X = X

    expl_sol1 = Euler1.solve()
    expl_sol2 = Euler2.solve()
    NN_sol = NN.get_solution().reshape(x1.shape)
    ana_sol = analytic(x2, t2)

    ta = np.argmin(abs(times1 - 0.2))
    tb = np.argmin(abs(times1 - T))
    tc = np.argmin(abs(times2 - 0.2))
    td = np.argmin(abs(times2 - T))

    f = go.Figure(go.Surface(x=x1, y=t1, z=NN_sol))
    f.show()

    for time1, time2 in zip([ta, tb], [tc, td]):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x1[time1], y=expl_sol1[time1],
                      mode="lines", name="Explicit solver", line=dict(width=5)))
        fig.add_trace(go.Scatter(x=x2[time2], y=expl_sol2[time2], mode="lines",
                      name="Explicit solver with smaller step size", line=dict(width=5)))
        fig.add_trace(go.Scatter(x=x2[time2], y=ana_sol[time2],
                      mode="lines", name="Analytic solution", line=dict(width=5)))
        fig.add_trace(go.Scatter(
            x=x1[time1], y=NN_sol[time1], mode="lines", name="Neural solver", line=dict(width=5)))
        fig.show()


if __name__ == "__main__":
    main()
