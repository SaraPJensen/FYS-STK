import numpy as np
from diffusion_forward_euler import Diffusion_1D
from Diffusion_eq_NN import Diffusion
import plotly.graph_objects as go



def analytic(x, t):
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)

def main():
    T = 1.5
    x1 = 1
    dx = 1e-1

    Explicit = Diffusion_1D(x1, T, dx)
    x, t = Explicit.mesh()
    times = Explicit.t

    X = np.concatenate((x.reshape(-1, 1), t.reshape(-1, 1)), axis=1)

    NN = Diffusion(nodes=[2, 35, 20, 10, 1],
                   activation="abs",
                   alpha=-1,
                   epochs=5000,
                   eta0=0.00005,
                   lmb=0,
                   gamma=0.95,
                   seed=2021,
                   load=False,
                   name="Lalala",
                   )
    NN.train(X)
    # NN.X = X

    expl_sol = Explicit.solve()
    NN_sol = NN.get_solution().reshape(x.shape)
    ana_sol = analytic(x, t)

    print(ana_sol.shape)
    t1 = np.argmin(abs(times - T / 2))
    t2 = np.argmin(abs(times - T))

    fig = go.Figure(data=[go.Surface(x=x, y=t, z=expl_sol)])
    fig.show()
    fig = go.Figure(data=[go.Surface(x=x, y=t, z=NN_sol)])
    fig.show()
    fig = go.Figure(data=[go.Surface(x=x, y=t, z=ana_sol)])
    fig.show()

    # for time in [t1, t2]:
    # for time in [t2]:

    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter(x=x[time], y=expl_sol[time], mode="lines", name="Explicit solver"))
    #     fig.add_trace(go.Scatter(x=x[time], y=ana_sol[time], mode="lines", name="Analytic solution"))
    #     fig.add_trace(go.Scatter(x=x[time], y=NN_sol[time], mode="lines", name="Neural solver"))
    #     fig.show()



    



if __name__ == "__main__":
    main()
