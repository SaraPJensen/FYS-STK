import autograd.numpy as np
import plotly.graph_objects as go
from Diffusion_eq_NN import  Diffusion

class Diffusion(Diffusion):
    def eta(self, epoch):
        r = epoch / self.epochs
        eta = self.ls(r) * self.eta0
        self.history["eta"][epoch] = eta
        return eta
    
const = lambda t: 1
lin_dec = lambda t: 1 - t
parabola = lambda t: 0.3 * t**4 - t**3 + 0.6 * t + 0.9
cos_up = lambda t: 2 - np.cos(t)
cos_down = lambda t: np.cos(t)

funcs = [const, lin_dec, parabola, cos_up, cos_down]
names = ["Constant", "Linear decrease", "Parabola", "Increasing cos", "Decreasing cos"]

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
    epochs = 5000

    ls = go.Figure()
    cost = go.Figure()
    costs = []

    for f, name in zip(funcs, names):
        Solver = Diffusion(
                        nodes=[2, 10, 20, 20, 10, 1], 
                        activation="tanh",
                        alpha=0,
                        epochs=epochs, 
                        eta0=0.002,
                        lmb=0, 
                        gamma=0.95, 
                        load=False, 
                        name=None, 
                        seed=2021,
                        )
        Solver.ls = f
        Solver.train(X)
        costs.append(Solver.history["cost"][-1])

        solution = Solver.get_solution()  # trial function solution
        solution = solution.reshape(x.shape)

        cost.add_trace(go.Scatter(x=Solver.t[int(epochs/5):], y=Solver.history["cost"][int(epochs/5):], mode="lines", name=name, line=dict(width=5)))
        ls.add_trace(go.Scatter(x=Solver.t, y=Solver.history["eta"], mode="lines", name=name, line=dict(width=5)))

    cost.update_layout(
        font_family="Garamond",
        font_size=35,
        title = "Cost during training for different learning schedules",
        xaxis_title="Epoch",
        yaxis_title= "Cost")
    ls.update_layout(
        font_family="Garamond",
        font_size=35,
        title = "Shape of learning schedules",
        xaxis_title="Epoch",
        yaxis_title= "Learning rate")
    cost.show()
    ls.show()

if __name__ == "__main__":
    main()

