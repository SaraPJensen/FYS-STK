import autograd.numpy as np
from autograd import elementwise_grad as ele_grad
import plotly.graph_objects as go
from PDE_solver_NN import PDE_solver_NN_base
import pandas as pd


class Symmetric_matrix(PDE_solver_NN_base):
    def diff_eq(self, X, P):
        dx = ele_grad(self, 0)(self.t_max, P)

        x = self(self.t_max, P).reshape(-1, 1)
        xxAx = self.xx0 * self.A @ x
        xAxx = (x.T @ self.A @ x) * x
        return dx - xxAx + xAxx

    def symmetrix(self, A, x0, t):
        self.A = A
        self.xx0 = x0.T @ x0

        self.t_max = t[-1].reshape(-1, 1)
        X = t.reshape(-1, 1)

        self.eigval = np.zeros(self.epochs)
        self.eigvec = np.zeros((self.epochs, self.nodes[-1]))

        self.train(X)

    def record(self, t, cf, X, P):
        super().record(t, cf, X, P)

        vec = self(self.t_max, P).reshape(-1, 1)
        vec /= np.linalg.norm(vec)
        val = (vec.T @ self.A @ vec) / (vec.T @ vec)

        self.eigval[t] = val
        self.eigvec[t] = vec[:, 0]

    def assess(self):
        true_vals, true_vecs = np.linalg.eig(self.A)

        vec = self.eigvec[-1, :].reshape(1, -1)
        val = self.eigval[-1]

        idx = np.argmin(abs(true_vals - val))
        true_val = true_vals[idx]
        true_vec = true_vecs[:, idx].reshape(1, -1)

        true_vec *= np.sign(true_vec[0])
        vec *= np.sign(vec[0])

        val_err = np.log10(abs(val - true_val) / true_val)
        vec_err = np.log10(abs(vec - true_vec) / true_vec)
        print("True vals: ", true_vals)
        print("True vec: ", true_vec)
        print("vec: ", vec)
        print("val and true val: ", val, true_val)
        print("val_err: ", val_err)
        print("vec errs: ", vec_err)


def main():
    seed = 2022
    np.random.seed(seed)
    n = 6
    Q = np.random.normal(size=(n, n))
    A = (Q.T + Q) / 2

    x = np.random.normal(size=n)
    print(x)
    tmax = 1e2
    nT = 2
    t = np.linspace(0, tmax, nT)

    Solver = Symmetric_matrix(nodes=[1, 30, 30, n],
                              activation="tanh",
                              epochs=50,
                              eta0=0.0003,
                              lmb=0,
                              gamma=0,
                              load=False,
                              name=None,
                              seed=seed,
                              )
    Solver.symmetrix(A, x, t)
    Solver.assess()

    # Apd = pd.DataFrame(A)
    # print(Apd.to_latex(escape=False, float_format="%.4f", index=False))

    fig = go.Figure()
    [fig.add_trace(go.Scatter(x=np.arange(Solver.epochs), y=Solver.eigvec[:, i],
                   mode="lines", line=dict(width=5), name=f"ν{i}")) for i in range(n)]
    fig.update_layout(
        font_family="Garamond",
        font_size=35,
        title="Evolution of eigenvector elements during training",
        xaxis_title="Epoch",
        yaxis_title="Value")
    fig.show()


if __name__ == "__main__":
    main()
