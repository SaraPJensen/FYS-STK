import autograd.numpy as np
from autograd import elementwise_grad as ele_grad
import plotly.graph_objects as go
from PDE_solver_NN import PDE_solver_NN_base


class Symmetric_matrix(PDE_solver_NN_base):
    def lhs(self, X, P):
        return ele_grad(self, 0)(self.t_max, P)

    def rhs(self, X, P):
        x = self(X, P)[-1].reshape(-1, 1)
        xxAx = self.xx0 * self.A @ x
        xAxx = x.T @ self.A @ x * x
        return xxAx - xAxx

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
        self.eigvec[t, :] = vec[:, 0]

    def eig(self):
        return self.eigval[self.epochs - 1], self.eigvec[self.epochs - 1]

    def assess(self):
        true_vals, true_vecs = np.linalg.eig(self.A)
        # val, vec = self.eig()
        vec = self(self.t_max, self.P).reshape(-1,)
        vec /= np.linalg.norm(vec)
        val = (vec.T @ self.A @ vec) / (vec.T @ vec)
        
        idx = np.argmin(abs(true_vals - val))
        true_val = true_vals[idx]
        true_vec = true_vecs[:, idx]
        
        true_vec *= np.sign(true_vec[0])
        vec *= np.sign(vec[0])

        val_err = val - true_val
        vec_err = np.log10(vec / true_vec)
        print(true_vals)
        print(val_err)
        print(vec_err)
        print(vec)
        print(val, true_val)
        return val_err, vec_err


def main():
    seed = 2021
    np.random.seed(seed)
    n = 6
    Q = np.random.normal(size=(n,n))
    A = (Q.T + Q) / 2
    
    x = np.random.normal(size=n)
    tmax = 1e4
    nT = 101
    t = np.linspace(0, tmax, nT)

    Solver = Symmetric_matrix(nodes=[1, 12, 12, n],
                              activation="sigmoid",
                              epochs=3000,
                              eta0=0.0000034,
                              lmb=0,
                              gamma=0.8,
                              load=False,
                              name=None,
                              seed=seed,
                              )
    Solver.symmetrix(A, x, t)
    # print(Solver.A)
    (Solver.assess())
    # fig = go.Figure(data=go.Scatter(y=Solver.history, mode="lines"))
    # fig.show()



if __name__ == "__main__":
    main()
