import autograd.numpy as np
from autograd import elementwise_grad as ele_grad
import plotly.graph_objects as go
from PDE_solver_NN import PDE_solver_NN_base


class Symmetric_matrix(PDE_solver_NN_base):
    def lhs(self, X, P):
        x, t = X.T
        return ele_grad(self.trial, 1)(x, t, X, P)

    def rhs(self, X, P):
        x, t = X.T
        x_trial = self.trial(x, t, X, P).T
        print(x_trial.shape)
        xx = np.sum(x_trial * x_trial, axis=0)
        xxF = np.ones((len(xx), self.A.shape[0])) * xx[:, None]
        
        xAx = np.sum(x_trial.T @ (self.A @ x_trial), axis=0)
        xAxF = np.ones((len(xAx), self.A.shape[0])) * xAx[:, None]
        print("xx")
        print(xx.shape)
        print(xxF.shape)
        print(xAx.shape)
        print(xAxF.shape)
        print(self.I.shape)
        m = xxF @ self.A
        print(m.shape)
        M = xxF @ self.A + (1 - xAxF) @ self.I
        print(M.shape)
        print("We did it!")
        print((M @ x_trial - x_trial).shape)
        exit()
        return M @ x_trial - x_trial

    def trial(self, x, t, X, P):
        return np.exp(-t[:, None]) * x[:, None] + (1 - np.exp(-t[:, None])) * self(X, P)

    def symmetrix(self, A, x0, t):
        self.N = A.shape[0]
        self.A = A
        self.I = np.eye(self.N)
        self.x0 = x0.reshape(-1, 1)
        self.t_max = t[-1].reshape(-1, 1)

        x, t = np.meshgrid(x0, t)
        X = np.concatenate((x.reshape(-1, 1), t.reshape(-1, 1)), axis=1)
        
        self.eigval = np.zeros(self.epochs)
        self.eigvec = np.zeros((self.epochs, self.N))

        self.train(X)

    def record(self, t, cf, X, P):
        super().record(t, cf)
        eigval, eigvec = self.eig(X, P)
        self.eigval[t] = eigval
        self.eigvec[t, :] = eigvec

    def eig(self, X, P):
        # v = self.trial(self.x0, self.t, X, P)
        # Av = self.A @ v
        return 0, self.x0



def main():
    np.random.seed(2021)
    n = 6
    Q = np.random.normal(size=(n,n))
    A = Q.T + Q
    
    x = np.random.normal(size=n)
    tmax = 1e4
    nT = 11
    t = np.linspace(0, tmax, nT)

    Solver = Symmetric_matrix(2,
                              nodes=[10,10],
                              output_node=n,
                              activation="relu",
                              alpha=0,
                              epochs=100,
                              eta0=0.001,
                              lmb=0,
                              gamma=0,
                              load=False,
                              name=None,
                              seed=2021,
                              )
    Solver.symmetrix(A, x, t)

if __name__ == "__main__":
    main()
