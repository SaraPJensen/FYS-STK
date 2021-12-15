import autograd.numpy as np
from NN_sym_eq import Symmetric_matrix


class Symmetric_matrix(Symmetric_matrix):
    def index(self):
        true_vals, true_vecs = np.linalg.eig(self.A)
        true_vals = sorted(true_vals, key=lambda x: abs(x))

        val = self.eigval[-1]
        if val == 0:
            return None

        return np.argmin(abs(true_vals - val))


def main():
    seed = None
    np.random.seed(seed)
    n = 6
    idxs = np.zeros(n)
    all = 0
    N = 1000
    for _ in range(N):
        Q = np.random.normal(size=(n, n))
        A = (Q.T + Q) / 2

        x = np.random.normal(size=n)
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

        i = Solver.index()
        if i is not None:
            all += 1
            idxs[i] += 1
    print(idxs / N)


if __name__ == "__main__":
    main()
