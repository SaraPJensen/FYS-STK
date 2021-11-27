import autograd.numpy as np
from autograd import elementwise_grad as ele_grad
import matplotlib.pyplot as plt
from tqdm import tqdm


class Activations:
    def sigmoid(self, x):
        return 1 / (1 - np.exp(-x))
    
    def ReLU(self, x):
        return np.where(x > 0, x, 0)
    
    def PReLU(self, x):
        return np.where(x > 0, x, self.alpha * x)


class PDE_solver_NN_base(Activations):
    def __init__(self, X, epochs, nodes, eta0, activation="relu"):
        self.alpha = 0.01  # for PReLU (leaky relu)
        self.nodes = np.array([X.shape[1], *nodes, 1])
        self.rng = np.random.default_rng(21345)

        self.epochs = epochs
        self.eta0 = eta0
        
        P = self.initialize_weights(activation)
    
        activations = {"sigmoid": self.sigmoid,
                       "relu": self.ReLU,
                       "prelu": self.PReLU,
                       }
        self.act_func = activations[activation]
        print(f"Initial cost: {self.cost_func(P, X):.4f}")
        self.P = self.train(X, P)
        self.X = X
        print(f"Final cost: {self.cost_func(P, X):.4f}")

    def initialize_weights(self, activation):
        # P = np.zeros(len(self.nodes) - 1, dtype=object)
        P = [None] * (len(self.nodes) - 1)
        for i in range(1, len(self.nodes)):
            n = self.nodes[i - 1]
            m = self.nodes[i]
            P[i - 1] = self.rng.normal(0, 1, (n + 1, m))
            P[i - 1][-1, :] = 0.01
        return P

    def train(self, X, P):
        cost_func_grad = ele_grad(self.cost_func, 0)
        pbar = tqdm(range(self.epochs), desc=f"{self.cost_func(P, X):.10f}")
        for t in pbar:
            try:
                gradient = cost_func_grad(P, X)
                for l in range(len(self.nodes) - 1):
                    P[l] = P[l] - self.eta(t) * gradient[l]
                pbar.set_description(f"{self.cost_func(P, X):.10f}")
            except KeyboardInterrupt:
                break
        return P

    def feed_forward(self, x, P):
        prev = x
        for l in range(len(self.nodes) - 1):
            prev = np.concatenate((prev, np.ones((prev.shape[0], 1))), axis=1)
            z = prev @ P[l]
            a = self.act_func(z)
            prev = a
        return z

    def __call__(self, x, P):
        return self.feed_forward(x, P)

    def cost_func(self, P, X):
        error = (self.lhs(X, P) - self.rhs(X, P)) ** 2
        return np.sum(error, axis=0) / error.shape[0]

    def lhs(self, X, P):
        raise NotImplementedError

    def rhs(self, X, P):
        raise NotImplementedError

    def trial(self, X, P):
        raise NotImplementedError

    def get_solution(self, *args):
        return self.trial(*args, self.X, self.P)


    
class Exp_Decay(PDE_solver_NN_base):
    def lhs(self, X, P):
        return ele_grad(self.trial, 0)(X, P)
    def rhs(self, X, P):
        return -2 * self.trial(X, P)

    def trial(self, x, P):
        return 10 + x * self(x, P)



def g_analytic(x, gamma = 2, g0 = 10):
    return g0*np.exp(-gamma*x)


def main():
    x = np.linspace(0, 1, 101 * 101).reshape(-1, 1)
    nodes = [10, 10]
    epochs = 600
    eta = 0.001

    Solver = Exp_Decay(x, epochs, nodes, eta)
    solution = Solver.get_solution(x)
    plt.plot(x, g_analytic(x))
    plt.plot(x, solution)
    plt.show()

if __name__ == "__main__":
    main()
