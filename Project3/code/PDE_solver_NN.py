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
    def __init__(self, X, epochs, nodes, eta0, activation="relu", load=False, name=None):
        self.alpha = 0.01  # for PReLU (leaky relu)
        self.nodes = np.array([X.shape[1], *nodes, 1])
        self.rng = np.random.default_rng(21345)

        self.epochs = epochs
        self.eta0 = eta0

        activations = {"sigmoid": self.sigmoid,
                       "relu": self.ReLU,
                       "prelu": self.PReLU,
                       }
        self.act_func = activations[activation]
        self.X = X

        if not load:
            P = self.initialize_weights(activation)

            print(f"Initial cost: {self.cost_func(P, X):.4f}")
            self.P = self.train(X, P)
            print(f"Final cost: {self.cost_func(P, X):.4f}")
            if name is not None:
                self.save(name)
        else:
            self.P = self.load(name)


    def initialize_weights(self, activation):
        P = [None] * (len(self.nodes) - 1)
        for i in range(1, len(self.nodes)):
            n = self.nodes[i - 1]
            m = self.nodes[i]
            s = np.sqrt(2 / n)
            P[i - 1] = self.rng.normal(0, s, (n + 1, m))
            P[i - 1][-1, :] = 0.01
        return P

    def train(self, X, P):
        cost_func_grad = ele_grad(self.cost_func, 0)
        self.history = np.zeros(self.epochs)
        pbar = tqdm(range(self.epochs), desc=f"{self.cost_func(P, X):.10f}")
        for t in pbar:
            try:
                gradient = cost_func_grad(P, X)
                for l in range(len(self.nodes) - 1):
                    P[l] = P[l] - self.eta(t) * gradient[l]
                cf = self.cost_func(P, X)
                pbar.set_description(f"{cf:.10f}")
                self.history[t] = cf
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

    def save(self, name):
        name = f"{name}_{self.act_func.__name__}"
        for i in self.nodes[1:-1]:
            name += "_" + str(i)
        np.save("./nets/" + name, np.asarray(self.P))

    def load(self, name):
        name = f"{name}_{self.act_func.__name__}"
        for i in self.nodes[1:-1]:
            name += "_" + str(i)
        fname = "./nets/" + name + ".npy"
        return list(np.load(fname, allow_pickle=True))




