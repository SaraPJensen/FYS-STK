import autograd.numpy as np
from autograd import elementwise_grad as ele_grad
from tqdm import tqdm
from collections import defaultdict


class Activations:
    """ Implemented activation functions """

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def PReLU(self, x):
        return np.where(x > 0, x, self.alpha * x)


class Optimizer:
    """ Optimizer for momentum gradient decent """

    def __init__(self, gamma=0, layer_sizes=None):
        self.g = min(1, max(gamma, 0))  # ensure between 0 and 1
        self.prev_v = [np.zeros((i + 1, 1)) for i in layer_sizes]

    def __call__(self, eta_grad, layer=0):
        v = self.g * self.prev_v[layer] + eta_grad
        self.prev_v[layer] = v
        return v


class PDE_solver_NN_base(Activations):
    """
    Feed-Forward Neural Network for solving PDEs
    """

    def __init__(self,
                 # nodes in each layer, including input and output
                 nodes=[1, 10, 1],
                 activation="relu",    # actuvation function
                 alpha=0,              # leaky relu
                 epochs=100,           # training epochs
                 # initial learning rate. Implement eta method to change schedule. is constant by default
                 eta0=0.001,
                 lmb=0,                # L2 regulizer
                 gamma=0,              # momentum
                 load=False,           # Load weights from file instead of training
                 name=None,            # name of file (nickname)
                 seed=21345,           # random seed for weight initialization
                 tol=1e-12             # lowest cost during training
                 ):
        self.nodes = np.asarray(nodes)
        self.rng = np.random.default_rng(seed)
        self.name = name
        self.tol = tol

        self.epochs = epochs
        self.eta0 = eta0
        self.lmb = lmb
        self.Optim = Optimizer(gamma, self.nodes)

        # linearity for activations below 0 (leaky relu factor)
        self.alpha = alpha
        activations = {"sigmoid": self.sigmoid,
                       "tanh": self.tanh,
                       "relu": self.PReLU,
                       }
        self.act_func = activations[activation]

        self.t = np.arange(self.epochs)

        if not load:
            self.initialize_weights()
        else:
            self.P = self.load(name)
            self.load_history(name)

    def initialize_weights(self):
        """
        Initializes network parameter weights and biases.
        Weights are initialised using Xavier for sigmoid and tanh,
        and He for prelu
        Biases are set to 0.01
        """
        P = [None] * (len(self.nodes) - 1)
        for i in range(1, len(self.nodes)):
            n = self.nodes[i - 1]
            m = self.nodes[i]
            if self.act_func.__name__ in ["sigmoid", "tanh"]:
                s = np.sqrt(6 / (n + m))  # Xavier initialization
                P[i - 1] = self.rng.uniform(-s, s, (n + 1, m))
            else:  # He initialization
                s = np.sqrt(6 / (1 + self.alpha**2) / (n + m))
                P[i - 1] = self.rng.normal(0, s, (n + 1, m))
            P[i - 1][-1, :] = 0.01  # biases
        self.P = P

    def train(self, X, save=True):
        """
        Trains neural network, updates network parameters
        """
        P = self.P
        self.X = X
        print(f"Initial cost: {self.cost_func(P, X):.4f}")

        cost_func_grad = ele_grad(self.cost_func, 0)
        self.history = defaultdict(lambda: np.zeros(self.epochs))
        pbar = tqdm(range(self.epochs), desc=f"{self.cost_func(P, X):.10f}")
        for t in pbar:
            try:
                gradient = cost_func_grad(P, X)
                for L in range(len(self.nodes) - 1):
                    Lgrad = gradient[L] + self.lmb * P[L]
                    update = self.Optim(self.eta(t) * Lgrad, L)
                    P[L] = P[L] - update

                if save:
                    cf = self.cost_func(P, X)

                    if cf < self.tol or np.isnan(cf):
                        break
                    pbar.set_description(f"{cf:.10f}")
                    self.record(t, cf, X, P)

            except KeyboardInterrupt:
                break

        self.epochs = t
        for key in self.history.keys():
            self.history[key] = self.history[key][:t]

        print(f"Final cost: {self.cost_func(P, X):.5f}")
        self.P = P
        if self.name is not None:
            self.save_history(self.name)
            self.save(self.name)

    def record(self, t, cf, X, P):
        """
        Can be reimplemented to record different stuff during training
        Takes the evaluated cost function, as this is already calculated for the pbar
        """
        self.history["cost"][t] = cf

    def feed_forward(self, x, P):
        prev = x
        for L in range(len(self.nodes) - 1):
            prev = np.concatenate((prev, np.ones((prev.shape[0], 1))), axis=1)
            z = prev @ P[L]
            a = self.act_func(z)
            prev = a
        return z

    def __call__(self, X, P):
        """ Used to evaluate network """
        return self.feed_forward(X, P)

    def cost_func(self, P, X):
        """
        Calculates error as difference between lhs and rhs of PDE
        returnes the normalized square error
        """
        error = self.diff_eq(X, P) ** 2
        return np.sum(error) / error.shape[0]

    def diff_eq(self, X, P):
        """
        Differential equation ordered so it should be 0.
        must be implemented separately through inheritance

        Should only take X and P as arguments,
        and return f(x, u(x), u'(x) ...) evaluated with the trial function
        Derivatives taken with autograd.elementwise_grad (ele_grad)

        Example:
        PDE: d^2/dx^2 u(x, t) = d/dt u(x, t)
        lhs = ele_grad(ele_grad(trial, 0), 0)(x, t, P)
        rhs = ele_grad(trial, 1)(x, t, P)
        diff_eq = lhs - rhs

        trial function enforces boundary/initial conditions.
        Must be a function of the weights and biases of the network M,
        as such: self(X, P), where X is the datapoints, and P the weights/biases
        Simply pass the X and P from lhs/rhs to the trial func, and furhter on to self()
        """
        raise NotImplementedError

    def eta(self, epoch):
        """
        Learning schedule

        Can be re-implemented to try different schedules
        """
        return self.eta0

    def get_solution(self):
        """
        Return solution to PDE after network is trained
        """
        x = [i[:, None] for i in self.X.T]
        return self.trial(*x, self.P)

    def save(self, name):
        """ Save weights and biases """
        name = f"{name}_{self.act_func.__name__}"
        for i in self.nodes[1:-1]:
            name += "_" + str(i)
        print(f"Saving net {name}")
        np.save("./nets/" + name, np.asarray(self.P, dtype=object))

    def load(self, name):
        """ Load weights and biases """
        name = f"{name}_{self.act_func.__name__}"
        for i in self.nodes[1:-1]:
            name += "_" + str(i)
        print(f"Loading net {name}")
        fname = "./nets/" + name + ".npy"
        return list(np.load(fname, allow_pickle=True))

    def save_history(self, name):
        """ Save train history for later visulaization """
        name = f"TH_{name}_{self.act_func.__name__}"
        for i in self.nodes[1:-1]:
            name += "_" + str(i)
        print(f"Saving train history of {name}")
        t = np.arange(self.epochs)
        self.history["epochs"] = np.arange(self.epochs)
        np.save("./nets/" + name, dict(self.history))

    def load_history(self, name):
        """ Reload train history """
        name = f"TH_{name}_{self.act_func.__name__}"
        for i in self.nodes[1:-1]:
            name += "_" + str(i)
        print(f"Loading train history of {name}")
        self.history = np.load("./nets/" + name + ".npy",
                               allow_pickle=True).item()
        self.t = self.history["epochs"]
        self.epochs = self.t[-1]
