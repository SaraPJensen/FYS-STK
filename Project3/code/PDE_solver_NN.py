import autograd.numpy as np
from autograd import elementwise_grad as ele_grad
from tqdm import tqdm
import warnings

# warnings.filterwarnings("error", category=RuntimeWarning)


class Activations:
    def sigmoid(self, x):
        return 1 / (1 - np.exp(-x))

    def abs(self, x):
        return np.abs(x)

    def PReLU(self, x):
        return np.where(x > 0, x, self.alpha * x)


class Optimizer:
    def __init__(self, gamma=0, layer_sizes=None):
        self.g = min(1, max(gamma, 0))  # ensure between 0 and 1
        self.prev_v = [np.zeros((i + 1, 1)) for i in layer_sizes]

    def __call__(self, eta_grad, layer=0):
        v = self.g * self.prev_v[layer] + eta_grad
        self.prev_v[layer] = v
        return v


class PDE_solver_NN_base(Activations):
    def __init__(self, 
                 inp_nodes,
                 nodes=[10,],          # nodes in each hidden layer
                 output_node=1,
                 activation="relu",    # actuvation function
                 alpha=0,
                 epochs=100,           # training epochs
                 eta0=0.001,           # initial learning rate. Implement eta method to change schedule. is constant by default
                 lmb=0,                # L2 regulizer
                 gamma=0,              # momentum
                 load=False,           # Load weights from file instead of training
                 name=None,            # name of file (nickname) 
                 seed=21345,           # random seed for weight initialization
                 tol=1e-12             # lowest cost during training
                 ):
        self.nodes = np.array([inp_nodes, *nodes, output_node])
        self.rng = np.random.default_rng(seed)
        self.name = name
        self.tol = tol

        self.epochs = epochs
        self.eta0 = eta0
        self.lmb = lmb
        self.Optim = Optimizer(gamma, self.nodes)

        
        self.alpha = alpha  # linearity for activations below 0 (leaky relu factor)
        activations = {"sigmoid": self.sigmoid,
                       "relu": self.PReLU,
                       "abs": self.abs
                       }
        self.act_func = activations[activation]

        self.t = np.arange(self.epochs)

        if not load:
            self.initialize_weights()
        else:
            self.P = self.load(name)
            self.load_history(name)

    def initialize_weights(self):
        P = [None] * (len(self.nodes) - 1)
        for i in range(1, len(self.nodes)):
            n = self.nodes[i - 1]
            m = self.nodes[i]
            if self.act_func.__name__ == "sigmoid":
                s = np.sqrt(1 / (n))
                P[i - 1] = self.rng.uniform(-0.1, s, (n + 1, m))
            else:
                s = np.sqrt(6 / (1 + self.alpha**2) / (n + m))
                P[i - 1] = self.rng.normal(0, s, (n + 1, m))
            P[i - 1][-1, :] = 0.01
        self.P = P

    def train(self, X):
        P = self.P
        self.X = X
        print(f"Initial cost: {self.cost_func(P, X):.4f}")

        cost_func_grad = ele_grad(self.cost_func, 0)
        self.history = np.zeros(self.epochs)
        pbar = tqdm(range(self.epochs), desc=f"{self.cost_func(P, X):.10f}")
        for t in pbar:
            try:
                gradient = cost_func_grad(P, X)
                for L in range(len(self.nodes) - 1):
                    Lgrad = gradient[L] + self.lmb * P[L]
                    update = self.Optim(self.eta(t) * Lgrad, L)
                    P[L] = P[L] - update

                cf = self.cost_func(P, X)

                if cf < self.tol or np.isnan(cf):
                    break

                pbar.set_description(f"{cf:.10f}")
                self.record(t, cf, X, P)

            except KeyboardInterrupt:
                break
        self.epochs = t
        self.history = self.history[:t]

        print(f"Final cost: {self.cost_func(P, X):.4f}")
        self.P = P
        if self.name is not None:
            self.save_history(self.name)
            self.save(self.name)

    def record(self, t, cf, X, P):
        """
        Can be reimplemented to record different stuff during training
        """
        self.history[t] = cf

    def feed_forward(self, x, P):
        prev = x
        for L in range(len(self.nodes) - 1):
            prev = np.concatenate((prev, np.ones((prev.shape[0], 1))), axis=1)
            z = prev @ P[L]
            a = self.act_func(z)
            prev = a
        return z

    def __call__(self, x, P):
        """ Used to evaluate network """
        return self.feed_forward(x, P)

    def cost_func(self, P, X):
        """
        Calculates error as difference between lhs and rhs of PDE
        returnes the normalized square error
        """
        error = (self.lhs(X, P) - self.rhs(X, P)) ** 2
        return np.sum(error) / error.shape[0]

    def lhs(self, X, P):
        """
        Left-hand side of differential equation,
        must be implemented separately through inheritance

        Should only take X and P as arguments,
        and return the lhs evaluated with the trial function
        Derivatives taken with autograd.elementwise_grad (ele_grad)

        Example:
        PDE: d^2/dx^2 u(x, t) = d/dt u(x, t)
        lhs = ele_grad(ele_grad(trial, 0), 0)(x, t, X, P)
        rhs = ele_grad(trial, 1)(x, t, X, P)

        trial function enforces boundary/initial conditions.
        Must be a function of the weights and biases of the network M,
        as such: self(X, P), where X is the datapoints, and P the weights/biases
        Simply pass the X and P from lhs/rhs to the trial func, and furhter on to self()
        """
        raise NotImplementedError

    def rhs(self, X, P):
        """
        Right-hand side of differential equation,
        must be implemented separately through inheritance
        See lhs for more documentation
        """
        raise NotImplementedError

    def trial(self, X, P):
        """
        Trial func is specific to PDE and boundary conditions,
        must be implemented separately through inheritance
        See lhs for more documentation
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
        return self.trial(*self.X.T, self.X, self.P)

    def save(self, name):
        """ Save weights and biases """
        name = f"{name}_{self.act_func.__name__}"
        for i in self.nodes[1:-1]:
            name += "_" + str(i)
        print(f"Saving net {name}")
        np.save("./nets/" + name, np.asarray(self.P))

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
        A = np.concatenate((t[:, None], self.history[:, None]), axis=1)
        np.save("./nets/" + name, A)

    def load_history(self, name):
        """ Reload train history """
        name = f"TH_{name}_{self.act_func.__name__}"
        for i in self.nodes[1:-1]:
            name += "_" + str(i)
        print(f"Loading train history of {name}")
        A = np.load("./nets/" + name + ".npy", allow_pickle=True)
        self.t, self.history = A.T
        self.epochs = self.t[-1]







