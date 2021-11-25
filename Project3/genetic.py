import numpy as np


class Chromosome:
    def __init__(self, genes):
        self.genes = genes
        self.g = 0
        self.equation = self.expression(self.genes[0])


    def read_genes(self):
        self.g += 1
        if self.g == 49:
            self.equation = "0"

        return self.genes[self.g]


    def eval_eq(self, x, t):
        sol = eval(self.equation)
        return sol


    def read_equation(self):
        print(self.equation)


    def expression(self, index):
        i = index % 6
        exp = {0: lambda: self.expression(self.read_genes()) + self.operator(self.read_genes()) + self.expression(self.read_genes()),
                1: lambda: self.expression(self.read_genes()), 2: lambda: self.func(self.read_genes()), 3: lambda: self.digit(self.read_genes()),
                4: lambda: "x", 5: lambda: "t"}
        return exp[i]()

    def operator(self, index):
        i = index % 4
        ope = {0: "+", 1: "-", 2: "*", 3: "/"}
        return ope[i]

    def func(self, index):
        i = index % 4
        func = {0: "np.sin", 1: "np.cos", 2: "np.exp", 3: "np.log"}
        return func[i] + "(" + self.expression(self.read_genes()) + ")"

    def digit(self, index):
        num = str(index % 10)
        return num


    def dx_sq(self, x, t):

        pass

    def dt(self, x, t):


        pass



    def fitness(self, x_range, t_range):
        for x in x_range:
            for t in t_range:
                E = (dx_sq(self, x, t) - dt(self, x, t))**2
                self.fitness += E



    def get_fitness(self):
        return self.fitness



print()
print()

x_range = np.linspace(0, 1, 100)
t_range = np.linspace(0, 1, 100)

x, t = meshgrid()

genes = [16, 3, 7, 4, 10, 28, 24, 1, 2, 4]

C = Chromosome(genes)

C.read_equation()

print (C.eval_eq(3.14/2, 0))
