#import numpy as np
from autograd import grad
import random
import autograd.numpy as np

import warnings
warnings.filterwarnings("ignore")


class Chromosome:
    def __init__(self, genes):
        self.genes = genes
        self.g = 0
        self.equation = self.expression(self.genes[0])

        if "stop" in self.equation:
            self.equation = "0.0"

        #print("Equation: ", self.equation)
        #print("")


    def read_genes(self):
        self.g += 1
        #print("G: ", self.g)
        if self.g > (len(self.genes) - 1):
            #print("This should stop now...")
            return "stop"

        return self.genes[self.g]


    def read_equation(self):
        print(self.equation)


    def expression(self, index, stop = False):
        #print("Access expression")
        if index == "stop":
            #print("Expression stop")
            return "stop"

        i = index % 6
        exp = {0: lambda: self.expression(self.read_genes()) + self.operator(self.read_genes()) + self.expression(self.read_genes()),
                1: lambda: "(" + self.expression(self.read_genes()) + ")", 2: lambda: self.func(self.read_genes()), 3: lambda: self.digit(self.read_genes()),
                4: lambda: "x", 5: lambda: "t"}
        return exp[i]()


    def operator(self, index, stop = False):
        #print("Access operator")
        if index == "stop":
            #print("Operator stop")
            return "stop"

        i = index % 4
        ope = {0: "+", 1: "-", 2: "*", 3: "/"}
        return ope[i]

    def func(self, index, stop = False):
        #print("Access func")
        if index == "stop":
            #print("Func stop")
            return "stop"

        i = index % 6
        func = {0: "np.sin", 1: "np.cos", 2: "np.exp", 3: "np.log", 4: "x**", 5: "t**"}
        return func[i] + "(" + self.expression(self.read_genes()) + ")"

    def digit(self, index, stop = False):
        #print("Access digit")
        if index == "stop":
            #print("Digit stop")
            return "stop"

        num = str(index % 10)+".0"

        return num


    def der(self, x, t):
        #print("Equation: ", self.equation)
        func = lambda x, t: eval(self.equation)

        #print("Function calculated")
        dM_dx = grad(grad(func, 0), 0)
        #print("dM_dx calculated")

        #print("Derivative: ", dM_dx(x, t))



        dM_dt = grad(func, 1)
        #print("dM_dt calculated")

        return dM_dx(x, t), dM_dt(x, t)


    def fitness(self, x_range, t_range):
        if ("x" not in self.equation) or ("t" not in self.equation):
            self.fitness = 1e5

        else:
            self.fitness = 0
            for x in x_range:
                for t in t_range:
                    dx, dt = self.der(x, t)
                    E = (dx - dt)**2
                    self.fitness += E

        #print("Fitness calculated")



    def get_fitness(self):
        return self.fitness



print()
print()

x_range = np.linspace(1, 20, 5)
t_range = np.linspace(1, 20, 5)




#genes = [0, 0, 4, 2, 5, 0, 5]   #x*x + t
'''
genes = [3, 3]

C = Chromosome(genes)

C.read_equation()

C.fitness(x_range, t_range)

print(C.get_fitness())

'''

np.random.seed(123)

class Population:
    def __init__(self, size_pop, size_chrom, generations, x_range, t_range):
        self.size_pop = size_pop    #no. of chromosomes
        self.size_chrom = size_chrom    #no. of genes in each chromosome

        self.generations = generations    #no. of generations

        self.Chromosomes = []

        self.x_range = x_range
        self.t_range = t_range

        for c in range (0, size_pop):
            genes = random.sample(range(0, 255), size_chrom)
            genes[0] =  random.choice([0, 2])   #ensures that the equation isn't too trivial
            c = Chromosome(genes)
            self.Chromosomes.append(c)


    def fitness(self):
        i = 0
        fitness_vals = []
        for c in self.Chromosomes:
            #print("Chromosome: ", i)
            c.read_equation()
            c.fitness(self.x_range, self.t_range)
            fitness_vals.append(c.get_fitness())
            #print("Fitness value: ", c.get_fitness())
            #print()
            i += 1

        zipped = zip(fitness_vals, self.Chromosomes)
        sorted_pairs = sorted(zipped)
        tuples = zip(*sorted_pairs)
        fitness_vals, self.sorted_Chromosomes = [list(tuple) for tuple in tuples]

        print("Fitness values")
        print(fitness_vals)

        print("Chromosome fitness vals:")
        for c in self.sorted_Chromosomes:
            print(c.get_fitness())



    def breed(self):
        pass

    def mutate(self):
        pass





Pop = Population(10, 50, 1)

Pop.fitness()

print("Is this the right document?")
