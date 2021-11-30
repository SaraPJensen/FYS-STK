#import numpy as np
from autograd import grad
import random
import autograd.numpy as np
from scipy.stats import halfnorm
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import warnings
warnings.filterwarnings("ignore")


class Chromosome:
    def __init__(self, genes):
        self.genes = genes
        self.g = 0
        self.equation = self.expression(self.genes[0])

        if "stop" in self.equation:
            self.equation = "0.0"


    def read_genes(self):
        self.g += 1

        if self.g > (len(self.genes) - 1):
            return "stop"

        return self.genes[self.g]


    def expression(self, index, stop = False):
        if index == "stop":
            return "stop"

        i = index % 5
        exp = {0: lambda: self.expression(self.read_genes()) + self.operator(self.read_genes()) + self.expression(self.read_genes()),
                1: lambda: "(" + self.expression(self.read_genes()) + ")", 2: lambda: self.func(self.read_genes()), 3: lambda: self.digit(self.read_genes()),
                4: lambda: "x"}
        return exp[i]()


    def operator(self, index, stop = False):
        if index == "stop":
            return "stop"

        i = index % 4
        ope = {0: "+", 1: "-", 2: "*", 3: "/"}
        return ope[i]

    def func(self, index, stop = False):
        if index == "stop":
            return "stop"

        i = index % 4
        func = {0: "np.sin", 1: "np.cos", 2: "np.exp", 3: "x**"}
        return func[i] + "(" + self.expression(self.read_genes()) + ")"


    def digit(self, index, stop = False):
        if index == "stop":
            return "stop"

        num = str(index % 10)+".0"

        return num






    def der_ODE(self, x):
        func = lambda x: eval(self.equation)
        dM_dx = grad(func, 0)

        try:
            diff = (dM_dx(x) - (2*x - func(x))/x)**2

        except ZeroDivisionError:
            diff = 1e10

        return diff


    def boundary_ODE(self, x_range):
        func = lambda x: eval(self.equation)

        try:
            y_0 = (func(0.1) - 20.1)**2

        except ZeroDivisionError:
            y_0 = 1e10

        return y_0



    def calc_fitness_ODE(self, x_range):
        func = lambda x: eval(self.equation)
        dM_dx = grad(func, 0)

        self.fitness = 0
        for x in x_range:
            try:
                #diff = (dM_dx(x) - (2*x - func(x))/x)**2

                diff = (dM_dx(x) - (1-func(x)*np.cos(x)/np.sin(x)))**2

            except ZeroDivisionError:
                diff = 1e10

            self.fitness += diff

        self.fitness /= len(x_range)

        try:
            #y_0 = (func(0.1) - 20.1)**2

            y_0 = 2.1/np.sin(0.1)

        except ZeroDivisionError:
            y_0 = 1e10

        self.fitness += y_0



    def __gt__(self, other):
        return self.fitness > other.fitness


    def get_fitness(self):
        return self.fitness


    def get_equation(self):
        return self.equation


    def return_genes(self):
        return self.genes



'''
gene = [0, 4, 0, 0, 3, 2, 3, 4]

sol = Chromosome(gene)


x_range = np.linspace(0.1, 1.0, 20)   #prevent division by zero
sol.calc_fitness_ODE(x_range)

print(sol.get_equation())
print(sol.get_fitness())
'''




class Population:
    def __init__(self, size_pop, size_chrom, generations, x_range):
        self.size_pop = size_pop    #no. of chromosomes
        self.size_chrom = size_chrom    #no. of genes in each chromosome
        self.generations = generations    #no. of generations
        self.Chromosomes = []
        self.x_range = x_range


        for c in range (0, size_pop):
            genes = random.sample(range(0, 255), size_chrom)
            genes[0] =  random.choice([0, 2])   #ensures that the equation isn't too trivial
            c = Chromosome(genes)
            self.Chromosomes.append(c)



    def fitness(self):
        i = 0
        fitness_vals = np.zeros(self.size_pop, dtype=Chromosome)    #does this do anything now??

        for c in self.Chromosomes:
            c.calc_fitness_ODE(self.x_range)

        print()
        for c in self.Chromosomes:
            if (math.isnan(c.get_fitness()) or math.isinf(c.get_fitness())) or not np.isfinite(c.get_fitness()):
            #math.isnan(c.get_fitness()) or
                self.Chromosomes.remove(c)


        self.remaining = len(self.Chromosomes)
        self.Chromosomes = sorted(self.Chromosomes)


        print()
        print("Final chromosome fitness vals:")
        for c in self.Chromosomes:
            #c.read_equation()
            print("Fitness value: ", c.get_fitness())


    def breed_mix(self):   #this gives poor results
        parents = 2*self.size_pop

        #Find the chromosomes to reproduce to the next generation by using half a normal distribution with
        #standard deviation 0.2*current size of population to ensure that the best individuals are reproduced
        chroms = halfnorm.rvs(loc = 0, scale = 0.2*self.remaining, size = parents).astype(int)

        self.past_gen = self.Chromosomes
        self.Chromosomes = []

        i = 0
        j = 0
        while i < parents:

            indices = np.random.randint(0, 49, 25)   #find which genes to swap
            new_genes = self.past_gen[chroms[i]].return_genes()

            for index in indices:
                new_genes[index] = self.past_gen[chroms[i+1]].return_genes()[index]

            if i % int(parents*0.1):   #do this for 10% of the chromosomes
                new_genes = self.mutate(new_genes)

            self.Chromosomes.append(Chromosome(new_genes))

            i += 2
            j += 1



    def breed_swap(self):
        parents = 2*self.size_pop
        chroms = halfnorm.rvs(loc = 0, scale = 0.2*self.remaining, size = parents).astype(int)
        self.past_gen = self.Chromosomes
        self.Chromosomes = []

        i = 0
        j = 0
        while i < parents:

            index = np.random.randint(0, 30)   #find where to swap
            new_genes = []

            new_genes[:index] = self.past_gen[chroms[i]].return_genes()[:index]  #use the first half of the genes from one chromosome, the second half of the other
            new_genes[index:] = self.past_gen[chroms[i+1]].return_genes()[index:]

            if i % 10 == 0:   #do this for 20% of the chromosomes
                new_genes = self.mutate(new_genes)

            new_genes = self.mutate(new_genes)
            self.Chromosomes.append(Chromosome(new_genes))

            i += 2
            j += 1




    def mutate(self, genes):
        #Makes a random mutation to one of the genes by replacing it with a random number
        for i in range(25):
            index = np.random.randint(1, 49)   #find where to swap, always keep the first gene as 0 or 2
            genes[index] = index = np.random.randint(0, 255)   #find where to swap
        return genes


    def print_eqs(self):
        for c in self.Chromosomes:
            print(c.get_equation())
            print(c.get_fitness())

            print()




x_range = np.linspace(0.1, 1.0, 15)   #prevent division by zero


Pop = Population(100, 50, 1, x_range)


generations = 30

for i in range(generations):
    print()
    print("Generation: ", i)
    print()
    Pop.fitness()
    Pop.breed_swap()


#Pop.fitness_print()
print()
Pop.fitness()
Pop.print_eqs()


print()
