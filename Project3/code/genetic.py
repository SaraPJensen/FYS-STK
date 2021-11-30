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

        if ("x" not in self.equation) or ("t" not in self.equation):
            self.equation += "+" + self.expression(self.genes[self.g])

        if "stop" in self.equation:
            self.equation = "0.0"


    def read_genes(self):
        self.g += 1

        if self.g > (len(self.genes) - 1):
            print(self.g)
            return "stop"

        return self.genes[self.g]


    def expression(self, index, stop = False):
        if index == "stop":
            return "stop"

        i = index % 6
        exp = {0: lambda: "(" + self.expression(self.read_genes()) + self.operator(self.read_genes()) + self.expression(self.read_genes()) + ")",
                1: lambda: "(" + self.expression(self.read_genes()) + ")", 2: lambda: self.func(self.read_genes()), 3: lambda: self.digit(self.read_genes()),
                4: lambda: "x", 5: lambda: "t"}


        return exp[i]()


    def operator(self, index, stop = False):
        if index == "stop":
            return "stop"

        i = index % 5
        ope = {0: "+", 1: "-", 2: "*", 3: "/", 4: "**"}
        return ope[i]

    def func(self, index, stop = False):
        if index == "stop":
            return "stop"

        i = index % 3
        func = {0: "np.sin", 1: "np.cos", 2: "np.exp"} #, 3: "x**", 4: "t**"}   #should these be included or not?
        return func[i] + "(" + self.expression(self.read_genes()) + ")"


    def digit(self, index, stop = False):
        if index == "stop":
            return "stop"

        num = str(index % 11)+".0"

        if num == "10.0":
            num = "np.pi"

        return num


    def der_diff(self, x, t):   #calculates the derivatives for the diffusion equation
        func = lambda x, t: eval(self.equation)

        dM_dx = grad(grad(func, 0), 0)
        dM_dt = grad(func, 1)

        try:
            diff = (dM_dx(x, t) - dM_dt(x, t))**2

        except ZeroDivisionError:
            diff = 1e10

        return diff



    def boundary_diff(self, func,  x_range, t_range):
        #func = lambda x, t: eval(self.equation)
        x_0 = 0
        x_L = 0
        t_0 = 0

        for t in t_range:
            try:
                x_0 += (func(0.0000001, t))**2      #prevent division by zero
                x_L += (func(x_range[-1], t))**2

            except ZeroDivisionError:
                x_0 += 1e10
                x_L += 1e10

        for x in x_range:
            try:
                t_0 += (func(x, 0.0000001) - np.sin(np.pi*x))**2

            except ZeroDivisionError:
                t_0 += 1e10

        return x_0 + x_L + t_0



    def calc_fitness(self, x_range, t_range):

        if ("x" not in self.equation) or ("t" not in self.equation):
            self.fitness = 1e10

        else:
            func = lambda x, t: eval(self.equation)
            dM_dx = grad(grad(func, 0), 0)
            dM_dt = grad(func, 1)

            self.fitness = 0
            for x in x_range:
                for t in t_range:
                    try:
                        diff = (dM_dx(x, t) - dM_dt(x, t))**2

                    except ZeroDivisionError:
                        diff = 1e10

                    self.fitness += diff


            boundary = self.boundary_diff(func, x_range, t_range)
            self.fitness += boundary*10




    def calc_fitness_print(self, x_range, t_range):

        if ("x" not in self.equation) or ("t" not in self.equation):
            self.fitness = np.inf

        else:
            print("Equation: ", self.equation)
            func = lambda x, t: eval(self.equation)
            dM_dx = grad(grad(func, 0), 0)
            dM_dt = grad(func, 1)

            self.fitness = 0
            for x in x_range:
                for t in t_range:
                    try:
                        diff = (dM_dx(x, t) - dM_dt(x, t))**2

                    except ZeroDivisionError:
                        diff = 1e10

                    self.fitness += diff

            print("Total diff eq deviance: ", self.fitness)


            boundary = self.boundary_diff(func, x_range, t_range)

            print("Boundary deviance: ", boundary)
            self.fitness += boundary*10

        print()



    def __gt__(self, other):
        return self.fitness > other.fitness


    def get_fitness(self):
        return self.fitness


    def get_equation(self):
        return self.equation


    def return_genes(self):
        return self.genes

'''
#Analytic solution
first = [0, 2, 2, 0, 3, 0, 1, 0, 3, 21, 2, 0, 3, 21, 2, 5, 2, 2, 0, 0, 3, 21, 2, 4]


Analytic = Chromosome(first)

print(Analytic.get_equation())


x_range = np.linspace(0.0000001, 1, 15)   #prevent division by zero
t_range = np.linspace(0.0000001, 1, 15)

Analytic.calc_fitness(x_range, t_range)
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
        fitness_vals = np.zeros(self.size_pop, dtype=Chromosome)    #does this do anything now??

        for c in self.Chromosomes:
            c.calc_fitness(self.x_range, self.t_range)

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



    def fitness_print(self):
        i = 0
        fitness_vals = np.zeros(self.size_pop, dtype=Chromosome)    #does this do anything now??

        for c in self.Chromosomes:
            c.calc_fitness_print(self.x_range, self.t_range)


        print()
        for c in self.Chromosomes:
            if (not c.get_fitness() != c.get_fitness() or math.isinf(c.get_fitness())) or not np.isfinite(c.get_fitness()):
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

            if i % 4 == 0:   #do this for 50% of the chromosomes
                new_genes = self.mutate(new_genes)

            new_genes = self.mutate(new_genes)
            self.Chromosomes.append(Chromosome(new_genes))

            i += 2
            j += 1


    def breed_report(self):

        pass




    def mutate(self, genes):
        #Makes a random mutation to one of the genes by replacing it with a random number
        for i in range(3):
            index = np.random.randint(1, 49)   #find where to swap, ensure that the first gene is 0 or 2
            genes[index] = index = np.random.randint(0, 255)   #find where to swap
        return genes


    def print_eqs(self):
        for c in self.Chromosomes:
            print(c.get_equation())
            print(c.get_fitness())

            print()




x_range = np.linspace(0.0000001, 1, 10)   #prevent division by zero
t_range = np.linspace(0.0000001, 1, 10)




Pop = Population(30, 50, 1, x_range, t_range)


generations = 30

for i in range(generations):
    print()
    print("Generation: ", i)
    print()
    Pop.fitness()
    Pop.breed_swap()


print()
Pop.fitness_print()
#Pop.print_eqs()


print()




#np.exp((np.sin(t)))/8.0

#t*(x*t-x)/8.0     #This came up as a solution with very low fitness, 0.22
