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
        ope = {0: "+", 1: "-", 2: "*", 3: "/"}#, 4: "**"}
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






    def calc_fitness_ODE(self, x_range):

        try:
            func = lambda x: eval(self.equation)

        except:
            print(self.equation)
            exit()

        dM_dx = grad(func, 0)
        dM_dx2 = grad(dM_dx, 0)

        self.fitness = 0
        for x in x_range:
            try:
                #diff = (dM_dx(x) - (2*x - func(x))/x)**2
                #diff = (dM_dx(x) - (1-func(x)*np.cos(x)/np.sin(x)))**2
                #diff = (x*dM_dx2(x) + (1-x)*dM_dx(x) + func(x))**2
                diff = (dM_dx(x) + 0.2*func(x) - np.exp(0.2*x)*np.cos(x))**2

            except: # ZeroDivisionError:
                diff = 1e10

            self.fitness += diff

        self.fitness /= len(x_range)

        try:
            #y_0 = (func(0.1) - 20.1)**2
            #y_0 = (func(0.1) - 2.1/np.sin(0.1))**2
            y_0 = (func(0))**2

            #y_0 = (func(0) -1)**2
            #y_1 = (func(1))**2

        except: # ZeroDivisionError:
            y_0 = 1e10
            #y_1 = 1e10

        self.fitness += y_0
        #self.fitness += y_1




    def __gt__(self, other):
        return self.fitness > other.fitness


    def get_fitness(self):
        return self.fitness


    def get_equation(self):
        return self.equation


    def return_genes(self):
        return self.genes






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
        for c in self.Chromosomes[:10]:
            print("Fitness value: ", c.get_fitness())



    def breed_mix(self, mutation):   #this gives poor results
        elite = self.size_pop // 20
        parents = 2*self.size_pop - elite*2

        #Find the chromosomes to reproduce to the next generation by using half a normal distribution with
        #standard deviation 0.2*current size of population to ensure that the best individuals are reproduced
        chroms = halfnorm.rvs(loc = 0, scale = 0.2*self.remaining, size = parents).astype(int)

        self.past_gen = self.Chromosomes
        self.Chromosomes = []

        i = 0
        j = 0

        for e in range(elite):   #pass on the best individuals to the next generation, must be an even number
            self.Chromosomes.append(self.past_gen[e])

        while i < parents:

            indices = np.random.randint(0, 49, 25)   #find which genes to swap
            new_genes = self.past_gen[chroms[i]].return_genes()

            for index in indices:
                new_genes[index] = self.past_gen[chroms[i+1]].return_genes()[index]

            if i % 4 == 0:   #do this for 50% of the chromosomes
                new_genes = self.mutate(new_genes, mutation)

            self.Chromosomes.append(Chromosome(new_genes))

            i += 2
            j += 1



    def breed_swap(self, mutation):
        elite = self.size_pop // 20
        parents = 2*self.size_pop - elite*2
        chroms = halfnorm.rvs(loc = 0, scale = 0.2*self.remaining, size = parents).astype(int)
        self.past_gen = self.Chromosomes
        self.Chromosomes = []

        i = 0
        j = 0

        for e in range(elite):   #pass on the best individuals to the next generation, must be an even number
            self.Chromosomes.append(self.past_gen[e])

        while i < parents:

            index = np.random.randint(0, 30)   #find where to swap
            new_genes = []

            new_genes[:index] = self.past_gen[chroms[i]].return_genes()[:index]  #use the first half of the genes from one chromosome, the second half of the other
            new_genes[index:] = self.past_gen[chroms[i+1]].return_genes()[index:]

            if i % 4 == 0:   #do this for 50% of the chromosomes
                new_genes = self.mutate(new_genes, mutation)

            self.Chromosomes.append(Chromosome(new_genes))

            i += 2
            j += 1



    def breed_tournament(self, mutation):
        elite = self.size_pop // 20
        self.past_gen = self.Chromosomes
        self.Chromosomes = []

        new = self.size_pop - elite

        for e in range(elite):   #pass on the best individuals to the next generation, must be an even number
            self.Chromosomes.append(self.past_gen[e])

        for c in range(new//2):   #tournament selection
            indices = np.sort(np.random.randint(0, 49, 5))  #pick out three random chromosomes, use the two best for reproduction

            index = np.random.randint(0, 49)   #where to split

            gene1 = self.past_gen[indices[0]].return_genes()
            gene2 = self.past_gen[indices[2]].return_genes()

            save_end = gene1[index:]

            gene1[index:] = gene2[index:]
            gene2[index:] = save_end

            gene1 = self.mutate(gene1, mutation)    #add mutations to half the new genes

            self.Chromosomes.append(Chromosome(gene1))
            self.Chromosomes.append(Chromosome(gene2))



    def mutate(self, genes, number):
        #Makes a random mutation to a number of the genes by replacing them with a random number
        for i in range(number):
            index = np.random.randint(1, 49)   #find where to swap, ensure that the first gene is 0 or 2
            genes[index] = index = np.random.randint(0, 255)   #find where to swap
        return genes


    def print_eqs(self):
        for c in self.Chromosomes[:10]:
            print(c.get_equation())
            print(c.get_fitness())

            print()




x_range = np.linspace(0, 1.0, 15)   #prevent division by zero


Pop = Population(1000, 50, 1, x_range)


generations = 40

for i in range(generations):
    print("Generation: ", i)
    print()
    Pop.fitness()
    Pop.breed_swap(10)



#Pop.fitness_print()
print()
Pop.fitness()
Pop.print_eqs()


print()
