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
from numba import jit

import warnings
warnings.filterwarnings("ignore")


class Chromosome:
    def __init__(self, genes, matrix, l):
        self.genes = genes
        self.matrix = matrix
        self.length = l
        self.equation = ""
        self.eq_generator()


    def eq_generator(self):
        for i in range(self.length-1):
            self.equation += str(self.genes[i]) + "*" + self.matrix[i] + "+"

        self.equation += str(self.genes[self.length-1]) + "*" + self.matrix[self.length-1]


    def boundary_diff(self, func,  x_range, t_range):
        x_0 = 0
        x_L = 0
        t_0 = 0

        for t in t_range:
            try:
                x_0 += (func(0.0000001, t))**2      #prevent division by zero
                x_L += (func(x_range[-1], t))**2

            except:
                x_0 += 1e10
                x_L += 1e10

        for x in x_range:
            try:
                t_0 += (func(x, 0.0000001) - np.sin(np.pi*x))**2

            except:
                t_0 += 1e10

        return (x_0 + x_L + t_0)/len(x_range)



    def calc_fitness(self, x_range, t_range):


        func = lambda x, t: eval(self.equation)
        dM_dx = grad(grad(func, 0), 0)
        dM_dt = grad(func, 1)

        self.fitness = 0
        for x in x_range:
            for t in t_range:
                try:
                    diff = (dM_dx(x, t) - dM_dt(x, t))**2

                except: # ZeroDivisionError:
                    diff = 1e10

                self.fitness += diff

        self.fitness /= (len(x_range))**2

        boundary = self.boundary_diff(func, x_range, t_range)
        self.fitness += boundary*10




    def calc_fitness_print(self, x_range, t_range):   #use to print out the deviance from the differential eq and boundary conditions

        print("Equation: ", self.equation)
        func = lambda x, t: eval(self.equation)
        dM_dx = grad(grad(func, 0), 0)
        dM_dt = grad(func, 1)

        self.fitness = 0
        for x in x_range:
            for t in t_range:
                try:
                    diff = (dM_dx(x, t) - dM_dt(x, t))**2

                except: # ZeroDivisionError:
                    diff = 1e10

                self.fitness += diff

        self.fitness /= (len(x_range))**2
        print("Total diff eq deviance: ", self.fitness)

        boundary = self.boundary_diff(func, x_range, t_range)

        print("Boundary deviance: ", boundary)
        self.fitness += boundary*10




    def __gt__(self, other):
        return self.fitness > other.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness

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
    def __init__(self, size_pop, poly, generations, x_range, t_range):
        self.size_pop = size_pop    #no. of chromosomes
        self.generations = generations    #no. of generations

        self.x_range = x_range
        self.t_range = t_range
        self.matrix, self.len = self.design_matrix(poly)

        self.Chromosomes = np.zeros(self.size_pop, dtype=Chromosome)

        for i in range (0, size_pop):
            genes = np.random.normal(0, 1, self.len)  #indices, normal distribution
            c = Chromosome(genes, self.matrix, self.len)
            self.Chromosomes[i] = c



    def design_matrix(self, poly):
        l = int((poly+1)*(poly+2)/2)		# Number of features
        X = []
        for i in range(0, poly+1):
            q = int((i)*(i+1)/2)

            for k in range(i+1):
                X.append(f"(x**{i-k})*(t**{k})")
        return X, l




    def fitness(self, write = True):
        for c in self.Chromosomes:
            c.calc_fitness(self.x_range, self.t_range)

        for c in self.Chromosomes:
            if (math.isnan(c.get_fitness()) or math.isinf(c.get_fitness())) or not np.isfinite(c.get_fitness()):
                c.set_fitness(1e10)

        self.Chromosomes = np.sort(self.Chromosomes)


        if write == True:
            fitness_vals = np.zeros(self.size_pop, dtype=Chromosome)    #does this do anything now??
            i = 0
            for c in self.Chromosomes:
                fitness_vals[i] = c.get_fitness()
                i += 1
            return fitness_vals, self.Chromosomes[0].get_equation()


        else:
            print("Final chromosome fitness vals:")
            for c in self.Chromosomes[:10]:
                print("Fitness value: ", c.get_fitness())




    def fitness_print(self):   #use to
        for c in self.Chromosomes:
            c.calc_fitness_print(self.x_range, self.t_range)


        print()
        for c in self.Chromosomes:
            if (not c.get_fitness() != c.get_fitness() or math.isinf(c.get_fitness())) or not np.isfinite(c.get_fitness()):
                self.Chromosomes.remove(c)

        self.Chromosomes = sorted(self.Chromosomes)

        print("Final chromosome fitness vals:")
        for c in self.Chromosomes:
            print("Fitness value: ", c.get_fitness())
            print()





    def breed_mix(self, mutation, genes):   #this gives poor results
        elite = self.size_pop // 20   #pass on the 5% best individuals
        parents = 2*self.size_pop - elite*2

        #Find the chromosomes to reproduce to the next generation by using half a normal distribution with
        #standard deviation 0.2*current size of population to ensure that the best individuals are reproduced
        chroms = halfnorm.rvs(loc = 0, scale = 0.2*self.size_pop, size = parents).astype(int)

        self.past_gen = self.Chromosomes
        self.Chromosomes = np.zeros(self.size_pop, dtype=Chromosome)


        for e in range(elite):   #pass on the best individuals to the next generation, must be an even number
            self.Chromosomes[e] = self.past_gen[e]

        i = 0
        j = elite

        while i < parents:

            indices = np.random.randint(0, genes-1, int(genes/2))   #find which genes to swap
            new_genes = self.past_gen[chroms[i]].return_genes()

            for index in indices:
                new_genes[index] = self.past_gen[chroms[i+1]].return_genes()[index]

            if i % 4 == 0:   #do this for 50% of the chromosomes
                new_genes = self.mutate(new_genes, mutation)

            self.Chromosomes[j] = Chromosome(new_genes, self.matrix, self.len)

            i += 2
            j += 1



    def breed_swap(self, mutation, genes):
        elite = self.size_pop // 20   #pass on the 5% best individuals
        parents = 2*self.size_pop - elite*2
        chroms = halfnorm.rvs(loc = 0, scale = 0.2*self.size_pop, size = parents).astype(int)
        self.past_gen = self.Chromosomes


        self.Chromosomes = np.zeros(self.size_pop, dtype=Chromosome)

        for e in range(elite):   #pass on the best individuals to the next generation, must be an even number
            self.Chromosomes[e] = self.past_gen[e]


        i = 0
        j = elite

        while i < parents:

            index = np.random.randint(0, 0.6*genes)   #find where to swap
            new_genes = np.zeros(genes)

            new_genes[:index] = self.past_gen[chroms[i]].return_genes()[:index]  #use the first half of the genes from one chromosome, the second half of the other
            new_genes[index:] = self.past_gen[chroms[i+1]].return_genes()[index:]

            if i % 4 == 0:   #do this for 50% of the chromosomes
                new_genes = self.mutate(new_genes, mutation)

            self.Chromosomes[j] = Chromosome(new_genes, self.matrix, self.len)

            i += 2
            j += 1



    def breed_tournament(self, mutation, genes):
        elite = self.size_pop // 20
        self.past_gen = self.Chromosomes
        self.Chromosomes = np.zeros(self.size_pop, dtype=Chromosome)

        new = self.size_pop - elite

        for e in range(elite):   #pass on the best individuals to the next generation, must be an even number
            self.Chromosomes[e] = self.past_gen[e]

        j = elite

        while j < self.size_pop:  #tournament selection
            indices = np.sort(np.random.randint(0, genes-1, 5))  #pick out three random chromosomes, use the two best for reproduction

            split = np.random.randint(0, genes-1)   #where to split

            gene1 = self.past_gen[indices[0]].return_genes()
            gene2 = self.past_gen[indices[1]].return_genes()

            save_end = gene1[split:]

            gene1[split:] = gene2[split:]
            gene2[split:] = save_end

            gene1 = self.mutate(gene1, mutation)    #add mutations to half the new genes

            self.Chromosomes[j] = Chromosome(gene1, self.matrix, self.len)
            self.Chromosomes[j+1] = Chromosome(gene2, self.matrix, self.len)

            j += 1




    def mutate(self, genes, mutations):
        #Makes a random mutation to a number of the genes by replacing them with a random number
        for i in range(mutations):
            index = np.random.randint(1, len(genes))
            genes[index] = np.random.uniform(-1, 1)  
        return genes


    def print_eqs(self, number):
        for c in self.Chromosomes[:number]:
            print(c.get_equation())
            print(c.get_fitness())
            print()







def main(filename):
    x_range = np.linspace(0.0, 1, 10)   #prevent division by zero
    t_range = np.linspace(0.0, 1, 10)

    pop_size = 50
    genes = 50
    mutation_rate = 10
    generations = 20

    Pop = Population(pop_size, genes, generations, x_range, t_range)

    file = open(f"data/{filename}.csv", "w")
    file.write(f"Pop_size: {pop_size} Genes: {genes} Method: swap Mutated: {mutation_rate} \n")
    file.write("Generation,avg_fitness_10,avg_fitness_70,top_fitness,top_equation \n")

    for i in range(generations):
        print()
        print()
        print("Generation: ", i)
        fitness, equation = Pop.fitness(write = True)

        length = len(fitness)
        avg10 = np.sum(fitness[:int(length*0.1)])/int(length*0.1)
        avg70 = np.sum(fitness[:int(length*0.7)])/int(length*0.7)
        best = fitness[0]

        file.write(f"{i},{avg10},{avg70},{best},{equation} \n")

        Pop.breed_swap(mutation_rate, genes)


    file.close()

    print()


main("tester")
