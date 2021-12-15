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
    '''
    Generates an object for each chromosome in the population. The genetic sequence is read and translated into an analytic expression in the constructor.
    '''
    def __init__(self, genome):
        self.genome = genome
        self.g = 0
        self.equation = self.expression(self.genome[0])

        if (("x" not in self.equation) or ("t" not in self.equation)) and self.g < 0.8*len(self.genome):  #keep looping if x or t is missing, but only if there is a substantial part of the genome left

            self.equation += "+" + self.expression(self.genome[self.g])

        if "stop" in self.equation:  #if no complete expression was found after reading all the genes, set the expression to "0.0"
            self.equation = "0.0"


    def read_genes(self):
        '''
        Read through the genetic sequence to construct the analytic expression
        '''
        self.g += 1

        if self.g > (len(self.genome) - 1):
            return "stop"

        return self.genome[self.g]


    def expression(self, index, stop = False):
        '''
        Inserts expression
        '''
        if index == "stop":
            return "stop"

        i = index % 6
        exp = {0: lambda: "(" + self.expression(self.read_genes()) + self.operator(self.read_genes()) + self.expression(self.read_genes()) + ")",
                1: lambda: "(" + self.expression(self.read_genes()) + ")", 2: lambda: self.func(self.read_genes()), 3: lambda: self.digit(self.read_genes()),
                4: lambda: "x", 5: lambda: "t"}


        return exp[i]()


    def operator(self, index, stop = False):
        '''
        Inserts operator
        '''
        if index == "stop":
            return "stop"

        i = index % 4
        ope = {0: "+", 1: "-", 2: "*", 3: "/"}
        return ope[i]


    def func(self, index, stop = False):
        '''
        Inserts functional expression
        '''
        if index == "stop":
            return "stop"

        i = index % 3
        func = {0: "np.sin", 1: "np.cos", 2: "np.exp"}
        return func[i] + "(" + self.expression(self.read_genes()) + ")"


    def digit(self, index, stop = False):
        '''
        Inserts digit
        '''
        if index == "stop":
            return "stop"

        num = str(index % 11)+".0"

        if num == "10.0":
            num = "np.pi"

        return num



    def boundary_diff(self, func,  x_range, t_range):
        '''
        Calculate and return the fitness due to deviance from the boundary conditions.
        '''
        x_0 = 0
        x_L = 0
        t_0 = 0

        for t in t_range:
            try:
                x_0 += (func(0.0, t))**2      #prevent division by zero and overflow
                x_L += (func(x_range[-1], t))**2

            except:
                x_0 += 1e10
                x_L += 1e10

        for x in x_range:
            try:
                t_0 += (func(x, 0.0) - np.sin(np.pi*x))**2

            except:
                t_0 += 1e10

        return (x_0 + x_L + t_0)/len(x_range)



    def calc_fitness(self, x_range, t_range):
        '''
        Calculate the total fitness, both due to deviance from boundary conditions and from the differential equation.
        It the expression does not contain either x or t, set the fitness very low (uninteresting answer).
        '''

        if ("x" not in self.equation) or ("t" not in self.equation):
            self.fitness = -1e10

        else:
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

                    self.fitness -= diff

            self.fitness /= (len(x_range))**2

            boundary = self.boundary_diff(func, x_range, t_range)
            self.fitness -= boundary*10





    def __gt__(self, other):
        '''
        Define how the object should be sorted, in reverse order.
        '''
        return self.fitness < other.fitness

    def set_fitness(self, fitness):
        '''
        Set the fitness by force form the outside
        '''
        self.fitness = fitness


    def get_fitness(self):
        return self.fitness


    def get_equation(self):
        return self.equation


    def return_genes(self):
        return self.genome



'''
#Analytic solution
first = [0, 2, 2, 0, 3, 0, 1, 0, 3, 21, 2, 0, 3, 21, 2, 5, 2, 2, 0, 0, 3, 21, 2, 4]


Analytic = Chromosome(first)

print(Analytic.get_equation())


x_range = np.linspace(0.0000001, 1, 15)   #prevent division by zero
t_range = np.linspace(0.0000001, 1, 15)

Analytic.calc_fitness(x_range, t_range)


test1 = [0, 2, 0, 4, 2, 5, 7, 3, 6, 8, 9, 2]
test2 = [0, 0, 0, 4, 2, 5, 7, 3, 6, 8, 9, 2]


Ex1 = Chromosome(test1)

print(Ex1.get_equation())

Ex2 = Chromosome(test2)

print(Ex2.get_equation())
'''



class Population:
    '''
    Construct the population containing all the chromosome-objects.
    The genes are selected randomly for the first generation.
    '''
    def __init__(self, size_pop, size_chrom, generations, x_range, t_range):
        self.size_pop = size_pop    #no. of chromosomes
        self.size_chrom = size_chrom    #no. of genes in each chromosome
        self.generations = generations    #no. of generations

        self.x_range = x_range
        self.t_range = t_range

        self.Chromosomes = np.zeros(self.size_pop, dtype=Chromosome)


        for i in range (0, size_pop):
            genome = random.sample(range(0, 255), self.size_chrom)
            genome[0] = random.choice([0, 2])   #ensures that the equation isn't too trivial
            c = Chromosome(genome)
            self.Chromosomes[i] = c



    def fitness(self, write = True):
        '''
        Calculate the fitness of each chromosome and sort them in order of fitness, the fittest first.
        If write = True, return the fitness values of all the chromosomes and the equation of the best choromosomes, so this can be written to file.
        '''
        for c in self.Chromosomes:
            c.calc_fitness(self.x_range, self.t_range)


        for c in self.Chromosomes:
            if (math.isnan(c.get_fitness()) or math.isinf(c.get_fitness())) or not np.isfinite(c.get_fitness()):
                c.set_fitness(-1e10)

        self.Chromosomes = np.sort(self.Chromosomes)


        if write == True:   #Use this when writing to file
            fitness_vals = np.zeros(self.size_pop, dtype=Chromosome)
            i = 0
            for c in self.Chromosomes:
                fitness_vals[i] = c.get_fitness()
                i += 1

            return fitness_vals, self.Chromosomes[0].get_equation()


        else:
            print("Final chromosome fitness vals:")
            for c in self.Chromosomes[:10]:
                print("Fitness value: ", c.get_fitness())






    def breed_mix(self, mutation, genes):
        '''
        Reproduction scheme. Pass on the elite unchanged, use probabilistic selection and mix the genes of the two parents perfectly to create a child,
        before mutations are applied to half of the new children.
        '''
        elite = self.size_pop // 10   #pass on the 10% best individuals
        parents = 2*self.size_pop - elite*2

        #Find the chromosomes to reproduce to the next generation by using half a normal distribution with
        #standard deviation 0.2*current size of population to ensure that the best individuals are reproduced
        chroms = halfnorm.rvs(loc = 0, scale = 0.2*self.size_pop, size = parents).astype(int)

        for i in range(len(chroms)):
            if chroms[i] > self.size_pop:
                chroms[i] = 0

        self.past_gen = self.Chromosomes
        self.Chromosomes = np.zeros(self.size_pop, dtype=Chromosome)


        for e in range(elite):   #pass on the best individuals to the next generation, must be an even number
            self.Chromosomes[e] = self.past_gen[e]

        i = 0
        j = elite

        while i < parents:

            indices = np.random.randint(0, genes - 1, int(genes/2))   #find which genes to swap
            new_genome = self.past_gen[chroms[i]].return_genes()

            for index in indices:
                new_genome[index] = self.past_gen[chroms[i+1]].return_genes()[index]

            if i % 4 == 0:   #do this for 50% of the chromosomes
            new_genome = self.mutate(new_genome, mutation)

            self.Chromosomes[j] = Chromosome(new_genome)

            i += 2
            j += 1




    def breed_tournament(self, mutation, genes):
        '''
        Reproduction scheme. Pass on the elite unchanged, use tournament selection to select the two parents. Use each pair of parents to create two new children using swapping.
        Mutations are applied to half the new children.
        '''
        elite = self.size_pop // 10
        self.past_gen = self.Chromosomes
        self.Chromosomes = np.zeros(self.size_pop, dtype=Chromosome)


        for e in range(elite):   #pass on the best individuals to the next generation, must be an even number
            self.Chromosomes[e] = self.past_gen[e]

        j = elite

        while j < self.size_pop - 1:  #tournament selection
            indices = np.sort(np.random.randint(0, self.size_pop - 1, 5))  #pick out three random chromosomes, use the two best for reproduction

            split = np.random.randint(0, genes-1)   #where to split

            genome1 = self.past_gen[indices[0]].return_genes()
            genome2 = self.past_gen[indices[1]].return_genes()

            save_end = genome1[split:]

            genome1[split:] = genome2[split:]
            genome2[split:] = save_end

            genome1 = self.mutate(genome1, mutation)    #add mutations to half the new genes
            genome2 = self.mutate(genome2, mutation)

            self.Chromosomes[j] = Chromosome(genome1)
            self.Chromosomes[j+1] = Chromosome(genome2)

            j += 1





    def breed_random(self, mutation, genes):
        '''
        Reproduction scheme. Pass on the elite unchanged. The rest of the chromosomes are created randomly.
        '''
        elite = self.size_pop // 10
        self.past_gen = self.Chromosomes
        self.Chromosomes = np.zeros(self.size_pop, dtype=Chromosome)


        for e in range(elite):   #pass on the best individuals to the next generation, must be an even number
            self.Chromosomes[e] = self.past_gen[e]

        j = elite

        while j < self.size_pop:  #generate new chromosomes with random genes
            genome = random.sample(range(0, 255), self.size_chrom)
            genome[0] = random.choice([0, 2])   #ensures that the equation isn't too trivial
            c = Chromosome(genome)
            self.Chromosomes[j] = c

            j += 1






    def mutate(self, genome, mutations):
        '''
        Apply "mutations" random mutations to the genome by replacing the genes with a random number.
        '''
        for i in range(mutations):
            index = np.random.randint(1, len(genome))   #find where to swap, ensure that the first gene is 0 or 2
            genome[index] = np.random.randint(0, 255)   #find where to swap
        return genome


    def print_eqs(self, number):
        '''
        Print out all the equations and their corresponding fitness, if you should so desire.
        '''
        for c in self.Chromosomes[:number]:
            print(c.get_equation())
            print(c.get_fitness())
            print()





def main():
    '''
    Run the simulation. Everything is hard-coded, so need to change the variables to decide on variables such as population size, genome size, mutation rate and number of generations.
    The results are written to a file.
    '''
    x_range = np.linspace(0.0, 1, 10)
    t_range = np.linspace(0.0, 1, 10)

    pop_size = 1000
    genes = 50
    mutation_rate = 3   #5 %
    generations = 500

    Pop = Population(pop_size, genes, generations, x_range, t_range)

    filename = "Diff_eq_tour" + str(np.random.randint(0, 1000000))

    file = open(f"data/{filename}.csv", "w")
    file.write(f"Diffusion equation - Pop_size: {pop_size} - Genes: {genes} - Method: tournament 5 - Mutated: {mutation_rate} - Mutation rate: 100% - Elite: 10% \n")
    file.write("Generation,avg_fitness_10,avg_fitness_70,top_fitness,top_equation \n")
    file.close()

    print("Filename: ", filename)



    for i in range(generations):

        print()
        print()
        print("Generation: ", i)
        fitness, equation = Pop.fitness(write = True)


        length = len(fitness)
        avg10 = np.sum(fitness[:int(length*0.1)])/int(length*0.1)
        avg70 = np.sum(fitness[:int(length*0.7)])/int(length*0.7)
        best = fitness[0]

        file = open(f"data/{filename}.csv", "a")
        file.write(f"{i},{avg10},{avg70},{best},{equation} \n")

        file.close()


        if best >= -1e-10:
            break


        Pop.breed_tournament(mutation_rate, genes)

    file.close()

    print("Filename: ", filename)
    print()


main()
