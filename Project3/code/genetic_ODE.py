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
    def __init__(self, genome):
        '''
        Generates an object for each chromosome in the population. The genetic sequence is read and translated into an analytic expression in the constructor.
        '''
        self.genome = genome
        self.g = 0
        self.equation = self.expression(self.genome[0])

        if ("x" not in self.equation) and self.g < 0.8*len(self.genome):  #keep looping if x or t is missing, but only if there is a substantial part of the genome left

            self.equation += "+" + self.expression(self.genome[self.g])

        if "stop" in self.equation:
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

        i = index % 5
        exp = {0: lambda: "(" + self.expression(self.read_genes()) + self.operator(self.read_genes()) + self.expression(self.read_genes()) + ")",
                1: lambda: "(" + self.expression(self.read_genes()) + ")", 2: lambda: self.func(self.read_genes()), 3: lambda: self.digit(self.read_genes()),
                4: lambda: "x"}


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

        num = str(index % 10) + ".0"

        return num




    def calc_fitness(self, x_range):
        '''
        Calculate the total fitness, both due to deviance from boundary conditions and from the differential equation.
        It the expression does not contain x, set the fitness very low (uninteresting answer).
        Comment in the desired expressions based on which ODE you are solving with what boundary conditions.
        '''

        if ("x" not in self.equation):
            self.fitness = -1e10


        else:
            func = lambda x: eval(self.equation)
            dM_dx = grad(func, 0)
            d2M_dx2 = grad(grad(func, 0), 0)

            self.fitness = 0
            for x in x_range:
                try:
                    #diff = (dM_dx(x) - (2*x - func(x))/x)**2   #ODE1
                    #diff = (d2M_dx2(x) - 6.0*dM_dx(x) + 9.0*func(x))**2   #ODE5
                    diff = (d2M_dx2(x) + 100*func(x))**2    #ODE4

                    self.fitness -= diff

                except:
                    diff = 1e10

            self.fitness /= len(x_range)

            try:
                #boundary = (func(0.1) - 20.1)**2    #ODE1
                #boundary = (func(0.0))**2  + (dM_dx(0.0) - 2)**2   #ODE5
                boundary = (func(0.0))**2  + (dM_dx(0.0) - 10)**2   #ODE4


            except:
                boundary =  1e10


            self.fitness -= boundary*10




    def calc_fitness_print(self, x_range):
        '''
        Same as above, but also prints out the deviance from the differential equation and boundary conditions,
        to better see what is goin on.
        '''


        if ("x" not in self.equation):
            self.fitness = -1e10


        else:
            print("Equation: ", self.equation)
            func = lambda x: eval(self.equation)
            dM_dx = grad(func)
            d2M_dx2 = grad(grad(func))

            self.fitness = 0
            for x in x_range:
                try:
                    #diff = (dM_dx(x) - (2.0*x - func(x))/x)**2   #ODE1
                    #diff = (d2M_dx2(x) - 6.0*dM_dx(x) + 9.0*func(x))**2   #ODE5
                    diff = (d2M_dx2(x) + 100*func(x))**2    #ODE4

                    self.fitness -= diff

                except:
                    diff = 1e10

            self.fitness /= len(x_range)

            print("Total diff eq deviance: ", self.fitness)

            #boundary = (func(0.1) - 20.1)**2    #ODE1
            #boundary = (func(0.0))**2  + (dM_dx(0.0) - 2.0)**2   #ODE5
            boundary = (func(0.0))**2  + (dM_dx(0.0) - 10)**2   #ODE4

            print("Boundary deviance: ", boundary)
            self.fitness -= boundary*10
            print()

        print()



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




class Population:
    def __init__(self, size_pop, size_chrom, generations, x_range):
        '''
        Construct the population containing all the chromosome-objects.
        The genes are selected randomly for the first generation.
        '''
        self.size_pop = size_pop    #no. of chromosomes
        self.size_chrom = size_chrom    #no. of genes in each chromosome
        self.generations = generations    #no. of generations

        self.x_range = x_range

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
            c.calc_fitness(self.x_range)

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
                print("Equation: ", c.get_equation())







    def breed_tournament(self, mutation, genes):
        '''
        Reproduction scheme. Pass on the elite unchanged, use tournament selection to select the two parents. Use each pair of parents to create two new children using swapping.
        Mutations are applied to half the new children.
        '''
        elite = self.size_pop // 20   #must be an even number
        self.past_gen = self.Chromosomes
        self.Chromosomes = np.zeros(self.size_pop, dtype=Chromosome)

        new = self.size_pop - elite

        for e in range(elite):   #pass on the best individuals to the next generation, must be an even number
            self.Chromosomes[e] = self.past_gen[e]

        j = elite

        while j < self.size_pop - 1:  #tournament selection
            indices = np.sort(np.random.randint(0, self.size_pop - 1, 5))  #pick out three random chromosomes, use the two best for reproduction

            split = np.random.randint(0, genes - 1)   #where to split

            genome1 = self.past_gen[indices[0]].return_genes()
            genome2 = self.past_gen[indices[1]].return_genes()

            save_end = genome1[split:]

            genome1[split:] = genome2[split:]
            genome2[split:] = save_end

            genome1 = self.mutate(genome1, mutation)    #add mutations to half the new genes

            self.Chromosomes[j] = Chromosome(genome1)
            self.Chromosomes[j+1] = Chromosome(genome2)

            j += 1



    def breed_random(self, mutation, genes):
        '''
        Reproduction scheme. Pass on the elite unchanged. The rest of the chromosomes are created randomly.
        '''
        elite = self.size_pop // 20   #must be an even number
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

        print(len(self.Chromosomes))







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
    Comment in the right variables depending on which ODE you are solving.
    '''
    #x_range = np.linspace(0.1, 1, 10)   #ODE1
    x_range = np.linspace(0, 1, 10)   #ODE5, ODE4

    pop_size = 1000
    genes = 50
    mutation_rate = 10
    generations = 1000

    Pop = Population(pop_size, genes, generations, x_range)



    filename = "ODE_tour" + str(np.random.randint(0, 1000000))

    file = open(f"data/{filename}.csv", "w")
    file.write(f"ODE - Pop_size: {pop_size} - Genes: {genes} - Method: tour 5 - Mutated: {mutation_rate} - Mutation rate: 50% \n")
    #file.write("Diff. equation: ODE1, solution: y(x) = x + 2/x \n")
    file.write("Diff. equation: ODE4, solution: y(x) = sin(10x) \n")
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
