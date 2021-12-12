from autograd import grad
import random
import autograd.numpy as np
from scipy.stats import halfnorm
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import mean_squared_error, r2_score


import warnings
warnings.filterwarnings("ignore")


class Chromosome:
    def __init__(self, genome, X_train, X_test, z_train, z_test):
        self.genome = genome
        self.X_train = X_train
        self.X_test = X_test
        self.z_train = z_train
        self.z_test = z_test


    def calc_fitness(self):
        z_model = self.X_train @ self.genome

        self.fitness = mean_squared_error(self.z_train, z_model)


    def get_variables(self):
        z_model = self.X_train @ self.genome
        z_predict = self.X_test @ self.genome

        mse_train = mean_squared_error(self.z_train, z_model)
        mse_test = mean_squared_error(self.z_test, z_predict)

        return z_model, z_predict, mse_train, mse_test


    def __gt__(self, other):
        return self.fitness > other.fitness


    def get_fitness(self):
        return self.fitness


    def get_equation(self):
        return self.equation

    def return_genes(self):
        return self.genome





class Population:
    def __init__(self, size_pop, poly, generations, X_train, X_test, z_train, z_test):
        self.genes = int((poly+1)*(poly+2)/2)

        self.X_train = X_train
        self.X_test = X_test
        self.z_train = z_train
        self.z_test = z_test


        self.size_pop = size_pop    #no. of chromosomes

        self.Chromosomes = np.zeros(self.size_pop, dtype=Chromosome)

        self.z_model = np.zeros(len(self.z_train), generations)   #save the results of the best chromosome over the generations in the population class
        self.z_predict = np.zeros(len(self.z_test), generations)
        self.mse_train = np.zeros(generations)
        self.mse_test = np.zeros(generations)



        for i in range (0, size_pop):
            genome = np.random.normal(0, 5, self.genes)  #indices, normal distribution
            c = Chromosome(genome, X_train, X_test, z_train, z_test)
            self.Chromosomes[i] = c




    def fitness(self, write = True, generation):

        for c in self.Chromosomes:
            c.calc_fitness()

        self.Chromosomes = np.sort(self.Chromosomes)


        if write == True:   #Use this when writing to file, only interested in the top chromosome
            z_model, z_predict, mse_train, mse_test = self.Chromosomes[0].get_variables()
            self.z_model[generation, :] = z_model   #one row for each generation
            self.z_predict[generation, :] = z_predict
            self.mse_train[generation] = mse_train
            self.mse_test[generation] = mse_test


    def get_variable(self):
        return self.z_model, self.z_predict, self.mse_train, self.mse_test






    def breed_tournament(self, mutation, genes):
        elite = self.size_pop // 20   #must be an even number
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

            self.Chromosomes[j] = Chromosome(genome1)
            self.Chromosomes[j+1] = Chromosome(genome2)

            j += 1





    def mutate(self, genome, mutations):
        #Makes a random mutation to a number of the genes by replacing them with a random number
        for i in range(mutations):
            index = np.random.randint(1, len(genome))   #find where to swap, ensure that the first gene is 0 or 2
            genome[index] =  np.random.uniform(-1, 1)

        return genome






def bias_variance(generations, filename):
    pop_size = 500
    genes = 50
    mutation_rate = 10
    generations = 1000
    B_runs = 100
    poly = 10


    X_train, X_test, z_train, z_test = Franke_data(n_points, noise, design, poly)


    filename = "genetic_bias_var" + name

    file = open(f"data_bv/{filename}.csv", "w")
    file.write("Generations,MSE_train,MSE_test,Bias,Variance\n")
    file.close()


    Pop_list = np.zeros(Sample, dtype = Population)

    for b in range(B_runs):   #create all the required populations with bootstrapping, one for each resampling
        X_train_boot, z_train_boot = resample(X_train, z_train)

        Pop_list[b] = Population(pop_size, poly, generations, X_train_boot, X_test, z_train_boot, z_test)


    z_predict_all = np.zeros(generations, len(z_test), B_runs)


    for pop, b in zip(Pop_list, range(B_runs)):   #go through each population and let it evolve over time
        for g in range(generations):

            pop.fitness(write = True)
            pop.breed_tournament(mutation_rate, genes)


        z_model, z_predict, mse_train, mse_test = pop.get_variables()

        z_predict_all[:, :, b] = z_predict


    MSE_test = np.zeros(generations)
    Bias = np.zeros(generations)
    Variance = np.zeros(generations)


    #Need to do something like this...

    z_test = self.z_test.reshape((-1, 1))

    for g in generations:

        #Which axis should these variables be calculated along? 1 or 2?? B_run is the final axis

        mse_test = np.mean( np.mean((z_test - z_predict_all[g, :, :])**2, axis=1, keepdims=True))
        bias = np.mean((z_test - np.mean(z_predict_all[g, :, :], axis=1, keepdims=True))**2)
        variance = np.mean(np.var(z_predict_all[g, :, :], axis=1, keepdims=True))

        file = open(f"data_bv/{filename}.csv", "a")
        file.write(f"{g},{mse_test},{bias},{variance}\n")
        file.close()

        MSE_test[g] = mse_test
        Bias[g] = bias
        Variance[g] = variance









    #Now get the variables from each population and use these to calculate the bias,
    #variance and average mse


    for g in range(0, self.poly+1):

        n = int((degree+1)*(degree+2)/2)

        X_train = self.X_train[:, :n]
        X_test = self.X_test[:, :n]



        z_predictions = np.zeros((len(self.z_test), B_runs))   #matrix containing the values for different bootstrap runs

        MSE_train_boot = []


        for i in range(B_runs):

            X_train_boot, z_train_boot = resample(X_train, self.z_train)

            beta = self.beta(X_train_boot, z_train_boot)

            z_model = X_train_boot @ beta
            z_predict = X_test @ beta

            MSE_train_boot.append(mean_squared_error(z_train_boot, z_model))

            z_predictions[:, i] = z_predict.ravel()

        z_test = self.z_test.reshape((-1, 1))

        mse_train = np.mean(MSE_train_boot)

        mse_test = np.mean( np.mean((z_test - z_predictions)**2, axis=1, keepdims=True) )
        bias = np.mean( (z_test - np.mean(z_predictions, axis=1, keepdims=True))**2 )
        variance = np.mean( np.var(z_predictions, axis=1, keepdims=True) )


        file = open(f"data_bv/{filename}.csv", "a")
        file.write(f"{degree},{mse_train},{mse_test},{bias},{variance}\n")
        file.close()
