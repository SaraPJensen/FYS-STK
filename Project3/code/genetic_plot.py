import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import pandas as pd

def Plotly(equation):
    func = lambda x, t: eval(equation)
    x_range = np.linspace(0.000001, 1, 20)   #prevent division by zero
    t_range = np.linspace(0.000001, 1, 20)

    x, t = np.meshgrid(x_range, t_range)
    T = func(x, t)

    fig = go.Figure(data=[go.Surface(x=x, y=t, z=T)])
    fig.show()



def Plot(equation):
    func = lambda x, t: eval(equation)
    x_range = np.linspace(0.000001, 1, 20)   #prevent division by zero
    t_range = np.linspace(0.000001, 1, 20)

    x, t = np.meshgrid(x_range, t_range)
    T = func(x, t)

    fig_predict = plt.figure()
    ax_predict = fig_predict.gca(projection='3d')

    surf_predict = ax_predict.plot_surface(x, t, T, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    fig_predict.colorbar(surf_predict, shrink=0.5, aspect=5)
    plt.show()



def progress_top(filename):
    path = "data/" + filename + ".csv"
    df = pd.read_csv(path, header=1, sep=",")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df["Generation"],
        y = df["top_fitness"],
        mode="lines", line=dict(width = 5)))

    fig.update_layout(
        font_family="Garamond",
        font_size=30,
        title = "Fitness of the best chromosome for each generation",
        xaxis_title="Generation",
        yaxis_title= "Fitness")

    fig.show()


def progress_avg10(filename):
    path = "data/" + filename + ".csv"

    df = pd.read_csv(path, header=1, sep=",")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df["Generation"],
        y = df["avg_fitness_10"],
        mode="lines", line=dict(width = 5)))

    fig.update_layout(
        font_family="Garamond",
        font_size=30,
        title = "Average fitness of the 10% best chromosomes",
        xaxis_title="Generation",
        yaxis_title= "Fitness")

    fig.show()


def progress_avg70(filename):
    path = "data/" + filename + ".csv"

    df = pd.read_csv(path, header=1, sep=",")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df["Generation"],
        y = df["avg_fitness_80"],
        mode="lines", line=dict(width = 5)))

    fig.update_layout(
        font_family="Garamond",
        font_size=30,
        title = "Average fitness of the 70% best chromosomes",
        xaxis_title="Generation",
        yaxis_title= "Fitness")

    fig.show()


def main():
    eq1 = "t*(x*t-x)/8.0"
    ex = "(np.cos((t+x))/np.exp(((x-x)+(np.exp(t)*np.exp(((np.exp((x))*t)-(x)))))))"

    progress_avg70("tester")

    #Plotly(ex)

main()
