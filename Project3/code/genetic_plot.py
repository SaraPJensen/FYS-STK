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



def progress_top():
    method = ["mix", "swap", "tour", "random"]
    Method = ["Mix", "Swap", "Tournament", "Random"]

    fig = go.Figure()

    for name, title in zip(method, Method):
        path = f"data/Diff_eq_{name}.csv"

        df = pd.read_csv(path, header=1, sep=",")


        fig.add_trace(go.Scatter(
            x = df["Generation"][:500],
            y = df["top_fitness"][:500],
            mode="lines",
            line=dict(width = 5),
            name = f"{title} selection"))

    fig.update_layout(
        font_family="Garamond",
        font_size=30,
        title = "Fitness of the best chromosome for the diffusion equation",
        xaxis_title="Generation",
        yaxis_title= "Fitness")

    fig.show()



def progress_avg10():

    method = ["mix", "swap", "tour", "random"]
    Method = ["Mix", "Swap", "Tournament", "Random"]

    fig = go.Figure()

    for name, title in zip(method, Method):
        path = f"data/Diff_eq_{name}.csv"

        df = pd.read_csv(path, header=1, sep=",")

        fig.add_trace(go.Scatter(
            x = df["Generation"][:500],
            y = df["avg_fitness_10"][:500],
            mode="lines",
            line=dict(width = 5),
            name = f"{title} selection"))

    fig.update_layout(
        font_family="Garamond",
        font_size=30,
        title = "Average fitness of the 10% best chromosomes for the diffusion equation",
        xaxis_title="Generation",
        yaxis_title= "Fitness")

    fig.show()



def progress_avg70():
    method = ["mix", "swap", "tour", "random"]
    Method = ["Mix", "Swap", "Tournament", "Random"]

    fig = go.Figure()

    for name, title in zip(method, Method):
        path = f"data/Diff_eq_{name}.csv"

        df = pd.read_csv(path, header=1, sep=",")

        fig.add_trace(go.Scatter(
            x = df["Generation"][2:500],
            y = df["avg_fitness_70"][2:500],
            mode="lines",
            line=dict(width = 5),
            name = f"{title} selection"))

    fig.update_layout(
        font_family="Garamond",
        font_size=30,
        title = "Average fitness of the 70% best chromosomes for the diffusion equation",
        xaxis_title="Generation",
        yaxis_title= "Fitness")

    fig.show()




def main():
    progress_top()
    #progress_avg10()
    #progress_avg70()

main()
