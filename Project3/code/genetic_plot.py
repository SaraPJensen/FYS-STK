import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import pandas as pd


def progress_top():
    '''
    Plots the fitness of the best chromosome at each generation for the different methods used to solve the diffusion equation.
    '''
    method = ["mix", "tour", "random"]
    Method = ["Mixing", "Tournament", "Random"]

    fig = go.Figure()

    for name, title in zip(method, Method):
        path = f"data/Diff_eq_{name}.csv"

        df = pd.read_csv(path, header=1, sep=",")


        fig.add_trace(go.Scatter(
            x = df["Generation"][:500],
            y = df["top_fitness"][:500],
            mode="lines",
            line=dict(width = 5),
            name = title))

    fig.update_layout(
        font_family="Garamond",
        font_size=35,
        title = "Fitness of the best chromosome",
        xaxis_title="Generation",
        yaxis_title= "Fitness")

    fig.show()



def progress_avg10():
    '''
    Plots the average fitness of the 10 % best chromosomes at each generation for the different methods used to solve the diffusion equation.
    '''

    method = ["mix", "tour"]
    Method = ["Mixing", "Tournament"]

    fig = go.Figure()

    for name, title in zip(method, Method):
        path = f"data/Diff_eq_{name}.csv"

        df = pd.read_csv(path, header=1, sep=",")

        fig.add_trace(go.Scatter(
            x = df["Generation"][:500],
            y = df["avg_fitness_10"][:500],
            mode="lines",
            line=dict(width = 5),
            name = title))

    fig.update_layout(
        font_family="Garamond",
        font_size=35,
        title = "Average fitness of the 10% best chromosomes",
        xaxis_title="Generation",
        yaxis_title= "Fitness")

    fig.show()



def progress_avg70():
    '''
    Plots the average fitness of the 70 % best chromosomes at each generation for the different methods used to solve the diffusion equation.
    '''
    method = ["mix", "tour"]
    Method = ["Mixing", "Tournament"]

    fig = go.Figure()

    for name, title in zip(method, Method):
        path = f"data/Diff_eq_{name}.csv"

        df = pd.read_csv(path, header=1, sep=",")

        fig.add_trace(go.Scatter(
            x = df["Generation"][:500],
            y = df["avg_fitness_70"][:500],
            mode="lines",
            line=dict(width = 5),
            name = title))

    fig.update_layout(
        font_family="Garamond",
        font_size=35,
        title = "Average fitness of the 70% best chromosomes",
        xaxis_title="Generation",
        yaxis_title= "Fitness")

    fig.show()


def variable_comparison(method):   #method = "mix" or "tour"
    '''
    Plots the fitness of the best chromosome at each generation for the different variables used to solve the diffusion equation, either using mixing or tournament selection.
    '''

    Var = ["", "_elite", "_mutation"]
    Title = ["Default variables", "Smaller elite", "Different mutation scheme"]

    if method == "tour":
        name = "tournament selection"

    else:
        name = "mixing"

    fig = go.Figure()

    for var, title in zip(Var, Title):

        path = f"data/Diff_eq_{method}{var}.csv"
        df = pd.read_csv(path, header=1, sep=",")

        fig.add_trace(go.Scatter(
            x = df["Generation"][:500],
            y = df["top_fitness"][:500],
            mode="lines",
            line=dict(width = 5),
            name = title))

    fig.update_layout(
        font_family="Garamond",
        font_size=35,
        title = f"Fitness of the best chromosome using {name}",
        xaxis_title="Generation",
        yaxis_title= "Fitness")

    fig.show()






def main():
    '''
    Calls of the functions of choice.
    '''
    #progress_top()
    #progress_avg10()
    #progress_avg70()
    #variable_comparison("mix")
    #variable_comparison("tour")

main()
