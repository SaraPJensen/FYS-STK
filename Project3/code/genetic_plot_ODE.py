import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import pandas as pd

def progress_top():
    '''
    Plots the fitness of the top chromosome over the generations for the different diff. eqs. solved.
    '''
    Equation = ["ODE1", "ODE2", "PDE"]

    method = "tour"

    fig = go.Figure()

    for eq in Equation:
        path = f"data/ODE_PDE/{eq}_{method}.csv"

        df = pd.read_csv(path, header=2, sep=",")
        fig.add_trace(go.Scatter(
            x = df["Generation"],
            y = df["top_fitness"],
            mode="lines",
            line=dict(width = 5),
            name = eq ))


    fig.update_layout(
        font_family="Garamond",
        font_size=35,
        title = "Fitness of the best chromosome",
        xaxis_title="Generation",
        yaxis_title= "Fitness")

    fig.show()





def progress_top_random(eq):   #eq == "ODE" or "PDE"
    '''
    Plots the fitness of the top chromosome over the generations for the chosen diff. eq. using tournament selection and random selection.
    '''

    Method = ["tour", "random"]
    Title = ["Tournament", "Random"]

    fig = go.Figure()


    for method, title in zip(Method, Title):

        path = f"data/ODE_PDE/{eq}_{method}.csv"

        df = pd.read_csv(path, header=2, sep=",")

        fig.add_trace(go.Scatter(
            x = df["Generation"],
            y = df["top_fitness"],
            mode="lines",
            line=dict(width = 5),
            name = f"{title} selection"))


    fig.update_layout(
        font_family="Garamond",
        font_size=35,
        title = f"Fitness of the best chromosome for the {eq}",
        xaxis_title="Generation",
        yaxis_title= "Fitness")

    fig.show()






def main():
    '''
    Calls on the function of choice.
    '''
    #progress_top_random("ODE")
    #progress_top()


main()
