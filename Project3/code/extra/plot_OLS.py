import plotly as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def OLS(filename):
    data = pd.read_csv(f"data_bv/{filename}.csv", header = 0, sep = ",")

    poly = data["Polynomial"]
    mse = data["MSE_train"]
    bias = data["Bias"]
    variance = data["Variance"]


    fig = go.Figure()

    fig.add_trace(go.Scatter(x=poly, y=bias,
         mode='lines+markers',
         line=dict(width=4, color="darkgoldenrod"),
         marker=dict(size=9),
         name="Bias"))


    fig.add_trace(go.Scatter(x=poly, y=variance,
         mode='lines+markers',
         line=dict(width=4, color = "firebrick"),
         marker=dict(size=9),
         name="Variance"))


    fig.add_trace(go.Scatter(x=poly, y=mse,
         mode='lines+markers',
         line=dict(width=4, color = "darkcyan"),
         marker=dict(size=9),
         name="MSE"))

    fig.update_layout(
         font_family="Garamond",
         font_size=33,
         title=f"Bias-variance tradeoff for incresing complexity for OLS regression",
         xaxis_title="Degrees of polynomial",
         yaxis_title="",
         legend=dict(yanchor="top", xanchor="left", x=0.01, y=0.99)
         )
    fig.show()



OLS("OLS_bias_var172440")
