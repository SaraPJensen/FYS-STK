from functions import *

np.random.seed(2018)

def main(exercise):
    # Generate data
    n = 20
    x = np.sort(np.random.uniform(0, 1, n))
    y = np.sort(np.random.uniform(0, 1, n))
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y) + 0.03*np.random.randn(n, n)

    x_flat = np.ravel(x)
    y_flat = np.ravel(y)
    z_flat = np.ravel(z)

    if exercise == 1:
        '''
        Exercise 1
        '''
        poly = 25

        X = design_matrix(x_flat, y_flat, poly)

        X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.2)

        #Plot the graph
        ThreeD_plot(x, y, z, "Function")

        #Plot prediction and calculate errors
        z_train_scaled, z_test_scaled, z_predict, z_model = OLS_Ridge(X_train, X_test, z_train, z_test, "standard", 0, poly, "plot_prediction")

        print('')
        r2_train = r2_score(z_train_scaled, z_model)
        print(f"R2, train: {r2_train:.5}")
        print('')
        r2_test = r2_score(z_test_scaled, z_predict)
        print(f"R2, test: {r2_test:.5}")
        print('')

        mse_train = mean_squared_error(z_train_scaled, z_model)
        print(f"MSE, train: {mse_train:.5}")
        print('')
        mse_test = mean_squared_error(z_test_scaled, z_predict)
        print(f"MSE, test: {mse_test:.5}")
        print('')


    elif exercise == 2:
        '''
        Exercise 2
        '''
        scaler = "standard"
        reg_method = "OLS"
        lamb = 0
        B_runs = 1
        k_fold = 0
        poly = 20
        dependency = "tradeoff"


        #Generate figure 2.11: see how MSE changes as a function of the degree of the polynomial
        MSE_train, MSE_test = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, lamb, dependency)


        deg_poly = [i for i in range(1, poly+1)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color="darkcyan"),
            marker=dict(size=9),
            name="Testing data"))


        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_train,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color = "firebrick"),
            marker=dict(size=9),
            name="Training data"))

        fig.update_layout(
            font_family="Garamond",
            font_size=33,
            title=f"Mean squared error as a function of complexity for {reg_method} regression",
            xaxis_title="Degrees of polynomial",
            yaxis_title="Mean Squared Error",
            legend=dict(yanchor="top", xanchor="left", x=0.01, y=0.99)
            )
        fig.show()

        #Generate fig. 2.11 with bootstrapping
        B_runs = 100

        #Generate figure 2.11: see how MSE changes as a function of the degree of the polynomial
        MSE_train, MSE_test = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, lamb, dependency)


        deg_poly = [i for i in range(1, poly+1)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color="darkcyan"),
            marker=dict(size=9),
            name="Testing data"))


        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_train,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color = "firebrick"),
            marker=dict(size=9),
            name="Training data"))

        fig.update_layout(
            font_family="Garamond",
            font_size=33,
            title=f"Mean squared error as a function of complexity for {reg_method} regression",
            xaxis_title="Degrees of polynomial",
            yaxis_title="Mean Squared Error",
            legend=dict(yanchor="top", xanchor="left", x=0.01, y=0.99)
            )
        fig.show()


        #Bootstrapping
        poly = 20
        B_runs = 100
        scaler = "standard"
        #scaler = "none"
        dependency = "poly"

        MSE, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, lamb, dependency)
        deg_poly = [i for i in range(1, poly+1)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=deg_poly, y=Bias,
            mode='lines+markers',
            line=dict(width=4, color="darkgoldenrod"),
            marker=dict(size=9),
            name="Bias"))

        fig.add_trace(go.Scatter(x=deg_poly, y=Variance,
            mode='lines+markers',
            line=dict(width=4, color = "firebrick"),
            marker=dict(size=9),
            name="Variance"))

        fig.add_trace(go.Scatter(x=deg_poly, y=MSE,
            mode='lines+markers',
            line=dict(width=4, color = "darkcyan"),
            marker=dict(size=9),
            name="MSE"))


        fig.update_layout(
            font_family="Garamond",
            font_size=33,
            title=f"Bias-variance tradeoff for incresing complexity for {reg_method} regression",
            xaxis_title="Degrees of polynomial",
            yaxis_title="",
            legend=dict(yanchor="top", xanchor="left", x=0.01, y=0.99)
            )
        fig.show()




    elif exercise == 3:
        scaler = "none"
        poly = 10
        k_fold = 5
        reg_method = "Ridge"
        lamb = 0.001
        dependency = "poly"

        CrossVal(x_flat, y_flat, z_flat, scaler, poly, k_fold, reg_method, lamb, dependency)



    elif exercise == 4:

        poly = 20
        B_runs = 100
        reg_method = "Ridge"
        lamb = 0.0001
        scaler = "standard"
        #scaler = "none"
        k_fold = 0
        dependency = "poly"


        #Look at test/training MSE without Bootstrapping
        #Generate figure 2.11: see how MSE changes as a function of the degree of the polynomial
        MSE_train, MSE_test = tradeoff(x_flat, y_flat, z_flat, scaler, poly, reg_method, lamb, B_runs, k_fold)
        deg_poly = [i for i in range(1, poly+1)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color="darkcyan"),
            marker=dict(size=9),
            name="Testing data"))


        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_train,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color = "firebrick"),
            marker=dict(size=9),
            name="Training data"))

        fig.update_layout(
            font_family="Garamond",
            font_size=33,
            title=f"Mean squared error as a function of complexity for {reg_method} regression",
            xaxis_title="Degrees of polynomial",
            yaxis_title="Mean Squared Error",
            legend=dict(yanchor="top", xanchor="center", x=0.3, y=0.99)
            )
        fig.show()



        #Bootstrapping for Ridge
        poly = 20

        #Look at Bias-Variance tradeoff with bootstrap
        MSE, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, lamb, dependency)
        deg_poly = [i for i in range(1, poly+1)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=deg_poly, y=Bias,
            mode='lines+markers',
            line=dict(width=4, color="darkgoldenrod"),
            marker=dict(size=9),
            name="Bias"))

        fig.add_trace(go.Scatter(x=deg_poly, y=Variance,
            mode='lines+markers',
            line=dict(width=4, color = "firebrick"),
            marker=dict(size=9),
            name="Variance"))

        fig.add_trace(go.Scatter(x=deg_poly, y=MSE,
            mode='lines+markers',
            line=dict(width=4, color = "darkcyan"),
            marker=dict(size=9),
            name="MSE"))


        fig.update_layout(
            font_family="Garamond",
            font_size=33,
            title=f"Bias-variance tradeoff for incresing complexity for {reg_method} regression",
            xaxis_title="Degrees of polynomial",
            yaxis_title="",
            legend=dict(yanchor="top", xanchor="center", x=0.3, y=0.99)
            )
        fig.show()



        #Look at dependence on lambda for a given polynomial
        dependency = "lambda"
        poly = 5

        MSE, LAMBDA = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, lamb, dependency)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.log10(LAMBDA), y=MSE,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color="firebrick"),
            marker=dict(size=9),
            name="Testing data"))


        fig.update_layout(
            font_family="Garamond",
            font_size=33,
            title=f"Mean squared error as a function of lambda for {reg_method} regression",
            xaxis_title="log10(lambda)",
            yaxis_title="Mean Squared Error",
            legend=dict(yanchor="top", xanchor="center", x=0.01, y=0.99)
            )
        fig.show()



        #Add function for finding for what values of poly and lambda MSE is lowest. Is it possible to use a different type of diagram for this?
        #Maybe similar to the 3d plot? Ask about this in group session.

    elif exercise == 5:
        poly = 50
        B_runs = 10
        reg_method = "Lasso"
        lamb = 0.01
        #scaler = "standard"
        scaler = "none"
        k_fold = 0
        dependency = "poly"

        X = design_matrix(x_flat, y_flat, poly)

        X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.2)

        #Plot prediction and calculate errors
        z_train_scaled, z_test_scaled, z_predict, z_model = Lasso(X_train, X_test, z_train, z_test, scaler, lamb, poly)

        print('')
        r2_train = r2_score(z_train_scaled, z_model)
        print(f"R2, train: {r2_train:.5}")
        print('')
        r2_test = r2_score(z_test_scaled, z_predict)
        print(f"R2, test: {r2_test:.5}")
        print('')

        mse_train = mean_squared_error(z_train_scaled, z_model)
        print(f"MSE, train: {mse_train:.5}")
        print('')
        mse_test = mean_squared_error(z_test_scaled, z_predict)
        print(f"MSE, test: {mse_test:.5}")
        print('')

        '''
        MSE, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, lamb, dependency)
        deg_poly = [i for i in range(1, poly+1)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=deg_poly, y=Bias,
            mode='lines+markers',
            line=dict(width=4, color="darkgoldenrod"),
            marker=dict(size=9),
            name="Bias"))

        fig.add_trace(go.Scatter(x=deg_poly, y=Variance,
            mode='lines+markers',
            line=dict(width=4, color = "firebrick"),
            marker=dict(size=9),
            name="Variance"))

        fig.add_trace(go.Scatter(x=deg_poly, y=MSE,
            mode='lines+markers',
            line=dict(width=4, color = "darkcyan"),
            marker=dict(size=9),
            name="MSE"))


        fig.update_layout(
            font_family="Garamond",
            font_size=33,
            title=f"Bias-variance tradeoff for incresing complexity for {reg_method} regression",
            xaxis_title="Degrees of polynomial",
            yaxis_title="",
            legend=dict(yanchor="top", xanchor="center", x=0.3, y=0.99)
            )
        fig.show()


        dependency = "tradeoff"

        MSE_train, MSE_test = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, lamb, dependency)
        deg_poly = [i for i in range(1, poly+1)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color="darkcyan"),
            marker=dict(size=9),
            name="Testing data"))


        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_train,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color = "firebrick"),
            marker=dict(size=9),
            name="Training data"))

        fig.update_layout(
            font_family="Garamond",
            font_size=33,
            title=f"Mean squared error as a function of complexity for {reg_method} regression",
            xaxis_title="Degrees of polynomial",
            yaxis_title="Mean Squared Error",
            legend=dict(yanchor="top", xanchor="left", x=0.01, y=0.99)
            )
        fig.show()
        '''




main(5)


'''
n = 20
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y) + 0.03*np.random.randn(n, n)

x_flat = np.ravel(x)
y_flat = np.ravel(y)
z_flat = np.ravel(z)


scaler = "none"
reg_method = "OLS"
lamb = 0
B_runs = 3
k_fold = 0
poly = 25
dependency = 'tradeoff'


#Generate figure 2.11: see how MSE changes as a function of the degree of the polynomial
MSE_train, MSE_test = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, lamb, dependency)


deg_poly = [i for i in range(1, poly+1)]

fig = go.Figure()
fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test,
    mode='lines+markers',
    line=dict(dash='solid', width=4, color="darkcyan"),
    marker=dict(size=9),
    name="Testing data"))


fig.add_trace(go.Scatter(x=deg_poly, y=MSE_train,
    mode='lines+markers',
    line=dict(dash='solid', width=4, color = "firebrick"),
    marker=dict(size=9),
    name="Training data"))

fig.update_layout(
    font_family="Garamond",
    font_size=33,
    title=f"Mean squared error as a function of complexity for {reg_method} regression",
    xaxis_title="Degrees of polynomial",
    yaxis_title="Mean Squared Error",
    legend=dict(yanchor="top", xanchor="left", x=0.01, y=0.99)
    )
fig.show()
'''
