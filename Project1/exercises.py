from functions import *

#np.random.seed(2018)

def main(exercise):
    # Generate data
    n = 20
    #x = np.arange(0,1,1/n)
    #y = np.arange(0,1,1/n)

    x = np.random.rand(n)
    y = np.random.rand(n)

    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y) + 0.05*np.random.randn(n, n)

    x_flat = np.ravel(x)
    y_flat = np.ravel(y)
    z_flat = np.ravel(z)

    if exercise == 1:
        '''
        Exercise 1
        '''
        poly = 1
        scaler = "standard"
        lamb = 0

        X = design_matrix(x_flat, y_flat, poly)

        X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.2)

        #Plot the graph
        #ThreeD_plot(x, y, z, "Function")

        #Plot prediction and calculate errors
        z_train_scaled, z_test_scaled, z_predict, z_model = OLS_Ridge(X_train, X_test, z_train, z_test, scaler, lamb, poly, "plot_prediction")

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
        scaler = "none"
        reg_method = "OLS"
        lamb = 0
        B_runs = 100
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
        #plot(fig)
        fig.show()

        #np.random.seed(2018)

        '''


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

        np.random.seed(2018)



        #Bootstrapping
        poly = 20
        B_runs = 100
        scaler = "none"
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
        '''



    elif exercise == 3:
        scaler = "none"
        poly = 20
        k_fold = 5
        reg_method = "OLS"
        lamb = 1
        dependency = "tradeoff"
        B_runs = 1

        #Calculate MSE values for test set with single validation set
        MSE_train, MSE_test = Bootstrap(x_flat, y_flat, z_flat, scaler, poly+1, B_runs, reg_method, 0, dependency)
        #Calcualte MSE for xval with k_fold folds
        mse = CrossVal(x_flat, y_flat, z_flat, scaler, poly, k_fold, reg_method, lamb, dependency)


        #Plotting
        deg_poly = [i for i in range(1, poly+2)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color="darkcyan"),
            marker=dict(size=9),
            name="No validation"))


        fig.add_trace(go.Scatter(x=deg_poly, y=mse.ravel(),
            mode='lines+markers',
            line=dict(dash='solid', width=4, color = "firebrick"),
            marker=dict(size=9),
            name="Cross-validation"))

        fig.update_layout(
            font_family="Garamond",
            font_size=33,
            title=f"MSE for training set with Cross-validation (k-fold = " + str(k_fold) + ") and no validation",
            xaxis_title="Degrees of polynomial",
            yaxis_title="Mean Squared Error",
            legend=dict(yanchor="top", xanchor="left", x=0.5, y=0.99)
            )
        plot(fig)
        fig.show()



    elif exercise == 4:

        poly = 20
        B_runs = 100
        reg_method = "Ridge"
        lamb = 0.01
        scaler = "none"
        #scaler = "none"
        k_fold = 0
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
        #plot(fig)
        fig.show()



        deg_poly = [i for i in range(1, poly+1)]

        #Plot MSE_test for 5 different lambdas
        MSE_train0, MSE_test0 = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 0, dependency)
        MSE_train1, MSE_test1 = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 0.001, dependency)
        MSE_train2, MSE_test2 = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 0.01, dependency)
        MSE_train3, MSE_test3 = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 0.1, dependency)
        MSE_train4, MSE_test4 = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 1, dependency)


        fig = go.Figure()

        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test0,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color = "orange"),
            marker=dict(size=9),
            name="Lambda = 0"))

        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test1,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color="darkcyan"),
            marker=dict(size=9),
            name="Lambda = 0.001"))

        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test2,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color = "firebrick"),
            marker=dict(size=9),
            name="Lambda = 0.01"))

        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test3,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color = "green"),
            marker=dict(size=9),
            name="Lambda = 0.1"))

        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test4,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color = "brown"),
            marker=dict(size=9),
            name="Lambda = 1"))



        fig.update_layout(
            font_family="Garamond",
            font_size=33,
            title=f"MSE as a function of complexity for {reg_method} regression",
            xaxis_title="Degrees of polynomial",
            yaxis_title="Mean Squared Error",
            legend=dict(yanchor="top", xanchor="left", x=0.01, y=0.99)
            )

        fig.show()



        dependency = "poly"   #plot bias-variance tradeoff

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
        poly = 8

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
        lamb = 0.001
        scaler = "standard"
        #scaler = "none"
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


main(2)

def terrain():

    # Load the terrain
    terrain1 = imread("SRTM_data_Norway_1.tif")
    #Dimensions of entire image: 3601 x 1801

    N = 1000
    start = 2000
    end = 3000
    poly = 2 # polynomial order
    terrain = terrain1[start:end,:N]

    # Creates mesh of image pixels
    x = np.linspace(0,1, np.shape(terrain)[0])
    y = np.linspace(0,1, np.shape(terrain)[1])
    x_mesh, y_mesh = np.meshgrid(x,y)

    z = terrain

    x_flat = np.ravel(x_mesh)
    y_flat = np.ravel(y_mesh)
    z_flat = np.ravel(z)

    scaler = Normalizer().fit(z)
    z_scaled = scaler.transform(z)

    #ThreeD_plot(x_mesh, y_mesh, z_scaled, "Terreng")

    lamb = 0


    X = design_matrix(x_flat, y_flat, poly)

    X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.2)

    #Plot prediction and calculate errors
    z_train_scaled, z_test_scaled, z_predict, z_model = OLS_Ridge(X_train, X_test, z_train, z_test, "standard", lamb, poly, "plot_prediction")

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



    # Show the terrain
    plt.figure()
    plt.title("Terrain over Norway 1")
    plt.imshow(terrain, cmap="gray")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
