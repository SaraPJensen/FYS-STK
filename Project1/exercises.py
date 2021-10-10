from functions import *



#seed = 2018
#rng = np.random.default_rng(np.random.MT19937(seed=seed))

np.random.seed(2018)



def main(exercise):
    # Generate data
    n = 20
    noise = 0.15

    x = np.arange(0,1,1/n)
    y = np.arange(0,1,1/n)

    x, y = np.meshgrid(x, y)

    z = FrankeFunction(x, y) + noise*np.random.randn(n, n)

    x_flat = np.ravel(x)
    y_flat = np.ravel(y)
    z_flat = np.ravel(z)

    if exercise == 1:
        '''
        Exercise 1
        '''
        poly = 5
        scaler = "scalerRobust"
        lamb = 0

        X = design_matrix(x_flat, y_flat, poly)

        X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.2)

        #Plot the graph
        ThreeD_plot(x, y, z, "Function")

        #Plot prediction and calculate errors
        z_train_scaled, z_test_scaled, z_predict, z_model = OLS_Ridge(X_train, X_test, z_train, z_test, scaler, lamb, poly, "plot_prediction")

        mse_train = mean_squared_error(z_train_scaled, z_model)
        print(f"MSE, train: {mse_train:.5}")
        print('')
        mse_test = mean_squared_error(z_test_scaled, z_predict)
        print(f"MSE, test: {mse_test:.5}")
        print('')

        r2_train = r2_score(z_train_scaled, z_model)
        print(f"R2, train: {r2_train:.5}")
        print('')
        r2_test = r2_score(z_test_scaled, z_predict)
        print(f"R2, test: {r2_test:.5}")
        print('')

        '''
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_sc = scaler.transform(X_train)
        #print(np.shape(z_train)[0])
        #print(np.shape(z_train.reshape(z_train.shape[0], 1)))
        z_train_sc = scaler.fit_transform(z_train.reshape(-1, 1))
        #print(np.shape(z_train_sc))
        '''
        beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
        #print(beta)
        k = len(beta)       # number of parameters
        #var_error = noise**2       # Variance in the standard normally distibuted noise
        ste_beta = np.zeros(k)
        width = np.zeros(k) # Width of the confidence interval
        a = 0.10            # 100(1-a)% CI
        n = len(X_train)    # Number of samples

        for i in range(k):
            ste_beta[i] = noise * np.sqrt((np.linalg.pinv(X_train.T @ X_train))[i, i])
            #print(ste_beta[i])
            width[i] = ste_beta[i] * 1.6499     # t-value for a = 0.1, df = 304 # n - (k+1)

        #plt.figure(i)
        plt.scatter(np.arange(len(beta)), beta)
        plt.errorbar(np.arange(len(beta)), beta, xerr = 0, yerr = width, linestyle='')

        plt.title("90% Confidense Intervals for $\\beta_i$\nwith noise = " + str(noise) + "$\\mathcal{N}$(0, 1)")
        plt.xlabel("$i$")
        plt.ylabel("$\hat{\\beta}_i \pm t_{\\alpha/2, n-(k+1)} \cdot s_{\hat{\\beta}_i}$")
        #plt.savefig("90CI_poly5.png")
    
        plt.show()

    elif exercise == 2:
        '''
        Exercise 2
        '''
        scaler = "none"
        reg_method = "OLS"
        lamb = 0
        B_runs = 1
        k_fold = 0
        poly = 20
        dependency = "bias_variance"


        '''
        #np.random.seed(123)

        #Generate figure 2.11: see how MSE changes as a function of the degree of the polynomial
        MSE_train, MSE_test, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, lamb, dependency)


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
            legend=dict(yanchor="top", xanchor="center", x=0.5, y=0.99)
            )
        #plot(fig)
        fig.show()

        '''



        np.random.seed(123)


        #Generate fig. 2.11 with bootstrapping
        B_runs = 100

        #Generate figure 2.11: see how MSE changes as a function of the degree of the polynomial
        MSE_train, MSE_test, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, lamb, dependency)


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


        np.random.seed(123)


        #Bootstrapping
        poly = 20
        B_runs = 100
        scaler = "none"
        dependency = "bias_variance"

        MSE_train, MSE_test, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, lamb, dependency)
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

        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test,
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
        scaler = "scalerNone"
        poly = 20
        k_fold = 5
        reg_method = "OLS"
        lamb = 0
        dependency = "bias_variance"
        B_runs = 400
        seed = int(time())#2018
        rng = np.random.default_rng(np.random.MT19937(seed=seed))


        mse_cv = CrossVal(x, y, z, scaler, poly, k_fold, reg_method, lamb, rng, dependency)
        mse_cv_ = CrossVal(x, y, z, scaler, poly, 10, reg_method, lamb, rng, dependency)
        MSE_train, MSE_test, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly-1, B_runs, reg_method, lamb, dependency)

        # plt.plot(np.arange(1,poly+1), mse_cv, label="CV, kfold = 5")
        # plt.plot(np.arange(1,poly+1), mse_cv_, label="CV, kfold = 10")
        # plt.plot(np.arange(1,poly+1), MSE_test, label="BS")
        # plt.legend()


        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(1,poly+1), y=mse_cv,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color="darkcyan"),
            marker=dict(size=9),
            name="Cross Validation, k_fold = 5"))


        fig.add_trace(go.Scatter(x=np.arange(1,poly+1), y=mse_cv_,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color = "firebrick"),
            marker=dict(size=9),
            name="Cross Validation, k_fold = 10"))

        fig.add_trace(go.Scatter(x=np.arange(1,poly+1), y=MSE_test,
            mode='lines+markers',
            line=dict(dash='solid', width=4),
            marker=dict(size=9),
            name="Bootstrapping, b_runs = 400"))

        fig.update_layout(
            font_family="Garamond",
            font_size=33,
            title=f"Mean squared error as a function of complexity for {reg_method} regression",
            xaxis_title="Degree of polynomial",
            yaxis_title="Mean Squared Error",
            legend=dict(yanchor="top", xanchor="center", x=0.5, y=0.99)
            )
        plot(fig)
        fig.show()




    elif exercise == 4.1:

        poly = 10
        reg_method = "Ridge"
        scaler = "scalerNone"
        #scaler = "none"
        k_fold = 0
        B_runs=100
        dependency = "bias_variance"


        lambdas = 10
        lambdas_ = np.logspace(-10,0,lambdas)

        #np.random.seed(123)
        mse = np.zeros((poly, lambdas))
        for p in range(poly):
            for l, lamb in enumerate(lambdas_):
                _, mse_temp , _, _ = Bootstrap(x_flat, y_flat, z_flat, scaler, p, B_runs, reg_method, lamb, dependency)
                mse[p, l] = np.mean(mse_temp)


        min_mse = np.where(mse == np.min(mse))

        min_p = int(min_mse[0])
        min_l = int(min_mse[1])

        print("Min poly: ", np.arange(1,poly+1)[min_p])
        print("Min lambda: ", lambdas_[min_l])
        print("Min MSE: ", np.min(mse))


        plt.figure(f"bootstrap", figsize=(11, 9), dpi=80)
        plt.plot(np.log10(lambdas_)[min_l], np.arange(1,poly+1)[min_p], "or", label="min", markersize=10)
        plt.contourf(np.log10(lambdas_), np.arange(1,poly+1), mse)
        plt.ylabel("degree",fontsize=14)
        plt.xlabel("log10(lambda)",fontsize=14)
        plt.colorbar()
        plt.legend()

        plt.show()


        '''
        MSE_train, MSE_test, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, lamb, dependency)


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



    elif exercise == 4:

        poly = 20
        B_runs = 100
        reg_method = "Ridge"
        lamb = 0.001
        scaler = "none"
        k_fold = 0
        dependency = "bias_variance"

        '''
        MSE_train, MSE_test, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, lamb, dependency)


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
        '''

        '''
        dependency = "bias_variance"

        B_runs= 100

        poly = 10

        deg_poly = [i for i in range(1, poly+1)]

        #Plot MSE_test for 5 different lambdas
        np.random.seed(123)
        MSE_train0, MSE_test0, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 0, dependency)
        np.random.seed(123)
        MSE_train1, MSE_test1, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 0.00001, dependency)
        np.random.seed(123)
        MSE_train2, MSE_test2, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 0.001, dependency)
        np.random.seed(123)
        MSE_train3, MSE_test3, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 0.1, dependency)
        np.random.seed(123)
        MSE_train4, MSE_test4, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 10, dependency)


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
            name="Lambda = 1E-5"))

        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test2,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color = "firebrick"),
            marker=dict(size=9),
            name="Lambda = 1E-3"))

        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test3,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color = "green"),
            marker=dict(size=9),
            name="Lambda = 0.1"))

        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test4,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color = "blue"),
            marker=dict(size=9),
            name="Lambda = 10"))


        fig.update_layout(
            font_family="Garamond",
            font_size=33,
            title=f"MSE as a function of complexity for {reg_method} regression",
            xaxis_title="Degrees of polynomial",
            yaxis_title="Mean Squared Error",
            legend=dict(yanchor="top", xanchor="left", x=0.2, y=0.99)
            )

        fig.show()

        '''

        np.random.seed(123)
        #Look at dependence on lambda for a given polynomial
        dependency = "lambda"
        poly = 7

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

        min_pos = np.argmin(MSE)
        min_lamb = LAMBDA[min_pos]

        print("Minimum MSE: ", min(MSE))
        print("Optimal lambda: ", min_lamb)


        '''
        #Bootstrapping
        poly = 20
        B_runs = 100
        scaler = "none"
        dependency = "bias_variance"
        lamb = 0.15703
        np.random.seed(123)

        MSE_train, MSE_test, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, lamb, dependency)
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

        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test,
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
             legend=dict(yanchor="top", xanchor="left", x=0.01, y=0.6)
             )
        fig.show()

        '''






    elif exercise == 5:

        B_runs = 100
        reg_method = "Lasso"
        scaler = "none"
        k_fold = 0


        '''
        #Plot MSE_test for 5 different lambdas
        dependency = "bias_variance"

        poly = 10

        deg_poly = [i for i in range(1, poly+1)]


        np.random.seed(123)
        MSE_train0, MSE_test0, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, "OLS", 0, dependency)
        np.random.seed(123)
        MSE_train1, MSE_test1, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 0.000001, dependency)
        np.random.seed(123)
        MSE_train2, MSE_test2, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 0.0001, dependency)
        np.random.seed(123)
        MSE_train3, MSE_test3, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 0.01, dependency)
        np.random.seed(123)
        MSE_train4, MSE_test4, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 0.1, dependency)


        fig = go.Figure()

        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test0,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color = "orange"),
            marker=dict(size=9),
            name="OLS"))

        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test1,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color="darkcyan"),
            marker=dict(size=9),
            name="Lambda = 1E-6"))

        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test2,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color = "firebrick"),
            marker=dict(size=9),
            name="Lambda = 1E-4"))

        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test3,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color = "green"),
            marker=dict(size=9),
            name="Lambda = 0.01"))

        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test4,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color = "blue"),
            marker=dict(size=9),
            name="Lambda = 0.1"))

        fig.update_layout(
            font_family="Garamond",
            font_size=33,
            title=f"MSE as a function of complexity for {reg_method} regression",
            xaxis_title="Degrees of polynomial",
            yaxis_title="Mean Squared Error",
            legend=dict(yanchor="top", xanchor="left", x=0.01, y=0.99)
            )

        fig.show()



        '''

        np.random.seed(123)

        #Look at dependence on lambda for a given polynomial
        dependency = "lambda"
        poly = 7
        lamb = 0

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

        min_pos = np.argmin(MSE)
        min_lamb = LAMBDA[min_pos]

        print("Minimum MSE: ", min(MSE))
        print("Optimal lambda: ", min_lamb)


main(1)


#main(4.1)

def terrain(part):

    # Load the terrain
    terrain1 = imread("SRTM_data_Norway_1.tif")
    #Dimensions of entire image: 3601 x 1801

    N = 200
    start = 2100
    end = start + N
    poly = 5
    terrain = terrain1[start:end, :N]

    # Creates mesh of image pixels
    x = np.linspace(0,1, np.shape(terrain)[0])
    y = np.linspace(0,1, np.shape(terrain)[1])
    x_mesh, y_mesh = np.meshgrid(x,y)

    z = terrain

    x_flat = np.ravel(x_mesh)
    y_flat = np.ravel(y_mesh)
    z_flat = np.ravel(z)

    if part == "OLS_tradeoff":
        #Look at dependency on complexity for OLS with bootstrapping, bias variance

        scaler = "minmax"
        poly = 25
        B_runs = 100
        reg_method = "OLS"
        lamb = 0
        dependency = "bias_variance"

        np.random.seed(123)

        MSE_train, MSE_test, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, lamb, dependency)


        deg_poly = [i for i in range(1, poly+1)]

        min_pos = np.argmin(MSE_test)
        min_poly = deg_poly[min_pos]

        print("Min MSE: ", min(MSE_test))
        print("Optimal poly: ", min_poly)

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

        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test,
             mode='lines+markers',
             line=dict(width=4, color = "darkcyan"),
             marker=dict(size=9),
             name="MSE"))


        fig.update_layout(
             font_family="Garamond",
             font_size=33,
             title=f"Bias-variance tradeoff for terrain data using {reg_method} regression",
             xaxis_title="Degrees of polynomial",
             yaxis_title="",
             legend=dict(yanchor="top", xanchor="left", x=0.01, y=0.99)
             )
        fig.show()

    if part == "Ridge_lambda":

        #look at dependency on lambda
        dependency = "bias_variance"
        scaler = "minmax"
        poly = 10
        B_runs = 100
        reg_method = "RIDGE"
        dependency = "bias_variance"


        deg_poly = [i for i in range(1, poly+1)]

        #Plot MSE_test for 5 different lambdas
        np.random.seed(123)
        MSE_train0, MSE_test0, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 0, dependency)
        np.random.seed(123)
        MSE_train1, MSE_test1, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 0.00001, dependency)
        np.random.seed(123)
        MSE_train2, MSE_test2, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 0.001, dependency)
        np.random.seed(123)
        MSE_train3, MSE_test3, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 0.1, dependency)
        np.random.seed(123)
        MSE_train4, MSE_test4, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 10, dependency)


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
            name="Lambda = 1E-5"))

        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test2,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color = "firebrick"),
            marker=dict(size=9),
            name="Lambda = 1E-3"))

        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test3,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color = "green"),
            marker=dict(size=9),
            name="Lambda = 0.1"))

        fig.add_trace(go.Scatter(x=deg_poly, y=MSE_test4,
            mode='lines+markers',
            line=dict(dash='solid', width=4, color = "blue"),
            marker=dict(size=9),
            name="Lambda = 10"))


        fig.update_layout(
            font_family="Garamond",
            font_size=33,
            title=f"MSE as a function of complexity for {reg_method} regression",
            xaxis_title="Degrees of polynomial",
            yaxis_title="Mean Squared Error",
            legend=dict(yanchor="top", xanchor="left", x=0.2, y=0.99)
            )

        fig.show()



        np.random.seed(123)
        #Look at dependence on lambda for a given polynomial
        dependency = "lambda"
        poly = 7

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

        min_pos = np.argmin(MSE)
        min_lamb = LAMBDA[min_pos]

        print("Minimum MSE: ", min(MSE))
        print("Optimal lambda: ", min_lamb)


    if part == "Ridge_contour":
        #find the optimal combination of lambda and poly


        poly = 10
        reg_method = "Ridge"
        scaler = "minmax"
        #scaler = "none"
        k_fold = 0
        B_runs=100
        dependency = "bias_variance"


        lambdas = 10
        lambdas_ = np.logspace(-10,0,lambdas)

        #np.random.seed(123)
        mse = np.zeros((poly, lambdas))
        for p in range(poly):
            for l, lamb in enumerate(lambdas_):
                _, mse_temp , _, _ = Bootstrap(x_flat, y_flat, z_flat, scaler, p, B_runs, reg_method, lamb, dependency)
                mse[p, l] = np.mean(mse_temp)


        min_mse = np.where(mse == np.min(mse))

        min_p = int(min_mse[0])
        min_l = int(min_mse[1])

        print("Min poly: ", np.arange(1,poly+1)[min_p])
        print("Min lambda: ", lambdas_[min_l])
        print("Min MSE: ", np.min(mse))


        plt.figure(f"bootstrap", figsize=(11, 9), dpi=80)
        plt.plot(np.log10(lambdas_)[min_l], np.arange(1,poly+1)[min_p], "or", label="min", markersize=10)
        plt.contourf(np.log10(lambdas_), np.arange(1,poly+1), mse)
        plt.ylabel("degree",fontsize=14)
        plt.xlabel("log10(lambda)",fontsize=14)
        plt.colorbar()
        plt.legend()

        plt.show()








    if part == "plots":
        ThreeD_plot(x_mesh, y_mesh, z, "Terreng")


        # Show the terrain
        plt.figure()
        plt.title("Terrain over Norway 1")
        plt.imshow(terrain, cmap="gray")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()


    if part == "scaler":
        #First get an idea of which scaler to use by running a single run of OLS
        #Conclusion: MinMax gives the best results, so use this for the rest of the analysis

        lamb = 0
        poly = 15

        np.random.seed(123)


        X = design_matrix(x_flat, y_flat, poly)

        X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.2)


        scaler = "none"

        z_train_scaled, z_test_scaled, z_predict, z_model = OLS_Ridge(X_train, X_test, z_train, z_test, scaler, lamb, poly, "plot_prediction")

        print("Scaler: None")
        r2_train = r2_score(z_train_scaled, z_model)
        print(f"R2, train: {r2_train:.5}")
        r2_test = r2_score(z_test_scaled, z_predict)
        print(f"R2, test: {r2_test:.5}")


        mse_train = mean_squared_error(z_train_scaled, z_model)
        print(f"MSE, train: {mse_train:.5}")
        mse_test = mean_squared_error(z_test_scaled, z_predict)
        print(f"MSE, test: {mse_test:.5}")
        print('')


        scaler = "standard"

        z_train_scaled, z_test_scaled, z_predict, z_model = OLS_Ridge(X_train, X_test, z_train, z_test, scaler, lamb, poly, "plot_prediction")

        print("Scaler: Standard")
        r2_train = r2_score(z_train_scaled, z_model)
        print(f"R2, train: {r2_train:.5}")
        r2_test = r2_score(z_test_scaled, z_predict)
        print(f"R2, test: {r2_test:.5}")

        mse_train = mean_squared_error(z_train_scaled, z_model)
        print(f"MSE, train: {mse_train:.5}")
        mse_test = mean_squared_error(z_test_scaled, z_predict)
        print(f"MSE, test: {mse_test:.5}")
        print('')



        scaler = "mean"

        z_train_scaled, z_test_scaled, z_predict, z_model = OLS_Ridge(X_train, X_test, z_train, z_test, scaler, lamb, poly, "plot_prediction")

        print("Scaler: Mean")
        r2_train = r2_score(z_train_scaled, z_model)
        print(f"R2, train: {r2_train:.5}")
        r2_test = r2_score(z_test_scaled, z_predict)
        print(f"R2, test: {r2_test:.5}")

        mse_train = mean_squared_error(z_train_scaled, z_model)
        print(f"MSE, train: {mse_train:.5}")
        mse_test = mean_squared_error(z_test_scaled, z_predict)
        print(f"MSE, test: {mse_test:.5}")
        print('')


        scaler = "minmax"

        z_train_scaled, z_test_scaled, z_predict, z_model = OLS_Ridge(X_train, X_test, z_train, z_test, scaler, lamb, poly, "plot_prediction")

        print("Scaler: MinMax")
        r2_train = r2_score(z_train_scaled, z_model)
        print(f"R2, train: {r2_train:.5}")
        r2_test = r2_score(z_test_scaled, z_predict)
        print(f"R2, test: {r2_test:.5}")

        mse_train = mean_squared_error(z_train_scaled, z_model)
        print(f"MSE, train: {mse_train:.5}")
        mse_test = mean_squared_error(z_test_scaled, z_predict)
        print(f"MSE, test: {mse_test:.5}")
        print('')


        scaler = "robust"

        z_train_scaled, z_test_scaled, z_predict, z_model = OLS_Ridge(X_train, X_test, z_train, z_test, scaler, lamb, poly, "plot_prediction")

        print("Scaler: Robust")
        r2_train = r2_score(z_train_scaled, z_model)
        print(f"R2, train: {r2_train:.5}")
        r2_test = r2_score(z_test_scaled, z_predict)
        print(f"R2, test: {r2_test:.5}")

        mse_train = mean_squared_error(z_train_scaled, z_model)
        print(f"MSE, train: {mse_train:.5}")
        mse_test = mean_squared_error(z_test_scaled, z_predict)
        print(f"MSE, test: {mse_test:.5}")
        print('')








terrain("OLS_tradeoff")
