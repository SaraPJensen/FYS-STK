from functions import *



#seed = 2018
#rng = np.random.default_rng(np.random.MT19937(seed=seed))

np.random.seed(2018)



def main(exercise, write_data = False):
    # Generate data
    n = 20
    noise = 0.05

    x = np.arange(0,1,1/n)
    y = np.arange(0,1,1/n)
    #x = np.random.rand(n)
    #y = np.random.rand(n)

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
        scaler = "none"
        lamb = 0

        X = design_matrix(x_flat, y_flat, poly)

        X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.2)

        #Plot the graph
        ThreeD_plot(x, y, z, "Franke function with low noise")

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
        plt.savefig(f"90CI_poly5_noise{noise}.png")

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
        poly = 25
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
        poly = 25
        k_fold = 5
        reg_method = "OLS"
        lamb = 0
        dependency = "bias_variance"
        B_runs = 100
        #seed = int(time())#2018
        seed = 123
        rng = np.random.default_rng(np.random.MT19937(seed=seed))

        #print("Starting CV, k=5")
        mse_cv = CrossVal(x_flat, y_flat, z_flat, scaler, poly, k_fold, reg_method, lamb, rng, dependency)
        #print("Starting CV, k=10")
        mse_cv_ = CrossVal(x_flat, y_flat, z_flat, scaler, poly, 10, reg_method, lamb, rng, dependency)
        #print("Starting Bootstrap")

        #mse_cv_2 = CrossVal(x_flat, y_flat, z_flat, scaler, poly, 200, reg_method, lamb, rng, dependency)
        np.random.seed(123)
        #perm = rng.permutation(np.arange(0, 400))

        MSE_train, MSE_test, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, lamb, dependency)

        '''
        #plt.plot(np.arange(0, poly+1), olss, label="ols")
        plt.plot(np.arange(0,poly+1), mse_cv, label="CV, kfold = 5")
        plt.plot(np.arange(0,poly+1), mse_cv_, label="CV, kfold = 10")
        plt.plot(np.arange(0,poly+1), mse_cv_2, label="CV, kfold = 200")
        #plt.plot(np.arange(0,poly+1), MSE_test, label="BS")
        plt.legend()
        plt.show()
        '''

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
            name=f"Bootstrapping, b_runs = {B_runs}"))

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

    elif exercise == 4.5:       # Ridge
        print("Comparison of the Ridge regession started.")

        scaler = "scalerNone"
        poly = 25
        k_fold = 5
        reg_method = "Ridge"
        dependency = "bias_variance"
        B_runs = 100
        #seed = int(time())#2018
        seed = 123
        rng = np.random.default_rng(np.random.MT19937(seed=seed))

        n_lambdas = 50
        ls = np.logspace(-5, 1, n_lambdas)
        
        cv5 = np.zeros((n_lambdas, poly+1))
        cv10 = np.zeros((n_lambdas, poly+1))
        boot = np.zeros((n_lambdas, poly+1))

        print("Completed necessary initializations.")

        for i, l in enumerate(ls):
            print(f"Iteration {i+1}/{n_lambdas}, lambda = {l}")
            #print("Starting CV, k=5")
            mse_cv5 = CrossVal(x_flat, y_flat, z_flat, scaler, poly, k_fold, reg_method, l, rng)
            #print("Starting CV, k=10")
            mse_cv10 = CrossVal(x_flat, y_flat, z_flat, scaler, poly, 10, reg_method, l, rng)
            #print("Starting Bootstrap")

            #mse_cv_2 = CrossVal(x_flat, y_flat, z_flat, scaler, poly, 200, reg_method, lamb, rng, dependency)
            np.random.seed(123)
            #perm = rng.permutation(np.arange(0, 400))

            MSE_train, MSE_test, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, l, dependency)

            cv5[i, :] = mse_cv5
            cv10[i, :] = mse_cv10
            boot[i, :] = MSE_test


        print("Lambda loop completed.")

        if write_data == True:
            with open("datafiles/current_params.txt", "w") as file:
                file.write(f"Max Poly: {poly}\nn_lambdas: {n_lambdas}\nBootstrap iterations: {B_runs}\nScaler: {scaler}\nRegression method: {reg_method}")
            np.savetxt("datafiles/ex4_cv5.csv", cv5, delimiter = ',')
            np.savetxt("datafiles/ex4_cv10.csv", cv10, delimiter = ',')
            np.savetxt("datafiles/ex4_boot.csv", boot, delimiter = ',')
            print("Datafiles written/overwritten.")
        '''
        #plt.plot(np.arange(0, poly+1), olss, label="ols")
        plt.plot(np.arange(0,poly+1), mse_cv, label="CV, kfold = 5")
        plt.plot(np.arange(0,poly+1), mse_cv_, label="CV, kfold = 10")
        plt.plot(np.arange(0,poly+1), mse_cv_2, label="CV, kfold = 200")
        #plt.plot(np.arange(0,poly+1), MSE_test, label="BS")
        plt.legend()
        plt.show()
        '''
        print("Task ended succesfully.")






    elif exercise == 5:

        #B_runs = 100
        #reg_method = "Lasso"
        #scaler = "none"
        #k_fold = 0

        print("Comparison of the Lasso regession started.")

        scaler = "scalerNone"
        poly = 25
        k_fold = 5
        reg_method = "Lasso"
        dependency = "bias_variance"
        B_runs = 100
        #seed = int(time())#2018
        seed = 123
        rng = np.random.default_rng(np.random.MT19937(seed=seed))

        n_lambdas = 50
        ls = np.logspace(-5, 1, n_lambdas)
        
        cv5 = np.zeros((n_lambdas, poly+1))
        cv10 = np.zeros((n_lambdas, poly+1))
        boot = np.zeros((n_lambdas, poly+1))

        print("Completed necessary initializations.")

        for i, l in enumerate(ls):
            print(f"Iteration {i+1}/{n_lambdas}, lambda = {l}")
            #print("Starting CV, k=5")
            mse_cv5 = CrossVal(x_flat, y_flat, z_flat, scaler, poly, k_fold, reg_method, l, rng)
            #print("Starting CV, k=10")
            mse_cv10 = CrossVal(x_flat, y_flat, z_flat, scaler, poly, 10, reg_method, l, rng)
            #print("Starting Bootstrap")

            #mse_cv_2 = CrossVal(x_flat, y_flat, z_flat, scaler, poly, 200, reg_method, lamb, rng, dependency)
            np.random.seed(123)
            #perm = rng.permutation(np.arange(0, 400))

            MSE_train, MSE_test, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, l, dependency)

            cv5[i, :] = mse_cv5
            cv10[i, :] = mse_cv10
            boot[i, :] = MSE_test


        print("Lambda loop completed.")

        if write_data == True:
            with open("datafiles/ex5_current_params.txt", "w") as file:
                file.write(f"Max Poly: {poly}\nn_lambdas: {n_lambdas}\nBootstrap iterations: {B_runs}\nScaler: {scaler}\nRegression method: {reg_method}")
            np.savetxt("datafiles/ex5_cv5.csv", cv5, delimiter = ',')
            np.savetxt("datafiles/ex5_cv10.csv", cv10, delimiter = ',')
            np.savetxt("datafiles/ex5_boot.csv", boot, delimiter = ',')
            print("Datafiles written/overwritten.")
        '''
        #plt.plot(np.arange(0, poly+1), olss, label="ols")
        plt.plot(np.arange(0,poly+1), mse_cv, label="CV, kfold = 5")
        plt.plot(np.arange(0,poly+1), mse_cv_, label="CV, kfold = 10")
        plt.plot(np.arange(0,poly+1), mse_cv_2, label="CV, kfold = 200")
        #plt.plot(np.arange(0,poly+1), MSE_test, label="BS")
        plt.legend()
        plt.show()
        '''
        print("Task ended succesfully.")


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
        '''

    elif exercise == "plot":
        
        for ex in ["ex4", "ex5"]:
            cv5 = np.loadtxt(f"datafiles/{ex}_cv5.csv", delimiter = ',')
            cv10 = np.loadtxt(f"datafiles/{ex}_cv10.csv", delimiter = ',')
            boot = np.loadtxt(f"datafiles/{ex}_boot.csv", delimiter = ',')

            for method in ["cv5", "cv10", "boot"]:
                result = eval(method)
                minarg = np.argmin(result)
                minarg = np.unravel_index(minarg, result.shape)
                #print(minarg)
                min_val = result[minarg]

                print(f"Minimum {method}, poly : {minarg[1]}, lambda : {minarg[0]}")
                print("min value: ", min_val)

                x_ax = np.linspace(0, 25, result.shape[1])
                y_ax = np.linspace(-5, 1, result.shape[0])

                plt.scatter(x_ax[minarg[1]], y_ax[minarg[0]], c='r', zorder = 5, label = f"Min MSE = {min_val:e}")
                plt.pcolormesh(x_ax, y_ax, result)
                #plt.contourf(result)       # Nicer looking, but less informative
                plt.colorbar() 

                if ex == "ex4":
                    reg = "Ridge regression"
                if ex == "ex5":
                    reg = "Lasso regression"

                if method == "cv5":
                    lalala = "Crossvalidation k = 5"
                if method == "cv10":
                    lalala = "Crossvalidation k = 10"
                if method == "boot":
                    lalala = "Bootstrap B = 100"
                plt.title(f"Mean Squared Error values\n{reg}, {lalala}")
                plt.legend()
                plt.xlabel("Polynomial degree")
                plt.ylabel("$\log{\lambda}$")
                #plt.yticks(y_ticks)

                plt.savefig(f"datafiles/{ex}{method}compare.png")
                plt.show()


main("plot", write_data = False)

def terrain(part):

    # Load the terrain
    terrain1 = imread("SRTM_data_Norway_1.tif")
    #Dimensions of entire image: 3601 x 1801

    N = 100
    start = 2200
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


        deg_poly = [i for i in range(5, poly+1)]

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
             legend=dict(yanchor="top", xanchor="left", x=0.85, y=0.99)
             )
        fig.show()

    if part == "Ridge_lambda":

        #look at dependency on lambda
        dependency = "bias_variance"
        scaler = "minmax"
        poly = 20
        B_runs = 10
        reg_method = "Ridge"
        dependency = "bias_variance"


        '''
        deg_poly = [i for i in range(5, poly+1)]

        #Plot MSE_test for 5 different lambdas
        np.random.seed(123)
        MSE_train0, MSE_test0, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 0, dependency)
        print("0")
        np.random.seed(123)
        MSE_train1, MSE_test1, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 0.00001, dependency)
        print("0.000001")
        np.random.seed(123)
        MSE_train2, MSE_test2, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 0.001, dependency)
        print("0.001")
        np.random.seed(123)
        MSE_train3, MSE_test3, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 0.1, dependency)
        print("0.1")
        np.random.seed(123)
        MSE_train4, MSE_test4, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 10, dependency)
        print("10")


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
            legend=dict(yanchor="top", xanchor="left", x=0.01, y=0.99)
            )

        fig.show()

        '''

        np.random.seed(123)
        lamb = 0
        #Look at dependence on lambda for a given polynomial
        dependency = "lambda"
        poly = 25

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

        poly = 15
        reg_method = "Ridge"
        scaler = "minmax"
        #scaler = "none"
        k_fold = 0
        B_runs=100
        dependency = "bias_variance"


        lambdas = 10
        lambdas_ = np.logspace(-7,2,lambdas)

        #np.random.seed(123)
        mse = np.zeros((poly, lambdas))
        for p in range(poly):
            print("Poly: ", p)
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


    if part == "Lasso":

        B_runs = 100
        reg_method = "Lasso"
        scaler = "minmax"
        k_fold = 0



        #Plot MSE_test for 5 different lambdas
        dependency = "bias_variance"

        poly = 20

        deg_poly = [i for i in range(5, poly+1)]

        np.random.seed(123)
        MSE_train0, MSE_test0, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, "OLS", 0, dependency)
        print("0")
        np.random.seed(123)
        MSE_train1, MSE_test1, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 0.000001, dependency)
        print("1E-6")
        np.random.seed(123)
        MSE_train2, MSE_test2, Bias, Variance = Bootstrap(x_flat, y_flat, z_flat, scaler, poly, B_runs, reg_method, 0.0001, dependency)
        print("1E-4")
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






    if part == "plots":

        ThreeD_plot(x_mesh, y_mesh, z, "Terrain")

        # Show the terrain
        plt.figure()
        plt.title("Terrain over Norway 1")
        plt.imshow(terrain, cmap="gray")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.zlabel("z")
        plt.legend()
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





#terrain("Ridge_lambda")
