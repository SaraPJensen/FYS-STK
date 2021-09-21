# FYS-STK

Overleaf: https://www.overleaf.com/project/61434511c27ecb747adb108b
 
Pseudokode:

* OLS-funksjon: (x, y, z, string="scaler", lambda) #Gjør Ridge, hvis lambda =/= 0
	return z_train_scaled, z_test_scaled, z_predict, z_model

* Bootstrap-funksjon: (string="metode", b_runs, poly, x, y, z, #string="scaler")
	return mse, bias, variance

* xVal: (string="metode", k_fold, poly, x, y , z, #string="scaler")
	return mse, bias, variance

* Lasso: (x,y,x, string="scaler")
	return z_train_scaled, z_test_scaled, z_predict, z_model

*Scaling-funksjoner: (X_train, X_test, z_train, z_test)
	return skalerte inputs
	
* MSE

* R2
	
	
MAIN:
Exercise 2:
OPPGAVE 1: Løkke som kjører OLS flere ganger og genererer figur 2.11
OPPGAVE 2: Bootstrap av OLS-test-data, analyse MSE
Exercise 3:
OPPGAVE 1: xVal av OLS-test-data, analyse MSE
Exercise 4: 
OPPGAVE 1: Bootstrap med Ridge, for ulike grader av polynom
