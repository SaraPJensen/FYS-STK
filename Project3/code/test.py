import numpy as np

x = np.linspace(0, 10, 11)
t = np.linspace(0, -10, 11)

x, t = np.meshgrid(x, t)

#print(x)

'''
def design_matrix(poly):
    l = int((poly+1)*(poly+2)/2)		# Number of features
    X = np.ones(l)

    for i in range(0, poly+1):
        q = int((i)*(i+1)/2)

        for k in range(i+1):
            X[q+k] = eval(f"(x**{i-k})*(y**{k})")
    return X

print(design_matrix(1))
'''

#print(np.random.uniform(-1, 1))


a = "word"
b = 9.8

print(type(a))
