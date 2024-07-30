import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.kernel_ridge import KernelRidge
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel

import cvxopt
from cvxopt import solvers
from cvxopt import matrix
from matplotlib import style
style.use("ggplot")
from numpy.random import randn # Gaussian random numbers
from scipy.stats import norm

###########################################################
#                    Initialization                       #
###########################################################

#- Parameters
lam = 0.5
n = 50
x = np.random.randn(n)
y = 1 + 0.5*x - 1.5*x**2 +0.25*x**3+0.5*x**4 + np.random.randn ( n ) #


def alpha (k,x,y) :
    n = len(x)
    K = np.zeros((n,n))
    for i in range(n) :
        for j in range(n) :
            K[i,j] = k( x[i] , x[j] )
    return np.linalg.inv(K + lam * np.identity(n)).dot(y)

#- Kernels Definition
def k_p (x,y) :
    return (np.dot ( x, y ) + 1)**3
def k_g (x,y) :
    return np.exp(-(x - y )**2 / 2)
def polynomial_kernel(x, y, degree=3, coef0=1):
    return (np.dot(x, y) + coef0)**degree


alpha_p = alpha ( k_p , x , y )
alpha_g = alpha ( k_g , x , y )
kernel = ExpSineSquared(length_scale=1.0, periodicity=3.0)
kernel2 = 1.0*ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(1e-1)


#- Fitting
z = np.sort(x) ; u = [ ] ; v = [ ]
x_reshaped = x.reshape(-1, 1)
z_reshaped = z.reshape(-1, 1)

#
krr = KernelRidge(kernel=k_p, alpha=1.0)
krr.fit(x_reshaped, y)

#
gaussian_process = GaussianProcessRegressor(kernel=kernel)
gaussian_process.fit(x_reshaped, y)

#
param_distributions = {"alpha": loguniform(1e0, 1e3),
    #"kernel__length_scale": loguniform(1e-2, 1e2),
    #"kernel__periodicity": loguniform(1e0, 1e1),
    }
krr_tuned = RandomizedSearchCV(krr,param_distributions=param_distributions,
                               n_iter=500,random_state=0,)
krr_tuned.fit(x_reshaped, y)

#
for j in range ( n ) :
    S = 0
    for i in range ( n ) :
        S = S + alpha_p [i] * k_p( x[i] , z[j] )
    u . append (S)
    S = 0
    for i in range ( n ) :
        S = S + alpha_g [i] * k_g( x[i] , z[j] )
    v . append (S)

###########################################################
#                    Ploting                              #
###########################################################
plt.scatter (x , y , facecolors='black' , edgecolors = "k" , marker = "o")
plt.plot( z , u , c = "r" , label = "Polynomial")
plt.plot( z , v , c = "r" , label = "Gaussian",linestyle="dashdot")
plt.plot(z, krr.predict(z_reshaped), c = "b" , label="sklearn krr", linestyle="dashdot",)
plt.plot(z, krr_tuned.predict(z_reshaped), c = "g" , label="sklearn krr tuned")

# mean_predictions_gpr, std_predictions_gpr = gaussian_process.predict(
#     z_reshaped,return_std=True,)
# plt.plot(
#     z,
#     mean_predictions_gpr,
#     c = "g" ,
#     label="Gaussian process regressor",
#     linewidth=2,
#     linestyle="dotted",
# )
# plt.fill_between(
#     z.ravel(),
#     mean_predictions_gpr - std_predictions_gpr,
#     mean_predictions_gpr + std_predictions_gpr,
#     color="tab:green",
#     alpha=0.2,
# )
plt.xlim(-1 , 1)
plt.ylim(-1 , 4)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Kernels Regression")
plt.legend ( loc = "upper left" , frameon = True , prop={'size': 14 } )
#plt.savefig("plot_lambda={}_.png".format(lam))
plt.savefig("plot.png")