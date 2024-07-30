import calendar
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime, timedelta
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, RBF, WhiteKernel, Sum
from sklearn.linear_model import Ridge
import warnings
from sklearn.kernel_approximation import PolynomialCountSketch
import multiprocessing as mp
#color=[RESET, RED, GREEN, BLUE, CYAN BOLD]
color=["\033[0;0m", "\033[1;31m" , "\033[0;32m",  "\033[1;34m", "\033[1;36m", "\033[;1m"] 

import time

import numpy as np
from scipy.optimize import minimize




def read_csv_to_matrix(file_path,column_index):
    # Initialize an empty matrix to store the CSV data
    matrix = []
    
    # Open the CSV file
    with open(file_path, mode='r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)
        # Iterate over each row in the CSV file
        for row in csv_reader:
            if len(column_index)!=0:
                temp=[]
                for i in column_index:
                    temp.append(row[i-1])
                matrix.append(temp)
            else:
                
                matrix.append(row)
    matrix = np.matrix(matrix)
    matrix=convert(matrix)
    print(color[3]+"o "+color[0]+"read_csv_to_matrix(): "+file_path+" has been transformed into a matrix\n")
    return matrix

def safe_convert(x):
    try:
        return float(x)
    except ValueError:
        return x  # or any other value indicating a failed conversion
convert=np.vectorize(safe_convert, otypes=[object])

############################################################
#                   CSV to basic matrices
############################################################

# Path to the CSV file
file_path = 'data/GDPNow_inputs.csv'
file_path2 = 'data/GDP&GDPplus.csv'
file_path3 = 'data/GDPNow.csv'
C1=[1,2,3,5]
C2=[1,2,28]
#C=[]
INPUT=read_csv_to_matrix(file_path,[])
GDPdata=read_csv_to_matrix(file_path2,C1)
GDP_now=read_csv_to_matrix(file_path3,C2)

############################################################
#                   Matrices
############################################################

#- timeline & dates
timeline=[]
timelineflt=[]
current_date = datetime.strptime('1960-01-31', '%Y-%m-%d')
t=1960+1/12
end_date = datetime.strptime('2023-12-31', '%Y-%m-%d')
while current_date <= end_date:
    timelineflt.append(t)
    timeline.append(current_date.strftime('%Y-%m-%d'))  # Convertir la date en string avant de l'ajouter
    
    days_in_month = calendar.monthrange(current_date.year, current_date.month)[1]
    current_date = current_date.replace(day=days_in_month) + timedelta(days=1)
    next_month = current_date.replace(day=28) + timedelta(days=4)
    current_date = next_month - timedelta(days=next_month.day)
    
    t+=1/12
N=len(timeline)

#- GDP
GDP=[]
for x in GDPdata[1:,2].A1:
    for k in range(3):
        GDP.append(x)

#- GDP Plus
GDPplus=[]
for x in GDPdata[1:,3].A1:
    for k in range(3):
        GDPplus.append(x)

#- GDP Now
GDPnow = []
dates = np.array([datetime.strptime(date.item(), '%Y-%m-%d') for date in GDP_now[1:, 0]])
def find_closest_date(target_date, dates):
    target_date = datetime.strptime(target_date, '%Y-%m-%d')
    same_month_dates = [date for date in dates if date.year == target_date.year and date.month == target_date.month]
    closest_date = min(same_month_dates, key=lambda date: abs(date - target_date))
    return closest_date
for end_of_month in timeline[timeline.index('2014-05-31'):]:
    closest_date = find_closest_date(end_of_month, dates)
    index = np.where(dates == closest_date)[0][0] + 1  # Ajustement pour l'index de la matrice GDP_now
    GDPnow.append(GDP_now[index, 2])

#- GDP3
GDP3=np.empty((N,3), dtype=object)
GDP3[:,0]=GDP
GDP3[:,1]=GDPplus
GDP3[N-len(GDPnow):,2]=GDPnow
# we want to predict starting 2014 => we train the model until 2014
n=timeline.index('2014-07-31')


#-Data Filter
INPUT=INPUT[1:,1:]

variable_cut_off=timeline.index('1970-01-31')
NBinit=INPUT.shape[1]
good_index=[]
for j in range(INPUT.shape[1]):
    if not '' in INPUT[variable_cut_off:, j]:
        good_index.append(j)
INPUT=INPUT[:,good_index]

for i in range(INPUT.shape[0]-1,0,-1):
    if '' in INPUT[i, :]:
        break
print(color[2]+"o "+color[0]+color[-1]+"Data Filter:"+color[0]+f" {NBinit} -> {len(good_index)} variables\n  1st_time={timeline[0]} -> {timeline[i+1]}\n")
   
INPUT=INPUT[i+1:,:]
timelineflt=timelineflt[i+1:]
timeline=timeline[i+1:]
GDP=GDP[i+1:]
INPUT = np.asarray(INPUT, dtype=float)

n=timeline.index('2014-07-31')


#- Normalization

EXTREM=[max(abs(INPUT[:,j])) for j in range(INPUT.shape[1])]
for i in range(INPUT.shape[0]):
    for j in range(INPUT.shape[1]):
        INPUT[i,j]=INPUT[i,j]/EXTREM[j]

#-Model for X,Y
X=INPUT[:n]
Y=np.asarray(GDP[:n], dtype=float)

############################################################
#                   Manual Regression
############################################################
def Alpha (k,x,y,lam) :
    n = len(x)
    K = np.zeros((n,n))
    for i in range(n) :
        for j in range(n) :
            K[i,j] = k( x[i] , x[j] )
    return np.linalg.inv(K + lam * np.identity(n)).dot(y)

#- Kernel Model



L1=1
lam1 = 0.01

L=0.3
lam2 = 10 #

#- Kernels Definition
def k_p (x,y) :
    return (np.dot ( x, y )+ 1)**3
def k_g (x,y) :
    return np.exp(-np.dot ( x-y, x-y))/np.sqrt(2*np.pi)
def k_e (x,y) :
    return 3/2*(1-4*(np.dot( x,y ))**2)
def k_ESQ (x,y) :
    return np.exp(L*np.sin(np.dot ( x,y)))/np.sqrt(2*np.pi)
def k_ESQ1 (x,y) :
    return np.exp(L1*np.sin(np.dot ( x, y)))/np.sqrt(2*np.pi)
print(color[-1]+"o "+color[0]+"Manual regressions have started", end='\r')


# print(color[4]+"o "+color[0]+"Linear regression done\n")
alpha_e = Alpha( k_e , X,Y,lam2)
alpha_esq1 = Alpha( k_ESQ1 , X,Y,lam1)
alpha_p = Alpha( k_p , X,Y,lam2 )
alpha_esq = Alpha( k_ESQ , X,Y,lam2 )
alpha_g = Alpha( k_g , X,Y,lam1 )
e, g, p, esq1, esq2= [], [], [], [], []
for j in range(len(INPUT)):
    S_e, S_g, S_p, S_esq1, S_esq2 = 0, 0, 0, 0, 0
    for i in range(n):
        S_e += alpha_e[i] * k_e(X[i], INPUT[j])
        S_g += alpha_g[i] * k_g(X[i], INPUT[j])
        S_p += alpha_p[i] * k_p(X[i], INPUT[j])
        S_esq2 += alpha_esq[i] * k_ESQ(X[i], INPUT[j])
        S_esq1 += alpha_esq1[i] * k_ESQ1(X[i], INPUT[j])
    e.append(S_e)
    g.append(S_g)
    p.append(S_p)
    esq2.append(S_esq2)
    esq1.append(S_esq1)


def perf(f,nb):
    R=[]
    P=[]
    SMAPE=[]
    P2=[]
    R2=[]
    for x in f:
        Rx=0
        Dx=0
        Dx2=0
        for i in range(n):
            Rx+=(x[i]-GDP[i])**2
            Dx+=(np.mean(GDP[:n])-GDP[i])**2
            Dx2+=(np.mean(x[n:n+nb])-GDP[i])**2
        temp=Rx/Dx
        temp=int(temp*1000)/1000
        R.append(temp)
        
        temp=Rx/Dx2
        temp=int(temp*1000)/1000
        R2.append(temp)
        
        Rx=0
        Dx=0
        Dx2=0
        div=0
        for i in range(n,n+nb):
            Rx+=(x[i]-GDP[i])**2
            Dx+=(np.mean(GDP[n:n+nb])-GDP[i])**2
            div+=(x[i]-GDP[i])**2/((abs(x[i])+abs(GDP[i]))/2)
            Dx2+=(np.mean(x[n:n+nb])-GDP[i])**2
        temp=Rx/Dx
        temp=int(temp*1000)/1000
        P.append(temp)
        
        temp=Rx/Dx2
        temp=int(temp*1000)/1000
        P2.append(temp)
        
        div=int((100/n)*div*1000)/1000
        SMAPE.append(div)
    return (R,P,R2,P2,SMAPE)

PREDICTION=[e,p,g, esq1, esq2]
e=np.matrix(e).transpose()
g=np.matrix(g).transpose()
p=np.matrix(p).transpose()
esq2=np.matrix(esq2).transpose()
esq1=np.matrix(esq1).transpose()

print(color[4]+"o "+color[0]+"Manual regression done          \n")



#########################################################################################################################
#                   KERNEL LEARNING
#########################################################################################################################
def squared_relative_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2 / y_true ** 2)
# Define the custom kernel function /w numpy
def custom_kernel_np(X, Y, theta, alpha):
    result = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            x = X[i]
            y = Y[j]
            norm_sq = np.linalg.norm(x - y) ** 2
            norm = np.linalg.norm(x - y)
            
            # Compute the components
            term1 = alpha[0] ** 2 * (np.dot(x, y) + theta[0] ** 2)
            term2 = alpha[1] ** 2 * (theta[1] ** 2 * np.dot(x, y) + theta[2] ** 2) ** abs(theta[3])
            term3 = alpha[2] ** 2 * np.exp(-norm_sq / (2 * theta[4] ** 2))
            term4 = alpha[3] ** 2 * np.exp(-norm / (2 * theta[5] ** 2))
            term5 = alpha[4] ** 2 * np.exp(-np.sin(np.pi * norm_sq / theta[6]) ** 2 / theta[7] ** 2) * np.exp(-norm_sq / theta[8] ** 2)
            term6 = alpha[5] ** 2 * np.exp(-np.sin(np.pi * norm_sq / theta[9]) ** 2 / theta[10] ** 2)
            term7 = alpha[6] ** 2 * np.exp(-np.sin(np.pi * norm / theta[11]) ** 2 / theta[12] ** 2) * np.exp(-norm / theta[13] ** 2)
            term8 = alpha[7] ** 2 * np.exp(-np.sin(np.pi * norm / theta[14]) ** 2 / theta[15] ** 2)
            term9 = alpha[8] ** 2 * (norm_sq + theta[16] ** 2) ** 0.5
            term10 = alpha[9] ** 2 * (theta[17] ** 2 + theta[18] ** 2 * norm_sq) ** -0.5
            term11 = alpha[10] ** 2 * (theta[19] ** 2 + theta[20] ** 2 * norm) ** -0.5
            term12 = alpha[11] ** 2 * (theta[21] ** 2 + norm) ** theta[22]
            term13 = alpha[12] ** 2 * (theta[23] ** 2 + norm_sq) ** theta[24]
            term14 = 0#alpha[13] ** 2 * (1 + (norm / theta[25]) ** -1) ** -1
            term15 = alpha[14] ** 2 * (1 + norm / theta[26] ** 2) ** -1
            term16 = alpha[15] ** 2 * (1 - norm_sq / (norm_sq + theta[27] ** 2))
            term17 = alpha[16] ** 2 * np.fmax(0, 1 - norm_sq / theta[28] ** 2)
            term18 = alpha[17] ** 2 * np.fmax(0, 1 - norm / theta[29] ** 2)
            term19 = alpha[18] ** 2 * np.log(norm ** theta[30] + 1)
            term20 = alpha[19] ** 2 * np.tanh(theta[31] * np.dot(x, y) + theta[32])
            term22 = alpha[21] ** 2 * np.exp(np.sin(np.dot(x,y))/theta[34] ** 2)/np.sqrt(2*np.pi)
            # Compute the term with the indicator function
            acos_argument = norm / theta[33] ** 2
            if acos_argument < 1:  # Indicator function condition
                term21 = alpha[20] ** 2 * (np.arccos(-acos_argument) - acos_argument * np.sqrt(1 - acos_argument ** 2))
            else:
                term21 = 0
                
            result[i, j] = (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + 
                     term11 + term12 + term13 + term14 + term15 + term16 + term17 + term18 + term19 + 
                     term20 + term21 + term22)
    return result
# Objective function for optimization /w numpy
def objective_np(theta, X, Y, alpha, lambda1, lambda2):
    K = custom_kernel_np(X, X, theta, alpha)
    # return mean_squared_error(Y, np.linalg.inv(K+lambda1*np.identity(X.shape[0])).dot(Y)) + lambda2 * np.linalg.norm(alpha, 1)
    K_inv_Y = np.linalg.inv(K + lambda1 * np.identity(X.shape[0])).dot(Y)
    error = squared_relative_error(Y, K_inv_Y)
    regularization = lambda2 * np.linalg.norm(alpha, 1)
    return error + regularization
def objectiveA_np(alpha, X, Y,theta, lambda1, lambda2):
    return objective_np(theta, X, Y, alpha, lambda1, lambda2)
# Sparse Kernel Flows algorithm /w numpy
def sparse_kernel_flows_np(it,X, Y, tau, alpha_init, theta_init, lambda1, lambda2):
    Nskf = X.shape[0] - tau
    X_tau = X[:Nskf]
    Y_tau = Y[tau:Nskf+tau]
    
    alpha = alpha_init
    theta = theta_init
    for _ in range(it):  # Iterate to optimize alpha and theta
        # Fix alpha, optimize theta
        start_time = time.perf_counter()
        res_theta = minimize(objective_np, theta, args=(X_tau, Y_tau, alpha, lambda1, lambda2), method='trust-constr')
        execution_time1 = time.perf_counter() - start_time
        
        print(color[3]+f"evolution of theta = {res_theta.x-theta}"+color[0])
        theta = res_theta.x
        # Fix theta, optimize alpha
        res_alpha = minimize(objectiveA_np, alpha, args=(X_tau, Y_tau, theta, lambda1, lambda2), method='trust-constr')
        execution_time2 = time.perf_counter() - start_time
        print(color[3]+f"evolution of alpha = {res_alpha.x-alpha}"+color[0])
        alpha=res_alpha.x
        # K = custom_kernel(X_tau, X_tau, theta, alpha)
        # alpha = np.linalg.lstsq(K, Y_tau, rcond=None)[0]
        # alpha = np.sign(alpha) * np.maximum(0, np.abs(alpha) - lambda2)
        if _ == 0:
            print(color[0]+f"Time measurments: 1 opti = {int(execution_time1/60)} min -> 1it = {int(execution_time2/60)} min"+color[1]+f"\ntotal time prediction : {int(execution_time2*it/3600)}h\n"+color[0])
        print(f"APRES: {objective_np(theta_init, X_tau, Y_tau,alpha_init, lambda1, lambda2)}->{objective_np(theta, X_tau, Y_tau,res_alpha.x, lambda1, lambda2)} |alpha={alpha}")
        print(color[2]+"> "+f"{100*(_+1)/it} %    "+color[0])
        
    return alpha, theta
def prediction(X,Y,alpha,theta, lambda1,INPUT):
    K_test = custom_kernel_np(INPUT, X, theta, alpha)
    ALPHA=np.linalg.inv(custom_kernel_np(X, X, theta, alpha)+lambda1*np.identity(X.shape[0])).dot(Y)
    return K_test.dot(ALPHA)



lambda1 = 1
lambda2 = 1e-7
tau=3
alpha_init=np.random.uniform(0, 1, 22) ; alpha_init[13]=0
theta_init = np.ones(35) ; theta_init[6]=0.147 ; theta_init[7]= 0.0053/np.sqrt(2); theta_init[8]=51856/np.sqrt(2)
theta_init[1]=np.sqrt(3.9) ;theta_init[2]= np.sqrt(4.27) ;theta_init[3]= 3.1
theta_init[10]= 0.0598 ;theta_init[9]= np.sqrt(0.91) ; theta_init[34]=np.sqrt(0.1)

alpha, theta = sparse_kernel_flows_np(1,X, Y, tau, alpha_init, theta_init, lambda1, lambda2)
Y_prediction=prediction(X,Y,alpha, theta, lambda1,INPUT)

PREDICTION.append(Y_prediction)
(R,P,R2,P2,SMAPE)=perf(PREDICTION,6)
print(color[2]+f"{R}"+color[0])
print(color[2]+f"{P}"+color[0])
print(color[2]+f"{SMAPE}"+color[0])
############################################################
#                   Sklearn KR Regression
############################################################

# #- Linear Model
# model = LinearRegression()
# model.fit(X,Y)
# Y_pred = model.predict(INPUT)
# print(color[4]+"o "+color[0]+"Linear regression done\n")

# # warnings.filterwarnings('ignore')

# #- Kernel Ridge Model
# print(color[-1]+"o "+color[0]+"KR Poly regression has started", end='\r')
# kernel_ridge = KernelRidge(kernel="polynomial",alpha=16.2,coef0=6.65,degree= 3.7,gamma=1.18)#
# # kernel_ridge = KernelRidge(kernel="polynomial",alpha=10,coef0=1,degree= 3,gamma=1)#
# kernel_ridge.fit(X,Y)
# Y_pred_3 = kernel_ridge.predict(INPUT)
# print(color[4]+"o "+color[0]+"KR Poly regression done        \n")

# #- Kernel Ridge Tuned Model
# print(color[-1]+"o "+color[0]+"KR Poly tuning has started", end='\r')
# alpha_distribution = {"alpha": np.arange(1, 20,0.1)}
# param_distributions2 = {
#     "alpha": np.arange(1, 30,0.1),
#     "degree": np.arange(1, 4, 0.01),  # Degree of the polynomial kernel
#     "coef0": np.arange(0, 25,0.05),  # Independent term in polynomial kernel
#     "gamma": np.arange(0.2, 10,0.01),  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
# }#{'gamma': 0.35000000000000014, 'degree': 1.01, 'coef0': 1.4000000000000001, 'alpha': 21.40000000000002}
# kernel_ridge_tuned =RandomizedSearchCV(
#     KernelRidge(kernel="polynomial",gamma= 0.35,
#                         degree=1.01, coef0=1.4, alpha=21.4),
#     param_distributions=param_distributions2,
#     n_iter=500,
#     random_state=0,)
# kr = GridSearchCV(
#     KernelRidge(kernel="polynomial"),
#     param_grid=param_distributions2,
# )

# # # kernel_ridge_tuned= KernelRidge(kernel="polynomial",gamma= 0.35,
# # #                         degree=1.01, coef0=1.4, alpha=21.4)
# kernel_ridge_tuned.fit(X,Y)
# Y_pred_2 = kernel_ridge_tuned.predict(INPUT)
# print(kernel_ridge_tuned.best_params_)#{'gamma': 0.0, 'degree': 2.0, 'coef0': 6.550000000000001, 'alpha': 10.0}
# print(color[4]+"o "+color[0]+"KR Poly tuned done               \n")

# #- Tuned ESQ Kernel Model
# print(color[-1]+"o "+color[0]+"KR ESQ regression has started", end='\r')
# kernel = Sum(ExpSineSquared(3.2, 0.76), WhiteKernel(noise_level=0.1))
# kr = KernelRidge(kernel=kernel,alpha=11.6)#
# alpha_distribution = {"alpha": np.arange(5, 20,0.1)}
# param_distributions1 = {
#     "alpha": np.arange(1, 20,0.1) ,
#     "kernel__k1__length_scale": np.arange(0.5, 10,0.01),
#     "kernel__k1__periodicity": np.arange(0.5, 10,0.01),
# }
# kr_tuned = RandomizedSearchCV(kr,
#     param_distributions=param_distributions1,n_iter=500,random_state=0,)
# # kr_tuned=kr
# kr_tuned.fit(X,Y)

# print(kr_tuned.best_params_)
#{'gamma': 0.35000000000000014, 'degree': 1.01, 'coef0': 1.4000000000000001, 'alpha': 21.40000000000002}

# Y_pred_KR = kr_tuned.predict(INPUT)

# # kr.fit(X,Y)
# # Y_pred_gauss = kr.predict(INPUT)
# print(color[4]+"o "+color[0]+"KR ESQ regression done              \n")

# #- Tuned ESQ Kernel Model
# print(color[-1]+"o "+color[0]+"KR ESQ tuning has started", end='\r')
# kernel2 = Sum(ExpSineSquared(3.2,0.76),WhiteKernel(noise_level=0.1))
# kr = KernelRidge(kernel=kernel2,alpha=11.9)#
# param_distributions = {
#     "alpha": loguniform(0.1, 1e2) ,
#     "kernel__k1__length_scale": loguniform(1e-5, 1e5),  # length_scale du premier sous-noyau (ExpSineSquared)
#     "kernel__k1__periodicity": loguniform(1e-5, 1e5),  # periodicity du premier sous-noyau (ExpSineSquared)
#     "kernel__k2__noise_level": loguniform(1e-2, 1)  # length_scale du deuxième sous-noyau (RBF)
# }
# param_distributions2 = {
#     "alpha": loguniform(0.1, 50) ,
#     "kernel__k1__length_scale": loguniform(1, 50),  # length_scale du premier sous-noyau (ExpSineSquared)
#     "kernel__k1__periodicity": loguniform(1, 1e3),  # periodicity du premier sous-noyau (ExpSineSquared)
#     }
# k_tuned = RandomizedSearchCV(kr,
#     param_distributions=param_distributions,n_iter=500,random_state=0,)
# # k_tuned =  KernelRidge(kernel=kernel,alpha=30)#1.0188485733850716
# k_tuned=kr
# k_tuned.fit(X,Y)
# print(k_tuned.best_params_)#{'alpha': 2.2581338402646867, 'kernel__k1__length_scale': 14.263761208061407, 'kernel__k1__periodicity': 6108.6096689612405, 'kernel__k2__noise_level': 0.10312825941082275}

# Y_pred_KR_tuned = k_tuned.predict(INPUT)
# print(color[4]+"o "+color[0]+"KR ESQ tuned regression done\n")

############################################################
#                   Sklearn Gaussian Regression
############################################################

# #- Tuned Gaussian Kernel Model
# kernel2 = Sum(ExpSineSquared(14.2637612, 6.6), WhiteKernel(noise_level=0.10312825941082275))
# gaussian_process = GaussianProcessRegressor(kernel=kernel,alpha=11.6)#
# alpha_distribution = {"alpha": np.arange(30, 50,0.1)}
# # param_distributions = {
# #     "alpha": loguniform(1, 1e3) ,
# #     "kernel__length_scale": loguniform(1e0, 1e1),
# #     "kernel__periodicity": loguniform(1e0, 1e2),
# # }
# gaussian_process_tuned = RandomizedSearchCV(gaussian_process,
#     param_distributions=param_distributions1,n_iter=500,random_state=0,)
# gaussian_process_tuned.fit(X,Y)

# print(gaussian_process_tuned.best_params_)
#{'kernel__k1__periodicity': 0.7600000000000002, 'kernel__k1__length_scale': 3.2000000000000024, 'alpha': 11.600000000000009}

# Y_pred_gauss = gaussian_process_tuned.predict(INPUT)

# gaussian_process.fit(X,Y)
# Y_pred_gauss = gaussian_process.predict(INPUT)
# print(color[4]+"o "+color[0]+"GaussianProcess regression done\n")

# #- Tuned Gaussian Kernel Model
# kernel2 = Sum(ExpSineSquared(46.1,627.9384),WhiteKernel(noise_level=0.10312825941082275))
# gaussian_process = GaussianProcessRegressor(kernel=kernel2,alpha=0.5)#
# # param_distributions = {
# #     "alpha": loguniform(0.1, 1e3) ,
# #     "kernel__k1__length_scale": loguniform(1e-5, 1e5),  # length_scale du premier sous-noyau (ExpSineSquared)
# #     "kernel__k1__periodicity": loguniform(1e-5, 1e5),  # periodicity du premier sous-noyau (ExpSineSquared)
# #     "kernel__k2__noise_level": loguniform(1e-2, 1)  # length_scale du deuxième sous-noyau (RBF)
# # }
# # param_distributions2 = {
# #     "alpha": np.arange(0.1, 20,0.1) ,
# #     "kernel__k1__length_scale": np.arange(1, 50,0.1),  # length_scale du premier sous-noyau (ExpSineSquared)
# #     "kernel__k1__periodicity": loguniform(1, 1e4),  # periodicity du premier sous-noyau (ExpSineSquared)
# #     }
# # k_tuned = RandomizedSearchCV(gaussian_process,
# #     param_distributions=param_distributions2,n_iter=100,random_state=0,)
# # k_tuned =  GaussianProcessRegressor(kernel=kernel,alpha=30)#1.0188485733850716
# k_tuned=gaussian_process
# k_tuned.fit(X,Y)
# # print(k_tuned.best_params_)#{'alpha': 2.2581338402646867, 'kernel__k1__length_scale': 14.263761208061407, 'kernel__k1__periodicity': 6108.6096689612405, 'kernel__k2__noise_level': 0.10312825941082275}
# print(color[4]+"o "+color[0]+"Gaussian tuned regression done\n")
# Y_pred_gauss_tuned = k_tuned.predict(INPUT)

# # krr = KernelRidge(kernel=k_p, alpha=1.0)
# # krr.fit(X,Y)
# # Y_pred_k = k_tuned.predict(X)

############################################################
#                   Prints & Plots
############################################################


plt.plot(timelineflt, GDP, color='grey', label='Real GDP',linewidth=3)

COLOR=["cyan","b","darkblue","y","g","darkgreen","orange","magenta","r","darkmagenta","darkred","chocolate","black","gray"]
CCC=range(len(PREDICTION))
Nom=["MKR epanechnikov","MKR polynomial","MKR Gaussian","MKR ExpSin²","MKR ExpSin²","Kernel flows"]
for tau in CCC:
    plt.plot(timelineflt[:n], (PREDICTION[tau])[:n], color=COLOR[tau], label=f"{Nom[tau]}",linewidth=1)
    plt.plot(timelineflt[n-1:], (PREDICTION[tau])[n-1:], color=COLOR[tau], dashes=[4, 2],linewidth=1)


# plt.plot(timelineflt[:n], e[:n], color='b', label='MKR epanechnikov',linewidth=1)
# plt.plot(timelineflt[n-1:], e[n-1:], color='b', dashes=[4, 2],linewidth=1)

# plt.plot(timelineflt[:n], p[:n], color='y', label='MKR polynomial',linewidth=1)
# plt.plot(timelineflt[n-1:], p[n-1:], color='y', dashes=[4, 2],linewidth=1)

# plt.plot(timelineflt[:n], esq1[:n], color='darkred', label='MKR ExpSin² lambda=0.01',linewidth=1)
# plt.plot(timelineflt[n-1:], esq1[n-1:], color='darkred', dashes=[4, 2],linewidth=1)

# plt.plot(timelineflt[:n], esq2[:n], color='green', label='MKR ExpSin² lambda=1',linewidth=1)
# plt.plot(timelineflt[n-1:], esq2[n-1:], color='green', dashes=[4, 2],linewidth=1)

# plt.plot(timelineflt[:n], g[:n], color='green', label='MKR Gaussian',linewidth=1)
# plt.plot(timelineflt[n-1:], g[n-1:], color='green', dashes=[4, 2],linewidth=1)


# plt.plot(timelineflt[:n], Y_pred_gauss[:n], color='g', label='G ExpSin²')
# plt.plot(timelineflt[n-1:], Y_pred_gauss[n-1:], color='g', linestyle="dotted")

# plt.plot(timelineflt[:n], Y_pred_gauss_tuned[:n], color='y', label='G ExpSin² tuned')
# plt.plot(timelineflt[n-1:], Y_pred_gauss_tuned[n-1:], color='y', linestyle="dotted")

# plt.plot(timelineflt[:n], Y_pred_3[:n], color='darkblue', label='KR Poly')
# plt.plot(timelineflt[n-1:], Y_pred_3[n-1:], color='darkblue', dashes=[4, 2])

# plt.plot(timelineflt[:n], Y_pred[:n], color='red', label='Forcast GDP linear',linewidth=1)
# plt.plot(timelineflt[n-1:], Y_pred[n-1:], color='red', dashes=[4, 2],linewidth=1)

# plt.plot(timelineflt[:n], Y_pred_2[:n], color='darkblue', label='KR Poly tuned')
# plt.plot(timelineflt[n-1:], Y_pred_2[n-1:], color='darkblue', dashes=[4, 2])

# plt.plot(timelineflt[:n], Y_pred_KR[:n], color='b', label='KR ExpSin²')
# plt.plot(timelineflt[n-1:], Y_pred_KR[n-1:], color='b', linestyle="dotted",linewidth=1)

# plt.plot(timelineflt[:n], Y_pred_KR_tuned[:n], color='darkolivegreen', label='KR ExpSin² tuned')
# plt.plot(timelineflt[n-1:], Y_pred_KR_tuned[n-1:], color='darkolivegreen', linestyle="dotted",linewidth=1)


print(color[2]+"------------------------"+color[0])
print(color[1]+"  END OF THE EXECUTION"+color[0])
print(color[2]+"------------------------"+color[0])
plt.title(f"GDP Forcast")
plt.legend()
plt.grid(False)
plt.xlim(timelineflt[n-60] , timelineflt[n+24] )
plt.ylim(-5 , 10)
# plt.savefig(f"Plot/GDP3.png")
plt.savefig(f"Plot/GDP/GDP4.png")



'''
EXECUTION DU PROGRAMME EN COURS...
'''