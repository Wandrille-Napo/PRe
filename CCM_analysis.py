# Define colors for output
# RED   = "\033[1;31m"  
# BLUE  = "\033[1;34m"
# CYAN  = "\033[1;36m"
# GREEN = "\033[0;32m"
# RESET = "\033[0;0m"
# BOLD    = "\033[;1m"
# REVERSE = "\033[;7m"
#color=[RESET, RED, GREEN, BLUE, CYAN BOLD]
color=["\033[0;0m", "\033[1;31m" , "\033[0;32m",  "\033[1;34m", "\033[1;36m", "\033[;1m"] 

import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from datetime import datetime, timedelta
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, RBF, WhiteKernel, Sum

lam = 0.01
def Alpha (k,x,y) :
    n = len(x)
    K = np.zeros((n,n))
    for i in range(n) :
        for j in range(n) :
            K[i,j] = k( x[i,0] , x[j,0] )
    return np.linalg.inv(K + lam * np.identity(n)).dot(y)

#- Kernels Definition
def k_p (x,y) :
    return (np.dot ( x, y ) + 1)**3
def k_g (x,y) :
    return np.exp(-(x - y )**2 / 2)
kernel2 = Sum(ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-7, 1e4)) , WhiteKernel(noise_level=1e-1) )


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
    print(color[4]+"o "+color[0]+"read_csv_to_matrix(): "+file_path+" has been transformed into a matrix\n")
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
file_path1 = 'data/CRSP_sample.csv'
file_path2 = 'data/CCM_annual_1989-2024.csv'
C2=[2,3,97,336]
CCM=read_csv_to_matrix(file_path2,C2)
DGS10=read_csv_to_matrix('data/DGS10.csv',[])
firm_column = CCM[:, 1].A1
count_firm=Counter(firm_column)
nb_firm_name=len(count_firm)
max_time_window=count_firm[max(count_firm, key=count_firm.get)]

############################################################
#                   Matrices
############################################################

#- timeline & dates
timeline=[]
current_date = datetime.strptime('1997-01-02', '%Y-%m-%d')
while current_date.strftime('%Y-%m-%d') != '2023-12-31':
    timeline.append(current_date.strftime('%Y-%m-%d'))
    current_date += timedelta(year=1)
dates = [datetime.strptime(date, '%Y-%m-%d') for date in timeline]
N=len(timeline)
print(timeline)
print(CCM)
#-Rf
Rf = np.empty((N,1), dtype=object)
for i in range(1,len(DGS10)):
    try:
        index = timeline.index(DGS10[i,0])
        Rf[index] = 0.01*float(DGS10[i,1])
    except:
        Rf[index] = None
        

# #- FIRMname, RET & SP500(=R_mkt)
# RET = np.empty((N,nb_firm_name), dtype=object)
# SP500 = np.empty((N,1), dtype=object)
# FIRMname=[]
# index_old = timeline.index(CCM[1,0])-1
# firm_index=0
# for i in range(1,len(CCM)):
#     index = timeline.index(CCM[i,0])
#     if(index_old>index): # when new firm
#         FIRMname.append(CCM[i-1,1]) # setting the name of the company "firm_index" to the newest name
#         firm_index+=1 # firm column change
#     if Rf[index] != None :
#         SP500[index] = CCM[i,3]
#     try:
#         RET[index,firm_index] = float(CCM[i,2])
#     except:
#         RET[index,firm_index] = None
#     index_old=index
# FIRMname.append(CCM[i-1,1])#adding the last firm name
# nb_firm=firm_index+1
# RET=RET[:,:nb_firm]

# #- equalizing nb of None regarding SP500 & Rf
# for i in range(N):
#     if SP500[i] == None or Rf[i] == None:
#         SP500[i] = None
#         Rf[i] = None
#         for k in range(nb_firm):
#             RET[i,k]= None

# ############################################################
# #                   CAPM
# ############################################################
# # for firm in range(nb_firm):
# firm=4
# #- Definition of X and Y
# Diff = np.empty((N,1), dtype=object)
# Ret = np.empty((N,1), dtype=object)

# m=np.nanmean(np.array([x if x is not None else np.nan for x in Rf], dtype=float))
# for i in range(N):
#     if (SP500[i] != None):
#         Diff[i] = SP500[i] - Rf[i]/365
#         Ret[i]  = RET[i,firm]
# Diff = Diff.astype(float)
# Ret = Ret.astype(float)

# valid_indices = np.where(~np.isnan(Diff).flatten() & ~np.isnan(Ret).flatten())[0]
# Diff = Diff[valid_indices]
# Ret = Ret[valid_indices]

# n=len(Diff)

# #- Regression

# #
# model = LinearRegression()
# model.fit(Diff[int(n/2)-180:int(n/2)],Ret[int(n/2)-180:int(n/2)])
# alpha = model.intercept_[0]
# beta = model.coef_[0][0]
# # print(f"({alpha},{beta})")
# print(color[4]+"o "+color[0]+" Linear regression done\n")

# #
# alpha_p = Alpha( k_p , Diff[int(n/2)-180:int(n/2)],Ret[int(n/2)-180:int(n/2)] )
# alpha_g = Alpha( k_g , Diff[int(n/2)-180:int(n/2)],Ret[int(n/2)-180:int(n/2)])
# u=[] ; v=[]
# for j in range ( n ) :
#     S = 0
#     for i in range ( 180 ) :
#         S = S + alpha_p [i] * k_p( Diff[i] , Diff[j] )
#     # u . append (S)
#     S = 0
#     for i in range ( 180 ) :
#         S = S + alpha_g [i] * k_g( Diff[i] , Diff[j] )
#     v . append (S)
# print(color[4]+"o "+color[0]+" Gaussian regression 1 done\n")

# #
# gaussian_process = GaussianProcessRegressor(kernel=kernel2)
# gaussian_process.fit(Diff[int(n/2)-180:int(n/2)],Ret[int(n/2)-180:int(n/2)])
# print(color[4]+"o "+color[0]+" Gaussian regression 2 done\n")


# #
# param_distributions = {
#     "kernel__k1__length_scale": loguniform(1e-2, 1e2),  # length_scale du premier sous-noyau (ExpSineSquared)
#     "kernel__k1__periodicity": loguniform(1e-1, 1e1),  # periodicity du premier sous-noyau (ExpSineSquared)
#     "kernel__k2__noise_level": loguniform(1e-2, 1e2)  # length_scale du deuxième sous-noyau (RBF)
# }
# k_tuned = RandomizedSearchCV(gaussian_process,param_distributions=param_distributions,n_iter=500,random_state=0,)
# k_tuned.fit(Diff[int(n/2)-180:int(n/2)],Ret[int(n/2)-180:int(n/2)])
# print(color[4]+"o "+color[0]+" Gaussian tuned regression done\n")
# # krr = KernelRidge(kernel=k_p, alpha=1.0)
# # krr.fit(Diff,Ret)


# Ret_pred = model.predict(Diff)
# Ret_pred_k = k_tuned.predict(Diff)
# # print(Ret_pred_k)

# #- Plot
# dates_Ret=(np.array(dates))[valid_indices]
# plt.plot(dates_Ret, Ret, color='grey', label='Actual Ret',linewidth=0.5)

# plt.plot(dates_Ret[:int(n/2)], Ret_pred[:int(n/2)], color='red', label='Expected Ret /w CAPM')
# plt.plot(dates_Ret[int(n/2):], Ret_pred[int(n/2):], color='red', label='Expected Ret /w CAPM', linestyle="dotted")

# plt.plot(dates_Ret[:int(n/2)], v[:int(n/2)], color='green', label='Exp kernel Ret')
# plt.plot(dates_Ret[int(n/2):], v[int(n/2):], color='green', label='Exp kernel Ret', linestyle="dotted")

# plt.plot(dates_Ret[:int(n/2)], Ret_pred_k[:int(n/2)], color='y', label='Exp kernel tuned Ret')
# plt.plot(dates_Ret[int(n/2):], Ret_pred_k[int(n/2):], color='y', label='Exp kernel Ret tuned', linestyle="dotted")


# plt.title(f"Return of {FIRMname[firm]} [firm={firm}]")
# # plt.legend()
# plt.grid(True)
# plt.xlim(dates_Ret[int(n/2)-50] , dates_Ret[int(n/2)+50] )
# plt.ylim(-0.10 , 0.15)
# plt.savefig(f"CAPM_forcast_firm${firm}$.png")

# ############################################################
# #                   Prints & Plots
# ############################################################

# # print(color[0]+"CCM:\n   >size: "+color[2]+"{}x{}".format(np.size(CCM[:,0].A1),np.size(CCM[0,:].A1))+color[0])
# # print(color[0]+f"   >Number of firms: "+color[2]+f"{nb_firm}")
# # print(color[0]+"   >Typical examples of rows:"+color[2])
# # print(CCM[:5,:])
# # print(color[0])
# plt.close()
# plt.plot(dates , SP500 , c = "r" , label = "S&P 500 return")
# plt.savefig("S&P500.png")

# # time=[]
# # for i in range(len(CCM[:,0])):
# #     for j in range(i,len(DGS10[:,0])):
# #         if CCM[i,0]==DGS10[j,0]:
# #             time.append(CCM[i,0])
# # time = enumerate(time)
# # for index, date in enumerate(time):
# #     # Imprimer une partie de la liste énumérée
# #     if index < 50:
# #         print(f"[{index}]: {date}")