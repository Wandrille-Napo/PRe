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
import warnings
import pandas as pd
import calendar
import csv
import matplotlib.pyplot as plt

from collections import Counter
from datetime import datetime, timedelta
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, RBF, WhiteKernel, Sum, Product, Matern, RationalQuadratic,DotProduct
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import mean_squared_error

# from optimparallel import minimize_parallel
import time

import numpy as np
from scipy.optimize import minimize


import torch
from torch.optim import SGD
from torch.nn.functional import mse_loss
import torch.nn.functional as F

def prctge(x):
    return 100*x
prctge=np.vectorize(prctge)

def AVG(L):
    avg=[]
    for i in range(len(L[0])):
        m=0
        j=0
        for x in L:
            m+=x[i]
            j+=1
        avg.append(m/j)
    return avg

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
            Rx+=(x[i]-Ret[i])**2
            Dx+=(np.mean(Ret[:n])-Ret[i])**2
            Dx2+=(np.mean(x[n:n+nb])-Ret[i])**2
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
            Rx+=(x[i]-Ret[i])**2
            Dx+=(np.mean(Ret[n:n+nb])-Ret[i])**2
            div+=(x[i]-Ret[i])**2/((abs(x[i])+abs(Ret[i]))/2)
            Dx2+=(np.mean(x[n:n+nb])-Ret[i])**2
        temp=Rx/Dx
        temp=int(temp*1000)/1000
        P.append(temp)
        
        temp=Rx/Dx2
        temp=int(temp*1000)/1000
        P2.append(temp)
        
        div=int((100/n)*div*1000)/1000
        SMAPE.append(div)
    return (R,P,R2,P2,SMAPE)

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
    # print(color[4]+"o "+color[0]+"read_csv_to_matrix(): "+file_path+" has been transformed into a matrix\n")
    return matrix

def safe_convert(x):
    try:
        return float(x)
    except ValueError:
        return x  # or any other value indicating a failed conversion
convert=np.vectorize(safe_convert, otypes=[object])

#####################################################################################################################################
#                   CSV to basic matrices
#####################################################################################################################################

# Path to the CSV file
file_path1 = 'data/CRSP_monthly.csv'
C=[2,9,58,63,7]#60

# file_path2 = 'data/CCM_annual_1989-2024.csv'
C2=[3,10,95,279,9]

# CCM=read_csv_to_matrix(file_path2,C2)
CRSP=read_csv_to_matrix(file_path1,C)
DGS10=read_csv_to_matrix('data/monthly_DGS10.csv',[])
firm_column = CRSP[:, 1].A1
count_firm=Counter(firm_column)
nb_firm_name=len(count_firm)
max_time_window=count_firm[max(count_firm, key=count_firm.get)]

#####################################################################################################################################
#                   Matrices timeline, FIRMname & Rf
#####################################################################################################################################

timeline=[]
timelineflt=[]
current_date = datetime.strptime('1997-01', '%Y-%m')
t=1997+1/12
end_date = datetime.strptime('2024-01', '%Y-%m')
while current_date <= end_date:
    timelineflt.append(t)
    timeline.append(current_date.strftime('%Y-%m'))  # Convertir la date en string avant de l'ajouter
    
    days_in_month = calendar.monthrange(current_date.year, current_date.month)[1]
    current_date = current_date.replace(day=days_in_month) + timedelta(days=1)
    next_month = current_date.replace(day=28) + timedelta(days=4)
    current_date = next_month - timedelta(days=next_month.day)
N=len(timeline)

#-Rf
Rf = np.empty((N,1), dtype=object)
for i in range(1,len(DGS10)):
    try:
        index = timeline.index(DGS10[i,0])
        Rf[index] = 0.01*float(DGS10[i,1])
    except:
        Rf[index] = None
        

#- FIRMname
FIRMname=[]
CUSIP=[CRSP[1,4]]
index_old = timeline.index(CRSP[1,0])-1
firm_index=0
FIRMname2=[CRSP[1,1]]
for i in range(1,len(CRSP)):
    index = timeline.index(CRSP[i,0])
    if(index_old>index): # when new firm
        CUSIP.append(CRSP[i,4])
        FIRMname.append(CRSP[i-1,1]) # setting the name of the company "firm_index" to the newest name
        FIRMname2.append(CRSP[i,1])# setting the name of the company "firm_index" to the oldest name
        firm_index+=1 # firm column change
    index_old=index
FIRMname.append(CRSP[i-1,1])#adding the last firm name
nb_firm=firm_index+1

#####################################################################################################################################
#                  Firm filter
#####################################################################################################################################


# firm_in_2_datasets=[]
# for x in CCM[:,1]:
#     if x in FIRMname2 and FIRMname2.index(x) not in firm_in_2_datasets:
#         firm_in_2_datasets.append(FIRMname2.index(x))
# print(f"{len(firm_in_2_datasets)}/{nb_firm}") #168/412
# print(firm_in_2_datasets)

# # firm_in_2_datasets=[223, 377, 165, 128, 205, 265, 293, 300, 407, 342, 246, 390, 121, 208, 21, 28, 33, 22, 11, 27, 14, 36, 35, 31, 37, 70, 67, 74, 93, 104, 101, 112, 109, 108, 118, 116, 99, 110, 119, 134, 25, 132, 139, 105, 133, 15, 144, 147, 140, 151, 167, 157, 146, 153, 131, 168, 173, 129, 149, 150, 181, 154, 183, 184, 197, 51, 360, 210, 198, 195, 124, 209, 227, 232, 247, 240, 218, 237, 239, 249, 236, 255, 248, 238, 250, 267, 73, 259, 217, 219, 71, 280, 285, 8, 272, 310, 308, 305, 312, 320, 318, 326, 330, 10, 161, 224, 369, 348, 334, 363, 30, 383, 284, 365, 373, 374, 380, 375, 296, 372, 396, 401, 410, 314, 395, 350, 289, 241, 287, 266, 402, 346, 276, 125, 353, 349, 303, 38, 319, 364, 69, 152, 370, 111, 88, 203, 55, 2, 244, 354, 100, 252, 193, 313, 229, 138, 19, 286, 301, 316, 90, 323, 169, 263, 309, 337, 77, 122]
# # [128, 121, 21, 28, 33, 22, 11, 27, 14, 36, 35, 31, 37, 70, 67, 74, 93, 104, 101, 112, 109, 108, 118, 116, 99, 110, 119, 134, 25, 132, 139, 105, 133, 15, 144, 147, 140, 151, 146, 153, 131, 129, 149, 150, 154, 51, 124, 73, 71, 8, 10, 30, 125, 38, 69, 152, 111, 88, 55, 2, 100, 138, 19, 90, 77, 122]
# firm_in_2_datasets=np.array(firm_in_2_datasets)

# CCM = np.array(CCM,dtype=object)

# valid_firm=[]
# FIRMname3=FIRMname2
# def count_company_occurrences(array, company_name):
#     count = 0
#     for row in array[1:]:  # Ignorer l'en-tête
#         if row[1] == company_name:
#             count += 1
#     return count

# for row in CCM[1:]:
#     if (row[2] == '') and (row[1] in FIRMname3):
#         FIRMname3[FIRMname3 == row[1]] = 'None'
# FIRMname3=[element for element in FIRMname3 if element != 'None']
# for firm in FIRMname3:
#     if (FIRMname2.index(firm) in firm_in_2_datasets) and (firm not in valid_firm) :
#         valid_firm.append(firm)
# for i in range(len(valid_firm)-1,0,-1):
#     if count_company_occurrences(CCM,valid_firm[i])<20:
#         valid_firm.pop(i)

# mask = np.isin(CCM[:, 1], valid_firm)
# CCM = CCM[mask]  
# print(valid_firm)
# # valid_firm=['IROQUOIS BANCORP INC', 'J & J SNACK FOODS CORP', 'PLEXUS CORP', 'NEWMIL BANCORP INC', 'SUN MICROSYSTEMS INC', 'TUESDAY MORNING CORP', 'ORACLE CORP', 'MICROSOFT CORP', 'SUNGARD DATA SYSTEMS INC', 'EAGLE BANCSHARES INC', 'TECH DATA CORP', 'REPLIGEN CORP', 'BAY VIEW CAPITAL CORP', 'SIGMA DESIGNS INC', 'MASSBANK CORP', 'TEKELEC', 'AUTOINFO INC', 'LINEAR TECHNOLOGY CORP', 'CYPRESS SEMICONDUCTOR CORP', 'AMCORE FINANCIAL INC', 'XOMA CORP', 'DAILY JOURNAL CORP', 'CANDELA CORP', 'ASTEC INDUSTRIES INC', 'NAVIGATORS GROUP INC', 'WERNER ENTERPRISES INC', 'SKYWEST INC', 'AMETEK INC', 'RESEARCH FRONTIERS INC', 'WARREN BANCORP INC', 'AMERICAN WOODMARK CORP', 'BROAD NATIONAL BANCORP', 'THERAGENICS CORP', 'LAKELAND INDUSTRIES INC', 'ACETO CORP', 'CYANOTECH CORP', 'PULSE BANCORP INC', 'FISERV INC', 'CHARTER ONE FINANCIAL INC', 'CARMIKE CINEMAS INC', 'HEMACARE CORP', 'BANK SOUTH CAROLINA CORP', 'SILICON GRAPHICS INC', 'HEARTLAND EXPRESS INC', 'GAINSCO INC', 'RENTRAK CORP', 'BRUNSWICK CORP', 'UNISYS CORP', 'WSFS FINANCIAL CORP', 'INVESTORS TITLE CO', 'CERNER CORP', 'FARMERS CAPITAL BANK CORP', 'FOUNTAIN POWERBOAT INDS INC', 'FLAG FINANCIAL CORP', 'WEBSTER FINANCIAL CORP', 'COMVERSE TECHNOLOGY INC', 'CENTENNIAL BANCORP', 'FIRST LONG ISLAND CORP', 'PSYCHEMEDICS CORP', 'ENVIRONMENTAL POWER CORP', 'AMERIANA BANCORP', 'PARLUX FRAGRANCES INC', 'AMERICAN VANGUARD CORP', 'PHOTRONICS INC', 'JENNIFER CONVERTIBLES INC']
# # ['BANCORP CONNECTICUT INC', 'ACETO CORP', 'CYANOTECH CORP', 'PULSE BANCORP INC', 'CHARTER ONE FINANCIAL INC', 'SILICON GRAPHICS INC', 'HEARTLAND EXPRESS INC', 'RENTRAK CORP', 'UNISYS CORP', 'INVESTORS TITLE CO', 'CERNER CORP', 'FARMERS CAPITAL BANK CORP']

# df_filtered = pd.DataFrame(CCM, columns=['Year', 'Company', 'at', 'emp', 'cusip'])

# output_csv_path = 'data/CCM2.csv'
# df_filtered.to_csv(output_csv_path, index=False)

#####################################################################################################################################
#                   Matrices 2: CCM, AT, EMP, RATIO, SP500, RET
#####################################################################################################################################


CCM=read_csv_to_matrix('data/CCM.csv',[])
FIRMnameCRSP=FIRMname
FIRMname=list(np.unique(CCM[1:,1].A1))
# print(FIRMname)
# ['ACETO CORP', 'AMCORE FINANCIAL INC', 'AMERIANA BANCORP', 'AMERICAN VANGUARD CORP', 'AMERICAN WOODMARK CORP', 'AMETEK INC', 'ASTEC INDUSTRIES INC', 'AUTOINFO INC', 'BANK SOUTH CAROLINA CORP', 'BAY VIEW CAPITAL CORP', 'BROAD NATIONAL BANCORP', 'BRUNSWICK CORP', 'CANDELA CORP', 'CARMIKE CINEMAS INC', 'CENTENNIAL BANCORP', 'CERNER CORP', 'CHARTER ONE FINANCIAL INC', 'COMVERSE TECHNOLOGY INC', 'CYANOTECH CORP', 'CYPRESS SEMICONDUCTOR CORP', 'DAILY JOURNAL CORP', 'EAGLE BANCSHARES INC', 'ENVIRONMENTAL POWER CORP', 'FARMERS CAPITAL BANK CORP', 'FIRST LONG ISLAND CORP', 'FISERV INC', 'FLAG FINANCIAL CORP', 'FOUNTAIN POWERBOAT INDS INC', 'GAINSCO INC', 'HEARTLAND EXPRESS INC', 'HEMACARE CORP', 'INVESTORS TITLE CO', 'IROQUOIS BANCORP INC', 'J & J SNACK FOODS CORP', 'JENNIFER CONVERTIBLES INC', 'LAKELAND INDUSTRIES INC', 'LINEAR TECHNOLOGY CORP', 'MASSBANK CORP', 'MICROSOFT CORP', 'NAVIGATORS GROUP INC', 'NEWMIL BANCORP INC', 'ORACLE CORP', 'PARLUX FRAGRANCES INC', 'PHOTRONICS INC', 'PLEXUS CORP', 'PSYCHEMEDICS CORP', 'PULSE BANCORP INC', 'RENTRAK CORP', 'REPLIGEN CORP', 'RESEARCH FRONTIERS INC', 'SIGMA DESIGNS INC', 'SILICON GRAPHICS INC', 'SKYWEST INC', 'SUN MICROSYSTEMS INC', 'SUNGARD DATA SYSTEMS INC', 'TECH DATA CORP', 'TEKELEC', 'THERAGENICS CORP', 'TUESDAY MORNING CORP', 'UNISYS CORP', 'WARREN BANCORP INC', 'WEBSTER FINANCIAL CORP', 'WERNER ENTERPRISES INC', 'WSFS FINANCIAL CORP', 'XOMA CORP']
CCM = np.array(CCM,dtype=object)
# ['ASTEC INDUSTRIES INC', 'DAILY JOURNAL CORP', 'J & J SNACK FOODS CORP', 'MICROSOFT CORP', 'PLEXUS CORP', 'REPLIGEN CORP', 'SKYWEST INC', 'WERNER ENTERPRISES INC', 'XOMA CORP']
nb_firm=len(FIRMname)

EMP=np.empty((N,nb_firm), dtype=object)
AT=np.empty((N,nb_firm), dtype=object)
for x in FIRMname:
    idx = list(CCM[:,1]).index(x)
    firmIDX = FIRMname.index(x)
    idx2 = idx + list(CCM[idx:,0]).index(1997)
    i=idx2
    while 12*(i-idx2)<AT.shape[0] and CCM[i,1]==x:
        for k in range(12):
            AT[12*(i-idx2)+k,firmIDX]=float(CCM[i,2])
        for k in range(12):
            EMP[12*(i-idx2)+k,firmIDX]=int(CCM[i,3]*1000)
        i+=1
RATIO=np.empty((N,nb_firm), dtype=object)
for i in range(AT.shape[0]):
    for j in range(AT.shape[1]):
        try:
            RATIO[i,j] = AT[i,j]/EMP[i,j]
        except:
            RATIO[i,j] = None
AT=np.array(AT)
EMP=np.array(EMP)
RATIO=np.array(RATIO)

RET = np.empty((N,nb_firm), dtype=object)
SP500 = np.empty((N,1), dtype=object)
for i in range(1,len(CRSP)):
    if CRSP[i,1] in FIRMname:
        firm_index = FIRMname.index(CRSP[i,1])
        index = timeline.index(CRSP[i,0])
        if Rf[index] != None :
            SP500[index] = CRSP[i,3]
        try:
            RET[index,firm_index] = float(CRSP[i,2])
        except:
            RET[index,firm_index] = None

CRSP=np.array(CRSP)
CRSP2= np.array([row for row in CRSP if row[1] in FIRMname])


# end=timeline.index('2024-01')
# CRSP=CRSP[:end,:]
# SP500=SP500[:end]
# Rf=Rf[:end]
# RET=RET[:end,:]
# AT=AT[:end,:]
# EMP=EMP[:end,:]
# RATIO=RATIO[:end,:]

# print(color[4]+"o "+color[0]+"Basics matrices done \n")
               
########################################################################################################################
#                   Choice of the studied firm
########################################################################################################################
#
# # for firm in range(nb_firm):
firm=0
print(color[3]+"o "+color[-1]+"firm: "+ FIRMname[firm] + color[0]+"\n")

########################################################################################################################
#                   Definitions and arrays setup
########################################################################################################################


#- equalizing nb of None regarding SP500 & Rf
for i in range(N):
    if SP500[i] == None or Rf[i] == None or RATIO[i,firm] == None:
        SP500[i] = None
        Rf[i] = None
        RET[i,firm]= None
        AT[i,firm]= None
        EMP[i,firm]= None
        RATIO[i,firm]= None

#- Definition of INPUT, X2 and Ret


INPUT=np.asarray(np.concatenate([SP500,Rf,np.matrix(AT[:,firm]).T,np.matrix(EMP[:,firm]).T,np.matrix(RATIO[:,firm]).T], axis=1))
Ret=np.array(RET[:,firm])
Ret = Ret.astype(float)
Ret=prctge(Ret)

#- Normalization of INPUT
for j in range(INPUT.shape[1]):
    max=np.nanmax(np.array([abs(x) if x is not None else np.nan for x in INPUT[:,j]], dtype=float))
    for i in range(INPUT.shape[0]):
        try:
            INPUT[i,j]=INPUT[i,j]/max
        except:
            INPUT[i,j]=None

#- Definition of X
n=timeline.index('2006-01')
X=INPUT[:n]
Y=np.asarray(Ret[:n], dtype=float)



#- Filter of None
invalid_x_mask = np.any(np.isnan(X.astype(float)) | (X == None), axis=1)
invalid_input_mask = np.any(np.isnan(INPUT.astype(float)) | (INPUT == None), axis=1)
invalid_y_mask = np.isnan(Y)
invalid_ret_mask = np.isnan(Ret)
mask1=invalid_x_mask|invalid_y_mask
mask2=invalid_input_mask|invalid_ret_mask
X =X[~mask1,:]
Y = Y[~mask1]
INPUT = INPUT[~mask2,:]
Ret = Ret[~mask2]

#- Definition of X2

X2=np.empty((N,1))
for i in range(N):
    try:
        X2[i] = SP500[i] - Rf[i]/365   
    except:
        X2[i] = None
max=np.nanmax(np.array([abs(x) if x is not None else np.nan for x in X2], dtype=float))
for i in range(X2.shape[0]):
    try:
        X2[i]=X2[i]/max
    except:
        X2[i]=None
X2 = X2.astype(float)
invalid_x2_mask = np.any(np.isnan(X2.astype(float)) | (X2 == None), axis=1)
mask3=invalid_x2_mask|invalid_ret_mask

X2_input=X2[~mask3]
X2=X2[:n]
X2=X2[~mask3[:n]]
########################################################################################################################
#                    Linear Regression
########################################################################################################################

#
model = LinearRegression()
model.fit(X2,Y)
# print(color[4]+"o "+color[0]+" Linear regression done\n")
Y_pred_linear = model.predict(X2_input)


########################################################################################################################
#                    Manual Regression
########################################################################################################################
# def Alpha (k,x,y,lam) :
#     n = len(x)
#     K = np.zeros((n,n))
#     for i in range(n) :
#         for j in range(n) :
#             K[i,j] = k( x[i] , x[j] )
#     return np.linalg.inv(K + lam * np.identity(n)).dot(y)

# #- Kernel Model
# L1=10
# lam1 = 0.1
# L2=1
# lam2 = 0.3

# def SUM_ABS(x):
#     s=0
#     for e in x:
#         s+=abs(e)
#     return s
# #- Kernels Definition
# def k_p (x,y) :
#     return (np.dot ( x, y )+ 1)**3
# def k_g (x,y) :
#     return np.exp(-SUM_ABS( x-y)**2)/np.sqrt(2*np.pi)
# def k_g_old (x,y) :
#     return np.exp(-np.dot( x-y, x-y))/np.sqrt(2*np.pi)
# def k_e (x,y) :
#     return 3/2*(1-4*(np.dot( x,y ))**2)
# def k_ESQ (x,y) :
#     return np.exp(L1*np.sin(np.dot ( x,y)))/np.sqrt(2*np.pi)
# def k_ESQ2 (x,y) :
#     return np.exp(L2*np.sin(np.dot ( x, y)))/np.sqrt(2*np.pi)
# print(color[-1]+"o "+color[0]+"Manual regressions have started", end='\r')

# alpha_e = Alpha( k_e , X,Y,lam2)
# alpha_esq2 = Alpha( k_ESQ2 , X,Y,lam2)
# alpha_p = Alpha( k_p , X,Y,lam1 )
# alpha_esq = Alpha( k_ESQ , X,Y,lam1 )
# alpha_g = Alpha( k_g , X,Y,lam2 )
# alpha_g_old = Alpha( k_g_old , X,Y,lam1 )
# e, g, g_old, p, esq1, esq2= [], [], [], [], [], []
# for j in range(len(INPUT)):
#     S_e, S_g, S_p, S_esq1, S_esq2, S_g_old = 0, 0, 0, 0, 0, 0
#     for i in range(n):
#         S_e += alpha_e[i] * k_e(X[i], INPUT[j])
#         S_g += alpha_g[i] * k_g(X[i], INPUT[j])
#         S_g_old += alpha_g_old[i] * k_g_old(X[i], INPUT[j])
#         S_p += alpha_p[i] * k_p(X[i], INPUT[j])
#         S_esq1 += alpha_esq[i] * k_ESQ(X[i], INPUT[j])
#         S_esq2 += alpha_esq2[i] * k_ESQ2(X[i], INPUT[j])
#     e.append(S_e)
#     g.append(S_g)
#     g_old.append(S_g_old)
#     p.append(S_p)
#     esq2.append(S_esq2)
#     esq1.append(S_esq1)

# (R,P)=perf((e,p,g,g_old, esq1),6)
# e=np.matrix(e).transpose()
# g=np.matrix(g).transpose()
# g_old=np.matrix(g_old).transpose()
# p=np.matrix(p).transpose()
# esq2=np.matrix(esq2).transpose()
# esq1=np.matrix(esq1).transpose()
# print(color[4]+"o "+color[0]+"Manual regression done          \n")
# print(color[1]+f"{R}"+color[0])
# print(color[1]+f"{P}"+color[0])

########################################################################################################################
#                    Sklearn KR Regression
########################################################################################################################
def train_model(kernel,param_distributions,it):
    tuned = RandomizedSearchCV(kernel,param_distributions=param_distributions,n_iter=it,random_state=0,)
    tuned.fit(X,Y)
    print(tuned.best_params_)
    print(color[4]+"o "+color[0]+"tuned regression done\n")
    return tuned.predict(INPUT)
warnings.filterwarnings('ignore')

#----------------------------- POLYNOMIAL ------------------------------------------------------------------------------------
POLY=[KernelRidge(kernel="polynomial",gamma= 4.65, degree= 3.4, coef0= 0.60, alpha= 14.98),KernelRidge(kernel="polynomial",gamma= 1.34, degree= 3.3, coef0= 0.65, alpha= 10.6)]
poly=POLY[firm]
# poly.fit(X,Y)
# Y_pred_p1 = poly.predict(INPUT)
# print(color[4]+"o "+color[0]+"KR Poly regression done\n")

#- Tuned Model
# print(color[-1]+"o "+color[0]+"KR Poly tuning has started", end='\r')
alpha_distribution = {"alpha": np.arange(1, 20,0.01)}
alpha_distributionL = {"alpha": np.arange(1e-3, 7,0.01)}
alpha_distributionH = {"alpha": np.arange(30, 100,0.1)}
param_poly2 = {
    "alpha": np.arange(1, 30,0.01),
    "degree": np.arange(3, 10, 0.01),  # Degree of the polynomial kernel
    "coef0": np.arange(0, 10,0.01),  # Independent term in polynomial kernel
    "gamma": np.arange(3, 10,0.01),  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
}
# Y_pred_p2 = train_model(poly,param_poly2,1000)

#------------------ Periodic Kernel: EXPSINESQUARED -----------------------------------------------------------------------------

ESQ=np.array([[KernelRidge(kernel=Sum(ExpSineSquared(2.16, 0.08), WhiteKernel(0.5)),alpha=8.69),KernelRidge(kernel=Sum(ExpSineSquared(0.17, 0.14), WhiteKernel(0.1)),alpha=12.3)],
              [KernelRidge(kernel=Sum(ExpSineSquared(0.06,0.91),WhiteKernel(noise_level=0.2)),alpha=1.68),KernelRidge(kernel=Sum(ExpSineSquared(0.175,0.52),WhiteKernel(noise_level=0.4)),alpha=6.79)]
              ])
esq=ESQ[firm,0]
# esq.fit(X,Y)
# Y_pred_esq=esq.predict(INPUT)
#- Tuned Model
# print(color[-1]+"o "+color[0]+"KR ESQ tuning has started", end='\r')

param_distributions = {
    "alpha": loguniform(0.1, 1e2) ,
    "kernel__k1__length_scale": loguniform(1e-2, 1e3),  # length_scale du premier sous-noyau (ExpSineSquared)
    "kernel__k1__periodicity": loguniform(1e-1, 1e3),  # periodicity du premier sous-noyau (ExpSineSquared)
    "kernel__k2__noise_level": loguniform(1e-1, 1)  
}
param_distributions2 = {
    "alpha": np.arange(1e-2, 13,0.01) ,
    "kernel__k1__length_scale": loguniform(0.1, 1e1),  # length_scale du premier sous-noyau (ExpSineSquared)
    "kernel__k1__periodicity": np.arange(0.01, 10,0.01),  # periodicity du premier sous-noyau (ExpSineSquared)
    "kernel__k2__noise_level": np.arange(0, 1,0.1)
    }
# Y_pred_esq2 = train_model(esq,param_distributions2,500) #{'kernel__k1__periodicity': 4.5, 'kernel__k1__length_scale': 11.4, 'alpha': 0.5}


#----------------- Locally Periodic Kernel -----------------------------------------------------------------------------
LPK=[KernelRidge(kernel=Product(ExpSineSquared(0.0053,0.147436),RBF(51856)),alpha=1.04),
     KernelRidge(kernel=Product(ExpSineSquared(0.0053,0.147436),RBF(51856)),alpha=1.04)]
lpk=LPK[firm]
# lpk.fit(X,Y)
# Y_pred_lpk=lpk.predict(INPUT)

param_lpk_global = {
    "alpha": np.arange(1e-1, 1e2,0.01) ,
    "kernel__k1__length_scale": loguniform(1e-3, 1e5),  # length_scale du premier sous-noyau (ExpSineSquared)
    "kernel__k1__periodicity": loguniform(1e-3, 1e3),  # periodicity du premier sous-noyau (ExpSineSquared)
    "kernel__k2__length_scale": loguniform(1e-3, 1e5)  # length_scale du deuxième sous-noyau (RBF)
}
param_lpk_fine = {
    "alpha": np.arange(1e-1, 10,0.01) ,
    "kernel__k1__length_scale": np.arange(1e-3, 1e-1,1e-3),  # length_scale du premier sous-noyau (ExpSineSquared)
    "kernel__k1__periodicity": np.arange(1e-2, 1e1,1e-3),  # periodicity du premier sous-noyau (ExpSineSquared)
    "kernel__k2__length_scale": loguniform(1e1, 1e3)  # length_scale du deuxième sous-noyau (RBF)
}
# Y_pred_lpk2=train_model(lpk,param_lpk_fine,2000)
#{'alpha': 3.0899999999999985, 'kernel__k1__length_scale': 0.068, 'kernel__k1__periodicity': 0.5209999999999996, 'kernel__k2__length_scale': 69.13775884432668}
# avg=AVG((Y_pred_lpk,Y_pred_lpk2))



param_lpk2_global = {
    "alpha": np.arange(1e-1, 1e2,0.01) ,
    "kernel__k1__length_scale": loguniform(1e-3, 1e3),  # length_scale du premier sous-noyau (ExpSineSquared)
    "kernel__k1__periodicity": loguniform(1e-3, 1e3),  # periodicity du premier sous-noyau (ExpSineSquared)
    "kernel__k2__degree": np.arange(1, 10,0.1) ,
    "kernel__k2__coef0": np.arange(0, 10,0.01),
    "kernel__k2__gamma": np.arange(3, 10,0.01)
}

# (R,P)=perf((Y_pred_esq,Y_pred_lpk,Y_pred_lpk2,avg),6)
# print(color[3]+f"{R}"+color[0])
# print(color[3]+f"{P}"+color[0])


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
            term14 = alpha[13] ** 2 * (1 + (norm / theta[25]) ** -1) ** -1
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
def k_np(x, y, theta,alpha):
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
    term14 = alpha[13] ** 2 * (1 + (norm / theta[25]) ** -1) ** -1
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
        
    return (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + 
                term11 + term12 + term13 + term14 + term15 + term16 + term17 + term18 + term19 + 
                term20 + term21 + term22)

def custom_kernel_np2(X, Y, omega):
    result = np.zeros((X.shape[0], Y.shape[0]))
    alpha=omega[:22]
    theta=omega[22:]
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
            term14 = alpha[13] ** 2 * (1 + (norm / theta[25]) ** -1) ** -1
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
def objective_np2(omega, X, Y, lambda1, lambda2):
    K = custom_kernel_np2(X, X, omega)
    # return mean_squared_error(Y, np.linalg.inv(K+lambda1*np.identity(X.shape[0])).dot(Y)) + lambda2 * np.linalg.norm(alpha, 1)
    K_inv_Y = np.linalg.inv(K + lambda1 * np.identity(X.shape[0])).dot(Y)
    error = squared_relative_error(Y, K_inv_Y)
    regularization = lambda2 * np.linalg.norm(omega[:22], 1)
    return error + regularization
def kernel_flows(it,X, Y, tau, omega_init, lambda1, lambda2):
    Nskf = X.shape[0] - tau
    X_tau = X[:Nskf]
    Y_tau = Y[tau:Nskf+tau]
    
    omega = omega_init
    for _ in range(it):  # Iterate to optimize alpha and theta
        # Fix alpha, optimize theta
        start_time = time.perf_counter()
        res_omega = minimize(objective_np2, omega, args=(X_tau, Y_tau, lambda1, lambda2), method='L-BFGS-B')
        execution_time = time.perf_counter() - start_time
        
        # print(color[3]+f"evolution of omega = {res_omega.x-omega}"+color[0])
        omega = res_omega.x
        # if _ == 0:
            # print(color[0]+f"Time measurments: 1 opti = {int(execution_time/60)} min"+color[1]+f"\ntotal time prediction : {int(execution_time*it/3600)}h\n"+color[0])
        print(f"{objective_np2(omega_init, X_tau, Y_tau, lambda1, lambda2)}->{objective_np2(omega, X_tau, Y_tau, lambda1, lambda2)} |omega={omega}")
        # print(color[2]+"> "+f"{100*(_+1)/it} %    "+color[0])
        
    return omega
def k(x, y, theta,alpha):
    norm_sq = torch.norm(x-y, p=2) ** 2
    norm = torch.norm(x-y, p=2)
    
    k = (alpha[0] ** 2 * (x @ y.T + theta[0] ** 2) +
         alpha[1] ** 2 * (theta[1] ** 2 * x @ y.T + theta[2] ** 2) ** abs(theta[3]) +
         alpha[2] ** 2 * torch.exp(-norm_sq / (2 * theta[4] ** 2)) +
         alpha[3] ** 2 * torch.exp(-norm / (2 * theta[5] ** 2)) +
         alpha[4] ** 2 * torch.exp(-torch.sin(torch.pi * norm_sq / theta[6]) ** 2 / theta[7] ** 2) *
         torch.exp(-norm_sq / theta[8] ** 2) +
         alpha[5] ** 2 * torch.exp(-torch.sin(torch.pi * norm_sq / theta[9]) ** 2 / theta[10] ** 2) +
         alpha[6] ** 2 * torch.exp(-torch.sin(torch.pi * norm / theta[11]) ** 2 / theta[12] ** 2) *
         torch.exp(-norm / theta[13] ** 2) +
         alpha[7] ** 2 * torch.exp(-torch.sin(torch.pi * norm / theta[14]) ** 2 / theta[15] ** 2) +
         alpha[8] ** 2 * (norm_sq + theta[16] ** 2) ** 0.5 +
         alpha[9] ** 2 * (theta[17] ** 2 + theta[18] ** 2 * norm_sq) ** -0.5 +
         alpha[10] ** 2 * (theta[19] ** 2 + theta[20] ** 2 * norm) ** -0.5 +
         alpha[11] ** 2 * (theta[21] ** 2 + norm) ** theta[22] +
         alpha[12] ** 2 * (theta[23] ** 2 + norm_sq) ** theta[24] +
         alpha[13] ** 2 * (1 + (norm / theta[25]) ** -1) ** -1 +
         alpha[14] ** 2 * (1 + norm / theta[26] ** 2) ** -1 +
         alpha[15] ** 2 * (1 - norm_sq / (norm_sq + theta[27] ** 2)) +
         alpha[16] ** 2 * torch.relu(1 - norm_sq / theta[28] ** 2) +
         alpha[17] ** 2 * torch.relu(1 - norm / theta[29] ** 2) +
         alpha[18] ** 2 * torch.log(norm ** theta[30] + 1) +
         alpha[19] ** 2 * torch.tanh(theta[31] * x @ y.T + theta[32]) +
         alpha[21] ** 2 * torch.exp(torch.sin(x @ y.T) / theta[34] ** 2) / torch.sqrt(torch.tensor(2 * torch.pi)))
    
    acos_argument = norm / theta[33] ** 2
    indicator = (acos_argument < 1).float()
    k += alpha[20] ** 2 * (torch.acos(-acos_argument) - acos_argument * torch.sqrt(1 - acos_argument ** 2)) * indicator

    return k
# Define the custom kernel function /w Pytorch
def custom_kernel(X, Y, theta, alpha):
    K = torch.zeros((X.shape[0], Y.shape[0]), dtype=torch.float32)
    h=False
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            
            # time1 = time.time() 
            oiu=k_np(X[i].clone().detach().numpy(), X[j].clone().detach().numpy(), theta.clone().detach().numpy(),alpha.clone().detach().numpy())
            # TT=time.time()-time1
            # if TT<1:
            #     h=False
            # else:
            #     h=True
            # if h==True:
            #     print(color[2]+f"k({i},{j}) : {TT} s"+color[0])
            # time1 = time.time() 
            K[i, j] = oiu
            # TT=time.time()-time1
            # print(color[2]+f"K[{i},{j}]<-oiu : {TT} s"+color[0])
    return K         
            
            # norm_sq = torch.norm(x - y, p=2) ** 2
            # norm = torch.norm(x - y, p=2)
          
            # term1 = alpha[0] ** 2 * (torch.dot(x, y) + theta[0] ** 2)
            # term2 = alpha[1] ** 2 * (theta[1] ** 2 * torch.dot(x, y) + theta[2] ** 2) ** torch.abs(theta[3])
            # term3 = alpha[2] ** 2 * torch.exp(-norm_sq / (2 * theta[4] ** 2))
            # term4 = alpha[3] ** 2 * torch.exp(-norm / (2 * theta[5] ** 2))
            # term5 = alpha[4] ** 2 * torch.exp(-torch.sin(torch.pi * norm_sq / theta[6]) ** 2 / theta[7] ** 2) * torch.exp(-norm_sq / theta[8] ** 2)
            # term6 = alpha[5] ** 2 * torch.exp(-torch.sin(torch.pi * norm_sq / theta[9]) ** 2 / theta[10] ** 2)
            # term7 = alpha[6] ** 2 * torch.exp(-torch.sin(torch.pi * norm / theta[11]) ** 2 / theta[12] ** 2) * torch.exp(-norm / theta[13] ** 2)
            # term8 = alpha[7] ** 2 * torch.exp(-torch.sin(torch.pi * norm / theta[14]) ** 2 / theta[15] ** 2)
            # term9 = alpha[8] ** 2 * (norm_sq + theta[16] ** 2) ** 0.5
            # term10 = alpha[9] ** 2 * (theta[17] ** 2 + theta[18] ** 2 * norm_sq) ** -0.5
            # term11 = alpha[10] ** 2 * (theta[19] ** 2 + theta[20] ** 2 * norm) ** -0.5
            # term12 = alpha[11] ** 2 * (theta[21] ** 2 + norm) ** theta[22]
            # term13 = alpha[12] ** 2 * (theta[23] ** 2 + norm_sq) ** theta[24]
            # term14 = alpha[13] ** 2 * (1 + (norm / theta[25]) ** -1) ** -1
            # term15 = alpha[14] ** 2 * (1 + norm / theta[26] ** 2) ** -1
            # term16 = alpha[15] ** 2 * (1 - norm_sq / (norm_sq + theta[27] ** 2))
            # term17 = alpha[16] ** 2 * torch.maximum(torch.tensor(0.0), 1 - norm_sq / theta[28] ** 2)
            # term18 = alpha[17] ** 2 * torch.maximum(torch.tensor(0.0), 1 - norm / theta[29] ** 2)
            # term19 = alpha[18] ** 2 * torch.log(norm ** theta[30] + 1)
            # term20 = alpha[19] ** 2 * torch.tanh(theta[31] * torch.dot(x, y) + theta[32])
            # term22 = alpha[21] ** 2 * torch.exp(torch.sin(torch.dot(x, y)) / theta[34] ** 2) / torch.sqrt(torch.tensor(2 * torch.pi))
            
            # acos_argument = norm / theta[33] ** 2
            # if acos_argument < 1:  # Indicator function condition
            #     term21 = alpha[20] ** 2 * (torch.acos(-acos_argument) - acos_argument * torch.sqrt(1 - acos_argument ** 2))
            # else:
            #     term21 = 0
                
            # K[i, j] = (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + 
            #                 term11 + term12 + term13 + term14 + term15 + term16 + term17 + term18 + term19 + 
            #                 term20 + term21 + term22)   
# Objective function for optimization /w Pytorch
def objective(theta, X, Y, alpha, lambda1, lambda2):
    K = custom_kernel(X, X, theta, alpha)
    K_inv = torch.linalg.inv(K + lambda1 * torch.eye(X.shape[0], dtype=X.dtype, device=X.device))
    mse = mse_loss(Y, K_inv.matmul(Y))
    l1_norm_alpha = lambda2 * torch.norm(alpha, 1)
    return mse + l1_norm_alpha
    
    # X = X.clone().detach().numpy()  # Detach X to avoid unnecessary gradient tracking
    # Y = Y.clone().detach().numpy()  # Detach Y to avoid unnecessary gradient tracking
    # alpha=alpha.clone().detach().numpy()
    # theta=theta.clone().detach().numpy()
    # K = custom_kernel_np(X, X, theta, alpha)
    # return mean_squared_error(Y, np.linalg.inv(K+lambda1*np.identity(X.shape[0])).dot(Y)) + lambda2 * np.linalg.norm(alpha, 1)
    # return objective_np(theta, X, Y, alpha, lambda1, lambda2)
def objectiveA(alpha, X, Y, theta, lambda1, lambda2):
    return objective(theta, X, Y,alpha, lambda1, lambda2)
def loss_function(alpha, theta, X_c, X, Y_c, Y, lambda1, lambda2):
    print("mark1")
    K_alpha_theta_c = custom_kernel(X_c, X_c, theta, alpha)
    print("mark2")
    print(torch.isinf(K_alpha_theta_c))
    print("mark3")
    K_alpha_theta_b = custom_kernel(X, X, theta, alpha)
    print("mark4")
    print(torch.isinf(K_alpha_theta_b))
    print("mark5")
    I_c = torch.eye(X_c.size(0), device=X_c.device)
    print("mark6")
    I_b = torch.eye(X.size(0), device=X.device)
    print("mark7")
    
    term1 = Y_c.T @ torch.linalg.solve(K_alpha_theta_c + lambda1 * I_c, Y_c)
    print("mark8")
    print(torch.isinf(term1))
    print("mark9")
    term2 = Y.T @ torch.linalg.solve(K_alpha_theta_b + lambda1 * I_b, Y)
    print("mark10")
    print(torch.isinf(term2))
    print("mark11")
    rho = 1 - (term1 / term2) + lambda2 * torch.norm(alpha, p=1)
    return rho
# Sparse Kernel Flows algorithm /w Pytorch
def sparse_kernel_flows(it, X, Y, tau, alpha_init, theta_init, lambda1, lambda2):
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    Nskf = X.shape[0] - tau
    X_tau = torch.tensor(X[:Nskf], dtype=torch.float32)
    Y_tau = torch.tensor(Y[tau:Nskf+tau], dtype=torch.float32)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    
    alpha = torch.tensor(alpha_init, dtype=torch.float32, requires_grad=True)
    theta = torch.tensor(theta_init, dtype=torch.float32, requires_grad=True)
    
    alpha_optimizer = SGD([alpha], lr=1e-2)
    theta_optimizer = SGD([theta], lr=1e-2)
    
    for _ in range(it):
        start_time = time.perf_counter()
        # Fix alpha, optimize theta
        OBJ=objective(theta, X_tau, Y_tau, alpha, lambda1, lambda2)
        t0=theta
        theta_optimizer.zero_grad()
        loss_theta = torch.tensor(objective(theta, X_tau, Y_tau, alpha, lambda1, lambda2), dtype=torch.float32, requires_grad=True)
        loss_theta.backward()
        theta_optimizer.step()
        print(color[3]+f"evolution of theta = {theta-t0}"+color[0])
        print(f"Theta after optimization: {theta}")
        print(color[2]+f"{OBJ}->{objective(theta, X_tau, Y_tau, alpha, lambda1, lambda2)}"+color[0])
        print(color[2]+f"Evolution obj: {objective(theta, X_tau, Y_tau, alpha, lambda1, lambda2)-OBJ}"+color[0])
        
        # Fix theta, optimize alpha
        print("test1")
        time1 = time.perf_counter() 
        mlkj=custom_kernel(X,X,theta,alpha)
        print(f"test2 : 1->2 in {time.perf_counter()-time1}")
        
        
        
        
        loss=loss_function(alpha,theta, X_tau,X, Y_tau,Y , lambda1, lambda2)
        
        a0=alpha
        alpha_optimizer.zero_grad()
        loss_alpha = torch.tensor(loss_function(alpha,theta, X_tau,X, Y_tau,Y , lambda1, lambda2), dtype=torch.float32, requires_grad=True)
        loss_alpha.backward()
        alpha_optimizer.step()
       
       # Print evolution
        execution_time = time.perf_counter() - start_time
        print(color[3]+f"evolution of alpha = {alpha-a0}"+color[0])
        print(f"Alpha after optimization: {alpha}")
        print(color[2]+f"{loss}->{loss_function(alpha,theta, X_tau,X, Y_tau,Y , lambda1, lambda2)}"+color[0])
        print(color[2]+f"Evolution obj: {loss_function(alpha,theta, X_tau,X, Y_tau,Y , lambda1, lambda2)-loss}"+color[0])
        
        if _ == 0:
            print(color[1]+f"\ntotal time prediction : {int(execution_time*it)} s\n"+color[0])
        print(color[1]+f"{100 * (_ + 1) / it} %"+color[0])
        
    return alpha.detach().numpy() ,theta.detach().numpy()


# Prediction
def prediction(X,Y,alpha,theta, lambda1,INPUT):
    K_test = custom_kernel_np(INPUT, X, theta, alpha)
    alphatemp=np.linalg.inv(custom_kernel_np(X, X, theta, alpha)+lambda1*np.identity(X.shape[0])).dot(Y)
    return K_test.dot(alphatemp)

# Initial parameters
alpha_init = 0.05*np.ones(22) ; alpha_init[4]=1 ; alpha_init[21]= 1; alpha_init[1]=1
theta_init = np.ones(35) ; theta_init[6]=0.147 ; theta_init[7]= 0.0053/np.sqrt(2); theta_init[8]=51856/np.sqrt(2)
theta_init[1]=np.sqrt(3.9) ;theta_init[2]= np.sqrt(4.27) ;theta_init[3]= 3.1
theta_init[10]= 0.0598 ;theta_init[9]= np.sqrt(0.91) ; theta_init[34]=np.sqrt(0.1)

omega=np.ones(57)
omega[:22]=alpha_init ; omega[22:]=theta_init
lambda1 = 5e-1
lambda2 = 1e-5
tau=5
# print(alpha_init)
# print(theta_init)
# Run Sparse Kernel Flows

theta0=[9.99999997e-01, 1.97482101e+00, 2.06639488e+00, 3.09997366e+00,1.00000001e+00, 1.00000005e+00, 1.46935844e-01, 9.14729754e-04,
 3.66677292e+04, 9.53910632e-01, 5.95061450e-02, 1.00001516e+00,1.00000230e+00, 9.99999794e-01, 1.00000008e+00, 9.99999962e-01,
 9.99999965e-01 ,9.99999982e-01, 1.00000000e+00, 1.00000002e+00,9.99999982e-01 ,9.99999982e-01, 1.00000007e+00, 9.99999890e-01,
 1.00000052e+00 ,9.99999700e-01, 9.99999892e-01, 9.99999978e-01, 9.99961542e-01, 9.99999385e-01, 9.99999164e-01, 1.00000001e+00,9.99999942e-01, 1.00001301e+00, 3.14400708e-01]
alpha0=[ 6.52142382e-03 , 2.14858833e-01 , 5.35227112e-02 , 4.32180725e-02,7.18572405e-01, 
-1.61647166e-03, -6.77084684e-01, -7.16287362e-02,
 -8.97466743e-02, -4.50714554e-02,  4.79673291e-02, -3.52821093e-01,9.77870738e+00, -9.02644026e-01, 
 -1.88029084e+00,  9.18210090e-01,2.01592225e+00,  1.51757420e+00, -2.27444510e+00,  2.23024542e-03,
 -4.47704836e-01 , 4.07936804e-01]

#lambda=(0.5 ; 1e-7)     ;    alpha = 0.05 sauf trois fois 1   ; tau=0
#obj=0.0264992956675196
#mse=0.92002151887543

# alpha0, theta0 = sparse_kernel_flows_np(1,X, Y, tau, alpha_init, theta_init, lambda1, lambda2)
#[107.592, 0.958]      pas d'évolution de obj du début à la fin : 0.08696048917448422
# [9.352, 1.063]      pour lambda 0.3;1e-8 et alpha 0.5 sauf qqe sqrt(2) -> le alpha n'a pas bouger
#mse = 9e+50

def k_optimized(x, y, theta,alpha,h):
    norm_sq = torch.norm(x - y, p=2) ** 2
    norm = torch.norm(x - y, p=2)

    results = 0
    
    # Profiling each kernel component
    start_time = time.time()
    results+= alpha[0] ** 2 * (x @ y.T + theta[0] ** 2)
    if h == True :
        print(f"Kernel 0 time: {time.time() - start_time}")

    start_time = time.time()
    results+= alpha[1] ** 2 * (theta[1] ** 2 * x @ y.T + theta[2] ** 2) ** abs(theta[3])
    if h == True :
        print(f"Kernel 1 time: {time.time() - start_time}")

    start_time = time.time()
    results+= alpha[2] ** 2 * torch.exp(-norm_sq / (2 * theta[4] ** 2))
    if h == True :
        print(f"Kernel 2 time: {time.time() - start_time}")

    start_time = time.time()
    results+= alpha[3] ** 2 * torch.exp(-norm / (2 * theta[5] ** 2))
    if h == True :
        print(f"Kernel 3 time: {time.time() - start_time}")

    start_time = time.time()
    results+= (alpha[4] ** 2 * torch.exp(-torch.sin(torch.pi * norm_sq / theta[6]) ** 2 / theta[7] ** 2) *
                   torch.exp(-norm_sq / theta[8] ** 2))
    if h == True :
        print(f"Kernel 4 time: {time.time() - start_time}")

    start_time = time.time()
    results+= alpha[5] ** 2 * torch.exp(-torch.sin(torch.pi * norm_sq / theta[9]) ** 2 / theta[10] ** 2)
    if h == True :
        print(f"Kernel 5 time: {time.time() - start_time}")

    start_time = time.time()
    results+= (alpha[6] ** 2 * torch.exp(-torch.sin(torch.pi * norm / theta[11]) ** 2 / theta[12] ** 2) *
                   torch.exp(-norm / theta[13] ** 2))
    if h == True :
        print(f"Kernel 6 time: {time.time() - start_time}")

    start_time = time.time()
    results+= alpha[7] ** 2 * torch.exp(-torch.sin(torch.pi * norm / theta[14]) ** 2 / theta[15] ** 2)
    if h == True :
        print(f"Kernel 7 time: {time.time() - start_time}")

    start_time = time.time()
    results+= alpha[8] ** 2 * (norm_sq + theta[16] ** 2) ** 0.5
    if h == True :
        print(f"Kernel 8 time: {time.time() - start_time}")

    start_time = time.time()
    results+= alpha[9] ** 2 * (theta[17] ** 2 + theta[18] ** 2 * norm_sq) ** -0.5
    if h == True :
        print(f"Kernel 9 time: {time.time() - start_time}")

    start_time = time.time()
    results+= alpha[10] ** 2 * (theta[19] ** 2 + theta[20] ** 2 * norm) ** -0.5
    if h == True :
        print(f"Kernel 10 time: {time.time() - start_time}")

    start_time = time.time()
    results+= alpha[11] ** 2 * (theta[21] ** 2 + norm) ** theta[22]
    if h == True :
        print(f"Kernel 11 time: {time.time() - start_time}")

    start_time = time.time()
    results+= alpha[12] ** 2 * (theta[23] ** 2 + norm_sq) ** theta[24]
    if h == True :
        print(f"Kernel 12 time: {time.time() - start_time}")

    start_time = time.time()
    results+= alpha[13] ** 2 * (1 + (norm / theta[25]) ** -1) ** -1
    if h == True :
        print(f"Kernel 13 time: {time.time() - start_time}")

    start_time = time.time()
    results+= alpha[14] ** 2 * (1 + norm / theta[26] ** 2) ** -1
    if h == True :
        print(f"Kernel 14 time: {time.time() - start_time}")

    start_time = time.time()
    results+= alpha[15] ** 2 * (1 - norm_sq / (norm_sq + theta[27] ** 2))
    if h == True :
        print(f"Kernel 15 time: {time.time() - start_time}")

    start_time = time.time()
    results+= alpha[16] ** 2 * torch.relu(1 - norm_sq / theta[28] ** 2)
    if h == True :
        print(f"Kernel 16 time: {time.time() - start_time}")

    start_time = time.time()
    results+= alpha[17] ** 2 * torch.relu(1 - norm / theta[29] ** 2)
    if h == True :
        print(f"Kernel 17 time: {time.time() - start_time}")

    start_time = time.time()
    results+= alpha[18] ** 2 * torch.log(norm ** theta[30] + 1)
    if h == True :
        print(f"Kernel 18 time: {time.time() - start_time}")

    start_time = time.time()
    results+= alpha[19] ** 2 * torch.tanh(theta[31] * x @ y.T + theta[32])
    if h == True :
        print(f"Kernel 19 time: {time.time() - start_time}")

    start_time = time.time()
    results+= alpha[21] ** 2 * torch.exp(torch.sin(torch.dot(x, y)) / theta[34] ** 2) / torch.sqrt(torch.tensor(2 * torch.pi))
    if h == True :
        print(f"Kernel 21 time: {time.time() - start_time}")

    acos_argument = norm / theta[33] ** 2
    if acos_argument < 1:  # Indicator function condition
        start_time = time.time()
        results+= alpha[20] ** 2 * (torch.acos(-acos_argument) - acos_argument * torch.sqrt(1 - acos_argument ** 2))
        if h == True :
            print(f"Kernel 20 time: {time.time() - start_time}")

    return results

# Y_prediction=prediction(X,Y,alpha0,theta0, lambda1,INPUT)

ALPHA=[];THETA=[];PREDICTION =[]
L2=[1e-7]

# omega=kernel_flows(1,X, Y, tau, omega, lambda1, lambda2)
# Y_prediction=prediction(X,Y,omega[:22],omega[22:], lambda1,INPUT)
# PREDICTION.append(Y_prediction)
for lambda2 in L2:
    print("")
    print(f"lambda2={lambda2}")
    print("")
    Nskf = X.shape[0] - tau
    X_tau = X[:Nskf]
    Y_tau = Y[tau:Nskf+tau]
    alpha = alpha_init
    theta = theta_init
    for _ in range(5):
        res_theta = minimize(objective_np, theta, args=(X_tau, Y_tau, alpha, lambda1, lambda2), method='L-BFGS-B')
        theta = res_theta.x
        res_alpha = minimize(objectiveA_np, alpha, args=(X_tau, Y_tau, theta, lambda1, lambda2), method='L-BFGS-B')
        alpha=res_alpha.x
    
    Y_prediction=prediction(X,Y,alpha,theta, lambda1,INPUT)
    ALPHA.append(alpha);THETA.append(theta)
    # PREDICTION.append(Y_prediction)
    
    omega=kernel_flows(1,X, Y, tau, omega, lambda1, lambda2)
    Y_prediction=prediction(X,Y,omega[:22],omega[22:], lambda1,INPUT)
    ALPHA.append(omega[:22]);THETA.append(omega[22:])
    PREDICTION.append(Y_prediction)
# # PREDICTION=[np.array([-1.61656420e-02,  2.31552316e-01,  6.77405496e-01, -2.23503312e-01,
# #        -7.68948722e-02, -2.24981851e-02,  2.98339288e-01,  1.40008374e-01,
# #         1.56265142e-01,  6.36430525e-02,  2.48982871e-01,  4.03686856e-01,
# #         8.25069066e-02,  1.39908940e-01,  2.14294113e-01,  5.60867157e-02,
# #         6.85256976e-02, -6.57447017e-03,  5.66027872e-02, -7.97443444e-01,
# #         1.57768616e-01, -2.13701810e-01,  1.60407651e-01,  1.50438700e-01,
# #        -1.63533390e-02, -3.31454881e-02,  1.73041669e-01, -5.62692765e-03,
# #         5.48957457e-02,  1.97247159e-01,  7.14290074e-02, -1.07343288e-01,
# #         3.17576171e-02,  1.22434140e-01, -2.70131846e-02,  1.07512410e-01,
# #        -2.15011127e-01, -1.38029472e-01, -3.13280352e-01,  4.31513134e-02,
# #        -1.30606467e-01, -3.24230785e-02, -8.77565640e-02,  2.10315788e-01,
# #         2.12730232e-04, -6.62349894e-02, -3.42496884e-01,  5.22944429e-02,
# #         6.65917788e-03,  4.69494540e-01,  1.07949385e-01,  3.90908224e-01,
# #        -2.08584560e-01,  7.50556271e-02, -1.24970738e-01,  1.65014305e-01,
# #        -1.99374952e-01, -3.07288449e-02,  1.26914345e-01, -1.46573233e-01,
# #        -5.52850971e-02,  1.37286184e-02, -1.95084460e-01,  3.27852444e-01,
# #        -1.28302456e-01,  3.92705836e-01,  1.80656739e-01, -3.98232222e-03,
# #        -8.92884916e-02,  1.13094183e-01,  1.32553446e-01, -5.46218705e-01,
# #        -2.73907675e-01, -4.14928483e-03,  1.55496052e-01,  6.96696431e-01,
# #        -2.95983793e-01,  1.73811228e-01,  1.19868319e-01,  1.42858800e-01,
# #         2.38470469e-02,  2.48969834e-01,  1.45312460e-01,  2.26701426e-01,
# #         1.27934184e-01,  1.44518385e-01,  1.12685984e-01, -6.01967908e-02,
# #         1.26114278e-01,  1.32295304e-01, -4.31396028e-01,  1.37232022e-01,
# #         1.53049446e-01,  1.42092066e-01, -1.81621734e-02,  1.22787155e-01,
# #        -1.61806574e-01,  8.88743160e-02, -1.03502180e-01, -1.22777493e-01,
# #         1.20164280e-01,  4.21682304e-02,  1.78258176e-01, -6.61489852e-02,
# #         4.13724905e-02, -1.04960451e-01,  2.02709954e-01, -2.75910462e-02,
# #         1.13522830e-01, -2.17816680e-02,  1.92147017e-02,  1.64143331e-02,
# #        -1.76170558e-02, -4.04712498e-02, -1.85263888e-02,  4.33987272e-02,
# #         8.37680968e-02,  9.74453012e-02,  5.73235847e-02,  4.51193409e-02,
# #         4.31261024e-01,  3.45319984e-02,  4.03448038e-01,  2.20061544e-01,
# #         4.70090966e-01,  1.88254011e-01,  1.58272425e-02,  4.26959448e-01,
# #         2.63117559e-01,  4.27465828e-01, -7.25961506e-01,  1.88946776e-01,
# #        -2.06827635e+00, -6.27728494e-01,  3.09164210e-01, -8.01966910e-02,
# #         3.92390367e-01, -4.65633492e+00,  1.99310549e-01,  3.99102997e-01,
# #        -4.73783097e+00, -1.01968971e+01, -2.61666815e+00, -1.31884915e-01,
# #        -6.25822117e+00, -1.41362667e+01, -5.08077955e+00, -5.75235653e+00,
# #        -6.36126454e-01,  4.56472539e-01, -1.11362598e+00,  2.43608051e-01,
# #         7.07468580e-02,  2.15195443e-01, -5.95175328e-01,  5.03308350e-01,
# #        -3.83415215e-01,  5.14207127e-01, -1.25098850e+01,  5.41650684e-01]), np.array([ 9.28078962e-03,  5.73766237e-02,  4.72870628e-01,  1.59408818e-01,
# #         4.40824506e-01, -1.24691787e-01,  4.57844101e-02,  3.55750920e-01,
# #         5.67673604e-02, -9.57890052e-01, -7.52599224e-01, -8.53860349e-01,
# #         4.88257922e-01,  1.48038499e+00,  8.94737616e-01,  6.39319063e-01,
# #         1.65322232e-01, -1.17470898e-01,  6.19406148e-02, -2.46891141e-01,
# #        -4.91950554e-03,  6.79455026e-01, -4.24253219e-02,  2.27727382e-01,
# #         6.67690136e-02,  9.18725232e-02, -2.89748816e-01, -1.04796979e-01,
# #         2.20568972e-01,  3.71129506e-01,  1.91182238e-01, -2.50351099e-01,
# #         6.78073612e-02,  6.83999181e-02,  6.10949883e-01, -3.24831836e-01,
# #        -4.95344340e-01,  2.33430321e-01, -3.20562617e-02,  4.48896451e-01,
# #         1.20667018e-01,  6.01794286e-01, -5.22628800e-02,  4.49908641e-01,
# #         3.93801242e-01, -1.49728931e-01, -3.74886725e-01,  1.28928268e-01,
# #        -1.11564492e-01,  3.40834802e-01, -6.43550484e-02, -6.50176575e-01,
# #         8.98202354e-02,  9.98893944e-02, -4.37811637e-01, -1.59079695e-01,
# #         7.64834429e-02,  2.91212497e-01, -3.36387279e-01,  5.50151754e-02,
# #        -1.28773498e-01, -3.60500527e-02, -8.89072714e-02, -1.64846110e-01,
# #        -4.06525057e-01,  1.05253110e+00, -1.07094448e+00,  1.48498620e-01,
# #        -1.15551116e+00, -5.74806265e-01,  2.88431682e-01,  1.50778853e+00,
# #         2.28373042e-02,  3.68254194e-01,  4.49293641e-01,  2.20426555e-01,
# #         9.55682461e-01,  2.20490697e+00,  1.82074899e-01, -1.61139145e-01,
# #        -5.01082477e-01, -1.09013902e-01, -8.75530034e-02, -1.97376540e-01,
# #         1.67777845e-02,  1.23106133e-01,  5.34561811e-01, -6.11503330e-01,
# #        -3.25762921e-01, -1.95314350e-01, -3.66627041e-01,  3.98862332e-02,
# #        -4.75804204e-02,  1.07174552e-01, -3.78349579e-01,  1.08909932e+00,
# #        -7.41003069e-01, -2.75493948e-01, -2.80857613e-01, -2.29439838e-01,
# #         8.55933898e-01, -1.06564224e-01, -4.95654606e-01, -6.15270094e-01,
# #        -1.72483155e-01, -3.38374868e-01, -4.22842964e-01,  9.68451397e-02,
# #         6.66139475e-01,  3.05584334e-01,  1.05980801e-01,  6.08752366e-02,
# #        -4.89240862e-01,  3.87782616e-01,  9.34576596e-02,  3.95201803e-01,
# #         8.13898496e-01,  1.60652375e+00,  2.08291180e-01,  8.50528144e-02,
# #         3.74719723e-01, -4.01132988e-02,  1.68918462e-01, -1.30628447e+00,
# #         1.26628043e+00,  2.81113133e-01,  8.67558641e-02,  2.89060389e-01,
# #        -1.34875808e+00,  1.23349582e-01, -9.38523046e-01, -1.26582197e+00,
# #         1.25333530e+00, -1.15478177e+00, -7.09888467e-01, -1.86933678e+00,
# #        -9.79287991e-01,  1.36315007e+00, -1.91746212e+00, -9.71323209e-01,
# #         3.33412774e+00,  4.03169062e+00, -1.80182927e-01,  2.41509158e+00,
# #        -4.65284393e+00, -7.99488831e+00, -7.84038728e+00, -8.05080450e+00,
# #        -7.35810165e+00, -5.72373683e+00, -7.09422318e+00, -7.55488738e+00,
# #        -8.44183144e+00, -7.22494428e+00, -8.11142623e+00, -6.02492090e+00,
# #        -6.56945056e+00, -7.56635072e+00, -1.04739162e+01, -5.38113650e+00]), np.array([-1.01314113e-01,  3.46104405e-01,  5.88967811e-01, -6.71532048e-02,
# #        -2.85859053e-02,  8.35182467e-02,  2.88904897e-01,  1.55559191e-01,
# #         1.17984176e-01,  2.38459366e-01,  1.93548453e-01,  1.03761862e-01,
# #        -3.48058556e-02,  2.52119379e-01, -1.93782165e-04,  3.62173880e-02,
# #        -1.52835155e-01, -1.04096055e-01, -3.90360232e-01, -7.81901965e-01,
# #        -6.94189833e-02,  7.37342158e-01,  1.54656039e-02, -1.26098130e-01,
# #         1.10850218e-01, -5.64257363e-03, -7.55402731e-02,  1.19323809e-01,
# #         9.72696222e-02,  3.52791826e-01,  1.11175102e-01, -2.00606457e-01,
# #         5.75293688e-02,  3.00157071e-01,  6.73983941e-02,  3.00923937e-01,
# #        -3.74835177e-01, -2.34377170e-01, -5.49042800e-01,  7.84434463e-02,
# #        -2.14724016e-01,  1.94035779e-02, -1.60413756e-02,  3.91407257e-01,
# #         1.25831472e-01, -2.04278635e-01,  3.84886425e-01, -6.80863821e-02,
# #        -2.70849577e-02, -3.54863438e-01,  1.13382543e-01, -6.49497814e-02,
# #         5.62936046e-02,  1.97894574e-01,  9.14638721e-02,  8.38404561e-02,
# #         3.99668298e-01,  4.13834634e-02,  1.04247977e-01, -8.68437315e-02,
# #         8.64548376e-02,  2.26697788e-02,  1.68356975e-01,  6.27871420e-02,
# #         9.60092160e-02,  1.21137716e-01,  1.71992466e-02,  6.59055744e-02,
# #        -2.05121461e-01,  2.00200054e-01, -8.13747548e-02, -1.72148870e-01,
# #        -2.05483143e-01, -1.18957719e-01,  1.58087624e-01,  3.11008546e-01,
# #         2.72100639e-02, -1.23579332e-02,  1.03543170e-01,  3.59717055e-02,
# #        -3.25006612e-02,  2.34350449e-01,  1.89036746e-01,  1.89338372e-01,
# #         7.03101809e-02,  1.34342299e-01, -6.69290224e-02, -6.74393316e-02,
# #         4.07802563e-02,  2.79734763e-02, -5.26271954e-03,  2.16959804e-01,
# #         1.75385925e-01,  1.21102348e-01,  3.76628919e-01,  2.94459586e-01,
# #        -1.68893916e-01,  6.55354512e-02, -1.27382673e-01, -1.34025604e-01,
# #         1.14410872e-01, -1.20044500e-01,  1.51903673e-01, -1.22691892e-01,
# #        -3.95865300e-02, -1.29334948e-01, -1.40874902e-02, -1.00487628e-01,
# #         1.12084490e-01, -5.96674604e-02,  1.15332138e-01,  1.70577296e-01,
# #         3.10159942e-01,  7.22235648e-02,  1.04627106e-01,  1.86606197e-01,
# #         1.28891727e-01,  1.17319188e-01,  2.22931676e-01,  1.85987726e-01,
# #        -2.51258122e-02, -2.97673584e-01, -1.60136174e-01,  5.09914331e-01,
# #         9.06755065e-02, -6.58202118e-02,  5.53684671e-02, -7.29910988e-02,
# #         3.90338105e-01, -1.55439451e-01, -2.81006153e-01, -6.81787577e-01,
# #        -7.99310393e-02, -1.00873884e+00, -1.46052531e+00, -2.97318054e-01,
# #        -7.89063991e-01,  1.33323384e+00, -7.31155918e-01, -7.54764855e-01,
# #         5.12885002e-01,  4.93253219e+00, -4.80523939e-01, -5.24129272e+00,
# #         1.07749679e+00,  4.78342463e+00,  1.31868423e+00,  2.05211331e+00,
# #         6.16811125e-01,  7.13132861e-01,  1.16085363e+00,  8.66433676e-01,
# #         9.45660936e-01,  4.34995416e-01,  7.86482960e-01,  5.16513180e-01,
# #         1.26546710e+00,  7.91691551e-01,  9.74731240e+00,  6.67647803e-01]), np.array([-2.25124107e-01,  2.13982924e-01,  6.18074843e-01, -2.39643394e-01,
# #         9.77246522e-02, -1.12058866e-01,  3.04094814e-01,  1.48034865e-01,
# #         4.02908909e-01,  1.71875111e-01,  2.00039236e-01,  3.15469420e-01,
# #         1.60606519e-02,  2.13337378e-01,  2.00980110e-01,  7.27219992e-03,
# #        -1.06579037e-01, -4.19352946e-02,  1.01302024e-01, -8.00683017e-01,
# #         2.27821232e-01,  1.02731853e-01,  1.80741177e-02,  3.04389239e-02,
# #         2.49617350e-02,  1.56391865e-01, -1.99582096e-01,  4.11886520e-02,
# #         5.80288477e-02,  3.15747566e-01,  5.85236393e-02, -2.29749920e-01,
# #        -9.18107968e-02,  4.40470686e-01,  9.13507445e-02,  4.00409582e-02,
# #        -4.86274201e-01, -1.20853551e-02, -5.29061395e-01,  3.03992737e-01,
# #        -1.02018341e-01, -1.32810112e-01, -3.12880035e-01,  3.89111593e-01,
# #         2.03644929e-01, -1.14873436e-01, -7.99083323e-02,  1.55896929e-01,
# #        -1.03154459e-01,  2.15132657e-01,  7.03756278e-02,  1.38709468e-01,
# #        -1.13257970e-01,  4.70343267e-01, -1.73947932e-01,  9.02461785e-02,
# #         6.99595601e-02, -5.81773367e-02,  1.52266538e-01, -1.45840979e-01,
# #        -4.68305953e-03,  8.68805457e-02,  3.71056913e-02,  2.09179138e-01,
# #        -8.90112467e-02,  1.62011006e-01,  3.79220910e-02,  7.01775107e-03,
# #        -1.48702199e-01,  1.71954697e-01,  6.27617553e-02, -3.30429898e-01,
# #        -1.57135257e-01, -4.31111604e-02,  8.45565608e-02,  5.06391132e-01,
# #        -7.59667539e-02,  7.41574336e-02,  1.44108908e-01,  4.20899579e-02,
# #         3.50188296e-02,  2.10253246e-01,  1.10681156e-01,  1.12524710e-01,
# #         7.92812766e-02,  2.11464294e-01,  7.10167774e-02, -1.31642876e-01,
# #         1.29242267e-01,  1.48785215e-01, -3.26819553e-01,  1.01773418e-01,
# #         8.37569323e-02,  2.07842467e-01,  2.26031420e-01,  9.38793268e-02,
# #        -1.98027988e-01, -8.23427772e-02, -1.36810548e-01, -2.80545446e-01,
# #         2.31753880e-01,  1.91656135e-01,  2.23606629e-01, -9.29511158e-02,
# #         5.22943468e-02, -1.39075152e-01,  1.78397839e-01, -1.43044360e-01,
# #         1.83556831e-02, -1.22619134e-01, -9.44246161e-02, -1.06073926e-01,
# #         1.09570717e-02, -9.01202949e-02, -8.85915013e-02, -1.17186686e-01,
# #        -8.47137749e-02, -8.06717795e-02, -2.92444810e-02, -2.92361690e-02,
# #         1.58862269e-01, -1.63195214e-01,  1.88648028e-01,  9.93727379e-02,
# #         1.92234135e-01, -2.26583419e-02, -1.41673305e-01,  1.72149157e-01,
# #         1.59686181e-01,  2.02002430e-01, -6.37849072e-01,  3.21348781e-01,
# #        -1.59997142e+00, -3.04699681e-01,  5.80401398e-01,  1.07663925e-01,
# #         3.80473915e-01, -4.09955499e+00,  2.39369285e-01,  3.80884059e-01,
# #        -3.86846257e+00, -8.21853675e+00, -1.94843708e+00,  1.33352056e+00,
# #        -3.71326388e+00, -1.11963265e+01, -4.89907831e+00, -5.73194736e+00,
# #        -8.34923803e-01,  4.27295733e-01, -1.38678662e+00,  1.47851844e-01,
# #        -1.85362248e-01,  2.51235075e-01, -7.96186711e-01,  3.88851249e-01,
# #        -4.84693828e-01,  1.59729518e-01, -1.48773249e+01,  1.60920458e-01]), np.array([ -0.42169807,  -0.13158939,   0.96778698,  -0.78297351,
# #         -0.6547281 ,   0.49673549,  -0.50782696,   0.58893624,
# #          0.32864114,   0.64570732,   0.55917364,  -0.09817036,
# #         -0.05174584,  -0.14825467,   0.56547678,  -0.12645445,
# #          0.05215895,   0.19277767,   0.21329079,  -1.40680185,
# #          0.59733138,   0.86081293,   0.29450233,   0.2028092 ,
# #         -0.32457543,  -0.08637548,  -0.05314981,   0.1109393 ,
# #          0.02659057,   0.23675422,   0.19636057,   0.38636543,
# #          0.15016398,  -0.93154272,  -0.59145492,  -0.45007764,
# #          0.53362448,   0.04662012,   3.08883185,   0.20347852,
# #          0.02745383,  -0.94780284,  -0.02909854,   0.21276148,
# #          0.5341429 ,   0.49677918,  -1.02881169,  -0.47442679,
# #          0.51808867,   0.36231673,  -1.82258196,  -1.98565321,
# #         -0.21009081,   0.12662592,   0.50450533,  -1.76137676,
# #          0.29276166,   0.33663434,  -1.63624766,  -0.05274442,
# #          0.27131696,   0.06526181,   0.70589043,  -0.9353724 ,
# #          0.67302453,   1.53368029,  -1.21559981,   0.29820903,
# #          0.37509377,  -1.23330467,   1.28417085,   4.07264642,
# #         -0.75558757,  -0.44043471,  -0.15059806,   1.53298095,
# #          0.89407928,  -0.75341022,   0.03243931,   0.13298847,
# #          0.40726452,   0.78381697,   0.04659292,   0.84470797,
# #          0.0845241 ,   0.08422092,  -0.26937041,   0.13469732,
# #          0.16835406,   0.11139545,  -0.70714448,  -0.16562053,
# #          0.02415943,   0.09565977,   0.63905396,  -0.58749611,
# #         -0.9275475 ,  -0.13757694,  -0.09646303,  -0.24916553,
# #         -0.54493864,  -0.59252484,   0.54043591,   0.36082253,
# #         -0.1399423 ,  -0.03790285,   0.45971342,  -0.53253791,
# #         -0.4087204 ,  -0.45380169,  -0.08408895,  -0.20609949,
# #         -0.06055083,  -0.66185352,  -0.44149935,  -0.30352726,
# #         -0.53612673,  -0.87978099,  -0.03160882,   0.0286609 ,
# #         -0.67420491,  -0.40226234,  -0.57030843,  -0.57585458,
# #         -1.49245151,  -0.13993041,  -0.35462699,  -0.63174403,
# #         -0.15018636,  -0.61206188,  -2.66897456,   0.16532256,
# #          3.64716684,  -3.08439492,  -1.97795631,   0.49866137,
# #         -0.82903252, -13.87406013,   0.12333971,  -0.80333791,
# #        -15.53142123,  -9.62591438,  -3.08051987,  -3.19539564,
# #        -17.6017599 , -15.32678784,   0.29364327,  -0.35267184,
# #          0.29033414,  -1.8348705 ,  -0.02636053,  -0.2558835 ,
# #         -0.43610339,  -2.47422524,   0.19340195,  -1.6509306 ,
# #         -3.92033688,  -0.45801521, -11.41795026,  -1.48280808]), np.array([ 8.68694085e-02,  2.99602419e-01,  1.26758971e+00,  1.28412690e-01,
# #         3.54047828e-01, -1.99715740e-01,  6.65072456e-02, -6.59446109e-01,
# #         3.09513891e-01,  7.36688829e-01,  2.93310157e-01, -4.74154284e-01,
# #        -4.39808615e-01,  2.10993544e-01, -1.02716846e-01, -3.33264252e-01,
# #        -2.51603213e-01, -6.19667639e-01, -4.82664788e-01, -7.18662688e-01,
# #         5.32023179e-02,  6.95253824e-01, -9.32774868e-02, -3.42005339e-01,
# #         7.55812630e-02, -1.68446535e-01, -1.66225439e-02,  1.83106440e-01,
# #         2.68099544e-01,  6.44475791e-01,  2.18365419e-01, -2.14043602e-02,
# #         1.21344929e-01,  7.69283424e-01,  1.62902456e-01,  3.21990637e-02,
# #        -8.30483584e-01, -4.77024885e-01, -1.10037776e+00,  3.05819675e-01,
# #        -3.89008550e-01, -1.07419633e-01, -8.72721567e-02,  8.11423423e-01,
# #         1.57011427e-01,  7.26604435e-02,  6.84377656e-01,  1.44705523e-01,
# #        -2.17536182e-01,  2.51226345e-01,  4.44617487e-01, -7.18516359e-01,
# #         1.61487817e-01,  5.69248033e-01, -3.44666411e-02,  4.55194286e-01,
# #         1.86377505e-01, -4.20113632e-01,  1.54718108e+00,  1.18399252e-01,
# #         4.47087745e-02,  3.57802927e-02, -6.51938080e-02,  3.20760841e-01,
# #        -3.56092198e-02, -1.48731207e+00, -5.62410295e-01, -5.49994756e-01,
# #        -2.39545717e-01,  1.04732121e-01, -7.25543068e-01,  5.32248499e-01,
# #         1.30484373e-01, -4.47307789e-01,  3.50416275e-01,  1.77292218e-01,
# #         2.32257952e-01, -3.63656319e-01,  6.14654538e-01, -7.69386495e-02,
# #        -3.32266950e-02,  3.45432490e-01,  4.20109708e-02,  3.52595451e-01,
# #         4.17607222e-01,  4.08150045e-01, -5.97791387e-01, -8.24811578e-02,
# #        -2.28246367e-01, -2.35996486e-01,  9.11976691e-01,  2.35450393e-02,
# #         2.22252732e-01,  4.20911876e-01,  3.74816055e-01,  3.12436845e-01,
# #        -6.32568250e-01, -1.60261975e-01,  8.37601252e-02, -4.50324795e-02,
# #         1.91998773e-01, -5.78373756e-01,  4.07261005e-01, -1.54184373e-01,
# #         6.16040598e-02,  2.95497563e-02,  6.40754514e-01,  3.26397680e-01,
# #         5.33265999e-01,  7.24375727e-01,  1.13881950e+00,  1.31496827e+00,
# #         1.00698630e+00,  1.06455214e+00,  1.19632990e+00,  1.17544812e+00,
# #         9.67640632e-01,  1.04756700e+00,  9.00303396e-01,  9.11875229e-01,
# #         1.03752577e+00,  5.48080241e-01,  6.57447804e-01,  2.20680421e+00,
# #         1.24853441e+00,  1.04825382e+00,  1.06783505e+00,  8.93759501e-01,
# #         1.47793669e+00,  6.04131563e-01, -1.04482048e+00, -1.60801550e+00,
# #        -2.96148820e+00, -4.02610552e+00, -5.24321391e+00,  6.38264773e-01,
# #        -1.77250902e+00,  5.63510689e+00, -1.83380603e+00, -1.63855131e+00,
# #         3.19858022e+00,  2.13058879e+01, -5.17613440e+00, -1.96831799e+01,
# #        -6.76717031e+00,  2.31100636e+01,  1.40063499e+01,  1.94852763e+01,
# #         7.86634338e+00,  3.84888670e+00,  9.21298656e+00,  5.37462302e+00,
# #         4.90704353e+00,  5.76370783e-01,  8.45496618e+00,  3.97757604e+00,
# #         2.51653180e+00,  5.45839832e+00,  7.66997226e+01,  4.66029818e+00]), np.array([-1.65081746e-01,  4.68298947e-01,  7.48419626e-01, -4.45677651e-01,
# #        -2.78436693e-01, -3.32708831e-02, -5.15439350e-02,  1.55410572e+00,
# #         4.21048255e-01, -2.96286920e-01,  3.72810572e-01, -1.11767276e+00,
# #         8.53964373e-01,  1.16271530e+00,  4.56019978e-01,  9.03542255e-01,
# #        -2.15100588e-02,  1.51480827e-01, -4.23475482e-01, -1.10800022e+00,
# #         4.16616240e-02,  1.00520858e+00, -1.80805095e-02,  2.59328366e-01,
# #        -1.03861513e-01, -2.67091849e-01, -4.85725390e-01,  1.14653315e-02,
# #         7.48399534e-02, -3.95294526e-03,  7.79612274e-02, -1.85959906e-01,
# #         3.03230642e-02, -2.21455871e-01, -2.01407959e-01, -1.64010280e-01,
# #        -1.29017909e-01,  1.00047684e-01,  4.41809424e-01,  7.18962235e-03,
# #         8.93031717e-02, -3.33206844e-01,  1.22060830e-01,  4.23481830e-02,
# #        -3.14535954e-01, -4.80568161e-01, -5.69369151e-02, -4.94287031e-01,
# #         1.99860269e-01,  2.34208019e-01, -3.75621092e-01, -9.98573577e-01,
# #        -1.70495613e-02, -1.80698460e-01,  5.45553100e-01, -6.60581839e-01,
# #         1.05215566e+00,  1.00950149e+00,  8.94370949e-01, -4.30209365e-01,
# #         3.83553915e-01, -1.89063967e-01,  6.46323638e-01, -5.59634988e-01,
# #         6.35676888e-01, -1.30372828e-01, -1.58570097e+00,  1.28195968e+00,
# #        -5.13636821e-01, -2.60000857e+00,  2.04136708e-01,  2.04517287e+00,
# #         5.93979051e-01,  1.11640013e-01, -3.73744911e-01,  1.23137218e+00,
# #         1.53210273e+00, -1.12162752e+00, -3.81258445e-01, -3.82740356e-02,
# #        -8.36428760e-01,  7.15764766e-01,  1.69262219e-01,  5.44372823e-01,
# #        -2.53868557e-01, -2.47615919e-01, -9.50592306e-02, -6.98162225e-01,
# #        -1.52277924e-01, -1.36030097e-02,  7.74700865e-02,  2.90551155e-01,
# #        -4.88815371e-02, -2.63502836e-01,  2.83425758e-01,  6.91101231e-01,
# #        -3.47365512e-01,  3.29272451e-01, -7.44344178e-02,  1.72012727e-01,
# #         5.92854856e-01,  6.54443581e-01, -2.20827554e-01,  2.28516588e-01,
# #         1.27553235e-01, -2.15608330e-02, -1.04391709e+00, -5.34866676e-01,
# #         6.19513499e-02, -6.79406756e-01, -4.45381133e-01, -4.42296684e-01,
# #         2.02944035e-02, -3.24582266e-01, -5.15085022e-01, -2.02054584e-01,
# #        -2.01337275e-01, -1.21412925e-01,  6.57850202e-02, -5.60665926e-02,
# #        -4.71056417e-01, -5.61101630e-01, -3.81034600e-01, -8.81480554e-01,
# #        -5.01992084e-01, -2.76132620e-01, -3.40756704e-01, -4.43267108e-01,
# #        -4.47141298e-01, -4.41138322e-01, -1.26630532e+00,  5.01278443e-01,
# #        -3.16137856e-01,  4.85580215e-01,  1.59498277e+00, -1.25636044e+00,
# #         3.49601558e-01, -9.86816393e-01,  7.46217060e-01,  2.58735326e-01,
# #         8.47869856e-01, -5.05699489e+00,  1.83793489e+00,  5.45122043e-01,
# #         7.08555242e+00,  2.04063660e+00, -1.41153937e+01, -1.52478895e+01,
# #        -5.52238025e+00,  4.79130733e-02, -6.54727606e+00, -2.64846182e+00,
# #        -3.30293313e+00,  1.55605013e+00, -5.52922815e+00, -1.08868223e+00,
# #        -1.47318433e-01, -3.06366009e+00, -2.74474803e+01, -1.37830392e+00]), np.array([-1.27339539e-01,  1.05882119e-01,  3.19808639e-01, -4.31263598e-01,
# #        -8.61400767e-02, -1.11656485e-01,  4.14645855e-01,  3.34464611e-01,
# #         3.37031210e-01,  2.26088423e-01,  2.60641222e-01,  5.24528832e-01,
# #         1.82981841e-01,  2.54639217e-01,  3.11045686e-01,  1.46215745e-01,
# #         4.44149587e-02,  1.20711576e-01,  2.00971329e-01, -8.19639857e-01,
# #         2.46271707e-01, -2.87040600e-01,  1.18240726e-01,  1.63099180e-01,
# #         4.53130969e-02,  1.29351358e-01, -6.13227398e-02, -9.22065587e-03,
# #         1.03207157e-02,  2.20857912e-01,  4.57273728e-02, -2.78181067e-01,
# #        -5.86418824e-02,  3.04928255e-01, -4.48125432e-02,  1.52755987e-01,
# #        -1.45807361e-01, -2.19195763e-02, -3.64217936e-01,  1.26396145e-01,
# #        -9.18483126e-02, -1.44570671e-01, -2.50622127e-01,  2.55172441e-01,
# #         4.95911060e-02, -2.01666200e-01, -3.24891679e-01,  7.55980003e-02,
# #        -6.69665975e-02,  4.62593343e-01,  9.61120512e-02,  1.96101403e-01,
# #        -2.33615871e-01,  2.60540114e-01, -1.21570302e-01,  1.30244030e-01,
# #        -7.54020338e-02,  9.03135808e-02,  1.58149757e-01, -2.07191827e-01,
# #         2.92857531e-02,  1.50261949e-01, -8.73188235e-02,  3.10556009e-01,
# #        -7.25982883e-02,  2.56026177e-01, -1.50040469e-02,  2.34981506e-01,
# #        -1.09821086e-01,  1.38486552e-01,  2.84940457e-01, -4.01007149e-01,
# #        -3.53116473e-01, -7.49906769e-02, -2.31328714e-02,  6.74851514e-01,
# #        -2.30113043e-01,  4.70405926e-01, -5.24652684e-02,  1.40355826e-01,
# #         4.47945121e-02,  1.25274618e-01,  1.30491079e-01,  8.10239103e-02,
# #        -9.60882141e-03,  5.88891689e-02,  7.50088660e-02, -5.11119030e-02,
# #         2.59864467e-01,  2.61311733e-01, -4.42876734e-01,  1.41491529e-01,
# #         2.42804858e-02,  5.94079575e-02,  5.69000981e-02,  7.89121129e-02,
# #        -9.29389978e-02,  2.58181417e-02, -1.97341325e-01, -2.23299583e-01,
# #         1.62020939e-01,  2.89557406e-01,  1.29686773e-01, -6.54414547e-02,
# #         3.38120243e-02, -1.85298521e-01,  3.89892423e-02, -2.19479735e-01,
# #        -8.37736773e-02, -2.74640676e-01, -3.45307078e-01, -4.12992570e-01,
# #        -2.41630376e-01, -3.67579878e-01, -3.83082137e-01, -3.83973314e-01,
# #        -2.84862238e-01, -2.74597530e-01, -2.40932347e-01, -2.37486724e-01,
# #         1.67180849e-01, -1.48676849e-01,  2.46757006e-01, -1.10767552e-01,
# #         1.80091682e-01, -5.77544972e-02, -2.06307293e-01,  1.99345670e-01,
# #         5.07681618e-02,  2.69329942e-01, -7.02169831e-01,  5.85263799e-01,
# #        -1.90119379e+00,  1.33645094e-01,  1.56966546e+00,  5.19029396e-02,
# #         8.00070068e-01, -6.05232493e+00,  6.15952845e-01,  7.80039503e-01,
# #        -5.28790905e+00, -1.28074423e+01, -2.14738845e+00,  6.50852707e+00,
# #        -2.01585874e+00, -1.45905159e+01, -8.03961642e+00, -9.86989695e+00,
# #        -2.15032817e+00, -2.81282156e-01, -3.31281194e+00, -8.03819044e-01,
# #        -1.00143403e+00,  1.24060142e-01, -2.27572583e+00, -3.18576161e-01,
# #        -1.08849176e+00, -7.13878885e-01, -2.77256580e+01, -5.65623063e-01]), np.array([-6.41946612e-02,  1.51476854e-01,  3.13936354e-01, -1.93733170e-01,
# #        -1.01012271e-01,  2.76295753e-02,  3.40994416e-01,  5.82815904e-01,
# #         1.73274128e-01,  3.17340317e-01,  2.10674768e-01,  9.01414978e-02,
# #        -4.75538967e-02,  4.95440051e-01,  7.06394140e-02, -4.71390386e-02,
# #        -5.18133794e-02,  1.63581947e-01, -5.31750953e-02, -8.51756714e-01,
# #         7.31853721e-02,  5.18159059e-01,  4.21205332e-02, -7.88552662e-03,
# #         2.36249313e-01, -6.11911845e-02, -2.99355718e-01,  1.68218783e-01,
# #         2.34860874e-03,  1.80734407e-01,  3.28665463e-02, -1.12292894e-01,
# #        -2.58405895e-03,  2.00899264e-01, -5.96528077e-02,  1.00358485e-01,
# #        -3.26725884e-01, -8.51394886e-02, -2.41923815e-01,  3.44345371e-02,
# #        -9.00263035e-02, -9.70385352e-02, -1.14634049e-01,  2.10492450e-01,
# #        -4.83330022e-02, -9.21132780e-02,  3.30405417e-01, -9.83409599e-02,
# #        -1.01676380e-01, -1.30621998e-01,  1.12201630e-01, -5.14432906e-01,
# #         6.08318973e-02,  2.24398018e-01,  4.71306389e-02,  1.13970506e-01,
# #         5.31798848e-01,  1.30014057e-01,  6.29170038e-01,  5.65403082e-02,
# #         6.61174676e-02,  7.87846179e-02,  3.64347795e-01,  1.36814924e-01,
# #         5.21000457e-02,  4.27465427e-02, -1.01165145e+00,  1.62617767e-01,
# #        -2.58272409e-01, -6.31291479e-01,  2.08642703e-01,  4.52583440e-01,
# #        -8.51284766e-02, -1.05299019e-01, -1.37580016e-04,  7.41445841e-01,
# #         3.25016419e-01, -1.80653867e-01,  6.56256509e-02,  6.25741689e-02,
# #        -2.40608271e-02,  1.58875925e-01,  4.99676417e-02,  1.44694591e-01,
# #         7.22082677e-02,  8.20752068e-02, -1.01745830e-01, -5.51051891e-02,
# #         7.41925654e-02,  7.43410810e-02,  2.32964303e-02,  1.56584411e-02,
# #         4.79121148e-02,  8.57452192e-02,  3.54356720e-01,  4.83655088e-02,
# #        -1.94925248e-01,  1.18529981e-01, -1.21357892e-01, -1.40717037e-01,
# #         1.77077824e-01,  1.11206057e-01,  8.24508658e-03, -2.32350767e-02,
# #         8.89838315e-02, -1.17228706e-01, -5.67887825e-02, -5.30470267e-02,
# #         9.54154051e-02, -5.77204653e-02, -2.64380004e-02, -4.55165379e-02,
# #         3.44971094e-02, -1.13576524e-01, -8.44075499e-02, -1.86694036e-02,
# #         1.71834647e-02,  2.06383683e-02,  3.07831793e-02,  2.45093614e-02,
# #        -2.38921217e-01, -3.47666606e-01, -2.15427561e-01,  1.27937921e-01,
# #        -2.33330354e-01, -2.96535847e-01, -2.80215835e-01, -2.26391730e-01,
# #         1.46782738e-01, -1.97226020e-01, -5.31870253e-01, -1.85040236e-01,
# #        -1.68238971e-01, -4.87857836e-01, -8.31484456e-02,  4.50516693e-02,
# #        -5.58562766e-02, -1.01053280e+00, -1.51673849e-01, -5.17002285e-02,
# #        -7.66849235e-01, -2.46681334e+00, -7.16662826e-01, -1.95996256e-02,
# #        -1.80965322e+00, -2.51212745e+00,  3.85814998e-01,  3.99129223e-01,
# #         3.00093279e-01,  1.27500484e-02,  7.81057786e-01,  4.32649704e-01,
# #         4.57924111e-01, -2.75989673e-02,  3.25725727e-01,  8.64530231e-02,
# #        -3.28507063e-01, -4.23013820e-02, -7.39462714e-02,  1.88273410e-02]), np.array([-2.05971722e-01,  6.18294244e-01,  5.08397296e-01, -1.89059307e-01,
# #        -1.33571426e-01,  3.15959123e-01,  3.55364849e-01,  4.25307665e-01,
# #         1.17422931e-01, -1.87715849e-01,  7.10211041e-02,  2.81566161e-01,
# #        -9.55651868e-02,  1.83835420e-01,  4.45196367e-02, -4.37045660e-02,
# #        -3.00483983e-01,  1.01677565e-01, -3.87505917e-01, -7.73729641e-01,
# #        -1.13099713e-02,  7.08771707e-01, -4.41127645e-02, -2.19815573e-01,
# #         7.73349807e-02, -3.27812135e-02,  4.71248903e-02,  1.97556789e-01,
# #         9.63758739e-02,  2.13663921e-01,  9.31115266e-02, -1.26477864e-01,
# #         4.76734924e-02, -1.21948613e-01,  1.31542201e-01,  4.81530710e-02,
# #        -3.91052666e-01, -2.00502911e-01, -5.47128175e-02,  1.24862579e-01,
# #        -1.88611974e-01,  5.52726927e-02, -5.68071152e-02,  2.05027827e-01,
# #         1.63156632e-01, -8.44097561e-02,  2.13321538e-01, -2.94983421e-02,
# #         8.56884587e-02, -5.98829650e-01,  2.93194974e-02,  1.94002943e-01,
# #         2.99966421e-02,  7.65087834e-02, -1.07003130e-02, -8.13111615e-02,
# #         2.77195658e-01,  1.37166608e-01, -4.88892322e-01, -7.77807724e-03,
# #         8.05002351e-03, -2.07373122e-02,  2.06962648e-01, -1.44854974e-01,
# #         2.06618472e-02,  1.03423524e+00,  5.91193449e-01,  1.05044877e-01,
# #        -1.62169688e-01,  3.49296845e-01, -1.11816163e-01, -6.63023261e-01,
# #        -2.33174865e-01,  1.16357060e-01,  8.29608819e-02,  3.49088600e-01,
# #        -3.34460824e-02, -2.01717976e-02, -3.42503115e-03,  1.21218035e-01,
# #        -1.00440898e-02,  3.65596956e-01,  1.55098017e-01,  2.76709714e-01,
# #         2.03127361e-02,  9.97517115e-02,  2.73421254e-01, -1.18011971e-01,
# #         1.82848973e-01,  2.08690029e-01, -4.68394085e-01,  1.66036839e-01,
# #         1.25354115e-01,  8.76413306e-02,  3.43098866e-01,  1.48201647e-01,
# #        -9.44702191e-02,  9.85928623e-02, -7.81597510e-02, -1.61032922e-01,
# #         1.27261713e-01, -1.82378685e-01,  1.12079354e-01, -2.00667254e-01,
# #         1.48419831e-02, -8.77163054e-02,  2.10038178e-01, -2.19501747e-02,
# #         2.37759989e-01,  5.91953468e-02,  2.01953672e-01,  2.66678341e-01,
# #         4.58724826e-01,  2.18803254e-01,  2.34449617e-01,  2.65072723e-01,
# #         2.56266332e-01,  2.51002746e-01,  2.67070047e-01,  2.33412007e-01,
# #         3.52690337e-01,  6.64146328e-03,  1.44815435e-01,  7.40325388e-01,
# #         5.25254628e-01,  3.68499779e-01,  4.24239273e-01,  2.82687480e-01,
# #         6.34277930e-01,  1.75172187e-01, -3.94919884e-02, -7.72853222e-01,
# #         3.81881443e-01, -1.00616391e+00, -1.81998437e+00, -9.24706779e-01,
# #        -7.42486682e-01,  3.23076367e-01, -8.83913623e-01, -6.83018189e-01,
# #        -9.89521558e-01, -5.67410255e+00, -4.97257189e-01, -7.25451680e+00,
# #        -3.11518787e+00, -1.13734214e+01, -7.23027722e-01,  4.48095530e-02,
# #        -5.20850613e-01,  1.36697141e+00,  3.27058095e-01,  1.31712717e+00,
# #         1.04068212e+00,  9.61552880e-01, -2.70331672e-01,  1.23503429e+00,
# #         2.34703861e+00,  1.42304044e+00,  1.27329802e+01,  1.55861647e+00])]
# # ALPHA= [np.array([0.0676222 , 0.5459778 , 0.0970425 , 0.8506866 , 0.01392905,
# #        0.24430682, 0.47075185, 0.0485893 , 0.66910386, 0.03055676,
# #        0.17255215, 0.98789144, 0.9300836 , 0.66342837, 0.95525104,
# #        0.22540115, 0.28437367, 0.8751638 , 0.1326497 , 0.7876431 ,
# #        0.41951478, 0.42377   ]), np.array([0.09019811, 0.8281453 , 0.651012  , 0.7218854 , 0.8979414 ,
# #        0.77603287, 0.4246073 , 0.11816986, 0.78230953, 0.53176105,
# #        0.7513376 , 0.82318324, 0.13579886, 0.23382305, 0.19098094,
# #        0.39020437, 0.5861438 , 0.3762712 , 0.5230755 , 0.88999337,
# #        0.91034967, 0.11100129]), np.array([0.8412019 , 0.15049455, 0.08707344, 0.50772834, 0.34092042,
# #        0.8343024 , 0.24328431, 0.09974807, 0.21993989, 0.8430233 ,
# #        0.99848056, 0.54048383, 0.35693055, 0.6627519 , 0.8148225 ,
# #        0.75317216, 0.46296844, 0.157068  , 0.27349994, 0.86395377,
# #        0.38850716, 0.56704885]), np.array([0.9151031 , 0.9353622 , 0.78554165, 0.6318496 , 0.9162669 ,
# #        0.8874302 , 0.5891699 , 0.95894545, 0.23151079, 0.35648066,
# #        0.6890923 , 0.29426607, 0.4572929 , 0.03682494, 0.8988547 ,
# #        0.83345747, 0.1879999 , 0.22934404, 0.78019583, 0.34945694,
# #        0.41996115, 0.642768  ]), np.array([0.59423304, 0.690818  , 0.18602206, 0.00827321, 0.22038382,
# #        0.4915409 , 0.12015531, 0.7595886 , 0.81175786, 0.19922754,
# #        0.75769264, 0.98636264, 0.88225627, 0.1735981 , 0.9042196 ,
# #        0.5945591 , 0.6895    , 0.71437067, 0.9784668 , 0.6531538 ,
# #        0.92720973, 0.02801668]), np.array([0.9044618 , 0.9947998 , 0.8894173 , 0.9799005 , 0.65986156,
# #        0.7350819 , 0.8375834 , 0.17101742, 0.27775475, 0.68007195,
# #        0.56292516, 0.6355288 , 0.49666235, 0.8076897 , 0.7716943 ,
# #        0.244525  , 0.37662253, 0.17709744, 0.90449435, 0.23366822,
# #        0.74330217, 0.71372855]), np.array([0.7339791 , 0.2977102 , 0.26087037, 0.12416782, 0.38028154,
# #        0.94399923, 0.56011474, 0.5588018 , 0.886005  , 0.31095546,
# #        0.6909143 , 0.3325868 , 0.7234799 , 0.7347048 , 0.669985  ,
# #        0.5439983 , 0.22943844, 0.9780644 , 0.9816883 , 0.6031849 ,
# #        0.3842651 , 0.24952465]), np.array([0.38827077, 0.39661017, 0.53400683, 0.5205485 , 0.4843285 ,
# #        0.60229933, 0.43070042, 0.2644685 , 0.48800662, 0.769768  ,
# #        0.9844121 , 0.49797165, 0.14487258, 0.16080895, 0.72074884,
# #        0.2335484 , 0.9456369 , 0.56691563, 0.03315782, 0.5927281 ,
# #        0.33556217, 0.54562247]), np.array([0.974863  , 0.7591993 , 0.53508675, 0.6070742 , 0.15304513,
# #        0.27661583, 0.567037  , 0.93972504, 0.675985  , 0.1072678 ,
# #        0.7623554 , 0.01756675, 0.22579539, 0.5307153 , 0.98394316,
# #        0.33248886, 0.52662873, 0.8690248 , 0.8696038 , 0.72291493,
# #        0.66755867, 0.16319276]), np.array([0.31790897, 0.32623568, 0.4063389 , 0.27256772, 0.42421633,
# #        0.6003264 , 0.72499746, 0.9173372 , 0.0242142 , 0.86271846,
# #        0.28379288, 0.23358826, 0.69697934, 0.9972797 , 0.25383437,
# #        0.8317118 , 0.04748168, 0.09194369, 0.62155825, 0.67147404,
# #        0.4390689 , 0.7374011 ])]
# print(PREDICTION)
# print("")
print(ALPHA)
# ALPHA=[[0.00652142382, 0.214858833, 0.0535227112, 0.0432180725, 0.718572405, -0.00161647166, -0.677084684, -0.0716287362, -0.0897466743, -0.0450714554, 0.0479673291, -0.352821093, 9.77870738, -0.902644026, -1.88029084, 0.91821009, 2.01592225, 1.5175742, -2.2744451, 0.00223024542, -0.447704836, 0.407936804], np.array([0.05000027, 1.00015722, 0.05000023, 0.05000089, 1.00016806,
#        0.05001151, 0.05000323, 0.05000181, 0.05000023, 0.05000031,
#        0.05000053, 0.0499989 , 0.05000005, 0.04999908, 0.05000121,
#        0.05000034, 0.04999995, 0.05000122, 0.04999863, 0.05000011,
#        0.05001184, 0.99968998]), np.array([0.05000027, 1.00015722, 0.05000023, 0.05000089, 1.00016806,
#        0.05001151, 0.05000323, 0.05000181, 0.05000023, 0.05000031,
#        0.05000053, 0.0499989 , 0.05000005, 0.04999908, 0.05000121,
#        0.05000034, 0.04999995, 0.05000122, 0.04999863, 0.05000011,
#        0.05001184, 0.99968998]), np.array([0.05000027, 1.00015722, 0.05000023, 0.05000089, 1.00016806,
#        0.05001151, 0.05000323, 0.05000181, 0.05000023, 0.05000031,
#        0.05000053, 0.0499989 , 0.05000005, 0.04999908, 0.05000121,
#        0.05000034, 0.04999995, 0.05000122, 0.04999863, 0.05000011,
#        0.05001184, 0.99968998]), np.array([0.05000027, 1.00015722, 0.05000023, 0.05000089, 1.00016806,
#        0.05001151, 0.05000323, 0.05000181, 0.05000023, 0.05000031,
#        0.05000053, 0.0499989 , 0.05000005, 0.04999908, 0.05000121,
#        0.05000034, 0.04999995, 0.05000122, 0.04999863, 0.05000011,
#        0.05001184, 0.99968998])]
# print("")
print(THETA)


# #firm=0
# ALPHA=[[0.00652142382, 0.214858833, 0.0535227112, 0.0432180725, 0.718572405, -0.00161647166, -0.677084684, -0.0716287362, -0.0897466743, -0.0450714554, 0.0479673291, -0.352821093, 9.77870738, -0.902644026, -1.88029084, 0.91821009, 2.01592225, 1.5175742, -2.2744451, 0.00223024542, -0.447704836, 0.407936804], np.array([0.05, 1.  , 0.05, 0.05, 1.  , 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
#        0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1.  ]), np.array([0.04999984, 0.99976647, 0.04999993, 0.04999723, 0.99923151,
#        0.04996281, 0.04998725, 0.04999291, 0.04999995, 0.04999986,
#        0.04999728, 0.05000541, 0.05000008, 0.0500051 , 0.04999469,
#        0.04999986, 0.0500002 , 0.04999454, 0.05000533, 0.04999992,
#        0.04999938, 1.00949347])]
# THETA=[[0.999999997, 1.97482101, 2.06639488, 3.09997366, 1.00000001, 1.00000005, 0.146935844, 0.000914729754, 36667.7292, 0.953910632, 0.059506145, 1.00001516, 1.0000023, 0.999999794, 1.00000008, 0.999999962, 0.999999965, 0.999999982, 1.0, 1.00000002, 0.999999982, 0.999999982, 1.00000007, 0.99999989, 1.00000052, 0.9999997, 0.999999892, 0.999999978, 0.999961542, 0.999999385, 0.999999164, 1.00000001, 0.999999942, 1.00001301, 0.314400708], np.array([9.99999986e-01, 1.97464248e+00, 2.06624217e+00, 3.09974055e+00,
    #    9.99999956e-01, 1.00000010e+00, 3.44345977e-02, 3.29485734e-03,
    #    3.66677292e+04, 9.53940526e-01, 5.98054976e-02, 1.00000018e+00,
    #    1.00000039e+00, 1.00000010e+00, 1.00000010e+00, 1.00000042e+00,
    #    9.99999987e-01, 9.99999889e-01, 9.99999847e-01, 1.00000007e+00,
    #    9.99999752e-01, 9.99999873e-01, 1.00000001e+00, 9.99999914e-01,
    #    9.99999962e-01, 9.99999777e-01, 1.00000019e+00, 9.99999966e-01,
    #    9.99999977e-01, 1.00000030e+00, 9.99999679e-01, 9.99999961e-01,
    #    9.99999943e-01, 1.00000234e+00, 4.45428601e-03]), np.array([9.99999986e-01, 1.97463995e+00, 2.06624020e+00, 3.09973726e+00,
    #    9.99999955e-01, 1.00000010e+00, 3.30071985e-02, 3.28911545e-03,
    #    3.66677292e+04, 9.53940542e-01, 5.98055673e-02, 1.00000018e+00,
    #    1.00000040e+00, 1.00000010e+00, 1.00000010e+00, 1.00000043e+00,
    #    9.99999987e-01, 9.99999888e-01, 9.99999845e-01, 1.00000008e+00,
    #    9.99999748e-01, 9.99999872e-01, 1.00000001e+00, 9.99999913e-01,
    #    9.99999961e-01, 9.99999774e-01, 1.00000020e+00, 9.99999966e-01,
    #    9.99999977e-01, 1.00000030e+00, 9.99999675e-01, 9.99999960e-01,
    #    9.99999943e-01, 1.00000237e+00, 5.00804813e-04])]

# ALPHA=[np.array([0.0535918 , 1.15262469, 0.0491477 , 0.06931729, 5.21158718,
#        1.34613464, 0.16571573, 0.1035349 , 0.04917956, 0.05179832,
#        0.06776219, 0.01094588, 0.05174302, 0.0130162 , 0.08949669,
#        0.05041833, 0.04210235, 0.0848374 , 0.01658894, 0.04941079,
#        1.3423789 , 1.7076353 ]), np.array([0.05000003, 1.00000734, 0.04999998, 0.05000057, 1.0001745 ,
#        0.0500166 , 0.05000232, 0.05000093, 0.05000002, 0.05      ,
#        0.05000056, 0.04999886, 0.05000003, 0.04999889, 0.05000112,
#        0.05000004, 0.04999971, 0.05000096, 0.04999892, 0.05000002,
#        0.05001269, 0.99992932])]
# THETA=[np.array([9.99999459e-01, 1.97484377e+00, 2.06639926e+00, 3.10000279e+00,
#        9.99999956e-01, 9.99999241e-01, 1.47410970e-01, 1.95967531e-03,
#        3.66677292e+04, 9.53950363e-01, 5.98013161e-02, 9.99999279e-01,
#        9.99999417e-01, 9.99999083e-01, 9.99999519e-01, 9.99999371e-01,
#        9.99999496e-01, 9.99999310e-01, 9.99999680e-01, 9.99999210e-01,
#        9.99999713e-01, 9.99999149e-01, 9.99999923e-01, 9.99999679e-01,
#        9.99999280e-01, 1.00000020e+00, 9.99999812e-01, 9.99999256e-01,
#        9.99999313e-01, 9.99999419e-01, 9.99999575e-01, 1.00000037e+00,
#        9.99999783e-01, 9.99994419e-01, 3.19121063e-01]), np.array([9.99999990e-01, 1.97484773e+00, 2.06640318e+00, 3.10000772e+00,
#        1.00000002e+00, 9.99999962e-01, 1.47410979e-01, 1.95949864e-03,
#        3.66677292e+04, 9.53948622e-01, 5.98010923e-02, 9.99999960e-01,
#        9.99999911e-01, 9.99999939e-01, 9.99999853e-01, 9.99999946e-01,
#        1.00000000e+00, 1.00000002e+00, 9.99999957e-01, 9.99999984e-01,
#        1.00000007e+00, 1.00000001e+00, 9.99999977e-01, 9.99999993e-01,
#        1.00000002e+00, 1.00000004e+00, 9.99999949e-01, 9.99999976e-01,
#        1.00000019e+00, 1.00000017e+00, 1.00000008e+00, 9.99999971e-01,
#        1.00000000e+00, 9.99995935e-01, 3.19117967e-01])]








# THETA=[[0.999999997, 1.97482101, 2.06639488, 3.09997366, 1.00000001, 1.00000005, 0.146935844, 0.000914729754, 36667.7292, 0.953910632, 0.059506145, 1.00001516, 1.0000023, 0.999999794, 1.00000008, 0.999999962, 0.999999965, 0.999999982, 1.0, 1.00000002, 0.999999982, 0.999999982, 1.00000007, 0.99999989, 1.00000052, 0.9999997, 0.999999892, 0.999999978, 0.999961542, 0.999999385, 0.999999164, 1.00000001, 0.999999942, 1.00001301, 0.314400708], np.array([1.00000004e+00, 1.97506181e+00, 2.06642304e+00, 3.10028070e+00,
#        1.00000014e+00, 1.00000027e+00, 1.47658025e-01, 8.15159655e-04,
#        3.66677292e+04, 9.53947127e-01, 5.98012263e-02, 1.00000015e+00,
#        9.99999988e-01, 1.00000013e+00, 1.00000006e+00, 1.00000019e+00,
#        1.00000016e+00, 1.00000022e+00, 9.99999912e-01, 9.99999995e-01,
#        1.00000017e+00, 9.99999869e-01, 9.99999951e-01, 1.00000021e+00,
#        9.99999948e-01, 1.00000036e+00, 1.00000005e+00, 9.99999971e-01,
#        1.00000025e+00, 1.00000017e+00, 1.00000000e+00, 1.00000006e+00,
#        9.99999902e-01, 9.99996302e-01, 3.18852833e-01]), np.array([1.00000004e+00, 1.97506181e+00, 2.06642304e+00, 3.10028070e+00,
#        1.00000014e+00, 1.00000027e+00, 1.47658025e-01, 8.15159655e-04,
#        3.66677292e+04, 9.53947127e-01, 5.98012263e-02, 1.00000015e+00,
#        9.99999988e-01, 1.00000013e+00, 1.00000006e+00, 1.00000019e+00,
#        1.00000016e+00, 1.00000022e+00, 9.99999912e-01, 9.99999995e-01,
#        1.00000017e+00, 9.99999869e-01, 9.99999951e-01, 1.00000021e+00,
#        9.99999948e-01, 1.00000036e+00, 1.00000005e+00, 9.99999971e-01,
#        1.00000025e+00, 1.00000017e+00, 1.00000000e+00, 1.00000006e+00,
#        9.99999902e-01, 9.99996302e-01, 3.18852833e-01]), np.array([1.00000004e+00, 1.97506181e+00, 2.06642304e+00, 3.10028070e+00,
#        1.00000014e+00, 1.00000027e+00, 1.47658025e-01, 8.15159655e-04,
#        3.66677292e+04, 9.53947127e-01, 5.98012263e-02, 1.00000015e+00,
#        9.99999988e-01, 1.00000013e+00, 1.00000006e+00, 1.00000019e+00,
#        1.00000016e+00, 1.00000022e+00, 9.99999912e-01, 9.99999995e-01,
#        1.00000017e+00, 9.99999869e-01, 9.99999951e-01, 1.00000021e+00,
#        9.99999948e-01, 1.00000036e+00, 1.00000005e+00, 9.99999971e-01,
#        1.00000025e+00, 1.00000017e+00, 1.00000000e+00, 1.00000006e+00,
#        9.99999902e-01, 9.99996302e-01, 3.18852833e-01]), np.array([1.00000004e+00, 1.97506181e+00, 2.06642304e+00, 3.10028070e+00,
#        1.00000014e+00, 1.00000027e+00, 1.47658025e-01, 8.15159655e-04,
#        3.66677292e+04, 9.53947127e-01, 5.98012263e-02, 1.00000015e+00,
#        9.99999988e-01, 1.00000013e+00, 1.00000006e+00, 1.00000019e+00,
#        1.00000016e+00, 1.00000022e+00, 9.99999912e-01, 9.99999995e-01,
#        1.00000017e+00, 9.99999869e-01, 9.99999951e-01, 1.00000021e+00,
#        9.99999948e-01, 1.00000036e+00, 1.00000005e+00, 9.99999971e-01,
#        1.00000025e+00, 1.00000017e+00, 1.00000000e+00, 1.00000006e+00,
#        9.99999902e-01, 9.99996302e-01, 3.18852833e-01])]

# [0.326, 0.358, 0.358, 0.358, 0.358, 0.358]
# [1.032, 1.697, 1.697, 1.697, 1.697, 1.697]
# [1.377, 2.147, 2.147, 2.147, 2.147, 2.147]


# ALPHA=[[0.00652142382, 0.214858833, 0.0535227112, 0.0432180725, 0.718572405, -0.00161647166, -0.677084684, -0.0716287362, -0.0897466743, -0.0450714554, 0.0479673291, -0.352821093, 9.77870738, -0.902644026, -1.88029084, 0.91821009, 2.01592225, 1.5175742, -2.2744451, 0.00223024542, -0.447704836, 0.407936804], np.array([ 1.04815696e-01,  1.95247181e+00,  1.97916647e-01,  5.18492352e-01,
#         2.75698232e+00,  1.38067574e+00,  7.07310277e-02,  1.61057326e-02,
#         3.95076279e-01,  7.14981456e-01,  6.87968925e-01,  2.18295049e-01,
#         5.96773259e-01,  6.10440558e-01,  8.54410038e-01,  2.99716010e-02,
#         1.37548248e-01,  1.80905221e-01,  1.22371765e-01,  4.59057551e-01,
#        -2.22448005e-03,  2.34988793e+00]), np.array([0.05000027, 1.00015722, 0.05000023, 0.05000089, 1.00016806,
#        0.05001151, 0.05000323, 0.05000181, 0.05000023, 0.05000031,
#        0.05000053, 0.0499989 , 0.05000005, 0.04999908, 0.05000121,
#        0.05000034, 0.04999995, 0.05000122, 0.04999863, 0.05000011,
#        0.05001184, 0.99968998])]
# THETA=[[0.999999997, 1.97482101, 2.06639488, 3.09997366, 1.00000001, 1.00000005, 0.146935844, 0.000914729754, 36667.7292, 0.953910632, 0.059506145, 1.00001516, 1.0000023, 0.999999794, 1.00000008, 0.999999962, 0.999999965, 0.999999982, 1.0, 1.00000002, 0.999999982, 0.999999982, 1.00000007, 0.99999989, 1.00000052, 0.9999997, 0.999999892, 0.999999978, 0.999961542, 0.999999385, 0.999999164, 1.00000001, 0.999999942, 1.00001301, 0.314400708], np.array([1.00019014e+00, 2.09206738e+00, 2.07830555e+00, 3.25150496e+00,
#        1.00014397e+00, 9.99850382e-01, 1.52121277e-01, 3.12292832e-03,
#        3.66677292e+04, 9.85036443e-01, 9.78781622e-02, 9.94441815e-01,
#        9.98677522e-01, 9.99473222e-01, 9.97929894e-01, 9.99946304e-01,
#        1.00004470e+00, 9.99966531e-01, 1.00001223e+00, 1.00016960e+00,
#        1.00030157e+00, 1.00017947e+00, 9.99958494e-01, 1.00020302e+00,
#        1.00006010e+00, 1.00016379e+00, 9.99969575e-01, 1.00004874e+00,
#        1.00035627e+00, 1.00036702e+00, 1.00000514e+00, 1.00020854e+00,
#        1.00031632e+00, 9.96887834e-01, 3.15781114e-01]), np.array([1.00000004e+00, 1.97506181e+00, 2.06642304e+00, 3.10028070e+00,
#        1.00000014e+00, 1.00000027e+00, 1.47658025e-01, 8.15159655e-04,
#        3.66677292e+04, 9.53947127e-01, 5.98012263e-02, 1.00000015e+00,
#        9.99999988e-01, 1.00000013e+00, 1.00000006e+00, 1.00000019e+00,
#        1.00000016e+00, 1.00000022e+00, 9.99999912e-01, 9.99999995e-01,
#        1.00000017e+00, 9.99999869e-01, 9.99999951e-01, 1.00000021e+00,
#        9.99999948e-01, 1.00000036e+00, 1.00000005e+00, 9.99999971e-01,
#        1.00000025e+00, 1.00000017e+00, 1.00000000e+00, 1.00000006e+00,
#        9.99999902e-01, 9.99996302e-01, 3.18852833e-01])]
        # algo1  algo2
# [0.358, 0.028, 0.358]
# [1.697, 2.921, 1.697]
# [2.147, 2.739, 2.147]

# alpha, theta = sparse_kernel_flows(3,X, Y, tau, alpha_init, theta_init, lambda1, lambda2)
# Y_prediction=prediction(X,Y,alpha,theta, lambda1,INPUT)
def fill_PRED(ALPHA,THETA):
    prediction1=[]
    for i in range(len(ALPHA)):
        pred=prediction(X,Y,ALPHA[i],THETA[i],lambda1,INPUT)
        prediction1.append(pred)
    return prediction1

# print(PREDICTION)
# print("")
# print(alpha)
# print("")
# print(theta)
# PREDICTION=[np.array([-3.92971322e-02,  2.83068031e-01,  6.50566451e-01, -2.29917566e-02,
#         1.08299117e-03,  7.79832558e-02,  3.16714383e-01, -2.75361310e-02,
#         5.99021396e-02,  2.86885290e-01,  1.61515621e-01,  2.35719874e-01,
#        -6.83376602e-03,  2.02099257e-01, -3.42687488e-02, -7.95325600e-03,
#         1.23536417e-01, -1.14849803e-01,  4.43898562e-03, -7.94232208e-01,
#        -1.05650445e-01,  5.43314482e-01, -9.48523975e-02,  3.51116682e-02,
#         3.11450675e-02,  8.58481162e-02,  3.23268659e-03,  6.15847439e-02,
#         8.99753586e-02,  2.91415845e-01,  8.19060846e-02, -2.08242285e-01,
#         2.97096195e-02,  3.80360349e-01, -1.16451976e-01,  3.10653057e-01,
#        -9.61105599e-02, -2.96637673e-01, -4.68078143e-01,  4.45856287e-02,
#        -2.72065371e-01, -1.39815837e-01, -1.14055374e-01,  3.15353720e-01,
#        -9.31961430e-02, -1.55613475e-01,  1.38008813e-01,  5.68315480e-02,
#        -8.96745503e-02,  8.00093540e-02,  1.62526597e-01, -1.73053471e-01,
#        -4.32708719e-02,  1.59845403e-01,  1.63844687e-02,  1.41116098e-01,
#         1.98983321e-01, -7.97672499e-02,  4.06961469e-01, -5.60786122e-02,
#         2.02972934e-02,  9.65568980e-03, -1.56658636e-02,  1.11227159e-01,
#         4.05562140e-03,  2.47411389e-01, -3.96937374e-01, -1.66488325e-01,
#        -1.76925582e-01, -8.47090566e-02,  2.24433778e-01, -1.99064081e-01,
#        -1.21134819e-01, -2.29731911e-01,  2.00282075e-01,  5.63358287e-01,
#        -2.84997268e-01,  2.02768508e-01,  2.03650430e-01,  3.02167453e-02,
#        -4.27176351e-02,  3.05563652e-01,  6.45542983e-02,  2.99303846e-01,
#         1.51305660e-01,  1.49864891e-01, -2.30178229e-01, -4.29832964e-02,
#        -4.27525731e-02, -5.72228630e-02,  3.37039387e-01,  5.77833135e-02,
#         1.04952487e-01,  1.50797237e-01,  2.81877804e-01,  2.04251363e-01,
#        -3.53692306e-01,  1.33476533e-01, -4.64338697e-02, -1.29446232e-01,
#         1.80349332e-01, -9.51732985e-02,  2.37388147e-01, -4.86899593e-02,
#         7.41029024e-02, -4.41646836e-02,  1.80588929e-01,  6.32343981e-02,
#         1.65078958e-01,  1.03779267e-01,  1.75591608e-01,  1.70398447e-01,
#         3.57534915e-02,  8.38600905e-02,  1.26389649e-01,  1.68410870e-01,
#         1.58293388e-01,  1.61519472e-01,  1.90544861e-01,  1.90221628e-01,
#         1.36397002e-01, -9.27332658e-02,  1.64913531e-01,  2.01249882e-01,
#         1.21911171e-01, -8.71189974e-02, -1.58860203e-01,  1.48676519e-01,
#         2.00703724e-01,  1.66084874e-01, -7.37112155e-01,  2.81569731e-02,
#        -2.20311138e+00, -1.08222395e+00, -3.48102073e-01,  3.48335041e-01,
#         1.20127954e-01, -3.76607239e+00,  5.99003292e-03,  1.35001824e-01,
#        -3.85627638e+00, -9.50449919e+00, -3.73952323e+00, -2.83555548e+00,
#        -2.06642367e+01, -2.64198986e+01, -1.44918825e+00, -1.57645930e+00,
#         9.06678228e-01,  2.69297722e-01,  1.35761122e+00,  7.34969056e-01,
#         6.82116763e-01, -7.42943592e-01,  9.61688599e-01,  4.18729860e-01,
#        -1.46763748e+00,  6.90430883e-01, -4.77986236e+00,  4.64123452e-01]), np.array([-2.69640811e-01,  4.10688100e-01,  4.90643300e-01, -2.62088566e-01,
#         6.23662128e-02, -3.58777584e-02,  3.17348647e-01,  2.02941199e-01,
#         4.59697329e-01,  4.56448155e-01,  2.80655360e-01,  4.57933338e-01,
#        -4.28041647e-02,  7.48835441e-02,  9.10390547e-02, -1.10673571e-02,
#        -9.39165635e-02, -5.98243370e-02,  1.28793085e-01, -7.12884528e-01,
#         1.39897305e-01,  2.93282060e-01, -5.91773578e-02, -3.61128231e-02,
#         1.93282685e-02,  1.36201619e-01, -1.83860693e-01,  1.12195819e-03,
#         1.20065255e-02,  2.20533818e-01, -6.00189632e-03, -2.72732775e-01,
#        -1.59051178e-01,  3.45692757e-01,  5.31141427e-02, -9.97299966e-02,
#        -5.39434743e-01, -8.95449864e-02, -5.07468740e-01,  2.66221690e-01,
#        -1.61648577e-01, -2.01152071e-01, -3.88186304e-01,  2.98819855e-01,
#         1.78861728e-01, -1.44962849e-01,  6.97413127e-02,  1.24321858e-01,
#         7.56576907e-02, -1.97819112e-01,  8.78684469e-02,  5.45724174e-02,
#         6.31757493e-02,  5.94333026e-01, -1.05560911e-01,  9.66317190e-02,
#        -2.13852115e-01,  1.22541526e-02,  8.68484355e-02,  2.80920924e-02,
#         7.12039384e-02,  1.44035377e-01,  1.64431538e-01,  1.52421816e-01,
#        -2.00245295e-02,  1.26890536e-01,  7.37886969e-02,  6.83918319e-02,
#         1.13332908e-01,  3.76299564e-01, -5.50107740e-02, -2.04187532e-01,
#        -1.30613112e-01, -1.93493417e-01, -5.73470745e-02,  7.23014454e-01,
#        -1.37616436e-01, -6.58121193e-01,  1.49328504e-01,  7.85825220e-02,
#         1.16863478e-01,  2.21182512e-01,  1.09261769e-01,  9.97589338e-02,
#         1.14927907e-01,  2.26286079e-01, -1.78552833e-01, -2.83751101e-02,
#         2.06812728e-01,  2.26700015e-01, -5.69109025e-02,  9.29706672e-02,
#         7.58063473e-02,  2.35440578e-01,  2.24827411e-01,  2.39208794e-02,
#        -1.55508235e-01, -1.11475172e-01, -1.20345734e-01, -3.03170807e-01,
#         2.09364315e-01,  2.13279748e-01,  3.03443068e-01, -5.05223167e-02,
#         1.01820374e-01, -9.92419924e-02,  2.87974699e-01, -1.25243940e-01,
#         5.22091799e-02, -6.03575307e-02, -7.91712805e-02, -1.16988494e-01,
#         1.84443351e-02, -1.52231088e-01, -1.37574775e-01, -9.59462115e-02,
#        -4.22820858e-02, -8.86580996e-02, -2.10570499e-02, -1.55143934e-02,
#        -2.19198349e-01, -3.29004429e-01, -1.73993113e-01, -4.99674402e-02,
#        -1.35786672e-01, -3.89237008e-01, -2.99602043e-01, -1.97976241e-01,
#        -1.39730722e-01, -1.52655474e-01, -3.20904107e-01, -1.37410590e-01,
#        -5.57713366e-01, -2.37888883e-01, -2.70207909e-01, -1.73099403e-01,
#        -6.79283554e-03, -9.04015191e-01, -1.58407737e-01,  5.08721232e-04,
#        -9.25361173e-01, -9.04167621e-01, -6.60551377e-01, -1.62305411e+00,
#        -1.28249703e+00, -1.45741841e+00,  7.10751036e-01,  1.01953192e+00,
#         1.45807839e-01,  4.86346908e-01,  3.61558485e-01,  4.57298060e-01,
#         3.86824728e-01,  1.81063691e-01,  3.84327339e-01,  5.53467392e-01,
#         5.30105781e-02,  5.60202547e-01,  3.30154545e+00,  2.69838110e-01]), np.array([-1.84000726e-01,  5.54204525e-01,  2.16653446e+00, -8.89838825e-02,
#        -1.54182141e-01,  3.76218475e-02,  9.60028166e-01,  1.50474266e+00,
#        -3.14617571e-02, -1.49879469e+00, -4.33711714e-01, -2.42969691e+00,
#         6.76026748e-01,  4.36428001e-01,  5.51051742e-01,  7.34736996e-01,
#         3.68295695e-01,  4.38126407e-01,  2.09362957e-01, -1.11351456e+00,
#         4.36946641e-01,  7.01612302e-01,  3.45091413e-01,  2.80651715e-01,
#        -4.27675416e-01, -6.16606429e-01, -3.05623058e-01, -2.87124368e-01,
#        -3.27895966e-01, -1.79429794e-01, -3.14819961e-01, -1.48782230e-01,
#        -2.46801387e-01, -3.44218606e-01,  1.64470211e-01, -1.05721237e-01,
#        -3.60675598e-01,  1.47851150e-01, -2.24698341e-02, -1.70591345e-01,
#         1.08556410e-01,  8.09092301e-02, -1.33892685e-01, -2.56838889e-01,
#        -6.90209410e-01, -2.86991033e-01,  4.42757684e-01, -3.61981004e-01,
#         3.00189055e-01,  1.12672007e+00, -5.97154850e-01, -2.76021017e-01,
#         1.92828153e-01,  1.25651834e-01,  1.63293178e-01, -6.52571094e-01,
#        -2.26662129e-01,  1.10691663e+00,  1.43310706e-01,  2.87750029e-01,
#         3.50775628e-01,  3.75692589e-01,  2.35093004e-01, -6.85520235e-01,
#         2.55594988e-01, -6.46543667e-01, -6.42829196e-01,  1.85668075e+00,
#        -4.31016712e-02, -2.36720470e+00,  6.16014014e-01,  1.69136192e+00,
#         2.84685176e-01,  7.26528043e-02, -3.50247303e-01,  7.10594729e-01,
#         1.30220567e+00, -1.60214714e+00, -8.62713064e-02,  2.37251176e-02,
#        -2.56230598e-01,  1.12293870e+00, -1.52027406e-01,  1.00976079e+00,
#         1.63137706e-02, -5.26427114e-02, -4.87377374e-02, -2.60599406e-01,
#        -3.54046457e-02,  8.32511601e-02, -1.65089614e-01, -2.12733967e-01,
#        -1.58344048e-01, -3.17229900e-02,  6.41682734e-01,  3.63264037e-01,
#        -1.51282017e-01,  2.45556068e-01, -4.93231622e-01,  4.33283176e-03,
#         1.63342933e-01,  1.20572293e+00, -3.03856379e-01,  2.23899861e-01,
#         3.50577367e-01, -3.87937045e-01, -6.22056443e-01, -3.83307861e-01,
#        -3.73428405e-01, -6.78048668e-01, -1.02319631e+00, -1.31855407e+00,
#        -1.54325049e+00, -1.39560560e+00, -1.39138937e+00, -1.14627025e+00,
#        -8.88453504e-01, -9.41854910e-01, -7.20674740e-01, -6.84451826e-01,
#        -9.75123847e-01, -1.41105995e+00, -8.22687914e-01, -1.46497438e+00,
#        -9.66874206e-01, -1.30608540e+00, -1.64512012e+00, -9.17116627e-01,
#        -1.21053768e+00, -7.72171649e-01, -1.99545595e+00,  2.63985873e-02,
#        -2.09003665e+00, -1.28023682e-01,  2.12884873e+00, -1.19255527e+00,
#         6.36832650e-01, -2.83793763e+00,  3.38512176e-01,  5.76130360e-01,
#        -2.00243940e+00, -5.52338687e+00, -7.75447905e-01,  2.88167448e+00,
#         8.96875823e-01, -2.42286555e+00, -1.75370962e+01, -1.86937079e+01,
#        -6.22754579e+00, -2.14163481e+00, -6.85873759e+00, -3.46220232e+00,
#        -3.77807489e+00, -1.05686282e+00, -6.34475738e+00, -2.23385143e+00,
#        -3.91790306e+00, -3.45921385e+00, -2.43632782e+01, -2.79138657e+00]), np.array([ 7.39792291e-02,  1.82921999e-01,  7.60715205e-01, -3.37652264e-01,
#         9.86046030e-04, -2.30470596e-01,  3.12183028e-01, -4.38744843e-02,
#         2.93229384e-01,  4.13679335e-01,  2.30705495e-01,  1.47028298e-01,
#         1.03769201e-01,  3.22336378e-01,  2.31998607e-01,  1.80548558e-01,
#        -4.25314661e-01,  4.01627706e-01, -5.13601002e-01, -7.46706123e-01,
#        -4.45001960e-02, -4.36704131e-01,  1.57302253e-01, -4.24262264e-01,
#         4.59043059e-01, -8.18829738e-01,  3.19316090e-01,  4.71919082e-01,
#         2.19938849e-01,  2.06416176e-01,  2.27903996e-01, -6.65364024e-02,
#         1.42595270e-01,  1.56447875e-01,  9.82318389e-02,  1.51315644e-01,
#        -2.06862863e-01, -2.84200269e-01, -5.30443611e-01,  7.51364489e-02,
#        -2.65609536e-01,  2.24483170e-03,  1.78837115e-02,  2.80501674e-01,
#        -6.44623860e-01,  9.10330842e-02,  5.40642099e-01,  2.75622823e-01,
#        -6.13512144e-01, -2.10071580e-01,  1.46049940e-01,  8.78780687e-02,
#        -5.94912731e-01,  2.38340974e-01, -7.68071253e-02,  2.94759283e-01,
#         2.00723152e-01,  9.67142600e-02,  1.09506873e+00, -6.00082061e-01,
#         1.94200722e-01,  4.42839519e-01, -5.34086308e-01,  7.01776287e-01,
#        -4.79194490e-02, -5.30543496e-01,  4.44475394e-01,  3.32285959e-01,
#        -1.95660852e-01,  2.93746539e-01, -6.85876944e-01, -9.45363272e-02,
#        -3.88649725e-01, -2.84549352e-01, -1.34649837e-01,  5.34567531e-01,
#        -1.61982069e-01,  6.75658720e-01, -2.99066268e-01,  2.55654641e-01,
#        -1.08102995e-01,  4.53515682e-01,  2.46716765e-01,  3.39348537e-01,
#        -1.52554824e-01, -2.27938119e-01, -2.25016768e-01, -5.07523477e-02,
#         4.68494469e-01,  4.29567956e-01,  2.79853976e-01,  3.85537715e-01,
#        -6.63271339e-02, -2.28231258e-01,  5.79770659e-01,  5.11168470e-01,
#        -4.33965397e-01,  3.38579605e-01, -3.35075779e-02, -1.31672965e-01,
#         1.02464719e-01, -4.89164804e-02, -1.12592672e-01, -1.49793244e-01,
#         1.29056194e-01, -6.51333898e-02,  2.44269171e-02,  3.58078418e-02,
#         1.67939920e-01,  1.46102976e-01,  2.61565042e-01,  3.59110526e-01,
#         5.33158667e-01,  6.45929907e-01,  5.61257155e-01,  2.53424358e-01,
#         1.67207677e-01,  1.53357547e-01,  2.97401180e-01,  2.89060862e-01,
#         2.29172541e+00,  9.03979342e-01,  1.88205913e+00,  2.50267416e+00,
#         2.35608383e+00,  2.24139653e+00,  1.32289852e+00,  2.13587836e+00,
#         2.24130017e+00,  1.89366015e+00, -4.08467332e+00, -5.96216596e-02,
#        -1.08604553e+01, -4.76148077e+00, -1.49397677e+00, -1.06073150e+00,
#         3.61725040e-01, -1.75619586e+01, -2.76562234e-01,  4.74438777e-01,
#        -1.82035042e+01, -2.80383018e+01, -1.34071151e+01, -4.35509634e+00,
#        -1.61038481e+01, -3.15925229e+01, -7.90641002e+00, -6.38699715e+00,
#        -1.29512569e+00,  5.01771399e+00, -2.01327232e+00,  3.65454300e+00,
#         3.07936897e+00,  3.57483594e+00, -1.01860902e+00,  4.34497919e+00,
#         2.09020409e+00,  3.95299378e+00,  1.92025692e+01,  4.83580668e+00]), np.array([-3.20457611e-02,  8.80986499e-02,  4.20296050e-01, -3.38782053e-01,
#        -1.40323558e-01, -5.75760184e-02,  4.00308135e-01,  2.88766937e-01,
#         1.75799973e-01,  1.83981864e-01,  2.99297888e-01,  4.89776623e-01,
#         2.85958460e-01,  1.63273843e-01,  1.39733383e-01,  3.01063856e-01,
#         1.13729302e-01,  1.04859859e-01, -4.88207559e-02, -8.40521467e-01,
#         3.33242683e-01, -2.79171628e-01,  1.44086810e-01,  6.23893266e-02,
#         7.10641082e-02,  1.80197905e-01,  6.43849956e-02,  7.04872658e-02,
#         3.28907602e-02,  2.06637191e-01,  1.38982513e-01, -3.21819417e-01,
#         6.67737283e-02,  2.76969655e-01, -1.24044909e-01,  3.39667158e-01,
#        -9.92465128e-02, -1.34612828e-01, -3.98058524e-01,  2.26049763e-02,
#        -1.44777746e-01, -1.33752236e-01, -3.37584482e-02,  2.27981032e-01,
#         4.89322642e-03, -3.07488920e-01, -3.39497002e-01, -3.32424456e-02,
#        -1.54583106e-01,  4.03941038e-01,  1.02751718e-01,  2.81961030e-01,
#        -3.13709527e-01, -3.51980469e-02,  1.25090818e-01,  1.17272863e-01,
#         4.66698741e-02,  1.47044500e-01,  1.96381566e-01, -4.80562841e-01,
#         7.64523253e-02,  6.51395133e-02, -1.43854429e-01,  3.30175597e-01,
#         1.27326061e-01,  2.34835456e-01,  1.33125145e-02,  2.87531789e-01,
#        -1.17879117e-01,  4.87119292e-02,  3.35120130e-01, -5.43286605e-01,
#        -3.69138224e-01, -3.03593712e-02, -1.39192181e-02,  7.35154784e-01,
#        -3.19937141e-01,  5.64083508e-01, -2.45917603e-01,  1.97934737e-01,
#        -9.17813445e-02,  5.11639111e-02,  3.21112988e-01,  4.83793709e-02,
#        -1.27826853e-01, -1.48058894e-01,  9.24622803e-02, -4.51961477e-02,
#         2.80788912e-01,  2.18942585e-01, -3.22567958e-01,  3.82809491e-01,
#         8.38595440e-02, -1.62154459e-01,  2.29841658e-01,  3.45287973e-01,
#        -8.91298575e-02,  2.33638522e-01, -1.79063682e-01, -8.21951843e-02,
#         8.39830729e-02,  1.65184369e-01,  9.98573930e-02, -1.11551115e-01,
#         2.96839485e-02, -1.75100538e-01, -4.13815264e-02, -2.37011812e-01,
#        -6.79179724e-02, -2.41421463e-01, -2.04187199e-01, -2.22645436e-01,
#        -2.92722709e-02, -9.35217886e-02, -1.54213666e-01, -2.34868871e-01,
#        -2.37591419e-01, -2.66828140e-01, -6.57494038e-02, -6.81766752e-02,
#         2.24970682e-01, -1.08598568e-01,  2.82259526e-01,  1.13177541e-01,
#         1.91363232e-01,  3.05950303e-02, -6.38140531e-02,  2.44931826e-01,
#         2.45343532e-01,  2.84761490e-01, -7.21051240e-01,  4.27480736e-01,
#        -2.10341875e+00, -2.57661626e-01,  1.11538632e+00,  8.75010272e-02,
#         6.35736584e-01, -5.60280527e+00,  5.19719033e-01,  6.31790408e-01,
#        -5.29527576e+00, -1.17087104e+01, -2.63546924e+00,  3.53716422e+00,
#        -4.70366599e+00, -1.54773558e+01, -6.34733912e+00, -7.83622509e+00,
#        -1.24213810e+00,  2.04857866e-01, -2.04302189e+00, -1.53770056e-01,
#        -4.36110702e-01,  3.16680154e-01, -1.34423845e+00,  7.84637822e-02,
#        -7.03457920e-01, -8.39073613e-02, -2.42049971e+01, -1.04895220e-02]), np.array([ 8.23287083e-02,  5.53100660e-02,  5.84349938e-01, -7.84590407e-02,
#         1.88923268e-01, -1.91622345e-01,  2.50133395e-01, -2.17295073e-01,
#         2.55127781e-01,  9.48239284e-01,  6.56171154e-02, -2.41501911e-02,
#        -8.82302647e-02,  6.37933158e-02,  3.61501746e-01, -1.28542782e-01,
#        -9.45005174e-02, -2.62258473e-01,  3.48252022e-02, -7.93515824e-01,
#         4.24966926e-01,  8.20572514e-02,  2.72757627e-01,  1.98799061e-01,
#        -1.52362926e-01,  1.23143652e-02, -5.92553014e-02, -1.18906927e-01,
#         3.24122432e-02,  4.62948379e-01,  1.80159498e-02, -1.95904317e-01,
#        -6.54634865e-02,  6.53923885e-01, -6.70292043e-02,  2.88015210e-01,
#         2.71564802e-02, -2.17485953e-01, -9.07923171e-01,  8.81725029e-02,
#        -2.31075218e-01, -1.92589077e-01, -2.49870993e-01,  5.34638056e-01,
#         7.95054137e-02, -1.03651215e-01,  3.27276922e-03,  6.70007465e-02,
#        -1.23890284e-01,  7.51930466e-01,  1.41412164e-01, -1.60592526e-01,
#        -5.09563434e-02,  4.64293577e-01,  8.14609735e-03,  1.43965538e-01,
#        -5.54499210e-02, -1.20267200e-01,  6.03027562e-01, -5.91321904e-02,
#         1.26400425e-01,  2.12558835e-01, -2.16509135e-01,  1.49286204e-01,
#         1.69051063e-02, -8.04095510e-01, -3.01908718e-01, -1.21593892e-01,
#        -1.60782094e-01,  2.58109201e-02,  1.20773785e-01,  8.20262712e-02,
#        -9.46284638e-02, -4.12754032e-01,  2.16071242e-01,  4.80493114e-01,
#        -1.87404965e-01,  2.64626790e-01,  3.28000273e-01,  1.55593316e-02,
#        -3.79858636e-02,  2.58966614e-01,  2.71100423e-02,  2.81356218e-01,
#         2.18901659e-01,  2.25590623e-01, -4.90376119e-01, -2.87510013e-02,
#        -2.16401947e-02, -3.90846452e-02,  4.75749592e-01, -1.40237055e-02,
#         7.79744250e-02,  2.36133852e-01,  9.22059310e-02,  2.07867086e-01,
#        -1.78726455e-01, -1.03586996e-01, -1.39296727e-02, -5.87608311e-02,
#         7.97770000e-02, -1.98426619e-02,  1.22350639e-01, -1.92893348e-02,
#         1.44318278e-02, -2.62654965e-02,  1.25757620e-01,  8.03336948e-03,
#         8.22061762e-03,  5.03107786e-02,  4.67935427e-02,  3.78361417e-02,
#         1.53196007e-01,  2.65222384e-02,  3.65816747e-02,  2.70249799e-02,
#         2.61752355e-02,  4.51911065e-02,  1.90440963e-02,  2.93046536e-02,
#         6.08195388e-02, -1.57194484e-01,  4.41467236e-02, -5.69800005e-02,
#         1.19128233e-01, -9.55486611e-02, -1.60038886e-01,  5.49788139e-02,
#        -9.35881612e-02,  4.11465111e-02, -9.17112679e-01, -1.46525256e-01,
#        -2.54501915e+00, -1.40482057e+00, -7.33350541e-01,  2.34294281e-01,
#        -1.37549109e-01, -1.55283383e+00, -2.13704461e-01, -1.19843755e-01,
#        -2.27507832e+00,  1.94159660e+00, -3.23844929e+00, -2.31833114e+00,
#        -5.77123634e+00,  1.41895682e+00, -5.99971375e+00, -6.34589216e+00,
#        -3.18788134e-01,  2.63168071e-01, -1.55628390e+00, -2.45487085e-01,
#        -4.19989337e-01, -6.78067149e-02, -3.13574100e-01,  7.49033036e-02,
#        -2.26739921e-01, -1.77712369e-03, -1.43453301e+01,  8.86726629e-02]), np.array([ 1.35594457e-02,  1.24118476e-01,  2.64933062e-01, -1.58378312e-01,
#         2.08639923e-01, -9.26433827e-02,  1.41521415e-01,  3.23384449e-01,
#         4.60004137e-01,  6.92053936e-02,  3.74030881e-01,  1.92260565e-01,
#         1.36186108e-01, -1.54210877e-01,  6.53171259e-01,  1.97510091e-02,
#        -1.11432312e-01, -1.00293504e+00,  8.44715124e-02, -4.35780554e-01,
#         6.28406478e-01, -1.40637570e-01,  3.83208048e-01,  4.41119197e-01,
#        -1.36367569e+00,  1.82314803e-01,  8.42411783e-01, -1.02548698e+00,
#         1.31356432e-01,  4.32712384e-01,  1.10633615e-01, -2.72811508e-01,
#        -9.48423372e-03,  3.31462269e-01,  1.80019181e-01, -9.48048134e-02,
#        -4.92769967e-01, -3.56792019e-02, -3.89597825e-01,  2.57674944e-01,
#        -8.53271300e-02,  4.57673500e-02, -2.00605472e-01,  4.98339332e-01,
#         2.83481538e-01, -1.53841387e-01, -5.33237331e-01,  9.69631220e-02,
#         9.28138209e-01,  2.19673574e+00,  2.18622591e-02,  1.91022156e-01,
#         3.30569832e-02,  3.33071247e-01, -1.84273454e-01,  1.86491344e-02,
#        -2.27496042e+00, -1.00377521e-01, -1.22901393e-01, -2.97824284e-02,
#        -2.79056940e-02, -4.73348724e-03, -9.76298339e-01,  9.37398657e-02,
#        -1.61770266e-01, -2.24503684e-01,  6.54124901e-01, -2.25363689e-02,
#         3.81384527e-02,  2.63404799e-01,  5.73507913e-02, -1.47729215e-02,
#        -1.57820956e-01, -8.54323857e-02,  1.10284723e-01,  1.54235464e-01,
#         3.89457333e-01,  1.33288418e-01,  2.39830307e-01,  1.20886586e-01,
#        -5.94004988e-02,  4.60262666e-01,  6.76043209e-02,  4.11547116e-01,
#         2.32899037e-01,  2.10930358e-01, -5.98632615e-02, -1.31589884e-01,
#         6.30449474e-02,  1.27011892e-01, -1.00767973e-01,  8.47555543e-02,
#         7.38073787e-02,  2.29667529e-01, -1.02782868e+00,  4.19335935e-01,
#        -2.70851738e-01, -2.55156297e-01, -1.36785546e-01, -2.52099184e-01,
#         7.65170042e-02,  9.47826198e-02,  6.22868876e-01, -2.18157824e-01,
#         3.58185993e-02, -1.53417222e-01,  1.07883365e+00,  2.55475664e-02,
#        -1.06154668e-01,  7.58732559e-02, -2.48674377e-03, -7.30686552e-02,
#         6.49785866e-02,  1.35974510e-01,  5.97363524e-02, -2.08759130e-01,
#        -1.17222300e-01,  6.75004478e-02, -1.62288874e-01, -6.19779523e-02,
#         1.58158435e-01,  6.10852901e-02,  2.11756161e-01, -2.13194174e+00,
#         3.79251272e-01,  1.10418674e-01,  1.50623973e-01,  1.70163846e-01,
#        -2.08697265e+00,  1.62570859e-01, -8.60995601e-02, -1.59092181e-02,
#         2.47848613e-01, -2.45301834e-01, -4.68733324e-02, -7.70423313e-02,
#         1.14224101e-01, -1.86057076e-01, -3.42775989e-01,  1.17724301e-01,
#        -6.35615627e-01,  1.80062787e-01,  5.18015699e-01, -1.97183153e-01,
#        -3.66191163e-01, -1.25044935e+00, -2.36017466e+00, -2.28209121e+00,
#        -9.57659062e-01, -2.92412172e-01, -2.23935537e+00, -2.93145799e+00,
#        -3.90664616e+00, -8.86091567e-01, -1.06672889e+00, -3.81451406e-01,
#        -6.90494904e-01,  6.27711410e-02, -3.59805383e-01, -3.52916713e-01]), np.array([ 2.09497157e-02,  9.71308602e-01,  5.19954668e-01, -1.44153841e-01,
#        -3.22009617e-02,  2.89806352e-01,  4.92025465e-01,  2.26261642e-01,
#         2.97538833e-02, -1.76961932e-03, -2.46527659e-01,  5.36374560e-02,
#        -3.79762312e-01, -6.33105301e-02, -1.42417644e-01, -3.69868973e-01,
#        -1.53824236e-01, -7.60263921e-01, -2.77180507e-01, -8.19146704e-01,
#         3.33069800e-01,  7.59119982e-01,  2.69044613e-01,  1.45145947e-01,
#        -2.69879992e-01,  5.49813944e-01,  2.33671635e-01, -1.84461081e-01,
#         1.74869284e-01,  4.50884043e-01,  1.77192826e-01, -1.83988869e-01,
#         5.88216565e-02,  3.45594253e-01, -1.55390708e-01,  2.76519824e-01,
#        -2.57256003e-02, -4.71523724e-01, -2.04138131e-01,  6.51749571e-02,
#        -4.17565851e-01, -1.61358524e-01, -1.77727035e-01,  4.80783188e-01,
#         3.85269769e-01, -8.00324559e-02, -1.31152515e-01, -2.50647650e-02,
#         2.80565797e-01,  4.37466714e-01, -1.63723747e-01, -3.49072817e-01,
#         2.28551514e-01,  8.35217229e-04,  1.81516441e-01, -3.22277251e-01,
#        -5.33113598e-02,  1.57704681e-01, -3.67016375e-01,  2.92118674e-01,
#         1.29128068e-01,  1.97774707e-02, -1.97796992e-01, -7.98294457e-01,
#         2.18183662e-01,  5.88043971e-01,  2.27448859e-01, -2.04867129e-01,
#        -1.23452205e-01, -1.07068078e-01,  7.72697307e-01, -4.95762371e-01,
#        -2.35583336e-01, -3.14895131e-01,  2.29798818e-01,  6.05675260e-01,
#        -5.66397151e-01,  4.37985273e-01,  1.74810946e-01,  9.74087613e-02,
#         1.03791261e-01,  4.60079028e-01,  1.56132227e-01,  4.30483518e-01,
#         1.27148821e-01,  1.70884113e-01, -3.04388593e-01,  7.32433378e-02,
#         1.32596986e-01,  1.01008099e-01,  3.25694110e-01,  1.55015722e-01,
#         1.73620379e-01,  1.64990540e-01, -2.82417450e-01,  3.50254316e-02,
#        -4.46621512e-02, -1.87928910e-01,  2.56442752e-01,  6.11646511e-02,
#        -4.11884725e-02, -8.36307291e-01,  3.44961730e-01, -1.38066421e-01,
#        -2.54505041e-01,  2.07066795e-01,  6.51158285e-01,  1.41086438e-01,
#         3.01466908e-01,  2.51156327e-01,  4.13940804e-01,  5.39005319e-01,
#         5.07740835e-01,  4.23186356e-01,  4.68829463e-01,  5.66160110e-01,
#         5.54351564e-01,  6.09255831e-01,  3.70221905e-01,  3.17100725e-01,
#         2.20599191e-01,  6.54376619e-01, -3.87232024e-02,  4.37388583e-01,
#         5.20248520e-01,  4.30301039e-01,  9.65950294e-01,  1.27568444e-01,
#         2.04946182e-01, -4.37991358e-02,  1.80425820e+00, -7.63169423e-01,
#         4.02923975e+00, -4.68614420e-01, -3.60228902e+00, -2.98709356e-01,
#        -1.60595827e+00,  8.94502488e+00, -1.04357123e+00, -1.54565050e+00,
#         6.83508728e+00,  1.06941920e+01,  2.91998333e+00, -1.46390795e+01,
#        -1.66029663e+01, -1.25965598e+01, -6.36950696e+00, -9.25609641e+00,
#         1.49213706e+00,  1.81478511e-01,  2.70843273e+00,  5.11269603e-01,
#        -1.28521110e-01, -1.23928343e+00,  1.94227739e+00,  8.79605888e-02,
#         8.73270084e-01,  1.20601347e+00, -6.19121084e+01,  8.72120337e-01]), np.array([-4.85715933e-02,  1.31454549e-01,  5.86586698e-01, -1.40167898e-01,
#         8.45265587e-02, -1.65001693e-01,  1.47367805e-01,  2.08741275e-02,
#         2.81686871e-01,  4.23208617e-01,  2.71204257e-01,  3.92684221e-01,
#        -1.14899408e-01, -7.03167742e-02,  4.82443175e-02, -1.68602966e-01,
#        -4.66180579e-02,  4.74096575e-02,  5.67814408e-02, -7.50377457e-01,
#         4.54640689e-01,  3.54581685e-01,  1.99019152e-01,  1.05104561e-01,
#         2.42965197e-01,  2.42441014e-01, -1.15697186e-01,  1.65019040e-01,
#         2.07351049e-02,  3.36412891e-01,  2.37490582e-02, -2.24891689e-01,
#        -9.24858878e-02,  6.19206880e-01, -1.68227540e-02,  3.34215991e-01,
#        -3.43867681e-01, -1.57954154e-01, -8.08974039e-01,  1.00969308e-01,
#        -1.92371240e-01, -8.39952968e-02, -2.76568079e-01,  3.85416809e-01,
#         2.99788088e-01, -1.64921610e-01,  1.86003772e-01,  3.44454571e-02,
#        -1.35328255e-01, -6.90295910e-02,  3.61259961e-01, -1.40447895e-01,
#        -7.97930261e-02,  3.36283283e-01, -1.25305412e-01,  3.77836211e-01,
#         2.68559757e-01, -1.57415463e-01,  6.87089018e-01, -1.01194606e-01,
#        -4.63738003e-03,  1.01275238e-01,  1.04660733e-01,  4.50270931e-01,
#        -1.06854589e-01, -5.80653658e-01, -7.67494965e-02, -1.17703696e-01,
#        -1.50698768e-01,  2.92424369e-01, -9.46013694e-02, -3.30781182e-01,
#        -2.61593151e-01, -1.37981782e-01,  2.39022209e-01,  4.14268469e-01,
#        -3.10525955e-01,  3.26461592e-01,  2.37812785e-01,  5.44843418e-02,
#         4.19597927e-02, -2.46328048e-02,  1.34302494e-01, -5.97138443e-02,
#         1.50011192e-01,  2.40505481e-01, -5.46712394e-02, -6.77802866e-02,
#         7.96518912e-02,  5.78989966e-02, -2.64668208e-01,  1.32780048e-01,
#         1.65856208e-01,  2.34840453e-01,  2.03582797e-01,  1.48630705e-02,
#         2.42679802e-02, -1.38209648e-01, -9.19258823e-02, -1.45865092e-01,
#         6.61342247e-02, -4.94836482e-04,  1.32108973e-01, -1.25065275e-01,
#        -6.74041938e-02, -1.05488926e-01,  1.20980757e-01, -1.27020985e-01,
#        -3.48837388e-02, -1.27311808e-01, -1.14150227e-01, -8.66378265e-02,
#        -9.14108500e-03, -1.10732150e-01, -1.09475073e-01, -6.51837926e-02,
#        -3.31679987e-02, -2.75246958e-03, -9.37914967e-02, -1.08836124e-01,
#        -1.40142072e-01, -1.63936475e-01, -1.50092497e-01,  4.48572152e-01,
#         9.82909552e-03, -3.06167503e-01, -1.56722103e-01, -1.39338405e-01,
#         3.66152556e-01, -1.23627226e-01,  4.32617235e-01, -1.45024567e-01,
#         8.11271638e-01,  5.05458742e-01,  1.34155833e-01,  3.07870554e-01,
#        -3.18707987e-02,  1.85844497e+00, -1.00566365e-01, -2.52707323e-02,
#         1.86949121e+00,  4.44386362e+00,  5.51953357e-01,  7.03890121e-01,
#         5.25731194e+00,  5.83241008e+00,  2.29703279e+00,  2.85179528e+00,
#         4.52218918e-01, -2.02354228e-01, -3.51850184e-01,  3.76231161e-01,
#         3.75269553e-01, -1.42927104e-01,  5.54023508e-01, -5.79682879e-02,
#         9.81195628e-02,  1.51618813e-01,  7.43635710e+00, -1.70900885e-01]), np.array([ 1.24168545e-02,  1.73938906e-01,  3.33350983e-01, -1.82806810e-01,
#        -5.13827709e-02, -7.41735964e-03,  3.32844560e-01,  2.13103222e-01,
#         2.57908033e-01,  1.46148001e-01,  5.21373361e-01,  1.89207376e-01,
#         5.07585930e-02, -5.86841466e-02,  2.35228831e-03,  7.04246557e-02,
#         1.03792951e-01, -4.86077052e-02, -8.20913851e-02, -7.21984172e-01,
#         1.13701088e-01,  2.11302187e-01,  1.00825016e-01,  8.44281067e-02,
#        -3.29658968e-02,  8.51465166e-02, -7.94112355e-03, -1.66278548e-02,
#         1.04446177e-01,  1.48938336e-01,  1.73442154e-01, -1.54816210e-01,
#         1.11617492e-01,  1.58584662e-01, -6.58453301e-02,  1.19470477e-01,
#        -2.08725376e-01, -1.02006501e-01, -1.08537991e-01,  1.03621221e-01,
#        -9.53237779e-02, -9.87722447e-02, -2.31870872e-02,  1.77308926e-01,
#         1.90599111e-01, -1.33755352e-01,  1.23003752e-01, -4.23578346e-02,
#        -3.04408560e-02,  1.95143497e-01,  1.49516355e-02, -2.46184832e-01,
#        -3.32396864e-03,  1.45962638e-01,  7.54506343e-02, -2.22937972e-03,
#        -1.33170452e-01,  4.13543434e-02, -4.61399696e-02, -7.43112657e-02,
#         6.44435774e-02,  6.60211180e-03, -3.69238437e-02,  1.36019109e-01,
#         5.59278972e-02,  1.02212229e-01,  9.03401001e-02,  3.76008003e-02,
#        -1.64592182e-01,  5.14657714e-01,  2.81466799e-01, -1.11337061e-01,
#         2.99195153e-02,  4.07502510e-02,  6.13999969e-02,  2.57670230e-01,
#        -1.81091261e-01,  2.04152947e-02,  6.44319936e-02,  1.16757610e-01,
#        -2.06183916e-01,  3.06406657e-01,  8.20203314e-02,  3.04123912e-01,
#         1.03544937e-01,  6.16717138e-02,  7.49349767e-02, -2.17477523e-01,
#         6.03309553e-02,  8.22256996e-02,  2.41962188e-02,  5.44268996e-02,
#         6.84313008e-02,  6.51275700e-02,  3.01498697e-01,  2.73973134e-01,
#        -4.43648150e-01,  1.38651948e-01, -1.34436286e-01, -2.35290352e-01,
#         1.77269901e-01, -1.03394582e-01,  2.32250262e-01, -1.47558757e-01,
#         1.81626437e-02, -1.42681798e-01,  1.32905399e-01, -3.30073420e-02,
#         5.12252267e-02, -1.81536895e-02,  1.45240845e-03, -2.67098066e-02,
#         8.79059949e-02, -4.46729072e-02, -3.58716403e-02, -3.83335258e-02,
#        -3.44099075e-02, -2.99472871e-02,  3.48941115e-02,  4.50587656e-02,
#        -1.38110102e-02, -1.03000721e-01,  6.73443286e-03,  1.28889035e-01,
#         2.03834725e-02, -7.30252313e-02, -4.42656081e-02, -6.13408892e-03,
#         1.55325618e-01,  6.68072475e-03, -5.62915302e-01, -1.93005621e-01,
#        -9.39016491e-01, -8.32813682e-01, -3.48481196e-01,  6.52265525e-01,
#         1.89029372e-02, -8.12778093e-01, -2.30879292e-01,  5.24018801e-02,
#        -1.09753061e+00, -3.88598701e+00, -1.06626234e+00, -4.26184412e-02,
#        -4.27335547e+00, -5.92885668e+00,  2.04573027e+00,  2.11775458e+00,
#         1.12489368e+00,  4.58384837e-02,  8.28301627e-01,  6.76052140e-01,
#         7.72221968e-01, -6.40427727e-01,  1.02803854e+00,  3.15840829e-01,
#        -8.97719566e-01,  6.17717962e-01,  5.45780562e+00,  2.37040072e-01]), np.array([-1.84980072e-01,  8.69533230e-03,  3.28828999e-01, -3.25675522e-01,
#        -1.57900964e-01, -2.21931639e-02,  6.28716066e-01,  4.59835101e-01,
#         3.15598789e-01, -1.54157326e-01,  4.96784200e-01,  5.26688686e-01,
#        -4.54916126e-03, -8.81532393e-02, -1.86539576e-01,  9.95768648e-03,
#         4.36077155e-01,  2.85923742e-01,  3.90733163e-01, -8.67947890e-01,
#        -7.42875730e-03,  2.57207046e-01, -1.80048215e-01, -5.50256990e-02,
#         4.78623602e-01,  2.69345195e-01, -4.71699914e-01,  3.48019774e-01,
#         9.52865937e-02,  1.96820553e-01,  1.02198918e-01, -2.09703721e-01,
#        -2.04987151e-02,  5.74853985e-01, -1.90324859e-01,  4.79940803e-01,
#        -4.38887559e-01, -1.53537817e-01, -2.74318998e-01,  1.14928350e-01,
#        -1.73573985e-01, -2.12976130e-01, -2.25651672e-01,  1.97249691e-01,
#         2.40891880e-01, -1.39923788e-01,  6.94725108e-02, -5.92131997e-03,
#        -4.84840439e-01, -3.34416626e-01,  5.33905910e-02, -3.82298534e-01,
#        -1.93516747e-01,  3.54158320e-01,  4.82366838e-03,  1.82715119e-01,
#         7.39837344e-01,  2.86918025e-02,  3.52677358e-01, -1.15149779e-01,
#         1.04179394e-01,  1.45970899e-01,  2.65993906e-01,  4.93938282e-01,
#         3.10935082e-02,  1.16017583e-01, -4.01741352e-01,  1.78603766e-01,
#        -2.31041086e-01,  1.94057553e-01,  7.31083366e-01, -4.07969902e-01,
#        -1.47691215e-01,  8.44669725e-02, -1.85128914e-02,  4.58403469e-01,
#        -3.64722200e-01,  3.74259878e-01, -5.10378918e-02,  5.26384937e-02,
#         1.07481883e-02,  1.45357608e-01,  5.85853048e-02,  8.95957971e-02,
#        -4.34388557e-02,  4.21872567e-02,  2.31407492e-01, -1.04132802e-01,
#         1.90124701e-01,  1.66701928e-01, -3.26798964e-01,  4.12081888e-02,
#         2.77018749e-03,  3.91815168e-02,  5.03901700e-01, -1.30591346e-01,
#        -4.84131301e-01,  1.57141204e-01, -2.78798677e-01, -3.63843504e-01,
#         2.63538086e-01,  2.57467756e-01,  6.57647375e-02, -6.24682962e-02,
#         1.35140085e-01, -2.45605015e-01, -1.76706688e-01, -1.48940717e-01,
#        -9.76645923e-03, -1.93554620e-01, -2.23455025e-01, -3.22524963e-01,
#        -5.79086255e-01, -4.55858294e-01, -3.98198250e-01, -2.61368223e-01,
#        -1.97982640e-01, -2.14537146e-01, -1.23984555e-01, -1.14557457e-01,
#         1.98769925e-02, -2.03129551e-01,  1.31904245e-01,  6.70382015e-01,
#        -4.54387748e-02, -2.97948619e-01, -4.55107897e-01,  6.42657621e-02,
#         6.84222908e-01,  1.55996216e-01, -7.99670357e-01,  5.10029196e-01,
#        -1.61839819e+00, -2.28509947e-01,  1.40264078e+00,  1.04926447e+00,
#         8.25216846e-01, -5.46958249e+00,  6.10719662e-01,  8.11878954e-01,
#        -4.53618186e+00, -1.41922042e+01, -1.90231407e+00,  6.87498534e+00,
#        -1.22576117e+01, -2.46409916e+01,  1.85524391e+00,  1.35147332e-01,
#         1.90227444e+00, -8.22668860e-02,  1.35844873e+00,  1.60391598e+00,
#         1.98269465e+00, -8.47466122e-01,  1.71435907e+00,  6.28478433e-01,
#        -3.47566295e+00,  5.21471601e-01, -2.01974789e+01,  1.50220866e-01]), np.array([-1.24205143e-01,  2.48677795e-01,  6.44200350e-01, -7.69079486e-02,
#         7.89086173e-02, -6.15239172e-03,  3.18356851e-01,  2.53110890e-02,
#         2.25719256e-01,  2.77782957e-01,  1.82575002e-01,  1.99393229e-01,
#        -5.06906713e-02,  2.99375500e-02,  9.65522507e-03, -2.12527499e-02,
#         1.72350708e-02, -6.85575371e-02,  5.00391711e-02, -7.94378651e-01,
#         6.41059392e-02,  4.32466998e-01, -8.36275056e-02,  6.98542115e-03,
#         8.92431787e-02,  9.53982548e-02, -1.18218897e-01,  1.24071352e-01,
#         6.64568533e-02,  2.99856577e-01,  4.28297192e-02, -2.11089690e-01,
#        -5.83279395e-02,  4.49020230e-01, -1.70476368e-02,  2.14539576e-01,
#        -2.16952731e-01, -2.04884544e-01, -4.89722587e-01,  1.56710676e-01,
#        -2.22926562e-01, -1.46625686e-01, -2.35921796e-01,  3.41292635e-01,
#         3.96438467e-02, -1.15821837e-01,  1.44272922e-01,  1.30556507e-01,
#        -1.50229956e-01,  4.08842123e-02,  1.18466436e-01, -4.85876825e-02,
#        -6.29788921e-02,  3.29208413e-01, -6.77451726e-02,  1.24556329e-01,
#         2.25162898e-01, -7.54968477e-02,  3.80436049e-01, -6.15731131e-02,
#         2.79573092e-02,  6.81237374e-02,  4.39389615e-02,  1.66617049e-01,
#        -2.85481860e-02,  1.57252901e-02, -1.05433527e-01, -1.05750547e-01,
#        -1.80729529e-01,  3.92842311e-02,  5.49072259e-02, -2.29706244e-01,
#        -1.06606323e-01, -1.75240199e-01,  1.48218057e-01,  4.90225711e-01,
#        -1.47770902e-01,  1.25342754e-01,  1.99684785e-01,  1.32947314e-02,
#         5.99830387e-03,  2.81680772e-01,  5.75846615e-02,  2.30201276e-01,
#         1.24970192e-01,  1.93636820e-01, -1.38524543e-01, -6.82098399e-02,
#         1.89713509e-02,  1.59089599e-02,  1.23953440e-01,  4.60736264e-02,
#         8.03265213e-02,  1.94621557e-01,  3.18280452e-01,  1.30687436e-01,
#        -3.56928232e-01,  4.09652395e-02, -8.45107345e-02, -1.92706540e-01,
#         2.23959623e-01, -8.78968988e-03,  2.07153461e-01, -6.87714721e-02,
#         9.36379563e-02, -7.43067397e-02,  1.86098179e-01,  1.26936724e-02,
#         1.87201320e-01,  1.07448610e-01,  1.79592459e-01,  1.77861377e-01,
#         3.94905262e-02,  9.71580055e-02,  1.34680085e-01,  1.86065909e-01,
#         1.85700040e-01,  1.85624334e-01,  2.03760388e-01,  1.97562493e-01,
#         2.31572795e-01, -7.80016895e-02,  2.28658979e-01,  3.81077432e-01,
#         2.46107962e-01,  9.83411240e-03, -1.18233435e-01,  2.34461706e-01,
#         3.48801069e-01,  2.39405766e-01, -1.03787347e+00, -5.58996485e-02,
#        -2.79390607e+00, -1.41011036e+00, -4.86584750e-01,  2.62710056e-01,
#         9.65011448e-02, -4.50920711e+00, -8.38765133e-02,  1.22292577e-01,
#        -4.88584673e+00, -1.06944084e+01, -4.09538048e+00, -2.61339883e+00,
#        -1.87963168e+01, -2.44794709e+01, -1.33938378e+00, -1.26177401e+00,
#         6.98185571e-01,  5.04251087e-01,  7.36163079e-01,  9.24570351e-01,
#         7.87383982e-01, -6.53421285e-01,  8.13050565e-01,  6.65894043e-01,
#        -1.43585246e+00,  7.70201422e-01, -1.77954341e+00,  6.25755155e-01])]
# ALPHA=[np.array([0.29807332, 0.29717696, 0.28761005, 0.40342692, 0.22180662,
#        0.46533814, 0.23290808, 0.7936003 , 0.040098  , 0.5959397 ,
#        0.68665993, 0.2563568 , 0.78304416, 0.3136372 , 0.07959852,
#        0.28887454, 0.4437329 , 0.50671345, 0.75288504, 0.88108057,
#        0.8242546 , 0.7580566 ]), np.array([0.54312783, 0.3160524 , 0.68391114, 0.41658413, 0.9682874 ,
#        0.5510217 , 0.30143932, 0.50246155, 0.31406122, 0.06630494,
#        0.49112192, 0.7853807 , 0.0703951 , 0.10507972, 0.9837269 ,
#        0.9456778 , 0.7887134 , 0.85592896, 0.64122385, 0.75711083,
#        0.41661102, 0.08679412]), np.array([0.7751733 , 0.20868498, 0.7374225 , 0.55788016, 0.32019505,
#        0.41570815, 0.6633579 , 0.13415709, 0.27806857, 0.58283323,
#        0.9479228 , 0.3442502 , 0.53336483, 0.7879667 , 0.5298292 ,
#        0.5109823 , 0.5909165 , 0.8786098 , 0.6016991 , 0.48600915,
#        0.01906334, 0.2101047 ]), np.array([0.42641824, 0.88432884, 0.18427657, 0.5354608 , 0.24708794,
#        0.83509135, 0.02935568, 0.58349776, 0.7929556 , 0.6908727 ,
#        0.074818  , 0.06991037, 0.8471104 , 0.86874783, 0.76129323,
#        0.9867806 , 0.29244   , 0.71936476, 0.7011408 , 0.45805773,
#        0.9827258 , 0.74922544]), np.array([0.67556304, 0.4728644 , 0.8728955 , 0.67381954, 0.02417446,
#        0.7768813 , 0.67088324, 0.3777765 , 0.03202364, 0.28677374,
#        0.3137759 , 0.19020188, 0.47409946, 0.8149911 , 0.00105311,
#        0.38853645, 0.51314014, 0.23630853, 0.6601728 , 0.62627316,
#        0.11174779, 0.47081798]), np.array([0.1541065 , 0.00141862, 0.465272  , 0.44496846, 0.446422  ,
#        0.28667575, 0.7027874 , 0.19161329, 0.88177866, 0.8281354 ,
#        0.68770814, 0.76795244, 0.77261764, 0.28383642, 0.8806148 ,
#        0.00943842, 0.97730553, 0.82355255, 0.38690946, 0.29724726,
#        0.5484669 , 0.7293643 ]), np.array([0.39583838, 0.6844688 , 0.01177795, 0.7666408 , 0.6659102 ,
#        0.6072844 , 0.9486013 , 0.44304118, 0.63933253, 0.64232916,
#        0.7198038 , 0.4171411 , 0.4787745 , 0.16390917, 0.23702186,
#        0.8173987 , 0.29356378, 0.98472744, 0.86326396, 0.2937148 ,
#        0.63177747, 0.1118447 ]), np.array([0.9141349 , 0.55335164, 0.37337676, 0.6015329 , 0.02629782,
#        0.0259124 , 0.18369524, 0.5406321 , 0.68360454, 0.49939135,
#        0.26635367, 0.54132754, 0.53075135, 0.29890156, 0.669048  ,
#        0.48516423, 0.49747443, 0.45066455, 0.09616227, 0.07027485,
#        0.793658  , 0.9015139 ]), np.array([0.828553  , 0.19707641, 0.5339927 , 0.8930001 , 0.51887697,
#        0.1792106 , 0.34214583, 0.5539142 , 0.47724754, 0.504058  ,
#        0.33315971, 0.63336545, 0.05767709, 0.2666583 , 0.4968595 ,
#        0.5952109 , 0.8453954 , 0.5168277 , 0.24041936, 0.60072803,
#        0.82678795, 0.24865809]), np.array([0.0630536 , 0.07596759, 0.39954296, 0.6657399 , 0.03235554,
#        0.54816943, 0.12957834, 0.35725865, 0.7467154 , 0.0910489 ,
#        0.8327059 , 0.18924229, 0.11849681, 0.18327138, 0.09492609,
#        0.09106185, 0.79929996, 0.12444192, 0.30931932, 0.8786104 ,
#        0.08975782, 0.3096414 ]), np.array([0.8658594 , 0.90852034, 0.82953703, 0.91165954, 0.4211833 ,
#        0.21105944, 0.54995835, 0.48338434, 0.984454  , 0.6926109 ,
#        0.19033204, 0.03389291, 0.13056597, 0.723754  , 0.4639835 ,
#        0.7176142 , 0.6089737 , 0.6010264 , 0.9186167 , 0.52127475,
#        0.7266668 , 0.5576527 ]), np.array([0.59009004, 0.28927478, 0.39289224, 0.09063202, 0.60966873,
#        0.48978105, 0.19092026, 0.26016158, 0.26971462, 0.38054103,
#        0.18188806, 0.38421443, 0.7660792 , 0.23609121, 0.0232273 ,
#        0.7378958 , 0.72447574, 0.16522773, 0.8804115 , 0.10049613,
#        0.97108454, 0.75787157])]
# THETA=[np.array([1.0000000e+00, 1.9748417e+00, 2.0663979e+00, 3.0999999e+00,
#        1.0000000e+00, 1.0000000e+00, 1.4700000e-01, 3.7476660e-03,
#        3.6667730e+04, 9.5393920e-01, 5.9799999e-02, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 3.1622776e-01]), np.array([1.0000000e+00, 1.9748417e+00, 2.0663979e+00, 3.0999999e+00,
#        1.0000000e+00, 1.0000000e+00, 1.4700000e-01, 3.7476660e-03,
#        3.6667730e+04, 9.5393920e-01, 5.9799999e-02, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 3.1622776e-01]), np.array([1.0000000e+00, 1.9748417e+00, 2.0663979e+00, 3.0999999e+00,
#        1.0000000e+00, 1.0000000e+00, 1.4700000e-01, 3.7476660e-03,
#        3.6667730e+04, 9.5393920e-01, 5.9799999e-02, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 3.1622776e-01]), np.array([1.0000000e+00, 1.9748417e+00, 2.0663979e+00, 3.0999999e+00,
#        1.0000000e+00, 1.0000000e+00, 1.4700000e-01, 3.7476660e-03,
#        3.6667730e+04, 9.5393920e-01, 5.9799999e-02, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 3.1622776e-01]), np.array([1.0000000e+00, 1.9748417e+00, 2.0663979e+00, 3.0999999e+00,
#        1.0000000e+00, 1.0000000e+00, 1.4700000e-01, 3.7476660e-03,
#        3.6667730e+04, 9.5393920e-01, 5.9799999e-02, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 3.1622776e-01]), np.array([1.0000000e+00, 1.9748417e+00, 2.0663979e+00, 3.0999999e+00,
#        1.0000000e+00, 1.0000000e+00, 1.4700000e-01, 3.7476660e-03,
#        3.6667730e+04, 9.5393920e-01, 5.9799999e-02, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 3.1622776e-01]), np.array([1.0000000e+00, 1.9748417e+00, 2.0663979e+00, 3.0999999e+00,
#        1.0000000e+00, 1.0000000e+00, 1.4700000e-01, 3.7476660e-03,
#        3.6667730e+04, 9.5393920e-01, 5.9799999e-02, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 3.1622776e-01]), np.array([1.0000000e+00, 1.9748417e+00, 2.0663979e+00, 3.0999999e+00,
#        1.0000000e+00, 1.0000000e+00, 1.4700000e-01, 3.7476660e-03,
#        3.6667730e+04, 9.5393920e-01, 5.9799999e-02, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 3.1622776e-01]), np.array([1.0000000e+00, 1.9748417e+00, 2.0663979e+00, 3.0999999e+00,
#        1.0000000e+00, 1.0000000e+00, 1.4700000e-01, 3.7476660e-03,
#        3.6667730e+04, 9.5393920e-01, 5.9799999e-02, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 3.1622776e-01]), np.array([1.0000000e+00, 1.9748417e+00, 2.0663979e+00, 3.0999999e+00,
#        1.0000000e+00, 1.0000000e+00, 1.4700000e-01, 3.7476660e-03,
#        3.6667730e+04, 9.5393920e-01, 5.9799999e-02, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 3.1622776e-01]), np.array([1.0000000e+00, 1.9748417e+00, 2.0663979e+00, 3.0999999e+00,
#        1.0000000e+00, 1.0000000e+00, 1.4700000e-01, 3.7476660e-03,
#        3.6667730e+04, 9.5393920e-01, 5.9799999e-02, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 3.1622776e-01]), np.array([1.0000000e+00, 1.9748417e+00, 2.0663979e+00, 3.0999999e+00,
#        1.0000000e+00, 1.0000000e+00, 1.4700000e-01, 3.7476660e-03,
#        3.6667730e+04, 9.5393920e-01, 5.9799999e-02, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
#        1.0000000e+00, 1.0000000e+00, 3.1622776e-01])]

PREDICTION=fill_PRED(ALPHA,THETA)
(R,P,R2,P2,SMAPE)=perf(PREDICTION,6)
print(color[2]+f"{R}"+color[0])
print(color[2]+f"{P}"+color[0])
print(color[2]+f"{SMAPE}"+color[0])
#########################################################################################################################
#                   Prints & Plots
##########################################################################################################################

datesRAW = [datetime.strptime(date, '%Y-%m') for date in timeline]
dates_Y=(np.array(datesRAW))[~mask3]
dates = (np.array(datesRAW))[~mask2]

plt.figure(1)
plt.plot(dates, Ret, color='grey', label='Actual Return',linewidth=3)
COLOR=["orange","b","g","cyan","darkblue","y","darkgreen","magenta","r","darkmagenta","darkred","chocolate","black","gray"]
CCC=range(len(PREDICTION))
for tau in CCC:
    plt.plot(dates[:n], (PREDICTION[tau])[:n], color=COLOR[tau], label=f"algo{1+tau}",linewidth=1)
    plt.plot(dates[n-1:], (PREDICTION[tau])[n-1:], color=COLOR[tau], dashes=[4, 2],linewidth=1)

# plt.plot(dates[:n], Y_prediction[:n], color='r', label=f"Learned Kernel",linewidth=1)
# plt.plot(dates[n-1:], Y_prediction[n-1:], color='r', dashes=[4, 2],linewidth=1)
# plt.plot(dates[:n], avg[:n], color="g", label=f"Learned Kernel avg",linewidth=1)
# plt.plot(dates[n-1:], avg[n-1:], color="g", dashes=[4, 2],linewidth=1)


plt.title(f"{FIRMname[firm]}'s Returns")
plt.legend(loc='lower left')
plt.grid(True,axis='y')
plt.plot([datetime.strptime('2006-01', '%Y-%m'),datetime.strptime('2006-01', '%Y-%m')],[-100,100],color='grey',linewidth=0.4)
    
[a,b]=[n-12*6, n+12*1+6]
plt.xlim(dates[a] , dates[b] )
# plt.ylim(-0.01 , 0.01)
plt.ylim(np.min(Ret[a:b])-0.1 , np.max(Ret[a:b]+0.1))
plt.ylim(np.min(Ret[a:b])+0.2 , np.max(Ret[a:b]-0.3))
plt.ylabel('return (%)')
plt.savefig(f"Plot/CRSP/kernel_learning/CRSP_learn[{firm}]_1.png")#[{tau}]


plt.figure(2)
plt.title(f"{FIRMname[firm]}'s Returns")
plt.plot(dates, Ret, color='grey', label='Actual Return',linewidth=3)
plt.grid(True,axis='y')
plt.plot([datetime.strptime('2006-01', '%Y-%m'),datetime.strptime('2006-01', '%Y-%m')],[-100,100],color='grey',linewidth=0.4)
plt.xlim(dates[n-12] , dates[n+18] )
plt.xticks(rotation = 30)
# plt.ylim(-0.5 , 0.4)
# plt.ylim(np.min(Ret[a:b])+0.1 , np.max(Ret[a:b]-0.1))
plt.ylim(np.min(Ret[a:b])+0.2 , np.max(Ret[a:b]-1.5))
plt.ylabel('return (%)')

for tau in CCC:
    plt.plot(dates[:n], (PREDICTION[tau])[:n], color=COLOR[tau], label=f"algo{1+tau}",linewidth=1)
    plt.plot(dates[n-1:], (PREDICTION[tau])[n-1:], color=COLOR[tau], dashes=[4, 2],linewidth=1)

# plt.plot(dates[:n], avg[:n], color="g", label=f"Learned Kernel avg",linewidth=1)
# plt.plot(dates[n-1:], avg[n-1:], color="g", dashes=[4, 2],linewidth=1)


# plt.plot(dates[:n], Y_prediction[:n], color='r', label=f"Learned Kernel",linewidth=1)
# plt.plot(dates[n-1:], Y_prediction[n-1:], color='r', dashes=[4, 2],linewidth=1)

plt.legend(loc='lower left')
plt.savefig(f"Plot/CRSP/kernel_learning/CRSP_learn[{firm}]_2.png")