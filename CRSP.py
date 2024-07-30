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
import numpy as np
from collections import Counter
from datetime import datetime, timedelta
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, RBF, WhiteKernel, Sum, Product
from sklearn.metrics.pairwise import polynomial_kernel

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
    avg=np.squeeze(avg)
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

print(color[4]+"o "+color[0]+"Basics matrices done \n")
               
########################################################################################################################
#                   Choice of the studied firm
########################################################################################################################
#
for firm in [1]:
    # firm=0
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
    def Alpha (k,x,y,lam) :
        n = len(x)
        K = np.zeros((n,n))
        for i in range(n) :
            for j in range(n) :
                K[i,j] = k( x[i] , x[j] )
        return np.linalg.inv(K + lam * np.identity(n)).dot(y)

    #- Kernel Model
    L1=10
    lam1 = 0.1
    L2=1
    lam2 = 1

    def SUM_ABS(x):
        s=0
        for e in x:
            s+=abs(e)
        return s
    #- Kernels Definition
    def k_p (x,y) :
        return (np.dot ( x, y )+ 1)**3
    def k_g (x,y) :
        return np.exp(-SUM_ABS( x-y)**2)/np.sqrt(2*np.pi)
    def k_g_old (x,y) :
        return np.exp(-np.dot( x-y, x-y))/np.sqrt(2*np.pi)
    def k_e (x,y) :
        return 3/2*(1-4*(np.dot( x,y ))**2)
    def k_ESQ (x,y) :
        return np.exp(L1*np.sin(np.dot ( x,y)))/np.sqrt(2*np.pi)
    def k_ESQ2 (x,y) :
        return np.exp(L2*np.sin(np.dot ( x, y)))/np.sqrt(2*np.pi)
    # print(color[-1]+"o "+color[0]+"Manual regressions have started", end='\r')

    alpha_e = Alpha( k_e , X,Y,lam2)
    alpha_esq2 = Alpha( k_ESQ2 , X,Y,lam2)
    alpha_p = Alpha( k_p , X,Y,lam1 )
    alpha_esq = Alpha( k_ESQ , X,Y,lam1 )
    alpha_g = Alpha( k_g , X,Y,lam2 )
    alpha_g_old = Alpha( k_g_old , X,Y,lam1 )
    e, g, g_old, p, esq1, esq2= [], [], [], [], [], []
    for j in range(len(INPUT)):
        S_e, S_g, S_p, S_esq1, S_esq2, S_g_old = 0, 0, 0, 0, 0, 0
        for i in range(n):
            S_e += alpha_e[i] * k_e(X[i], INPUT[j])
            S_g += alpha_g[i] * k_g(X[i], INPUT[j])
            S_g_old += alpha_g_old[i] * k_g_old(X[i], INPUT[j])
            S_p += alpha_p[i] * k_p(X[i], INPUT[j])
            S_esq1 += alpha_esq[i] * k_ESQ(X[i], INPUT[j])
            S_esq2 += alpha_esq2[i] * k_ESQ2(X[i], INPUT[j])
        e.append(S_e)
        g.append(S_g)
        g_old.append(S_g_old)
        p.append(S_p)
        esq2.append(S_esq2)
        esq1.append(S_esq1)

   
    
    e=np.matrix(e).transpose()
    g=np.matrix(g).transpose()
    g_old=np.matrix(g_old).transpose()
    p=np.matrix(p).transpose()
    esq2=np.matrix(esq2).transpose()
    esq1=np.matrix(esq1).transpose()
    print(color[4]+"o "+color[0]+"Manual regression done          \n")

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
    POLY=[KernelRidge(kernel="polynomial",gamma= 4.65, degree= 3.4, coef0= 0.60, alpha= 14.98),
        KernelRidge(kernel="polynomial",gamma= 1.34, degree= 3.3, coef0= 0.65, alpha= 10.6),
        KernelRidge(kernel="polynomial"),
        KernelRidge(kernel="polynomial"),
        KernelRidge(kernel="polynomial"),
        KernelRidge(kernel="polynomial"),
        ]
    poly=POLY[firm]
    poly.fit(X,Y)
    Y_pred_p1 = poly.predict(INPUT)
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
    Y_pred_p2 = train_model(poly,param_poly2,1000)

    #------------------ Periodic Kernel: EXPSINESQUARED -----------------------------------------------------------------------------

    ESQ=np.array([[KernelRidge(kernel=Sum(ExpSineSquared(2.16, 0.08), WhiteKernel(0.5)),alpha=8.69),KernelRidge(kernel=Sum(ExpSineSquared(0.17, 0.14), WhiteKernel(0.1)),alpha=12.3)],
                [KernelRidge(kernel=Sum(ExpSineSquared(0.06,0.91),WhiteKernel(noise_level=0.2)),alpha=1.68),KernelRidge(kernel=Sum(ExpSineSquared(0.175,0.52),WhiteKernel(noise_level=0.4)),alpha=6.79)],
                [KernelRidge(kernel=Sum(ExpSineSquared(0.06,0.91),WhiteKernel(noise_level=0.2)),alpha=1.68),KernelRidge(kernel=Sum(ExpSineSquared(0.175,0.52),WhiteKernel(noise_level=0.4)),alpha=6.79)],
                [KernelRidge(kernel=Sum(ExpSineSquared(0.06,0.91),WhiteKernel(noise_level=0.2)),alpha=1.68),KernelRidge(kernel=Sum(ExpSineSquared(0.175,0.52),WhiteKernel(noise_level=0.4)),alpha=6.79)],
                [KernelRidge(kernel=Sum(ExpSineSquared(0.06,0.91),WhiteKernel(noise_level=0.2)),alpha=1.68),KernelRidge(kernel=Sum(ExpSineSquared(0.175,0.52),WhiteKernel(noise_level=0.4)),alpha=6.79)],
                [KernelRidge(kernel=Sum(ExpSineSquared(0.06,0.91),WhiteKernel(noise_level=0.2)),alpha=1.68),KernelRidge(kernel=Sum(ExpSineSquared(0.175,0.52),WhiteKernel(noise_level=0.4)),alpha=6.79)],
                            
                ])
    esq=ESQ[firm,0]
    esq.fit(X,Y)
    Y_pred_esq=esq.predict(INPUT)
    #- Tuned Model
    # print(color[-1]+"o "+color[0]+"KR ESQ tuning has started", end='\r')

    param_distributions = {
        "alpha": loguniform(0.1, 1e2) ,
        "kernel__k1__length_scale": loguniform(1e-3, 1e3),  # length_scale du premier sous-noyau (ExpSineSquared)
        "kernel__k1__periodicity": loguniform(1e-3, 1e3),  # periodicity du premier sous-noyau (ExpSineSquared)
        "kernel__k2__noise_level": loguniform(1e-1, 1)  
    }
    param_distributions2 = {
        "alpha": np.arange(1e-2, 5,0.01) ,
        "kernel__k1__length_scale": loguniform(0.1, 1e1),  # length_scale du premier sous-noyau (ExpSineSquared)
        "kernel__k1__periodicity": np.arange(0.01, 10,0.01),  # periodicity du premier sous-noyau (ExpSineSquared)
        "kernel__k2__noise_level": np.arange(0, 1,0.1)
        }
    Y_pred_esq2 = train_model(esq,param_distributions,500) #{'kernel__k1__periodicity': 4.5, 'kernel__k1__length_scale': 11.4, 'alpha': 0.5}


    #----------------- Locally Periodic Kernel -----------------------------------------------------------------------------
    LPK=[KernelRidge(kernel=Product(ExpSineSquared(0.068,0.521),RBF(69.13776)),alpha=3.09),
        KernelRidge(kernel=Product(ExpSineSquared(0.0053,0.147436),RBF(51856)),alpha=1.04),
        KernelRidge(kernel=Product(ExpSineSquared(),RBF())),
        KernelRidge(kernel=Product(ExpSineSquared(),RBF())),
        KernelRidge(kernel=Product(ExpSineSquared(),RBF())),
        KernelRidge(kernel=Product(ExpSineSquared(),RBF())),
        ]
    lpk=LPK[firm]
    lpk.fit(X,Y)
    Y_pred_lpk=lpk.predict(INPUT)

    param_lpk_global = {
        "alpha": np.arange(1e-1, 1e2,0.01) ,
        "kernel__k1__length_scale": loguniform(1e-3, 1e5),  # length_scale du premier sous-noyau (ExpSineSquared)
        "kernel__k1__periodicity": loguniform(1e-3, 1e0),  # periodicity du premier sous-noyau (ExpSineSquared)
        "kernel__k2__length_scale": loguniform(1e-3, 1e4)  # length_scale du deuxième sous-noyau (RBF)
    }
    param_lpk_fine = {
        "alpha": np.arange(1e-1, 5,0.01) ,
        "kernel__k1__length_scale": np.arange(1e-3, 1e0,1e-3),  # length_scale du premier sous-noyau (ExpSineSquared)
        "kernel__k1__periodicity": np.arange(1e1, 1e2,1e-3),  # periodicity du premier sous-noyau (ExpSineSquared)
        "kernel__k2__length_scale": loguniform(1e2, 1e5)  # length_scale du deuxième sous-noyau (RBF)
    }
    Y_pred_lpk2=train_model(lpk,param_lpk_global,500)
    #{'alpha': 3.0899999999999985, 'kernel__k1__length_scale': 0.068, 'kernel__k1__periodicity': 0.5209999999999996, 'kernel__k2__length_scale': 69.13775884432668}

    param_lpk2_global = {
        "alpha": np.arange(1e-1, 1e2,0.01) ,
        "kernel__k1__length_scale": loguniform(1e-3, 1e3),  # length_scale du premier sous-noyau (ExpSineSquared)
        "kernel__k1__periodicity": loguniform(1e-3, 1e3),  # periodicity du premier sous-noyau (ExpSineSquared)
        "kernel__k2__degree": np.arange(1, 10,0.1) ,
        "kernel__k2__coef0": np.arange(0, 10,0.01),
        "kernel__k2__gamma": np.arange(3, 10,0.01)
    }

    
    
    PREDICTION=(Y_pred_lpk,esq1,Y_pred_esq2,e,p,Y_pred_p2,Y_pred_linear)
    
    (R,P,R2,P2,SMAPE)=perf(PREDICTION,6)
    print(color[3]+f"{R2}"+color[0])
    print(color[3]+f"{P2}"+color[0])
    print(color[2]+f"{R}"+color[0])
    print(color[2]+f"{P}"+color[0])
    print(color[2]+f"{SMAPE}"+color[0])
    

    # print(perf(Y_pred_lpk,avg),18)

    #########################################################################################################################
    #                   Prints & Plots
    ##########################################################################################################################

    datesRAW = [datetime.strptime(date, '%Y-%m') for date in timeline]
    dates_Y=(np.array(datesRAW))[~mask3]
    dates = (np.array(datesRAW))[~mask2]
    
    [a,b]=[n-12*6, n+12*1+6]
    op=1
    #----------------- Manual Regression -------------------------------------
    plt.figure(1)
    plt.title(f"{FIRMname[firm]}'s Returns")
    plt.plot(dates, Ret, color='grey', label='Actual Return',linewidth=3)
    plt.grid(True,axis='y')
    plt.plot([datetime.strptime('2006-01', '%Y-%m'),datetime.strptime('2006-01', '%Y-%m')],[-100,100],color='grey',linewidth=0.4)
    plt.xlim(dates[a] , dates[b] )
    # plt.ylim(-0.01 , 0.01)
    plt.ylim(np.min(Ret[a:b])-0.1 , np.max(Ret[a:b]+0.1))
    # plt.ylim(np.min(Ret[a:b])+0.2 , np.max(Ret[a:b]-1.2))
    plt.ylabel('return (%)')
    

    plt.plot(dates[:n], e[:n], color='b', label='epanechnikov',linewidth=1, alpha=op)
    plt.plot(dates[n-1:], e[n-1:], color='b', dashes=[4, 2],linewidth=1, alpha=1)

    # plt.plot(dates[:n], p[:n], color='darkblue', label='MKR polynomial',linewidth=1, alpha=op)
    # plt.plot(dates[n-1:], p[n-1:], color='darkblue', dashes=[4, 2],linewidth=1, alpha=1)

    plt.plot(dates[:n], g[:n], color='green', label='MKR Gaussian',linewidth=1)
    plt.plot(dates[n-1:], g[n-1:], color='green', dashes=[4, 2],linewidth=1)

    # plt.plot(dates[:n], g_old[:n], color='darkgreen', label='MKR Old Gaussian',linewidth=1)
    # plt.plot(dates[n-1:], g_old[n-1:], color='darkgreen', dashes=[4, 2],linewidth=1)

    plt.plot(dates[:n], esq1[:n], color='darkred', label='MKR ExpSine²',linewidth=1, alpha=op)
    plt.plot(dates[n-1:], esq1[n-1:], color='darkred', dashes=[4, 2],linewidth=1, alpha=1)

    # plt.plot(timelineflt[:n], esq2[:n], color='green', label='MKR ExpSin² lambda=1',linewidth=1)
    # plt.plot(timelineflt[n-1:], esq2[n-1:], color='green', dashes=[4, 2],linewidth=1)
    plt.legend()
    plt.savefig(f"Plot/CRSP/CRSP_firm2#{firm}_1.png")  
    
    
    
    
    #----------------- Sklearn Regression -------------------------------------   
    plt.figure(2)
    plt.title(f"{FIRMname[firm]}'s Returns")
    plt.plot(dates, Ret, color='grey', label='Actual Return',linewidth=3)
    plt.grid(True,axis='y')
    plt.plot([datetime.strptime('2006-01', '%Y-%m'),datetime.strptime('2006-01', '%Y-%m')],[-100,100],color='grey',linewidth=0.4)
    plt.xlim(dates[a] , dates[b] )
    # plt.ylim(-0.01 , 0.01)
    # plt.ylim(np.min(Ret[a:b])+0.1 , np.max(Ret[a:b]-0.1))
    plt.ylim(np.min(Ret[a:b])+0.1 ,0.6)
    
    # plt.ylim(np.min(Ret[a:b])+0.2 , np.max(Ret[a:b]-1.2))
    plt.ylabel('return (%)')


    # plt.plot(dates_Y[:n], Y_pred_linear[:n], color='y', label='CAPM',linewidth=1)
    # plt.plot(dates_Y[n-1:], Y_pred_linear[n-1:], color='y', dashes=[4, 2],linewidth=1)

    # plt.plot(dates[:n], Y_pred_p1[:n], color='r', label='Poly',linewidth=1)
    # plt.plot(dates[n-1:], Y_pred_p1[n-1:], color='r', dashes=[4, 2],linewidth=1)

    plt.plot(dates[:n], Y_pred_p2[:n], color='darkblue', label='Poly tuned',linewidth=1, alpha=op)
    plt.plot(dates[n-1:], Y_pred_p2[n-1:], color='darkblue', dashes=[4, 2],linewidth=1, alpha=1)

    # plt.plot(dates[:n], Y_pred_esq[:n], color='red', label='ExpSin²',linewidth=1)
    # plt.plot(dates[n-1:], Y_pred_esq[n-1:], color='red', dashes=[4, 2],linewidth=1)

    plt.plot(dates[:n], Y_pred_esq2[:n], color='darkred', label='ExpSine² tuned',linewidth=1, alpha=op)
    plt.plot(dates[n-1:], Y_pred_esq2[n-1:], color='darkred', dashes=[4, 2],linewidth=1, alpha=1)

    # plt.plot(dates[:n], Y_pred_lpk[:n], color='r', label='lpk',linewidth=1, alpha=op)
    # plt.plot(dates[n-1:], Y_pred_lpk[n-1:], color='r', dashes=[4, 2],linewidth=1, alpha=1)

    plt.plot(dates[:n], Y_pred_lpk2[:n], color='red', label='lpk tuned',linewidth=1)
    plt.plot(dates[n-1:], Y_pred_lpk2[n-1:], color='red', dashes=[4, 2],linewidth=1)

    # plt.plot(dates[:n], avg[:n], color='r', label='avg',linewidth=1)
    # plt.plot(dates[n-1:], avg[n-1:], color='r', dashes=[4, 2],linewidth=1)

    # plt.plot(dates[:n], Y_pred_gauss[:n], color='g', label='G ExpSin²')
    # plt.plot(dates[n-1:], Y_pred_gauss[n-1:], color='g', linestyle="dotted")

    # plt.plot(dates[:n], Y_pred_gauss_tuned[:n], color='y', label='G ExpSin² tuned')
    # plt.plot(dates[n-1:], Y_pred_gauss_tuned[n-1:], color='y', linestyle="dotted")
    plt.legend()
    plt.savefig(f"Plot/CRSP/CRSP_firm2#{firm}_2.png")
    
    
     #-----------------  COMP ------------------------------------- 
    
    plt.figure(3)
    plt.title(f"{FIRMname[firm]}'s Returns")
    plt.plot(dates, Ret, color='grey', label='Actual Return',linewidth=3)
    plt.grid(True,axis='y')
    plt.plot([datetime.strptime('2006-01', '%Y-%m'),datetime.strptime('2006-01', '%Y-%m')],[-100,100],color='grey',linewidth=0.4)
    plt.xlim(dates[n-12] , dates[n+18] )
    plt.xticks(rotation = 30)
    plt.ylim(-0.7 , 1)
    # plt.ylim(np.min(Ret[a:b])+0.1 , np.max(Ret[a:b]-0.1))
    # plt.ylim(np.min(Ret[a:b])+0.2 , np.max(Ret[a:b]-1.2))
    plt.ylabel('return (%)')

    # plt.plot(dates[:n], e[:n], color='b', label='epanechnikov',linewidth=1)
    # plt.plot(dates[n-1:], e[n-1:], color='b', dashes=[4, 2],linewidth=1)

    # plt.plot(dates[:n], p[:n], color='y', label='MKR polynomial',linewidth=1)
    # plt.plot(dates[n-1:], p[n-1:], color='y', dashes=[4, 2],linewidth=1)

    # plt.plot(dates[:n], g[:n], color='green', label='Gaussian',linewidth=1)
    # plt.plot(dates[n-1:], g[n-1:], color='green', dashes=[4, 2],linewidth=1)

    # plt.plot(dates[:n], g_old[:n], color='darkgreen', label='MKR Old Gaussian',linewidth=1)
    # plt.plot(dates[n-1:], g_old[n-1:], color='darkgreen', dashes=[4, 2],linewidth=1)

    # plt.plot(dates[:n], esq1[:n], color='darkred', label='ExpSine²',linewidth=1)
    # plt.plot(dates[n-1:], esq1[n-1:], color='darkred', dashes=[4, 2],linewidth=1)

    # plt.plot(timelineflt[:n], esq2[:n], color='green', label='MKR ExpSin² lambda=1',linewidth=1)
    # plt.plot(timelineflt[n-1:], esq2[n-1:], color='green', dashes=[4, 2],linewidth=1)

    plt.plot(dates_Y[:n], Y_pred_linear[:n], color='y', label='CAPM',linewidth=1)
    plt.plot(dates_Y[n-1:], Y_pred_linear[n-1:], color='y', dashes=[4, 2],linewidth=1)

    # plt.plot(dates[:n], Y_pred_p1[:n], color='r', label='Poly',linewidth=1)
    # plt.plot(dates[n-1:], Y_pred_p1[n-1:], color='r', dashes=[4, 2],linewidth=1)

    plt.plot(dates[:n], Y_pred_p2[:n], color='darkblue', label='Poly tuned',linewidth=1)
    plt.plot(dates[n-1:], Y_pred_p2[n-1:], color='darkblue', dashes=[4, 2],linewidth=1)

    # plt.plot(dates[:n], Y_pred_esq[:n], color='darkred', label='ExpSin²',linewidth=1)
    # plt.plot(dates[n-1:], Y_pred_esq[n-1:], color='darkred', dashes=[4, 2],linewidth=1)

    # plt.plot(dates[:n], Y_pred_esq2[:n], color='darkred', label='ExpSin² tuned',linewidth=1)
    # plt.plot(dates[n-1:], Y_pred_esq2[n-1:], color='darkred', dashes=[4, 2],linewidth=1)

    # plt.plot(dates[:n], Y_pred_lpk[:n], color='r', label='lpk',linewidth=1)
    # plt.plot(dates[n-1:], Y_pred_lpk[n-1:], color='r', dashes=[4, 2],linewidth=1)

    plt.plot(dates[:n], Y_pred_lpk2[:n], color='red', label='lpk tuned',linewidth=1)
    plt.plot(dates[n-1:], Y_pred_lpk2[n-1:], color='red', dashes=[4, 2],linewidth=1)

    # plt.plot(dates[:n], avg[:n], color='r', label='avg',linewidth=1)
    # plt.plot(dates[n-1:], avg[n-1:], color='r', dashes=[4, 2],linewidth=1)

    # plt.plot(dates[:n], Y_pred_gauss[:n], color='g', label='G ExpSin²')
    # plt.plot(dates[n-1:], Y_pred_gauss[n-1:], color='g', linestyle="dotted")

    # plt.plot(dates[:n], Y_pred_gauss_tuned[:n], color='y', label='G ExpSin² tuned')
    # plt.plot(dates[n-1:], Y_pred_gauss_tuned[n-1:], color='y', linestyle="dotted")
    
    # plt.plot(dates[n-1:], avg[n-1:], color='darkorange', dashes=[4, 1],linewidth=1)
    plt.legend()
    plt.savefig(f"Plot/CRSP/CRSP_firm2#{firm}_3.png")