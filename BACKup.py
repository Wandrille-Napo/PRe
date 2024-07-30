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

#########################################################################
#                   CSV to basic matrices
#########################################################################

# Path to the CSV file
file_path1 = 'data/monthly_CRSP2.csv'
C=[2,9,50,63,7]#60

file_path2 = 'data/CCM_annual_1989-2024.csv'
C2=[3,10,95,279,9]

# CCM=read_csv_to_matrix(file_path2,C2)
CRSP=read_csv_to_matrix(file_path1,C)
DGS10=read_csv_to_matrix('data/monthly_DGS10.csv',[])
firm_column = CRSP[:, 1].A1
count_firm=Counter(firm_column)
nb_firm_name=len(count_firm)
max_time_window=count_firm[max(count_firm, key=count_firm.get)]

#########################################################################
#                   Matrices    cusip   at/emp  monthly var    eric so
#########################################################################

#- timeline & dates
# timeline=[]
# current_date = datetime.strptime('1997-01-02', '%Y-%m-%d')
# while current_date.strftime('%Y-%m-%d') != '2023-12-31':
#     timeline.append(current_date.strftime('%Y-%m-%d'))
#     current_date += timedelta(days=1)
# dates = [datetime.strptime(date, '%Y-%m-%d') for date in timeline]
# N=len(timeline)

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

#########################################################################
#                  Firm filter
#########################################################################


# firm_in_2_datasets=[]
# for x in CCM[:,1]:
#     if x in FIRMname2 and FIRMname2.index(x) not in firm_in_2_datasets:
#         firm_in_2_datasets.append(FIRMname2.index(x))
# print(f"{len(firm_in_2_datasets)}/{nb_firm}")
# print(firm_in_2_datasets)

# firm_in_2_datasets=[128, 121, 21, 28, 33, 22, 11, 27, 14, 36, 35, 31, 37, 70, 67, 74, 93, 104, 101, 112, 109, 108, 118, 116, 99, 110, 119, 134, 25, 132, 139, 105, 133, 15, 144, 147, 140, 151, 146, 153, 131, 129, 149, 150, 154, 51, 124, 73, 71, 8, 10, 30, 125, 38, 69, 152, 111, 88, 55, 2, 100, 138, 19, 90, 77, 122]
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
#     if (row[2] == '' or row[3] == '') and (row[1] in FIRMname3):
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
# # valid_firm=['A T C GROUP SERVICES INC', 'DALLAS GOLD & SILVER EXCHANGE IN', 'GOOD GUYS INC', 'FRANKLIN BANK NATL ASSOC', 'ROCKY MOUNTAIN CHOCOLATE FAC INC', 'CENTURY COMMUNICATIONS CORP', 'ADAMS EXPRESS CO', 'ALEX BROWN INC', 'V W R SCIENTIFIC PRODUCTS CORP', 'U I C I', 'ALPHA TECHNOLOGIES GROUP INC', '4 KIDS ENTERTAINMENT INC', 'ORACLE CORP', 'MICROSOFT CORP', 'ACCLAIM ENTERTAINMENT INC', '20TH CENTURY INDUSTRIES', 'PIPER JAFFRAY COS INC', 'MEDFORD SAVINGS BANK', 'ACKERLEY COMMUNICATIONS INC', 'ELECTRIC & GAS TECHNOLOGY INC', 'FAMILY STEAK HOUSES FL INC', 'ATLANTIC POWER CORP', 'WESTCORP INC', 'PLASTI LINE INC', 'WATER JEL TECHNOLOGIES INC', 'CAMELOT INFORMATION SYSTEMS INC', 'AMERICAN BRANDS INC', 'ELMIRA SAVINGS BANK FSB NY', 'SIGMA DESIGNS INC', 'SENETEK PLC', 'BRENDLES INC', 'TOMPKINS COUNTY TRUSTCO INC', 'CENTER BANKS INC', 'BALCHEM CORP', 'XOMA CORP', 'ABINGTON SAVINGS BANK', 'AVONDALE INDUSTRIES INC', 'BUTLER INTERNATIONAL INC NEW', 'ADVANCED MAGNETICS INC', 'ASARCO INC', 'CANDELA CORP', 'CYTOGEN CORP', 'A T C COMMUNICATIONS INC', 'TRENWICK GROUP INC', 'WERNER ENTERPRISES INC', 'INTERLEAF INC', 'TRANSMEDIA NETWORK INC', 'CAPSTONE PHARMACY SERVICES INC', 'GENICOM CORP', 'A S A INTERNATIONAL LTD', 'ALPNET INC', 'RYAN BECK & CO INC', 'L A GEAR INC', 'COMMNET CELLULAR INC']

# df_filtered = pd.DataFrame(CCM, columns=['Year', 'Company', 'at', 'emp', 'cusip'])

# output_csv_path = 'data/CCM.csv'
# df_filtered.to_csv(output_csv_path, index=False)

#########################################################################
#                   Matrices 2
#########################################################################


CCM=read_csv_to_matrix('data/CCM.csv',[])
CCM_FIRMname=list(np.unique(CCM[1:,1].A1))
CCM = np.array(CCM,dtype=object)
# ['ASTEC INDUSTRIES INC', 'DAILY JOURNAL CORP', 'J & J SNACK FOODS CORP', 'MICROSOFT CORP', 'PLEXUS CORP', 'REPLIGEN CORP', 'SKYWEST INC', 'WERNER ENTERPRISES INC', 'XOMA CORP']
nb_firm=len(CCM_FIRMname)

EMP=np.empty((N,nb_firm), dtype=object)
AT=np.empty((N,nb_firm), dtype=object)
for x in CCM_FIRMname:
    idx = list(CCM[:,1]).index(x)
    firmIDX = CCM_FIRMname.index(x)
    idx2 = idx + list(CCM[idx:,0]).index(1997)
    i=idx2
    while 12*(i-idx2)<AT.shape[0] and CCM[i,1]==x:
        for k in range(12):
            AT[12*(i-idx2)+k,firmIDX]=float(CCM[idx2,2])
        for k in range(12):
            EMP[12*(i-idx2)+k,firmIDX]=int(CCM[idx2,3]*1000)
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
    if CRSP[i,1] in CCM_FIRMname:
        firm_index = CCM_FIRMname.index(CRSP[i,1])
        index = timeline.index(CRSP[i,0])
        if CRSP[i,2] != '':
            print(f"{firm_index}/{nb_firm}  |  {CRSP[i,2]}")
        if Rf[index] != None :
            SP500[index] = CRSP[i,3]
        try:
            RET[index,firm_index] = float(CRSP[i,2])
            print(CRSP[i,2])
        except:
            RET[index,firm_index] = None
print()
# print(RET)
print()
CRSP=np.array(CRSP)
CRSP2= np.array([row for row in CRSP if row[1] in CCM_FIRMname])

print(CRSP2)
end=timeline.index('2023-12')

CRSP[:end]
SP500[:end]
Rf[:end]
RET[:end]

#- equalizing nb of None regarding SP500 & Rf
for i in range(N):
    if SP500[i] == None or Rf[i] == None:
        SP500[i] = None
        Rf[i] = None
        for k in range(nb_firm):
            RET[i,k]= None

#trouvez des entreprises qui a at, emp, RET(X)
#prendre un panel plus large dans CRSP




############################################################
#                   Regressions
############################################################
# # for firm in range(nb_firm):
# firm=1
# #trouver le emp et le at de la firm en question.
# #- Definition of X and Y
# X = np.empty((N,3), dtype=object)
# Y = np.empty((N,1), dtype=object)

# m=np.nanmean(np.array([x if x is not None else np.nan for x in Rf], dtype=float))
# for i in range(N):
#     if (SP500[i] != None):
#         X[i,0] = SP500[i] - Rf[i]/365
#         Y[i]  = Y[i,firm]      
# X = X.astype(float)
# Y = Y.astype(float)

# valid_indices = np.where(~np.isnan(X).flatten() & ~np.isnan(Y).flatten())[0]
# X = X[valid_indices]
# Y = Y[valid_indices]

# n=len(X)

# #- Regression

# #
# model = LinearRegression()
# model.fit(X[int(n/2)-180:int(n/2)],Y[int(n/2)-180:int(n/2)])
# alpha = model.intercept_[0]
# beta = model.coef_[0][0]
# # print(f"({alpha},{beta})")
# print(color[4]+"o "+color[0]+" Linear regression done\n")

# #
# alpha_p = Alpha( k_p , X[int(n/2)-180:int(n/2)],Y[int(n/2)-180:int(n/2)] )
# alpha_g = Alpha( k_g , X[int(n/2)-180:int(n/2)],Y[int(n/2)-180:int(n/2)])
# u=[] ; v=[]
# for j in range ( n ) :
#     S = 0
#     for i in range ( 180 ) :
#         S = S + alpha_p [i] * k_p( X[i] , X[j] )
#     # u . append (S)
#     S = 0
#     for i in range ( 180 ) :
#         S = S + alpha_g [i] * k_g( X[i] , X[j] )
#     v . append (S)
# print(color[4]+"o "+color[0]+" Gaussian regression 1 done\n")

# #
# gaussian_process = GaussianProcessRegressor(kernel=kernel2)
# gaussian_process.fit(X[int(n/2)-180:int(n/2)],Y[int(n/2)-180:int(n/2)])
# print(color[4]+"o "+color[0]+" Gaussian regression 2 done\n")


# #
# # param_distributions = {
# #     "kernel__k1__length_scale": loguniform(1e-2, 1e2),  # length_scale du premier sous-noyau (ExpSineSquared)
# #     "kernel__k1__periodicity": loguniform(1e-1, 1e1),  # periodicity du premier sous-noyau (ExpSineSquared)
# #     "kernel__k2__noise_level": loguniform(1e-2, 1e2)  # length_scale du deuxième sous-noyau (RBF)
# # }
# # k_tuned = RandomizedSearchCV(gaussian_process,param_distributions=param_distributions,n_iter=500,random_state=0,)
# # k_tuned.fit(X[int(n/2)-180:int(n/2)],Y[int(n/2)-180:int(n/2)])
# # print(color[4]+"o "+color[0]+" Gaussian tuned regression done\n")
# # krr = KernelRidge(kernel=k_p, alpha=1.0)
# # krr.fit(X,Y)


# Y_pred = model.predict(X)
# # Y_pred_k = k_tuned.predict(X)
# # print(Y_pred_k)

# #- Plot
# dates_Y=(np.array(dates))[valid_indices]
# plt.plot(dates_Y, Y, color='grey', label='Actual Y',linewidth=0.5)

# plt.plot(dates_Y[:int(n/2)], Y_pred[:int(n/2)], color='red', label='CAPM')
# plt.plot(dates_Y[int(n/2):], Y_pred[int(n/2):], color='red', linestyle="dotted")

# plt.plot(dates_Y[:int(n/2)], v[:int(n/2)], color='green', label='Exp kernel Ret')
# plt.plot(dates_Y[int(n/2):], v[int(n/2):], color='green', label='Exp kernel Ret', linestyle="dotted")

# # plt.plot(dates_Y[:int(n/2)], Y_pred_k[:int(n/2)], color='y', label='Exp kernel tuned Y')
# # plt.plot(dates_Y[int(n/2):], Y_pred_k[int(n/2):], color='y', label='Exp kernel Y tuned', linestyle="dotted")


# plt.title(f"Yurn of {FIRMname[firm]} [firm={firm}]")
# # plt.legend()
# plt.grid(True)
# plt.xlim(dates_Y[int(n/2)-50] , dates_Y[int(n/2)+50] )
# plt.ylim(-0.10 , 0.15)
# plt.savefig(f"CAPM_forcast_firm${firm}$.png")

# ############################################################
# #                   Prints & Plots
# ############################################################

# # print(color[0]+"CRSP:\n   >size: "+color[2]+"{}x{}".format(np.size(CRSP[:,0].A1),np.size(CRSP[0,:].A1))+color[0])
# # print(color[0]+f"   >Number of firms: "+color[2]+f"{nb_firm}")
# # print(color[0]+"   >Typical examples of rows:"+color[2])
# # print(CRSP[:5,:])
# # print(color[0])
# plt.close()
# plt.plot(dates , SP500 , c = "r" , label = "S&P 500 Yurn")
# plt.savefig("S&P500.png")

# # time=[]
# # for i in range(len(CRSP[:,0])):
# #     for j in range(i,len(DGS10[:,0])):
# #         if CRSP[i,0]==DGS10[j,0]:
# #             time.append(CRSP[i,0])
# # time = enumerate(time)
# # for index, date in enumerate(time):
# #     # Imprimer une partie de la liste énumérée
# #     if index < 50:
# #         print(f"[{index}]: {date}")



import pandas as pd

# Chargement du fichier CSV en spécifiant les types de données
file_path = 'data/test5183.csv'  # Remplacez par le chemin de votre fichier
output_file_path = 'data/test5183m.csv'  # Remplacez par le chemin de sortie souhaité


df = pd.read_csv(file_path, dtype={
    'NAMEENDT': str, 'SHRCLS': str, 'TSYMBOL': str, 'DCLRDT': str, 'DLPDT': str,
    'NEXTDT': str, 'PAYDT': str, 'RCRDDT': str, 'SHRFLG': str
}, na_values=[''], low_memory=False)



# Convertir la colonne 'date' en datetime
df['date'] = pd.to_datetime(df['date'])

# Extraire l'année et le mois
df['year_month'] = df['date'].dt.to_period('M')

# Sélectionner les colonnes numériques pour calculer les moyennes
numeric_cols = df.select_dtypes(include=['number']).columns

# Calculer les moyennes mensuelles pour chaque entreprise (PERMNO) en ignorant les NaN
monthly_avg_numeric = df.groupby(['PERMNO', 'year_month'], as_index=False)[numeric_cols].mean()

# Sélectionner les colonnes non numériques pour la première occurrence du mois
non_numeric_cols = ['NAMEENDT', 'NCUSIP', 'TICKER', 'COMNAM', 'SHRCLS', 'TSYMBOL', 
                    'NAICS', 'PRIMEXCH', 'TRDSTAT', 'SECSTAT', 'CUSIP', 'DCLRDT', 
                    'DLPDT', 'NEXTDT', 'PAYDT', 'RCRDDT']
monthly_first_non_numeric = df.groupby(['PERMNO', 'year_month'], as_index=False)[non_numeric_cols].first()

# Fusionner les résultats en utilisant 'PERMNO' et 'year_month' comme clés de fusion
monthly_avg = pd.merge(monthly_avg_numeric, monthly_first_non_numeric, on=['PERMNO', 'year_month'])

# Réorganiser les colonnes dans l'ordre spécifié
desired_order = ['PERMNO', 'year_month', 'NAMEENDT', 'SHRCD', 'EXCHCD', 'SICCD', 'NCUSIP', 'TICKER', 'COMNAM', 'SHRCLS',
                 'TSYMBOL', 'NAICS', 'PRIMEXCH', 'TRDSTAT', 'SECSTAT', 'PERMCO', 'ISSUNO', 'HEXCD', 'HSICCD',
                 'CUSIP', 'DCLRDT', 'DLAMT', 'DLPDT', 'DLSTCD', 'NEXTDT', 'PAYDT', 'RCRDDT', 'SHRFLG', 'HSICMG',
                 'HSICIG', 'DISTCD', 'DIVAMT', 'FACPR', 'FACSHR', 'ACPERM', 'ACCOMP', 'SHRENDDT', 'NWPERM', 'DLRETX',
                 'DLPRC', 'DLRET', 'TRTSCD', 'NMSIND', 'MMCNT', 'NSDINX', 'BIDLO', 'ASKHI', 'PRC', 'VOL', 'RET',
                 'BID', 'ASK', 'SHROUT', 'CFACPR', 'CFACSHR', 'OPENPRC', 'NUMTRD', 'RETX', 'vwretd', 'vwretx', 'ewretd',
                 'ewretx', 'sprtrn']
monthly_avg = monthly_avg.reindex(columns=desired_order)

# Reformater la colonne 'year_month' pour l'affichage
monthly_avg['date'] = pd.to_datetime(monthly_avg['year_month'].dt.to_timestamp())

# Enregistrer le résultat dans un nouveau fichier CSV


monthly_avg.to_csv(output_file_path, index=False)

print(f"Moyennes mensuelles enregistrées dans {output_file_path}")
