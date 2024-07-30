import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import calendar
c=['a'',''a'',''b'',''c'',''c'',''c']
# print(c.index('a'))
# print(c.index('b'))
# print(c.index('c'))
# print(c.index('c'))
# print(c.index('a'))
y='2000-12-04'
# print(y[:5])
# print(len([128',' 121',' 21',' 28',' 33',' 22',' 11',' 27',' 14',' 36',' 35',' 31',' 37',' 70',' 67',' 74',' 93',' 104',' 101',' 112',' 109',' 108',' 118',' 116',' 99',' 110',' 119',' 134',' 25',' 132',' 139',' 105',' 133',' 15',' 144',' 147',' 140',' 151',' 146',' 153',' 131',' 129',' 149',' 150',' 154',' 51',' 124',' 73',' 71',' 8',' 10',' 30',' 125',' 38',' 69',' 152',' 111',' 88',' 55',' 2',' 100',' 138',' 19',' 90',' 77',' 122]))
# print(len(['ENERGY WEST INC'',' 'SOUTH ALABAMA BANCORPORATION INC'',' 'A T C GROUP SERVICES INC'',' 'DENSE PACIFIC MICROSYSTEMS INC'',' 'FIGGIE INTERNATIONAL INC DEL'',' 'I F R SYSTEMS INC'',' 'A E P INDUSTRIES INC'',' 'DALLAS GOLD & SILVER EXCHANGE IN'',' 'FRANKLIN BANK NATL ASSOC'',' 'XIOX CORP'',' 'ROCKY MOUNTAIN CHOCOLATE FAC INC'',' 'CENTURY COMMUNICATIONS CORP'',' 'HANGER ORTHOPEDIC GROUP INC'',' 'FREE STATE CONSOL GOLD CO LTD'',' 'ADAMS EXPRESS CO'',' 'FRANKLIN TELECOMMUNICATIONS CORP'',' 'V W R SCIENTIFIC PRODUCTS CORP'',' 'U I C I'',' '4 KIDS ENTERTAINMENT INC'',' 'AMERICAN CLAIMS EVALUATION INC'',' 'HALSEY DRUG INC'',' '20TH CENTURY INDUSTRIES'',' 'PIPER JAFFRAY COS INC'',' 'MEDFORD SAVINGS BANK'',' 'ACKERLEY COMMUNICATIONS INC'',' 'ONE VALLEY BANCORP INC'',' 'PARIS BUSINESS FORMS INC'',' 'LA MAN CORP'',' 'NEWNAN HOLDINGS INC'',' 'ALLEGHENY POWER SYSTEMS INC'',' 'T ROWE PRICE ASSOC INC'',' 'SEALRIGHT COMPANY INC'',' 'ONCOGENE SCIENCE INC'',' 'ALLIED SIGNAL INC'',' 'E M C CORP MA'',' 'TRANS LEASING INTL INC'',' 'ELECTRIC & GAS TECHNOLOGY INC'',' 'NORTH SIDE SAVINGS BANK BRONX'',' 'SANDATA INC'',' 'X RITE INC'',' 'FAMILY STEAK HOUSES FL INC'',' 'BARRYS JEWELERS INC'',' 'J M C GROUP INC'',' 'GOLDEN BOOKS FAMILY ENT INC'',' 'R C S B FINANCIAL INC'',' 'AKORN INC'',' 'MERRILL CORP'',' 'C TEC CORP'',' 'ATLANTIC POWER CORP'',' 'LAWRENCE SAVINGS BANK NEW'',' 'WESTCORP INC'',' 'PLASTI LINE INC'',' 'WATER JEL TECHNOLOGIES INC'',' 'A F P IMAGING CORP'',' 'CAMELOT INFORMATION SYSTEMS INC'',' 'AMERICAN BRANDS INC'',' 'ANDOVER BANCORP INC DEL'',' 'COMMUNITY BANKSHARES INC NH'',' 'ELMIRA SAVINGS BANK FSB NY'',' 'SEATTLE FILMWORKS INC'',' 'BALDWIN & LYONS INC'',' 'POLLUTION RESEARCH & CTRL CORP'',' 'GANTOS INC NEW'',' 'INDEPENDENT BANK CORP MA'',' 'NEW SKY COMMMUNICATIONS INC'',' 'MICROLOG CORP'',' 'DIGITAL SOLUTIONS INC'',' 'T CELL SCIENCES INC'',' 'SENETEK PLC'',' 'WHEELABRATOR TECHNOLOGIES INC NW'',' 'HAVERTY FURNITURE COS INC'',' 'NEW HAMPSHIRE THRIFT BNCSHRS INC'',' 'TOMPKINS COUNTY TRUSTCO INC'',' 'CENTER BANKS INC'',' 'BALCHEM CORP'',' 'BAKER J INC'',' 'ABINGTON SAVINGS BANK'',' 'ADAPTEC INC'',' 'BUTLER INTERNATIONAL INC NEW'',' 'ADVANCED MAGNETICS INC'',' 'T C F FINANCIAL CORP'',' 'A T C COMMUNICATIONS INC'',' 'TRENWICK GROUP INC'',' 'G H S INC'',' 'A T & T CORP'',' 'TRANSMEDIA NETWORK INC'',' 'CAPSTONE PHARMACY SERVICES INC'',' 'JONES MEDICAL INDUSTRIES INC'',' 'A S A INTERNATIONAL LTD'',' 'K L L M TRANSPORT SVCS INC'',' 'F R P PROPERTIES INC'',' 'INAMED CORP'',' 'TUESDAY MORNING CORP'',' 'TYCO TOYS INC'',' 'CANYON RESOURCES CORP'',' 'SOMATIX THERAPY CORP'',' 'GENZYME CORP'',' 'PRIME CAPITAL CORP'',' 'ANALYSIS & TECHNOLOGY INC'',' 'INTERLEAF INC'',' 'SOUND ADVICE INC'',' 'ROYCE LABORATORIES INC'',' 'SOMERSET GROUP INC'',' 'HEALTHPLEX INC'',' 'AMCORE FINANCIAL INC'',' 'BAY VIEW CAPITAL CORP'',' 'EAGLE BANCSHARES INC'',' 'IROQUOIS BANCORP INC'',' 'MASSBANK CORP'',' 'NAVIGATORS GROUP INC'',' 'NEWMIL BANCORP INC'',' 'TRANS FINANCIAL INC'',' 'ROYCE GLOBAL TRUST INC'',' 'INFINITY BROADCASTING CORP']))

# T=['gvkey','datadate','fyear','indfmt','consol','popsrc','datafmt','tic','cusip','conm','acctchg','acctstd','acqmeth','adrr','ajex','ajp','bspr','compst','curcd','curncd','currtr','curuscn','final','fyr','ismod','ltcm','ogm','pddur','scf','src','stalt','udpl','upd','apdedate','fdate','pdate','acchg','acco','accrt','acdo','aco','acodo','acominc','acox','acoxar','acqao','acqcshi','acqgdwl','acqic','acqintan','acqinvt','acqlntal','acqniintc','acqppe','acqsc','act','adpac','aedi','afudcc','afudci','aldo','am','amc','amdc','amgw','ano','ao','aocidergl','aociother','aocipen','aocisecgl','aodo','aol2','aoloch','aox','ap','apalch','apb','apc','apofs','aqa','aqc','aqd','aqeps','aqi','aqp','aqpl1','aqs','arb','arc','arce','arced','arceeps','artfs','at','aul3','autxr','balr','banlr','bast','bastr','batr','bcef','bclr','bcltbl','bcnlr','bcrbl','bct','bctbl','bctr','billexce','bkvlps','bltbl','ca','capr1','capr2','capr3','caps','capsft','capx','capxv','cb','cbi','cdpac','cdvc','ceiexbill','ceq','ceql','ceqt','cfbd','cfere','cfo','cfpdo','cga','cgri','cgti','cgui','ch','che','chech','chs','ci','cibegni','cicurr','cidergl','cimii','ciother','cipen','cisecgl','citotal','cld2','cld3','cld4','cld5','clfc','clfx','clg','clis','cll','cllc','clo','clrll','clt','cmp','cnltbl','cogs','cpcbl','cpdoi','cpnli','cppbl','cprei','crv','crvnli','cshfd','cshi','csho','cshpri','cshr','cshrc','cshrp','cshrso','cshrt','cshrw','cstk','cstkcv','cstke','dbi','dc','dclo','dcom','dcpstk','dcs','dcvsr','dcvsub','dcvt','dd','dd1','dd2','dd3','dd4','dd5','depc','derac','deralt','derhedgl','derlc','derllt','dfpac','dfs','dfxa','diladj','dilavx','dlc','dlcch','dltis','dlto','dltp','dltr','dltsub','dltt','dm','dn','do','donr','dp','dpacb','dpacc','dpacli','dpacls','dpacme','dpacnr','dpaco','dpacre','dpact','dpc','dpdc','dpltb','dpret','dpsc','dpstb','dptb','dptc','dptic','dpvieb','dpvio','dpvir','drc','drci','drlt','ds','dt','dtea','dted','dteeps','dtep','dudd','dv','dvc','dvdnp','dvintf','dvp','dvpa','dvpd','dvpdp','dvpibb','dvrpiv','dvrre','dvsco','dvt','dxd2','dxd3','dxd4','dxd5','ea','ebit','ebitda','eiea','emol','emp','epsfi','epsfx','epspi','epspx','esopct','esopdlt','esopnr','esopr','esopt','esub','esubc','excadj','exre','fatb','fatc','fatd','fate','fatl','fatn','fato','fatp','fca','fdfr','fea','fel','ffo','ffs','fiao','finaco','finao','fincf','finch','findlc','findlt','finivst','finlco','finlto','finnp','finrecc','finreclt','finrev','finxint','finxopr','fopo','fopox','fopt','fsrco','fsrct','fuseo','fuset','gbbl','gdwl','gdwlam','gdwlia','gdwlid','gdwlieps','gdwlip','geqrv','gla','glcea','glced','glceeps','glcep','gld','gleps','gliv','glp','govgr','govtown','gp','gphbl','gplbl','gpobl','gprbl','gptbl','gwo','hedgegl','iaeq','iaeqci','iaeqmi','iafici','iafxi','iafxmi','iali','ialoi','ialti','iamli','iaoi','iapli','iarei','iasci','iasmi','iassi','iasti','iatci','iati','iatmi','iaui','ib','ibadj','ibbl','ibc','ibcom','ibki','ibmii','icapt','idiis','idilb','idilc','idis','idist','idit','idits','iire','initb','intan','intano','intc','intpn','invch','invfg','invo','invofs','invreh','invrei','invres','invrm','invt','invwip','iobd','ioi','iore','ip','ipabl','ipc','iphbl','iplbl','ipobl','iptbl','ipti','ipv','irei','irent','irii','irli','irnli','irsi','iseq','iseqc','iseqm','isfi','isfxc','isfxm','isgr','isgt','isgu','islg','islgc','islgm','islt','isng','isngc','isngm','isotc','isoth','isotm','issc','issm','issu','ist','istc','istm','isut','itcb','itcc','itci','ivaco','ivaeq','ivao','ivch','ivgod','ivi','ivncf','ivpt','ivst','ivstch','lcabg','lcacl','lcacr','lcag','lcal','lcalt','lcam','lcao','lcast','lcat','lco','lcox','lcoxar','lcoxdr','lct','lcuacu','li','lif','lifr','lifrp','lloml','lloo','llot','llrci','llrcr','llwoci','llwocr','lno','lo','lol2','loxdr','lqpl1','lrv','ls','lse','lst','lt','lul3','mib','mibn','mibt','mii','mrc1','mrc2','mrc3','mrc4','mrc5','mrct','mrcta','msa','msvrv','mtl','nat','nco','nfsr','ni','niadj','nieci','niint','niintpfc','niintpfp','niit','nim','nio','nipfc','nipfp','nit','nits','nopi','nopio','np','npanl','npaore','nparl','npat','nrtxt','nrtxtd','nrtxteps','oancf','ob','oiadp','oibdp','opeps','opili','opincar','opini','opioi','opiri','opiti','oprepsx','optca','optdr','optex','optexd','optfvgr','optgr','optlife','optosby','optosey','optprcby','optprcca','optprcex','optprcey','optprcgr','optprcwa','optrfr','optvol','palr','panlr','patr','pcl','pclr','pcnlr','pctr','pdvc','pi','pidom','pifo','pll','pltbl','pnca','pncad','pncaeps','pncia','pncid','pncieps','pncip','pncwia','pncwid','pncwieps','pncwip','pnlbl','pnli','pnrsho','pobl','ppcbl','ppegt','ppenb','ppenc','ppenli','ppenls','ppenme','ppennr','ppeno','ppent','ppevbb','ppeveb','ppevo','ppevr','pppabl','ppphbl','pppobl','ppptbl','prc','prca','prcad','prcaeps','prebl','pri','prodv','prsho','prstkc','prstkcc','prstkpc','prvt','pstk','pstkc','pstkl','pstkn','pstkr','pstkrv','ptbl','ptran','pvcl','pvo','pvon','pvpl','pvt','pwoi','radp','ragr','rari','rati','rca','rcd','rceps','rcl','rcp','rdip','rdipa','rdipd','rdipeps','rdp','re','rea','reajo','recch','recco','recd','rect','recta','rectr','recub','ret','reuna','reunr','revt','ris','rll','rlo','rlp','rlri','rlt','rmum','rpag','rra','rrd','rreps','rrp','rstche','rstchelt','rvbci','rvbpi','rvbti','rvdo','rvdt','rveqt','rvlrv','rvno','rvnt','rvri','rvsi','rvti','rvtxr','rvupi','rvutx','saa','sal','sale','salepfc','salepfp','sbdc','sc','sco','scstkc','secu','seq','seqo','seta','setd','seteps','setp','siv','spce','spced','spceeps','spi','spid','spieps','spioa','spiop','sppe','sppiv','spstkc','sret','srt','ssnp','sstk','stbo','stio','stkco','stkcpa','tdc','tdscd','tdsce','tdsg','tdslg','tdsmm','tdsng','tdso','tdss','tdst','teq','tf','tfva','tfvce','tfvl','tie','tii','tlcf','transa','tsa','tsafc','tso','tstk','tstkc','tstkme','tstkn','tstkp','txach','txbco','txbcof','txc','txdb','txdba','txdbca','txdbcl','txdc','txdfed','txdfo','txdi','txditc','txds','txeqa','txeqii','txfed','txfo','txndb','txndba','txndbl','txndbr','txo','txp','txpd','txr','txs','txt','txtubadjust','txtubbegin','txtubend','txtubmax','txtubmin','txtubposdec','txtubposinc','txtubpospdec','txtubpospinc','txtubsettle','txtubsoflimit','txtubtxtr','txtubxintbs','txtubxintis','txva','txw','uaoloch','uaox','uapt','ucaps','uccons','uceq','ucustad','udcopres','udd','udfcc','udmb','udolt','udpco','udpfa','udvp','ufretsd','ugi','ui','uinvt','ulcm','ulco','uniami','unl','unnp','unnpl','unopinc','unwcc','uois','uopi','uopres','updvp','upmcstk','upmpf','upmpfs','upmsubp','upstk','upstkc','upstksf','urect','urectr','urevub','uspi','ustdnc','usubdvp','usubpstk','utfdoc','utfosc','utme','utxfed','uwkcapc','uxinst','uxintd','vpac','vpo','wcap','wcapc','wcapch','wda','wdd','wdeps','wdp','xacc','xad','xago','xagt','xcom','xcomi','xdepl','xdp','xdvre','xeqo','xi','xido','xidoc','xindb','xindc','xins','xinst','xint','xintd','xintopt','xivi','xivre','xlr','xnbi','xnf','xnins','xnitb','xobd','xoi','xopr','xoprar','xoptd','xopteps','xore','xpp','xpr','xrd','xrdp','xrent','xs','xsga','xstf','xstfo','xstfws','xt','xuw','xuwli','xuwnli','xuwoi','xuwrei','xuwti','exchg','cik','costat','fic','naicsh','sich','cshtr_c','dvpsp_c','dvpsx_c','prcc_c','prch_c','prcl_c','adjex_c','cshtr_f','dvpsp_f','dvpsx_f','mkvalt','prcc_f','prch_f','prcl_f','adjex_f','rank','au','auop','auopic','ceoso','cfoso','busdesc','city','conml','county','dlrsn','ein','fyrc','ggroup','gind','gsector','gsubind','idbflag','incorp','loc','naics','phone','prican','prirow','priusa','sic','spcindcd','spcseccd','spcsrc','state','stko','weburl','dldte','ipodate']

# timeline=[]
# timelineflt=[]
# current_date = datetime.strptime('1997-01', '%Y-%m')
# t=1997+1/12
# end_date = datetime.strptime('2024-01', '%Y-%m')
# while current_date <= end_date:
#     timelineflt.append(t)
#     timeline.append(current_date.strftime('%Y-%m'))  # Convertir la date en string avant de l'ajouter
    
#     days_in_month = calendar.monthrange(current_date.year, current_date.month)[1]
#     current_date = current_date.replace(day=days_in_month) + timedelta(days=1)
#     next_month = current_date.replace(day=28) + timedelta(days=4)
#     current_date = next_month - timedelta(days=next_month.day)
# N=len(timeline)
# print(timeline[N-5:])
# print(N%12)
# print(N/12)
import torch
import torch.nn.functional as F
from torch.optim import SGD

W= np.array([[None, 3.10, None, None, None],
[None, None ,None, None ,None],
 [None, -7.5 ,None ,2.04, None]])
INPUT = np.array([
    [0.3760599435648854, 0.9554993901442279, 1.0, 1.0, 1.0],
    [0.04808857665915871, 0.9323238780870847, 1.0, 1.0, 1.0],
    [-0.2923730477699084, 0.9722669659360974, 1.0, 1.0, 1.0],
    # ... (autres lignes) ...
    [None, None, None, None, None],
    [None, None, None, None, None],
    [None, None, None, None, None]
], dtype=object)
# print(W[0])
# print([abs(x) if x is not None else np.nan for x in W[:,1]])
alpha=[-2.48825680e-01,  4.12221928e+00, -3.02599012e-02,  1.50381941e-01,
  2.12813498e+00 , 3.14464521e+00 ,-9.89452556e-02 , 4.33635259e-02,
 -9.39073131e-02 ,-1.98110176e-01, -6.61288819e-02 ,-3.96943662e-02,
  8.20134796e-03, -9.55913660e-02 , 1.93333220e-02,  1.90945690e-01,
  1.02505543e-01 , 1.75961303e-02 ,-7.52663100e-02, -8.07873082e-02,
  3.38844865e-03]
print(sum([abs(x) for x in alpha]))
print(np.linalg.norm(alpha, 1))
print(np.linalg.norm(alpha, 2))


tau=8
X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)
Nskf = X.shape[0] - tau
X_tau = torch.tensor(X[:Nskf], dtype=torch.float32)
Y_tau = torch.tensor(Y[tau:Nskf+tau], dtype=torch.float32)

alpha = torch.tensor(alpha_init, dtype=torch.float32, requires_grad=True)
theta = torch.tensor(theta_init, dtype=torch.float32, requires_grad=True)



def k(x, y, alpha, theta):
    k = []
    norm_sq = torch.norm(x - y, p=2) ** 2
    norm = torch.norm(x - y, p=2)
    
    k.append(alpha[0] ** 2 * (x @ y.T + theta[0] ** 2))
    k.append(alpha[1] ** 2 * (theta[1] ** 2 * x @ y.T + theta[2] ** 2) ** abs(theta[3]))
    k.append(alpha[2] ** 2 * torch.exp(-norm_sq / (2 * theta[4] ** 2)))
    k.append(alpha[3] ** 2 * torch.exp(-norm / (2 * theta[5] ** 2)))
    k.append(alpha[4] ** 2 * torch.exp(-torch.sin(torch.pi * norm_sq / theta[6]) ** 2 / theta[7] ** 2) *
             torch.exp(-norm_sq / theta[8] ** 2))
    k.append(alpha[5] ** 2 * torch.exp(-torch.sin(torch.pi * norm_sq / theta[9]) ** 2 / theta[10] ** 2))
    k.append(alpha[6] ** 2 * torch.exp(-torch.sin(torch.pi * norm / theta[11]) ** 2 / theta[12] ** 2) *
             torch.exp(-norm / theta[13] ** 2))
    k.append(alpha[7] ** 2 * torch.exp(-torch.sin(torch.pi * norm / theta[14]) ** 2 / theta[15] ** 2))
    k.append(alpha[8] ** 2 * (norm_sq + theta[16] ** 2) ** 0.5)
    k.append(alpha[9] ** 2 * (theta[17] ** 2 + theta[18] ** 2 * norm_sq) ** -0.5)
    k.append(alpha[10] ** 2 * (theta[19] ** 2 + theta[20] ** 2 * norm) ** -0.5)
    k.append(alpha[11] ** 2 * (theta[21] ** 2 + norm) ** theta[22])
    k.append(alpha[12] ** 2 * (theta[23] ** 2 + norm_sq) ** theta[24])
    k.append(alpha[13] ** 2 * (1 + (norm / theta[25]) ** -1) ** -1)
    k.append(alpha[14] ** 2 * (1 + norm / theta[26] ** 2) ** -1)
    k.append(alpha[15] ** 2 * (1 - norm_sq / (norm_sq + theta[27] ** 2)))
    k.append(alpha[16] ** 2 * torch.relu(1 - norm_sq / theta[28] ** 2))
    k.append(alpha[17] ** 2 * torch.relu(1 - norm / theta[29] ** 2))
    k.append(alpha[18] ** 2 * torch.log(norm ** theta[30] + 1))
    k.append(alpha[19] ** 2 * torch.tanh(theta[31] * x @ y.T + theta[32]))
    k.append(alpha[21] ** 2 * torch.exp(torch.sin(torch.dot(x, y)) / theta[34] ** 2) / torch.sqrt(torch.tensor(2 * torch.pi)))
    acos_argument = norm / theta[33] ** 2
    if acos_argument < 1:  # Indicator function condition
        k.append(alpha[20] ** 2 * (torch.acos(-acos_argument) - acos_argument * torch.sqrt(1 - acos_argument ** 2)))
    return sum(k)
def k_alpha_theta(X, alpha, theta):
    K = torch.zeros(X.shape[0], X.shape[0])
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            K[i, j] = k(X[i], X[j], alpha, theta)
    return K

# Définir la fonction de perte
def loss_function(alpha, theta, X_c, X_b, Y_c, Y_b, lambda1, lambda2):
    K_alpha_theta_c = k_alpha_theta(X_c, alpha, theta)
    K_alpha_theta_b = k_alpha_theta(X_b, alpha, theta)
    
    term1 = Y_c.T @ torch.inverse(K_alpha_theta_c + lambda1 * torch.eye(X_c.shape[0])) @ Y_c
    term2 = Y_b.T @ torch.inverse(K_alpha_theta_b + lambda1 * torch.eye(X_b.shape[0])) @ Y_b
    
    rho = 1 - (term1 / term2) + lambda2 * torch.norm(alpha, p=1)
    return rho

# Optimisation
X_c = torch.rand(n_samples, n_features)  # données d'apprentissage
X_b = torch.rand(n_samples, n_features)  # données de validation
Y_c = torch.rand(n_samples, 1)  # étiquettes d'apprentissage
Y_b = torch.rand(n_samples, 1)  # étiquettes de validation

optimizer = SGD([alpha, theta], lr=0.01)

for epoch in range(100):  # nombre d'itérations d'entraînement
    optimizer.zero_grad()
    loss = loss_function(alpha, theta, X_c, X_b, Y_c, Y_b, lambda1, lambda2)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

print("Optimisation terminée.")