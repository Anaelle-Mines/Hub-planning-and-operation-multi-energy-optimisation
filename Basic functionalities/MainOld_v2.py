# region Importation of modules
import os

if os.path.basename(os.getcwd()) == "Basic functionalities":
    os.chdir('..')  ## to work at project root  like in any IDE

import numpy as np
import pandas as pd
import csv
import time

import datetime
import copy
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import sys
from joblib import Parallel, delayed
import multiprocessing
from itertools import product

from dynprogstorage.wrappers import GenCostFunctionFromMarketPrices
from dynprogstorage.wrappers import GenCostFunctionFromMarketPrices_dict

from Functions.f_operationModels import *
from Functions.f_optimization import *
from Functions.f_graphicalTools import *
#from Functions.f_operationModelsAna import *
#from Functions.f_operationModelsAna_copy import *
from Functions.f_operationModelsAna_test import *
from Functions.f_graphicalToolsAna import *
from Functions.f_AnalyseToolsAna import *

# endregion

# region Solver and data location definition
InputFolder = 'Data/Input/'
OutputFolder = 'Data/Output/'
if sys.platform != 'win32':
    myhost = os.uname()[1]
else:
    myhost = ""
if (myhost == "jupyter-sop"):
    ## for https://jupyter-sop.mines-paristech.fr/ users, you need to
    #  (1) run the following to loanch the license server
    if (os.system(
            "/opt/mosek/9.2/tools/platform/linux64x86/bin/lmgrd -c /opt/mosek/9.2/tools/platform/linux64x86/bin/mosek.lic -l lmgrd.log") == 0):
        os.system("/opt/mosek/9.2/tools/platform/linux64x86/bin/lmutil lmstat -c 27007@127.0.0.1 -a")
    #  (2) definition of license
    os.environ["MOSEKLM_LICENSE_FILE"] = '@jupyter-sop'

BaseSolverPath = '/Users/robin.girard/Documents/Code/Packages/solvers/ampl_macosx64'  ### change this to the folder with knitro ampl ...
## in order to obtain more solver see see https://ampl.com/products/solvers/open-source/
## for eduction this site provides also several professional solvers, that are more efficient than e.g. cbc
sys.path.append(BaseSolverPath)
solvers = ['gurobi', 'knitro', 'cbc']  # try 'glpk', 'cplex'
solverpath = {}
for solver in solvers: solverpath[solver] = BaseSolverPath + '/' + solver
solver = 'mosek'  ## no need for solverpath with mosek.


# endregion

#region Ramp Single area : loading parameters case with ramp constraints
Zones="PACA"
year=2013
other='_WithH2'
PrixRes='fixe'
Selected_TECHNOLOGIES=['OldNuke','Solar','WindOnShore','pac','electrolysis','Solar_PPA','WindOnShore_PPA']
areaConsumption, availabilityFactor, TechParameters, conversionFactor, ResParameters, Calendrier,StorageParameters = loadingParameters(Selected_TECHNOLOGIES,'Data/Input/',Zones,year,other,PrixRes)
#endregion

#region Single area with storage : solving and loading results
start=time.time()
model = My_GetElectricSystemModel_PlaningSingleNode_MultiRessources_WithStorage(areaConsumption, availabilityFactor, TechParameters, ResParameters,conversionFactor,StorageParameters,Calendrier)
opt = SolverFactory(solver)
opt.solve(model)
end=time.time()
Complexity=end-start
print('temps de calcul = ',Complexity, 's')
Variables = getVariables_panda(model)  #### obtain optimized variables in panda form
Constraints = getConstraintsDual_panda(model)

#endregion

#region Visualisation des résultats

# Capacity Costs & Importation Costs
PlotCapacityAndImportationCosts(Variables)

# Carbon Costs
#PlotCarbonCosts(Variables)

# Installed Capacity and Total energy production
PlotCapacityAndEnegyProduction(Variables)

# Energy variation
PlotRessourceVariation(Variables)

# Electricity production
PlotElectricityProduction(Variables)

# H2 production
PlotH2Production(Variables)

#endregion

#region Analyse

power_use=Variables['power'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='power')
Prix_elec=(Variables['capacityCosts'].set_index('TECHNOLOGIES').loc['Solar']['capacityCosts']+Variables['capacityCosts'].set_index('TECHNOLOGIES').loc['WindOnShore']['capacityCosts']+Variables['importCosts'].set_index('RESSOURCES').loc['electricity']['importCosts'])/(sum(power_use['WindOnShore'])+sum(power_use['Solar'])) #€/MWh
#lagrange=Constraints['energyCtr'].set_index('TIMESTAMP')
#Prix_elec_lagrange=sum(lagrange['energyCtr']*((power_use['WindOnShore']+power_use['Solar'])/(sum(power_use['WindOnShore'])+sum(power_use['Solar']))))
capa_cout=sum(Variables['capacityCosts']['capacityCosts'])
import_cout=sum(Variables['importCosts']['importCosts'])
turpe_cout=Variables['turpeCosts'].dropna().sum()['turpeCosts']
Prix_H2_MW=(capa_cout+import_cout+turpe_cout)/sum(power_use['electrolysis'])#€/MWh
Prix_H2_kg=Prix_H2_MW/1000*33.33 #€/kg

Capa_H2=Variables['capacity'].set_index('TECHNOLOGIES').loc['electrolysis']['capacity']
nbh_H2=sum(power_use['electrolysis'])/Capa_H2
Prod_H2=sum(power_use['electrolysis'])/1000000

import_elec=Variables['importation'].groupby('RESSOURCES').agg({'importation':'sum'}).loc['electricity']['importation']/1000000 #TWh
EnR_elec=(Variables['power'].groupby('TECHNOLOGIES').agg({'power':'sum'}).loc['Solar']['power']+Variables['power'].groupby('TECHNOLOGIES').agg({'power':'sum'}).loc['WindOnShore']['power'])/1000000  #TWh
total_elec=import_elec+EnR_elec
elec_H2=sum(power_use['electrolysis'])*1.43/1000000
elec_inject=Variables['injection'].groupby('RESSOURCES').agg({'injection':'sum'}).loc['electricity']['injection']/1000000 #TWh
local_elec_H2=EnR_elec-elec_inject
reseau_elec_H2=import_elec

print('Prix total hydrogène = ',Prix_H2_MW,' €/MWh soit ',Prix_H2_kg,' €/kg')
print('Prix électricité = ',Prix_elec,' €/MWh')
print('Capacité électrolyseur = ',Capa_H2/1000,' GW')
print('Nombre heure de fonctionnement électrolyseur = ',nbh_H2,' h soit facteur de charge =',nbh_H2/8760*100,'%')
print('Prod hdyrogène =',Prod_H2,' TWh')
print('Electricité locale EnR = ',local_elec_H2,' TWh soit ',local_elec_H2/(elec_H2)*100,'%')
print('Electricité réseau = ',reseau_elec_H2,' TWh')
print('Electricité réinjectée = ',elec_inject,' TWh')

# Coûts système comparaison :

df1=pd.read_csv('Coûts_Isolé.csv',sep=',',decimal='.',skiprows=0).rename(columns={'Coût':'Isolé'}).set_index('Unnamed: 0')/(1.94*1000000000)*33.3
df2=pd.read_csv('Coûts_Local.csv',sep=',',decimal='.',skiprows=0).rename(columns={'Coût':'Local'}).set_index('Unnamed: 0')/(1.94*1000000000)*33.3
df3=pd.read_csv('Coûts_PPA.csv',sep=',',decimal='.',skiprows=0).rename(columns={'Coût':'PPA'}).set_index('Unnamed: 0')/(1.94*1000000000)*33.3
df4=pd.DataFrame({'Isolé':0.34*89860*1000/(1.94*1000000000)*33.3,'Local':0.25*89860*1000/(1.94*1000000000)*33.3,'PPA':0.27*89860*1000/(1.94*1000000000)*33.3},index=['capacité H2'])
df=pd.concat([df1,df2,df3],axis=1)
df=pd.concat([df,df4],axis=0)
df.loc['capacités']=df.loc['capacités']-df.loc['capacité H2']

fig, ax = plt.subplots()

# Create light blue Bars
plt.bar(df.columns, list(df.loc['capacité H2']), color='#88B3F8', label="Capacité H2")
# Create dark blue Bars
plt.bar(df.columns, list(df.loc['capacités']), bottom=list(df.loc['capacité H2']), color='#165BA0', label="Capacités EnR")
# Create orange Bars
ax.bar(list(df.columns), list(df.loc['importations']), bottom=[i + j for i, j in zip(list(df.loc['capacité H2']), list(df.loc['capacités']))], color='#F46E23', label="Réseau")
# Create green Bars
plt.bar(df.columns, list(df.loc['TURPE']), bottom=[i + j + k for i, j, k in zip(list(df.loc['capacité H2']), list(df.loc['capacités']),list(df.loc['importations']))], color='#14A01B', label="TURPE")
# Create yellow bar
plt.bar(df.columns, list(df.loc['Stockage']), bottom=[i + j + k + l for i, j, k, l in zip(list(df.loc['capacité H2']), list(df.loc['capacités']),list(df.loc['importations']),list(df.loc['TURPE']))], color='#F0C300', label="Stockage")
ax.set_ylabel('Coûts (€/kgH2)')
ax.set_title("Répartition des coûts système")
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('Répartition coûts système')
plt.show()

#endregion

#region Stockage

#Stockage H2

Storage_H2=Variables['StockNiv_var'].set_index(['TIMESTAMP','RESSOURCES']).loc[(slice(None),'hydrogen'),'StockNiv_var']
TIMESTAMP_d = pd.date_range(start=str(year) + "-01-01 00:00:00", end=str(year) + "-12-31 23:00:00", freq="1H")
Storage_H2.index = TIMESTAMP_d
fig=px.line(Storage_H2,title="Variation de l'énergie dans le stockage")
fig = fig.update_layout(xaxis_title="temps", yaxis_title="MWh")
plotly.offline.plot(fig, filename='stockage_H2.html')

print('Etat initial du stockage H2 = ',Storage_H2.loc['2013-01-01 00:00:00']/1000,' GWh soit ',Storage_H2.loc['2013-01-01 00:00:00']/33.33,' tonnes')
print('Taille de stockage H2 nécessaire (Cmax) = ',Storage_H2.max()/1000,' GWh soit',Storage_H2.max()/33.33,' tonne')
print('Valeur du débit max H2 nécessaire (Pmax) = ',Variables['Pmax_var'].set_index('RESSOURCES').loc['hydrogen','Pmax_var']/1000,' GW soit',Variables['Pmax_var'].set_index('RESSOURCES').loc['hydrogen','Pmax_var']/33.33,' tonne/h')

#Stockage elec

Storage_elec = Variables['StockNiv_var'].set_index(['TIMESTAMP','RESSOURCES']).loc[(slice(None),'electricity'),'StockNiv_var']
TIMESTAMP_d = pd.date_range(start=str(year) + "-01-01 00:00:00", end=str(year) + "-12-31 23:00:00", freq="1H")
Storage_elec.index = TIMESTAMP_d
fig=px.line(Storage_elec,title="Variation de l'énergie dans le stockage")
fig = fig.update_layout(xaxis_title="temps", yaxis_title="MWh")
plotly.offline.plot(fig, filename='stockage_elec.html')

print('Etat initial du stockage élec = ',Storage_elec.loc['2013-01-01 00:00:00']/1000,' GWh')
print('Taille de stockage élec nécessaire (Cmax) = ',Storage_elec.max()/1000,' GWh')
print('Valeur du débit max élec nécessaire (Pmax) = ',Variables['Pmax_var'].set_index('RESSOURCES').loc['electricity','Pmax_var']/1000,' GW')

#endregion

Cout_import=sum(Variables['importCosts_var']['importCosts_var'])
Cout_capa=sum(Variables['capacityCosts_var']['capacityCosts_var'])
Cout_Turpe=sum(Variables['turpeCosts_var'][Variables['turpeCosts_var'].RESSOURCES=='electricity']['turpeCosts_var'])
Cout_stock=sum(Variables['storageCosts_var']['storageCosts_var'])
df=pd.DataFrame([Cout_import,Cout_capa,Cout_Turpe,Cout_stock],index=['importations','capacités','TURPE','Stockage'],columns=['Coût'])
df.to_csv('Coûts_Isolé.csv')
Variables['Cmax_var'].to_csv('Cmax.csv')
