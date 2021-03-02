# region Importation of modules
import os

if os.path.basename(os.getcwd()) == "Basic functionalities":
    os.chdir('..')  ## to work at project root  like in any IDE

import numpy as np
import pandas as pd
import csv

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
from Functions.f_operationModelsAna import *
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
Zones="FR"
year=2013
Selected_TECHNOLOGIES=['OldNuke','Solar','WindOnShore','HydroReservoir','HydroRiver','TAC','CCG','pac','electrolysis']
areaConsumption,availabilityFactor, TechParameters, conversionFactor, ResParameters = loadingParameters()

#endregion

#region Ramp Single area : solving and loading results
model = My_GetElectricSystemModel_PlaningSingleNode_MultiRessources(areaConsumption,availabilityFactor,TechParameters,ResParameters,conversionFactor)
opt = SolverFactory(solver)
results=opt.solve(model)
Variables=getVariables_panda_indexed(model)
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

#### lagrange multipliers
Constraints= getConstraintsDual_panda(model)

#endregion

#region Chiffres principaux
Capa_totale = sum(Variables['capacity']['capacity'])/1000
power_use=Variables['power'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='power')
Prod_elec = sum(Variables['power']['power'])/1000000#-sum(power_use['electrolysis'])*1.43/1000000
#Prod_H2 = sum(power_use['electrolysis'])/1000000
Capa_cost = sum(Variables['capacityCosts']['capacityCosts'])/1000000
Import_cost = sum(Variables['importCosts']['importCosts'])/1000000
Prod_gaz=Variables['power'].loc[Variables['power']['TECHNOLOGIES']=='CCG'].sum(axis=0)['power']/1000000+Variables['power'].loc[Variables['power']['TECHNOLOGIES']=='TAC'].sum(axis=0)['power']/1000000
#Emission_costs = sum(Variables['carbonCosts']['carbonCosts'])/1000000

print('Capa_totale = ',Capa_totale, ' GW')
print('Prod_elec = ',Prod_elec,' TWh')
print('Prod_gaz = ',Prod_gaz,' TWh')
#print('Prod_H2 = ',Prod_H2,' TWh')
print('Capa_cost = ',Capa_cost,' M€')
print('Import_cost = ',Import_cost,' M€')
#print('Emission_costs = ',Emission_costs,' M€')

#endregion

#region Ordres de grandeur, ratio H2/gaz solving and loading results

#### Boucle pour la variation du CAPEX H2 + la variation du prix du gaz
# Prix du gaz borné à 100€/MWh, au delà => biogaz
# Prix du CAPEX H2 = -50% à +50% par rapport à la référence.
variation_CAPEX_H2=[-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5]
variation_prix_GazNat=[10,20,30,40,50,60,70,80,90,100]
alpha_df=Boucle_SensibiliteAlphaSimple(areaConsumption, availabilityFactor, TechParameters, ResParameters,conversionFactor,variation_CAPEX_H2,variation_prix_GazNat)
alpha_df.to_csv('Alpha_matrice_test.csv')  # enregistrement de la matrice dans un csv

### Tracé Heatmap alpha
variation_CAPEX_H2=['-50%', '-40%', '-30%', '-20%', '-10%', '0%', '10%', '20%', '30%', '40%','50%']
variation_prix_GazNat=[10,20,30,40,50,60,70,80,90,100]
alpha_df=pd.read_csv('Alpha_matrice_test.csv',sep=',',decimal='.',skiprows=0)[['PrixGaz','Capex','alpha']]# Récupération de la matrice à partir du csv
PlotHeatmapAlpha(alpha_df.rename(columns={'alpha':'value'}),variation_prix_GazNat,variation_CAPEX_H2)

### Tracé de Scatter alpha
variation_CAPEX_H2=['-50%', '-40%', '-30%', '-20%', '-10%', '0%', '10%', '20%', '30%', '40%','50%']
alpha_df=pd.read_csv('Alpha_matrice_test.csv',sep=',',decimal='.',skiprows=0)[['PrixGaz','Capex','alpha']] # Récupération de la matrice à partir du csv
PlotScatterAlphe(alpha_df.rename(columns={'alpha':'value'}),variation_CAPEX_H2)
#endregion

#region Sensibilité Alpha Parallel, solving and loading results

def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
    return pd.concat(retLst)

def expand_grid(dictionary):
   return pd.DataFrame([row for row in product(*dictionary.values())],
                       columns=dictionary.keys())

Variations=expand_grid({"variation_CAPEX_H2" : [-0.5, -0.4 ,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5],
            "variation_prix_GazNat": [10, 20, 30,40,50,60,70,80,90,100]})

alpha_df=applyParallel(Variations.groupby(Variations.index), SensibiliteAlphaSimple)

alpha_df.to_csv('Alpha_matrice_test.csv')  # enregistrement de la matrice dans un csv
#endregion

#region Sensibilité Alpha Parallel with storage, solving and loading results

def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
    return pd.concat(retLst)

def expand_grid(dictionary):
   return pd.DataFrame([row for row in product(*dictionary.values())],
                       columns=dictionary.keys())


#"variation_CAPEX_H2" : [-0.5, -0.4 ,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5]
#"variation_prix_GazNat": [10, 20, 30,40,50,60,70,80,90,100]

Variations=expand_grid({"variation_CAPEX_H2" : [-0.5, -0.4 ],
            "variation_prix_GazNat": [10, 20]})

alpha_df=applyParallel(Variations.groupby(Variations.index), SensibiliteAlpha_WithStorage)

alpha_df.to_csv('Alpha_matrice_test.csv')  # enregistrement de la matrice dans un csv
#endregion

#region Construction pandas avec toutes les matrices alpha
Scenarios=['base','57','35']
list_df=[]

for s in Scenarios :
    df=pd.read_csv('Alpha_matrice_'+s+'.csv',sep=',',decimal='.',skiprows=0) # Récupération de la matrice à partir du csv
    df.drop(columns='Unnamed: 0',inplace=True)
    df['scenario']=s
    list_df.append(df)
matrice_melted=pd.concat(list_df)
matrice_melted.to_csv('Alpha_matrice_melted.csv') # enregistrement de la matrice dans un csv

#endregion

#region Régression linéaire

Imp=pd.read_csv('Alpha_matrice_melted.csv',sep=',',decimal='.',skiprows=0) # Récupération de la matrice à partir du csv
Imp.drop(columns='Unnamed: 0', inplace=True)
MyData = Imp[Imp.scenario == 'base']
MyData.rename(columns={"value": "Simulation"}, inplace=True)
MyData = MyData[MyData.Simulation > 0.0001]
Rdeux,Predictions,Parameters=regression(MyData)

MyData.loc[:,"Prediction"] =Predictions['simple']
MyData=MyData.melt( id_vars=['PrixGaz',"Capex"],var_name="Type",value_vars=["Simulation","Prediction"])
fig2 = px.scatter_3d(MyData, x='PrixGaz', y='Capex', z='value',color='Type')
plotly.offline.plot(fig2, filename='régression aplha.html')

#endregion

#region Construction pandas avec les valeurs de la régréssion opt pour chaque scénario

param_PrixCapexGazCarreCross={}
param_PrixCapexGazCarreCross['EnR100']=Parameters['PrixCapexGazCarreCross']
scenario_list=list(param_PrixCapexGazCarreCross.keys())
param_list=["const","PrixGaz", "Capex", "CapexCarre","PrixGazCarre","Cross"]
pd_values={"const" : [],"PrixGaz": [], "Capex": [], "CapexCarre": [],"PrixGazCarre": [],"Cross": []}
for i in scenario_list :
    pd_values['const'].append(param_PrixCapexGazCarreCross[i][0])
    pd_values['PrixGaz'].append(param_PrixCapexGazCarreCross[i][1])
    pd_values['Capex'].append(param_PrixCapexGazCarreCross[i][2])
    pd_values['CapexCarre'].append(param_PrixCapexGazCarreCross[i][3])
    pd_values['PrixGazCarre'].append(param_PrixCapexGazCarreCross[i][4])
    pd_values['Cross'].append(param_PrixCapexGazCarreCross[i][5])

param_df=pd.DataFrame(pd_values,index=scenario_list)
param_df.to_csv('Parameters_PrixCapexGazCarreCross.csv') # enregistrement de la matrice dans un csv

## contrainte volume

df_test=Variables['importation'].set_index('RESSOURCES')
df_test=df_test.loc['gaz']
df_test=df_test.set_index('TIMESTAMP')
gaz=df_test.sum(axis=0)/1000000

#endregion

#region Ramp Single area with storage : solving and loading results
tol=exp(-4)
n=10
p_max=5000 ### 7 GW de STEP + 10 GW de batteries
storageParameters={"p_max" : p_max , "c_max": p_max*10,"efficiency_in": 0.9,"efficiency_out" : 0.9}
variation_CAPEX_H2=[-0.5,-0.4]#,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5]
variation_prix_GazNat=[10,20,]#30,40,50,60,70,80,90,100]
alpha_df=Boucle_SensibiliteAlphaSimple_With1Storage(areaConsumption, availabilityFactor, TechParameters, ResParameters,
                                                conversionFactor,variation_CAPEX_H2,variation_prix_GazNat,storageParameters,tol,n)
alpha_df.to_csv('Alpha_matrice_test.csv')  # enregistrement de la matrice dans un csv
#endregion

#region Analyse
MyData=pd.read_csv('Alpha_matrice_parallel.csv',sep=',',decimal='.',skiprows=0) # Récupération de la matrice à partir du csv
MyData.drop(columns='Unnamed: 0',inplace=True)
MyData = MyData[MyData.alpha > 0.0001]
Rdeux,Predictions,Parameters=regression(MyData)
MyData.loc[:,"PredictionAlpha"] =Predictions['simple']
Param=Parameters['simple']
def reg_alpha0(reg,PrixGaz):
    Capex=-(reg['const']+reg['PrixGaz']*PrixGaz)/reg['Capex']
    return Capex

PrixGaz_list=np.arange(0,151,1)
Capex_list=reg_alpha0(Param,PrixGaz_list)

#endregion