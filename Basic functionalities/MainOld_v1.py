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
Zones="PACA"
year=2013
other='_WithH2'
PrixRes='horaire'
Selected_TECHNOLOGIES=['OldNuke','Solar','WindOnShore','pac','electrolysis']
areaConsumption, availabilityFactor, TechParameters, conversionFactor, ResParameters, Calendrier = loadingParameters(Selected_TECHNOLOGIES,'Data/Input/',Zones,year,other,PrixRes)

#endregion

#region Ramp Single area : solving and loading results
model = My_GetElectricSystemModel_PlaningSingleNode_MultiRessources(areaConsumption,availabilityFactor,TechParameters,ResParameters,conversionFactor,Calendrier)
opt = SolverFactory(solver)
results=opt.solve(model)
Variables=getVariables_panda_indexed(model)
Cout_import=sum(Variables['importCosts']['importCosts'])
Cout_capa=sum(Variables['capacityCosts']['capacityCosts'])
Cout_Turpe=sum(Variables['turpeCosts'][Variables['turpeCosts'].RESSOURCES=='electricity']['turpeCosts'])
df=pd.DataFrame([Cout_import,Cout_capa,Cout_Turpe],index=['importations','capacités','TURPE'],columns=['Coût'])
df.to_csv('test.csv')
#### lagrange multipliers
Constraints= getConstraintsDual_panda(model)
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

#region Chiffres principaux
Capa_totale = sum(Variables['capacity']['capacity'])/1000
power_use=Variables['power'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='power')
Prod_elec = sum(Variables['power']['power'])/1000000-sum(power_use['electrolysis'])*1.43/1000000
Prod_H2 = sum(power_use['electrolysis'])/1000000
Capa_cost = sum(Variables['capacityCosts']['capacityCosts'])/1000000
Import_cost = sum(Variables['importCosts']['importCosts'])/1000000
Prod_gaz=Variables['power'].loc[Variables['power']['TECHNOLOGIES']=='CCG'].sum(axis=0)['power']/1000000+Variables['power'].loc[Variables['power']['TECHNOLOGIES']=='TAC'].sum(axis=0)['power']/1000000
#Emission_costs = sum(Variables['carbonCosts']['carbonCosts'])/1000000

print('Capa_totale = ',Capa_totale, ' GW')
print('Prod_elec = ',Prod_elec,' TWh')
print('Prod_gaz = ',Prod_gaz,' TWh')
print('Prod_H2 = ',Prod_H2,' TWh')
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

alpha_df.to_csv('Alpha_matrice_0-100-S.csv')  # enregistrement de la matrice dans un csv
#endregion

#region Sensibilité Alpha Parallel with storage, solving and loading results

def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
    return pd.concat(retLst)

def expand_grid(dictionary):
   return pd.DataFrame([row for row in product(*dictionary.values())],
                       columns=dictionary.keys())

Variations=expand_grid({"variation_CAPEX_H2" : [-0.5, -0.4 ,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5],
            "variation_prix_GazNat": [10, 20, 30,40,50,60,70,80,90,100]})

alpha_df=applyParallel(Variations.groupby(Variations.index), SensibiliteAlpha_WithStorage)

alpha_df.to_csv('Alpha_matrice_37-100-A.csv')  # enregistrement de la matrice dans un csv
#endregion

#region Construction pandas avec toutes les matrices alpha
#Scenarios=['base','50-50-S','50-100-S','50-100-A','50-150-S','37-50-S','37-100-S','37-100-A','37-150-S','24-50-S','24-100-S','24-100-A','24-150-S','11-50-S','11-100-S','11-100-A','11-150-S','0-50-S','0-100-S','0-100-A','0-150-S']
#Scenarios=['base','base-H2','50-100-S','50-100-S-H2','37-100-S','37-100-S-H2','24-100-S','24-100-S-H2','11-100-S','11-100-S-H2','0-100-S','0-100-S-H2']
Scenarios=['base','50-100-S','50-100-A','50-100-S-H2','37-100-S','37-100-A','37-100-S-H2','24-100-S','24-100-A','24-100-S-H2','11-100-S','11-100-S-H2','11-100-A','0-100-S','0-100-S-H2','0-100-A']

list_df=[]

for s in Scenarios :
    df=pd.read_csv('Alpha_matrice_'+s+'.csv',sep=',',decimal='.',skiprows=0) # Récupération de la matrice à partir du csv
    df.drop(columns='Unnamed: 0',inplace=True)
    df['scenario']=s
    list_df.append(df)
matrice_melted=pd.concat(list_df)
matrice_melted.to_csv('Alpha_matrice_melted-H2.csv') # enregistrement de la matrice dans un csv

#endregion

#region Régression linéaire

Imp=pd.read_csv('Alpha_matrice_melted-H2.csv',sep=',',decimal='.',skiprows=0) # Récupération de la matrice à partir du csv
Imp.drop(columns='Unnamed: 0', inplace=True)
MyData = Imp[Imp.scenario == 'base']
MyData = MyData[MyData.alpha > 0.0001]
Rdeux,Predictions,Parameters=regression(MyData)

MyData.rename(columns={"alpha": "Simulation"}, inplace=True)
MyData.loc[:,"Prediction"] =Predictions['simple']
MyData=MyData.melt( id_vars=['PrixGaz',"Capex"],var_name="Type",value_vars=["Simulation","Prediction"])
fig2 = px.scatter_3d(MyData, x='PrixGaz', y='Capex', z='value',color='Type')
plotly.offline.plot(fig2, filename='régression aplha H2.html')

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
alpha_df.to_csv('Alpha_matrice_24-100-S.csv')  # enregistrement de la matrice dans un csv
#endregion

#region Analyses
def reg_alpha0(reg,PrixGaz):
    Capex=-(reg['const']+reg['PrixGaz']*PrixGaz)/reg['Capex']
    return Capex

#Scenarios=[['base'],['50-50-S','50-100-S','50-100-A','50-150-S'],['37-50-S','37-100-S','37-100-A','37-150-S'],['24-50-S','24-100-S','24-100-A','24-150-S'],['11-50-S','11-100-S','11-100-A','11-150-S'],['0-50-S','0-100-S','0-100-A','0-150-S']]
Scenarios=[['base','base-H2'],['50-100-S','50-100-S-H2'],['37-100-S','37-100-S-H2'],['24-100-S','24-100-S-H2'],['11-100-S','11-100-S-H2'],['0-100-S','0-100-S-H2']]

Alpha_melted=pd.read_csv('Alpha_matrice_melted-H2.csv',sep=',',decimal='.',skiprows=0) # Récupération de la matrice à partir du csv
Alpha_melted.drop(columns='Unnamed: 0',inplace=True)
Alpha_melted = Alpha_melted[Alpha_melted.alpha > 0.0001]

df=pd.DataFrame(np.arange(0,151,1),columns={'PrixGaz'})

fig1=go.Figure()
fig2=go.Figure()
fig3=go.Figure()
style=[[(37, 253, 233),(44, 117, 255),(22, 184, 78),(237, 0, 0),(243, 214, 23),(223, 115, 255)],['lines','markers','markers+lines','lines'],[1,1,1,0.3]]

c1=0

for s_list in Scenarios :
    c2=0
    for s in s_list :
        Rdeux,Predictions,Parameters=regression(Alpha_melted.loc[Alpha_melted['scenario']==s])
        Alpha_melted.loc[Alpha_melted['scenario']==s,"PredictionAlpha"] =Predictions['simple']
        Alpha_melted.loc[Alpha_melted['scenario'] == s, "Alpha-PredictionAlpha"] = Alpha_melted.loc[Alpha_melted['scenario']==s,"PredictionAlpha"]-Alpha_melted.loc[Alpha_melted['scenario']==s,"alpha"]
        Param=Parameters['simple']
        df.loc[:,s] = reg_alpha0(Param, df.PrixGaz)
        fig1=fig1.add_trace(go.Scatter(x=df.PrixGaz,y=df[s],marker_color='rgb'+str(style[0][c1]),mode=style[1][c2],opacity=style[2][c2],name=s))
        fig2 = fig2.add_trace(go.Scatter(x=Alpha_melted.loc[Alpha_melted['scenario'] == s, "alpha"], y=Alpha_melted.loc[Alpha_melted['scenario'] == s, "PredictionAlpha"],mode='markers',name=s))
        fig3=fig3.add_trace(go.Scatter(x=Alpha_melted.loc[Alpha_melted['scenario'] == s, "alpha"], y=Alpha_melted.loc[Alpha_melted['scenario'] == s, "Alpha-PredictionAlpha"],mode='markers',name=s))
        c2=c2+1
    c1=c1+1

fig1=fig1.update_layout(title='Equations Alpha=0 pour chaque scnéario',xaxis_title='Prix du gaz (€/MWh)', yaxis_title='Capex H2 (€/MWh/an)')
plotly.offline.plot(fig1, filename='Alpha=0.html')
fig2=fig2.update_layout(title='Alpha_Prédiction=f(Alpha) pour chaque scnéario',xaxis_title='Alpha', yaxis_title='Alpha_Prédiction')
fig2 = fig2.add_trace(go.Scatter(x=np.arange(0,1.1,0.1), y=np.arange(0,1.1,0.1),mode='lines',name='Alpha_Prédiction=Alpha'))
plotly.offline.plot(fig2, filename='Alpha_Prediction=f(Alpha).html')
fig3=fig3.update_layout(title='Alpha-Alpha_Prédiction=f(Alpha) pour chaque scnéario',xaxis_title='Alpha', yaxis_title='Alpha-Alpha_Prédiction')
plotly.offline.plot(fig3, filename='Alpha-Alpha_Prédiction=f(Alpha).html')
#endregion

#region Analyse 2
Alpha_melted=pd.read_csv('Alpha_matrice_24-100-S-H2.csv',sep=',',decimal='.',skiprows=0) # Récupération de la matrice à partir du csv
Alpha_melted.drop(columns='Unnamed: 0',inplace=True)
areaConsumption,availabilityFactor, TechParameters, conversionFactor, ResParameters = loadingParameters()
Data=Alpha_melted[Alpha_melted.Capex ==453618.0]
Data.loc[:,'DemRes']=(areaConsumption.groupby('RESSOURCES').agg({"areaConsumption" : "sum"}).loc['electricity']['areaConsumption']/(10**6))\
                     -Data['HydroReservoir_Prod']-Data['HydroRiver_Prod']-Data['Solar_Prod']-Data['WindOnShore_Prod']-Data['OldNuke_Prod']
Data.loc[:,'DemResFactor']=Data.loc[:,'DemRes']/(Data.loc[:,'DemResMax']*8760)
Data.loc[:,'nbh_H2']=Data['electrolysis_Prod']*(10**3)/Data['electrolysis_Capa']
fig=px.scatter(Data,Data.DemRes,Data.nbh_H2,color=Data.scenario,title='nbh_H2=f(DemRes) CAPEX = +50%')
plotly.offline.plot(fig, filename='nbh_H2=f(DemRes)_CAPEX+50.html')

#endregion

#region Analyse 3

#region Utilisation de plotly
Scenarios=['base','50-100-S','50-100-A','50-100-S-H2','37-100-S','37-100-A','37-100-S-H2','24-100-S','24-100-A','24-100-S-H2','11-100-S','11-100-S-H2','11-100-A','0-100-S','0-100-S-H2','0-100-A']

Alpha_melted=pd.read_csv('Alpha_matrice_melted-H2.csv',sep=',',decimal='.',skiprows=0) # Récupération de la matrice à partir du csv
Alpha_melted.drop(columns='Unnamed: 0',inplace=True)

fig1=go.Figure()
My_data=Alpha_melted[(Alpha_melted.Capex==302412) & (Alpha_melted.scenario=='37-100-S')]
fig1.add_trace(go.Scatter(y=My_data.alpha,x=My_data.PrixGaz,name='Sans stockage élec, sans demande H2'))
My_data=Alpha_melted[(Alpha_melted.Capex==302412) & (Alpha_melted.scenario=='37-100-A')]
fig1.add_trace(go.Scatter(y=My_data.alpha,x=My_data.PrixGaz,name='Avec stockage élec, sans demande H2'))
My_data=Alpha_melted[(Alpha_melted.Capex==302412) & (Alpha_melted.scenario=='37-100-S-H2')]
fig1.add_trace(go.Scatter(y=My_data.alpha,x=My_data.PrixGaz,name='Sans stockage élec, avec demande H2'))
fig1.update_layout(title='CAPEX H2 fixe et capacité nucléaire fixe (37 GW)', yaxis_title='alpha',xaxis_title='Prix du gaz €/MWh')
plotly.offline.plot(fig1, filename='alpha=f(prix gaz)')


fig2=go.Figure()

My_data=Alpha_melted[(Alpha_melted.Capex==302412) & (Alpha_melted.PrixGaz==60)]
df1=My_data[My_data.scenario=='50-100-S']
df2=My_data[My_data.scenario=='37-100-S']
df3=My_data[My_data.scenario=='24-100-S']
df4=My_data[My_data.scenario=='11-100-S']
df5=My_data[My_data.scenario=='0-100-S']
My_data=pd.concat([df1,df2,df3,df4,df5])
fig2.add_trace(go.Scatter(y=My_data.alpha,x=My_data.OldNuke_Capa,name='Sans stockage élec, sans demande H2'))

My_data=Alpha_melted[(Alpha_melted.Capex==302412) & (Alpha_melted.PrixGaz==60)]
df1=My_data[My_data.scenario=='50-100-A']
df2=My_data[My_data.scenario=='37-100-A']
df3=My_data[My_data.scenario=='24-100-A']
df4=My_data[My_data.scenario=='11-100-A']
df5=My_data[My_data.scenario=='0-100-A']
My_data=pd.concat([df1,df2,df3,df4,df5])
fig2.add_trace(go.Scatter(y=My_data.alpha,x=My_data.OldNuke_Capa,name='Avec stockage élec, sans demande H2'))

My_data=Alpha_melted[(Alpha_melted.Capex==302412) & (Alpha_melted.PrixGaz==60)]
df1=My_data[My_data.scenario=='50-100-S-H2']
df2=My_data[My_data.scenario=='37-100-S-H2']
df3=My_data[My_data.scenario=='24-100-S-H2']
df4=My_data[My_data.scenario=='11-100-S-H2']
df5=My_data[My_data.scenario=='0-100-S-H2']
My_data=pd.concat([df1,df2,df3,df4,df5])
fig2.add_trace(go.Scatter(y=My_data.alpha,x=My_data.OldNuke_Capa,name='Sans stockage élec, avec demande H2'))

fig2.update_layout(title='CAPEX H2 fixe et prix gaz fixe (60€/MWh)', yaxis_title='alpha',xaxis_title='Capacité nucléaire installée (GW)')
plotly.offline.plot(fig2, filename='alpha=f(EnR)')

fig3=go.Figure()

My_data=Alpha_melted[(Alpha_melted.PrixGaz==60) & (Alpha_melted.scenario=='37-100-S')]
fig3.add_trace(go.Scatter(y=My_data.alpha,x=My_data.Capex,name='Sans stockage élec, sans demande H2'))
My_data=Alpha_melted[(Alpha_melted.PrixGaz==60) & (Alpha_melted.scenario=='37-100-A')]
fig3.add_trace(go.Scatter(y=My_data.alpha,x=My_data.Capex,name='Avec stockage élec, sans demande H2'))
My_data=Alpha_melted[(Alpha_melted.PrixGaz==60) & (Alpha_melted.scenario=='37-100-S-H2')]
fig3.add_trace(go.Scatter(y=My_data.alpha,x=My_data.Capex,name='Sans stockage élec, avec demande H2'))
fig3.update_layout(title='Prix gaz fixe (60€/MWh) et capacité nucléaire fixe (37 GW)', yaxis_title='alpha',xaxis_title='CAPEX H2 €/MWh installés')
plotly.offline.plot(fig3, filename='alpha=f(CAPEX)')

#endregion

#region utilisation matplotlib
Alpha_melted=pd.read_csv('Alpha_matrice_melted-H2.csv',sep=',',decimal='.',skiprows=0) # Récupération de la matrice à partir du csv
Alpha_melted.drop(columns=['Unnamed: 0','HydroReservoir_Capa','electrolysis_Capa','TAC_Capa','pac_Capa','Solar_Capa','CCG_Capa','WindOnShore_Capa',
                           'HydroRiver_Capa', 'CCG_Prod', 'HydroReservoir_Prod', 'HydroRiver_Prod','OldNuke_Prod', 'Solar_Prod', 'TAC_Prod', 'WindOnShore_Prod',
                           'electrolysis_Prod', 'pac_Prod', 'gaz_Conso'],inplace=True)

data_tot=Alpha_melted[Alpha_melted.alpha>0.01]
data_tot.loc[data_tot.OldNuke_Capa<0.1,"OldNuke_Capa"]=0
#NUKE=list(set(data_tot.OldNuke_Capa))
CAPEX=list(set(data_tot.Capex))
#orange'#F46E23',bleu'#165BA0',vert'#14A01B'
test=['0','11','24','37','50']

fig=plt.figure(figsize=(10,10))
ax = [fig.add_subplot(511),fig.add_subplot(512),fig.add_subplot(513),fig.add_subplot(514),fig.add_subplot(515)]

#Sans H2, sans stockage
for i in np.arange(0,5,1) :
    My_data=data_tot[data_tot.scenario==test[i]+'-100-S']
    My_data1=[] # Prix Capex H2
    My_data2=[] # Prix min du gaz
    for capex in CAPEX :
        l=list(My_data.loc[My_data.Capex==capex,'PrixGaz'])
        if len(l)!=0 :
            My_data1.append(capex/1000)
            My_data2.append(min(l))
    ax[i].plot(My_data1,My_data2,color='#165BA0',label='Sans H2\nSans stockage')

#Sans H2, avec stockage
for i in np.arange(0,5,1) :
    My_data=data_tot[data_tot.scenario==test[i]+'-100-A']
    My_data1=[] # Prix Capex H2
    My_data2=[] # Prix min du gaz
    for capex in CAPEX :
        l=list(My_data.loc[My_data.Capex==capex,'PrixGaz'])
        if len(l)!=0 :
            My_data1.append(capex/1000)
            My_data2.append(min(l))
    ax[i].scatter(My_data1,My_data2,color='#F46E23',label='Sans H2\nAvec stockage')

#Avec H2, sans stockage
for i in np.arange(0,5,1) :
    My_data=data_tot[data_tot.scenario==test[i]+'-100-S-H2']
    My_data1=[] # Prix Capex H2
    My_data2=[] # Prix min du gaz
    for capex in CAPEX :
        l=list(My_data.loc[My_data.Capex==capex,'PrixGaz'])
        if len(l)!=0 :
            My_data1.append(capex/1000)
            My_data2.append(min(l))
    ax[i].scatter(My_data1,My_data2,color='#14A01B',marker='x',label='Avec H2\nSans stockage')


for i in np.arange(0,5,1) :
    ax[i].set_title('Capacité nucléaire = '+test[i]+' GW',loc='left')
    ax[i].set_ylabel('Prix du gaz (€/MWh)')

plt.xlabel('Coûts fixes H2 (k€/MWh)')
plt.tight_layout()
ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.subplots_adjust(right=0.8)
plt.savefig('test')
plt.show()

## avec group bars

test=['11','24','37','50']

fig=plt.figure(figsize=(25,3))
ax = [fig.add_subplot(141),fig.add_subplot(142),fig.add_subplot(143),fig.add_subplot(144)]

CAPEX = sorted(list(set(data_tot.Capex)))
labels=[int(CAPEX[i]/1000) for i in np.arange(0,len(CAPEX),1)]
x = np.arange(len(labels))  # the label locations
width = 0.5  # the width of the bars

#Sans H2, sans stockage
for i in np.arange(0,4,1) :
    My_data=data_tot[data_tot.scenario==test[i]+'-100-S']
    My_data1=[] # Prix Capex H2
    My_data2=[] # Prix min du gaz
    for capex in CAPEX :
        l=list(My_data.loc[My_data.Capex==capex,'PrixGaz'])
        if len(l)!=0 :
            My_data1.append(capex)
            My_data2.append(min(l))
    ax[i].bar(x - width / 3, My_data2, width/3,color='#14A01B', label='Sans H2\nSans stockage')

#Sans H2, avec stockage
for i in np.arange(0,4,1) :
    My_data=data_tot[data_tot.scenario==test[i]+'-100-A']
    My_data1=[] # Prix Capex H2
    My_data2=[] # Prix min du gaz
    for capex in CAPEX :
        l=list(My_data.loc[My_data.Capex==capex,'PrixGaz'])
        if len(l)!=0 :
            My_data1.append(capex/1000)
            My_data2.append(min(l))
    ax[i].bar(x, My_data2, width/3,color='#F46E23', label='Sans H2\nAvec stockage')

#Sans H2, sans stockage
for i in np.arange(0,4,1) :
    My_data=data_tot[data_tot.scenario==test[i]+'-100-S-H2']
    My_data1=[] # Prix Capex H2
    My_data2=[] # Prix min du gaz
    for capex in CAPEX :
        l=list(My_data.loc[My_data.Capex==capex,'PrixGaz'])
        if len(l)!=0 :
            My_data1.append(capex/1000)
            My_data2.append(min(l))
        else :
            My_data1.append(capex/1000)
            My_data2.append(0)
    ax[i].bar(x + width / 3, My_data2, width/3,color='#165BA0', label='Avec H2\nSans stockage')

for i in np.arange(0,4,1) :
    ax[i].set_title('Capacité nucléaire = '+test[i]+' GW',loc='left')
    ax[i].set_xlabel('Coûts fixes H2 (k€/MW)')
    ax[i].set_xticks(x)
    ax[i].set_xticklabels(labels)
    ax[i].set_ylim((0,100))
    ax[i].grid(axis='y',linewidth=0.3)
ax[0].set_ylabel('Prix limite du gaz (€/MWh)')
plt.tight_layout()
ax[-1].legend(loc='upper left', bbox_to_anchor=(1, 0.85))
plt.subplots_adjust(top=0.76,right=0.92)
plt.suptitle("Prix limite du gaz naturel pour lequel on utilise l'hydrogène à la pointe",fontsize=16)
plt.savefig('Prix limite gaz')
plt.show()

#region test

test=['11','24','37','50']

fig=plt.figure(figsize=(8,6))
ax = [fig.add_subplot(221),fig.add_subplot(222),fig.add_subplot(223),fig.add_subplot(224)]

CAPEX = sorted(list(set(data_tot.Capex)))
labels=[int(CAPEX[i]/1000) for i in np.arange(0,len(CAPEX),1)]
x = np.arange(len(labels))  # the label locations
width = 0.5  # the width of the bars

#Sans H2, sans stockage
for i in np.arange(0,4,1) :
    My_data=data_tot[data_tot.scenario==test[i]+'-100-S']
    My_data1=[] # Prix Capex H2
    My_data2=[] # Prix min du gaz
    for capex in CAPEX :
        l=list(My_data.loc[My_data.Capex==capex,'PrixGaz'])
        if len(l)!=0 :
            My_data1.append(capex)
            My_data2.append(min(l))
    ax[i].bar(x - width / 3, My_data2, width/3,color='#14A01B', label='Sans H2, Sans stockage')

#Sans H2, avec stockage
for i in np.arange(0,4,1) :
    My_data=data_tot[data_tot.scenario==test[i]+'-100-A']
    My_data1=[] # Prix Capex H2
    My_data2=[] # Prix min du gaz
    for capex in CAPEX :
        l=list(My_data.loc[My_data.Capex==capex,'PrixGaz'])
        if len(l)!=0 :
            My_data1.append(capex/1000)
            My_data2.append(min(l))
    ax[i].bar(x, My_data2, width/3,color='#F46E23', label='Sans H2, Avec stockage')

#Sans H2, sans stockage
for i in np.arange(0,4,1) :
    My_data=data_tot[data_tot.scenario==test[i]+'-100-S-H2']
    My_data1=[] # Prix Capex H2
    My_data2=[] # Prix min du gaz
    for capex in CAPEX :
        l=list(My_data.loc[My_data.Capex==capex,'PrixGaz'])
        if len(l)!=0 :
            My_data1.append(capex/1000)
            My_data2.append(min(l))
        else :
            My_data1.append(capex/1000)
            My_data2.append(0)
    ax[i].bar(x + width / 3, My_data2, width/3,color='#165BA0', label='Avec H2, Sans stockage')

for i in np.arange(0,4,1) :
    ax[i].set_title('Capacité nucléaire = '+test[i]+' GW',loc='left')
    ax[i].set_xticks(x)
    ax[i].set_xticklabels(labels)
    ax[i].set_ylim((0,100))
    ax[i].grid(axis='y',linewidth=0.3)
ax[0].set_ylabel('Prix limite du gaz (€/MWh)')
ax[2].set_ylabel('Prix limite du gaz (€/MWh)')
ax[2].set_xlabel('Coûts limite fixes H2 (k€/MW)')
ax[3].set_xlabel('Coûts limite fixes H2 (k€/MW)')
ax[2].legend(loc='upper left', bbox_to_anchor=(0, -0.27))
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.suptitle("Prix limite du gaz naturel pour lequel on utilise l'hydrogène à la pointe",fontsize=16)
plt.savefig('Prix limite gaz')
plt.show()

#endregion

#endregion

#endregion

#region Analyse cas PACA
Variables=getVariables_panda_indexed(model)
power_use=Variables['power'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='power')
Prix_elec=(Variables['capacityCosts'].set_index('TECHNOLOGIES').loc['Solar']['capacityCosts']+Variables['capacityCosts'].set_index('TECHNOLOGIES').loc['WindOnShore']['capacityCosts']+Variables['importCosts'].set_index('RESSOURCES').loc['electricity']['importCosts'])/(sum(power_use['WindOnShore'])+sum(power_use['Solar'])) #€/MWh
#lagrange=Constraints['energyCtr'].set_index('TIMESTAMP')
#Prix_elec_lagrange=sum(lagrange['energyCtr']*((power_use['WindOnShore']+power_use['Solar'])/(sum(power_use['WindOnShore'])+sum(power_use['Solar']))))
capa_cout=sum(Variables['capacityCosts']['capacityCosts'])
import_cout=sum(Variables['importCosts']['importCosts'])
turpe_cout=Variables['turpeCosts'].set_index('RESSOURCES').loc['electricity']['turpeCosts']
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

#Stockage

Surplu=Variables['energy'].set_index(['TIMESTAMP','RESSOURCES']).join(areaConsumption)
Surplu['surplu']=Surplu['energy']-Surplu['areaConsumption']
Surplu=Surplu.reset_index().set_index('TIMESTAMP').dropna()
Surplu_H2=Surplu[Surplu['RESSOURCES']=='hydrogen']

cumul_H2=[Surplu_H2['surplu'].loc[1]]
for t in np.arange(2,8761,1) :
    cumul_H2.append(cumul_H2[t-2]+Surplu_H2['surplu'].loc[t])

Cumul_H2=pd.DataFrame(cumul_H2,index=Surplu_H2.index,columns=['Cumul_H2'])
TIMESTAMP_d = pd.date_range(start=str(year) + "-01-01 00:00:00", end=str(year) + "-12-31 23:00:00", freq="1H")
Cumul_H2.index = TIMESTAMP_d
fig=px.line(Cumul_H2,title="Variation de l'énergie dans le stockage")
fig = fig.update_layout(xaxis_title="temps", yaxis_title="MWh")
plotly.offline.plot(fig, filename='stockage_H2.html')
s_0=min(Cumul_H2.Cumul_H2)
if s_0<0 : Cumul_H2.Cumul_H2=Cumul_H2.Cumul_H2-min(Cumul_H2.Cumul_H2)
s_max=max(Cumul_H2.Cumul_H2)

print('Etat initial du stockage = ',-s_0/1000,' GWh soit ',-s_0*33.33,' tonnes')
print('Taille de stockage nécessaire = ',s_max/1000,' GWh soit',s_max/33.33,' tonne')

# Coûts système :

plt.pie([capa_cout,import_cout,turpe_cout],labels=['capacité','élec réseau','TURPE'],shadow=True,autopct='%1.1f%%',startangle=90,colors = [  'lightskyblue', 'lightcoral','yellowgreen','gold'])
plt.title('Répartition des coûts système')
plt.savefig('SystemCosts.png')
plt.show()

# Coûts système comparaison :

df1=pd.read_csv('Coûts_Totale.csv',sep=',',decimal='.',skiprows=0).rename(columns={'Coût':'Autoconso\nuniquement'}).set_index('Unnamed: 0')/(1.94*1000000000)*33.3
df2=pd.read_csv('Coûts_Locale.csv',sep=',',decimal='.',skiprows=0).rename(columns={'Coût':'Autoconso \n+ soutirage'}).set_index('Unnamed: 0')/(1.94*1000000000)*33.3
df3=pd.read_csv('Coûts_Décentralisée.csv',sep=',',decimal='.',skiprows=0).rename(columns={'Coût':'EnR délocalisées (PPA)\n+ soutirage'}).set_index('Unnamed: 0')/(1.94*1000000000)*33.3
df4=pd.DataFrame({'Autoconso\nuniquement':0.41*89860*1000/(1.94*1000000000)*33.3,'Autoconso \n+ soutirage':0.33*89860*1000/(1.94*1000000000)*33.3,'EnR délocalisées (PPA)\n+ soutirage':0.39*89860*1000/(1.94*1000000000)*33.3},index=['capacité H2'])
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
plt.bar(df.columns, list(df.loc['TURPE']), bottom=[i + j + k for i, j,k in zip(list(df.loc['capacité H2']), list(df.loc['capacités']),list(df.loc['importations']))], color='#14A01B', label="TURPE")
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
