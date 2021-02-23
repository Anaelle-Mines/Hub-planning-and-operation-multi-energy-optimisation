# region Importation of modules
import os

if os.path.basename(os.getcwd()) == "Basic functionalities":
    os.chdir('..')  ## to work at project root  like in any IDE

import numpy as np
import pandas as pd
import csv

import datetime
import copy
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import sys

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
#### reading CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0).set_index(["TIMESTAMP","RESSOURCES"])
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0).set_index(["TIMESTAMP","TECHNOLOGIES"])
TechParameters = pd.read_csv(InputFolder+'Planing-RAMP2_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0,comment="#").set_index(["TECHNOLOGIES"])
conversionFactor = pd.read_csv(InputFolder+'Ressources_conversionFactors.csv',sep=',',decimal='.',skiprows=0,comment="#").set_index(["RESSOURCES","TECHNOLOGIES"])
ResParameters = pd.read_csv(InputFolder+'Ressources_set.csv',sep=',',decimal='.',skiprows=0,comment="#").set_index(["RESSOURCES"])

#### Selection of subset
availabilityFactor=availabilityFactor.loc[(slice(None),Selected_TECHNOLOGIES),:]
conversionFactor=conversionFactor.loc[(slice(None),Selected_TECHNOLOGIES),:]
TechParameters=TechParameters.loc[Selected_TECHNOLOGIES,:]
TechParameters.loc["OldNuke",'RampConstraintMoins']=0.01 ## a bit strong to put in light the effect
TechParameters.loc["OldNuke",'RampConstraintPlus']=0.02 ## a bit strong to put in light the effect
#endregion

#region Ramp Single area : solving and loading results
model = My_GetElectricSystemModel_PlaningSingleNode_MultiRessources(areaConsumption,availabilityFactor,TechParameters,ResParameters,conversionFactor)
opt = SolverFactory(solver)
results=opt.solve(model)
Variables=getVariables_panda_indexed(model)

#pour avoir la production en KWh de chaque moyen de prod chaque heure
#production_df=Variables['power'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='power')
#energy_variation=
### Check sum Prod = Consumption
#Delta=(production_df.sum(axis=1) - areaConsumption.areaConsumption);
#abs(Delta).max()
#endregion

#region Visualisation des résultats

# Capacity Costs
df=Variables['capacityCosts']
df.drop(df.loc[df['capacityCosts']<1000000].index,inplace=True)
df=pd.DataFrame([list(Variables['capacityCosts']['capacityCosts']/1000000)], columns=list(Variables['capacityCosts']['TECHNOLOGIES']))
fig1 = px.bar(df, title="Capacity costs",width=600)
fig1=fig1.update_layout(xaxis_title="Scenario",yaxis_title="M€")
plotly.offline.plot(fig1, filename='Capacity costs.html')

# Importation Costs
df=Variables['importCosts']
df.drop(df.loc[df['importCosts']<1000000].index,inplace=True)
df=pd.DataFrame([list(Variables['importCosts']['importCosts']/1000000)], columns=list(Variables['importCosts']['RESSOURCES']))
fig2 = px.bar(df, title="Importation costs",width=600)
fig2=fig2.update_layout(xaxis_title="Scenario",yaxis_title="M€")
plotly.offline.plot(fig2, filename='Importation costs.html')

# Carbon Costs
#df=Variables['carbonCosts']
#df.drop(df.loc[df['carbonCosts']<1].index,inplace=True)
#df=pd.DataFrame([list(Variables['carbonCosts']['carbonCosts']/1000000)], columns=list(Variables['carbonCosts']['TECHNOLOGIES']))
#fig6 = px.bar(df, title="Emission costs",width=600)
#fig6=fig6.update_layout(xaxis_title="Scenario",yaxis_title="M€")
#plotly.offline.plot(fig6, filename='Emissioncosts.html')

# Installed Capacity and Total energy production
df1=Variables['capacity']
df1.drop(df1.loc[df1['capacity']<0.1].index,inplace=True)
df1=df1.set_index('TECHNOLOGIES')
df2=Variables['power'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='power')
df2=df2.sum(axis=0)
df2=pd.DataFrame(df2,columns={'energy'})
df2.drop(df2.loc[df2['energy']<100].index,inplace=True)
df=pd.concat([df1,df2],axis=1)
df=df.reset_index()
df1=pd.DataFrame([list(df['capacity']/1000)], columns=list(df['TECHNOLOGIES']))
df2=pd.DataFrame([list(df['energy']/1000000)], columns=list(df['TECHNOLOGIES']))
fig3 = px.bar(df1,barmode='stack',title='Installed capacity')
fig3=fig3.update_layout(xaxis_title="Scenario",yaxis_title="GW",width=600)
plotly.offline.plot(fig3, filename='Installed capacity.html')
fig4 = px.bar(df2,barmode='stack',title='Produced energy',width=600)
fig4=fig4.update_layout(xaxis_title="Scenario",yaxis_title="TWh")
plotly.offline.plot(fig4, filename='Produced energy.html')

# Energy variation
energy_variation=Variables['energy'].pivot(index="TIMESTAMP",columns='RESSOURCES', values='energy')
TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
energy_variation.index=TIMESTAMP_d;
del energy_variation['uranium']
fig5=MyStackedPlotly(y_df=energy_variation)
fig5=fig5.update_layout(title_text="Variation par énergie (production nette + importation) (en MWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig5, filename='Energy variation.html')

# Use of production mean
power_use=Variables['power'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='power')
TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
power_use.index=TIMESTAMP_d;
fig6=MyStackedPlotly(y_df=power_use)
fig6=fig6.update_layout(title_text="Utilisation des moyens de production (en MW)", xaxis_title="heures de l'année")
plotly.offline.plot(fig6, filename='Power.html')

#TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
#productionH2_df.index=TIMESTAMP_d;
#fig2=MyStackedPlotly(y_df=productionH2_df)
#fig2=fig2.update_layout(title_text="Production H2 (en KWh)", xaxis_title="heures de l'année")
#plotly.offline.plot(fig, filename='file.html') ## offline
#fig2.show()

#### lagrange multipliers
Constraints= getConstraintsDual_panda(model)

# Analyse energyCtr
#energyCtrDual=Constraints['energyCtr']; energyCtrDual['energyCtr']=energyCtrDual['energyCtr']*1000000
#energyCtrDual
#round(energyCtrDual.energyCtr,2).unique()

# Analyse CapacityCtr
#CapacityCtrDual=Constraints['CapacityCtr'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='CapacityCtr')*1000000;
#round(CapacityCtrDual,2)
#round(CapacityCtrDual.OldNuke,2).unique() ## if you increase by Delta the installed capacity of nuke you decrease by xxx the cost when nuke is not sufficient
#round(CapacityCtrDual.CCG,2).unique() ## increasing the capacity of CCG as no effect on prices
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

#region Ordres de grandeur, ratio H2/gaz

Zones="FR"
year=2013
Selected_TECHNOLOGIES=['OldNuke','Solar','WindOnShore','HydroReservoir','HydroRiver','TAC','CCG','pac','electrolysis']
#### reading CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0).set_index(["TIMESTAMP","RESSOURCES"])
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0).set_index(["TIMESTAMP","TECHNOLOGIES"])
TechParameters = pd.read_csv(InputFolder+'Planing-RAMP2_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0,comment="#").set_index(["TECHNOLOGIES"])
conversionFactor = pd.read_csv(InputFolder+'Ressources_conversionFactors.csv',sep=',',decimal='.',skiprows=0,comment="#").set_index(["RESSOURCES","TECHNOLOGIES"])
ResParameters = pd.read_csv(InputFolder+'Ressources_set.csv',sep=',',decimal='.',skiprows=0,comment="#").set_index(["RESSOURCES"])

#### Selection of subset
availabilityFactor=availabilityFactor.loc[(slice(None),Selected_TECHNOLOGIES),:]
conversionFactor=conversionFactor.loc[(slice(None),Selected_TECHNOLOGIES),:]
TechParameters=TechParameters.loc[Selected_TECHNOLOGIES,:]
TechParameters.loc["OldNuke",'RampConstraintMoins']=0.01 ## a bit strong to put in light the effect
TechParameters.loc["OldNuke",'RampConstraintPlus']=0.02 ## a bit strong to put in light the effect

#### Boucle pour la variation du CAPEX H2 + la variation du prix du gaz

# Prix du gaz borné à 100€/MWh, au delà => biogaz
# Prix du CAPEX H2 = -50% à +50% par rapport à la référence.
variation_CAPEX_H2=[-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5]
variation_prix_GazNat=[10,20,30,40,50,60,70,80,90,100]
CAPEX_electrolysis=TechParameters['capacityCost']['electrolysis']
CAPEX_pac=TechParameters['capacityCost']['pac']
Prix_gaz=ResParameters['importCost']['gaz']
alpha_list=[]
Prod_EnR_list=[]
Prod_elec_list= []
Prod_gaz_list=[]
Prod_H2_list = []
Prod_Nuke_list=[]
Conso_gaz_list=[]
Capa_gaz_list=[]
Capa_EnR_list=[]
Capa_electrolysis_list = []
Capa_PAC_list = []

for var2 in variation_prix_GazNat  :
    ResParameters['importCost']['gaz'] = var2
    for var1 in variation_CAPEX_H2 :
        TechParameters['capacityCost']['electrolysis'] = CAPEX_electrolysis * (1 + var1)
        TechParameters['capacityCost']['pac'] = CAPEX_pac * (1 + var1)
        model = My_GetElectricSystemModel_PlaningSingleNode_MultiRessources(areaConsumption, availabilityFactor, TechParameters,ResParameters, conversionFactor)
        opt = SolverFactory(solver)
        results = opt.solve(model)
        Variables = getVariables_panda_indexed(model)
        ener = Variables['power'].pivot(index="TIMESTAMP", columns='TECHNOLOGIES', values='power')
        ener = ener.sum(axis=0)
        ener = pd.DataFrame(ener, columns={'energy'})
        alpha=ener.loc['pac']/(ener.loc['CCG']+ener.loc['TAC']+ener.loc['pac'])
        alpha_list=alpha_list+[alpha['energy']]
        Prod_elec_list =Prod_elec_list+[ener.sum(axis=0)['energy']/1000000]
        Prod_EnR_list = Prod_EnR_list+[(ener.loc['Solar']['energy']+ener.loc['HydroReservoir']['energy']+ener.loc['HydroRiver']['energy']+ener.loc['WindOnShore']['energy'])/1000000]
        Prod_gaz_list = Prod_gaz_list+[(ener.loc['CCG']['energy']+ener.loc['TAC']['energy'])/1000000]
        Prod_H2_list = Prod_H2_list+[ener.loc['pac']['energy']/1000000]
        Prod_Nuke_list = Prod_Nuke_list+[ener.loc['OldNuke']['energy']/1000000]
        Conso_gaz_list = Conso_gaz_list+[Variables['importation'].loc[Variables['importation']['RESSOURCES']=='gaz'].sum(axis=0)['importation']/1000000]
        capa=Variables['capacity'].set_index('TECHNOLOGIES')
        Capa_gaz_list = Capa_gaz_list+[(capa.loc['TAC']['capacity']+capa.loc['CCG']['capacity'])/1000]
        Capa_EnR_list = Capa_EnR_list+ [(capa.loc['Solar']['capacity']+capa.loc['HydroReservoir']['capacity']+capa.loc['HydroRiver']['capacity']+capa.loc['WindOnShore']['capacity'])/1000]
        Capa_electrolysis_list = Capa_electrolysis_list+[capa.loc['electrolysis']['capacity']/1000]
        Capa_PAC_list =Capa_PAC_list+[capa.loc['pac']['capacity']/1000]
        print(alpha_list)

### récupérer matrice puis dataframe à partir de la liste des résultats

PrixGaz_list=[]
CAPEX_list=[]
CAPEX_H2=[]
for var1 in variation_CAPEX_H2:
    CAPEX_H2.append(round(CAPEX_electrolysis * (1 + var1)+CAPEX_pac * (1 + var1),1))

for i in variation_prix_GazNat :
    for j in CAPEX_H2 :
        PrixGaz_list.append(i)
        CAPEX_list.append(j)

alpha_df=pd.DataFrame()
alpha_df['PrixGaz']=PrixGaz_list
alpha_df['Capex']=CAPEX_list
alpha_df['value']=alpha_list
alpha_df['Prod_elec']=Prod_elec_list
alpha_df['Prod_EnR']=Prod_EnR_list
alpha_df['Prod_gaz']=Prod_gaz_list
alpha_df['Prod_H2']=Prod_H2_list
alpha_df['Prod_Nuke']=Prod_Nuke_list
alpha_df['Conso_gaz']=Conso_gaz_list
alpha_df['Capa_gaz']=Capa_gaz_list
alpha_df['Capa_EnR']=Capa_EnR_list
alpha_df['Capa_electrolysis']=Capa_electrolysis_list
alpha_df['Capa_PAC']=Capa_PAC_list
alpha_df.to_csv('Alpha_matrice_base.csv') # enregistrement de la matrice dans un csv

### Tracé Heatmap alpha

test=pd.read_csv('Alpha_matrice.csv',sep=',',decimal='.',skiprows=0).set_index('Unnamed: 0') # Récupération de la matrice à partir du csv
fig = go.Figure(data=go.Heatmap(z=test,y=variation_prix_GazNat,x=['-50%','-40%','-30%','-20%','-10%','0%','10%','20%','30%','40%','50%']))
fig.update_layout(title='Proportion NRJ PAC / NRJ TAC + CCG',yaxis_title='Prix du gaz €/MWh',xaxis_title='Variation en % des CAPEX électrolyseurs et PAC par rapport à la référence')
plotly.offline.plot(fig, filename='Abaque aplha.html')

### Tracé de Scatter alpha

fig1 = px.line(test)
fig1.update_layout(title='Proportion NRJ PAC / NRJ TAC + CCG en fonction du prix du gaz pour différentes valeur de CAPEX H2',yaxis_title='Proportion NRJ PAC / NRJ TAC',xaxis_title='Prix du gaz (€/MWh')
plotly.offline.plot(fig1, filename='Scatter aplha.html')

### Tracé selon ratio des index

test=pd.read_csv('Alpha_matrice.csv',sep=',',decimal='.',skiprows=0) # Récupération de la matrice à partir du csv
test.drop(columns='Unnamed: 0',inplace=True)
variation_CAPEX_str=['50%','40%','30%','20%','10%','0%','-10%','-20%','-30%','-40%','-50%']
test.columns=variation_CAPEX_str

beta_list=[]
fig3 = go.Figure()
i=0
for y in variation_CAPEX_H2 :
    beta_list = [y / x for x in variation_prix_GazNat]
    df=test.loc[:,[variation_CAPEX_str[i]]]
    df['abscisse']=beta_list
    print(df)
    fig3.add_trace(go.Scatter(x=df['abscisse'], y=df[variation_CAPEX_str[i]], name=variation_CAPEX_str[i],line_shape='linear'))
    del df
    i = i + 1

fig3.update_layout(title='Proportion NRJ PAC / NRJ TAC + CCG en fonction du ratio Valeur CAPEX H2/Prix Gaz',yaxis_title='Proportion NRJ PAC / NRJ TAC',xaxis_title='ratio Valeur CAPEX H2/Prix Gaz')
plotly.offline.plot(fig3, filename='Alpha_ratio 1D.html')

#endregion

#region Construction pandas avec toutes les matrices alpha
Scenarios=['base']
list_df=[]

for s in Scenarios :
    df=pd.read_csv('Alpha_matrice_'+s+'.csv',sep=',',decimal='.',skiprows=0) # Récupération de la matrice à partir du csv
    df.drop(columns='Unnamed: 0',inplace=True)
    df['scenario']=s
    list_df.append(df)
matrice_melted=pd.concat(list_df)
df.to_csv('Alpha_matrice_melted.csv') # enregistrement de la matrice dans un csv

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

#region Storage

model = My_GetElectricSystemModel_PlaningSingleNode_MultiRessources(areaConsumption,availabilityFactor,TechParameters,ResParameters,conversionFactor)
opt = SolverFactory(solver)

tol=exp(-4)
n=10
p_max=5000 ### 7 GW de STEP + 10 GW de batteries
StorageParameters={"p_max" : p_max , "c_max": p_max*10,"efficiency_in": 0.9,"efficiency_out" : 0.9}
# results = opt.solve(model)
# Variables = getVariables_panda(model) #### obtain optimized variables in panda form
# Constraints = getConstraintsDual_panda(model)  #### obtain lagrange multipliers in panda form

##### Loop
PrixTotal = {}
Consommation = {}
LMultipliers = {}
DeltaPrix = {}
Deltazz = {}
CostFunction = {}
TotalCols = {}
zz = {}
# p_max_out=100.; p_max_in=100.; c_max=10.;

areaConsumption["NewConsumption"] = areaConsumption["areaConsumption"]

nbTime = len(areaConsumption["areaConsumption"])
cpt = 0
for i in model.areaConsumption:
    if i[1]=='electricity':
        model.areaConsumption[i] = areaConsumption.NewConsumption[i]

DeltaPrix_ = tol + 1
while ((DeltaPrix_ > tol) & (n > cpt)):
    print(cpt)
    if (cpt == 0):
        zz[cpt] = [0] * nbTime
    else:
        zz[cpt] = areaConsumption["Storage"].tolist()

    #if solver=="mosek" :
     #    results = opt.solve(model, options= {"dparam.optimizer_max_time":  100.0, "iparam.outlev" : 2,                                                 "iparam.optimizer":     mosek.optimizertype.primal_simplex},tee=True)
    #else :
    #if (solver == 'cplex') | (solver == 'cbc'):
     #   results = opt.solve(model, warmstart=True)
    #else:
    results = opt.solve(model)
    Constraints = getConstraintsDual_panda(model)
    # if solver=='cbc':
    #    Variables = getVariables_panda(model)['energy'].set_index(['TIMESTAMP','TECHNOLOGIES'])
    #    for i in model.energy:  model.energy[i] = Variables.energy[i]

    TotalCols[cpt] = getVariables_panda_indexed(model)['powerCosts'].sum()[1] + getVariables_panda_indexed(model)['importCosts'].sum()[1]
    Prix = Constraints["energyCtr"].assign(Prix=lambda x: x.energyCtr).Prix.to_numpy()
    Prix[Prix <= 0] = 0.0000000001
    valueAtZero = TotalCols[cpt] - Prix * zz[cpt]
    tmpCost = GenCostFunctionFromMarketPrices_dict(Prix, r_in=StorageParameters['efficiency_in'],
                                                   r_out=StorageParameters['efficiency_out'],
                                                   valueAtZero=valueAtZero)
    if (cpt == 0):
        CostFunction[cpt] = GenCostFunctionFromMarketPrices(Prix, r_in=StorageParameters['efficiency_in'],
                                                            r_out=StorageParameters['efficiency_out'],
                                                            valueAtZero=valueAtZero)
    else:
        tmpCost = GenCostFunctionFromMarketPrices_dict(Prix, r_in=StorageParameters['efficiency_in'],
                                                       r_out=StorageParameters['efficiency_out'],
                                                       valueAtZero=valueAtZero)
        tmpCost2 = CostFunction[cpt - 1]
        if StorageParameters['efficiency_in'] * StorageParameters['efficiency_out'] == 1:
            tmpCost2.Maxf_1Breaks_withO(tmpCost['S1'], tmpCost['B1'], tmpCost[
                'f0'])
        else:
            tmpCost2.Maxf_2Breaks_withO(tmpCost['S1'], tmpCost['S2'], tmpCost['B1'], tmpCost['B2'], tmpCost[
                'f0'])  ### etape clé, il faut bien comprendre ici l'utilisation du max de deux fonction de coût
        CostFunction[cpt] = tmpCost2
    LMultipliers[cpt] = Prix
    if cpt > 0:
        DeltaPrix[cpt] = sum(abs(LMultipliers[cpt] - LMultipliers[cpt - 1])) / sum(abs(LMultipliers[cpt]))
        if sum(abs(pd.DataFrame(zz[cpt]))) > 0:
            Deltazz[cpt] = sum(abs(pd.DataFrame(zz[cpt]) - pd.DataFrame(zz[cpt - 1]))) / sum(abs(pd.DataFrame(zz[cpt])))
        else:
            Deltazz[cpt] = 0
        DeltaPrix_ = DeltaPrix[cpt]

    areaConsumption.loc[:, "Storage"] = CostFunction[cpt].OptimMargInt(
        [-StorageParameters['p_max'] / StorageParameters['efficiency_out']] * nbTime,
        [StorageParameters['p_max'] * StorageParameters['efficiency_in']] * nbTime,
        [0] * nbTime,
        [StorageParameters['c_max']] * nbTime)

    areaConsumption.loc[areaConsumption.loc[:, "Storage"] > 0, "Storage"] = areaConsumption.loc[areaConsumption.loc[:,
                                                                                                "Storage"] > 0, "Storage"] / \
                                                                            StorageParameters['efficiency_in']
    areaConsumption.loc[areaConsumption.loc[:, "Storage"] < 0, "Storage"] = areaConsumption.loc[areaConsumption.loc[:,
                                                                                                "Storage"] < 0, "Storage"] * \
                                                                            StorageParameters['efficiency_out']
    areaConsumption.loc[:, "NewConsumption"] = areaConsumption.loc[:, "areaConsumption"] + areaConsumption.loc[:,
                                                                                           "Storage"]
    for i in model.areaConsumption:
        if i[1]=='electricity':
            model.areaConsumption[i] = areaConsumption.NewConsumption[i]
    cpt = cpt + 1

results = opt.solve(model)
stats = {"DeltaPrix": DeltaPrix, "Deltazz": Deltazz}



#endregion
