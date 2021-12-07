#region importation of modules
import os
if os.path.basename(os.getcwd())=="BasicFunctionalities":
    os.chdir('..') ## to work at project root  like in any IDE
import sys
if sys.platform != 'win32':
    myhost = os.uname()[1]
else : myhost = ""
if (myhost=="jupyter-sop"):
    ## for https://jupyter-sop.mines-paristech.fr/ users, you need to
    #  (1) run the following in a terminal
    if (os.system("/opt/mosek/9.2/tools/platform/linux64x86/bin/lmgrd -c /opt/mosek/9.2/tools/platform/linux64x86/bin/mosek.lic -l lmgrd.log")==0):
        os.system("/opt/mosek/9.2/tools/platform/linux64x86/bin/lmutil lmstat -c 27007@127.0.0.1 -a")
    #  (2) definition of license
    os.environ["MOSEKLM_LICENSE_FILE"] = '@jupyter-sop'

import numpy as np
import pandas as pd
import csv
#import docplex
import datetime
import copy
import plotly
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn import linear_model
import sys
import time

from Functions.f_multiResourceModels import *
from Functions.f_optimization import *
from Functions.f_graphicalTools import *
# Change this if you have other solvers obtained here
## https://ampl.com/products/solvers/open-source/
## for eduction this site provides also several professional solvers, that are more efficient than e.g. cbc
#endregion

#region Solver and data location definition
InputFolder='Data/input/'
solver= 'mosek' ## no need for solverpath with mosek.
BaseSolverPath='/Users/robin.girard/Documents/Code/Packages/solvers/ampl_macosx64'
sys.path.append(BaseSolverPath)

solvers= ['gurobi','knitro','cbc'] # 'glpk' is too slow 'cplex' and 'xpress' do not work
solverpath= {}
for solver in solvers : solverpath[solver]=BaseSolverPath+'/'+solver
cplexPATH='/Applications/CPLEX_Studio1210/cplex/bin/x86-64_osx'
sys.path.append(cplexPATH)
solverpath['cplex']=cplexPATH+"/"+"cplex"
solver = 'mosek'
#endregion

#region I - Simple single area : loading parameters
Zones="PACA" ; year=2013; PrixRes='horaire'
Selected_TECHNOLOGIES = ['OldNuke', 'Solar', 'WindOnShore', 'CCG', 'SMR_class']
#### reading areaConsumption availabilityFactor and TechParameters CSV files
areaConsumption,availabilityFactor,TechParameters,conversionFactor,ResParameters,Calendrier,StorageParameters,storageFactors,Economics=loadingParameters(Selected_TECHNOLOGIES,Zones=Zones,year=year,PrixRes=PrixRes)
#endregion

#region I - Simple single area  : Solving and loading results
model = GetElectricSystemModel_MultiResources_SingleNode(areaConsumption, availabilityFactor, TechParameters, ResParameters,conversionFactor)

if solver in solverpath :  opt = SolverFactory(solver,executable=solverpath[solver])
else : opt = SolverFactory(solver)
results=opt.solve(model)
# result analysis
Variables=getVariables_panda_indexed(model)

#pour avoir la production en KWh de chaque moyen de prod chaque heure
production_df=Variables['energy_Pvar'].pivot(index="TIMESTAMP",columns='RESOURCES', values='energy_Pvar')
### Check sum Prod = Consumption
areaConsumption_df=areaConsumption.reset_index().pivot(index="TIMESTAMP",columns='RESOURCES', values='areaConsumption')
Delta=(production_df.sum(axis=0) - areaConsumption_df.sum(axis=0));
abs(Delta).max()

print(production_df.sum(axis=0)/10**6) ### energies produites TWh (ne comprends pas ce qui est consommé par le système)
print(Variables['capacityCosts_Pvar']) #pour avoir le coût de chaque moyen de prod à l'année
print(Variables['powerCosts_Pvar'])
print(Variables['importCosts_Pvar'])
#endregion

#region I - Simple single area  : visualisation and lagrange multipliers
### representation des résultats
power_use=Variables['power_Dvar'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='power_Dvar').drop(columns='electrolysis')
elecConsumption=pd.DataFrame(areaConsumption.loc[(slice(None),'electricity'),'areaConsumption'])

TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
power_use.index=TIMESTAMP_d; elecConsumption.index=TIMESTAMP_d;
fig=MyStackedPlotly(y_df=power_use,Conso = elecConsumption)
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline
#fig.show()

power_use=Variables['power_Dvar'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='power_Dvar').drop(columns=['CCG','OldNuke'])
H2Consumption=pd.DataFrame(areaConsumption.loc[(slice(None),'hydrogen'),'areaConsumption'])

TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
power_use.index=TIMESTAMP_d; H2Consumption.index=TIMESTAMP_d;
fig=MyStackedPlotly(y_df=power_use,Conso = H2Consumption)
fig=fig.update_layout(title_text="Production hydrogène (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline
#fig.show()

#### lagrange multipliers
Constraints= getConstraintsDual_panda(model)

# Analyse energyCtr
energyCtrDual=Constraints['energyCtr']; energyCtrDual['energyCtr']=energyCtrDual['energyCtr']
energyCtrDual
round(energyCtrDual.energyCtr,2).unique()

# Analyse CapacityCtr
CapacityCtrDual=Constraints['CapacityCtr'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='CapacityCtr')*1000000;
round(CapacityCtrDual,2)
round(CapacityCtrDual.OldNuke,2).unique() ## if you increase by Delta the installed capacity of nuke you decrease by xxx the cost when nuke is not sufficient
round(CapacityCtrDual.CCG,2).unique() ## increasing the capacity of CCG as no effect on prices
#endregion

#region II - Storage single area : loading parameters
Zones="PACA" ; year=2013 ; PrixRes='fixe'
Selected_TECHNOLOGIES=['OldNuke','WindOnShore', 'Solar','electrolysis'] ## try adding 'HydroRiver', 'HydroReservoir'
Selected_STECH=['Battery','tankH2_G']
areaConsumption,availabilityFactor,TechParameters,conversionFactor,ResParameters,Calendrier,StorageParameters,storageFactors,Economics=loadingParameters(Selected_TECHNOLOGIES,Selected_STECH,Zones=Zones,year=year,PrixRes=PrixRes)
#endregion

#region II Storage single area : solving and loading results
model= GetElectricSystemModel_MultiResources_SingleNode_WithStorage(areaConsumption, availabilityFactor, TechParameters,ResParameters,conversionFactor,StorageParameters,storageFactors)

start_clock=time.time()
if solver in solverpath :  opt = SolverFactory(solver,executable=solverpath[solver])
else : opt = SolverFactory(solver)
results=opt.solve(model)
Variables = getVariables_panda_indexed(model)
Constraints = getConstraintsDual_panda(model)
end_clock=time.time()
Clock=end_clock-start_clock
print('temps de calcul = ',Clock, 's')

#Stockage
StockageIn=Variables['storageIn_Pvar'].pivot(index=['TIMESTAMP','RESOURCES'],columns='STOCK_TECHNO',values='storageIn_Pvar').sum(axis=1)
StockageOut=Variables['storageOut_Pvar'].pivot(index=['TIMESTAMP','RESOURCES'],columns='STOCK_TECHNO',values='storageOut_Pvar').sum(axis=1)
StockageConsumption=Variables['storageConsumption_Pvar'].pivot(index=['TIMESTAMP','RESOURCES'],columns='STOCK_TECHNO',values='storageConsumption_Pvar').sum(axis=1)
areaConsumption['NewConsumption']=areaConsumption['areaConsumption']+StockageIn-StockageOut

#Energie disponible
production_df=Variables['energy_Pvar'].pivot(index="TIMESTAMP",columns='RESOURCES', values='energy_Pvar')
Delta=production_df.sum(axis=0)-areaConsumption.reset_index().pivot(index="TIMESTAMP",columns='RESOURCES', values='areaConsumption').sum(axis=0)
print("Vérification équilibre O/D : \n",Delta)
print(Variables['capacity_Dvar'])

#electricity
production_elec=pd.DataFrame(production_df['electricity'])
elecConsumption=pd.DataFrame(areaConsumption.loc[(slice(None),'electricity'),['areaConsumption','NewConsumption']])
production_elec['Stockage']=StockageIn.loc[(slice(None),'electricity')]-StockageOut.loc[(slice(None),'electricity')]
TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
production_elec.index=TIMESTAMP_d; elecConsumption.index=TIMESTAMP_d;
fig=MyStackedPlotly(y_df=production_elec,Conso = elecConsumption)
fig=fig.update_layout(title_text="Production électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline
#fig.show()

power_use=Variables['power_Dvar'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='power_Dvar').drop(columns='electrolysis')
TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
power_use.index=TIMESTAMP_d; 
fig=MyStackedPlotly(y_df=power_use)
fig=fig.update_layout(title_text="Gestion électrique (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline

#hydrogen
production_H2=pd.DataFrame(production_df['hydrogen'])
H2Consumption=pd.DataFrame(areaConsumption.loc[(slice(None),'hydrogen'),['areaConsumption','NewConsumption']])
production_H2['Stockage']=StockageIn.loc[(slice(None),'hydrogen')]-StockageOut.loc[(slice(None),'hydrogen')]
TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
production_H2.index=TIMESTAMP_d; H2Consumption.index=TIMESTAMP_d;
fig=MyStackedPlotly(y_df=production_H2,Conso = H2Consumption)
fig=fig.update_layout(title_text="Production H2 (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline

df=Variables['stockLevel_Pvar'].pivot(index='TIMESTAMP',columns='STOCK_TECHNO',values='stockLevel_Pvar')
TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
df.index=TIMESTAMP_d
fig=px.line(df)
fig=fig.update_layout(title_text="Niveau des stocks (en KWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline

#endregion

#region III - Multi-tempo single area : loading parameters
Zones="PACA" ; year='2020-2050'; PrixRes='horaire'
Selected_TECHNOLOGIES = ['OldNuke', 'Solar', 'WindOnShore', 'electrolysis','SMR_class','SMR_CCS1','SMR_CCS2','SMR_elec','SMR_elecCCS1']
dic_eco = {2020: 1, 2030: 2, 2040: 3, 2050: 4}
#### reading areaConsumption availabilityFactor and TechParameters CSV files
areaConsumption,availabilityFactor,TechParameters,conversionFactor,ResParameters,Calendrier,StorageParameters,storageFactors,Economics=loadingParameters_MultiTempo(Selected_TECHNOLOGIES,Zones=Zones,year=year,PrixRes=PrixRes,dic_eco=dic_eco)
#endregion

#region III - Multi-tempo single area  : Solving and loading results
model = GetElectricSystemModel_MultiResources_MultiTempo_SingleNode(areaConsumption, availabilityFactor, TechParameters, ResParameters,conversionFactor,Economics)

start_clock=time.time()
if solver in solverpath :  opt = SolverFactory(solver,executable=solverpath[solver])
else : opt = SolverFactory(solver)
results=opt.solve(model)
end_clock=time.time()
Clock=end_clock-start_clock
print('temps de calcul = ',Clock, 's')

# result analysis
Variables=getVariables_panda_indexed(model)

#pour avoir la production en KWh de chaque moyen de prod chaque heure
production_df=Variables['energy_Pvar'].pivot(index=["TIMESTAMP",'YEAR_op'],columns='RESOURCES', values='energy_Pvar')

### Check sum Prod = Consumption
areaConsumption_df=areaConsumption.reset_index().pivot(index=["TIMESTAMP",'YEAR'],columns='RESOURCES', values='areaConsumption')
Delta=(production_df.sum(axis=0) - areaConsumption_df.sum(axis=0));
abs(Delta).max()

print(production_df.sum(axis=0)/10**6) ### energies produites TWh (ne comprends pas ce qui est consommé par le système)
print(Variables['capacityInvest_Dvar'].set_index('YEAR_invest').loc['1'])
print(Variables['capacityInvest_Dvar'].set_index('YEAR_invest').loc['2'])
print(Variables['capacityInvest_Dvar'].set_index('YEAR_invest').loc['3'])
print(Variables['capacity_Pvar'].set_index('YEAR_op').loc['4'])

power_use=Variables['power_Dvar'].set_index('YEAR_op').loc['4'].pivot(index=['TIMESTAMP'],columns='TECHNOLOGIES',values='power_Dvar').drop(columns=['OldNuke','Solar','WindOnShore'])
year=2050
TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
power_use.index=TIMESTAMP_d;
fig=MyStackedPlotly(y_df=power_use)
fig=fig.update_layout(title_text="Gestion hdyrogen (en MWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline

#endregion



