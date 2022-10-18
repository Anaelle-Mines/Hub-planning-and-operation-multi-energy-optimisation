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
import datetime
import seaborn as sb

from Functions.f_multiResourceModels import *
from Functions.f_optimization import *
from Functions.f_graphicalTools import *
from Functions.f_optimModel_elec import *
from Functions.f_InputScenario import *

# Change this if you have other solvers obtained here
## https://ampl.com/products/solvers/open-source/
## for eduction this site provides also several professional solvers, that are more efficient than e.g. cbc
#endregion

#region Solver and data location definition

#region Choice of Scenario
Scenario='Ref' # Possible Choice : 'Ref', 'eSMR', 'EnR', 'Grid', 'GN', 'BG', 'EnR+','Crack'
ScenarioName='Ref_gasUp_v4'
ScenarioNameFr='Ref_v4'
#endregion

InputFolder='Data/Input/Input_'+ScenarioName+'/'
OutputFolder='Data/output/'
Simul_date='2022-10-18'
SimulName=Simul_date+'_'+ScenarioName
DataCreation_date='2022-10-14'
SimulNameFr=DataCreation_date+'_'+ScenarioNameFr+'_Fr'


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

#region Other models

#region I - Simple single area : loading parameters
Zones="PACA" ; year=2013; PrixRes='horaire'
Selected_TECHNOLOGIES = ['OldNuke','SMR_class']
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
Zones="Fr" ; year=2013 ; PrixRes='fixe'
Selected_TECHNOLOGIES=['OldNuke','WindOnShore', 'Solar','CCG','TAC','Coal_p'] ## try adding 'HydroRiver', 'HydroReservoir'
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
fig=fig.update_layout(title_text="Production électrique (en MWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline
#fig.show()

power_use=Variables['power_Dvar'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='power_Dvar').drop(columns='electrolysis')
TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
power_use.index=TIMESTAMP_d; 
fig=MyStackedPlotly(y_df=power_use)
fig=fig.update_layout(title_text="Gestion électrique (en MWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline

#hydrogen
production_H2=pd.DataFrame(production_df['hydrogen'])
H2Consumption=pd.DataFrame(areaConsumption.loc[(slice(None),'hydrogen'),['areaConsumption','NewConsumption']])
production_H2['Stockage']=StockageIn.loc[(slice(None),'hydrogen')]-StockageOut.loc[(slice(None),'hydrogen')]
TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
production_H2.index=TIMESTAMP_d; H2Consumption.index=TIMESTAMP_d;
fig=MyStackedPlotly(y_df=production_H2,Conso = H2Consumption)
fig=fig.update_layout(title_text="Production H2 (en MWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline

df=Variables['stockLevel_Pvar'].pivot(index='TIMESTAMP',columns='STOCK_TECHNO',values='stockLevel_Pvar')
TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
df.index=TIMESTAMP_d
fig=px.line(df)
fig=fig.update_layout(title_text="Niveau des stocks (en MWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline

#endregion

#region III - Multi-tempo single area : loading parameters
Zones="Fr" ; year='2013'; PrixRes='fixe'
Selected_TECHNOLOGIES = ['OldNuke', 'NewNuke','Solar', 'WindOnShore', 'WindOffShore','HydroRiver','CCG',]
dic_eco = {2020: 1, 2030: 2, 2040: 3, 2050: 4}
#TransFactors=pd.DataFrame({'TECHNO1':['SMR_class_ex','SMR_class_ex','SMR_class','SMR_class','SMR_elec','SMR_CCS1'],'TECHNO2':['SMR_CCS1','SMR_CCS2','SMR_CCS1','SMR_CCS2','SMR_elecCCS1','SMR_CCS2'],'TransFactor':[1,1,1,1,1,1]}).set_index(['TECHNO1','TECHNO2'])
TransFactors=pd.DataFrame({'TECHNO1':[],'TECHNO2':[],'TransFactor':[]}).set_index(['TECHNO1','TECHNO2'])
#### reading areaConsumption availabilityFactor and TechParameters CSV files
areaConsumption,availabilityFactor,TechParameters,conversionFactor,ResParameters,Calendrier,StorageParameters,storageFactors,Economics=loadingParameters_MultiTempo(Selected_TECHNOLOGIES,Zones=Zones,year=year,PrixRes=PrixRes,dic_eco=dic_eco)
#endregion

#region III - Multi-tempo single area  : Solving and loading results
model = GetElectricSystemModel_MultiResources_MultiTempo_SingleNode(areaConsumption, availabilityFactor, TechParameters, ResParameters,conversionFactor,Economics,Calendrier,TransFactors,isAbstract=False)

start_clock=time.time()
if solver in solverpath :  opt = SolverFactory(solver,executable=solverpath[solver])
else : opt = SolverFactory(solver)
results=opt.solve(model)
end_clock=time.time()
Clock=end_clock-start_clock
print('temps de calcul = ',Clock, 's')

# result analysis
Variables=getVariables_panda(model)
Constraints= getConstraintsDual_panda(model)


Variables['transInvest_Dvar'].loc[Variables['transInvest_Dvar'].transInvest_Dvar>0]
Variables['capacityInvest_Dvar'].loc[Variables['capacityInvest_Dvar'].capacityInvest_Dvar>0]
Variables['capacity_Pvar'].loc[Variables['capacity_Pvar'].capacity_Pvar>0]

#del production_df,areaConsumption_df,Delta,production_elec,elecConsumption,power_use,production_H2,H2Consumption

Year_results=2
year={v: k for k, v in dic_eco.items()}[Year_results]

### Check sum Prod = Consumption
production_df=Variables['energy_Pvar'].set_index('YEAR_op').loc[Year_results].pivot(index="TIMESTAMP",columns='RESOURCES', values='energy_Pvar')
areaConsumption_df=areaConsumption.reset_index().set_index('YEAR').loc[Year_results].pivot(index=["TIMESTAMP"],columns='RESOURCES', values='areaConsumption')
Delta=(production_df.sum(axis=0) - areaConsumption_df.sum(axis=0));
abs(Delta).max()
print("Vérification équilibre O/D : \n",Delta)
print("Production par énergie (TWh) : \n",production_df.sum(axis=0)/10**6) ### energies produites TWh (ne comprends pas ce qui est consommé par le système)

# Capacité
print(Variables['capacity_Pvar'].set_index('YEAR_op').loc[Year_results])

#electricity
production_elec=pd.DataFrame(production_df['electricity'])
elecConsumption=pd.DataFrame(areaConsumption.loc[(Year_results,slice(None),'electricity'),['areaConsumption']])
TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
production_elec.index=TIMESTAMP_d; elecConsumption.index=TIMESTAMP_d;
fig=MyStackedPlotly(y_df=production_elec,Conso = elecConsumption)
fig=fig.update_layout(title_text="Production électrique (en MWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline
#fig.show()

#Gestion élec
power_use=Variables['power_Dvar'].set_index('YEAR_op').loc[Year_results].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='power_Dvar')
TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
power_use.index=TIMESTAMP_d;
fig=MyStackedPlotly(y_df=power_use)
fig=fig.update_layout(title_text="Gestion électrique (en MWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline

#Gestion H2
power_use=Variables['power_Dvar'].set_index('YEAR_op').loc[Year_results].pivot(index=['TIMESTAMP'],columns='TECHNOLOGIES',values='power_Dvar').drop(columns=['OldNuke','Solar','WindOnShore'])
TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
power_use.index=TIMESTAMP_d;
fig=MyStackedPlotly(y_df=power_use)
fig=fig.update_layout(title_text="Gestion hdyrogen (en MWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline

#hydrogen
production_H2=pd.DataFrame(production_df['hydrogen'])
H2Consumption=pd.DataFrame(areaConsumption.loc[(Year_results,slice(None),'hydrogen'),['areaConsumption']])
TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
production_H2.index=TIMESTAMP_d; H2Consumption.index=TIMESTAMP_d;
fig=MyStackedPlotly(y_df=production_H2,Conso = H2Consumption)
fig=fig.update_layout(title_text="Production H2 (en MWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline

Variables['carbon_Pvar'].pivot(index="TIMESTAMP",columns='YEAR_op', values='carbon_Pvar').sum()/1000000

#endregion

#endregion

#region IV - Input data creation

#region Descritpion of scenario

Param_list={'MixElec': {'plus':'100%','ref':'75%','moins':'50%'} ,
            'CAPEX_EnR': {'plus':0.9,'ref':0.75,'moins':0.6},
            'CAPEX_eSMR': {'plus':1000,'ref':850,'moins':700},
            'CAPEX_CCS': {'plus':0.9,'ref':0.75,'moins':0.5},
            'CAPEX_elec': {'plus':0.7,'ref':0.55,'moins':0.4},
            'BlackC_price' : {'plus':138,'ref':0,'moins':0},
            'Biogaz_price': {'plus':120,'ref':80,'moins':40},
            'Gaznat_price': {'plus':5,'ref':10,'moins':1},
            'CarbonTax' : {'plus':200,'ref':165,'moins':130},
            'Local_RE':{'plus':[3000,300,2000],'ref':[150,150,1000],'moins':[150,150,1000]}}

Scenario_list={'Ref':{'MixElec':'ref','CAPEX_EnR': 'ref','CAPEX_eSMR': 'ref','CAPEX_CCS': 'ref','CAPEX_elec': 'ref','Biogaz_price': 'ref','Gaznat_price': 'ref','CarbonTax' :'ref','Local_RE':'ref','BlackC_price':'ref'},
               'eSMR':{'MixElec':'ref','CAPEX_EnR': 'ref','CAPEX_eSMR': 'moins','CAPEX_CCS': 'ref','CAPEX_elec': 'plus','Biogaz_price': 'ref','Gaznat_price': 'moins','CarbonTax' :'plus','Local_RE':'ref','BlackC_price':'ref'},
               'EnR':{'MixElec':'moins','CAPEX_EnR': 'moins','CAPEX_eSMR': 'plus','CAPEX_CCS': 'ref','CAPEX_elec': 'moins','Biogaz_price': 'ref','Gaznat_price': 'ref','CarbonTax' :'ref','Local_RE':'ref','BlackC_price':'ref'},
               'EnR+':{'MixElec':'moins','CAPEX_EnR': 'moins','CAPEX_eSMR': 'plus','CAPEX_CCS': 'ref','CAPEX_elec': 'moins','Biogaz_price': 'ref','Gaznat_price': 'ref','CarbonTax' :'ref','Local_RE':'plus','BlackC_price':'ref'},
               'Grid':{'MixElec':'plus','CAPEX_EnR': 'plus','CAPEX_eSMR': 'plus','CAPEX_CCS': 'ref','CAPEX_elec': 'moins','Biogaz_price': 'ref','Gaznat_price': 'ref','CarbonTax' :'ref','Local_RE':'ref','BlackC_price':'ref'},
               'GN':{'MixElec':'ref','CAPEX_EnR': 'plus','CAPEX_eSMR': 'ref','CAPEX_CCS': 'moins','CAPEX_elec': 'ref','Biogaz_price': 'plus','Gaznat_price': 'moins','CarbonTax' :'moins','Local_RE':'ref','BlackC_price':'ref'},
               'BG':{'MixElec':'ref','CAPEX_EnR': 'plus','CAPEX_eSMR': 'ref','CAPEX_CCS': 'plus','CAPEX_elec': 'plus','Biogaz_price': 'moins','Gaznat_price': 'plus','CarbonTax' :'plus','Local_RE':'ref','BlackC_price':'ref'},
               'Crack':{'MixElec':'ref','CAPEX_EnR': 'ref','CAPEX_eSMR': 'ref','CAPEX_CCS': 'ref','CAPEX_elec': 'ref','Biogaz_price': 'ref','Gaznat_price': 'ref','CarbonTax' :'ref','Local_RE':'ref','BlackC_price':'plus'}}

ElecMix= {'100%':{'Solar':[52,100,130],'WindOnShore':[45,70,95],'WindOffShore':[5,35,60],'OldNuke':[54,30,0],'NewNuke':[0,0,0],'HydroRiver':[15,15,15],'HydroReservoir':[15,15,15],'Coal_p':[6,0,0],'TAC':[10,5,0],'CCG':[7,5,5],'Interco':[13,26,39],'curtailment':[10,10,10],'Battery':[10,10,30],'STEP':[5,3,2]},
          '75%':{'Solar':[45,55,75],'WindOnShore':[40,52,70],'WindOffShore':[6,15,45],'OldNuke':[54,45,15],'NewNuke':[0,5,13],'HydroRiver':[15,15,15],'HydroReservoir':[15,15,15],'Coal_p':[6,0,0],'TAC':[10,5,0],'CCG':[7,10,17],'Interco':[13,26,39],'curtailment':[10,20,30],'Battery':[10,10,30],'STEP':[5,3,2]},
          '50%':{'Solar':[40,40,40],'WindOnShore':[45,45,45],'WindOffShore':[5,10,25],'OldNuke':[54,49,29],'NewNuke':[0,5,25],'HydroRiver':[15,15,15],'HydroReservoir':[15,15,15],'Coal_p':[6,0,0],'TAC':[10,5,0],'CCG':[7,10,17],'Interco':[13,26,39],'curtailment':[10,20,30],'Battery':[10,10,30],'STEP':[5,3,2]}}

#endregion

create_data(Scenario,ScenarioName,Scenario_list,Param_list,ElecMix)

#region Mix elec

os.chdir(OutputFolder)
os.chdir(SimulNameFr)
v_list = ['capacityInvest_Dvar','transInvest_Dvar','capacity_Pvar','capacityDel_Pvar','capacityDem_Dvar', 'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
         'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar','carbon_Pvar','powerCosts_Pvar','capacityCosts_Pvar','importCosts_Pvar','storageCosts_Pvar','turpeCosts_Pvar','Pmax_Pvar','max_PS_Dvar','carbonCosts_Pvar']
Variables = {v : pd.read_csv(v + '_' + SimulNameFr + '.csv').drop(columns='Unnamed: 0') for v in v_list}
os.chdir('..')
os.chdir('..')
os.chdir('..')
dic_eco = {2020: 1, 2030: 2, 2040: 3, 2050: 4}
elecProd=Variables['power_Dvar'].set_index('YEAR_op').rename(index=dic_eco).set_index(['TIMESTAMP','TECHNOLOGIES'],append=True)

Prod=elecProd.groupby(['YEAR_op','TECHNOLOGIES']).sum()
Capa=Variables['capacity_Pvar'].set_index(['YEAR_op','TECHNOLOGIES'])

TECHNO=list(elecProd.index.get_level_values('TECHNOLOGIES').unique())
YEAR=list(elecProd.index.get_level_values('YEAR_op').unique())
l_tech=len(TECHNO)
l_year=len(YEAR)

Fossils={y:(Prod.loc[(y,'CCG')]+Prod.loc[(y,'Coal_p')]+Prod.loc[(y,'NewNuke')]+Prod.loc[(y,'OldNuke')]+Prod.loc[(y,'TAC')])['power_Dvar']/Prod.loc[(y,slice(None))].sum()['power_Dvar'] for y in [2,3,4]}
EnR={y:(Prod.loc[(y,'Interco')]+Prod.loc[(y,'Solar')]+Prod.loc[(y,'WindOnShore')]+Prod.loc[(y,'WindOffShore')]+Prod.loc[(y,'HydroRiver')]+Prod.loc[(y,'HydroReservoir')])['power_Dvar']/Prod.loc[(y,slice(None))].sum()['power_Dvar'] for y in [2,3,4]}
test={y:Fossils[y]+EnR[y] for y in [2,3,4]}

sb.set_palette('muted')

fig, ax = plt.subplots()
width= 0.60
x = np.arange(l_year)
cpt=1
for tech in TECHNO :
    l=list(Prod.loc[(slice(None),tech),'power_Dvar']/1000000)
    ax.bar(x + cpt*width/l_tech, l, width/l_tech, label=tech)
    cpt=cpt+1

plt.xticks(x,['2030','2040','2050'])
plt.title('Electricity production')
plt.ylabel('TWh/an')
plt.legend()

os.chdir(OutputFolder)
os.chdir(SimulNameFr)
plt.savefig('Mix prod élec')
os.chdir('..')
os.chdir('..')
os.chdir('..')

plt.show()

fig, ax = plt.subplots()
width= 0.60
x = np.arange(l_year)
cpt=1
for tech in TECHNO :
    l = list(Capa.loc[(slice(None), tech), 'capacity_Pvar']/1000)
    ax.bar(x + cpt * width / l_tech, l, width / l_tech, label=tech)
    cpt=cpt+1

plt.xticks(x,['2030','2040','2050'])
plt.title('Installed capacity')
plt.ylabel('GW')
plt.legend()

os.chdir(OutputFolder)
os.chdir(SimulNameFr)
plt.savefig('Capacités installées')
os.chdir('..')
os.chdir('..')
os.chdir('..')

plt.show()

#endregion

marketPrice=ElecPrice_optim(ScenarioName,SimulNameFr,solver='mosek',InputFolder = 'Data/Input/',OutputFolder = 'Data/output/')

#region Monotones de prix

os.chdir(OutputFolder)
os.chdir(SimulNameFr)
marketPrice=pd.read_csv('marketPrice.csv').set_index(['YEAR_op','TIMESTAMP'])
marketPrice['NewPrice_NonAct'].loc[marketPrice['NewPrice']>250]=250
marketPrice['OldPrice_NonAct'].loc[marketPrice['energyCtr']>250]=250
os.chdir('..')
os.chdir('..')
os.chdir('..')


#Année 2030
MonotoneNew=marketPrice.NewPrice_NonAct.loc[(2030,slice(None))].value_counts(bins=100)
MonotoneNew.sort_index(inplace=True,ascending=False)
NbVal=MonotoneNew.sum()
MonotoneNew_Cumul=[]
MonotoneNew_Price=[]
val=0
for i in MonotoneNew.index :
    val=val+MonotoneNew.loc[i]
    MonotoneNew_Cumul.append(val/NbVal*100)
    MonotoneNew_Price.append(i.right)

MonotoneOld=marketPrice.OldPrice_NonAct.loc[(2030,slice(None))].value_counts(bins=100)
MonotoneOld.sort_index(inplace=True,ascending=False)
NbVal=MonotoneOld.sum()
MonotoneOld_Cumul=[]
MonotoneOld_Price=[]
val=0
for i in MonotoneOld.index :
    val=val+MonotoneOld.loc[i]
    MonotoneOld_Cumul.append(val/NbVal*100)
    MonotoneOld_Price.append(i.right)

sb.set_palette('muted',color_codes=True)
plt.plot(MonotoneNew_Cumul,MonotoneNew_Price,'b-',label='NewPrice 2030')
plt.plot(MonotoneOld_Cumul,MonotoneOld_Price,'b--',label='OldPrice 2030')

#Année 2040
MonotoneNew=marketPrice.NewPrice_NonAct.loc[(2040,slice(None))].value_counts(bins=100)
MonotoneNew.sort_index(inplace=True,ascending=False)
NbVal=MonotoneNew.sum()
MonotoneNew_Cumul=[]
MonotoneNew_Price=[]
val=0
for i in MonotoneNew.index :
    val=val+MonotoneNew.loc[i]
    MonotoneNew_Cumul.append(val/NbVal*100)
    MonotoneNew_Price.append(i.right)

MonotoneOld=marketPrice.OldPrice_NonAct.loc[(2040,slice(None))].value_counts(bins=100)
MonotoneOld.sort_index(inplace=True,ascending=False)
NbVal=MonotoneOld.sum()
MonotoneOld_Cumul=[]
MonotoneOld_Price=[]
val=0
for i in MonotoneOld.index :
    val=val+MonotoneOld.loc[i]
    MonotoneOld_Cumul.append(val/NbVal*100)
    MonotoneOld_Price.append(i.right)

sb.set_palette('muted',color_codes=True)
plt.plot(MonotoneNew_Cumul,MonotoneNew_Price,'r-',label='NewPrice 2040')
plt.plot(MonotoneOld_Cumul,MonotoneOld_Price,'r--',label='OldPrice 2040')

# Année 2050
MonotoneNew = marketPrice.NewPrice_NonAct.loc[(2050, slice(None))].value_counts(bins=100)
MonotoneNew.sort_index(inplace=True, ascending=False)
NbVal = MonotoneNew.sum()
MonotoneNew_Cumul = []
MonotoneNew_Price = []
val = 0
for i in MonotoneNew.index:
    val = val + MonotoneNew.loc[i]
    MonotoneNew_Cumul.append(val / NbVal * 100)
    MonotoneNew_Price.append(i.right)

MonotoneOld = marketPrice.OldPrice_NonAct.loc[(2050, slice(None))].value_counts(bins=100)
MonotoneOld.sort_index(inplace=True, ascending=False)
NbVal = MonotoneOld.sum()
MonotoneOld_Cumul = []
MonotoneOld_Price = []
val = 0
for i in MonotoneOld.index:
    val = val + MonotoneOld.loc[i]
    MonotoneOld_Cumul.append(val / NbVal * 100)
    MonotoneOld_Price.append(i.right)

sb.set_palette('muted',color_codes=True)
plt.plot(MonotoneNew_Cumul, MonotoneNew_Price, 'k-', label='NewPrice 2050')
plt.plot(MonotoneOld_Cumul, MonotoneOld_Price, 'k--', label='OldPrice 2050')


plt.legend()
plt.xlabel('% du temps')
plt.ylabel('Prix (€/MWh)')

os.chdir(OutputFolder)
os.chdir(SimulNameFr)
plt.savefig('Monotone de prix élec _ wo high prices')
os.chdir('..')
os.chdir('..')
os.chdir('..')

plt.show()

#endregion

Carbon_content=pd.read_csv(OutputFolder+SimulNameFr+'/carbon_'+SimulNameFr+'.csv').rename(columns={'YEAR_op':'YEAR'}).set_index(['YEAR','TIMESTAMP'])
marketPrice=pd.read_csv(OutputFolder+SimulNameFr+'/marketPrice.csv').rename(columns={'YEAR_op':'YEAR'}).set_index(['YEAR','TIMESTAMP'])
create_data_PACA(Scenario, ScenarioName, marketPrice,Carbon_content, Param_list, Scenario_list, ElecMix,InputFolder='Data/Input/')

#endregion

#region IV - Multi-tempo with storage single area : loading parameters

#region FR
Zones="Fr" ; year='2020-2050'; PrixRes='horaire'
Selected_TECHNOLOGIES = ['OldNuke','Solar', 'WindOnShore','WindOffShore','CCG','NewNuke','WindOffShore','TAC','HydroRiver','HydroReservoir','curtailment','Interco','Coal_p']
Selected_STECH= ['Battery','STEP']
dic_eco = {2020: 1, 2030: 2, 2040: 3, 2050: 4}
TransFactors=pd.DataFrame({'TECHNO1':[],'TECHNO2':[],'TransFactor':[]}).set_index(['TECHNO1','TECHNO2'])
#### reading areaConsumption availabilityFactor and TechParameters CSV files
areaConsumption,availabilityFactor,TechParameters,conversionFactor,ResParameters,Calendrier,StorageParameters,storageFactor,Economics,CarbonTax=loadingParameters_MultiTempo(Selected_TECHNOLOGIES,Selected_STECH,InputFolder=InputFolder,Zones=Zones,year=year,PrixRes=PrixRes,dic_eco=dic_eco)
#endregion

#region PACA
Zones="PACA" ; year='2020-2050'; PrixRes='horaire'
Selected_TECHNOLOGIES =['Solar', 'WindOnShore','WindOffShore','SMR_class_ex','SMR_elec','SMR_elecCCS1','SMR_class','SMR_CCS1','SMR_CCS2','CCS1','CCS2','electrolysis_PEMEL','electrolysis_AEL','cracking']
Selected_STECH= ['Battery','tankH2_G']
dic_eco = {2020: 1, 2030: 2, 2040: 3, 2050: 4}
TransFactors=pd.DataFrame({'TECHNO1':['SMR_class_ex','SMR_class_ex','SMR_class','SMR_class','SMR_CCS1','SMR_elec'],'TECHNO2':['SMR_CCS1','SMR_CCS2','SMR_CCS1','SMR_CCS2','SMR_CCS2','SMR_elecCCS1'],'TransFactor':[1,1,1,1,1,1]}).set_index(['TECHNO1','TECHNO2'])
#### reading areaConsumption availabilityFactor and TechParameters CSV files
areaConsumption,availabilityFactor,TechParameters,conversionFactor,ResParameters,Calendrier,StorageParameters,storageFactor,Economics,CarbonTax=loadingParameters_MultiTempo_SMR(Selected_TECHNOLOGIES,Selected_STECH,InputFolder=InputFolder,Zones=Zones,year=year,PrixRes=PrixRes,dic_eco=dic_eco)
#endregion

#endregion

#region IV - Multi-tempo with storage single area  : Solving and loading results
model = GetElectricSystemModel_MultiResources_MultiTempo_SingleNode_WithStorage(areaConsumption, availabilityFactor, TechParameters, ResParameters,conversionFactor,Economics,Calendrier,StorageParameters,storageFactor,TransFactors,CarbonTax)

start_clock=time.time()
if solver in solverpath :  opt = SolverFactory(solver,executable=solverpath[solver])
else : opt = SolverFactory(solver)
results=opt.solve(model)
end_clock=time.time()
Clock=end_clock-start_clock
print('temps de calcul = ',Clock, 's')

# result analysis
Variables=getVariables_panda(model)
Constraints= getConstraintsDual_panda(model)
#del production_df,areaConsumption_df, Delta, StockageIn, StockageOut, StockageConsumption,production_elec,elecConsumption,power_use,production_H2,H2Consumption,df

Year_results=4
year={v: k for k, v in dic_eco.items()}[Year_results]

### Check sum Prod = Consumption
production_df=Variables['energy_Pvar'].set_index('YEAR_op').loc[Year_results].pivot(index="TIMESTAMP",columns='RESOURCES', values='energy_Pvar')
areaConsumption_df=areaConsumption.reset_index().set_index('YEAR').loc[Year_results].pivot(index=["TIMESTAMP"],columns='RESOURCES', values='areaConsumption')
Delta=(production_df.sum(axis=0) - areaConsumption_df.sum(axis=0));
abs(Delta).max()
print("Vérification équilibre O/D : \n",Delta)
print("Production par énergie (TWh) : \n",production_df.sum(axis=0)/10**6) ### energies produites TWh (ne comprends pas ce qui est consommé par le système)


### save results
var_name=list(Variables.keys())
cons_name=list(Constraints.keys())
os.chdir(OutputFolder)
d=datetime.date.today()
SimulName=str(d.year)+'-'+str(d.month)+'-'+str(d.day)+'_'+ScenarioName
SimulNameZone=SimulName+'_'+Zones
os.mkdir(SimulNameZone)
os.chdir(SimulNameZone)

for var in var_name :
    Variables[var].to_csv(var+'_'+SimulNameZone+'.csv',index=True)
for cons in cons_name :
    Constraints[cons].to_csv(cons+'_'+SimulNameZone+'.csv',index=True)

dic_an={1:2020, 2:2030, 3:2040, 4:2050}
Prix_elec=Constraints['energyCtr'].set_index('RESOURCES').loc['electricity'].set_index('YEAR_op').rename(index=dic_an)
Carbon=Variables['carbon_Pvar'].set_index(['YEAR_op','TIMESTAMP'])
Prod_elec=Variables['power_Dvar'].groupby(['YEAR_op','TIMESTAMP']).sum()
Carbon_content=Carbon['carbon_Pvar']/Prod_elec['power_Dvar']
Carbon_content=Carbon_content.reset_index().set_index('YEAR_op').rename(index=dic_an,columns={0:'carbonContent'})
Prix_elec.to_csv('elecPrice_'+SimulNameZone+'.csv',index=True)
Carbon_content.to_csv('carbon_'+SimulNameZone+'.csv',index=True)
os.chdir('..')
os.chdir('..')
os.chdir('..')

#endregion

#region IV - Multi-tempo with storage single area  : analysis
Zones='PACA'
SimulName=SimulName+'_'+Zones

#Import results
os.chdir(OutputFolder)
os.chdir(SimulName)
v_list = ['capacityInvest_Dvar','transInvest_Dvar','capacity_Pvar','capacityDem_Dvar','capacityDel_Pvar', 'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
         'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar','carbon_Pvar','powerCosts_Pvar','capacityCosts_Pvar','importCosts_Pvar','storageCosts_Pvar','turpeCosts_Pvar','Pmax_Pvar','max_PS_Dvar','carbonCosts_Pvar']
Variables = {v : pd.read_csv(v + '_' + SimulName + '.csv').drop(columns='Unnamed: 0') for v in v_list}
carbon_content=pd.read_csv('carbon_' + SimulName + '.csv')
os.chdir('..')
os.chdir(SimulNameFr)
elec_price=pd.read_csv('elecPrice_' + SimulNameFr + '.csv')
marketPrice=pd.read_csv('marketPrice.csv')
os.chdir('..')
os.chdir(SimulName)

#region Tracé mix prod H2 et EnR
df0=pd.DataFrame({'YEAR_op':[1],'TECHNOLOGIES':['SMR_class_ex'],'capacity_Pvar':[Variables['capacityInvest_Dvar'].set_index(['YEAR_invest','TECHNOLOGIES']).loc[(1,'SMR_class_ex')].capacityInvest_Dvar]})
df=Variables['capacity_Pvar'].append(df0)
df=df.pivot(columns='TECHNOLOGIES',values='capacity_Pvar', index='YEAR_op').rename(columns={
    "electrolysis_AEL": "Alkaline electrolysis",
    "electrolysis_PEMEL": "PEM electrolysis",
    'SMR_class': "SMR w/o CCUS",
    'SMR_CCS1':  'SMR + CCUS 50%',
    'SMR_CCS2':  'SMR + CCUS 75%',
    'SMR_elec': 'eSMR w/o CCUS',
    'SMR_elecCCS1': 'eSMR + CCUS 50%',
    'cracking': 'Methane cracking'
}).fillna(0)

#LoadFactors
EnR_loadFactor={y : (Variables['power_Dvar'].groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']/(Variables['capacity_Pvar'].set_index(['YEAR_op','TECHNOLOGIES'])['capacity_Pvar']*8760)).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values=0).loc[y,['WindOnShore','Solar','WindOffShore']].fillna(0)  for y in [2,3,4]}
H2_loadFactor={y : (Variables['power_Dvar'].groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']/(Variables['capacity_Pvar'].set_index(['YEAR_op','TECHNOLOGIES'])['capacity_Pvar']*8760)).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values=0).loc[y,['electrolysis_PEMEL','electrolysis_AEL','SMR_class_ex','SMR_class','SMR_CCS1','SMR_CCS2','SMR_elec','SMR_elecCCS1','cracking']].fillna(0) for y in [2,3,4]}
for y in [2,3,4] : H2_loadFactor[y].loc[H2_loadFactor[y]<0]=0
for y in [2,3,4] : H2_loadFactor[y].loc[H2_loadFactor[y]>1]=0

df['SMR w/o CCUS']+=df['SMR_class_ex']

fig, ax = plt.subplots(2,1,sharex=True)
width= 0.30
labels=list(df.index)
x = np.arange(len(labels))
col=sb.color_palette('muted')

# Create dark grey Bar
l1=list(df['SMR w/o CCUS'])
ax[0].bar(x - width/2, l1,width, color=col[7], label="SMR w/o CCUS")
# Create dark bleu Bar
l2=list(df['SMR + CCUS 50%'])
ax[0].bar(x - width/2,l2,width, bottom=l1,color='#005E9E', label="SMR + CCUS 50%")
#Create turquoise bleu Bar
l3=list(df['SMR + CCUS 75%'])
ax[0].bar(x - width/2,l3,width, bottom=[i+j for i,j in zip(l1,l2)], color=col[9] ,label="SMR + CCUS 75%")
#Create orange Bar
l4=list(df['eSMR w/o CCUS'])
ax[0].bar(x - width/2,l4,width, bottom=[i+j+k for i,j,k in zip(l1,l2,l3)], color=col[1],label="eSMR w/o CCUS")
# Create yellow Bars
l5=list(df['eSMR + CCUS 50%'])
ax[0].bar(x - width/2,l5,width, bottom=[i+j+k+l for i,j,k,l in zip(l1,l2,l3,l4)], color='#F8B740',label="eSMR + CCUS 50%")
# Create pink bar
l6=list(df['Methane cracking'])
ax[0].bar(x - width/2,l6,width, bottom=[i+j+k+l+m for i,j,k,l,m in zip(l1,l2,l3,l4,l5)], color=col[6],label="Methane cracking")
# Create green Bars
l7=list(df['Alkaline electrolysis']+df['PEM electrolysis'])
ax[0].bar(x + width/2,l7,width, color=col[2],label="Water electrolysis")

# Create red bar
l8=list(df['Solar'])
ax[1].bar(x ,l8,width, color=col[3],label="Solar")
# Create violet bar
l9=list(df['WindOnShore'])
ax[1].bar(x,l9,width,  bottom=l8,color=col[4],label="WindOnShore")
# Create pink bar
l10=list(df['WindOffShore'])
ax[1].bar(x,l10,width,  bottom=[i+j for i,j in zip(l8,l9)],color=col[6],label="Wind Offshore")

#add Load factors
for i in np.arange(1,len(x)):
    ax[0].text((x + width/2)[i], l7[i]/2, str(round(H2_loadFactor[i+1]['electrolysis_AEL']*100)) +'%',ha='center')
    #ax[1].text((x)[i], l8[i]/2, str(round(EnR_loadFactor[i + 1]['Solar'] * 100)) + '%', ha='center')
   #ax[1].text((x)[i], l8[i]+l9[i]/2, str(round(EnR_loadFactor[i + 1]['WindOnShore'] * 100)) + '%', ha='center')
   # ax[1].text((x)[i], l8[i]+l9[i]+l10[i]/2, str(round(EnR_loadFactor[i + 1]['WindOffShore'] * 100)) + '%', ha='center')

ax[0].set_ylim([0,2000])
ax[1].set_ylim([0,5000])
ax[0].set_ylabel('Installed capacity (MW)')
ax[1].set_ylabel('Installed capacity (MW)')
ax[0].set_title("Evolution of H2 production assets")
ax[1].set_title("Evolution of EnR assets")
plt.xticks(x, ['2020','2020-2030','2030-2040', '2040-2050'])
# Shrink current axis by 20%
box = ax[0].get_position()
ax[0].set_position([box.x0, box.y0, box.width * 0.68, box.height])
# Put a legend to the right of the current axis
ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
box = ax[1].get_position()
ax[1].set_position([box.x0, box.y0, box.width * 0.68, box.height])
# Put a legend to the right of the current axis
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('Evolution mix prod')
plt.show()

#endregion

#region Tracé qté H2
df0=pd.DataFrame({'YEAR_op':[1],'TECHNOLOGIES':['SMR_class_ex'],'power_Dvar':[Variables['capacityInvest_Dvar'].set_index(['YEAR_invest','TECHNOLOGIES']).loc[(1,'SMR_class_ex')].capacityInvest_Dvar*8760]}).set_index(['YEAR_op','TECHNOLOGIES'])
df1=Variables['power_Dvar'].groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP').append(df0).reset_index()
df1=df1.pivot(columns='TECHNOLOGIES',values='power_Dvar', index='YEAR_op').rename(columns={
    "electrolysis_AEL": "Alkaline electrolysis",
    "electrolysis_PEMEL": "PEM electrolysis",
    'SMR_class': "SMR w/o CCUS",
    'SMR_CCS1':  'SMR + CCUS 50%',
    'SMR_CCS2':  'SMR + CCUS 75%',
    'SMR_elec': 'eSMR w/o CCUS',
    'SMR_elecCCS1': 'eSMR + CCUS 50%',
    'cracking': 'Methane cracking'
}).fillna(0)

df1['SMR w/o CCUS']+=df1['SMR_class_ex']
df1=df1/1000


fig, ax = plt.subplots()
width= 0.35
col=sb.color_palette('muted')
labels=list(df1.index)
x = np.arange(len(labels))

# Create dark grey Bar
l1=list(df1['SMR w/o CCUS'])
ax.bar(x - width/2, l1,width, color=col[7], label="SMR w/o CCUS")
# Create dark bleu Bar
l2=list(df1['SMR + CCUS 50%'])
ax.bar(x - width/2,l2,width, bottom=l1,color=col[0], label="SMR + CCUS 50%")
#Create turquoise bleu Bar
l3=list(df1['SMR + CCUS 75%'])
ax.bar(x - width/2,l3,width, bottom=[i+j for i,j in zip(l1,l2)], color=col[9] ,label="SMR + CCUS 75%")
#Create orange Bar
l4=list(df1['eSMR w/o CCUS'])
ax.bar(x - width/2,l4,width, bottom=[i+j+k for i,j,k in zip(l1,l2,l3)], color=col[1],label="eSMR w/o CCUS")
# Create yellow Bars
l5=list(df1['eSMR + CCUS 50%'])
ax.bar(x - width/2,l5,width, bottom=[i+j+k+l for i,j,k,l in zip(l1,l2,l3,l4)], color=col[8],label="eSMR + CCUS 50%")
# Create pink bar
l6=list(df1['Methane cracking'])
ax.bar(x - width/2,l6,width, bottom=[i+j+k+l+m for i,j,k,l,m in zip(l1,l2,l3,l4,l5)], color=col[6],label="Methane cracking")
# Create green Bars
l7=list(df1['Alkaline electrolysis']+df1['PEM electrolysis'])
ax.bar(x + width/2,l7,width, color=col[2],label="Water electrolysis")

ax.set_ylabel('H2 production (GWh)')
ax.set_title("Use of assets")
plt.xticks(x, ['2020','2030','2040', '2050'])
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.74, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('H2 production')
plt.show()

chargeFacteur=df1/(df*8.760).fillna(0)

#endregion

#region Tracé Evolution des technos
df0=Variables['transInvest_Dvar'].set_index(['YEAR_invest','TECHNOLOGIES','TECHNOLOGIES.1'])
df=pd.DataFrame(index=[1,2,3],columns=['SMR w/o CCUS','eSMR w/o CCUS','+ CCUS 50%','+ CCUS 75%','+ CCUS 25%','+ eCCUS 50%'])
df['+ CCUS 50%']=df0.loc[(slice(None),'SMR_class_ex','SMR_CCS1')]+df0.loc[(slice(None),'SMR_class','SMR_CCS1')]
df['+ CCUS 75%']=df0.loc[(slice(None),'SMR_class_ex','SMR_CCS2')]+df0.loc[(slice(None),'SMR_class','SMR_CCS2')]
df['+ CCUS 25%']=df0.loc[(slice(None),'SMR_CCS1','SMR_CCS2')]
df['+ CCUS 50%']-=df['+ CCUS 25%']
df['+ eCCUS 50%']=df0.loc[(slice(None),'SMR_elec','SMR_elecCCS1')]
capa_ex=Variables['capacityInvest_Dvar'].set_index(['YEAR_invest','TECHNOLOGIES']).loc[(1,'SMR_class_ex')].capacityInvest_Dvar
df.loc[0]=0
df=df.sort_index()
for i in np.arange(len(list(df.index))-1) :
    df.loc[i+1]+=df.loc[i]
df['SMR w/o CCUS']=[capa_ex]+[i+j for i,j in zip(list(Variables['capacity_Pvar'].set_index(['YEAR_op','TECHNOLOGIES']).loc[(slice(None),'SMR_class_ex'),'capacity_Pvar']),list(Variables['capacity_Pvar'].set_index(['YEAR_op','TECHNOLOGIES']).loc[(slice(None),'SMR_class'),'capacity_Pvar']))]
df['eSMR w/o CCUS']=[0]+list(Variables['capacity_Pvar'].set_index(['YEAR_op','TECHNOLOGIES']).loc[(slice(None),'SMR_elec'),'capacity_Pvar'])


fig, ax = plt.subplots()
width= 0.35
col=sb.color_palette('muted')
labels=list(df.index)
x = np.arange(len(labels))

#Create Ref Bars
l1=list(df['SMR w/o CCUS'])
ax.bar(x - width/2, l1,width, color=col[7], label="SMR w/o CCUS")
l2=list(df['eSMR w/o CCUS'])
ax.bar(x + width/2, l2,width, color=col[8], label="eSMR w/o CCUS")

#Create Transfo Bars
l3=list(df['+ CCUS 50%'])
ax.bar(x - width/2, l3,width,bottom=l1, color=col[0], label="SMR + CCUS 50%")
l4=list(df['+ CCUS 75%'])
ax.bar(x - width/2, l4,width, bottom=[i+j for i,j in zip(l1,l3)], color=col[9], label="SMR + CCUS 75%")
l5=list(df['+ CCUS 25%'])
ax.bar(x - width/2, l5,width, bottom=[i+j+k for i,j,k in zip(l1,l3,l4)], color=col[9], label="SMR + CCUS 75%")
l6=list(df['+ eCCUS 50%'])
ax.bar(x + width/2, l6,width, bottom=l2, color=col[1], label="eSMR + CCUS 50%")

ax.set_ylabel('Capacité (MW)')
ax.set_title("Evolution des technologies SMR")
plt.xticks(x, ['2020','2020-2030','2030-2040', '2040-2050'])
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.74, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('Evolution SMR')
plt.show()
#endregion

##region Tracé moyenné élec
def weekly_average(df):
    df['week'] = df.index // 168
    return df.groupby('week').sum() / 1000

os.chdir('..')
os.chdir('..')
os.chdir('..')

convFac = pd.read_csv(InputFolder + 'conversionFactors_RESxTECH.csv', sep=',', decimal='.', skiprows=0,comment="#").set_index(["RESOURCES", "TECHNOLOGIES"])

period = {2: "2030", 3: "2040", 4: "2050"}

v = 'power_Dvar'
Pel = {y: Variables[v].loc[Variables[v]['YEAR_op'] == y].pivot(columns='TECHNOLOGIES', values='power_Dvar', index='TIMESTAMP').drop(columns=['CCS1','CCS2']) for y in (2, 3, 4)}
for y in (2, 3, 4) :
    for tech in list(Pel[y].columns):
        Pel[y][tech]=Pel[y][tech]*convFac.loc[('electricity',tech)].conversionFactor
v = 'storageOut_Pvar'
Pel_stock_out = {y: Variables[v][np.logical_and(Variables[v]['YEAR_op'] == y, Variables[v]['RESOURCES'] == 'electricity')].pivot(columns='STOCK_TECHNO',values=v,index='TIMESTAMP') for y in (2, 3, 4)}
v = 'storageIn_Pvar'
Pel_stock_in = {y: Variables[v][np.logical_and(Variables[v]['YEAR_op'] == y, Variables[v]['RESOURCES'] == 'electricity')].pivot(columns='STOCK_TECHNO',values=v,index='TIMESTAMP') for y in (2, 3, 4)}
v = 'importation_Dvar'
Pel_imp = {y: Variables[v][np.logical_and(Variables[v]['YEAR_op'] == y, Variables[v]['RESOURCES'] == 'electricity')].pivot(columns='RESOURCES',values=v,index='TIMESTAMP') for y in (2, 3, 4)}

Pel_exp = {y: -np.minimum(Pel_imp[y], 0) for y in Pel_imp.keys()}
Pel_imp = {y: np.maximum(Pel_imp[y], 0) for y in Pel_imp.keys()}


fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=True)

for k, yr in enumerate((2, 3, 4)):
    ax[k].yaxis.grid(linestyle='--', linewidth=0.5)

    # power_Dvar
    Pel[yr] = weekly_average(Pel[yr])
    # storageOut_Pvar
    Pel_stock_out[yr] = weekly_average(Pel_stock_out[yr])
    # storageIn_Pvar
    Pel_stock_in[yr] = weekly_average(Pel_stock_in[yr])
    # importation_Dvar
    Pel_imp[yr] = weekly_average(Pel_imp[yr])
    Pel_exp[yr] = weekly_average(Pel_exp[yr])

    # Elec production
    ax[k].bar(Pel[yr].index, Pel[yr]['Solar'], label='Solar - PV',color='#ffb233', zorder=-1)
    ax[k].bar(Pel[yr].index, Pel[yr]['Solar'] + Pel[yr]['WindOffShore'], label='Wind - Offshore',color='#33caff', zorder=-2)
    ax[k].bar(Pel[yr].index, Pel[yr]['Solar'] + Pel[yr]['WindOffShore'] + Pel[yr]['WindOnShore'], label='Wind - Onshore', color='#3b8ff9',zorder=-3)
    ax[k].bar(Pel_stock_out[yr].index, Pel_stock_out[yr]['Battery'] + Pel[yr]['WindOnShore'] + Pel[yr]['Solar'],label='Battery - Out',color='#fd46c8', zorder=-4)
    ax[k].bar(Pel_stock_out[yr].index,Pel_stock_out[yr]['Battery'] + Pel[yr]['WindOnShore'] + Pel[yr]['Solar'] + Pel_imp[yr]['electricity'],label='Imports',color='#f74242',  zorder=-5)

    # Elec consumption
    ax[k].bar(Pel[yr].index, Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL'], label='Electrolysis',color='#52de57', zorder=-1)
    ax[k].bar(Pel[yr].index, Pel[yr]['SMR_elec'] + Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL'], label='eSMR',color='#f4f72e', zorder=-2)
    ax[k].bar(Pel[yr].index, Pel[yr]['SMR_CCS1'] + Pel[yr]['SMR_CCS2'] + Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL'] + Pel[yr]['SMR_elec'] , label='CCUS',color='#7c7c7c', zorder=-3)
    ax[k].bar(Pel_stock_in[yr].index, -Pel_stock_in[yr]['Battery'] + Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL'] + Pel[yr]['SMR_elec'] + Pel[yr]['SMR_CCS1'] + Pel[yr]['SMR_CCS2']  , label='Battery - In', color='#d460df' ,zorder=-4)
    ax[k].bar(Pel_stock_in[yr].index,-Pel_stock_in[yr]['Battery'] + Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL'] + Pel[yr]['SMR_elec'] - Pel_exp[yr]['electricity'] + Pel[yr]['SMR_CCS1'] + Pel[yr]['SMR_CCS2'], label='Exports',color='#ff7f7f',zorder=-5)

    ax[k].set_ylabel('Weakly production (GWh)')
    m=(Pel_stock_out[yr]['Battery'] + Pel[yr]['WindOnShore'] + Pel[yr]['Solar'] + Pel_imp[yr]['electricity']).max()+100
    ax[k].set_ylim([-m, m])
    ax[k].set_title(period[yr])
    # Shrink all axis by 20%
    box = ax[k].get_position()
    ax[k].set_position([box.x0, box.y0, box.width * 0.74, box.height])

ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
ax[-1].set_xlabel('Week')
os.chdir(OutputFolder)
os.chdir(SimulName)
plt.savefig('Gestion elec')
plt.show()

#endregion

##region Tracé moyenné H2
def weekly_average(df):
    df['week'] = df.index // 168
    return df.groupby('week').sum() / 1000

os.chdir('..')
os.chdir('..')
os.chdir('..')

convFac = pd.read_csv(InputFolder + 'conversionFactors_RESxTECH.csv', sep=',', decimal='.', skiprows=0,comment="#").set_index(["RESOURCES", "TECHNOLOGIES"])
areaConsumption = pd.read_csv(InputFolder + 'areaConsumption2020-2050_PACA_SMR_TIMExRESxYEAR.csv', sep=',', decimal='.', skiprows=0,comment="#")
areaConsumption['YEAR'] = areaConsumption['YEAR'].replace([2030,2040,2050],[2,3,4])
Conso={y: areaConsumption.loc[areaConsumption['YEAR']==y].pivot(columns='RESOURCES',values='areaConsumption',index='TIMESTAMP') for y in (2, 3, 4) }

period = {2: "2030", 3: "2040", 4: "2050"}

v = 'power_Dvar'
Pel = {y: Variables[v].loc[Variables[v]['YEAR_op'] == y].pivot(columns='TECHNOLOGIES', values='power_Dvar', index='TIMESTAMP').drop(columns=['CCS1','CCS2']) for y in (2, 3, 4)}
for y in (2, 3, 4) :
    for tech in list(Pel[y].columns):
        Pel[y][tech]=Pel[y][tech]*convFac.loc[('hydrogen',tech)].conversionFactor
v = 'storageOut_Pvar'
Pel_stock_out = {y: Variables[v][np.logical_and(Variables[v]['YEAR_op'] == y, Variables[v]['RESOURCES'] == 'hydrogen')].pivot(columns='STOCK_TECHNO',values=v,index='TIMESTAMP') for y in (2, 3, 4)}
v = 'storageIn_Pvar'
Pel_stock_in = {y: Variables[v][np.logical_and(Variables[v]['YEAR_op'] == y, Variables[v]['RESOURCES'] == 'hydrogen')].pivot(columns='STOCK_TECHNO',values=v,index='TIMESTAMP') for y in (2, 3, 4)}
v = 'importation_Dvar'
Pel_imp = {y: Variables[v][np.logical_and(Variables[v]['YEAR_op'] == y, Variables[v]['RESOURCES'] == 'hydrogen')].pivot(columns='RESOURCES',values=v,index='TIMESTAMP') for y in (2, 3, 4)}

Pel_exp = {y: -np.minimum(Pel_imp[y], 0) for y in Pel_imp.keys()}
Pel_imp = {y: np.maximum(Pel_imp[y], 0) for y in Pel_imp.keys()}


fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=True)

for k, yr in enumerate((2, 3, 4)):
    ax[k].yaxis.grid(linestyle='--', linewidth=0.5)

    # power_Dvar
    Pel[yr] = weekly_average(Pel[yr])
    # storageOut_Pvar
    Pel_stock_out[yr] = weekly_average(Pel_stock_out[yr])
    # storageIn_Pvar
    Pel_stock_in[yr] = weekly_average(Pel_stock_in[yr])
    #Demand H2
    Conso[yr] = weekly_average(Conso[yr])
    # importation_Dvar
    Pel_imp[yr] = weekly_average(Pel_imp[yr])
    Pel_exp[yr] = weekly_average(Pel_exp[yr])

    # H2 production
    ax[k].bar(Pel[yr].index, Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL'], label='Electrolysis',color='#52de57', zorder=-1)
    ax[k].bar(Pel[yr].index, Pel[yr]['SMR_elec'] + Pel[yr]['SMR_elecCCS1'] + Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL'], label='eSMR',color='#f4f72e', zorder=-2)
    ax[k].bar(Pel[yr].index, Pel[yr]['SMR_class_ex'] + Pel[yr]['SMR_class'] + Pel[yr]['SMR_CCS1'] + Pel[yr]['SMR_CCS2'] + Pel[yr]['SMR_elec'] + Pel[yr]['SMR_elecCCS1'] + Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL'], label='SMR',color='#7c7c7c', zorder=-3)
    ax[k].bar(Pel[yr].index, Pel[yr]['SMR_class_ex'] + Pel[yr]['SMR_class'] + Pel[yr]['SMR_CCS1'] + Pel[yr]['SMR_CCS2'] + Pel[yr]['SMR_elec'] + Pel[yr]['SMR_elecCCS1'] + Pel[yr]['electrolysis_AEL'] + Pel[yr]['electrolysis_PEMEL']+ Pel[yr]['cracking'],label='Methane cracking', color='#33caff', zorder=-4)
    ax[k].bar(Pel_stock_out[yr].index, Pel_stock_out[yr]['tankH2_G'] + Pel[yr]['SMR_class_ex'] + Pel[yr]['SMR_class'] + Pel[yr]['SMR_CCS1'] + Pel[yr]['SMR_CCS2'] + Pel[yr]['SMR_elec'] + Pel[yr]['SMR_elecCCS1'] + Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL']+ Pel[yr]['cracking'],label='Tank - Out',color='#fd46c8', zorder=-5)
    ax[k].bar(Pel_stock_out[yr].index,Pel_stock_out[yr]['tankH2_G'] + Pel[yr]['SMR_class_ex'] + Pel[yr]['SMR_class'] + Pel[yr]['SMR_CCS1'] + Pel[yr]['SMR_CCS2'] + Pel[yr]['SMR_elec'] + Pel[yr]['SMR_elecCCS1'] + Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL']+ Pel[yr]['cracking']+ Pel_imp[yr]['hydrogen'],label='Imports',color='#f74242',  zorder=-6)

    # H2 concumption
    ax[k].bar(Pel[yr].index, -Conso[yr]['hydrogen'], label='Consumption',color='#ffb233', zorder=-1)
    ax[k].bar(Pel_stock_in[yr].index,-Pel_stock_in[yr]['tankH2_G'] - Conso[yr]['hydrogen'], label='Tank - In',color='#d460df',zorder=-2)

    ax[k].set_ylabel('Weakly production (GWh)')
    m=(Pel_stock_in[yr]['tankH2_G'] + Conso[yr]['hydrogen']).max()+100
    ax[k].set_ylim([-m, m])
    ax[k].set_title(period[yr])
    # Shrink all axis by 20%
    box = ax[k].get_position()
    ax[k].set_position([box.x0, box.y0, box.width * 0.74, box.height])

ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
ax[-1].set_xlabel('Week')
os.chdir(OutputFolder)
os.chdir(SimulName)
plt.savefig('Gestion H2')
plt.show()

#endregion

#region Tracé du stock
def weekly_average(df):
    df['week'] = df.index // 168
    return df.groupby('week').mean() / 1000

os.chdir('..')
os.chdir('..')
os.chdir('..')

os.chdir(OutputFolder)
os.chdir(SimulName)

stock={y:Variables['stockLevel_Pvar'].loc[Variables['stockLevel_Pvar']['YEAR_op']==y].pivot(index='TIMESTAMP',columns='STOCK_TECHNO',values='stockLevel_Pvar') for y in (2,3,4)}

# hourly
fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=True,sharey=True)
for k,yr in enumerate((2, 3, 4)):
    ax[k].plot(stock[yr].index,stock[yr]['tankH2_G']/1000,label='Stock hydrogen')
    ax[k].plot(stock[yr].index, stock[yr]['Battery']/1000, label='Stock electricity')
    ax[k].set_ylabel('Storage (GWh)')
    # Shrink all axis by 20%
    box = ax[k].get_position()
    ax[k].set_position([box.x0, box.y0, box.width * 0.74, box.height])
ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
ax[-1].set_xlabel('Hour')
plt.savefig('Gestion stockage')
plt.show()

# weekly
fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=True,sharey=True)
for k,yr in enumerate((2, 3, 4)):
    stock[yr]=weekly_average(stock[yr])
    ax[k].plot(stock[yr].index,stock[yr]['tankH2_G'],label='Stock hydrogen')
    ax[k].plot(stock[yr].index, stock[yr]['Battery'], label='Stock electricity')
    ax[k].set_ylabel('Mean weakly storage (GWh)')
    # Shrink all axis by 20%
    box = ax[k].get_position()
    ax[k].set_position([box.x0, box.y0, box.width * 0.74, box.height])
ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
ax[-1].set_xlabel('Week')
plt.savefig('Gestion stockage week')
plt.show()

#endregion

##region Tracé qq jours élec

os.chdir('..')
os.chdir('..')
os.chdir('..')

convFac = pd.read_csv(InputFolder + 'conversionFactors_RESxTECH.csv', sep=',', decimal='.', skiprows=0,comment="#").set_index(["RESOURCES", "TECHNOLOGIES"])

period = {2: "2030", 3: "2040", 4: "2050"}

v = 'power_Dvar'
Pel = {y: Variables[v].loc[Variables[v]['YEAR_op'] == y].pivot(columns='TECHNOLOGIES', values='power_Dvar', index='TIMESTAMP').drop(columns=['CCS1','CCS2']) for y in (2, 3, 4)}
for y in (2, 3, 4) :
    for tech in list(Pel[y].columns):
        Pel[y][tech]=Pel[y][tech]*convFac.loc[('electricity',tech)].conversionFactor
v = 'storageOut_Pvar'
Pel_stock_out = {y: Variables[v][np.logical_and(Variables[v]['YEAR_op'] == y, Variables[v]['RESOURCES'] == 'electricity')].pivot(columns='STOCK_TECHNO',values=v,index='TIMESTAMP') for y in (2, 3, 4)}
v = 'storageIn_Pvar'
Pel_stock_in = {y: Variables[v][np.logical_and(Variables[v]['YEAR_op'] == y, Variables[v]['RESOURCES'] == 'electricity')].pivot(columns='STOCK_TECHNO',values=v,index='TIMESTAMP') for y in (2, 3, 4)}
v = 'importation_Dvar'
Pel_imp = {y: Variables[v][np.logical_and(Variables[v]['YEAR_op'] == y, Variables[v]['RESOURCES'] == 'electricity')].pivot(columns='RESOURCES',values=v,index='TIMESTAMP') for y in (2, 3, 4)}

Pel_exp = {y: -np.minimum(Pel_imp[y], 0) for y in Pel_imp.keys()}
Pel_imp = {y: np.maximum(Pel_imp[y], 0) for y in Pel_imp.keys()}

winterDays=[385,433] # 17 et 18 janvier
summerDays=[5833,5881] # 1 et 2 août
x=list(np.arange(0,49))
y0=list(np.zeros(49))

#region Winter

fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=True)

for k, yr in enumerate((2, 3, 4)):
    ax[k].yaxis.grid(linestyle='--', linewidth=0.5)

    # Elec production
    y1=Pel[yr].loc[winterDays[0]:winterDays[1]]['Solar']
    ax[k].fill_between(x, y0, y1, color='#ffb233', label='Solar - PV', linewidth=0)
    y2=y1+Pel[yr].loc[winterDays[0]:winterDays[1]]['WindOnShore']
    ax[k].fill_between(x, y1, y2, color='#3b8ff9', label='Wind - Onshore', linewidth=0)
    y3=y2+Pel_stock_out[yr].loc[winterDays[0]:winterDays[1]]['Battery']
    ax[k].fill_between(x, y2, y3, color='#fd46c8', label='Battery - Out', linewidth=0)
    y4=y3+Pel_imp[yr].loc[winterDays[0]:winterDays[1]]['electricity']
    ax[k].fill_between(x, y3, y4, color='#f74242', label='Imports', linewidth=0)

    # Elec consumption
    y5=Pel[yr].loc[winterDays[0]:winterDays[1]]['electrolysis_AEL']
    ax[k].fill_between(x, y0, y5, color='#52de57', label='Electrolysis', linewidth=0)
    y6=y5+Pel[yr].loc[winterDays[0]:winterDays[1]]['SMR_elec']
    ax[k].fill_between(x, y5, y6, color='#f4f72e', label='eSMR', linewidth=0)
    y7 = y6 + Pel[yr].loc[winterDays[0]:winterDays[1]]['SMR_CCS1']+Pel[yr].loc[winterDays[0]:winterDays[1]]['SMR_CCS2']
    ax[k].fill_between(x, y6, y7, color='#7c7c7c', label='CCUS', linewidth=0)
    y8=y7-Pel_stock_in[yr].loc[winterDays[0]:winterDays[1]]['Battery']
    ax[k].fill_between(x, y7, y8, color='#d460df', label='Battery - In', linewidth=0)
    y9=y8+Pel_exp[yr].loc[winterDays[0]:winterDays[1]]['electricity']
    ax[k].fill_between(x, y8, y9, color='#ff7f7f', label='Exports', linewidth=0)

    ax[k].set_ylabel('Hourly production (MWh)')
    m=y4.max()+100
    ax[k].set_ylim([-m, m])
    ax[k].set_title(period[yr])
    # Shrink all axis by 20%
    box = ax[k].get_position()
    ax[k].set_position([box.x0, box.y0, box.width * 0.74, box.height])

ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
ax[-1].set_xlabel('hour')
os.chdir(OutputFolder)
os.chdir(SimulName)
plt.savefig('Journée elec hiver')
plt.show()

#endregion

#region Summer

fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=True)

for k, yr in enumerate((2, 3, 4)):
    ax[k].yaxis.grid(linestyle='--', linewidth=0.5)

    # Elec production
    y1=Pel[yr].loc[summerDays[0]:summerDays[1]]['Solar']
    ax[k].fill_between(x, y0, y1, color='#ffb233', label='Solar - PV', linewidth=0)
    y2=y1+Pel[yr].loc[summerDays[0]:summerDays[1]]['WindOnShore']
    ax[k].fill_between(x, y1, y2, color='#3b8ff9', label='Wind - Onshore', linewidth=0)
    y3=y2+Pel_stock_out[yr].loc[summerDays[0]:summerDays[1]]['Battery']
    ax[k].fill_between(x, y2, y3, color='#fd46c8', label='Battery - Out', linewidth=0)
    y4=y3+Pel_imp[yr].loc[summerDays[0]:summerDays[1]]['electricity']
    ax[k].fill_between(x, y3, y4, color='#f74242', label='Imports', linewidth=0)

    # Elec consumption
    y5=Pel[yr].loc[summerDays[0]:summerDays[1]]['electrolysis_AEL']
    ax[k].fill_between(x, y0, y5, color='#52de57', label='Electrolysis - PV', linewidth=0)
    y6=y5+Pel[yr].loc[summerDays[0]:summerDays[1]]['SMR_elec']
    ax[k].fill_between(x, y5, y6, color='#f4f72e', label='eSMR - PV', linewidth=0)
    y7 = y6 + Pel[yr].loc[summerDays[0]:summerDays[1]]['SMR_CCS1']+Pel[yr].loc[summerDays[0]:summerDays[1]]['SMR_CCS2']
    ax[k].fill_between(x, y6, y7, color='#7c7c7c', label='CCUS', linewidth=0)
    y8=y7-Pel_stock_in[yr].loc[summerDays[0]:summerDays[1]]['Battery']
    ax[k].fill_between(x, y7, y8, color='#d460df', label='Battery - In', linewidth=0)
    y9=y8+Pel_exp[yr].loc[summerDays[0]:summerDays[1]]['electricity']
    ax[k].fill_between(x, y8, y9, color='#ff7f7f', label='Exports', linewidth=0)

    ax[k].set_ylabel('Hourly production (MWh)')
    m=y4.max()+100
    ax[k].set_ylim([-m, m])
    ax[k].set_title(period[yr])
    # Shrink all axis by 20%
    box = ax[k].get_position()
    ax[k].set_position([box.x0, box.y0, box.width * 0.74, box.height])

ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
ax[-1].set_xlabel('hour')
plt.savefig('Journée elec été')
plt.show()

#endregion

#endregion

##region qq jours H2

os.chdir('..')
os.chdir('..')
os.chdir('..')

convFac = pd.read_csv(InputFolder + 'conversionFactors_RESxTECH.csv', sep=',', decimal='.', skiprows=0,comment="#").set_index(["RESOURCES", "TECHNOLOGIES"])
areaConsumption = pd.read_csv(InputFolder + 'areaConsumption2020-2050_PACA_SMR_TIMExRESxYEAR.csv', sep=',', decimal='.', skiprows=0,comment="#")
areaConsumption['YEAR'] = areaConsumption['YEAR'].replace([2030,2040,2050],[2,3,4])
Conso={y: areaConsumption.loc[areaConsumption['YEAR']==y].pivot(columns='RESOURCES',values='areaConsumption',index='TIMESTAMP') for y in (2, 3, 4) }

period = {2: "2030", 3: "2040", 4: "2050"}

v = 'power_Dvar'
Pel = {y: Variables[v].loc[Variables[v]['YEAR_op'] == y].pivot(columns='TECHNOLOGIES', values='power_Dvar', index='TIMESTAMP').drop(columns=['CCS1','CCS2']) for y in (2, 3, 4)}
for y in (2, 3, 4) :
    for tech in list(Pel[y].columns):
        Pel[y][tech]=Pel[y][tech]*convFac.loc[('hydrogen',tech)].conversionFactor
v = 'storageOut_Pvar'
Pel_stock_out = {y: Variables[v][np.logical_and(Variables[v]['YEAR_op'] == y, Variables[v]['RESOURCES'] == 'hydrogen')].pivot(columns='STOCK_TECHNO',values=v,index='TIMESTAMP') for y in (2, 3, 4)}
v = 'storageIn_Pvar'
Pel_stock_in = {y: Variables[v][np.logical_and(Variables[v]['YEAR_op'] == y, Variables[v]['RESOURCES'] == 'hydrogen')].pivot(columns='STOCK_TECHNO',values=v,index='TIMESTAMP') for y in (2, 3, 4)}
v = 'importation_Dvar'
Pel_imp = {y: Variables[v][np.logical_and(Variables[v]['YEAR_op'] == y, Variables[v]['RESOURCES'] == 'hydrogen')].pivot(columns='RESOURCES',values=v,index='TIMESTAMP') for y in (2, 3, 4)}

Pel_exp = {y: -np.minimum(Pel_imp[y], 0) for y in Pel_imp.keys()}
Pel_imp = {y: np.maximum(Pel_imp[y], 0) for y in Pel_imp.keys()}

winterDays=[385,433] # 17 et 18 janvier
summerDays=[5833,5881] # 1 et 2 août
x=list(np.arange(0,49))
y0=list(np.zeros(49))

#region Winter

fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=True)

for k, yr in enumerate((2, 3, 4)):
    ax[k].yaxis.grid(linestyle='--', linewidth=0.5)

    # H2 production
    y1=Pel[yr].loc[winterDays[0]:winterDays[1]]['electrolysis_AEL']
    ax[k].fill_between(x, y0, y1, color='#52de57', label='Electrolysis', linewidth=0)
    y2=y1+Pel[yr].loc[winterDays[0]:winterDays[1]]['SMR_elec']+Pel[yr].loc[winterDays[0]:winterDays[1]]['SMR_elecCCS1']
    ax[k].fill_between(x, y1, y2, color='#f4f72e', label='eSMR', linewidth=0)
    y3=y2+Pel[yr].loc[winterDays[0]:winterDays[1]]['SMR_class_ex']+Pel[yr].loc[winterDays[0]:winterDays[1]]['SMR_class']+Pel[yr].loc[winterDays[0]:winterDays[1]]['SMR_CCS1']+Pel[yr].loc[winterDays[0]:winterDays[1]]['SMR_CCS2']
    ax[k].fill_between(x, y2, y3, color='#7c7c7c', label='SMR', linewidth=0)
    y4=y3+Pel_stock_out[yr].loc[winterDays[0]:winterDays[1]]['tankH2_G']
    ax[k].fill_between(x, y3, y4, color='#fd46c8', label='Tank - Out', linewidth=0)

    # H2 consumption
    y5=-Conso[yr].loc[winterDays[0]:winterDays[1]]['hydrogen']
    ax[k].fill_between(x, y0, y5, color='#ffb233', label='Consumption', linewidth=0)
    y6=y5-Pel_stock_in[yr].loc[winterDays[0]:winterDays[1]]['tankH2_G']
    ax[k].fill_between(x, y5, y6, color='#d460df', label='Tank - In', linewidth=0)

    ax[k].set_ylabel('Hourly production (MWh)')
    m=y4.max()+100
    ax[k].set_ylim([-m, m])
    ax[k].set_title(period[yr])
    # Shrink all axis by 20%
    box = ax[k].get_position()
    ax[k].set_position([box.x0, box.y0, box.width * 0.74, box.height])

ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
ax[-1].set_xlabel('hour')
os.chdir(OutputFolder)
os.chdir(SimulName)
plt.savefig('Journée H2 hiver')
plt.show()

#endregion

#region Summer

fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=True)

for k, yr in enumerate((2, 3, 4)):
    ax[k].yaxis.grid(linestyle='--', linewidth=0.5)

    # H2 production
    y1=Pel[yr].loc[summerDays[0]:summerDays[1]]['electrolysis_AEL']
    ax[k].fill_between(x, y0, y1, color='#52de57', label='Electrolysis', linewidth=0)
    y2=y1+Pel[yr].loc[summerDays[0]:summerDays[1]]['SMR_elec']+Pel[yr].loc[summerDays[0]:summerDays[1]]['SMR_elecCCS1']
    ax[k].fill_between(x, y1, y2, color='#f4f72e', label='eSMR', linewidth=0)
    y3=y2+Pel[yr].loc[summerDays[0]:summerDays[1]]['SMR_class_ex']+Pel[yr].loc[summerDays[0]:summerDays[1]]['SMR_class']+Pel[yr].loc[summerDays[0]:summerDays[1]]['SMR_CCS1']+Pel[yr].loc[summerDays[0]:summerDays[1]]['SMR_CCS2']
    ax[k].fill_between(x, y2, y3, color='#7c7c7c', label='SMR', linewidth=0)
    y4=y3+Pel_stock_out[yr].loc[summerDays[0]:summerDays[1]]['tankH2_G']
    ax[k].fill_between(x, y3, y4, color='#fd46c8', label='Tank - Out', linewidth=0)

    # H2 consumption
    y5=-Conso[yr].loc[summerDays[0]:summerDays[1]]['hydrogen']
    ax[k].fill_between(x, y0, y5, color='#ffb233', label='Consumption', linewidth=0)
    y6=y5-Pel_stock_in[yr].loc[summerDays[0]:summerDays[1]]['tankH2_G']
    ax[k].fill_between(x, y5, y6, color='#d460df', label='Tank - In', linewidth=0)

    ax[k].set_ylabel('Hourly production (MWh)')
    m=y4.max()+100
    ax[k].set_ylim([-m, m])
    ax[k].set_title(period[yr])
    # Shrink all axis by 20%
    box = ax[k].get_position()
    ax[k].set_position([box.x0, box.y0, box.width * 0.74, box.height])

ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
ax[-1].set_xlabel('hour')
plt.savefig('Journée H2 été')
plt.show()

#endregion

#endregion

#region qq jours stock

os.chdir('..')
os.chdir('..')
os.chdir('..')

os.chdir(OutputFolder)
os.chdir(SimulName)

stock={y:Variables['stockLevel_Pvar'].loc[Variables['stockLevel_Pvar']['YEAR_op']==y].pivot(index='TIMESTAMP',columns='STOCK_TECHNO',values='stockLevel_Pvar') for y in (2,3,4)}
winterDays=[385,433] # 17 et 18 janvier
summerDays=[5833,5881] # 1 et 2 août
x=list(np.arange(0,49))

# winter
fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=True,sharey=True)
for k,yr in enumerate((2, 3, 4)):
    ax[k].plot(x,stock[yr].loc[winterDays[0]:winterDays[1]]['tankH2_G']/1000,label='Stock hydrogen')
    ax[k].plot(x, stock[yr].loc[winterDays[0]:winterDays[1]]['Battery']/1000, label='Stock electricity')
    ax[k].set_ylabel('Storage (GWh)')
    # Shrink all axis by 20%
    box = ax[k].get_position()
    ax[k].set_position([box.x0, box.y0, box.width * 0.74, box.height])
ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
ax[-1].set_xlabel('Hour')
plt.savefig('Journée stock hiver')
plt.show()

# summer
fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=True,sharey=True)
for k,yr in enumerate((2, 3, 4)):
    ax[k].plot(x,stock[yr].loc[summerDays[0]:summerDays[1]]['tankH2_G']/1000,label='Stock hydrogen')
    ax[k].plot(x, stock[yr].loc[summerDays[0]:summerDays[1]]['Battery']/1000, label='Stock electricity')
    ax[k].set_ylabel('Storage (GWh)')
    # Shrink all axis by 20%
    box = ax[k].get_position()
    ax[k].set_position([box.x0, box.y0, box.width * 0.74, box.height])
ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
ax[-1].set_xlabel('Hour')
plt.savefig('journée stock été')
plt.show()

#endregion

#region Tracé CO2
os.chdir('..')
os.chdir('..')
os.chdir('..')

carbon=Variables['carbon_Pvar'].groupby(by='YEAR_op').sum().drop(columns='TIMESTAMP')/1000000
carbon.loc[1]=932
carbon=carbon.sort_index()
fig=plt.plot([2020,2030,2040,2050],carbon.carbon_Pvar,label='Total CO2 emissions')
plt.title('CO2 Emission')
plt.legend()
plt.ylabel('kt/yr')
plt.xlabel('year')
os.chdir(OutputFolder)
os.chdir(SimulName)
plt.savefig('Emissions')
plt.show()
#endregion

#region Tracé répartition coût H2
os.chdir('..')
os.chdir('..')
os.chdir('..')

os.chdir(InputFolder)
convFac=pd.read_csv('conversionFactors_RESxTECH.csv').set_index(['RESOURCES','TECHNOLOGIES'])
Tech=pd.read_csv('set2020-2050_PACA_SMR_TECHxYEAR.csv').set_index('YEAR').rename(index={2020:2,2030:3,2040:4}).set_index('TECHNOLOGIES',append=True)
TaxC=pd.read_csv('CarbonTax_YEAR.csv').rename(columns={'Unnamed: 0':'YEAR'}).set_index('YEAR')
os.chdir('..')
os.chdir('..')
os.chdir('..')

os.chdir(OutputFolder)
os.chdir(SimulName)

Grid_car=carbon_content.rename(columns={'YEAR_op':'YEAR'}).set_index('YEAR').rename(index={2030:2,2040:3,2050:4}).set_index('TIMESTAMP',append=True)
df1=Variables['powerCosts_Pvar'].rename(columns={'YEAR_op':'YEAR','powerCosts_Pvar':'powerCosts'}).set_index(['YEAR','TECHNOLOGIES'])
df1['capacityCosts']=Variables['capacityCosts_Pvar'].rename(columns={'YEAR_op':'YEAR'}).set_index(['YEAR','TECHNOLOGIES'])
df1['Prod']=Variables['power_Dvar'].rename(columns={'YEAR_op':'YEAR'}).groupby(['YEAR','TECHNOLOGIES']).sum().drop(columns=['TIMESTAMP'])
df2=Variables['importCosts_Pvar'].rename(columns={'YEAR_op':'YEAR','importCosts_Pvar':'importCosts'}).set_index(['YEAR','RESOURCES'])
df2['TURPE']=Variables['turpeCosts_Pvar'].rename(columns={'YEAR_op':'YEAR'}).set_index(['YEAR','RESOURCES'])
df3=Variables['capacityCosts_Pvar'].rename(columns={'YEAR_op':'YEAR'}).set_index(['YEAR','TECHNOLOGIES'])
df4=Variables['storageCosts_Pvar'].rename(columns={'YEAR_op':'YEAR','storageCosts_Pvar':'storageCosts'}).set_index(['YEAR','STOCK_TECHNO'])
df5=Variables['carbonCosts_Pvar'].rename(columns={'YEAR_op':'YEAR','carbonCosts_Pvar':'carbon'}).set_index('YEAR')

Y=list(df1.index.get_level_values('YEAR').unique())

for y in Y:
    for tech in ['CCS1','CCS2','WindOnShore','WindOffShore','Solar']:
        df1.drop((y,tech),inplace=True)

# Energy use
TECHNO = list(df1.index.get_level_values('TECHNOLOGIES').unique())
TIMESTAMP=list(Variables['power_Dvar'].set_index('TIMESTAMP').index.get_level_values('TIMESTAMP').unique())

df1['elecUse'] = 0
df1['gasUse'] = 0
df1['carbon'] = 0

for tech in TECHNO:
    df1.loc[(slice(None),tech),'elecUse']=df1.loc[(slice(None),tech),'Prod']*(-convFac.loc[('electricity',tech),'conversionFactor'])
    df1.loc[(slice(None), tech), 'gasUse'] = df1.loc[(slice(None), tech), 'Prod'] * (-convFac.loc[('gaz', tech), 'conversionFactor'])

Elecfac=pd.DataFrame(Y,columns=['YEAR']).set_index('YEAR')
imp=Variables['importation_Dvar'].rename(columns={'YEAR_op':'YEAR'}).set_index(['YEAR','TIMESTAMP','RESOURCES']).loc[(slice(None),slice(None),'electricity')].groupby('YEAR').sum()
for y in Y: Elecfac.loc[y,'ElecFac']=imp.loc[y,'importation_Dvar']/df1['elecUse'].groupby('YEAR').sum().loc[y]

for tech in TECHNO:
    Grid_car[tech]=Variables['power_Dvar'].rename(columns={'YEAR_op':'YEAR'}).set_index(['YEAR','TIMESTAMP','TECHNOLOGIES']).loc[(slice(None),slice(None),tech)]*(-convFac.loc[('electricity',tech),'conversionFactor'])
    Grid_car[tech]=Grid_car[tech]*Grid_car['carbonContent']
    df1.loc[(slice(None), tech), 'carbon'] = (df1.loc[(slice(None), tech), 'Prod'] * ((-convFac.loc[('gaz', tech), 'conversionFactor']) * 203.5 + Tech.loc[(slice(None),tech),'EmissionCO2']) + Grid_car[tech].groupby('YEAR').sum()*Elecfac['ElecFac'])*TaxC['carbonTax']

df1['prodPercent']=0
for y in Y:
    df1.loc[(y,slice(None)),'elecPercent']=df1.loc[(y,slice(None)),'elecUse']/df1['elecUse'].groupby('YEAR').sum().loc[y]
    df1.loc[(y, slice(None)), 'gasPercent'] = df1.loc[(y, slice(None)), 'gasUse']/df1['gasUse'].groupby('YEAR').sum().loc[y]
    df1.loc[(y, 'electrolysis_AEL'), 'prodPercent'] = df1.loc[(y, 'electrolysis_AEL'), 'Prod'] / (df1.loc[(y, 'electrolysis_AEL'), 'Prod']+df1.loc[(y, 'electrolysis_PEMEL'), 'Prod'])
    df1.loc[(y, 'electrolysis_PEMEL'), 'prodPercent'] = df1.loc[(y, 'electrolysis_PEMEL'), 'Prod'] / (df1.loc[(y, 'electrolysis_AEL'), 'Prod'] + df1.loc[(y, 'electrolysis_PEMEL'), 'Prod'])

#regroupement
df1['type'] = 'None'
df1.loc[(slice(None), 'SMR_CCS1'), 'type']='SMR'
df1.loc[(slice(None), 'SMR_CCS2'), 'type']='SMR'
df1.loc[(slice(None), 'SMR_class'), 'type']='SMR'
df1.loc[(slice(None), 'SMR_class_ex'), 'type']='SMR'
df1.loc[(slice(None), 'SMR_elec'), 'type']='eSMR'
df1.loc[(slice(None), 'SMR_elecCCS1'), 'type']='eSMR'
df1.loc[(slice(None), 'electrolysis_AEL'), 'type']='Electrolysis'
df1.loc[(slice(None), 'electrolysis_PEMEL'), 'type']='Electrolysis'
df1.loc[(slice(None), 'cracking'), 'type']='Cracking'

# Repartition coût

for y in Y:
    df1.loc[(y,slice(None)),'importElec']=df1.loc[(y,slice(None)),'elecPercent']*df2.loc[(y,'electricity')]['importCosts']
    df1.loc[(y, slice(None)), 'TURPE'] = df1.loc[(y, slice(None)), 'elecPercent'] * df2.loc[(y, 'electricity')]['TURPE']
    df1.loc[(y,slice(None)),'capexElec']=df1.loc[(y,slice(None)),'elecPercent']*(df3.loc[(y,'WindOnShore')]['capacityCosts_Pvar']+df3.loc[(y,'WindOffShore')]['capacityCosts_Pvar']+df3.loc[(y,'Solar')]['capacityCosts_Pvar'])
    df1.loc[(y, slice(None)), 'importGas'] = df1.loc[(y,slice(None)),'gasPercent']*(df2.loc[(y, 'gazNat')]['importCosts']+df2.loc[(y, 'gazBio')]['importCosts'])
    df1.loc[(y,slice(None)),'storageElec']=df1.loc[(y,slice(None)),'elecPercent']*df4.loc[(y,'Battery')]['storageCosts']
    df1.loc[(y, slice(None)), 'storageH2'] = df1.loc[(y, slice(None)), 'prodPercent'] * df4.loc[(y, 'tankH2_G')]['storageCosts']

df1['Prod'].loc[df1['Prod']<0.0001]=0

TECH=['Electrolysis','SMR','eSMR','Cracking']
df={tech:df1.loc[df1['type']==tech].groupby('YEAR').sum() for tech in TECH}
df_cocon=pd.DataFrame(index=Y)

for tech in TECH:
    if df[tech]['Prod'].sum()==0:
        df.pop(tech)
    else :
        for y in Y:
            if df[tech].loc[y]['Prod']==0:
                df_cocon.loc[y,tech] = df[tech]['capacityCosts'].loc[y]+df[tech]['storageElec'].loc[y]+df[tech]['storageH2'].loc[y]+df[tech]['capexElec'].loc[y]
                df[tech].loc[y]=0
                df[tech].loc[y]['Prod']=1


fig, ax = plt.subplots(figsize=(8,5))
width= 0.20
labels=list(df['Electrolysis'].index)
x = np.arange(len(labels))
col=sb.color_palette('muted')
#code couleur Mines
bl='#2ba9ff'
rd='#fca592'
ye='#F8B740'
br='#e1a673'
gr='#cccccc'
parameters={'axes.labelsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
          'figure.titlesize': 15,
            'legend.fontsize':12}
plt.rcParams.update(parameters)

B=list(df.keys())
B_nb=len(B)
if B_nb%2>0:
    n=B_nb//2
    X=[-1.25*(i+1)*width  for i in np.arange(n)]+[0]+[1.25*(i+1)*width  for i in np.arange(n)]
else:
    n=B_nb/2
    X=[-0.625*(i+1)*width for i in np.arange(n)]+[0.625*(i+1)*width  for i in np.arange(n)]

# Create blue Bars
a={}
for i in np.arange(B_nb):
    a[i]=list(df[B[i]]['capacityCosts']/(df[B[i]]['Prod']*30))
    plt.bar(x + X[i], a[i], width, color=bl,label="Fixed Costs" if i==0 else "")

# Create brown Bars
b={}
for i in np.arange(B_nb):
    b[i]=list(df[B[i]]['importGas']/(df[B[i]]['Prod']*30))
    plt.bar(x + X[i], b[i], width, bottom=a[i], color=br,label="Gas" if i==0 else "")

# Create green Bars
c={}
for i in np.arange(B_nb):
    c[i]=list(df[B[i]]['capexElec']/(df[B[i]]['Prod']*30))
    plt.bar(x + X[i], c[i], width, bottom=[i + j for i, j in zip(a[i],b[i])], color=col[2],label="Local electricity" if i==0 else "")

# Create light red Bars
d={}
for i in np.arange(B_nb):
    d[i]=list(df[B[i]]['importElec']/(df[B[i]]['Prod']*30))
    plt.bar(x + X[i], d[i], width, bottom=[i + j + k for i, j, k in zip(a[i],b[i],c[i])], color=rd,label="Grid electricity" if i==0 else "")

# Create dark red Bars
e={}
for i in np.arange(B_nb):
    e[i]=list(df[B[i]]['TURPE']/(df[B[i]]['Prod']*30))
    plt.bar(x + X[i], e[i], width,  bottom=[i + j + k + l for i, j, k, l in zip(a[i],b[i],c[i],d[i])], color=col[3],label="TURPE" if i==0 else "")

# Create yellow Bars
f={}
for i in np.arange(B_nb):
    f[i]=list(df[B[i]]['storageH2']/(df[B[i]]['Prod']*30))
    plt.bar(x + X[i], f[i], width,   bottom=[i + j + k + l + m for i, j, k, l, m in  zip(a[i],b[i],c[i],d[i],e[i])], color=ye,label="H2 storage capa" if i==0 else "")

# Create orange Bars
g={}
for i in np.arange(B_nb):
    g[i]=list(df[B[i]]['storageElec']/(df[B[i]]['Prod']*30))
    plt.bar(x + X[i], g[i], width,   bottom=[i + j + k + l + m + n for i, j, k, l, m, n in zip(a[i],b[i],c[i],d[i],e[i],f[i])], color=col[1],label="Elec storage capa" if i==0 else "")

# Create grey Bars
h={}
for i in np.arange(B_nb):
    h[i]=list(df[B[i]]['carbon']/(df[B[i]]['Prod']*30))
    plt.bar(x + X[i], h[i], width,   bottom=[i + j + k + l + m + n + o for i, j, k, l, m, n, o in zip(a[i],b[i],c[i],d[i],e[i],f[i],g[i])], color=gr,label="Carbon tax" if i==0 else "")

for i in np.arange(B_nb):
    for j in x:
        ax.text((x+X[i])[j], [k + l + m + n + o + p + q + r + 0.05 for k, l, m, n, o, p, q, r in zip(a[i],b[i],c[i],d[i],e[i],f[i],g[i],h[i])][j],B[i],ha='center',rotation=70)
    print([k + l + m + n + o + p + q + r for k, l, m, n, o, p, q, r in zip(a[i],b[i],c[i],d[i],e[i],f[i],g[i],h[i])])


ax.set_ylabel('Costs (€/kgH2)')
x=list(x)
plt.xticks(x, ['2030','2040', '2050'])
ax.set_ylim([0,5.5])
ax.set_title("Hydrogen production costs")
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.67, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('H2 costs')
plt.show()

#endregion

os.chdir('..')
os.chdir('..')
os.chdir('..')
#endregion

#region Bazare
#region
print(Variables['capacity_Pvar'].set_index('YEAR_op').loc[Year_results])
print(Variables['Cmax_Pvar'].set_index('YEAR_op').loc[Year_results])
print(Variables['Pmax_Pvar'].set_index('YEAR_op').loc[Year_results])

# Stockage
StockageIn=Variables['storageIn_Pvar'].set_index('YEAR_op').loc[Year_results].pivot(index=['TIMESTAMP','RESOURCES'],columns='STOCK_TECHNO',values='storageIn_Pvar').sum(axis=1)
StockageOut=Variables['storageOut_Pvar'].set_index('YEAR_op').loc[Year_results].pivot(index=['TIMESTAMP','RESOURCES'],columns='STOCK_TECHNO',values='storageOut_Pvar').sum(axis=1)
StockageConsumption=Variables['storageConsumption_Pvar'].set_index('YEAR_op').loc[Year_results].pivot(index=['TIMESTAMP','RESOURCES'],columns='STOCK_TECHNO',values='storageConsumption_Pvar').sum(axis=1)
areaConsumption['NewConsumption']=areaConsumption['areaConsumption']+StockageIn-StockageOut

#electricity
production_elec=pd.DataFrame(production_df['electricity'])
elecConsumption=pd.DataFrame(areaConsumption.loc[(Year_results,slice(None),'electricity'),['areaConsumption','NewConsumption']])
production_elec['Stockage']=StockageIn.loc[(slice(None),'electricity')]-StockageOut.loc[(slice(None),'electricity')]
TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
production_elec.index=TIMESTAMP_d; elecConsumption.index=TIMESTAMP_d;
fig=MyStackedPlotly(y_df=production_elec,Conso = elecConsumption)
fig=fig.update_layout(title_text="Production électrique (en MWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline
#fig.show()

#Gestion élec
power_use=Variables['power_Dvar'].set_index('YEAR_op').loc[Year_results].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='power_Dvar')#.drop(columns=['electrolysis','SMR_class','SMR_CCS1','SMR_CCS2','SMR_elec','SMR_elecCCS1','SMR_class_ex','CCS1','CCS2'])
TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
power_use.index=TIMESTAMP_d;
fig=MyStackedPlotly(y_df=power_use)
fig=fig.update_layout(title_text="Gestion électrique (en MWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline

#Gestion H2
power_use=Variables['power_Dvar'].set_index('YEAR_op').loc[Year_results].pivot(index=['TIMESTAMP'],columns='TECHNOLOGIES',values='power_Dvar').drop(columns=['OldNuke','Solar','WindOnShore','CCS1','CCS2'])
TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
power_use.index=TIMESTAMP_d;
fig=MyStackedPlotly(y_df=power_use)
fig=fig.update_layout(title_text="Gestion hdyrogen (en MWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline

#hydrogen
production_H2=pd.DataFrame(production_df['hydrogen'])
H2Consumption=pd.DataFrame(areaConsumption.loc[(Year_results,slice(None),'hydrogen'),['areaConsumption','NewConsumption']])
production_H2['Stockage']=StockageIn.loc[(slice(None),'hydrogen')]-StockageOut.loc[(slice(None),'hydrogen')]
TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
production_H2.index=TIMESTAMP_d; H2Consumption.index=TIMESTAMP_d;
fig=MyStackedPlotly(y_df=production_H2,Conso = H2Consumption)
fig=fig.update_layout(title_text="Production H2 (en MWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline

df=Variables['stockLevel_Pvar'].set_index('YEAR_op').loc[Year_results].pivot(index='TIMESTAMP',columns='STOCK_TECHNO',values='stockLevel_Pvar')
TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
df.index=TIMESTAMP_d
fig=px.line(df)
fig=fig.update_layout(title_text="Niveau des stocks (en MWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig, filename='file.html') ## offline

Variables['carbon_Pvar'].pivot(index="TIMESTAMP",columns='YEAR_op', values='carbon_Pvar').sum()/1000000
#endregion

# def week_avarage(df):
#     df['week']=df.index//168
#     return df.groupby('week').mean()
#
# df1=pd.DataFrame(ResParameters.loc[(2,slice(None),'electricity')]).reset_index().set_index('TIMESTAMP').drop(columns=['RESOURCES','YEAR'])
# df2=pd.DataFrame(availabilityFactor.loc[(2,slice(None),'WindOnShore')])
# df3=pd.DataFrame(availabilityFactor.loc[(2,slice(None),'Solar')])
#
# df11=week_avarage(df1)
# df21=week_avarage(df2)
# df31=week_avarage(df3)
# fig, ax = plt.subplots(2, 1, figsize=(6, 10), sharex=True)
#
# ax[0].plot(df11.index,df11,label='Elec Prices')
# ax[1].plot(df21.index,df21+df31,label='EnR availability')
#
#
# ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
# #plt.savefig('Gestion stockage week')
# plt.show()

def week_avarage(df):
     df['week']=df.index//168
     return df.groupby('week').sum()

prod_elec = Variables['power_Dvar'].loc[Variables['power_Dvar']['TECHNOLOGIES']=='electrolysis']
Capa_elec= Variables['capacity_Pvar'].loc[Variables['capacity_Pvar']['TECHNOLOGIES']=='electrolysis']
prod_enr=availabilityFactor.loc[(slice(None),slice(None),['Solar','WindOnShore'])].reset_index()


prod_elec={y:prod_elec.loc[prod_elec['YEAR_op']==y].pivot(index='TIMESTAMP',columns='TECHNOLOGIES',values='power_Dvar') for y in (2,3,4)}
capa_elec={y:Capa_elec.loc[Capa_elec['YEAR_op']==y]['capacity_Pvar'] for y in (2,3,4)}
prod_enr={y:prod_enr.loc[prod_enr['YEAR']==y].pivot(index='TIMESTAMP',columns='TECHNOLOGIES',values='availabilityFactor') for y in (2,3,4)}


fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
for k,yr in enumerate((2, 3, 4)):
    prod_elec[yr] = prod_elec[yr]/capa_elec[yr].values
    ax[k].plot(prod_elec[yr].index,prod_elec[yr],label='test')
    ax[k].plot(prod_enr[yr].index, prod_enr[yr]['WindOnShore'], label='WindOnShore')
    ax[k].plot(prod_enr[yr].index, prod_enr[yr]['Solar'], label='Solar')
    ax[k].set_ylabel('test (-)')
    # Shrink all axis by 20%

ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
ax[-1].set_xlabel('Week')
plt.xlim([4000,4048])
plt.show()

prod_elec = Variables['power_Dvar'].loc[Variables['power_Dvar']['TECHNOLOGIES']=='electrolysis']
prod_SMR = Variables['power_Dvar'].loc[Variables['power_Dvar']['TECHNOLOGIES']=='SMR_elec']
Capa_elec= Variables['capacity_Pvar'].loc[Variables['capacity_Pvar']['TECHNOLOGIES']=='electrolysis']
prod_wind=Variables['power_Dvar'].loc[Variables['power_Dvar']['TECHNOLOGIES']=='WindOnShore']
prod_PV=Variables['power_Dvar'].loc[Variables['power_Dvar']['TECHNOLOGIES']=='Solar']
res=Variables['importation_Dvar'].loc[Variables['importation_Dvar']['RESOURCES']=='electricity']

prod_elec={y:prod_elec.loc[prod_elec['YEAR_op']==y].pivot(index='TIMESTAMP',columns='TECHNOLOGIES',values='power_Dvar') for y in (2,3,4)}
prod_SMR={y:prod_SMR.loc[prod_SMR['YEAR_op']==y].pivot(index='TIMESTAMP',columns='TECHNOLOGIES',values='power_Dvar') for y in (2,3,4)}
capa_elec={y:Capa_elec.loc[Capa_elec['YEAR_op']==y]['capacity_Pvar'] for y in (2,3,4)}
prod_wind={y:prod_wind.loc[prod_wind['YEAR_op']==y].pivot(index='TIMESTAMP',columns='TECHNOLOGIES',values='power_Dvar') for y in (2,3,4)}
prod_PV={y:prod_PV.loc[prod_PV['YEAR_op']==y].pivot(index='TIMESTAMP',columns='TECHNOLOGIES',values='power_Dvar') for y in (2,3,4)}
res={y:res.loc[res['YEAR_op']==y].pivot(index='TIMESTAMP',columns='RESOURCES',values='importation_Dvar') for y in (2,3,4)}


fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
for k,yr in enumerate((2, 3, 4)):
    prod_elec[yr] = prod_elec[yr]
    ax[k].plot(prod_elec[yr].index,prod_elec[yr]/capa_elec[yr].values,label='electrolyse')
    ax[k].plot(prod_wind[yr].index, (prod_wind[yr].values+prod_PV[yr].values)/capa_elec[yr].values*0.7, label='EnR')
    ax[k].plot(prod_wind[yr].index, (prod_wind[yr].values + prod_PV[yr].values + res[yr].values) / capa_elec[yr].values * 0.7,'--',
               label='EnR+res')
    ax[k].set_ylabel('test (-)')
    # Shrink all axis by 20%

ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
ax[-1].set_xlabel('Week')
plt.xlim([4000,4248])
plt.show()

# prod_elec = Variables['power_Dvar'].loc[Variables['power_Dvar']['TECHNOLOGIES']=='electrolysis']
# Capa_elec= Variables['capacity_Pvar'].loc[(slice(None),['WindOnShore','Solar','electrolysis'])]
# prod_enr=availabilityFactor.loc[(slice(None),slice(None),['Solar','WindOnShore'])].reset_index()
#
#
# prod_elec={y:prod_elec.loc[prod_elec['YEAR_op']==y].pivot(index='TIMESTAMP',columns='TECHNOLOGIES',values='power_Dvar') for y in (2,3,4)}
# capa_elec={y:Capa_elec.loc[Capa_elec['YEAR_op']==y]['capacity_Pvar'] for y in (2,3,4)}
# prod_enr={y:prod_enr.loc[prod_enr['YEAR']==y].pivot(index='TIMESTAMP',columns='TECHNOLOGIES',values='availabilityFactor') for y in (2,3,4)}
#
#
# fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
# for k,yr in enumerate((2, 3, 4)):
#     prod_elec[yr] = prod_elec[yr]/capa_elec[yr].values
#     ax[k].plot(prod_elec[yr].index,prod_elec[yr],label='test')
#     ax[k].plot(prod_enr[yr].index, prod_enr[yr]['WindOnShore'], label='WindOnShore')
#     ax[k].plot(prod_enr[yr].index, prod_enr[yr]['Solar'], label='Solar')
#     ax[k].set_ylabel('test (-)')
#     # Shrink all axis by 20%
#
# ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
# ax[-1].set_xlabel('Week')
# plt.xlim([4000,4048])
# plt.show()


##region Test année 2013
Lagrange=Constraints['energyCtr'].loc[Constraints['energyCtr']['YEAR_op']==2].loc[Constraints['energyCtr']['RESOURCES']=='electricity']
Lagrange['energyCtr'].loc[Lagrange['energyCtr']>1000]=1000
spot=pd.read_csv(InputFolder + 'set2013_horaire_TIMExRES.csv', sep=',', decimal='.', skiprows=0,comment="#").set_index(["TIMESTAMP",'RESOURCES'])
Lagrange['PrixSpot']=list(spot.loc[(slice(None),'electricity'),'importCost'])
Lagrange=Lagrange.set_index('TIMESTAMP').drop(columns=['YEAR_op','RESOURCES'])
fig1=px.line(Lagrange)
plotly.offline.plot(fig1, filename='file.html') ## offline

Carbon=Variables['carbon_Pvar'].loc[Variables['carbon_Pvar']['YEAR_op']==4].drop(columns='YEAR_op').set_index('TIMESTAMP')
RTE=pd.read_csv(InputFolder + 'CO2_RTE.csv', sep=',', decimal='.', skiprows=0,comment="#").set_index("TIMESTAMP")
Carbon['Co2_RTE']=RTE['Co2_RTE']
fig2=px.line(Carbon)
plotly.offline.plot(fig2, filename='file.html') ## offline

#endregion

#endregion

#region Analyse fonctionnement SMR
Zones="PACA"
SimulName=SimulName+'_'+Zones

#Import results
os.chdir(OutputFolder)
os.chdir(SimulName)
v_list = ['capacityInvest_Dvar','transInvest_Dvar','capacity_Pvar','capacityDem_Dvar','capacityDel_Pvar', 'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
         'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar','carbon_Pvar','powerCosts_Pvar','capacityCosts_Pvar','importCosts_Pvar','storageCosts_Pvar','turpeCosts_Pvar','Pmax_Pvar','max_PS_Dvar','carbonCosts_Pvar']
Variables = {v : pd.read_csv(v + '_' + SimulName + '.csv').drop(columns='Unnamed: 0') for v in v_list}
carbon_content=pd.read_csv('carbon_' + SimulName + '.csv')
os.chdir('..')
os.chdir(SimulNameFr)
elec_price=pd.read_csv('elecPrice_' + SimulNameFr + '.csv')
marketPrice=pd.read_csv('marketPrice.csv')
os.chdir('..')
os.chdir(SimulName)

Power=Variables['power_Dvar'].pivot(index=['YEAR_op','TIMESTAMP'],columns='TECHNOLOGIES',values='power_Dvar')[['SMR_class','SMR_class_ex','SMR_CCS1','SMR_CCS2','SMR_elec','SMR_elecCCS1']]
Capa=Variables['capacity_Pvar'].pivot(index='YEAR_op',columns='TECHNOLOGIES',values='capacity_Pvar')[['SMR_class','SMR_class_ex','SMR_CCS1','SMR_CCS2','SMR_elec','SMR_elecCCS1']]

YEAR=list(Power.index.get_level_values('YEAR_op').unique())
TIMESTAMP=list(Power.index.get_level_values('TIMESTAMP').unique())
TECHNO=['SMR_class','SMR_CCS','SMR_elec','SMR_elecCCS']

PWR=pd.DataFrame(Power['SMR_class']+Power['SMR_class_ex']).rename(columns={0:'SMR_class'})
PWR['SMR_CCS']=Power['SMR_CCS1']+Power['SMR_CCS2']
PWR['SMR_elec']=Power['SMR_elec']
PWR['SMR_elecCCS']=Power['SMR_elecCCS1']

CP=pd.DataFrame(Capa['SMR_class']+Capa['SMR_class_ex']).rename(columns={0:'SMR_class'})
CP['SMR_CCS']=Capa['SMR_CCS1']+Capa['SMR_CCS2']
CP['SMR_elec']=Capa['SMR_elec']
CP['SMR_elecCCS']=Capa['SMR_elecCCS1']

Runs={2:{},3:{},4:{}}
Starts={2:{},3:{},4:{}}
Fails={2:{},3:{},4:{}}
Fails_percent={2:{},3:{},4:{}}
GasFails={2:{},3:{},4:{}}
ElecFails={2:{},3:{},4:{}}
GasFails_percent={2:{},3:{},4:{}}
ElecFails_percent={2:{},3:{},4:{}}
Gas_loss={2:{},3:{},4:{}}
Elec_loss={2:{},3:{},4:{}}
Gas_loss_percent={2:{},3:{},4:{}}
Elec_loss_percent={2:{},3:{},4:{}}

for y in YEAR:
    run=0
    start=0
    fail=0
    for tech in TECHNO:
        if CP.loc[y][tech]>0:
            run = PWR.loc[PWR[tech] > 0][tech].loc[(y, slice(None))].count()
            Runs[y][tech]=run
            if tech in ['SMR_class','SMR_elec']:
                temp=pd.DataFrame((PWR.loc[PWR[tech]>0][tech]/CP.loc[y][tech]<0.2).loc[(y,slice(None))])
                fail=temp.loc[temp[tech]==True][tech].count()
                Fails[y][tech]=fail
                temp2=pd.DataFrame((PWR.loc[PWR[tech]>0][tech]-CP.loc[y][tech]*0.2).loc[(y,slice(None))])
                temp2=temp2.loc[temp2[tech]<0]
                if tech=='SMR_class':
                    GasFails[y][tech]=temp2[tech].sum()*(-1.8)
                    GasFails_percent[y][tech]=GasFails[y][tech]/(PWR[tech].sum()*1.8)*100
                else :
                    GasFails[y][tech] = temp2[tech].sum() * (-0.91)
                    GasFails_percent[y][tech] = GasFails[y][tech] / (PWR[tech].sum() * 0.91) * 100
                    ElecFails[y][tech] = temp2[tech].sum() * (-0.75)
                    ElecFails_percent[y][tech] = ElecFails[y][tech] / (PWR[tech].sum() * 0.75) * 100
            if tech in ['SMR_CCS','SMR_elecCCS']:
                temp=pd.DataFrame((PWR.loc[PWR[tech]>0][tech]/CP.loc[y][tech]<0.5).loc[(y,slice(None))])
                fail=temp.loc[temp[tech]==True][tech].count()
                Fails[y][tech]=fail
                temp2=pd.DataFrame((PWR.loc[PWR[tech]>0][tech]-CP.loc[y][tech]*0.5).loc[(y,slice(None))])
                temp2=temp2.loc[temp2[tech]<0]
                if tech == 'SMR_CCS' :
                    GasFails[y][tech]=temp2[tech].sum()*(-1.8)
                    GasFails_percent[y][tech]=GasFails[y][tech]/(PWR[tech].sum()*1.8)*100
                    ElecFails[y][tech]=temp2[tech].sum()*(-0.34)
                    ElecFails_percent[y][tech]=ElecFails[y][tech]/(PWR[tech].sum()*0.34)*100
                else :
                    GasFails[y][tech]=temp2[tech].sum()*(-0.91)
                    GasFails_percent[y][tech]=GasFails[y][tech]/(PWR[tech].sum()*0.91)*100
                    ElecFails[y][tech]=temp2[tech].sum()*(-0.92)
                    ElecFails_percent[y][tech]=ElecFails[y][tech]/(PWR[tech].sum()*0.92)*100
            for t in TIMESTAMP[1:]:
                if (PWR.loc[(y,t-1)][tech]==0 and PWR.loc[(y,t)][tech]>0):
                    start+=1
            Starts[y][tech]=start
            Fails_percent[y][tech]=Fails[y][tech]/Runs[y][tech]*100
            if tech in ['SMR_class','SMR_CCS']:
                Gas_loss[y][tech]=round(Starts[y][tech]*1.05,2)
                Gas_loss_percent[y][tech]=round(Starts[y][tech]*1.05*100/(Starts[y][tech]*1.05+Runs[y][tech]*1.8),2)
            if tech in ['SMR_elec', 'SMR_elecCCS']:
                Elec_loss[y][tech]=round(Starts[y][tech]*0.75,2)
                Elec_loss_percent[y][tech]=round(Starts[y][tech]*0.75/(Starts[y][tech]*0.75+Runs[y][tech]*0.75)*100,2)

Results=pd.DataFrame([[a,b] for a in YEAR for b in TECHNO],columns=['YEAR','TECHNOLOGIES']).set_index(['YEAR','TECHNOLOGIES'])

for y in YEAR:
    for tech in TECHNO:
        if CP.loc[y][tech]>0:
            Results.loc[(y,tech),'Runs']=Runs[y][tech]
            Results.loc[(y, tech), 'Starts'] = Starts[y][tech]
            Results.loc[(y, tech), 'Fails'] = Fails[y][tech]
            Results.loc[(y, tech), 'Fails_percent'] = Fails_percent[y][tech]
            if tech in ['SMR_class','SMR_CCS']:
                Results.loc[(y, tech), 'Gas_loss'] = Gas_loss[y][tech]
                Results.loc[(y, tech), 'Gas_loss_percent'] = Gas_loss_percent[y][tech]
                Results.loc[(y, tech), 'GasFails'] = GasFails[y][tech]
                Results.loc[(y, tech), 'GasFails_percent'] = GasFails_percent[y][tech]
                if tech == 'SMR_CCS':
                    Results.loc[(y, tech), 'ElecFails'] = ElecFails[y][tech]
                    Results.loc[(y, tech), 'ElecFails_percent'] = ElecFails_percent[y][tech]
            if tech in ['SMR_elec', 'SMR_elecCCS']:
                Results.loc[(y, tech), 'Elec_loss'] = Elec_loss[y][tech]
                Results.loc[(y, tech), 'Elec_loss_percent'] = Elec_loss_percent[y][tech]
                Results.loc[(y, tech), 'GasFails'] = GasFails[y][tech]
                Results.loc[(y, tech), 'GasFails_percent'] = GasFails_percent[y][tech]
                Results.loc[(y, tech), 'ElecFails'] = ElecFails[y][tech]
                Results.loc[(y, tech), 'ElecFails_percent'] = ElecFails_percent[y][tech]

Results.to_csv('SMR_analysis_'+SimulName+'.csv',index=True)

#endregion

#region Analyse Prix EnR
Zones="PACA"
SimulName=SimulName+'_'+Zones

#Import results
os.chdir(OutputFolder)
os.chdir(SimulName)
v_list = ['capacityInvest_Dvar','transInvest_Dvar','capacity_Pvar','capacityDem_Dvar','capacityDel_Pvar', 'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
         'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar','carbon_Pvar','powerCosts_Pvar','capacityCosts_Pvar','importCosts_Pvar','storageCosts_Pvar','turpeCosts_Pvar','Pmax_Pvar','max_PS_Dvar','carbonCosts_Pvar']
Variables = {v : pd.read_csv(v + '_' + SimulName + '.csv').drop(columns='Unnamed: 0') for v in v_list}
carbon_content=pd.read_csv('carbon_' + SimulName + '.csv')
os.chdir('..')
os.chdir(SimulNameFr)
elec_price=pd.read_csv('elecPrice_' + SimulNameFr + '.csv')
marketPrice=pd.read_csv('marketPrice.csv')
os.chdir('..')
os.chdir(SimulName)

Power=Variables['power_Dvar'].pivot(index=['YEAR_op','TIMESTAMP'],columns='TECHNOLOGIES',values='power_Dvar')[['Solar','WindOnShore','WindOffShore']].groupby('YEAR_op').sum()
CapaCosts=Variables['capacityCosts_Pvar'].pivot(index='YEAR_op',columns='TECHNOLOGIES',values='capacityCosts_Pvar')[['Solar','WindOnShore','WindOffShore']]
EnR_elecPrice=CapaCosts/Power

EnR_usePrice=Variables['importCosts_Pvar'].loc[Variables['importCosts_Pvar']['RESOURCES']=='electricity'].groupby('YEAR_op').sum()['importCosts_Pvar']/Variables['importation_Dvar'].loc[Variables['importation_Dvar']['RESOURCES']=='electricity'].groupby('YEAR_op').sum()['importation_Dvar']

#endregion

#region Curtailment Analysis

Zones="PACA"
SimulName=SimulName+'_'+Zones

#Import results
os.chdir(OutputFolder)
os.chdir(SimulName)
v_list = ['capacityInvest_Dvar','transInvest_Dvar','capacity_Pvar','capacityDem_Dvar','capacityDel_Pvar', 'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
         'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar','carbon_Pvar','powerCosts_Pvar','capacityCosts_Pvar','importCosts_Pvar','storageCosts_Pvar','turpeCosts_Pvar','Pmax_Pvar','max_PS_Dvar','carbonCosts_Pvar']
Variables = {v : pd.read_csv(v + '_' + SimulName + '.csv').drop(columns='Unnamed: 0') for v in v_list}
carbon_content=pd.read_csv('carbon_' + SimulName + '.csv')
os.chdir('..')
os.chdir(SimulNameFr)
elec_price=pd.read_csv('elecPrice_' + SimulNameFr + '.csv')
marketPrice=pd.read_csv('marketPrice.csv')
os.chdir('..')
os.chdir('..')
os.chdir('..')

ait=pd.read_csv(InputFolder + 'availabilityFactor' + str(year) + '_' + str(Zones) + '_TIMExTECHxYEAR.csv',sep=',', decimal='.', skiprows=0).rename(columns={'YEAR':'YEAR_op'}).pivot(index=['YEAR_op','TIMESTAMP'],columns='TECHNOLOGIES',values='availabilityFactor').rename(index={2030:2,2040:3,2050:4})[['Solar','WindOnShore','WindOffShore']]
Power=Variables['power_Dvar'].pivot(index=['YEAR_op','TIMESTAMP'],columns='TECHNOLOGIES',values='power_Dvar')[['Solar','WindOnShore','WindOffShore']]
Capa=Variables['capacity_Pvar'].pivot(index='YEAR_op',columns='TECHNOLOGIES',values='capacity_Pvar')[['Solar','WindOnShore','WindOffShore']]
maxPower=Capa*ait
Diff=maxPower-Power

Curtailment= {2:Diff.loc[2].sum()/maxPower.loc[2].sum()*100,3:Diff.loc[3].sum()/maxPower.loc[3].sum()*100,4:Diff.loc[4].sum()/maxPower.loc[4].sum()*100}


#endregion

#region Elec analysis

#Import results
os.chdir(OutputFolder)
os.chdir(SimulNameFr)
v_list = ['capacityInvest_Dvar','transInvest_Dvar','capacity_Pvar','capacityDem_Dvar','capacityDel_Pvar', 'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
         'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar','carbon_Pvar','powerCosts_Pvar','capacityCosts_Pvar','importCosts_Pvar','storageCosts_Pvar','turpeCosts_Pvar','Pmax_Pvar','max_PS_Dvar','carbonCosts_Pvar']
Variables = {v : pd.read_csv(v + '_' + SimulNameFr + '.csv').drop(columns='Unnamed: 0') for v in v_list}
carbon_content=pd.read_csv('carbon_' + SimulNameFr + '.csv')
elec_price=pd.read_csv('elecPrice_' + SimulNameFr + '.csv')
marketPrice=pd.read_csv('marketPrice.csv')
PriceCorrection=pd.read_csv('priceCorrection.csv')
os.chdir('..')
os.chdir('..')
os.chdir('..')

marketPrice.set_index(['YEAR_op','TIMESTAMP']).loc[(2030,slice(None))]['NewPrice'].mean()
marketPrice.set_index(['YEAR_op','TIMESTAMP']).loc[(2040,slice(None))]['NewPrice'].mean()
marketPrice.set_index(['YEAR_op','TIMESTAMP']).loc[(2050,slice(None))]['NewPrice'].mean()
PriceCorrection.loc[PriceCorrection['AjustFac']>0]

P=Variables['power_Dvar'].set_index(['YEAR_op','TIMESTAMP','TECHNOLOGIES'])

sum_EnR=P.loc[(slice(None),slice(None),'Solar')]+P.loc[(slice(None),slice(None),'WindOnShore')]+P.loc[(slice(None),slice(None),'WindOffShore')]+P.loc[(slice(None),slice(None),'HydroRiver')]+P.loc[(slice(None),slice(None),'HydroReservoir')]
Dif=areaConsumption.loc[(slice(None),slice(None),'electricity')]['areaConsumption']-sum_EnR['power_Dvar']

marketPrice_year={y:marketPrice.set_index('YEAR_op').rename(index={2030:2,2040:3,2050:4}).set_index('TIMESTAMP',append=True).loc[(y,slice(None))] for y in [2,3,4]}
EnR_marg_wo={y:sum(marketPrice_year[y].loc[marketPrice_year[y]['LastCalled'] == tech]['LastCalled'].count() for tech in ['Solar','WindOnShore','WindOffShore','HydroRiver'])  for y in [2,3,4]}
EnR_marg_w={y:EnR_marg_wo[y]+marketPrice_year[y].loc[marketPrice_year[y]['LastCalled'] == 'HydroReservoir']['LastCalled'].count() for y in [2,3,4]}

marketPrice_year[2].loc[marketPrice_year[2]['NewPrice_NonAct']<50]
marketPrice_year[3].loc[marketPrice_year[3]['NewPrice_NonAct']<50]
marketPrice_year[4].loc[marketPrice_year[4]['NewPrice_NonAct']<50]
#endregion

# #test
# Variables['power_Dvar'].groupby(['YEAR_op','TECHNOLOGIES']).sum().loc[(slice(None),'SMR_class_ex'),'power_Dvar']
# Variables['capacityDem_Dvar'].loc[Variables['capacityDem_Dvar']['capacityDem_Dvar']>0]
# Variables['capacityInvest_Dvar'].pivot(index='YEAR_invest',columns='TECHNOLOGIES',values='capacityInvest_Dvar')[['SMR_class_ex','SMR_CCS1','SMR_CCS2','CCS1','CCS2']]
# Variables['capacity_Pvar'].pivot(index='YEAR_op',columns='TECHNOLOGIES',values='capacity_Pvar')[['SMR_class_ex','SMR_CCS1','SMR_CCS2','CCS1','CCS2']]
# Variables['capacityDem_Dvar'].rename(columns={'YEAR_invest.1':'YEAR_dem'}).pivot(index=['YEAR_invest','YEAR_dem'],columns='TECHNOLOGIES',values='capacityDem_Dvar')[['SMR_class_ex','SMR_CCS1','SMR_CCS2','CCS1','CCS2']]
# Variables['transInvest_Dvar'].loc[Variables['transInvest_Dvar']['transInvest_Dvar']>0]
# Variables['power_Dvar'].groupby(['YEAR_op', 'TECHNOLOGIES']).sum().loc[(slice(None),'cracking'),'power_Dvar']*0.125/1000

