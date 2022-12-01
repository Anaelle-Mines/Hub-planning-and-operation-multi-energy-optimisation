#region Importation of modules
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
import sys
import time
import datetime
#import seaborn as sb

# from Models.MultiRessource.f_Models_TP_Robin import *
# from EnergyAlternativesPlaning.f_tools import *
# from Models.MultiRessource.scenarios_TP import *

from Functions.f_Models_TP_Robin import *
from Functions.f_optimization import *
from Basic_functionalities.scenarios_TP import *



#endregion

#region Solver and data location definition
outputPath='Data/output/'
solver= 'mosek' ## no need for solverpath with mosek.
#inputPath = 'Models/MultiRessource/Data/'
inputPath='Data/Raw_TP/'
#endregion

#region Heat Model
#Dates = pd.DataFrame.from_dict({"ind" : range(1,8761),"Dates": availabilityFactor.reset_index().Date.unique()}).set_index(["ind"])
#Calend=pd.read_csv(inputPath + 'H2Demand_TIME.csv', sep=',', decimal='.', skiprows=0, comment="#").rename(columns={'TIMESTAMP': "Dates"})
#Calend.loc[:,"Dates"] = list(Dates.loc[list(Calend.Dates),"Dates"])
#Calend.to_csv(inputPath+'H2Demand_TIME.csv',index=False)

scenario=Scenario_Heat(2030,inputPath=inputPath)

print('Building model...')
Parameters = loadScenario(scenario, False)
model = systemModel_MultiResource_WithStorage(Parameters)
#start_clock = time.time()
print('Calculating model...')
opt = SolverFactory(solver)
results = opt.solve(model)
#end_clock = time.time()
#print('Computational time: {:.0f} s'.format(end_clock - start_clock))

res_HEAT = {
    'variables': getVariables_panda(model),
    'constraints': getConstraintsDual_panda(model)
}
#endregion

#region Analysis Heat results

TECHNO_heat=list(scenario['conversionTechs'].transpose().loc[scenario['conversionTechs'].transpose()['Category']=='Heat production'].index)
TECHNO_elec=list(scenario['conversionTechs'].transpose().loc[scenario['conversionTechs'].transpose()['Category']=='Electricity production'].index)

Power=res_HEAT['variables']['power_Dvar'].set_index(['Date','TECHNOLOGIES'])
PowerPercent_heat=res_HEAT['variables']['power_Dvar'].groupby('TECHNOLOGIES').sum().loc[TECHNO_heat,'power_Dvar']/res_HEAT['variables']['power_Dvar'].groupby('TECHNOLOGIES').sum().loc[TECHNO_heat,'power_Dvar'].sum()
PowerPercent_elec=res_HEAT['variables']['power_Dvar'].groupby('TECHNOLOGIES').sum().loc[TECHNO_elec,'power_Dvar']/res_HEAT['variables']['power_Dvar'].groupby('TECHNOLOGIES').sum().loc[TECHNO_elec,'power_Dvar'].sum()
PowerPercent=pd.concat([PowerPercent_heat,PowerPercent_elec]).reset_index().rename(columns={'power_Dvar':'powerPercent'}).set_index('TECHNOLOGIES')
Import=res_HEAT['variables']['importation_Dvar'].loc[res_HEAT['variables']['importation_Dvar']['RESOURCES']=='electricity'].rename(columns={'RESOURCES':'TECHNOLOGIES','importation_Dvar':'power_Dvar'}).set_index(['Date','TECHNOLOGIES']).rename(index={'electricity':'Importation'})
Stock=res_HEAT['variables']['storageOut_Dvar'].groupby(['Date','RESOURCES']).sum().reset_index().rename(columns={'RESOURCES':'TECHNOLOGIES','storageOut_Dvar':'power_Dvar'}).set_index(['Date','TECHNOLOGIES']).rename(index={'electricity':'Battery','heat':'Tank'})-res_HEAT['variables']['storageIn_Dvar'].groupby(['Date','RESOURCES']).sum().reset_index().rename(columns={'RESOURCES':'TECHNOLOGIES','storageIn_Dvar':'power_Dvar'}).set_index(['Date','TECHNOLOGIES']).rename(index={'electricity':'Battery','heat':'Tank'})
df_power=pd.concat([Power,Import,Stock])

Power_heat=df_power.loc[(slice(None),TECHNO_heat+['Tank']),'power_Dvar'].reset_index()
fig1=px.area(Power_heat,x='Date',y='power_Dvar',color='TECHNOLOGIES',title='Heat production')
fig1 = fig1.update_layout(yaxis_title="Heat production (MWh)")
plotly.offline.plot(fig1, filename='Heat production.html')

Power_elec=df_power.loc[(slice(None),TECHNO_elec+['Importation','Battery']),'power_Dvar'].reset_index()
fig2=px.area(Power_elec,x='Date',y='power_Dvar',color='TECHNOLOGIES',title='Electricity production')
fig2 = fig2.update_layout(yaxis_title="Electricity production (MWh)")
plotly.offline.plot(fig2, filename='Electricity production.html')

Capacity=res_HEAT['variables']['capacityCosts_Pvar'].set_index('TECHNOLOGIES').rename(columns={'capacityCosts_Pvar':'Costs'})
Capacity['Type']='Fixed Costs'
Capacity['Costs']=Capacity['Costs']/res_HEAT['variables']['power_Dvar'].groupby('TECHNOLOGIES').sum()['power_Dvar']
Variable=res_HEAT['variables']['powerCosts_Pvar'].set_index('TECHNOLOGIES').rename(columns={'powerCosts_Pvar':'Costs'})
Variable['Type']='Variable Costs'
Variable['Costs']=Variable['Costs']/res_HEAT['variables']['power_Dvar'].groupby('TECHNOLOGIES').sum()['power_Dvar']
Import=Capacity.copy()
Import['Type']='Importation'
Import['Costs']=0
Import.loc[TECHNO_heat,'Costs']=PowerPercent.loc[TECHNO_heat,'powerPercent']*res_HEAT['variables']['importation_Dvar']['importation_Dvar'].sum()
Import['Costs']=Import['Costs']/res_HEAT['variables']['power_Dvar'].groupby('TECHNOLOGIES').sum()['power_Dvar']
TURPE=Capacity.copy()
TURPE['Type']='TURPE'
TURPE['Costs']=0
TURPE.loc[TECHNO_heat,'Costs']=PowerPercent.loc[TECHNO_heat,'powerPercent']*res_HEAT['variables']['turpeCosts_Pvar']['turpeCosts_Pvar'].sum()
TURPE['Costs']=TURPE['Costs']/res_HEAT['variables']['power_Dvar'].groupby('TECHNOLOGIES').sum()['power_Dvar']
Storage=Capacity.copy()
Storage['Type']='Storage'
Storage['Costs']=0
Storage.loc[TECHNO_heat,'Costs']=PowerPercent.loc[TECHNO_heat,'powerPercent']*res_HEAT['variables']['storageCosts_Pvar'].set_index('STOCK_TECHNO').loc['Tank']['storageCosts_Pvar']
Storage.loc[TECHNO_elec,'Costs']=PowerPercent.loc[TECHNO_elec,'powerPercent']*res_HEAT['variables']['storageCosts_Pvar'].set_index('STOCK_TECHNO').loc['battery']['storageCosts_Pvar']
Storage['Costs']=Storage['Costs']/res_HEAT['variables']['power_Dvar'].groupby('TECHNOLOGIES').sum()['power_Dvar']

df_costs=pd.concat([Capacity,Variable,Import,TURPE,Storage]).reset_index()
fig3 = px.bar(df_costs, x="TECHNOLOGIES", y="Costs", color="Type", title="Conversion technologies costs")
fig3 = fig3.update_layout(yaxis_title="LCOE (MWh)")
plotly.offline.plot(fig3, filename='Costs.html')

#endregion

#region H2 model
scenario=Scenario_H2(2030,inputPath=inputPath)

print('Building model...')
Parameters = loadScenario(scenario, False)
model = systemModel_MultiResource_WithStorage(Parameters)
start_clock = time.time()
print('Calculating model...')
opt = SolverFactory(solver)
results = opt.solve(model)
end_clock = time.time()
print('Computational time: {:.0f} s'.format(end_clock - start_clock))

res_H2 = {
    'variables': getVariables_panda(model),
    'constraints': getConstraintsDual_panda(model)
}

#endregion

#region Analysis H2 results

TECHNO_H2=list(scenario['conversionTechs'].transpose().loc[scenario['conversionTechs'].transpose()['Category']=='H2 production'].index)
TECHNO_elec=list(scenario['conversionTechs'].transpose().loc[scenario['conversionTechs'].transpose()['Category']=='Electricity production'].index)

Power=res_H2['variables']['power_Dvar'].set_index(['Date','TECHNOLOGIES'])
PowerPercent_H2=res_H2['variables']['power_Dvar'].groupby('TECHNOLOGIES').sum().loc[TECHNO_H2,'power_Dvar']/res_H2['variables']['power_Dvar'].groupby('TECHNOLOGIES').sum().loc[TECHNO_H2,'power_Dvar'].sum()
PowerPercent_elec=res_H2['variables']['power_Dvar'].groupby('TECHNOLOGIES').sum().loc[TECHNO_elec,'power_Dvar']/res_H2['variables']['power_Dvar'].groupby('TECHNOLOGIES').sum().loc[TECHNO_elec,'power_Dvar'].sum()
PowerPercent=pd.concat([PowerPercent_H2,PowerPercent_elec]).reset_index().rename(columns={'power_Dvar':'powerPercent'}).set_index('TECHNOLOGIES')
Import=res_H2['variables']['importation_Dvar'].loc[res_H2['variables']['importation_Dvar']['RESOURCES']=='electricity'].rename(columns={'RESOURCES':'TECHNOLOGIES','importation_Dvar':'power_Dvar'}).set_index(['Date','TECHNOLOGIES']).rename(index={'electricity':'Importation'})
Stock=res_H2['variables']['storageOut_Dvar'].groupby(['Date','RESOURCES']).sum().reset_index().rename(columns={'RESOURCES':'TECHNOLOGIES','storageOut_Dvar':'power_Dvar'}).set_index(['Date','TECHNOLOGIES']).rename(index={'electricity':'Battery','hydrogen':'TankH2'})-res_H2['variables']['storageIn_Dvar'].groupby(['Date','RESOURCES']).sum().reset_index().rename(columns={'RESOURCES':'TECHNOLOGIES','storageIn_Dvar':'power_Dvar'}).set_index(['Date','TECHNOLOGIES']).rename(index={'electricity':'Battery','hydrogen':'TankH2'})
df_power=pd.concat([Power,Import,Stock])

Power_H2=df_power.loc[(slice(None),TECHNO_H2+['TankH2']),'power_Dvar'].reset_index()
fig1=px.area(Power_H2,x='Date',y='power_Dvar',color='TECHNOLOGIES',title='H2 production')
fig1 = fig1.update_layout(yaxis_title="H2 production (MWh)")
plotly.offline.plot(fig1, filename='H2 production.html')

Power_elec=df_power.loc[(slice(None),TECHNO_elec+['Importation','Battery']),'power_Dvar'].reset_index()
fig2=px.area(Power_elec,x='Date',y='power_Dvar',color='TECHNOLOGIES',title='Electricity production')
fig2 = fig2.update_layout(yaxis_title="Electricity production (MWh)")
plotly.offline.plot(fig2, filename='Electricity production.html')

Capacity=res_H2['variables']['capacityCosts_Pvar'].set_index('TECHNOLOGIES').rename(columns={'capacityCosts_Pvar':'Costs'})
Capacity['Type']='Fixed Costs'
Capacity['Costs']=Capacity['Costs']/res_H2['variables']['power_Dvar'].groupby('TECHNOLOGIES').sum()['power_Dvar']
Variable=res_H2['variables']['powerCosts_Pvar'].set_index('TECHNOLOGIES').rename(columns={'powerCosts_Pvar':'Costs'})
Variable['Type']='Variable Costs'
Variable['Costs']=Variable['Costs']/res_H2['variables']['power_Dvar'].groupby('TECHNOLOGIES').sum()['power_Dvar']
Import=Capacity.copy()
Import['Type']='Importation'
Import['Costs']=0
Import.loc[TECHNO_H2,'Costs']=PowerPercent.loc[TECHNO_H2,'powerPercent']*res_H2['variables']['importation_Dvar']['importation_Dvar'].sum()
Import['Costs']=Import['Costs']/res_H2['variables']['power_Dvar'].groupby('TECHNOLOGIES').sum()['power_Dvar']
TURPE=Capacity.copy()
TURPE['Type']='TURPE'
TURPE['Costs']=0
TURPE.loc[TECHNO_H2,'Costs']=PowerPercent.loc[TECHNO_H2,'powerPercent']*res_H2['variables']['turpeCosts_Pvar']['turpeCosts_Pvar'].sum()
TURPE['Costs']=TURPE['Costs']/res_H2['variables']['power_Dvar'].groupby('TECHNOLOGIES').sum()['power_Dvar']
Storage=Capacity.copy()
Storage['Type']='Storage'
Storage['Costs']=0
Storage.loc[TECHNO_H2,'Costs']=PowerPercent.loc[TECHNO_H2,'powerPercent']*res_H2['variables']['storageCosts_Pvar'].set_index('STOCK_TECHNO').loc['TankH2']['storageCosts_Pvar']
Storage.loc[TECHNO_elec,'Costs']=PowerPercent.loc[TECHNO_elec,'powerPercent']*res_H2['variables']['storageCosts_Pvar'].set_index('STOCK_TECHNO').loc['battery']['storageCosts_Pvar']
Storage['Costs']=Storage['Costs']/res_H2['variables']['power_Dvar'].groupby('TECHNOLOGIES').sum()['power_Dvar']

df_costs=pd.concat([Capacity,Variable,Import,TURPE,Storage]).reset_index()
fig3 = px.bar(df_costs, x="TECHNOLOGIES", y="Costs", color="Type", title="Conversion technologies costs")
fig3 = fig3.update_layout(yaxis_title="LCOE (MWh)")
plotly.offline.plot(fig3, filename='Costs.html')

#endregion