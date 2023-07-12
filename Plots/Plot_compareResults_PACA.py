
import os
import numpy as np
import pandas as pd
import csv

from Functions.f_graphicTools import *
from Basic_functionalities.scenario_creation import scenarioDict
from Basic_functionalities.scenario_creation_REsensibility import scenarioDict_RE

outputPath='Data/output/'

# plot costs comparison

ScenarioName='Ref'
outputFolder=outputPath+ScenarioName+'_Base_PACA'
df=extract_costs(scenarioDict[ScenarioName],outputFolder)
df['AEL']=df.pop('Electrolysis')

ScenarioName='ElecOnly'
outputFolder=outputPath+ScenarioName+'_var4_PACA'
df2=extract_costs(scenarioDict[ScenarioName],outputFolder)
df2.pop('SMR')
df2['AEL only']=df2.pop('Electrolysis')

df.update(df2)

plot_costs(df,outputPath,comparaison=True)

# comparaison wo SMR in 2050

ScenarioName='Ref'
outputFolder=outputPath+ScenarioName+'_Base_PACA'
df=extract_costs(scenarioDict[ScenarioName],outputFolder)
df['AEL']=df.pop('Electrolysis')

ScenarioName='Ref'
outputFolder=outputPath+ScenarioName+'_var6woSMR_PACA'
df2=extract_costs(scenarioDict[ScenarioName],outputFolder)
df2['SMR wo SMR']=df2.pop('SMR')
df2['AEL wo SMR']=df2.pop('Electrolysis')


df.update(df2)

plot_costs(df,outputPath,comparaison=True)

# comparaison EnR*2

ScenarioName='Ref'
outputFolder=outputPath+ScenarioName+'_Base_PACA'
df=extract_costs(scenarioDict[ScenarioName],outputFolder)
df['AEL']=df.pop('Electrolysis')

ScenarioName='Ref'
outputFolder=outputPath+ScenarioName+'_var2_PACA'
df2=extract_costs(scenarioDict[ScenarioName],outputFolder)
df2['SMR EnRx2']=df2.pop('SMR')
df2['AEL EnRx2']=df2.pop('Electrolysis')


df.update(df2)

plot_costs(df,outputPath,comparaison=True)

# comparaison EnR inf

ScenarioName='Ref'
outputFolder=outputPath+ScenarioName+'_Base_PACA'
df=extract_costs(scenarioDict[ScenarioName],outputFolder)
df['AEL']=df.pop('Electrolysis')

ScenarioName='Ref'
outputFolder=outputPath+ScenarioName+'_EnR_inf_PACA'
df2=extract_costs(scenarioDict[ScenarioName],outputFolder)
df2['SMR EnR inf']=df2.pop('SMR')
df2['AEL EnR inf']=df2.pop('Electrolysis')

df.update(df2)

plot_costs(df,outputPath,comparaison=True)

# comparaison Biomethane price

ScenarioName='BM_80'
outputFolder=outputPath+ScenarioName+'_PACA'
df=extract_costs(scenarioDict[ScenarioName],outputFolder)
df['AEL BM=80€']=df.pop('AEL')
df['SMR BM=80€']=df.pop('SMR')


for l in df.keys() : df[l]=df[l].reset_index().loc[df[l].reset_index().YEAR==2050].set_index('YEAR')

ScenarioName='Ref'
outputFolder=outputPath+ScenarioName+'_PACA'
df2=extract_costs(scenarioDict[ScenarioName],outputFolder)
df2['AEL BM=90€']=df2.pop('AEL')
df2['SMR BM=90€']=df2.pop('SMR')


for l in df2.keys() : df2[l]=df2[l].reset_index().loc[df2[l].reset_index().YEAR==2050].set_index('YEAR')

df.update(df2)

ScenarioName='BM_100'
outputFolder=outputPath+ScenarioName+'_PACA'
df3=extract_costs(scenarioDict[ScenarioName],outputFolder)
df3['AEL BM=100€']=df3.pop('AEL')
df3['SMR BM=100€']=df3.pop('SMR')


for l in df3.keys() : df3[l]=df3[l].reset_index().loc[df3[l].reset_index().YEAR==2050].set_index('YEAR')

df.update(df3)

plot_costs2050(df,outputPath,comparaison=True)

# comparaison wo biogas

ScenarioName='Ref'
outputFolder=outputPath+ScenarioName+'_Base_PACA'
df=extract_costs(scenarioDict[ScenarioName],outputFolder)
df['AEL']=df.pop('AEL')

ScenarioName='Ref'
outputFolder=outputPath+ScenarioName+'_var6last_PACA'
df2=extract_costs(scenarioDict[ScenarioName],outputFolder)
df2['SMR wo Bm']=df2.pop('SMR')
df2['AEL wo Bm']=df2.pop('AEL')

df.update(df2)

plot_costs(df,outputPath,comparaison=True)

# comparaison EnR 100

ScenarioName='Ref'
outputFolder=outputPath+ScenarioName+'_Base_PACA'
df=extract_costs(scenarioDict[ScenarioName],outputFolder)
df['AEL']=df.pop('Electrolysis')

ScenarioName='EnR'
outputFolder=outputPath+ScenarioName+'_Base_PACA'
df2=extract_costs(scenarioDict[ScenarioName],outputFolder)
df2['SMR 100']=df2.pop('SMR')
df2['AEL 100']=df2.pop('Electrolysis')

df.update(df2)

plot_costs(df,outputPath,comparaison=True)

# comparaison woSMR2050 et woSMR2040
ScenarioName='Ref'
outputFolder=outputPath+ScenarioName+'_var6woSMR_PACA'
df=extract_costs(scenarioDict[ScenarioName],outputFolder)
df['SMR 50']=df.pop('SMR')
df['AEL 50']=df.pop('Electrolysis')

ScenarioName='Ref'
outputFolder=outputPath+ScenarioName+'_var6woSMR2040_PACA'
df2=extract_costs(scenarioDict[ScenarioName],outputFolder)
df2['SMR 40']=df2.pop('SMR')
df2['AEL 40']=df2.pop('Electrolysis')

df.update(df2)

plot_costs(df,outputPath,comparaison=True)

# comparaison Grid
ScenarioName='Ref'
outputFolder=outputPath+ScenarioName+'_Base_PACA'
df=extract_costs(scenarioDict[ScenarioName],outputFolder)
df['AEL']=df.pop('Electrolysis')

ScenarioName='Grid'
outputFolder=outputPath+ScenarioName+'_Base_PACA'
df2=extract_costs(scenarioDict[ScenarioName],outputFolder)
df2['SMR 50']=df2.pop('SMR')
df2['AEL 50']=df2.pop('Electrolysis')

df.update(df2)

plot_costs(df,outputPath,comparaison=True)

# comparaison all scenarios

dico_costs={}
scenarioList=['Ref', 'BM_100','woSMR_2030', 'woSMR_2040', 'woSMR_2050', 'CO2_10', 'CO2_100','Re_x2', 'Re_inf',]
scenarioNames=['Ref','BM=100€/MWh','wo SMR from 2030','wo SMR from 2040','wo SMR from 2050','CO$_2$=10€/t','CO$_2$=100€/t','Renewables x2','Renewables \nunbounded']

for s in scenarioList:
    outputFolder=outputPath+s+'_PACA'
    dico_costs[s]=extract_costs(scenarioDict[s],outputFolder)

plot_carbonCosts(dico_costs,scenarioNames)

# comparaison SMR flex

dico_costs={}
scenarioList=['Ref','woSMR_2030', 'woSMR_2040', 'woSMR_2050']
scenarioNames=['Ref','wo SMR from 2030','wo SMR from 2040','wo SMR from 2050']
for s in scenarioList:
    outputFolder=outputPath+s+'_PACA'
    dico_costs[s]=extract_costs(scenarioDict[s],outputFolder)

# plot_compare_elecPrice(dico_costs,scenarioNames)
plot_carbonCosts(dico_costs,scenarioNames)

# comparaison energy and carbon

# dico_ener={}
# scenarioList=['CO2_10','Ref','CO2_100']
# scenarioNames=['CO$_2$=10€/t$_{captured}$','CO$_2$=50€/t$_{captured}$','CO$_2$=100€/t$_{captured}$']

dico_ener={}
scenarioList=['BM_80','Ref','BM_100']
scenarioNames=['BM=80€/MWh$_{CH4}$','BM=90€/MWh$_{CH4}$','BM=100€/MWh$_{CH4}$']

# dico_ener={}
# scenarioList=['woSMR_2030', 'woSMR_2040', 'woSMR_2050']
# scenarioNames=['wo SMR from 2030','wo SMR from 2040','wo SMR from 2050']

for s in scenarioList:
    outputFolder=outputPath+s+'_PACA'
    dico_ener[s]=extract_energy(scenarioDict[s],outputFolder)

plot_compare_energy_and_carbon(dico_ener,scenarioNames)

# comparaison energy and costs

dico_ener={}
scenarioList=['Ref','Re_x2', 'Re_inf'] #['Ref', 'EnR', 'SmrOnly', 'Grid', 'Re_x2', 'Re_inf', 'BM_100','Manosque','woSMR_2030', 'woSMR_2040', 'woSMR_2050', 'CO2_10', 'CO2_100','Free']
scenarioNames=['Reference','Renewables x2','Renewables \nunbounded']
for s in scenarioList:
    outputFolder=outputPath+s+'_PACA'
    dico_ener[s]=extract_energy(scenarioDict[s],outputFolder)

plot_compare_energy_and_costs(dico_ener,scenarioNames)

#Comparison of annual CO2 emissions

scenarios=['Ref', 'EnR', 'Grid', 'Re_x2', 'Re_inf','Manosque', 'BM_100','SmrOnly','woSMR_2030', 'woSMR_2040', 'woSMR_2050', 'CO2_10', 'CO2_100']
labels=['Ref (N1 75% RE)','MO 100% RE','N3 50% RE','Renewables x2','Renewables unbounded','with salt caverns','BM=100€/MWh','SMR only','wo SMR from 2030','wo SMR from 2040','wo SMR from 2050','CO$_2$ treat.=50€/t','CO$_2$ treat.=100€/t']

plot_annual_co2_emissions(scenarios,labels)

#test Robin

dico_costs={}
scenarioList=['Ref', 'CO2_10', 'CO2_100','BM_100','BM_80','woSMR_2030', 'woSMR_2040', 'woSMR_2050','conv_SmrOnly','SmrOnly', 'Re_x2', 'Re_inf', 'Caverns','CavernREinf']
scenarioNames=['Ref (N1 75% RE)','CO$_2$ treat.=10€/t','CO$_2$ treat.=100€/t','BM=100€/MWh','BM=80€/MWh','wo SMR from 2030','wo SMR from 2040','wo SMR from 2050',' Conventional SMR only','SMR only','Renewables x2','Renewables unbounded','with salt caverns', 'with caverns & unbounded RE'] #'M0 100% RE','N3 50% RE'
for s in scenarioList:
    outputFolder=outputPath+s+'_PACA'
    dico_costs[s]=extract_costs(scenarioDict[s],outputFolder)

plot_total_co2_emissions_and_costs(dico_costs,scenarioNames)


# Sensibility RE
outputPath=outputPath+'Sensibility_RE/'

# SMR flexibility value
scenarioList=list(scenarioDict_RE.keys())
del scenarioList[0]
scenarioNames=['Re_inf','Re_inf_TC200','Re_inf_CCS10','Caverns','Caverns_TC200','Caverns_CCS10','CavernREinf','CavernREinf_TC200','CavernREinf_CCS10']
labels=['Unbounded','Unbounded TC=200€/t','Unbounded CCS=10€/t','Caverns','Caverns TC=200€/t','Caverns CCS=10€/t','Unbounded + Caverns','Unbounded \n+ Caverns TC=200€/t','Unbounded \n+ Caverns CCS=10€/t']

dico_costs={}
for s in scenarioList:
    outputFolder=outputPath+s+'_PACA'
    dico_costs[s]=extract_costs(scenarioDict_RE[s],outputFolder)

flex=plot_total_co2_emissions_and_flexSMR(dico_costs,scenarioNames,labels,outputPath=outputPath)

# Total Carbon and costs
outputPath=outputPath+'Sensibility_RE/'
scenarioDict=scenarioDict_RE

scenarioList=['Re_inf','Re_inf_TC200','Re_inf_CCS10','Caverns','Caverns_TC200','Caverns_CCS10','CavernREinf','CavernREinf_TC200','CavernREinf_CCS10']
scenarioNames=['Re_inf','Re_inf_TC200','Re_inf_CCS10','Caverns','Caverns_TC200','Caverns_CCS10','CavernREinf','CavernREinf_TC200','CavernREinf_CCS10']

dico_costs={}
for s in scenarioList:
    outputFolder=outputPath+s+'_PACA'
    dico_costs[s]=extract_costs(scenarioDict[s],outputFolder)

plot_total_co2_emissions_and_costs(dico_costs,scenarioNames,outputPath=outputPath)

