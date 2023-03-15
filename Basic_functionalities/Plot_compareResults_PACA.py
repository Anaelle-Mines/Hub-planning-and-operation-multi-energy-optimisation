
import os
import numpy as np
import pandas as pd
import csv

from Functions.f_graphicTools import *
from Basic_functionalities.scenario_creation import scenarioDict

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

# comparaison Biogas*1.25

ScenarioName='Ref'
outputFolder=outputPath+ScenarioName+'_Base_PACA'
df=extract_costs(scenarioDict[ScenarioName],outputFolder)
df['SMR Ref']=df.pop('SMR')
df['AEL Ref']=df.pop('Electrolysis')

for l in df.keys() : df[l]=df[l].reset_index().loc[df[l].reset_index().YEAR==2050].set_index('YEAR')

ScenarioName='Ref'
outputFolder=outputPath+ScenarioName+'_Var_BM_100_PACA'
df2=extract_costs(scenarioDict[ScenarioName],outputFolder)
df2['SMR Var_BM_100']=df2.pop('SMR')
df2['AEL Var_BM_100']=df2.pop('Electrolysis')

for l in df2.keys() : df2[l]=df2[l].reset_index().loc[df2[l].reset_index().YEAR==2050].set_index('YEAR')

df.update(df2)

plot_costs(df,outputPath,comparaison=True)

# comparaison wo biogas

ScenarioName='Ref'
outputFolder=outputPath+ScenarioName+'_Base_PACA'
df=extract_costs(scenarioDict[ScenarioName],outputFolder)
df['AEL']=df.pop('Electrolysis')

ScenarioName='Ref'
outputFolder=outputPath+ScenarioName+'_var6last_PACA'
df2=extract_costs(scenarioDict[ScenarioName],outputFolder)
df2['SMR wo Bm']=df2.pop('SMR')
df2['AEL wo Bm']=df2.pop('Electrolysis')

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
scenarioList=[('Ref','Base'),('Ref','Var_BM_100'),('ElecOnly','Var_woSMR_2030'),('Ref','Var_woSMR_2040'),('Ref','Var_woSMR_2050'),('Ref','Var_CO2_10'),('Ref','Var_CO2_100'),('Ref','Var_RE_x2'),('Ref','Var_RE_inf')]
scenarioNames=['Ref','Var_BM_100','Var_woSMR_2030','Var_woSMR_2040','Var_woSMR_2050','Var_CO2_10','Var_CO2_100','Var_RE_x2','Var_RE_inf']

for s,k in scenarioList:
    outputFolder=outputPath+s+'_'+k+'_PACA'
    dico_costs[s+'_'+k]=extract_costs(scenarioDict[s],outputFolder)

plot_carbonCosts(dico_costs,scenarioNames)

# comparaison electrolysis

dico_costs={}
scenarioList=[('Ref','Base'),('ElecOnly','Var_woSMR_2030'),('Ref','Var_woSMR_2040'),('Ref','Var_woSMR_2050')]
scenarioNames=['Ref','Var_woSMR_2030','Var_woSMR_2040','Var_woSMR_2050']
for s,k in scenarioList:
    outputFolder=outputPath+s+'_'+k+'_PACA'
    dico_costs[s+'_'+k]=extract_costs(scenarioDict[s],outputFolder)

plot_compare_elecPrice(dico_costs,scenarioNames)
plot_carbonCosts(dico_costs,scenarioNames)

# comparaison energy

dico_ener={}
scenarioList=[('Ref','Base'),('Ref','Var_CO2_10'),('Ref','Var_CO2_100')]
scenarioNames=['$CO_2=50€/t$','$CO_2=10€/t$','$CO_2=1000€/t$']
# scenarioList=[('Ref','Base'),('Ref','Var_RE_x2'),('Ref','Var_RE_inf')]
# scenarioNames=['Ref','Var_RE_x2','Var_RE_inf']
for s,k in scenarioList:
    outputFolder=outputPath+s+'_'+k+'_PACA'
    dico_ener[s+'_'+k]=extract_energy(scenarioDict[s],outputFolder)

plot_compare_test(dico_ener,scenarioNames)


# comparaison energy and costs

dico_ener={}
scenarioList=[('Ref','Base'),('Ref','Var_RE_x2'),('Ref','Var_RE_inf')]
scenarioNames=['Ref','Var_RE_x2','Var_RE_inf']
for s,k in scenarioList:
    outputFolder=outputPath+s+'_'+k+'_PACA'
    dico_ener[s+'_'+k]=extract_energy(scenarioDict[s],outputFolder)

plot_compare_energy_costs(dico_ener,scenarioNames)
