
import os

import numpy as np
import pandas as pd
import csv
import datetime
import copy
import time

from Functions.f_multiResourceModels import *
from Functions.f_optimization import *
from Functions.f_InputScenario import *
from Basic_functionalities.scenario_creation import scenarioDict

outputPath='Data/output/'
solver= 'mosek' ## no need for solverpath with mosek.

ScenarioName='test' # Possible Choice : 'Ref', 'ElecOnly','SmrOnly','Grid','EnR',
outputFolder=outputPath+ScenarioName+'_CO2_100_PACA'
# outputFolderFr=outputPath+ScenarioName+'_Fr'
outputFolderFr=outputPath+'Ref_wH2_Fr'

#pd.set_option('display.max_columns', 500)

#region PACA calculation

print('Building model PACA...')
model = systemModel_MultiResource_WithStorage(scenarioDict[ScenarioName],'PACA',isAbstract=False)

start_clock = time.time()
print('Calculating model PACA...')
opt = SolverFactory(solver)
results = opt.solve(model)
end_clock = time.time()
print('Computational time: {:.0f} s'.format(end_clock - start_clock))

res = {
    'variables': getVariables_panda(model),
    'constraints': getConstraintsDual_panda(model)
}


### Check sum Prod = Consumption
year_results=2020
production_df=res['variables']['energy_Pvar'].set_index('YEAR_op').loc[year_results].pivot(index="TIMESTAMP",columns='RESOURCES', values='energy_Pvar')
areaConsumption_df=scenarioDict[ScenarioName]['resourceDemand'].set_index('YEAR').loc[year_results]
Delta=(production_df.sum(axis=0) - areaConsumption_df.sum(axis=0))
abs(Delta).max()
print("Vérification équilibre O/D : \n",Delta)
print("Production par énergie (TWh) : \n",production_df.sum(axis=0)/10**6) ### energies produites TWh (ne comprends pas ce qui est consommé par le système)


### save results
var_name=list(res['variables'].keys())
cons_name=list(res['constraints'].keys())


try:
    os.mkdir(outputFolder)
except:
    pass

for k, v in res['variables'].items():
    v.to_csv(outputFolder +'/' + k + '.csv',index=True)

#endregion

importation=pd.read_csv(outputFolder+'/importation_Dvar.csv').groupby(['YEAR_op','RESOURCES']).sum()
localEnR=pd.read_csv(outputFolder+'/power_Dvar.csv').groupby(['YEAR_op','TECHNOLOGIES']).sum().loc[(slice(None),['WindOnShore','WindOffShore_flot','Solar']),'power_Dvar'].reset_index().groupby('YEAR_op').sum()
