
import os
import numpy as np
import pandas as pd
import csv
import datetime
import time
import datetime

from Functions.f_multiResourceModels import *
from Functions.f_optimization import *
from Functions.f_InputScenario import *
#from Basic_functionalities.scenarios_ref_Fr import scenarioFr
# from Basic_functionalities.scenarios_100_Fr import scenarioFr100
from Basic_functionalities.scenarios_ref_woConstraint_Fr import scenarioFr
from Basic_functionalities.scenarios_2060_Fr import scenarioFr2060

#region Solver and data location definition
outputPath='Data/output/'
solver= 'mosek' ## no need for solverpath with mosek.
#endregion

ScenarioName='Ref' # Possible Choice : 'Ref', 'eSMR', 'EnR', 'Grid', 'GN', 'BG', 'EnR+','Crack'
outputFolderFr=outputPath+ScenarioName+'_test2060_Fr'

#pd.set_option('display.max_columns', 500)

#region Electricity data creation (France scenario)

print('Building model France...')
model = systemModel_MultiResource_WithStorage(scenarioFr2060,'Fr',isAbstract=False)

start_clock = time.time()
print('Calculating model France...')
opt = SolverFactory(solver)
results = opt.solve(model)
end_clock = time.time()
print('Computational time: {:.0f} s'.format(end_clock - start_clock))

res = {
    'variables': getVariables_panda(model),
    'constraints': getConstraintsDual_panda(model)
}


try:
    os.mkdir(outputFolderFr)
except:
    pass

for k, v in res['variables'].items():
    v.to_csv(outputFolderFr +'/' + k + '.csv',index=True)

Prix_elec = round(res['constraints']['energyCtr'].set_index('RESOURCES').loc['electricity'].set_index(['YEAR_op','TIMESTAMP']),2)
Carbon = res['variables']['carbon_Pvar'].set_index(['YEAR_op', 'TIMESTAMP'])
Carbon.loc[Carbon['carbon_Pvar'] < 0.01] = 0
Prod_elec = res['variables']['energy_Pvar'].loc[res['variables']['energy_Pvar']['RESOURCES']=='electricity'].groupby(['YEAR_op', 'TIMESTAMP']).sum()
Carbon_content = Carbon['carbon_Pvar'] / Prod_elec['energy_Pvar']
Carbon_content = round(Carbon_content.reset_index().set_index('YEAR_op').rename(columns={0: 'carbonContent'}).set_index('TIMESTAMP', append=True))
Prix_elec.to_csv(outputFolderFr +'/elecPrice.csv',index=True)

Carbon_content.to_csv(outputFolderFr +'/carbon.csv',index=True)

marketPrice,elec_prod=ElecPrice_optim(scenarioFr2060,IntercoOut=100,solver='mosek',outputFolder = outputFolderFr,testOnly=True)

# AdjustFac,marketPrice,elec_var,elec_prod=ElecPrice_optim(scenarioFr,IntercoOut=100,solver='mosek',outputFolder = outputFolderFr,testOnly=False)
#
# OldDifference = elec_var['OldRevenus'].set_index(['YEAR_op','TECHNOLOGIES'])['OldRevenus'] - elec_var['TotalCosts'].set_index(['YEAR_op','TECHNOLOGIES'])['TotalCosts']
# MW = elec_prod.groupby(['YEAR_op', 'TECHNOLOGIES']).sum()['power_Dvar']
# MW = MW.loc[(MW != 0)]
# raport1 = OldDifference / MW
# print(raport1)
#
# NewDifference = elec_var['Revenus'].set_index(['YEAR_op','TECHNOLOGIES'])['Revenus'] - elec_var['TotalCosts'].set_index(['YEAR_op','TECHNOLOGIES'])['TotalCosts']
# raport2 = NewDifference / MW
# print(raport2)

# marketPrice.loc[(2020,slice(None)),'OldPrice_NonAct'].mean()
# marketPrice.loc[marketPrice['OldPrice_NonAct']<50] .loc[(2020,slice(None)),'OldPrice_NonAct']
# marketPrice.loc[(2020,slice(None)),'NewPrice_NonAct'].mean()
# marketPrice.loc[marketPrice['NewPrice_NonAct']<50] .loc[(2020,slice(None)),'NewPrice_NonAct']

#endregion

