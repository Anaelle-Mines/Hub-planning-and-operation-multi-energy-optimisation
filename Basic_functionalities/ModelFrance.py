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
from Basic_functionalities.scenarios_ref_Fr import scenarioFr

#region Solver and data location definition
outputPath='Data/output/'
solver= 'mosek' ## no need for solverpath with mosek.
#endregion

ScenarioName='Ref' # Possible Choice : 'Ref', 'eSMR', 'EnR', 'Grid', 'GN', 'BG', 'EnR+','Crack'
outputFolderFr=outputPath+ScenarioName+'_Fr'

#pd.set_option('display.max_columns', 500)

#region Electricity data creation (France scenario)

print('Building model France...')
model = systemModel_MultiResource_WithStorage(scenarioFr,isAbstract=False)

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

marketPrice=ElecPrice_optim(scenarioFr,solver='mosek',outputFolder = outputFolderFr)

#endregion

