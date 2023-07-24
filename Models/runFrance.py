import os
import numpy as np
import pandas as pd
import csv
import datetime
import time
import datetime

os.sys.path.append(r'../')

from Functions.f_multiResourceModels import *
from Functions.f_optimization import *
from scenarios_ref_Fr import scenarioFr

#region Solver and data location definition
outputPath='../Data/output/'
solver= 'mosek' ## no need for solverpath with mosek.
#endregion

ScenarioName='Ref'
outputFolderFr=outputPath+ScenarioName+'_Fr_5'

# region Electricity data creation (France scenario)

print('Building model France...')
model = systemModel_MultiResource_WithStorage(scenarioFr,'Fr',isAbstract=False)

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

elecPrice = round(res['constraints']['energyCtr'].set_index('RESOURCES').loc['electricity'].set_index(['YEAR_op','TIMESTAMP']),2)
Carbon = res['variables']['carbon_Pvar'].set_index(['YEAR_op', 'TIMESTAMP'])
Carbon.loc[Carbon['carbon_Pvar'] < 0.01] = 0
Prod_elec = res['variables']['energy_Pvar'].loc[res['variables']['energy_Pvar']['RESOURCES']=='electricity'].groupby(['YEAR_op', 'TIMESTAMP']).sum()
Carbon_content = Carbon['carbon_Pvar'] / Prod_elec['energy_Pvar']
Carbon_content = round(Carbon_content.reset_index().set_index('YEAR_op').rename(columns={0: 'carbonContent'}).set_index('TIMESTAMP', append=True))

elecPrice.to_csv(outputFolderFr +'/elecPrice.csv',index=True)
Carbon_content.to_csv(outputFolderFr +'/carbon.csv',index=True)
marketPrice,elec_prod=ElecPrice_optim(scenarioFr,IntercoOut=50,solver='mosek',outputFolder = outputFolderFr)


#endregion

