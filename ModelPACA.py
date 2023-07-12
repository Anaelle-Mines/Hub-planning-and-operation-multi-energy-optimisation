
import os

import numpy as np
import pandas as pd
import csv
import datetime
import copy
import time
from pyomo.environ import value

from Functions.f_multiResourceModels import *
from Functions.f_optimization import *
from scenario_creation import scenarioDict
# from scenario_creation_REsensibility import scenarioDict_RE

outputPath='Data/output/'
solver= 'mosek' ## no need for solverpath with mosek.

ScenarioList=['Ref'] #['EnR', 'SmrOnly','conv_SmrOnly', 'Grid', 'Re_x2', 'Re_inf', 'BM_100','BM_80','Cavern','CavernREinf','woSMR_2030', 'woSMR_2040', 'woSMR_2050', 'CO2_10', 'CO2_100','TC120']

for ScenarioName in ScenarioList :

    outputFolder=outputPath+ScenarioName
    outputFolderFr=outputPath+'Ref_Fr'

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
        'constraints': getConstraintsDual_panda(model),
    }

    print(str(results.Problem))
    print('Valeur de la fonction objectif = ',res['variables']['objective_Pvar'].sum())

    ### Check sum Prod = Consumption
    year_results=2030
    production_df=res['variables']['energy_Pvar'].set_index('YEAR_op').loc[year_results].pivot(index="TIMESTAMP",columns='RESOURCES', values='energy_Pvar')
    areaConsumption_df=scenarioDict[ScenarioName]['resourceDemand'].set_index('YEAR').loc[year_results]
    Delta=(production_df.sum(axis=0) - areaConsumption_df.sum(axis=0))
    abs(Delta).max()
    print("Vérification équilibre O/D : \n",Delta)
    print("Production par énergie (TWh) : \n",production_df.sum(axis=0)/10**6) ### Energy that is getting out of the system (doesn't take into account the enrgy that is produced and then consumed by the capacities of the model)


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

