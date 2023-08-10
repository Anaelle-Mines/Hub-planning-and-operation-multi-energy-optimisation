
import os
import numpy as np
import pandas as pd
import csv
import datetime
import copy
import time
from pyomo.environ import value
os.sys.path.append(r'../')
from Functions.f_multiResourceModels import *
from Functions.f_optimization import *
from scenario_creation import scenarioDict
from scenario_creation_REsensibility import scenarioDict_RE

scenarioDict.update(scenarioDict_RE)

outputPath='../Data/output/'
solver= 'mosek' ## no need for solverpath with mosek.

scenarioList = os.sys.argv[1:]


for scenarioName in scenarioList :

    outputFolder=outputPath+scenarioName+'_PACA_test'
    # outputFolderFr=outputPath+'Ref_Fr'

    #region PACA calculation

    # print(scenarioDict[scenarioName]['resourceImportPrices']['gazBio'].unique())

    print('Building model PACA...')
    model = systemModel_MultiResource_WithStorage(scenarioDict[scenarioName],'PACA',isAbstract=False)

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
    print('Valeur de la fonction objectif = ',res['variables']['objective_Pvar'].sum()['objective_Pvar'])

    ### Check sum Prod = Consumption
    year_results=2030
    production_df=res['variables']['energy_Pvar'].set_index('YEAR_op').loc[year_results].pivot(index="TIMESTAMP",columns='RESOURCES', values='energy_Pvar')
    areaConsumption_df=scenarioDict[scenarioName]['resourceDemand'].set_index('YEAR').loc[year_results]
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

