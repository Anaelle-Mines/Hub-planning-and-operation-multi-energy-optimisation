
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
from Functions.f_InputScenario import *
# from Basic_functionalities.scenario_creation import scenarioDict
# from Basic_functionalities.scenario_creation_REsensibility import scenarioDict_RE
from Basic_functionalities.scenarios_ISGT import *
# pd.set_option('display.max_columns', 500)

outputPath='Data/output/'
solver= 'mosek' ## no need for solverpath with mosek.

# ScenarioList= ['Export']#['Ref', 'EnR', 'SmrOnly','conv_SmrOnly', 'Grid', 'Re_x2', 'Re_inf', 'BM_100','BM_80','Cavern','CavernREinf','woSMR_2030', 'woSMR_2040', 'woSMR_2050', 'CO2_10', 'CO2_100','Free','TC120'] #'test2060', 'test',Export
# ScenarioList=[ 'Caverns_CCS10', 'Caverns_CCS10_woSMR', 'CavernREinf_CCS10', 'CavernREinf_CCS10_woSMR'] #'Re_inf_CCS10', 'Re_inf_CCS10_woSMR',
# scenarioDict=scenarioDict_RE
# ScenarioList=[ 'scenario2_tdemi', 'scenario2_tdouble']#['scenario1','scenario2','scenario3','scenario4','scenario3_d10', 'scenario4_d10', 'scenario3_d50', 'scenario4_d50', 'scenario3_d250', 'scenario4_d250', 'scenario3_d500', 'scenario4_d500', 'scenario1_tdemi', 'scenario1_tdouble', 'scenario2_tdemi', 'scenario2_tdouble', 'scenario3_tdemi', 'scenario3_tdouble', 'scenario4_tdemi', 'scenario4_tdouble']
# scenarioDict=scenarioDict_ISGT
ScenarioList=['Ref','Re_x2', 'Re_inf']

for ScenarioName in ScenarioList :

    outputFolder=outputPath+'ISGT/'+ScenarioName
    outputFolderFr=outputPath+'Ref_wH2_Fr'

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

    # importation=pd.read_csv(outputFolder+'/importation_Dvar.csv').groupby(['YEAR_op','RESOURCES']).sum()
    # localEnR=pd.read_csv(outputFolder+'/power_Dvar.csv').groupby(['YEAR_op','TECHNOLOGIES']).sum().loc[(slice(None),['WindOnShore','WindOffShore_flot','Solar']),'power_Dvar'].reset_index().groupby('YEAR_op').sum()
