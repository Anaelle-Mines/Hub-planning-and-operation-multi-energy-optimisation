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
from Basic_functionalities.scenarios_ref_PACA import scenarioPACA

outputPath='Data/output/'
solver= 'mosek' ## no need for solverpath with mosek.

ScenarioName='Ref' # Possible Choice : 'Ref', 'eSMR', 'EnR', 'Grid', 'GN', 'BG', 'EnR+','Crack'
outputFolder=outputPath+ScenarioName
outputFolderFr=outputPath+ScenarioName+'_Fr'

#region Descritpion of scenario

Param_list={'MixElec': {'plus':'100%','ref':'75%','moins':'50%'} ,
            'CAPEX_EnR': {'plus':0.9,'ref':0.75,'moins':0.6},
            'CAPEX_eSMR': {'plus':1000,'ref':850,'moins':700},
            'CAPEX_CCS': {'plus':0.9,'ref':0.75,'moins':0.5},
            'CAPEX_elec': {'plus':0.7,'ref':0.55,'moins':0.4},
            'BlackC_price' : {'plus':138,'ref':0,'moins':0},
            'Biogaz_price': {'plus':120,'ref':100,'moins':80},
            'Gaznat_price': {'plus':5,'ref':3,'moins':1},
            'CarbonTax' : {'plus':200,'ref':165,'moins':130},
            'Local_RE':{'plus':[300,300,2000],'ref':[150,150,1000],'moins':[150,150,1000]}}

Scenario_list={'Ref':{'MixElec':'ref','CAPEX_EnR': 'ref','CAPEX_eSMR': 'ref','CAPEX_CCS': 'ref','CAPEX_elec': 'ref','Biogaz_price': 'ref','Gaznat_price': 'ref','CarbonTax' :'ref','Local_RE':'ref','BlackC_price':'ref'},
               'eSMR':{'MixElec':'ref','CAPEX_EnR': 'ref','CAPEX_eSMR': 'moins','CAPEX_CCS': 'ref','CAPEX_elec': 'plus','Biogaz_price': 'ref','Gaznat_price': 'moins','CarbonTax' :'plus','Local_RE':'ref','BlackC_price':'ref'},
               'EnR':{'MixElec':'moins','CAPEX_EnR': 'moins','CAPEX_eSMR': 'plus','CAPEX_CCS': 'ref','CAPEX_elec': 'moins','Biogaz_price': 'ref','Gaznat_price': 'ref','CarbonTax' :'ref','Local_RE':'ref','BlackC_price':'ref'},
               'EnR+':{'MixElec':'moins','CAPEX_EnR': 'moins','CAPEX_eSMR': 'plus','CAPEX_CCS': 'ref','CAPEX_elec': 'moins','Biogaz_price': 'ref','Gaznat_price': 'ref','CarbonTax' :'ref','Local_RE':'plus','BlackC_price':'ref'},
               'Grid':{'MixElec':'plus','CAPEX_EnR': 'plus','CAPEX_eSMR': 'plus','CAPEX_CCS': 'ref','CAPEX_elec': 'moins','Biogaz_price': 'ref','Gaznat_price': 'ref','CarbonTax' :'ref','Local_RE':'ref','BlackC_price':'ref'},
               'GN':{'MixElec':'ref','CAPEX_EnR': 'plus','CAPEX_eSMR': 'ref','CAPEX_CCS': 'moins','CAPEX_elec': 'ref','Biogaz_price': 'plus','Gaznat_price': 'moins','CarbonTax' :'moins','Local_RE':'ref','BlackC_price':'ref'},
               'BG':{'MixElec':'ref','CAPEX_EnR': 'plus','CAPEX_eSMR': 'ref','CAPEX_CCS': 'plus','CAPEX_elec': 'plus','Biogaz_price': 'moins','Gaznat_price': 'plus','CarbonTax' :'plus','Local_RE':'ref','BlackC_price':'ref'},
               'Crack':{'MixElec':'ref','CAPEX_EnR': 'ref','CAPEX_eSMR': 'ref','CAPEX_CCS': 'ref','CAPEX_elec': 'ref','Biogaz_price': 'ref','Gaznat_price': 'ref','CarbonTax' :'ref','Local_RE':'ref','BlackC_price':'plus'}}

ElecMix= {'100%':{'Solar':[52,100,130],'WindOnShore':[45,70,95],'WindOffShore':[5,35,60],'OldNuke':[54,30,0],'NewNuke':[0,0,0],'HydroRiver':[15,15,15],'HydroReservoir':[15,15,15],'Coal_p':[6,0,0],'TAC':[10,5,0],'CCG':[7,5,5],'Interco':[13,26,39],'curtailment':[10,10,10],'Battery':[10,10,30],'STEP':[5,3,2]},
          '75%':{'Solar':[45,55,75],'WindOnShore':[40,52,70],'WindOffShore':[6,15,45],'OldNuke':[54,45,15],'NewNuke':[0,5,13],'HydroRiver':[15,15,15],'HydroReservoir':[15,15,15],'Coal_p':[6,0,0],'TAC':[10,5,0],'CCG':[7,10,17],'Interco':[13,26,39],'curtailment':[10,20,30],'Battery':[10,10,30],'STEP':[5,3,2]},
          '50%':{'Solar':[40,40,40],'WindOnShore':[45,45,45],'WindOffShore':[5,10,25],'OldNuke':[54,49,29],'NewNuke':[0,5,25],'HydroRiver':[15,15,15],'HydroReservoir':[15,15,15],'Coal_p':[6,0,0],'TAC':[10,5,0],'CCG':[7,10,17],'Interco':[13,26,39],'curtailment':[10,20,30],'Battery':[10,10,30],'STEP':[5,3,2]}}

#endregion


#pd.set_option('display.max_columns', 500)

#region PACA scenario

print('Building model PACA...')
model = systemModel_MultiResource_WithStorage(scenarioPACA,isAbstract=False)

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
areaConsumption_df=scenarioPACA['resourceDemand'].set_index('YEAR').loc[year_results]
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
