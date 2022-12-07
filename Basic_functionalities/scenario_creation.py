
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
from Basic_functionalities.tech_eco_data import *

outputPath='Data/output/'
solver= 'mosek' ## no need for solverpath with mosek.

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

Scenario_list={'EnR':{'MixElec':'moins','CAPEX_EnR': 'moins','CAPEX_eSMR': 'plus','CAPEX_CCS': 'ref','CAPEX_elec': 'moins','Biogaz_price': 'ref','Gaznat_price': 'ref','CarbonTax' :'ref','Local_RE':'ref'},
               'Grid':{'MixElec':'plus','CAPEX_EnR': 'plus','CAPEX_eSMR': 'plus','CAPEX_CCS': 'ref','CAPEX_elec': 'moins','Biogaz_price': 'ref','Gaznat_price': 'ref','CarbonTax' :'ref','Local_RE':'ref'},
               'GN':{'MixElec':'ref','CAPEX_EnR': 'plus','CAPEX_eSMR': 'ref','CAPEX_CCS': 'moins','CAPEX_elec': 'ref','Biogaz_price': 'plus','Gaznat_price': 'moins','CarbonTax' :'moins','Local_RE':'ref'},
               'BG':{'MixElec':'ref','CAPEX_EnR': 'plus','CAPEX_eSMR': 'ref','CAPEX_CCS': 'plus','CAPEX_elec': 'plus','Biogaz_price': 'moins','Gaznat_price': 'plus','CarbonTax' :'plus','Local_RE':'ref'}}


scenarioDict={'Ref':scenarioPACA}

# Scenario EnR
scenarioEnR=scenarioPACA.copy()

tech = 'Solar'
capex=np.zeros([len(scenarioEnR['yearList'][:-1])])
opex=np.zeros([len(scenarioEnR['yearList'][:-1])])
lifespan=np.zeros([len(scenarioEnR['yearList'][:-1])])
max_install_capacity = [0, 200, 200, 200]
max_cumul_capacity = [0, 300, 300, 300]

for k,year in enumerate(scenarioEnR['yearList'][:-1]):
    capex[k], opex[k], lifespan[k] = get_capex_new_tech_RTE(tech, hyp='low', year=year, var=None)

scenarioEnR['conversionTechs'][tech].loc['investCost']=capex
