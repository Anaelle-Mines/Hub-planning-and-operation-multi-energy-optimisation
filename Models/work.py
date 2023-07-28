import os
import numpy as np
import pandas as pd
import csv
os.sys.path.append(r'../')
from Functions.f_multiResourceModels import *
from Functions.f_optimization import *
from scenario_creation import scenarioDict
from scenario_creation_REsensibility import scenarioDict_RE
from scenarios_ref_Fr import scenarioFr

scenarioDict.update(scenarioDict_RE)
# print(scenarioDict.keys())


# import1=pd.read_csv('../Data/output/Ref_Fr_new/capacity_Pvar.csv').set_index(['YEAR_op','TECHNOLOGIES'])
# import2=pd.read_csv('../Data/output/Ref_Fr/capacity_Pvar.csv').set_index(['YEAR_op','TECHNOLOGIES'])
import1=pd.read_csv('../Data/output/Ref_Fr_new/Cmax_Pvar.csv').set_index(['YEAR_op','STOCK_TECHNO'])
import2=pd.read_csv('../Data/output/Ref_Fr/Cmax_Pvar.csv').set_index(['YEAR_op','STOCK_TECHNO'])


print(import1.loc[import1['Cmax_Pvar']>0].loc[(2040,slice(None))].sort_index())
print(import2.loc[import2['Cmax_Pvar']>0].loc[(2040,slice(None))].sort_index())

pd.set_option('display.max_columns', 500)
print(loadScenario(scenarioFr)['storageParameters'])