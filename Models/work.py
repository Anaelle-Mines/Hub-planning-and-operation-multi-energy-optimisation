import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
os.sys.path.append(r'../')
from Functions.f_multiResourceModels import *
from Functions.f_optimization import *
from scenario_creation import scenarioDict
from scenario_creation_REsensibility import scenarioDict_RE
from scenarios_ref_Fr import scenarioFr

scenarioDict.update(scenarioDict_RE)
# print(scenarioDict.keys())


# import1=pd.read_csv('../Data/output/Ref_Fr_5/capacity_Pvar.csv').set_index(['YEAR_op','TECHNOLOGIES'])
# import2=pd.read_csv('../Data/output/Ref_Fr/capacity_Pvar.csv').set_index(['YEAR_op','TECHNOLOGIES'])
# import1=pd.read_csv('../Data/output/Ref_Fr_oldBGPrice/Pmax_Pvar.csv').set_index(['YEAR_op','STOCK_TECHNO'])
# import2=pd.read_csv('../Data/output/Ref_Fr/Pmax_Pvar.csv').set_index(['YEAR_op','STOCK_TECHNO'])

import1=pd.read_csv('../Data/output/Ref_Fr_oldBGPrice/importation_Dvar.csv').set_index(['YEAR_op','RESOURCES'])
import2=pd.read_csv('../Data/output/Ref_Fr/importation_Dvar.csv').set_index(['YEAR_op','RESOURCES'])

importCosts1=pd.read_csv('../Data/raw/set2020_horaire_TIMExRES.csv').set_index(['YEAR_op','RESOURCES'])
importCosts2=pd.read_csv('../Data/output/Ref_Fr/importCosts_Pvar.csv').set_index(['YEAR_op','RESOURCES'])


import3=pd.read_csv('../Data/output/Ref_Fr_oldBGPrice/power_Dvar.csv').set_index(['YEAR_op','TECHNOLOGIES'])
import4=pd.read_csv('../Data/output/Ref_Fr/power_Dvar.csv').set_index(['YEAR_op','TECHNOLOGIES'])



# print(import1.loc[import1['capacity_Pvar']>0].loc[(2020,slice(None))].sort_index())
# print(import2.loc[import2['capacity_Pvar']>0].loc[(2020,slice(None))].sort_index())


# print(import1.groupby(['YEAR_op','RESOURCES']).sum())
# print(import2.groupby(['YEAR_op','RESOURCES']).sum())

print(import3.groupby(['YEAR_op','TECHNOLOGIES']).sum().loc[(slice(2020,2030),slice(None))])
print(import4.groupby(['YEAR_op','TECHNOLOGIES']).sum().loc[(slice(2020,2030),slice(None))])




pd.set_option('display.max_columns', 500)
# print(loadScenario(scenarioFr)['storageParameters'])