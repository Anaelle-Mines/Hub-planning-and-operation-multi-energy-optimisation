#region Importation of modules
import os
if os.path.basename(os.getcwd())=="BasicFunctionalities":
    os.chdir('..') ## to work at project root  like in any IDE
import sys
if sys.platform != 'win32':
    myhost = os.uname()[1]
else : myhost = ""
if (myhost=="jupyter-sop"):
    ## for https://jupyter-sop.mines-paristech.fr/ users, you need to
    #  (1) run the following in a terminal
    if (os.system("/opt/mosek/9.2/tools/platform/linux64x86/bin/lmgrd -c /opt/mosek/9.2/tools/platform/linux64x86/bin/mosek.lic -l lmgrd.log")==0):
        os.system("/opt/mosek/9.2/tools/platform/linux64x86/bin/lmutil lmstat -c 27007@127.0.0.1 -a")
    #  (2) definition of license
    os.environ["MOSEKLM_LICENSE_FILE"] = '@jupyter-sop'

import numpy as np
import pandas as pd
import csv
#import docplex
import datetime
import copy
import plotly
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn import linear_model
import sys
import time
import datetime
import seaborn as sb

from Functions.f_multiResourceModels import *
from Functions.f_optimization import *
from Functions.f_optimModel_elec import *
from Functions.f_InputScenario import *
from Basic_functionalities.scenarios_ref_Fr import scenarioFr

#endregion

#region Solver and data location definition
outputPath='Data/output/'
solver= 'mosek' ## no need for solverpath with mosek.
#endregion

outputFolder=outputPath+'Ref_Fr'
scenario=scenarioFr

inputDict = loadScenario(scenario, False)

elecProd = pd.read_csv(outputFolder + '/power_Dvar.csv').drop(columns='Unnamed: 0').set_index(
    ['YEAR_op', 'TIMESTAMP', 'TECHNOLOGIES'])
carbon_content = pd.read_csv(outputFolder + '/carbon.csv')
elec_price = pd.read_csv(outputFolder + '/elecPrice.csv')

marketPrice = elec_price.set_index(['YEAR_op', 'TIMESTAMP'])
marketPrice['LastCalled'] = ""

for i in marketPrice.index:
    if elecProd.loc[(i[0], i[1], 'Coal_p')]['power_Dvar'] > 0:
        marketPrice.LastCalled.loc[i] = 'Coal_p'
    elif elecProd.loc[(i[0], i[1], 'TAC')]['power_Dvar'] > 0:
        marketPrice.LastCalled.loc[i] = 'TAC'
    elif elecProd.loc[(i[0], i[1], 'CCG')]['power_Dvar'] > 0:
        marketPrice.LastCalled.loc[i] = 'CCG'
    elif elecProd.loc[(i[0], i[1], 'OldNuke')]['power_Dvar'] > 0:
        marketPrice.LastCalled.loc[i] = 'OldNuke'
    elif elecProd.loc[(i[0], i[1], 'NewNuke')]['power_Dvar'] > 0:
        marketPrice.LastCalled.loc[i] = 'NewNuke'
    elif elecProd.loc[(i[0], i[1], 'WindOffShore')]['power_Dvar'] > 0:
        marketPrice.LastCalled.loc[i] = 'WindOffShore'
    elif elecProd.loc[(i[0], i[1], 'WindOnShore')]['power_Dvar'] > 0:
        marketPrice.LastCalled.loc[i] = 'WindOnShore'
    elif elecProd.loc[(i[0], i[1], 'Solar')]['power_Dvar'] > 0:
        marketPrice.LastCalled.loc[i] = 'Solar'
    else:
        marketPrice.LastCalled.loc[i] = 'Undetermined'

capaCosts = pd.read_csv(outputFolder + '/capacityCosts_Pvar.csv').drop(columns='Unnamed: 0').set_index(
    ['YEAR_op', 'TECHNOLOGIES'])
carbonContent = carbon_content.set_index(['YEAR_op', 'TIMESTAMP'])
ResParameters = inputDict['resParameters'].loc[
    (slice(None), slice(None), ['electricity', 'gaz', 'hydrogen', 'uranium'])].reset_index().rename(
    columns={'YEAR': 'YEAR_op'}).set_index(['YEAR_op', 'TIMESTAMP', 'RESOURCES'])
gazPrice = (pd.DataFrame(pd.read_csv(outputFolder + '/importCosts_Pvar.csv').drop(columns='Unnamed: 0').set_index(
    ['YEAR_op', 'RESOURCES']).loc[(slice(None), ['gazBio', 'gazNat']), 'importCosts_Pvar']).fillna(0).groupby(
    'YEAR_op').sum()).join(pd.DataFrame(
    pd.read_csv(outputFolder + '/importation_Dvar.csv').groupby(['YEAR_op', 'RESOURCES']).sum().drop(
        columns=['Unnamed: 0', 'TIMESTAMP']).loc[(slice(None), ['gazBio', 'gazNat']), 'importation_Dvar']).fillna(
    0).groupby('YEAR_op').sum())
gazPrice['gazPrice'] = (gazPrice['importCosts_Pvar'] / gazPrice['importation_Dvar']).fillna(0)
for yr in [2020, 2030, 2040, 2050]: ResParameters.loc[(yr, slice(None), ['gaz']), 'importCost'] = gazPrice.loc[yr][
    'gazPrice']

model = GetElectricPriceModel(elecProd, marketPrice, ResParameters, inputDict['techParameters'], capaCosts,
                              carbonContent, inputDict['conversionFactor'], inputDict['carbonTax'], isAbstract=False)
opt = SolverFactory(solver)
results = opt.solve(model)
elec_var = getVariables_panda(model)

AjustFac = elec_var['AjustFac'].set_index(['YEAR_op', 'TECHNOLOGIES']).fillna(0)
AjustFac.loc[AjustFac['AjustFac'] < 0] = 0
NewPrice = []
for i in marketPrice.index:
    Ajustement = AjustFac.loc[i[0], marketPrice.LastCalled.loc[i]]
    NewPrice.append(marketPrice.loc[i]['energyCtr'] + Ajustement['AjustFac'])

marketPrice['NewPrice'] = NewPrice

#region test

test = marketPrice.NewPrice == marketPrice.energyCtr
print(test.loc[test == False])

# test='energyCtr'
test = 'NewPrice'

TECHNO = list(elecProd.index.get_level_values('TECHNOLOGIES').unique())
TIMESTAMP = list(elecProd.index.get_level_values('TIMESTAMP').unique())
RES = list(ResParameters.index.get_level_values('RESOURCES').unique())
RES.remove('hydrogen')
RES.remove('electricity')
YEAR = sorted(list(elecProd.index.get_level_values('YEAR_op').unique()))
dy = YEAR[1] - YEAR[0]
elecProd['Revenus'] = elecProd['power_Dvar'] * marketPrice[test]
Revenus = elecProd.Revenus.groupby(['YEAR_op', 'TECHNOLOGIES']).sum()
TotalCosts = elecProd.groupby(['YEAR_op', 'TECHNOLOGIES']).sum().drop(columns=['power_Dvar', 'Revenus'])

for tech in TECHNO:
    df = pd.DataFrame({'YEAR_op': [2020] * 8760 + [2030] * 8760 + [2040] * 8760 + [2050] * 8760,
                       'TIMESTAMP': TIMESTAMP + TIMESTAMP + TIMESTAMP + TIMESTAMP}).set_index(['YEAR_op', 'TIMESTAMP'])
    for res in RES:
        df[res] = elecProd['power_Dvar'].loc[(slice(None), slice(None), tech)] * ResParameters['importCost'].loc[
            (slice(None), slice(None), res)] * (-inputDict['conversionFactor']['conversionFactor'].loc[(res, tech)])
    for y in YEAR:
        TotalCosts.loc[(y, tech), 'import'] = df.groupby('YEAR_op').sum().sum(axis=1).loc[y]

for y in YEAR:
    for tech in TECHNO:
        TotalCosts.loc[(y, tech), 'variable'] = elecProd['power_Dvar'].groupby(['YEAR_op', 'TECHNOLOGIES']).sum().loc[
                                                    (y, tech)] * inputDict['techParameters']['powerCost'].loc[
                                                    (y - dy, tech)]
        TotalCosts.loc[(y, tech), 'carbon'] = elecProd['power_Dvar'].groupby(['YEAR_op', 'TECHNOLOGIES']).sum().loc[
                                                  (y, tech)] * inputDict['carbonTax']['carbonTax'].loc[y]

TotalCosts['capacity'] = capaCosts['capacityCosts_Pvar']
TotalCosts['total'] = TotalCosts['import'] + TotalCosts['variable'] + TotalCosts['carbon'] + TotalCosts['capacity']
Difference = TotalCosts['total'] - Revenus
delta = Difference.loc[(Difference != 0)]
MW = elecProd.groupby(['YEAR_op', 'TECHNOLOGIES']).sum()['power_Dvar']
MW = MW.loc[(MW != 0)]
raport = -delta / MW
print(elec_var['AjustFac'].loc[elec_var['AjustFac']['AjustFac'] > 0])
print(raport)

marketPrice.loc[marketPrice['NewPrice'] < 0] = 0

for yr in [2020, 2030, 2040, 2050]:
    marketPrice.loc[(yr, slice(None)), 'OldPrice_NonAct'] = marketPrice.loc[(yr, slice(None)), 'energyCtr'] / (
                (1 + scenario['economicParameters']['discountRate'].loc[0]) ** (-10 * (yr - dy)))
    marketPrice.loc[(yr, slice(None)), 'NewPrice_NonAct'] = marketPrice.loc[(yr, slice(None)), 'NewPrice'] / (
                (1 + scenario['economicParameters']['discountRate'].loc[0]) ** (-10 * (yr - dy)))

marketPrice = round(marketPrice.reset_index().set_index(['YEAR_op', 'TIMESTAMP']), 2)

#endregion