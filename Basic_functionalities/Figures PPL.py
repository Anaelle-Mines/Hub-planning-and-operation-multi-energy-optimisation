#region importation of modules
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
from Functions.f_graphicalTools import *
from Functions.f_optimModel_elec import *
from Functions.f_InputScenario import *

# Change this if you have other solvers obtained here
## https://ampl.com/products/solvers/open-source/
## for eduction this site provides also several professional solvers, that are more efficient than e.g. cbc
#endregion

#region Solver and data location definition

#region Choice of Scenario
Scenario='BG' # Possible Choice : 'Ref', 'eSMR', 'EnR', 'Grid', 'GN', 'BG', 'EnR+','Crack'
ScenarioName='BG_v4'
ScenarioNameFr='BG_v4'
#endregion

InputFolder='Data/Input/Input_'+ScenarioName+'/'
OutputFolder='Data/output/'
Simul_date='2022-10-19'
SimulName=Simul_date+'_'+ScenarioName
DataCreation_date='2022-10-19'
SimulNameFr=DataCreation_date+'_'+ScenarioNameFr+'_Fr'


solver= 'mosek' ## no need for solverpath with mosek.
BaseSolverPath='/Users/robin.girard/Documents/Code/Packages/solvers/ampl_macosx64'
sys.path.append(BaseSolverPath)

solvers= ['gurobi','knitro','cbc'] # 'glpk' is too slow 'cplex' and 'xpress' do not work
solverpath= {}
for solver in solvers : solverpath[solver]=BaseSolverPath+'/'+solver
cplexPATH='/Applications/CPLEX_Studio1210/cplex/bin/x86-64_osx'
sys.path.append(cplexPATH)
solverpath['cplex']=cplexPATH+"/"+"cplex"
solver = 'mosek'
#endregion

#region Tracé mix prod H2 et EnR

#region import données
Zones='PACA'
SimulName=SimulName+'_'+Zones

#Import results
os.chdir(OutputFolder)
os.chdir(SimulName)
v_list = ['capacityInvest_Dvar','transInvest_Dvar','capacity_Pvar','capacityDem_Dvar','capacityDel_Pvar', 'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
         'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar','carbon_Pvar','powerCosts_Pvar','capacityCosts_Pvar','importCosts_Pvar','storageCosts_Pvar','turpeCosts_Pvar','Pmax_Pvar','max_PS_Dvar','carbonCosts_Pvar']
Variables = {v : pd.read_csv(v + '_' + SimulName + '.csv').drop(columns='Unnamed: 0') for v in v_list}
carbon_content=pd.read_csv('carbon_' + SimulName + '.csv')
os.chdir('..')
os.chdir(SimulNameFr)
elec_price=pd.read_csv('elecPrice_' + SimulNameFr + '.csv')
marketPrice=pd.read_csv('marketPrice.csv')
os.chdir('..')
os.chdir(SimulName)
#endregion

df0=pd.DataFrame({'YEAR_op':[1],'TECHNOLOGIES':['SMR_class_ex'],'capacity_Pvar':[Variables['capacityInvest_Dvar'].set_index(['YEAR_invest','TECHNOLOGIES']).loc[(1,'SMR_class_ex')].capacityInvest_Dvar]})
df=Variables['capacity_Pvar'].append(df0)
df=df.pivot(columns='TECHNOLOGIES',values='capacity_Pvar', index='YEAR_op').rename(columns={
    "electrolysis_AEL": "Alkaline electrolysis",
    "electrolysis_PEMEL": "PEM electrolysis",
    'SMR_class': "SMR w/o CCUS",
    'SMR_CCS1':  'SMR + CCUS 50%',
    'SMR_CCS2':  'SMR + CCUS 75%',
    'SMR_elec': 'eSMR w/o CCUS',
    'SMR_elecCCS1': 'eSMR + CCUS 50%',
    'cracking': 'Methane cracking'
}).fillna(0)

#LoadFactors
EnR_loadFactor={y : (Variables['power_Dvar'].groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']/(Variables['capacity_Pvar'].set_index(['YEAR_op','TECHNOLOGIES'])['capacity_Pvar']*8760)).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values=0).loc[y,['WindOnShore','Solar','WindOffShore_flot']].fillna(0)  for y in [2,3,4]}
H2_loadFactor={y : (Variables['power_Dvar'].groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']/(Variables['capacity_Pvar'].set_index(['YEAR_op','TECHNOLOGIES'])['capacity_Pvar']*8760)).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values=0).loc[y,['electrolysis_PEMEL','electrolysis_AEL','SMR_class_ex','SMR_class','SMR_CCS1','SMR_CCS2','SMR_elec','SMR_elecCCS1','cracking']].fillna(0) for y in [2,3,4]}
for y in [2,3,4] : H2_loadFactor[y].loc[H2_loadFactor[y]<0]=0
for y in [2,3,4] : H2_loadFactor[y].loc[H2_loadFactor[y]>1]=0

df['SMR w/o CCUS']+=df['SMR_class_ex']

fig, ax = plt.subplots(2,1,sharex=True, figsize=(5,3.5))
width= 0.30
labels=list(df.index)
x = np.arange(len(labels))
col=sb.color_palette('muted')

# Create dark grey Bar
l1=list(df['SMR w/o CCUS'])
ax[0].bar(x - width/2, l1,width, color=col[7], label="SMR")
# Create dark bleu Bar
l2=list(df['SMR + CCUS 50%'])
ax[0].bar(x - width/2,l2,width, bottom=l1,color='#005E9E', label="SMR + CCUS")
# Create green Bars
l7=list(df['Alkaline electrolysis']+df['PEM electrolysis'])
ax[0].bar(x + width/2,l7,width, color=col[2],label="Water electrolysis")

#Create electrolyse load factor

xi=[x[i]-0.05 for i in np.arange(1,len(x))]
yiel = [(H2_loadFactor[i+1]['electrolysis_AEL']*100) for i in np.arange(1,len(x))]
ax[1].plot(xi,yiel, color=col[2], label='Water electrolysis')

#ax2=ax[1].twinx()
#
#
# #add Load factors
# xi=[x[i]-0.05 for i in np.arange(1,len(x))]
# yiel = [(H2_loadFactor[i+1]['electrolysis_AEL']*100) for i in np.arange(1,len(x))]
#
#
# ax2.plot(xi,yiel, color='k', label='Water electrolysis', zorder=-1)
#
#
# ax2.set_ylim([0,100])
# ax2.set_ylabel('Load factor (%)')
ax[0].set_ylim([0,1800])
ax[0].set_ylabel('Installed capacity (MW)')
ax[1].set_ylim([0,100])
ax[1].set_ylabel('Load Factor (%)')

plt.xticks(x, ['2020','2030','2040', '2050'])
# Shrink current axis by 20%
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.68, box.height])
# Put a legend to the right of the current axis

#box = ax[0].get_position()
#ax[0].set_position([box.x0, box.y0, box.width * 0.68, box.height])
# Put a legend to the right of the current axis
ax[0].legend(loc='center left', bbox_to_anchor=(1.05, 0.7), title='Installed capacity')
ax[1].legend(loc='center left', bbox_to_anchor=(1.05, 0.7), title='Load factor')
plt.tight_layout()
plt.savefig('Evolution mix prod w load factor.png',bbox_inches='tight',dpi=300)
plt.show()

os.chdir('..')
os.chdir('..')
os.chdir('..')

#endregion

#region Tracé comparaison coût tot

#region import
r=0.04
# Ref
SimulName1='2022-10-19_Ref_v4_PACA'
os.chdir(OutputFolder)
os.chdir(SimulName1)
Capa1=pd.read_csv('capacityCosts_Pvar_' + SimulName1 + '.csv').drop(columns='Unnamed: 0').groupby('YEAR_op').sum()
Import1=pd.read_csv('importCosts_Pvar_' + SimulName1 + '.csv').drop(columns='Unnamed: 0').groupby('YEAR_op').sum()
Carbon1=pd.read_csv('carbonCosts_Pvar_' + SimulName1 + '.csv').drop(columns='Unnamed: 0').groupby('YEAR_op').sum()
Power1=pd.read_csv('powerCosts_Pvar_' + SimulName1 + '.csv').drop(columns='Unnamed: 0').groupby('YEAR_op').sum()
Storage1=pd.read_csv('storageCosts_Pvar_' + SimulName1 + '.csv').drop(columns='Unnamed: 0').groupby('YEAR_op').sum()
Turpe1=pd.read_csv('turpeCosts_Pvar_' + SimulName1 + '.csv').drop(columns='Unnamed: 0').groupby('YEAR_op').sum()
Ener1=pd.read_csv('energy_Pvar_' + SimulName1 + '.csv').drop(columns='Unnamed: 0')
os.chdir('..')
os.chdir('..')
os.chdir('..')

Ener1={y:Ener1.pivot(index=['YEAR_op','TIMESTAMP'],columns='RESOURCES',values='energy_Pvar').loc[y,['hydrogen']].sum()['hydrogen'] for y in [2,3,4]}
Costs1={y:Capa1.loc[y]['capacityCosts_Pvar']+Import1.loc[y]['importCosts_Pvar']+Carbon1.loc[y]['carbonCosts_Pvar']+Power1.loc[y]['powerCosts_Pvar']+Storage1.loc[y]['storageCosts_Pvar']+Turpe1.loc[y]['turpeCosts_Pvar'] for y in [2,3,4]}
kg_Costs1={y:Costs1[y]/(Ener1[y]*30)*1/((1+r)**(-10*(y-1))) for y in [2,3,4]}

# Ref EnR+
SimulName2='2022-10-19_Ref_EnR+_PACA'
os.chdir(OutputFolder)
os.chdir(SimulName2)
Capa2=pd.read_csv('capacityCosts_Pvar_' + SimulName2 + '.csv').drop(columns='Unnamed: 0').groupby('YEAR_op').sum()
Import2=pd.read_csv('importCosts_Pvar_' + SimulName2 + '.csv').drop(columns='Unnamed: 0').groupby('YEAR_op').sum()
Carbon2=pd.read_csv('carbonCosts_Pvar_' + SimulName2 + '.csv').drop(columns='Unnamed: 0').groupby('YEAR_op').sum()
Power2=pd.read_csv('powerCosts_Pvar_' + SimulName2 + '.csv').drop(columns='Unnamed: 0').groupby('YEAR_op').sum()
Storage2=pd.read_csv('storageCosts_Pvar_' + SimulName2 + '.csv').drop(columns='Unnamed: 0').groupby('YEAR_op').sum()
Turpe2=pd.read_csv('turpeCosts_Pvar_' + SimulName2 + '.csv').drop(columns='Unnamed: 0').groupby('YEAR_op').sum()
Ener2=pd.read_csv('energy_Pvar_' + SimulName2 + '.csv').drop(columns='Unnamed: 0')
os.chdir('..')
os.chdir('..')
os.chdir('..')

Ener2={y:Ener2.pivot(index=['YEAR_op','TIMESTAMP'],columns='RESOURCES',values='energy_Pvar').loc[y,['hydrogen']].sum()['hydrogen'] for y in [2,3,4]}
Costs2={y:Capa2.loc[y]['capacityCosts_Pvar']+Import2.loc[y]['importCosts_Pvar']+Carbon2.loc[y]['carbonCosts_Pvar']+Power2.loc[y]['powerCosts_Pvar']+Storage2.loc[y]['storageCosts_Pvar']+Turpe2.loc[y]['turpeCosts_Pvar'] for y in [2,3,4]}
kg_Costs2={y:Costs2[y]/(Ener2[y]*30)*1/((1+r)**(-10*(y-1))) for y in [2,3,4]}


# 20% subvention offshore flottant
SimulName3='2022-10-21_Ref_test_Offshore5_PACA'
os.chdir(OutputFolder)
os.chdir(SimulName3)
Capa3=pd.read_csv('capacityCosts_Pvar_' + SimulName3 + '.csv').drop(columns='Unnamed: 0').groupby('YEAR_op').sum()
Import3=pd.read_csv('importCosts_Pvar_' + SimulName3 + '.csv').drop(columns='Unnamed: 0').groupby('YEAR_op').sum()
Carbon3=pd.read_csv('carbonCosts_Pvar_' + SimulName3 + '.csv').drop(columns='Unnamed: 0').groupby('YEAR_op').sum()
Power3=pd.read_csv('powerCosts_Pvar_' + SimulName3 + '.csv').drop(columns='Unnamed: 0').groupby('YEAR_op').sum()
Storage3=pd.read_csv('storageCosts_Pvar_' + SimulName3 + '.csv').drop(columns='Unnamed: 0').groupby('YEAR_op').sum()
Turpe3=pd.read_csv('turpeCosts_Pvar_' + SimulName3 + '.csv').drop(columns='Unnamed: 0').groupby('YEAR_op').sum()
Ener3=pd.read_csv('energy_Pvar_' + SimulName3 + '.csv').drop(columns='Unnamed: 0')
os.chdir('..')
os.chdir('..')
os.chdir('..')

Ener3={y:Ener3.pivot(index=['YEAR_op','TIMESTAMP'],columns='RESOURCES',values='energy_Pvar').loc[y,['hydrogen']].sum()['hydrogen'] for y in [2,3,4]}
Costs3={y:Capa3.loc[y]['capacityCosts_Pvar']+Import2.loc[y]['importCosts_Pvar']+Carbon2.loc[y]['carbonCosts_Pvar']+Power2.loc[y]['powerCosts_Pvar']+Storage2.loc[y]['storageCosts_Pvar']+Turpe2.loc[y]['turpeCosts_Pvar'] for y in [2,3,4]}
kg_Costs3={y:Costs3[y]/(Ener3[y]*30)*1/((1+r)**(-10*(y-1))) for y in [2,3,4]}
#endregion

df=pd.concat([pd.DataFrame(kg_Costs1,index=['Reference']),pd.DataFrame(kg_Costs2,index=['Renewables favorable']),pd.DataFrame(kg_Costs3,index=['20% subvention for offshore'])])

fig, ax = plt.subplots(1,1, figsize=(5,2))
col=sb.color_palette('muted')

ax.plot(df.loc['Reference'],color='#005e9e',label='Reference')
ax.plot(df.loc['Renewables favorable'],color='#2ba9ff',label='Renewables favorable')
ax.plot(df.loc['20% subvention for offshore'],color=col[2],label='20% subvention \n for offshore')

#ax.set_ylim([0,3.5])
ax.set_ylabel('LCOH (€/kg)')

plt.xticks([2,3,4], ['2030','2040', '2050'])

ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.80), title='Scenario')

plt.tight_layout()
plt.savefig('ComparaisonEnR_PPL.png',bbox_inches='tight',dpi=300)
plt.show()

#endregion