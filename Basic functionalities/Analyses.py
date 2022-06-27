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

# Change this if you have other solvers obtained here
## https://ampl.com/products/solvers/open-source/
## for eduction this site provides also several professional solvers, that are more efficient than e.g. cbc
#endregion

#region Solver and data location definition
InputFolder='Data/Input_Ref/'
OutputFolder='Data/output/'
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

#region bazare
#region import results
SimulName='2022-5-13_PACA_Ref'
os.chdir(OutputFolder)
os.chdir(SimulName)
v_list = ['capacityInvest_Dvar','capacityDem_Dvar','transInvest_Dvar','capacity_Pvar','capacityDel_Pvar', 'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
         'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar','carbon_Pvar','powerCosts_Pvar','capacityCosts_Pvar','importCosts_Pvar','storageCosts_Pvar','turpeCosts_Pvar','Pmax_Pvar','max_PS_Dvar','carbonCosts_Pvar']
Variables = {v : pd.read_csv(v + '_' + SimulName + '.csv').drop(columns='Unnamed: 0') for v in v_list}
carbon_content=pd.read_csv('carbon_' + SimulName + '.csv').set_index(['YEAR_op','TIMESTAMP'])
os.chdir('..')
SimulFranceName='2022-5-13_Fr_Ref'
os.chdir(SimulFranceName)
elec_price=pd.read_csv('elecPrice_' + SimulFranceName + '.csv').set_index(['YEAR_op','TIMESTAMP'])
marketPrice=pd.read_csv('marketPrice.csv').set_index(['YEAR_op','TIMESTAMP'])
os.chdir('..')
os.chdir('..')
os.chdir('..')
#endregion

#Elec from grid
grid_meanPrice={y : marketPrice.groupby('YEAR_op').mean().loc[y]['NewPrice'] for y in [2,3,4]}
grid_varCoef={y :  marketPrice.groupby('YEAR_op').std().loc[y]['NewPrice'] / grid_meanPrice[y] for y in [2,3,4]}
user_meanPrice={y : Variables['importCosts_Pvar'].set_index(['YEAR_op','RESOURCES']).loc[(y,'electricity')]['importCosts_Pvar']/Variables['importation_Dvar'].set_index('RESOURCES').loc['electricity'].groupby('YEAR_op').sum().loc[y]['importation_Dvar'] for y in [2,3,4]}

#Elec from local EnR
Zones="PACA" ; year='2020-2050'; PrixRes='horaire'
availabilityFac={y:pd.read_csv(InputFolder + 'availabilityFactor' + str(year) + '_' + str(Zones) + '_TIMExTECHxYEAR.csv',sep=',', decimal='.', skiprows=0).rename(columns={'YEAR':'YEAR_op'}).set_index("YEAR_op").rename(index={2030:2,2040:3,2050:4}).set_index(["TIMESTAMP", "TECHNOLOGIES"],append=True).groupby(['YEAR_op','TECHNOLOGIES']).mean().reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values='availabilityFactor').loc[y,['WindOnShore','Solar','WindOffShore']] for y in [2,3,4]}
EnR_meanPrice={y:Variables['capacityCosts_Pvar'].pivot(index='YEAR_op',columns='TECHNOLOGIES',values='capacityCosts_Pvar').loc[y,['WindOnShore','Solar','WindOffShore']].sum()/Variables['power_Dvar'].groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP').reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values='power_Dvar').loc[y,['WindOnShore','Solar','WindOffShore']].sum() for y in [2,3,4]}
EnR_meanPrice_byTechno={y:(Variables['capacityCosts_Pvar'].set_index(['YEAR_op','TECHNOLOGIES'])['capacityCosts_Pvar']/Variables['power_Dvar'].groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values=0).loc[y,['WindOnShore','Solar','WindOffShore']].dropna() for y in [2,3,4]}
maxEnR={y:Variables['capacity_Pvar'].pivot(index='YEAR_op',columns='TECHNOLOGIES',values='capacity_Pvar').loc[y,['WindOnShore','Solar','WindOffShore']]*availabilityFac[y]*8760 for y in [2,3,4]}
EnR_curtailment={y:maxEnR[y]-Variables['power_Dvar'].groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar'].reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values='power_Dvar').loc[y,['WindOnShore','Solar','WindOffShore']]for y in [2,3,4]}
EnR_curtailmentFac={y: EnR_curtailment[y].sum()/maxEnR[y].sum()*100  for y in [2,3,4]}
EnR_loadFactor={y : (Variables['power_Dvar'].groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']/(Variables['capacity_Pvar'].set_index(['YEAR_op','TECHNOLOGIES'])['capacity_Pvar']*8760)).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values=0).loc[y,['WindOnShore','Solar','WindOffShore']].dropna()  for y in [2,3,4]}

#GazNat
gazNat_meanPrice={y : Variables['importCosts_Pvar'].set_index(['YEAR_op','RESOURCES']).loc[(y,'gazNat')]['importCosts_Pvar']/Variables['importation_Dvar'].set_index('RESOURCES').loc['gazNat'].groupby('YEAR_op').sum().loc[y]['importation_Dvar'] for y in [2,3,4]}

#Fonctionnement prod H2
H2_loadFactor={y : (Variables['power_Dvar'].groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']/(Variables['capacity_Pvar'].set_index(['YEAR_op','TECHNOLOGIES'])['capacity_Pvar']*8760)).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values=0).loc[y,['electrolysis','SMR_class_ex','SMR_class','SMR_CCS1','SMR_CCS2','SMR_elec','SMR_elecCCS1']].dropna() for y in [2,3,4]}

#Répartition énergies
Elec={y:Variables['importation_Dvar'].pivot(index=['YEAR_op','TIMESTAMP'],columns='RESOURCES',values='importation_Dvar').loc[y,['electricity']].sum()['electricity'] for y in [2,3,4]}
NatGaz={y:Variables['importation_Dvar'].pivot(index=['YEAR_op','TIMESTAMP'],columns='RESOURCES',values='importation_Dvar').loc[y,['gazNat']].sum()['gazNat'] for y in [2,3,4]}
BioGaz={y:Variables['importation_Dvar'].pivot(index=['YEAR_op','TIMESTAMP'],columns='RESOURCES',values='importation_Dvar').loc[y,['gazBio']].sum()['gazBio'] for y in [2,3,4]}
EnR={y:(Variables['power_Dvar'].groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values='power_Dvar').loc[y,['WindOnShore','Solar','WindOffShore']].sum() for y in [2,3,4]}

sb.set_palette('muted')
fig,ax = plt.subplots(3,1,figsize=(7, 8))
labels=['Grid','Natural gaz','Local EnR']
meanP={y: [str(round(user_meanPrice[y],1))+'€/MWh',str(round(EnR_meanPrice[y],1))+'€/MWh',str(round(gazNat_meanPrice[y],1))+'€/MWh'] for y in [2,3,4]}
dic_an={2:2030,3:2040,4:2050}

for i in np.arange(2,5) :
    ax[i-2].pie([Elec[i],NatGaz[i],EnR[i]],autopct='%1.1f%%',labels=meanP[i],shadow=True,labeldistance=1.2,startangle=0)
    ax[i-2].set_title(dic_an[i],loc='left')
    ax[i-2].axis('equal')

ax[1].legend(labels,title='Energy used',loc="lower right")
os.chdir(OutputFolder)
os.chdir(SimulName)
plt.savefig('Repartition.png')
os.chdir('..')
os.chdir('..')
os.chdir('..')
plt.show()


test=(availabilityFactor.loc[(4,slice(None),'Solar')]*2665)+(availabilityFactor.loc[(4,slice(None),'WindOnShore')]*2589)
test['>1785']=test['availabilityFactor']
test.loc[test['>1785']>1785,'>1785']=1785
test['>1785'].sum()/1000
#endregion

#region comparaison results

#region import
# Ref
SimulName1='2022-5-23_Ref_bis_PACA'
os.chdir(OutputFolder)
os.chdir(SimulName1)
Capa1=pd.read_csv('capacity_Pvar_' + SimulName1 + '.csv').drop(columns='Unnamed: 0')
Power1=pd.read_csv('power_Dvar_' + SimulName1 + '.csv').drop(columns='Unnamed: 0')
Import1=pd.read_csv('importation_Dvar_' + SimulName1 + '.csv').drop(columns='Unnamed: 0')
os.chdir('..')
os.chdir('..')
os.chdir('..')

EnR_loadFactor1={y : (Power1.groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']/(Capa1.set_index(['YEAR_op','TECHNOLOGIES'])['capacity_Pvar']*8760)).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values=0).loc[y,['WindOnShore','Solar','WindOffShore']].fillna(0)  for y in [2,3,4]}
H2_loadFactor1={y : (Power1.groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']/(Capa1.set_index(['YEAR_op','TECHNOLOGIES'])['capacity_Pvar']*8760)).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values=0).loc[y,['electrolysis','SMR_class_ex','SMR_class','SMR_CCS1','SMR_CCS2','SMR_elec','SMR_elecCCS1']].fillna(0) for y in [2,3,4]}
df1={y:Capa1.loc[Capa1['YEAR_op']==y].drop(columns='YEAR_op') for y in [2,3,4]}
for y in [2,3,4] : df1[y]['Scenario']='Ref'

Elec1={y:Import1.pivot(index=['YEAR_op','TIMESTAMP'],columns='RESOURCES',values='importation_Dvar').loc[y,['electricity']].sum()['electricity'] for y in [2,3,4]}
NatGaz1={y:Import1.pivot(index=['YEAR_op','TIMESTAMP'],columns='RESOURCES',values='importation_Dvar').loc[y,['gazNat']].sum()['gazNat'] for y in [2,3,4]}
BioGaz1={y:Import1.pivot(index=['YEAR_op','TIMESTAMP'],columns='RESOURCES',values='importation_Dvar').loc[y,['gazBio']].sum()['gazBio'] for y in [2,3,4]}
EnR1={y:(Power1.groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values='power_Dvar').loc[y,['WindOnShore','Solar','WindOffShore']].sum() for y in [2,3,4]}
dr1={y:pd.DataFrame([EnR1[y],Elec1[y],NatGaz1[y],BioGaz1[y]],index=['EnR','Grid','gazNat','biogaz'],columns=['energy']).reset_index() for y in [2,3,4]}
for y in [2,3,4] : dr1[y]['Scenario']='Ref'

# eSMR
SimulName2='2022-5-23_eSMR_ter_PACA'
os.chdir(OutputFolder)
os.chdir(SimulName2)
Capa2=pd.read_csv('capacity_Pvar_' + SimulName2 + '.csv').drop(columns='Unnamed: 0')
Power2=pd.read_csv('power_Dvar_' + SimulName2 + '.csv').drop(columns='Unnamed: 0')
Import2=pd.read_csv('importation_Dvar_' + SimulName2 + '.csv').drop(columns='Unnamed: 0')
os.chdir('..')
os.chdir('..')
os.chdir('..')
EnR_loadFactor2={y : (Power2.groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']/(Capa2.set_index(['YEAR_op','TECHNOLOGIES'])['capacity_Pvar']*8760)).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values=0).loc[y,['WindOnShore','Solar','WindOffShore']].fillna(0)  for y in [2,3,4]}
H2_loadFactor2={y : (Power2.groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']/(Capa2.set_index(['YEAR_op','TECHNOLOGIES'])['capacity_Pvar']*8760)).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values=0).loc[y,['electrolysis','SMR_class_ex','SMR_class','SMR_CCS1','SMR_CCS2','SMR_elec','SMR_elecCCS1']].fillna(0) for y in [2,3,4]}
df2={y:Capa2.loc[Capa2['YEAR_op']==y].drop(columns='YEAR_op') for y in [2,3,4]}
for y in [2,3,4] : df2[y]['Scenario']='eSMR'

Elec2={y:Import2.pivot(index=['YEAR_op','TIMESTAMP'],columns='RESOURCES',values='importation_Dvar').loc[y,['electricity']].sum()['electricity'] for y in [2,3,4]}
NatGaz2={y:Import2.pivot(index=['YEAR_op','TIMESTAMP'],columns='RESOURCES',values='importation_Dvar').loc[y,['gazNat']].sum()['gazNat'] for y in [2,3,4]}
BioGaz2={y:Import2.pivot(index=['YEAR_op','TIMESTAMP'],columns='RESOURCES',values='importation_Dvar').loc[y,['gazBio']].sum()['gazBio'] for y in [2,3,4]}
EnR2={y:(Power2.groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values='power_Dvar').loc[y,['WindOnShore','Solar','WindOffShore']].sum() for y in [2,3,4]}
dr2={y:pd.DataFrame([EnR2[y],Elec2[y],NatGaz2[y],BioGaz2[y]],index=['EnR','Grid','gazNat','biogaz'],columns=['energy']).reset_index() for y in [2,3,4]}
for y in [2,3,4] : dr2[y]['Scenario']='eSMR'

# EnR
SimulName3='2022-5-23_EnR_bis_PACA'
os.chdir(OutputFolder)
os.chdir(SimulName3)
Capa3=pd.read_csv('capacity_Pvar_' + SimulName3 + '.csv').drop(columns='Unnamed: 0')
Power3=pd.read_csv('power_Dvar_' + SimulName3 + '.csv').drop(columns='Unnamed: 0')
Import3=pd.read_csv('importation_Dvar_' + SimulName3 + '.csv').drop(columns='Unnamed: 0')
os.chdir('..')
os.chdir('..')
os.chdir('..')
EnR_loadFactor3={y : (Power3.groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']/(Capa3.set_index(['YEAR_op','TECHNOLOGIES'])['capacity_Pvar']*8760)).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values=0).loc[y,['WindOnShore','Solar','WindOffShore']].fillna(0)  for y in [2,3,4]}
H2_loadFactor3={y : (Power3.groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']/(Capa3.set_index(['YEAR_op','TECHNOLOGIES'])['capacity_Pvar']*8760)).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values=0).loc[y,['electrolysis','SMR_class_ex','SMR_class','SMR_CCS1','SMR_CCS2','SMR_elec','SMR_elecCCS1']].fillna(0) for y in [2,3,4]}
df3={y:Capa3.loc[Capa3['YEAR_op']==y].drop(columns='YEAR_op')for y in [2,3,4]}
for y in [2,3,4]:df3[y]['Scenario']='EnR'

Elec3={y:Import3.pivot(index=['YEAR_op','TIMESTAMP'],columns='RESOURCES',values='importation_Dvar').loc[y,['electricity']].sum()['electricity'] for y in [2,3,4]}
NatGaz3={y:Import3.pivot(index=['YEAR_op','TIMESTAMP'],columns='RESOURCES',values='importation_Dvar').loc[y,['gazNat']].sum()['gazNat'] for y in [2,3,4]}
BioGaz3={y:Import3.pivot(index=['YEAR_op','TIMESTAMP'],columns='RESOURCES',values='importation_Dvar').loc[y,['gazBio']].sum()['gazBio'] for y in [2,3,4]}
EnR3={y:(Power3.groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values='power_Dvar').loc[y,['WindOnShore','Solar','WindOffShore']].sum() for y in [2,3,4]}
dr3={y:pd.DataFrame([EnR3[y],Elec3[y],NatGaz3[y],BioGaz3[y]],index=['EnR','Grid','gazNat','biogaz'],columns=['energy']).reset_index() for y in [2,3,4]}
for y in [2,3,4]:dr3[y]['Scenario']='EnR'

# Grid
SimulName4='2022-5-23_Grid_bis_PACA'
os.chdir(OutputFolder)
os.chdir(SimulName4)
Capa4=pd.read_csv('capacity_Pvar_' + SimulName4 + '.csv').drop(columns='Unnamed: 0')
Power4=pd.read_csv('power_Dvar_' + SimulName4 + '.csv').drop(columns='Unnamed: 0')
Import4=pd.read_csv('importation_Dvar_' + SimulName4 + '.csv').drop(columns='Unnamed: 0')
os.chdir('..')
os.chdir('..')
os.chdir('..')
EnR_loadFactor4={y : (Power4.groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']/(Capa4.set_index(['YEAR_op','TECHNOLOGIES'])['capacity_Pvar']*8760)).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values=0).loc[y,['WindOnShore','Solar','WindOffShore']].fillna(0)  for y in [2,3,4]}
H2_loadFactor4={y : (Power4.groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']/(Capa4.set_index(['YEAR_op','TECHNOLOGIES'])['capacity_Pvar']*8760)).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values=0).loc[y,['electrolysis','SMR_class_ex','SMR_class','SMR_CCS1','SMR_CCS2','SMR_elec','SMR_elecCCS1']].fillna(0) for y in [2,3,4]}
df4={y:Capa4.loc[Capa4['YEAR_op']==y].drop(columns='YEAR_op')for y in [2,3,4]}
for y in [2,3,4]:df4[y]['Scenario']='Grid'

Elec4={y:Import4.pivot(index=['YEAR_op','TIMESTAMP'],columns='RESOURCES',values='importation_Dvar').loc[y,['electricity']].sum()['electricity'] for y in [2,3,4]}
NatGaz4={y:Import4.pivot(index=['YEAR_op','TIMESTAMP'],columns='RESOURCES',values='importation_Dvar').loc[y,['gazNat']].sum()['gazNat'] for y in [2,3,4]}
BioGaz4={y:Import4.pivot(index=['YEAR_op','TIMESTAMP'],columns='RESOURCES',values='importation_Dvar').loc[y,['gazBio']].sum()['gazBio'] for y in [2,3,4]}
EnR4={y:(Power4.groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values='power_Dvar').loc[y,['WindOnShore','Solar','WindOffShore']].sum() for y in [2,3,4]}
dr4={y:pd.DataFrame([EnR4[y],Elec4[y],NatGaz4[y],BioGaz4[y]],index=['EnR','Grid','gazNat','biogaz'],columns=['energy']).reset_index() for y in [2,3,4]}
for y in [2,3,4]:dr4[y]['Scenario']='Grid'

# Gaz Nat
SimulName5='2022-5-23_GN_bis_PACA'
os.chdir(OutputFolder)
os.chdir(SimulName5)
Capa5=pd.read_csv('capacity_Pvar_' + SimulName5 + '.csv').drop(columns='Unnamed: 0')
Power5=pd.read_csv('power_Dvar_' + SimulName5 + '.csv').drop(columns='Unnamed: 0')
Import5=pd.read_csv('importation_Dvar_' + SimulName5 + '.csv').drop(columns='Unnamed: 0')
os.chdir('..')
os.chdir('..')
os.chdir('..')
EnR_loadFactor5={y : (Power5.groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']/(Capa5.set_index(['YEAR_op','TECHNOLOGIES'])['capacity_Pvar']*8760)).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values=0).loc[y,['WindOnShore','Solar','WindOffShore']].fillna(0)  for y in [2,3,4]}
H2_loadFactor5={y : (Power5.groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']/(Capa5.set_index(['YEAR_op','TECHNOLOGIES'])['capacity_Pvar']*8760)).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values=0).loc[y,['electrolysis','SMR_class_ex','SMR_class','SMR_CCS1','SMR_CCS2','SMR_elec','SMR_elecCCS1']].fillna(0) for y in [2,3,4]}
df5={y:Capa5.loc[Capa5['YEAR_op']==y].drop(columns='YEAR_op') for y in [2,3,4]}
for y in [2,3,4] : df5[y]['Scenario']='GazNat'

Elec5={y:Import5.pivot(index=['YEAR_op','TIMESTAMP'],columns='RESOURCES',values='importation_Dvar').loc[y,['electricity']].sum()['electricity'] for y in [2,3,4]}
NatGaz5={y:Import5.pivot(index=['YEAR_op','TIMESTAMP'],columns='RESOURCES',values='importation_Dvar').loc[y,['gazNat']].sum()['gazNat'] for y in [2,3,4]}
BioGaz5={y:Import5.pivot(index=['YEAR_op','TIMESTAMP'],columns='RESOURCES',values='importation_Dvar').loc[y,['gazBio']].sum()['gazBio'] for y in [2,3,4]}
EnR5={y:(Power5.groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values='power_Dvar').loc[y,['WindOnShore','Solar','WindOffShore']].sum() for y in [2,3,4]}
dr5={y:pd.DataFrame([EnR5[y],Elec5[y],NatGaz5[y],BioGaz5[y]],index=['EnR','Grid','gazNat','biogaz'],columns=['energy']).reset_index() for y in [2,3,4]}
for y in [2,3,4] : dr5[y]['Scenario']='GazNat'

# Gaz Bio
SimulName6='2022-5-23_BG_tetra_PACA'
os.chdir(OutputFolder)
os.chdir(SimulName6)
Capa6=pd.read_csv('capacity_Pvar_' + SimulName6 + '.csv').drop(columns='Unnamed: 0')
Power6=pd.read_csv('power_Dvar_' + SimulName6 + '.csv').drop(columns='Unnamed: 0')
Import6=pd.read_csv('importation_Dvar_' + SimulName6 + '.csv').drop(columns='Unnamed: 0')
os.chdir('..')
os.chdir('..')
os.chdir('..')
EnR_loadFactor6={y : (Power6.groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']/(Capa6.set_index(['YEAR_op','TECHNOLOGIES'])['capacity_Pvar']*8760)).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values=0).loc[y,['WindOnShore','Solar','WindOffShore']].fillna(0)  for y in [2,3,4]}
H2_loadFactor6={y : (Power6.groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']/(Capa6.set_index(['YEAR_op','TECHNOLOGIES'])['capacity_Pvar']*8760)).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values=0).loc[y,['electrolysis','SMR_class_ex','SMR_class','SMR_CCS1','SMR_CCS2','SMR_elec','SMR_elecCCS1']].fillna(0) for y in [2,3,4]}
df6={y: Capa6.loc[Capa6['YEAR_op']==y].drop(columns='YEAR_op') for y in [2,3,4]}
for y in [2,3,4] : df6[y]['Scenario']='GazBio'

Elec6={y:Import6.pivot(index=['YEAR_op','TIMESTAMP'],columns='RESOURCES',values='importation_Dvar').loc[y,['electricity']].sum()['electricity'] for y in [2,3,4]}
NatGaz6={y:Import6.pivot(index=['YEAR_op','TIMESTAMP'],columns='RESOURCES',values='importation_Dvar').loc[y,['gazNat']].sum()['gazNat'] for y in [2,3,4]}
BioGaz6={y:Import6.pivot(index=['YEAR_op','TIMESTAMP'],columns='RESOURCES',values='importation_Dvar').loc[y,['gazBio']].sum()['gazBio'] for y in [2,3,4]}
EnR6={y:(Power6.groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values='power_Dvar').loc[y,['WindOnShore','Solar','WindOffShore']].sum() for y in [2,3,4]}
dr6={y:pd.DataFrame([EnR6[y],Elec6[y],NatGaz6[y],BioGaz6[y]],index=['EnR','Grid','gazNat','biogaz'],columns=['energy']).reset_index() for y in [2,3,4]}
for y in [2,3,4] : dr6[y]['Scenario']='GazBio'

#endregion

#region Tracé mix prod H2 et EnR

year=2050
dico_an={2030:2,2040:3,2050:4}
yr=dico_an[year]

df=pd.concat([df1[yr],df2[yr],df3[yr],df4[yr],df5[yr],df6[yr]])
df=df.pivot(columns='TECHNOLOGIES',values='capacity_Pvar',index='Scenario').rename(columns={
    "electrolysis": "Water electrolysis",
    'SMR_class': "SMR w/o CCUS",
    'SMR_CCS1':  'SMR + CCUS 50%',
    'SMR_CCS2':  'SMR + CCUS 75%',
    'SMR_elec': 'eSMR w/o CCUS',
    'SMR_elecCCS1': 'eSMR + CCUS 50%'
}).fillna(0)
df['SMR w/o CCUS']+=df['SMR_class_ex']

dr=pd.concat([dr1[yr],dr2[yr],dr3[yr],dr4[yr],dr5[yr],dr6[yr]])
dr=dr.pivot(columns='index',values='energy',index='Scenario')

EnR_loadFactor=pd.DataFrame(EnR_loadFactor1[yr]).rename(columns={yr:'Ref'})
EnR_loadFactor['eSMR']=EnR_loadFactor2[yr]
EnR_loadFactor['EnR']=EnR_loadFactor3[yr]
EnR_loadFactor['Grid']=EnR_loadFactor4[yr]
EnR_loadFactor['GazNat']=EnR_loadFactor5[yr]
EnR_loadFactor['GazBio']=EnR_loadFactor6[yr]

H2_loadFactor=pd.DataFrame(H2_loadFactor1[yr]).rename(columns={yr:'Ref'})
H2_loadFactor['eSMR']=H2_loadFactor2[yr]
H2_loadFactor['EnR']=H2_loadFactor3[yr]
H2_loadFactor['Grid']=H2_loadFactor4[yr]
H2_loadFactor['GazNat']=H2_loadFactor5[yr]
H2_loadFactor['GazBio']=H2_loadFactor6[yr]

fig, ax = plt.subplots(3,1,sharex=True,figsize=(8, 8))
width= 0.30
labels=list(df.index)
x = np.arange(len(labels))
col=sb.color_palette('muted')

# Create dark grey Bar
l1=list(df['SMR w/o CCUS']/1000)
ax[0].bar(x - width/2, l1,width, color=col[7], label="SMR w/o CCUS")
# Create dark bleu Bar
l2=list(df['SMR + CCUS 50%']/1000)
ax[0].bar(x - width/2,l2,width, bottom=l1,color='#005E9E', label="SMR + CCUS 50%")
#Create turquoise bleu Bar
l3=list(df['SMR + CCUS 75%']/1000)
ax[0].bar(x - width/2,l3,width, bottom=[i+j for i,j in zip(l1,l2)], color=col[9] ,label="SMR + CCUS 75%")
#Create orange Bar
l4=list(df['eSMR w/o CCUS']/1000)
ax[0].bar(x - width/2,l4,width, bottom=[i+j+k for i,j,k in zip(l1,l2,l3)], color=col[1],label="eSMR w/o CCUS")
# Create yellow bar
l5=list(df['eSMR + CCUS 50%']/1000)
ax[0].bar(x - width/2,l5,width, bottom=[i+j+k+l for i,j,k,l in zip(l1,l2,l3,l4)], color='#F8B740',label="eSMR + CCUS 50%")
# Create green Bars
l6=list(df['Water electrolysis']/1000)
ax[0].bar(x + width/2,l6,width, color=col[2],label="Water electrolysis")

# Create red bar
l7=list(df['Solar']/1000)
ax[1].bar(x ,l7,width, color=col[3],label="Solar")
# Create violet bar
l8=list(df['WindOnShore']/1000)
ax[1].bar(x,l8,width,  bottom=l7,color=col[4],label="Wind")

# Create brown Bar
l1=list(dr['gazNat']/1000000)
ax[2].bar(x - width/2, l1,width, color=col[5], label="Natural gas")
# Create light brown Bar
l2=list(dr['biogaz']/1000000)
ax[2].bar(x - width/2,l2,width, bottom=l1,color=col[8], label="Biogas")
#Create pink Bar
l3=list(dr['Grid']/1000000)
ax[2].bar(x + width/2,l3,width, color=col[6] ,label="Electricity from grid")
#Create blue Bar
l4=list(dr['EnR']/1000000)
ax[2].bar(x + width/2,l4,width, bottom=l3, color='#2BA9FF',label="Local renewables")


#add Load factors
for i in x :
    ax[0].text((x + width/2)[i], l6[i]/2, str(round(H2_loadFactor[labels[i]]['electrolysis']*100))+'%',ha='center')
    #ax[0].text((x - width / 2)[i], l1[i] / 2, str(round(H2_loadFactor[labels[i]]['SMR_class_ex'] * 100)) + '%',ha='center')
    #ax[1].text(x[i], l7[i]/2, str(round(EnR_loadFactor[labels[i]]['Solar'] * 100)) + '%', ha='center')
    #ax[1].text(x[i], l7[i]+l8[i]/2, str(round(EnR_loadFactor[labels[i]]['WindOnShore'] * 100)) + '%', ha='center')

#ax[0].set_ylim([0,1100])
ax[1].set_ylim([0,5.5])
ax[0].set_ylabel('Installed capacity (GW)')
ax[1].set_ylabel('Installed capacity (GW)')
ax[2].set_ylabel('Energy source (TWh)')
ax[0].set_title("Evolution of H2 production assets")
ax[1].set_title("Evolution of EnR assets")
ax[2].set_title("Origin of energy for H2 production")
plt.xticks(x, labels)
# Shrink current axis by 20%
box = ax[0].get_position()
ax[0].set_position([box.x0, box.y0, box.width * 0.74, box.height])
# Put a legend to the right of the current axis
ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
box = ax[1].get_position()
ax[1].set_position([box.x0, box.y0, box.width * 0.74, box.height])
# Put a legend to the right of the current axis
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
box = ax[2].get_position()
ax[2].set_position([box.x0, box.y0, box.width * 0.74, box.height])
# Put a legend to the right of the current axis
ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('Evolution mix prod_'+str(year))
plt.show()

#endregion

#endregion

#region Analyse de sensibilité

#region import
# Ref
SimulName1='2022-5-23_Ref_bis_PACA'
os.chdir(OutputFolder)
os.chdir(SimulName1)
Capa1=pd.read_csv('capacity_Pvar_' + SimulName1 + '.csv').drop(columns='Unnamed: 0')
#Power1=pd.read_csv('power_Dvar_' + SimulName1 + '.csv').drop(columns='Unnamed: 0')
#Import1=pd.read_csv('importation_Dvar_' + SimulName1 + '.csv').drop(columns='Unnamed: 0')
os.chdir('..')
os.chdir('..')
os.chdir('..')

df1=Capa1.pivot(columns='TECHNOLOGIES',values='capacity_Pvar', index='YEAR_op').rename(columns={
    "electrolysis": "Water electrolysis",
    'SMR_class': "SMR w/o CCUS",
    'SMR_CCS1':  'SMR + CCUS 50%',
    'SMR_CCS2':  'SMR + CCUS 75%',
    'SMR_elec': 'eSMR w/o CCUS',
    'SMR_elecCCS1': 'eSMR + CCUS 50%'
}).fillna(0)
df1['SMR w/o CCUS']+=df1['SMR_class_ex']
df1['SMR w CCUS']=df1['SMR + CCUS 50%']+df1['SMR + CCUS 75%']
df1['eSMR']=df1['eSMR w/o CCUS']+df1['eSMR + CCUS 50%']

# eSMR
SimulName2='2022-5-23_eSMR_ter_PACA'
os.chdir(OutputFolder)
os.chdir(SimulName2)
Capa2=pd.read_csv('capacity_Pvar_' + SimulName2 + '.csv').drop(columns='Unnamed: 0')
#Power2=pd.read_csv('power_Dvar_' + SimulName2 + '.csv').drop(columns='Unnamed: 0')
#Import2=pd.read_csv('importation_Dvar_' + SimulName2 + '.csv').drop(columns='Unnamed: 0')
os.chdir('..')
os.chdir('..')
os.chdir('..')
df2=Capa2.pivot(columns='TECHNOLOGIES',values='capacity_Pvar', index='YEAR_op').rename(columns={
    "electrolysis": "Water electrolysis",
    'SMR_class': "SMR w/o CCUS",
    'SMR_CCS1':  'SMR + CCUS 50%',
    'SMR_CCS2':  'SMR + CCUS 75%',
    'SMR_elec': 'eSMR w/o CCUS',
    'SMR_elecCCS1': 'eSMR + CCUS 50%'
}).fillna(0)
df2['SMR w/o CCUS']+=df2['SMR_class_ex']
df2['SMR w CCUS']=df2['SMR + CCUS 50%']+df2['SMR + CCUS 75%']
df2['eSMR']=df2['eSMR w/o CCUS']+df2['eSMR + CCUS 50%']

# EnR
SimulName3='2022-5-23_EnR_bis_PACA'
os.chdir(OutputFolder)
os.chdir(SimulName3)
Capa3=pd.read_csv('capacity_Pvar_' + SimulName3 + '.csv').drop(columns='Unnamed: 0')
#Power3=pd.read_csv('power_Dvar_' + SimulName3 + '.csv').drop(columns='Unnamed: 0')
#Import3=pd.read_csv('importation_Dvar_' + SimulName3 + '.csv').drop(columns='Unnamed: 0')
os.chdir('..')
os.chdir('..')
os.chdir('..')
df3=Capa3.pivot(columns='TECHNOLOGIES',values='capacity_Pvar', index='YEAR_op').rename(columns={
    "electrolysis": "Water electrolysis",
    'SMR_class': "SMR w/o CCUS",
    'SMR_CCS1':  'SMR + CCUS 50%',
    'SMR_CCS2':  'SMR + CCUS 75%',
    'SMR_elec': 'eSMR w/o CCUS',
    'SMR_elecCCS1': 'eSMR + CCUS 50%'
}).fillna(0)
df3['SMR w/o CCUS']+=df3['SMR_class_ex']
df3['SMR w CCUS']=df3['SMR + CCUS 50%']+df3['SMR + CCUS 75%']
df3['eSMR']=df3['eSMR w/o CCUS']+df3['eSMR + CCUS 50%']

# Grid
SimulName4='2022-5-23_Grid_bis_PACA'
os.chdir(OutputFolder)
os.chdir(SimulName4)
Capa4=pd.read_csv('capacity_Pvar_' + SimulName4 + '.csv').drop(columns='Unnamed: 0')
#Power4=pd.read_csv('power_Dvar_' + SimulName4 + '.csv').drop(columns='Unnamed: 0')
#Import4=pd.read_csv('importation_Dvar_' + SimulName4 + '.csv').drop(columns='Unnamed: 0')
os.chdir('..')
os.chdir('..')
os.chdir('..')
df4=Capa4.pivot(columns='TECHNOLOGIES',values='capacity_Pvar', index='YEAR_op').rename(columns={
    "electrolysis": "Water electrolysis",
    'SMR_class': "SMR w/o CCUS",
    'SMR_CCS1':  'SMR + CCUS 50%',
    'SMR_CCS2':  'SMR + CCUS 75%',
    'SMR_elec': 'eSMR w/o CCUS',
    'SMR_elecCCS1': 'eSMR + CCUS 50%'
}).fillna(0)
df4['SMR w/o CCUS']+=df4['SMR_class_ex']
df4['SMR w CCUS']=df4['SMR + CCUS 50%']+df4['SMR + CCUS 75%']
df4['eSMR']=df4['eSMR w/o CCUS']+df4['eSMR + CCUS 50%']

# Gaz Nat
SimulName5='2022-5-23_GN_bis_PACA'
os.chdir(OutputFolder)
os.chdir(SimulName5)
Capa5=pd.read_csv('capacity_Pvar_' + SimulName5 + '.csv').drop(columns='Unnamed: 0')
#Power5=pd.read_csv('power_Dvar_' + SimulName5 + '.csv').drop(columns='Unnamed: 0')
#Import5=pd.read_csv('importation_Dvar_' + SimulName5 + '.csv').drop(columns='Unnamed: 0')
os.chdir('..')
os.chdir('..')
os.chdir('..')
df5=Capa5.pivot(columns='TECHNOLOGIES',values='capacity_Pvar', index='YEAR_op').rename(columns={
    "electrolysis": "Water electrolysis",
    'SMR_class': "SMR w/o CCUS",
    'SMR_CCS1':  'SMR + CCUS 50%',
    'SMR_CCS2':  'SMR + CCUS 75%',
    'SMR_elec': 'eSMR w/o CCUS',
    'SMR_elecCCS1': 'eSMR + CCUS 50%'
}).fillna(0)
df5['SMR w/o CCUS']+=df5['SMR_class_ex']
df5['SMR w CCUS']=df5['SMR + CCUS 50%']+df5['SMR + CCUS 75%']
df5['eSMR']=df5['eSMR w/o CCUS']+df5['eSMR + CCUS 50%']

# Gaz Bio
SimulName6='2022-5-23_BG_tetra_PACA'
os.chdir(OutputFolder)
os.chdir(SimulName6)
Capa6=pd.read_csv('capacity_Pvar_' + SimulName6 + '.csv').drop(columns='Unnamed: 0')
Power6=pd.read_csv('power_Dvar_' + SimulName6 + '.csv').drop(columns='Unnamed: 0')
Import6=pd.read_csv('importation_Dvar_' + SimulName6 + '.csv').drop(columns='Unnamed: 0')
os.chdir('..')
os.chdir('..')
os.chdir('..')
df6=Capa6.pivot(columns='TECHNOLOGIES',values='capacity_Pvar', index='YEAR_op').rename(columns={
    "electrolysis": "Water electrolysis",
    'SMR_class': "SMR w/o CCUS",
    'SMR_CCS1':  'SMR + CCUS 50%',
    'SMR_CCS2':  'SMR + CCUS 75%',
    'SMR_elec': 'eSMR w/o CCUS',
    'SMR_elecCCS1': 'eSMR + CCUS 50%'
}).fillna(0)
df6['SMR w/o CCUS']+=df6['SMR_class_ex']
df6['SMR w CCUS']=df6['SMR + CCUS 50%']+df6['SMR + CCUS 75%']
df6['eSMR']=df6['eSMR w/o CCUS']+df6['eSMR + CCUS 50%']

#endregion

df_min=df1.copy()
df_max=df1.copy()

for y in list(df1.index):
    for tech in list(df1.columns):
        df_min.loc[y,tech]=min([df1.loc[y,tech],df2.loc[y,tech],df3.loc[y,tech],df4.loc[y,tech],df5.loc[y,tech],df6.loc[y,tech]])
        df_max.loc[y, tech]=max([df1.loc[y, tech], df2.loc[y, tech], df3.loc[y, tech], df4.loc[y, tech], df5.loc[y, tech],df6.loc[y, tech]])


fig, ax = plt.subplots(2,1,sharex=True,figsize=(7,5))
parameters={'axes.labelsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
          'figure.titlesize': 15,
            'legend.fontsize':12}
plt.rcParams.update(parameters)

#width= 0.30
labels=['2020-2030','2030-2040','2040-200']
x = np.arange(len(labels))
col=sb.color_palette('muted')

# # Create purple area
# l1=list(df_max['SMR w/o CCUS']/1000)
# ax[0].plot(l1, color=col[9])
# l2=list(df_min['SMR w/o CCUS']/1000)
# ax[0].plot(l2,color=col[9])
# ax[0].fill_between(x,l1,l2,color=col[8],label='SMR w/o CCUS')
# # Create blue area
# l3=list(df_max['SMR w CCUS']/1000)
# ax[0].plot(l3, color=col[1])
# l4=list(df_min['SMR w CCUS']/1000)
# ax[0].plot(l4,color=col[1])
# ax[0].fill_between(x,l3,l4,color=col[0],label='SMR + CCUS')
col1='#005E9E'

# Create purple area
l1=list(df1['SMR w/o CCUS']/1000)
ax[0].plot(l1, color=col1)
l11=list(df2['SMR w/o CCUS']/1000)
ax[0].plot(l11,color=col1)
l111=list(df3['SMR w/o CCUS']/1000)
ax[0].plot(l111,color=col1)
l1111=list(df4['SMR w/o CCUS']/1000)
ax[0].plot(l1111,color=col1)
l11111=list(df5['SMR w/o CCUS']/1000)
ax[0].plot(l11111,color=col1)
l111111=list(df6['SMR w/o CCUS']/1000)
ax[0].plot(l111111,color=col1,label='SMR w/o CCUS')

# Create blue area
col2='#F8B740'
l2=list(df1['SMR w CCUS']/1000)
ax[0].plot(l2, color=col2)
l22=list(df2['SMR w CCUS']/1000)
ax[0].plot(l22,color=col2)
l222=list(df3['SMR w CCUS']/1000)
ax[0].plot(l222,color=col2)
l2222=list(df4['SMR w CCUS']/1000)
ax[0].plot(l2222,color=col2)
l22222=list(df5['SMR w CCUS']/1000)
ax[0].plot(l22222,color=col2)
l222222=list(df6['SMR w CCUS']/1000)
ax[0].plot(l222222,color=col2,label="SMR w CCUS")

# # Create orange area
# l7=list(df_max['eSMR']/1000)
# ax[1].plot(l7, color=col[7])
# l8=list(df_min['eSMR']/1000)
# ax[1].plot(l8,color=col[7])
# ax[1].fill_between(x,l7,l8,color=col[6],label='eSMR')

# Create Green area
col3=col[2]
l3=list(df1['Water electrolysis']/1000)
ax[1].plot(l3, color=col3)
l33=list(df2['Water electrolysis']/1000)
ax[1].plot(l33, color=col3)
l333=list(df3['Water electrolysis']/1000)
ax[1].plot(l333, color=col3)
l3333=list(df4['Water electrolysis']/1000)
ax[1].plot(l3333, color=col3)
l33333=list(df5['Water electrolysis']/1000)
ax[1].plot(l33333, color=col3)
l333333=list(df6['Water electrolysis']/1000)
ax[1].plot(l333333, color=col3,label='Water \n electrolysis')


#ax[0].set_ylim([0,1100])
#ax[1].set_ylim([0,5.5])
ax[0].set_ylabel('Installed capacity (GW)')
ax[1].set_ylabel('Installed capacity (GW)')
#ax[2].set_ylabel('Installed capacity (GW)')
ax[0].set_title("SMR capacity")
#ax[1].set_title("eSMR capacity")
ax[1].set_title("Water electrolysis capacity")
plt.xticks(x, labels)
# Shrink current axis by 20%
box = ax[0].get_position()
ax[0].set_position([box.x0, box.y0, box.width * 0.67, box.height])
# Put a legend to the right of the current axis
ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
box = ax[1].get_position()
ax[1].set_position([box.x0, box.y0, box.width * 0.67, box.height])
# Put a legend to the right of the current axis
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
#box = ax[2].get_position()
#ax[2].set_position([box.x0, box.y0, box.width * 0.74, box.height])
# Put a legend to the right of the current axis
#ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('test')
plt.show()
#endregion