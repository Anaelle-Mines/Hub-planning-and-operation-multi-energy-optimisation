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

from Functions.f_multiResourceModels import *
from Functions.f_optimization import *
from Functions.f_graphicalTools import *
# Change this if you have other solvers obtained here
## https://ampl.com/products/solvers/open-source/
## for eduction this site provides also several professional solvers, that are more efficient than e.g. cbc
#endregion

InputFolder='Data/Input/'
OutputFolder='Data/output/'

##region exemple
dic_eco = {2020: 1, 2030: 2, 2040: 3, 2050: 4}
Demande=pd.DataFrame({'Acierie':[0,51,143,184.5],'Raffinerie':[92.1,63.6,27.6,10.1],'Methanol':[3.4,3.4,3.4,3.4]},index=[2020,2030,2040,2050])
Prod=pd.DataFrame({'ChloreAlkali':[11.3,11.3,11.3,11.3],'SMRclassique':[84.2,60,10,0],'Autres':[0,46.7,152.7,186.7]},index=[2020,2030,2040,2050])

fig,ax=plt.subplots()
x=list(Demande.index)
y0=[0,0,0,0]
#production
y1=list(Demande.Methanol)
y2=list(Demande.Raffinerie+Demande.Methanol)
y3=list(Demande.Acierie+Demande.Raffinerie+Demande.Methanol)
ax.fill_between(x,y0,y1,color='#f3f700',label='Methanol')
ax.fill_between(x,y1,y2,color='#fc6f58',label='Raffinerie')
ax.fill_between(x,y2,y3,color='#d186ce',label='Aciérie')
#consommation
y4=list(-Prod.ChloreAlkali)
y5=list(-Prod.ChloreAlkali-Prod.SMRclassique)
y6=list(-Prod.ChloreAlkali-Prod.SMRclassique-Prod.Autres)
ax.fill_between(x,y0,y4,color='#70a6e4',label='Chlore-Alkali')
ax.fill_between(x,y4,y5,color='#e3ad60',label='SMR classique')
ax.fill_between(x,y5,y6,color='#8fe977',label='Autres')

ax.set_xlabel('Années de référence')
ax.set_ylabel('kt/an')
plt.grid(True, axis='x', linewidth=0.1)
plt.title('Evolution de la demande hydrogène')

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0*2.5, box.width , box.height* 0.85])
# Put a legend to the right of the current axis
ax.legend(ncol=2,loc='lower center', bbox_to_anchor=(0.5, -0.43))

plt.savefig('test')
plt.show()
##endregion

##region réalisé
convFac = pd.read_csv(InputFolder + 'conversionFactors_SMR_RESxTECH.csv', sep=',', decimal='.', skiprows=0,comment="#").set_index(["RESOURCES", "TECHNOLOGIES"])

SimulName='carbonTaxeVar_corrige_2022-2-23'
os.chdir(OutputFolder)
os.chdir(SimulName)
v_list = ['capacityInvest_Dvar','transInvest_Dvar','capacity_Pvar','capacityDel_Dvar', 'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
         'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar','carbon_Pvar','powerCosts_Pvar','capacityCosts_Pvar','importCosts_Pvar','storageCosts_Pvar','turpeCosts_Pvar','Pmax_Pvar','max_PS_Dvar','carbonCosts_Pvar']
Variables = {v : pd.read_csv(v + '_' + SimulName + '.csv').drop(columns='Unnamed: 0') for v in v_list}


dic_eco = {2020: 1, 2030: 2, 2040: 3, 2050: 4}
Demande=pd.DataFrame({'Acierie':[0,51,143,184.5],'Raffinerie':[92.1,63.6,27.6,10.1],'Methanol':[3.4,3.4,3.4,3.4]},index=[2020,2030,2040,2050])
Alkali=[-11.3,-11.3,-11.3,-11.3]
v = 'power_Dvar'
Ph2 = {y: Variables[v].loc[Variables[v]['YEAR_op'] == y].pivot(columns='TECHNOLOGIES', values='power_Dvar', index='TIMESTAMP').drop(columns=['CCS1','CCS2']) for y in (2, 3, 4)}
Ph2={y:Ph2[y].sum() for y in (2,3,4)}
for y in (2, 3, 4) :
    for tech in list(Ph2[y].index):
        Ph2[y][tech]=Ph2[y][tech]*convFac.loc[('hydrogen',tech)].conversionFactor

Ph2=pd.concat((Ph2[y] for y in (2,3,4)),axis=1).rename(columns={0:1,1:2,2:3})
init=Ph2.drop(columns=[1,2,3])
init[0]=0
init.loc['SMR_class_ex']=320*8760
Ph2=pd.concat([init,Ph2],axis=1)

fig,ax=plt.subplots()
x=list(Demande.index)
y0=[0,0,0,0]
#production
y1=list(Demande.Methanol)
y2=list(Demande.Raffinerie+Demande.Methanol)
y3=list(Demande.Acierie+Demande.Raffinerie+Demande.Methanol)
ax.fill_between(x,y0,y1,color='#3556fc',label='Methanol',linewidth=0)
ax.fill_between(x,y1,y2,color='#fc6f58',label='Raffinerie',linewidth=0)
ax.fill_between(x,y2,y3,color='#d186ce',label='Aciérie',linewidth=0)
#consommation
y4=Alkali
y5=[i-j for i,j in zip(y4,list(Ph2.loc['SMR_class_ex']*30/1000000+Ph2.loc['SMR_class']*30/1000000))]
y6=[i-j for i,j in zip(y5,list(Ph2.loc['SMR_CCS1']*30/1000000+Ph2.loc['SMR_CCS2']*30/1000000))]
y7=[i-j for i,j in zip(y6,list(Ph2.loc['SMR_elec']*30/1000000+Ph2.loc['SMR_elecCCS1']*30/1000000))]
y8=[i-j for i,j in zip(y7,list(Ph2.loc['electrolysis']*30/1000000))]
ax.fill_between(x,y0,y4,color='#ff9ae0',label='Chlore-Alkali',linewidth=0)
ax.fill_between(x,y4,y5,color='#e3ad60',label='SMR classique',linewidth=0)
ax.fill_between(x,y5,y6,color='#3ec5ff',label='SMR w CCS',linewidth=0)
ax.fill_between(x,y6,y7,color='#f3f700',label='SMR elec w/wo CCS',linewidth=0)
ax.fill_between(x,y7,y8,color='#8fe977',label='electrolyse',linewidth=0)

ax.set_xlabel('Années de référence')
ax.set_ylabel('kt/an')
plt.grid(True, axis='x', linewidth=0.1)
plt.title('Evolution de la demande hydrogène')

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0*2.5, box.width , box.height* 0.85])
# Put a legend to the right of the current axis
ax.legend(ncol=2,loc='lower center', bbox_to_anchor=(0.5, -0.44))

plt.savefig('Evolution demande')
plt.show()

os.chdir('..')
os.chdir('..')
os.chdir('..')
##endregion