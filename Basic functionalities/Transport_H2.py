# region Importation of modules
import os

if os.path.basename(os.getcwd()) == "Basic functionalities":
    os.chdir('..')  ## to work at project root  like in any IDE

import numpy as np
import pandas as pd
import csv

import datetime
import copy
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import linear_model
import sys

from Functions.f_operationModels import *
from Functions.f_optimization import *
from Functions.f_graphicalTools import *

# endregion

#region Model
def Transport_H2(Parameters,T,d,D,r) :

    # Set variables
    Parameters.loc[Parameters['SYMBOLE'] == 'T', 'valeur'] = T
    Parameters.loc[Parameters['SYMBOLE'] == 'd', 'valeur'] = d
    Parameters.loc[Parameters['SYMBOLE'] == 'D', 'valeur'] = D

    # Set parameters
    Parameters.loc[Parameters['SYMBOLE'] == 'Q', 'valeur'] =np.pi*D**2/4*Parameters.loc[Parameters['SYMBOLE'] == 'vpipe', 'valeur'].iloc[0] * 1000
    Parameters.loc[Parameters['SYMBOLE'] == 'Vtot', 'valeur'] = Parameters.loc[Parameters['SYMBOLE'] == 'Q', 'valeur'].iloc[0] * T*8760
    Parameters.loc[Parameters['SYMBOLE'] == 'Vc','valeur']=Parameters.loc[Parameters['SYMBOLE'] == 'Pc','valeur'].iloc[0]*20
    Parameters.loc[Parameters['SYMBOLE'] == 'Tc','valeur']=d*2/Parameters.loc[Parameters['SYMBOLE'] == 'vc','valeur'].iloc[0]*8760/2000
    Parameters.loc[Parameters['SYMBOLE'] == 'Vv', 'valeur']=Parameters.loc[Parameters['SYMBOLE'] == 'Q','valeur'].iloc[0]*Parameters.loc[Parameters['SYMBOLE'] == 'Tc','valeur'].iloc[0]
    Parameters.loc[Parameters['SYMBOLE'] == 'Nbc', 'valeur'] = ceil(Parameters.loc[Parameters['SYMBOLE'] == 'Vv', 'valeur'].iloc[0] / Parameters.loc[Parameters['SYMBOLE'] == 'Vc', 'valeur'].iloc[0])
    Parameters.loc[Parameters['SYMBOLE'] == 'Nbv', 'valeur'] = Parameters.loc[Parameters['SYMBOLE'] == 'Vtot', 'valeur'].iloc[0] / Parameters.loc[Parameters['SYMBOLE'] == 'Vv', 'valeur'].iloc[0]
    Parameters.loc[Parameters['SYMBOLE'] == 'dtotale', 'valeur'] = Parameters.loc[Parameters['SYMBOLE'] == 'Nbv', 'valeur'].iloc[0] * Parameters.loc[Parameters['SYMBOLE'] == 'Nbc', 'valeur'].iloc[0] * d * 2
    Parameters.loc[Parameters['SYMBOLE'] == 'Tconduite', 'valeur'] = Parameters.loc[Parameters['SYMBOLE'] == 'dtotale', 'valeur'].iloc[0] / Parameters.loc[Parameters['SYMBOLE'] == 'vc', 'valeur'].iloc[0]
    Parameters.loc[Parameters['SYMBOLE'] == 'Ic', 'valeur'] = Parameters.loc[Parameters['SYMBOLE'] == 'Nbc', 'valeur'].iloc[0] * Parameters.loc[Parameters['SYMBOLE'] == 'Ccamion', 'valeur'].iloc[0]
    Parameters.loc[Parameters['SYMBOLE'] == 'Cc', 'valeur'] = Parameters.loc[Parameters['SYMBOLE'] == 'OPEXcamion', 'valeur'].iloc[0] * Parameters.loc[Parameters['SYMBOLE'] == 'Ic', 'valeur'].iloc[0] / 100 \
                                                              + Parameters.loc[Parameters['SYMBOLE'] == 'Cchauffeur', 'valeur'].iloc[0] * Parameters.loc[Parameters['SYMBOLE'] == 'Tconduite', 'valeur'].iloc[0] \
                                                              + Parameters.loc[Parameters['SYMBOLE'] == 'Diesel', 'valeur'].iloc[0] * Parameters.loc[Parameters['SYMBOLE'] == 'Conso', 'valeur'].iloc[0] * Parameters.loc[Parameters['SYMBOLE'] == 'dtotale', 'valeur'].iloc[0]/100
    Parameters.loc[Parameters['SYMBOLE'] == 'Cpipe', 'valeur'] = 2.2e-3*D**2+0.86*D+247.5
    Parameters.loc[Parameters['SYMBOLE'] == 'Ip', 'valeur'] = Parameters.loc[Parameters['SYMBOLE'] == 'Cpipe', 'valeur'].iloc[0]*d*1000
    Parameters.loc[Parameters['SYMBOLE'] == 'Cp', 'valeur'] = Parameters.loc[Parameters['SYMBOLE'] == 'OPEXpipe', 'valeur'].iloc[0]*Parameters.loc[Parameters['SYMBOLE'] == 'Ip', 'valeur'].iloc[0]/100

    Ltcam = Parameters.loc[Parameters['SYMBOLE'] == 'Ltcamion', 'valeur'].iloc[0]
    Ltp = Parameters.loc[Parameters['SYMBOLE'] == 'Ltpipe', 'valeur'].iloc[0]

    Parameters = Parameters.set_index('SYMBOLE')

    num_pipe=0
    den_pipe=0
    for i in np.arange(1,Ltp+1,1) :
        if i==1 :
            num_pipe=num_pipe+(Parameters['valeur']['Cp']+Parameters['valeur']['Ip'])/((1+r)**i)
        else :
            num_pipe = num_pipe + Parameters['valeur']['Cp'] / ((1 + r) ** i)
        den_pipe=den_pipe+Parameters['valeur']['Vtot']*2.99/1000/((1+r)**i)
    LCOE_pipe=num_pipe/den_pipe

    num_cam=0
    den_cam=0
    for i in np.arange(1,Ltcam+1,1) :
        if i==1 :
            num_cam=num_cam+(Parameters['valeur']['Cc']+Parameters['valeur']['Ic'])/((1+r)**i)
        else :
            num_cam = num_cam + Parameters['valeur']['Cc'] / ((1 + r) ** i)
        den_cam=den_cam+Parameters['valeur']['Vtot']*2.99/1000/((1+r)**i)
    LCOE_cam=num_cam/den_cam


    return LCOE_pipe/LCOE_cam

#endregion

#region Loading arguments
InputFolder = 'Data/Input/'
Parameters= pd.read_csv(InputFolder+'Transport'+'.csv',sep=',',decimal='.',skiprows=0)

T=1
r=0.04

d_list=[1,10,20,30,50,75,100,125,150,175,200] #Distance en km
D_list=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9,1] #Diamètre pipe en m
Q_list=[np.pi*x**2/4*30000 for x in D_list] #Débit H2 en m3(N)/h
alpha_list=[]
for D in D_list :
    for d in d_list :
        alpha=Transport_H2(Parameters,T,d,D,r)
        if alpha>=1 :
            alpha_list.append(1)
        else :
            alpha_list.append(0)

alpha_matrice=np.zeros([11,14])
for i in np.arange(0,11,1) :
    alpha_matrice[i,:]=alpha_list[14*i:14*(i+1)]

alpha_df=pd.DataFrame(alpha_matrice,index=d_list,columns=Q_list)

fig = go.Figure(data=go.Heatmap(z=alpha_df,x=Q_list,y=d_list))
fig.update_layout(title='LCOE_pipe/LCOE_camion',xaxis_title='Débit de H2 (Nm3/h)',yaxis_title='Distance (km)')
plotly.offline.plot(fig, filename='Abaque transport.html')


#endregion