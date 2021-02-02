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
def Transport_H2(Parameters,T,d,D) :

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

    Parameters = Parameters.set_index('SYMBOLE')



    return

#endregion

#region Loading arguments
InputFolder = 'Data/Input/'
Parameters= pd.read_csv(InputFolder+'Transport'+'.csv',
                                sep=',',decimal='.',skiprows=0)
T=1
d=100
D=0.1
Transport_H2(Parameters,T,d,D)


#endregion