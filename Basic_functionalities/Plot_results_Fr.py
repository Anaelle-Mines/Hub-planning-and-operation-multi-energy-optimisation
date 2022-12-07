#region Importation of modules
import os
import numpy as np
import pandas as pd
import csv

from Functions.f_graphicTools import *

#endregion

outputPath='Data/output/'

ScenarioName='Ref' # Possible Choice : 'Ref', 'eSMR', 'EnR', 'Grid', 'GN', 'BG', 'EnR+','Crack'
outputFolder=outputPath+ScenarioName
outputFolderFr=outputPath+ScenarioName+'_Fr'

#pd.set_option('display.max_columns', 500)
EnR,Fossils=plot_mixProdElec(outputFolderFr)


timeRange=range(1,1+24*7)
plot_hourlyProduction(2040,timeRange,outputFolderFr)

plot_monotone(outputFolderFr)