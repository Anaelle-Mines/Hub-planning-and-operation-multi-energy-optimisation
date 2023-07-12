import numpy as np
import pandas as pd
import csv
import datetime
import copy
import matplotlib.pyplot as plt
import seaborn as sb

from scenario_creation import *
from Functions.f_extract_data import *


allCosts=pd.DataFrame()
allEnergy=pd.DataFrame()
allCapa=pd.DataFrame()

outputPath='Data/output/'

ScenarioList=['Ref']

for ScenarioName in ScenarioList :
    outputFolder = outputPath + ScenarioName
    scenario = ScenarioName
    temp1=extract_costs(scenarioDict[ScenarioName],outputFolder)
    temp2=extract_energy(scenarioDict[ScenarioName],outputFolder)
    temp3=extract_capa(scenarioDict[ScenarioName],outputFolder)

    for tech in temp1.keys():
        temp1[tech]['SCENARIO']=ScenarioName
        temp1[tech]['TECHNOLOGIE']=tech
        temp1[tech]=temp1[tech].reset_index().set_index(['YEAR','SCENARIO','TECHNOLOGIE'])
        allCosts=allCosts.append(temp1[tech])
    
    temp2['SCENARIO']=ScenarioName
    temp2=temp2.reset_index().set_index(['YEAR_op','SCENARIO'])
    allEnergy=allEnergy.append(temp2)
    temp3['SCENARIO']=ScenarioName
    temp3=temp3.reset_index().set_index(['YEAR_op','SCENARIO'])
    allCapa=allCapa.append(temp3)

allCosts.to_csv(outputPath+'allCosts.csv')
allEnergy.to_csv(outputPath+'allEnergy.csv')
allCapa.to_csv(outputPath+'allCapa.csv')