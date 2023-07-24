import os
import numpy as np
import pandas as pd
import csv

os.sys.path.append(r'../')

from Functions.f_extract_data import extract_costs
from Functions.f_graphicTools import plot_costs
from Models.scenario_creation import scenarioDict

#First : execute runPACA.py

outputPath='../Data/output/'

ScenarioName='Ref'
outputFolder=outputPath+ScenarioName+'_PACA'
scenario=scenarioDict[ScenarioName]

df=extract_costs(scenario,outputFolder)

plot_costs(df,outputFolder)