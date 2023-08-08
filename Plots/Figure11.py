
import os
os.sys.path.append(r'../')
from Functions.f_extract_data import extract_costs
from Functions.f_graphicTools import plot_carbonCosts
from Models.scenario_creation import scenarioDict

#First : execute runPACA.py Ref woSMR_2030 woSMR_2040 woSMR_2050

outputPath='../Data/output/'

dico_costs={}
scenarioList=['Ref','woSMR_2030', 'woSMR_2040', 'woSMR_2050']
scenarioNames=['Ref','No SMR from 2030','No SMR from 2040','No SMR from 2050']
for s in scenarioList:
    outputFolder=outputPath+s+'_PACA'
    dico_costs[s]=extract_costs(scenarioDict[s],outputFolder)

plot_carbonCosts(dico_costs,scenarioNames)