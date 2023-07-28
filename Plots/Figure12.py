
import os
os.sys.path.append(r'../')
from Functions.f_graphicTools import plot_H2Mean2050
from Models.scenario_creation import scenarioDict

#First : execute runPACA.py Re_inf

outputPath='../Data/output/'
scenario='Re_inf'
outputFolder=outputPath+scenario+'_PACA'

plot_H2Mean2050(scenarioDict[scenario],outputFolder)