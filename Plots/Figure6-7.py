
import os
os.sys.path.append(r'../')
from Functions.f_extract_data import extract_energy
from Functions.f_graphicTools import plot_compare_energy_and_carbon
from Models.scenario_creation import scenarioDict

#First : execute runPACA.py CO2_10 Ref CO2_100

outputPath='../Data/output/'

dico_ener={}
scenarioList=['CO2_10','Ref','CO2_100']
scenarioNames=['CO$_2$=10€/t$_{captured}$','CO$_2$=50€/t$_{captured}$','CO$_2$=100€/t$_{captured}$']

for s in scenarioList:
    outputFolder=outputPath+s+'_PACA'
    dico_ener[s]=extract_energy(scenarioDict[s],outputFolder)

plot_compare_energy_and_carbon(dico_ener,scenarioNames,outputPath)
