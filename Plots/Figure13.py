
import os
os.sys.path.append(r'../')
from Functions.f_extract_data import extract_costs
from Functions.f_graphicTools import plot_total_co2_emissions_and_flexSMR
from Models.scenario_creation_REsensibility import scenarioDict_RE

#First : execute runPACA.py Re_inf Re_inf_woSMR Re_inf_CCS10 Re_inf_CCS10_woSMR Re_inf_TC200 Re_inf_TC200_woSMR Caverns Caverns_woSMR Caverns_CCS10 Caverns_CCS10_woSMR Caverns_TC200 Caverns_TC200_woSMR CavernREinf CavernREinf_woSMR CavernREinf_CCS10 CavernREinf_CCS10_woSMR CavernREinf_TC200 CavernREinf_TC200_woSMR

outputPath='../Data/output/'

scenarioList=list(scenarioDict_RE.keys())
del scenarioList[0]
scenarioNames=['Re_inf','Re_inf_TC200','Re_inf_CCS10','Caverns','Caverns_TC200','Caverns_CCS10','CavernREinf','CavernREinf_TC200','CavernREinf_CCS10']
labels=['Unbounded','Unbounded TC=200€/t','Unbounded CCS=10€/t','Caverns','Caverns TC=200€/t','Caverns CCS=10€/t','Unbounded + Caverns','Unbounded \n+ Caverns TC=200€/t','Unbounded \n+ Caverns CCS=10€/t']

dico_costs={}
for s in scenarioList:
    outputFolder=outputPath+s+'_PACA'
    dico_costs[s]=extract_costs(scenarioDict_RE[s],outputFolder)

flex=plot_total_co2_emissions_and_flexSMR(dico_costs,scenarioNames,labels,outputPath=outputPath)