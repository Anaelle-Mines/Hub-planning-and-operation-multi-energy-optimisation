
import os
import numpy as np
import pandas as pd
import csv

from Functions.f_graphicTools import *
from Basic_functionalities.scenario_creation import scenarioDict
from Basic_functionalities.scenario_creation_REsensibility import scenarioDict_RE


outputPath='Data/output/'

# ScenarioList=['Ref', 'conv_SmrOnly','SmrOnly',  'Re_x2', 'Re_inf','BM_80', 'BM_100','Caverns','CavernREinf','woSMR_2030', 'woSMR_2040', 'woSMR_2050', 'CO2_10', 'CO2_100'] #'test2060', 'test','Free','EnR','Grid'

# ScenarioList=list(scenarioDict_RE.keys())
# del ScenarioList[0]
# scenarioDict=scenarioDict_RE

ScenarioName='Ref'


for ScenarioName in ScenarioList :

    outputFolder=outputPath+ScenarioName+'_PACA'
    # outputFolder = outputPath +'Sensibility_RE/'+ ScenarioName + '_PACA'

    # plot installed capacity by technologies
    capacity=plot_capacity(outputFolder)

    # plot H2 production by technologies
    energy=plot_energy(outputFolder)

    # calculation of charge factors
    chargeFactors=(energy/(capacity*8.760)).fillna(0)

    # plot evolution of SMR techno
    plot_evolve(outputFolder)

    # plot electricity weekly management
    plot_elecMean(scenarioDict[ScenarioName],outputFolder)

    # plot H2 weekly management
    plot_H2Mean(scenarioDict[ScenarioName],outputFolder)

    # plot stock level
    plot_stock(outputFolder)

    # plot carbon emissions
    plot_carbon(outputFolder)

    # plot costs
    plot_costs(extract_costs(scenarioDict[ScenarioName],outputFolder),outputFolder)

allCosts=pd.DataFrame()
allEnergy=pd.DataFrame()
allCapa=pd.DataFrame()

outputPath=outputPath #+ 'Sensibility_RE/'

for ScenarioName in ScenarioList :
    outputFolder = outputPath + ScenarioName + '_PACA'
    scenario = ScenarioName
    # temp1=extract_costs(scenarioDict[ScenarioName],outputFolder)
    # temp2=extract_energy(scenarioDict[ScenarioName],outputFolder)
    temp3=extract_capa(scenarioDict[ScenarioName],outputFolder)

    # for tech in temp1.keys():
    #     temp1[tech]['SCENARIO']=ScenarioName
    #     temp1[tech]['TECHNOLOGIE']=tech
    #     temp1[tech]=temp1[tech].reset_index().set_index(['YEAR','SCENARIO','TECHNOLOGIE'])
    #     allCosts=allCosts.append(temp1[tech])
    #
    # temp2['SCENARIO']=ScenarioName
    # temp2=temp2.reset_index().set_index(['YEAR_op','SCENARIO'])
    # allEnergy=allEnergy.append(temp2)

    temp3['SCENARIO']=ScenarioName
    temp3=temp3.reset_index().set_index(['YEAR_op','SCENARIO'])
    allCapa=allCapa.append(temp3)

# allCosts.to_csv(outputPath+'allCosts1.csv')
# allEnergy.to_csv(outputPath+'allEnergy1.csv')
allCapa.to_csv(outputPath+'allCapa1.csv')
