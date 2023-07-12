
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

# #region test coûts
# def actualisationFactor(y):
#     r = 0.04
#     dy = 10
#     y0 = 2010
#     return (1 + r) ** (-(y + dy / 2 - y0))
#
# actualisation=pd.DataFrame(columns=['YEAR_op','actualisation'],data=([y,actualisationFactor(y)] for y in [2020,2030,2040,2050])).set_index('YEAR_op')
#
#
# for ScenarioName in ScenarioList :
#
#     # outputFolder=outputPath+'Sensibility_RE/'+ScenarioName+'_testOBJ_PACA'
#     outputFolder = outputPath + ScenarioName + '_PACA'
#     # obj=(pd.read_csv(outputFolder+'/objective_Pvar.csv').drop(columns='Unnamed: 0').set_index('YEAR_op')['objective_Pvar']/actualisation['actualisation']).sum()
#     power=pd.read_csv(outputFolder+'/power_Dvar.csv').drop(columns='Unnamed: 0').groupby('TECHNOLOGIES').sum().drop(columns=['TIMESTAMP','YEAR_op'])
#     df = extract_costs(scenarioDict[ScenarioName], outputFolder)
#     c1 = (pd.read_csv(outputFolder+'/powerCosts_Pvar.csv').groupby('YEAR_op').sum()['powerCosts_Pvar']/actualisation['actualisation']).sum()
#     c1bis = sum(df[k][['powerCosts']].sum(axis=1) for k in list(df.keys())).sum()
#     c2 = (pd.read_csv(outputFolder+'/capacityCosts_Pvar.csv').groupby('YEAR_op').sum()['capacityCosts_Pvar']/actualisation['actualisation']).sum()
#     c2bis = sum(df[k][['capacityCosts']].sum(axis=1) for k in list(df.keys())).sum() + sum(df[k][['capexElec']].sum(axis=1) for k in list(df.keys())).sum()
#     c3 = (pd.read_csv(outputFolder+'/importCosts_Pvar.csv').groupby('YEAR_op').sum()['importCosts_Pvar']/actualisation['actualisation']).sum()
#     c3bis = sum(df[k][['importElec', 'importGas']].sum(axis=1) for k in list(df.keys())).sum()
#     c4 = (pd.read_csv(outputFolder+'/turpeCosts_Pvar.csv').groupby('YEAR_op').sum()['turpeCosts_Pvar']/actualisation['actualisation']).sum()
#     c4bis = sum(df[k][['TURPE']].sum(axis=1) for k in list(df.keys())).sum()
#     c5 = (pd.read_csv(outputFolder+'/storageCosts_Pvar.csv').groupby('YEAR_op').sum()['storageCosts_Pvar']/actualisation['actualisation']).sum()
#     c5bis = sum(df[k][['storageElec', 'storageH2']].sum(axis=1) for k in list(df.keys())).sum()
#     c6 = (pd.read_csv(outputFolder+'/carbonCosts_Pvar.csv').groupby('YEAR_op').sum()['carbonCosts_Pvar']/actualisation['actualisation']).sum()
#     c6bis = sum(df[k][['carbon']].sum(axis=1) for k in list(df.keys())).sum()
#     car=(pd.read_csv(outputFolder+'/carbon_Pvar.csv').groupby('YEAR_op').sum()['carbon_Pvar']).sum()/1e6 #en kt
#     costs=sum(df[k][['powerCosts', 'capacityCosts', 'capexElec', 'importElec', 'importGas', 'storageElec', 'storageH2','carbon', 'TURPE']].sum(axis=1) for k in list(df.keys())).sum()
#     prod=sum((df[k]['Prod'] * 30) for k in list(df.keys())).sum()
#     H2=(power.loc['SMR']['power_Dvar']+power.loc['SMR + CCS1']['power_Dvar']+power.loc['SMR + CCS2']['power_Dvar']+power.loc['electrolysis_AEL']['power_Dvar'])*30
#     print(ScenarioName)
#     # print('La valeur de la fonction objectif pour le scenario ',ScenarioName,' est : ',obj/1e6,' Mln€')
#     print('La valeur des coûts calculés pour le scenario ',ScenarioName,' est : ',costs/1e6,' Mln€')
#     print('La valeur des coûts calculés pour le scenario ', ScenarioName, ' est : ', (c1+c2+c3+c4+c5+c6) / 1e6, ' Mln€')
#     # print('Prix moyen de H2 pour le scenario ',ScenarioName,' est : ',obj/H2,' €/kg')
#     # print('c1 : ',c1/1e6,' VS c1bis :', c1bis/1e6)
#     # print('c2 : ', c2 / 1e6, ' VS c2bis :', c2bis / 1e6)
#     # print('c3 : ', c3 / 1e6, ' VS c3bis :', c3bis / 1e6)
#     # print('c4 : ', c4 / 1e6, ' VS c4bis :', c4bis / 1e6)
#     # print('c5 : ', c5 / 1e6, ' VS c5bis :', c5bis / 1e6)
#     print('c6 : ', c6 / 1e6, ' VS c6bis :', c6bis / 1e6)
# #endregion

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
