
import os

import numpy as np
import pandas as pd
import csv
import copy

from Functions.f_multiResourceModels import *
from Functions.f_optimization import *

def ElecPrice_optim(scenario,IntercoOut=0,solver='mosek',outputFolder = 'Data/output/',testOnly=False):

    # scenario=scenarioFr
    # outputFolder=outputFolderFr

    inputDict = loadScenario(scenario, False)

    TECH_elec=list(inputDict['conversionTechs'].transpose().loc[inputDict['conversionTechs'].transpose()['Category']=='Electricity production'].index.unique())

    elecProd = pd.read_csv(outputFolder+'/power_Dvar.csv').drop(columns='Unnamed: 0').set_index(['YEAR_op','TIMESTAMP', 'TECHNOLOGIES']).loc[(slice(None),slice(None),TECH_elec)]

    YEAR = sorted(list(elecProd.index.get_level_values('YEAR_op').unique()))
    dy=YEAR[1]-YEAR[0]
    y0=YEAR[0]-dy

    carbon_content = pd.read_csv(outputFolder+'/carbon.csv')
    elec_price = pd.read_csv(outputFolder+'/elecPrice.csv')
    capaCosts = pd.read_csv(outputFolder + '/capacityCosts_Pvar.csv').drop(columns='Unnamed: 0').set_index(['YEAR_op', 'TECHNOLOGIES'])
    carbonContent = carbon_content.set_index(['YEAR_op', 'TIMESTAMP'])
    ResParameters = inputDict['resParameters'].loc[
        (slice(None), slice(None), ['electricity', 'gaz', 'hydrogen', 'uranium'])].reset_index().rename(
        columns={'YEAR': 'YEAR_op'}).set_index(['YEAR_op', 'TIMESTAMP', 'RESOURCES'])
    gazPrice = (pd.DataFrame(pd.read_csv(outputFolder + '/importCosts_Pvar.csv').drop(columns='Unnamed: 0').set_index(
        ['YEAR_op', 'RESOURCES']).loc[(slice(None), ['gazBio', 'gazNat']), 'importCosts_Pvar']).fillna(0).groupby(
        'YEAR_op').sum()).join(pd.DataFrame(
        pd.read_csv(outputFolder + '/importation_Dvar.csv').groupby(['YEAR_op', 'RESOURCES']).sum().drop(
            columns=['Unnamed: 0', 'TIMESTAMP']).loc[(slice(None), ['gazBio', 'gazNat']), 'importation_Dvar']).fillna(
        0).groupby('YEAR_op').sum())
    gazPrice['gazPrice'] = (gazPrice['importCosts_Pvar'] / gazPrice['importation_Dvar']).fillna(0)
    Capacities=pd.read_csv(outputFolder+'/capacity_Pvar.csv').drop(columns='Unnamed: 0').set_index(['YEAR_op', 'TECHNOLOGIES'])

    marketPrice = elec_price.set_index(['YEAR_op','TIMESTAMP'])
    for yr in YEAR:
        marketPrice.loc[(yr, slice(None)), 'OldPrice_NonAct'] = marketPrice.loc[(yr, slice(None)), 'energyCtr'] / (
                (1 + scenario['economicParameters']['discountRate'].loc[0]) ** (-(yr - y0)))
        capaCosts.loc[(yr,slice(None)),'capacityCosts_NonAct']=capaCosts.loc[(yr,slice(None)),'capacityCosts_Pvar']/ ((1 + scenario['economicParameters']['discountRate'].loc[0]) ** (-(yr - y0)))
        ResParameters.loc[(yr, slice(None), ['gaz']), 'importCost'] = gazPrice.loc[yr]['gazPrice'] / (
                (1 + scenario['economicParameters']['discountRate'].loc[0]) ** (-(yr - y0)))

    export_TECH=['OldNuke', 'WindOnShore', 'Solar', 'WindOffShore', 'NewNuke','HydroRiver']

    availableCapa=inputDict['availabilityFactor'].reset_index().rename(columns={'YEAR':'YEAR_op'}).set_index(['YEAR_op','TIMESTAMP','TECHNOLOGIES']).loc[(slice(None),slice(None),export_TECH)]
    for yr in YEAR :
        for tech in export_TECH:
            availableCapa.loc[(yr,slice(None),tech),'maxCapa']=availableCapa.loc[(yr,slice(None),tech),'availabilityFactor']*Capacities.loc[(yr,tech)]['capacity_Pvar']
    availableCapa['availableCapa']=availableCapa['maxCapa']-elecProd.loc[(slice(None),slice(None),export_TECH)]['power_Dvar']
    availableCapa.loc[availableCapa['availableCapa'] < 0]=0

    marketPrice['LastCalled'] = ""

    for i in marketPrice.index:
        if elecProd.loc[(i[0], i[1],'IntercoIn')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'IntercoIn'
        elif elecProd.loc[(i[0], i[1], 'Coal_p')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'Coal_p'
        elif elecProd.loc[(i[0], i[1], 'TAC')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'TAC'
        elif elecProd.loc[(i[0], i[1], 'CCG')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'CCG'
        elif elecProd.loc[(i[0], i[1], 'TAC_H2')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'TAC_H2'
        elif elecProd.loc[(i[0], i[1], 'CCG_H2')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'CCG_H2'
        elif elecProd.loc[(i[0], i[1], 'OldNuke')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'OldNuke'
        elif elecProd.loc[(i[0], i[1], 'NewNuke')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'NewNuke'
        elif elecProd.loc[(i[0], i[1], 'HydroRiver')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'HydroRiver'
        elif elecProd.loc[(i[0], i[1], 'WindOffShore')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'WindOffShore'
        elif elecProd.loc[(i[0], i[1], 'WindOnShore')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'WindOnShore'
        elif elecProd.loc[(i[0], i[1], 'Solar')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'Solar'
        else:
            marketPrice.LastCalled.loc[i] = 'Undetermined'

    if testOnly==False :

        model = GetElectricPriceModel(elecProd,availableCapa,IntercoOut, marketPrice, ResParameters, inputDict['techParameters'].loc[(slice(None),TECH_elec),slice(None)], capaCosts.loc[(slice(None),TECH_elec),slice(None)], carbonContent,inputDict['conversionFactor'].loc[(slice(None),TECH_elec),slice(None)],inputDict['carbonTax'], isAbstract=False)
        opt = SolverFactory(solver)
        opt.solve(model)
        elec_var = getVariables_panda(model)

        AjustFac = elec_var['AjustFac'].set_index(['YEAR_op', 'TECHNOLOGIES']).fillna(0)
        AjustFac.loc[AjustFac['AjustFac'] < 0] = 0
        NewPrice_NonAct = []
        for i in marketPrice.index:
            Ajustement = AjustFac.loc[i[0], marketPrice.LastCalled.loc[i]]
            NewPrice_NonAct.append(marketPrice.loc[i]['OldPrice_NonAct'] + Ajustement['AjustFac'])

        marketPrice['NewPrice_NonAct'] = NewPrice_NonAct
        print(elec_var['AjustFac'].loc[elec_var['AjustFac']['AjustFac'] > 0])
        AjustFac.to_csv(outputFolder + '/priceCorrection.csv')

        marketPrice.loc[marketPrice['NewPrice_NonAct'] < 0] = 0

    marketPrice = round(marketPrice.reset_index().set_index(['YEAR_op', 'TIMESTAMP']), 2)

    marketPrice.to_csv(outputFolder + '/marketPrice.csv')

    if testOnly==False : return AjustFac,marketPrice,elec_var,elecProd
    else : return marketPrice,elecProd

