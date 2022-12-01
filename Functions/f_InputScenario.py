#region importation of modules
import os
if os.path.basename(os.getcwd())=="BasicFunctionalities":
    os.chdir('..') ## to work at project root  like in any IDE
import sys
if sys.platform != 'win32':
    myhost = os.uname()[1]
else : myhost = ""
if (myhost=="jupyter-sop"):
    ## for https://jupyter-sop.mines-paristech.fr/ users, you need to
    #  (1) run the following in a terminal
    if (os.system("/opt/mosek/9.2/tools/platform/linux64x86/bin/lmgrd -c /opt/mosek/9.2/tools/platform/linux64x86/bin/mosek.lic -l lmgrd.log")==0):
        os.system("/opt/mosek/9.2/tools/platform/linux64x86/bin/lmutil lmstat -c 27007@127.0.0.1 -a")
    #  (2) definition of license
    os.environ["MOSEKLM_LICENSE_FILE"] = '@jupyter-sop'

import numpy as np
import pandas as pd
import csv
#import docplex
import datetime
import copy
import plotly
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn import linear_model
import sys
import time
import datetime
import seaborn as sb

from Functions.f_multiResourceModels import *
from Functions.f_optimization import *
from Functions.f_optimModel_elec import *

# Change this if you have other solvers obtained here
## https://ampl.com/products/solvers/open-source/
## for eduction this site provides also several professional solvers, that are more efficient than e.g. cbc
#endregion

def modif_CAPEX(Scenario,techno,Stechno,Param_list,Scenario_list) :
    techno=techno.append(techno.loc[2020].rename(index={2020:2030}))
    techno=techno.append(techno.loc[2020].rename(index={2020:2040}))
    techno=techno.set_index('TECHNOLOGIES',append=True)
    Stechno=Stechno.append(Stechno.loc[2020].rename(index={2020:2030}))
    Stechno=Stechno.append(Stechno.loc[2020].rename(index={2020:2040}))
    Stechno = Stechno.set_index('STOCK_TECHNO', append=True)

    # Thermique
    for yr in [2030,2040]:
        techno.loc[(yr,'TAC'), 'investCost'] = 600000
        techno.loc[(yr, 'TAC'), 'operationCost'] = 36000
        techno.loc[(yr, 'CCG'), 'investCost'] = 900000
        techno.loc[(yr, 'CCG'), 'operationCost'] = 36000
        techno.loc[(yr, 'Coal_p'), 'investCost'] = 1100000
        techno.loc[(yr, 'Coal_p'), 'operationCost'] = 36000

    # EnR
    for tech in ['Solar','WindOnShore','WindOffShore','WindOffShore_flot']:
        techno.loc[(2040,tech),'investCost']=techno.loc[(2020,tech),'investCost']*Param_list['CAPEX_EnR'][Scenario_list[Scenario]['CAPEX_EnR']]
        techno.loc[(2030,tech),'investCost']=(techno.loc[(2020,tech),'investCost']+techno.loc[(2040,tech),'investCost'])/2
        techno.loc[(2040,tech),'operationCost']=techno.loc[(2020,tech),'operationCost']*Param_list['CAPEX_EnR'][Scenario_list[Scenario]['CAPEX_EnR']]
        techno.loc[(2030,tech),'operationCost']=(techno.loc[(2020,tech),'operationCost']+techno.loc[(2040,tech),'operationCost'])/2

    # Electrolysis
    techno.loc[(2040,'electrolysis_PEMEL'),'investCost']=techno.loc[(2020,'electrolysis_PEMEL'),'investCost']*(Param_list['CAPEX_elec'][Scenario_list[Scenario]['CAPEX_elec']]-0.15)
    techno.loc[(2030,'electrolysis_PEMEL'),'investCost']=(techno.loc[(2020,'electrolysis_PEMEL'),'investCost']+techno.loc[(2040,'electrolysis_PEMEL'),'investCost'])/2
    techno.loc[(2040,'electrolysis_PEMEL'),'operationCost']=techno.loc[(2040,'electrolysis_PEMEL'),'investCost']*0.03
    techno.loc[(2030,'electrolysis_PEMEL'),'operationCost']=techno.loc[(2030,'electrolysis_PEMEL'),'investCost']*0.03
    techno.loc[(2040,'electrolysis_AEL'),'investCost']=techno.loc[(2020,'electrolysis_AEL'),'investCost']*Param_list['CAPEX_elec'][Scenario_list[Scenario]['CAPEX_elec']]
    techno.loc[(2030,'electrolysis_AEL'),'investCost']=(techno.loc[(2020,'electrolysis_AEL'),'investCost']+techno.loc[(2040,'electrolysis_AEL'),'investCost'])/2
    techno.loc[(2040,'electrolysis_AEL'),'operationCost']=techno.loc[(2040,'electrolysis_AEL'),'investCost']*0.03
    techno.loc[(2030,'electrolysis_AEL'),'operationCost']=techno.loc[(2030,'electrolysis_AEL'),'investCost']*0.03


    # eSMR
    techno.loc[(2040,'SMR_elec'),'investCost']=Param_list['CAPEX_eSMR'][Scenario_list[Scenario]['CAPEX_eSMR']]*100000
    techno.loc[(2030,'SMR_elec'),'investCost']=(techno.loc[(2020,'SMR_elec'),'investCost']+techno.loc[(2040,'SMR_elec'),'investCost'])/2
    techno.loc[(2040,'SMR_elec'),'operationCost']=techno.loc[(2040,'SMR_elec'),'investCost']*0.05
    techno.loc[(2030,'SMR_elec'),'operationCost']=techno.loc[(2030,'SMR_elec'),'investCost']*0.05

    # cracking
    techno.loc[(2020, 'cracking'), 'powerCost'] = -Param_list['BlackC_price'][Scenario_list[Scenario]['BlackC_price']]
    techno.loc[(2030, 'cracking'), 'powerCost'] = -Param_list['BlackC_price'][Scenario_list[Scenario]['BlackC_price']]
    techno.loc[(2040,'cracking'),'powerCost'] = -Param_list['BlackC_price'][Scenario_list[Scenario]['BlackC_price']]

    # CCS
    for tech in ['CCS1','CCS2']:
        techno.loc[(2040,tech),'investCost']=techno.loc[(2020,tech),'investCost']*Param_list['CAPEX_CCS'][Scenario_list[Scenario]['CAPEX_CCS']]
        techno.loc[(2030,tech),'investCost']=(techno.loc[(2020,tech),'investCost']+techno.loc[(2040,tech),'investCost'])/2

    # SMR CCS1
    for yr in [2030,2040]:
        techno.loc[(yr,'SMR_CCS1'),'investCost']=techno.loc[(yr,'SMR_class'),'investCost']+techno.loc[(yr,'CCS1'),'investCost']
        techno.loc[(yr,'SMR_CCS1'),'operationCost']=techno.loc[(yr,'SMR_CCS1'),'investCost']*0.05

    # SMR CCS2
    for yr in [2030,2040]:
        techno.loc[(yr,'SMR_CCS2'),'investCost']=techno.loc[(yr,'SMR_class'),'investCost']+techno.loc[(yr,'CCS1'),'investCost']+techno.loc[(yr,'CCS2'),'investCost']
        techno.loc[(yr,'SMR_CCS2'),'operationCost']=techno.loc[(yr,'SMR_CCS2'),'investCost']*0.05

    # eSMR CCS1
    for yr in [2030,2040]:
        techno.loc[(yr,'SMR_elecCCS1'),'investCost']=techno.loc[(yr,'SMR_elec'),'investCost']+techno.loc[(yr,'CCS1'),'investCost']
        techno.loc[(yr,'SMR_elecCCS1'),'operationCost']=techno.loc[(yr,'SMR_elecCCS1'),'investCost']*0.05

    return techno,Stechno

def capacity_Lim(Scenario,techno,Stechno,ElecMix,Param_list,Scenario_list):
    TECH_Fr=['OldNuke','Solar', 'WindOnShore','WindOffShore','CCG','NewNuke','WindOffShore','TAC','HydroRiver','HydroReservoir','curtailment','Interco','Coal_p']
    TECH_SMR=['Solar', 'WindOnShore','WindOffShore_flot','SMR_class_ex','SMR_class','SMR_elec','SMR_elecCCS1','SMR_CCS1','SMR_CCS2','CCS1','CCS2','electrolysis_PEMEL','electrolysis_AEL','cracking']
    STECH_Fr=['Battery','STEP']
    STECH_SMR=['Battery','tankH2_G']
    technoFr=techno.loc[slice(None),TECH_Fr,:]
    technoSMR=techno.loc[slice(None),TECH_SMR,:]
    StechnoFr=Stechno.loc[(slice(None),STECH_Fr),:]
    StechnoSMR=Stechno.loc[(slice(None),STECH_SMR),:]

    for yr in [2020,2030,2040] :
        dic={2020:0,2030:1,2040:2}
        if yr==2020 :
            for tech in TECH_Fr :
                technoFr.loc[(yr,tech),'minCapacity']=ElecMix[Param_list['MixElec'][Scenario_list[Scenario]['MixElec']]][tech][dic[yr]]*1000
                technoFr.loc[(yr, tech),'maxCapacity'] = ElecMix[Param_list['MixElec'][Scenario_list[Scenario]['MixElec']]][tech][dic[yr]] * 1000
                technoFr.loc[(yr, tech),'capacityLim'] = ElecMix[Param_list['MixElec'][Scenario_list[Scenario]['MixElec']]][tech][dic[yr]] * 1000
        else :
            for tech in TECH_Fr :
                if ElecMix[Param_list['MixElec'][Scenario_list[Scenario]['MixElec']]][tech][dic[yr]]*1000>ElecMix[Param_list['MixElec'][Scenario_list[Scenario]['MixElec']]][tech][dic[yr]-1]*1000 :
                    technoFr.loc[(yr, tech),'minCapacity'] = ElecMix[Param_list['MixElec'][Scenario_list[Scenario]['MixElec']]][tech][dic[yr]]*1000 - ElecMix[Param_list['MixElec'][Scenario_list[Scenario]['MixElec']]][tech][dic[yr]-1]*1000
                    technoFr.loc[(yr, tech),'maxCapacity'] = ElecMix[Param_list['MixElec'][Scenario_list[Scenario]['MixElec']]][tech][dic[yr]]*1000 - ElecMix[Param_list['MixElec'][Scenario_list[Scenario]['MixElec']]][tech][dic[yr]-1]*1000
                else :
                    technoFr.loc[(yr, tech), 'minCapacity']=0
                    technoFr.loc[(yr, tech), 'maxCapacity']=0
                if tech=='Solar' :
                    if ElecMix[Param_list['MixElec'][Scenario_list[Scenario]['MixElec']]][tech][2]*1000 > ElecMix[Param_list['MixElec'][Scenario_list[Scenario]['MixElec']]][tech][1]*1000 :
                        technoFr.loc[(2040, tech),'minCapacity'] = ElecMix[Param_list['MixElec'][Scenario_list[Scenario]['MixElec']]][tech][2]*1000 - ElecMix[Param_list['MixElec'][Scenario_list[Scenario]['MixElec']]][tech][1]*1000 + ElecMix[Param_list['MixElec'][Scenario_list[Scenario]['MixElec']]][tech][0]*1000
                        technoFr.loc[(2040, tech),'maxCapacity'] = ElecMix[Param_list['MixElec'][Scenario_list[Scenario]['MixElec']]][tech][2]*1000 - ElecMix[Param_list['MixElec'][Scenario_list[Scenario]['MixElec']]][tech][1]*1000 + ElecMix[Param_list['MixElec'][Scenario_list[Scenario]['MixElec']]][tech][0]*1000
                    else :
                        technoFr.loc[(2040, tech), 'minCapacity']=0
                        technoFr.loc[(2040, tech), 'maxCapacity']=0
                technoFr.loc[(yr, tech), 'capacityLim'] = ElecMix[Param_list['MixElec'][Scenario_list[Scenario]['MixElec']]][tech][dic[yr]] * 1000

    for yr in [2020,2030,2040] :
        dic = {2020: 0, 2030: 1, 2040: 2}
        for stech in STECH_Fr :
            StechnoFr.loc[(yr, stech), 'p_max'] = ElecMix[Param_list['MixElec'][Scenario_list[Scenario]['MixElec']]][stech][dic[yr]] * 1000
            if stech=='Battery' :
                StechnoFr.loc[(yr, stech), 'c_max'] = ElecMix[Param_list['MixElec'][Scenario_list[Scenario]['MixElec']]][stech][dic[yr]] * 10000
            elif stech=='STEP' :
                StechnoFr.loc[(yr, stech), 'c_max'] = ElecMix[Param_list['MixElec'][Scenario_list[Scenario]['MixElec']]][stech][dic[yr]] * 30000

    for yr in [2020,2030,2040] :
        for tech in TECH_SMR :
            if tech == 'SMR_class_ex' :
                if yr == 2020 :
                    technoSMR.loc[(yr, tech), 'minCapacity'] = 320
                    technoSMR.loc[(yr, tech), 'maxCapacity'] = 320
                    technoSMR.loc[(yr, tech), 'capacityLim'] = 320
                else :
                    technoSMR.loc[(yr, tech), 'minCapacity'] = 0
                    technoSMR.loc[(yr, tech), 'maxCapacity'] = 0
                    technoSMR.loc[(yr, tech), 'capacityLim'] = 320
            elif tech == 'Solar':
                technoSMR.loc[(yr, tech), 'minCapacity'] = 0
                technoSMR.loc[(yr, tech), 'capacityLim'] = Param_list['Local_RE'][Scenario_list[Scenario]['Local_RE']][0]
                technoSMR.loc[(yr, tech), 'maxCapacity'] = technoSMR.loc[(yr, tech), 'capacityLim']*2/3
            elif tech == 'WindOnShore':
                technoSMR.loc[(yr, tech), 'minCapacity'] = 0
                technoSMR.loc[(yr, tech), 'capacityLim'] = Param_list['Local_RE'][Scenario_list[Scenario]['Local_RE']][1]
                technoSMR.loc[(yr, tech), 'maxCapacity'] = technoSMR.loc[(yr, tech), 'capacityLim']*2/3
            elif tech == 'WindOffShore':
                technoSMR.loc[(yr, tech), 'minCapacity'] = 0
                technoSMR.loc[(yr, tech), 'capacityLim'] = Param_list['Local_RE'][Scenario_list[Scenario]['Local_RE']][2]
                technoSMR.loc[(yr, tech), 'maxCapacity'] = technoSMR.loc[(yr, tech), 'capacityLim']*2/3
            else :
                technoSMR.loc[(yr, tech), 'minCapacity'] = 0
                technoSMR.loc[(yr, tech), 'maxCapacity'] = 100000
                technoSMR.loc[(yr, tech), 'capacityLim'] = 100000

    for yr in [2020,2030,2040] :
        for stech in STECH_SMR :
            StechnoSMR.loc[(yr, stech), 'p_max'] = 1000
            if stech=='Battery' :
                StechnoSMR.loc[(yr, stech), 'c_max'] = 10000
            elif stech=='tankH2_G' :
                StechnoSMR.loc[(yr, stech), 'c_max'] = 100000

    return technoFr, technoSMR, StechnoFr, StechnoSMR

def Res_Price(Scenario,Res_ref,marketPrice,carbonContent,Param_list,Scenario_list,type='Fr'):
    Res=Res_ref.copy()
    dic = {2030: 2, 2040: 1, 2050: 0}
    #GazNat
    Res.loc[(slice(None),slice(None),'gazNat'),'importCost']=Res_ref.loc[(slice(None),slice(None),'gazNat'),'importCost']*Param_list['Gaznat_price'][Scenario_list[Scenario]['Gaznat_price']]
    #BioGaz
    for yr in [2030,2040,2050] :
        Res.loc[(yr, slice(None), 'gazBio'), 'importCost']=Param_list['Biogaz_price'][Scenario_list[Scenario]['Biogaz_price']]+dic[yr]*20
    #electricity
    if type=='SMR':
        Res.loc[(slice(None), slice(None), 'electricity'),'importCost'] = marketPrice.loc[(slice(None),slice(None))]['NewPrice_NonAct']
        Res.loc[(slice(None), slice(None), 'electricity'),'emission'] = carbonContent.loc[(slice(None),slice(None))]['carbonContent']

    return Res

def create_data(Scenario,ScenarioName,Scenario_list,Param_list,ElecMix,solver='mosek',InputFolder = 'Data/Input/',OutputFolder = 'Data/output/'):

    ImportFolder = 'Data/Input_reference_v3/'
    areaConsumption, areaConsumptionSMR, availabilityFactorFr, availabilityFactorPACA, Calendrier, Convfac, sConvfac, Economics, Stechno_ref, techno_ref, Res_ref, marketPrice_ref, carbon_ref = loading_reference(ImportFolder)
    carbonTax_ref = {2: 0.1, 3: 0.115, 4: 0.13}

    # Creation of data

    techno, Stechno = modif_CAPEX(Scenario, techno_ref, Stechno_ref, Param_list, Scenario_list)
    technoFr, technoSMR, StechnoFr, StechnoSMR = capacity_Lim(Scenario, techno, Stechno, ElecMix, Param_list,Scenario_list)
    ResFr = Res_Price(Scenario, Res_ref, marketPrice_ref, carbon_ref, Param_list, Scenario_list)
    carbonTax = carbonTax_ref
    carbonTax[4] = Param_list['CarbonTax'][Scenario_list[Scenario]['CarbonTax']] / 1000
    carbonTax[3] = (carbonTax[2] + carbonTax[4]) / 2
    carbonTax_pd=pd.DataFrame(carbonTax.values(),index=carbonTax.keys()).rename(columns={0:'carbonTax'})

    # Export csv Scenario Fr
    Zones = "Fr";
    year = '2020-2050';
    InputName = 'Input_' + ScenarioName
    os.chdir(InputFolder)
    os.mkdir(InputName)
    os.chdir(InputName)

    areaConsumption.to_csv('areaConsumption' + year + '_' + Zones + '_TIMExRESxYEAR.csv', index=True)
    availabilityFactorFr.to_csv('availabilityFactor' + year + '_' + Zones + '_TIMExTECHxYEAR.csv', index=True)
    Calendrier.to_csv('calendrierHPHC_TIME.csv', index=True)
    Convfac.to_csv('conversionFactors_RESxTECH.csv', index=True)
    sConvfac.to_csv('storageFactor_RESxSTECH.csv', index=True)
    Economics.to_csv('Economics.csv', index=True)
    ResFr.to_csv('set' + year + '_fixe_TIMExRESxYEAR.csv', index=True)
    technoFr.to_csv('set' + year + '_' + Zones + '_TECHxYEAR.csv', index=True)
    StechnoFr.to_csv('set' + year + '_' + Zones + '_STECHxYEAR.csv', index=True)
    carbonTax_pd.to_csv('CarbonTax_YEAR.csv', index=True)
    os.chdir('..')
    os.chdir('..')
    os.chdir('..')

    # Optimization Fr

    # Import parameters
    Zones = "Fr";
    year = '2020-2050';
    PrixRes = 'fixe'
    Selected_TECHNOLOGIES = ['OldNuke', 'Solar', 'WindOnShore', 'WindOffShore', 'CCG', 'NewNuke', 'WindOffShore', 'TAC','HydroRiver', 'HydroReservoir', 'curtailment', 'Interco', 'Coal_p']
    Selected_STECH = ['Battery', 'STEP']
    dic_eco = {2020: 1, 2030: 2, 2040: 3, 2050: 4}
    TransFactors = pd.DataFrame({'TECHNO1': [], 'TECHNO2': [], 'TransFactor': []}).set_index(['TECHNO1', 'TECHNO2'])
    # reading areaConsumption availabilityFactor and TechParameters CSV files
    areaConsumption, availabilityFactor, TechParameters, conversionFactor, ResParameters, Calendrier, StorageParameters, storageFactor, Economics,carbonTax = loadingParameters_MultiTempo(Selected_TECHNOLOGIES, Selected_STECH, InputFolder=InputFolder + 'Input_' + ScenarioName + '/', Zones=Zones,year=year, PrixRes=PrixRes, dic_eco=dic_eco)

    # Optimization
    model = GetElectricSystemModel_MultiResources_MultiTempo_SingleNode_WithStorage(areaConsumption, availabilityFactor,TechParameters, ResParameters,conversionFactor, Economics,Calendrier, StorageParameters,storageFactor, TransFactors,carbonTax)

    start_clock = time.time()
    opt = SolverFactory(solver)
    results = opt.solve(model)
    end_clock = time.time()
    Clock = end_clock - start_clock
    print('temps de calcul = ', Clock, 's')

    Variables = getVariables_panda(model)
    Constraints = getConstraintsDual_panda(model)

    # save results
    d = datetime.date.today()
    SimulName = str(d.year) + '-' + str(d.month) + '-' + str(d.day) + '_' + ScenarioName + '_Fr'
    var_name = list(Variables.keys())
    cons_name = list(Constraints.keys())
    os.chdir(OutputFolder)
    os.mkdir(SimulName)
    os.chdir(SimulName)

    for var in var_name:
        Variables[var].to_csv(var + '_' + SimulName + '.csv', index=True)
    for cons in cons_name:
        Constraints[cons].to_csv(cons + '_' + SimulName + '.csv', index=True)

    dic_an = {1: 2020, 2: 2030, 3: 2040, 4: 2050}
    Prix_elec = round(Constraints['energyCtr'].set_index('RESOURCES').loc['electricity'].set_index('YEAR_op').rename(index=dic_an),2)
    Carbon = Variables['carbon_Pvar'].set_index(['YEAR_op', 'TIMESTAMP'])
    Carbon.loc[Carbon['carbon_Pvar']<0.01]=0
    Prod_elec = Variables['power_Dvar'].groupby(['YEAR_op', 'TIMESTAMP']).sum()
    Carbon_content = Carbon['carbon_Pvar'] / Prod_elec['power_Dvar']
    Carbon_content = round(Carbon_content.reset_index().set_index('YEAR_op').rename(index=dic_an,columns={0: 'carbonContent'}).set_index(
        'TIMESTAMP', append=True))
    Prix_elec.to_csv('elecPrice_' + SimulName + '.csv', index=True)
    Carbon_content.to_csv('carbon_' + SimulName + '.csv', index=True)
    os.chdir('..')
    os.chdir('..')
    os.chdir('..')

    return

def ElecPrice_optim(scenario,solver='mosek',outputFolder = 'Data/output/'):

    inputDict = loadScenario(scenario, False)


    elecProd = pd.read_csv(outputFolder+'/power_Dvar.csv').drop(columns='Unnamed: 0').set_index(['YEAR_op','TIMESTAMP', 'TECHNOLOGIES'])
    carbon_content = pd.read_csv(outputFolder+'/carbon.csv')
    elec_price = pd.read_csv(outputFolder+'/elecPrice.csv')

    YEAR = sorted(list(elecProd.index.get_level_values('YEAR_op').unique()))
    dy=YEAR[1]-YEAR[0]
    y0=YEAR[0]-dy

    marketPrice = elec_price.set_index(['YEAR_op','TIMESTAMP'])
    marketPrice['LastCalled'] = ""

    for i in marketPrice.index:
        if elecProd.loc[(i[0], i[1], 'Coal_p')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'Coal_p'
        elif elecProd.loc[(i[0], i[1], 'TAC')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'TAC'
        elif elecProd.loc[(i[0], i[1], 'CCG')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'CCG'
        elif elecProd.loc[(i[0], i[1], 'OldNuke')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'OldNuke'
        elif elecProd.loc[(i[0], i[1], 'NewNuke')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'NewNuke'
        elif elecProd.loc[(i[0], i[1], 'WindOffShore')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'WindOffShore'
        elif elecProd.loc[(i[0], i[1], 'WindOnShore')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'WindOnShore'
        elif elecProd.loc[(i[0], i[1], 'Solar')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'Solar'
        else:
            marketPrice.LastCalled.loc[i] = 'Undetermined'

    capaCosts = pd.read_csv(outputFolder+'/capacityCosts_Pvar.csv').drop(columns='Unnamed: 0').set_index(['YEAR_op','TECHNOLOGIES'])
    carbonContent = carbon_content.set_index(['YEAR_op','TIMESTAMP'])
    ResParameters = inputDict['resParameters'].loc[(slice(None), slice(None), ['electricity', 'gaz', 'hydrogen', 'uranium'])].reset_index().rename(columns={'YEAR':'YEAR_op'}).set_index(['YEAR_op','TIMESTAMP','RESOURCES'])
    gazPrice = (pd.DataFrame(pd.read_csv(outputFolder+'/importCosts_Pvar.csv').drop(columns='Unnamed: 0').set_index(['YEAR_op', 'RESOURCES']).loc[(slice(None), ['gazBio', 'gazNat']), 'importCosts_Pvar']).fillna(0).groupby('YEAR_op').sum()).join(pd.DataFrame(pd.read_csv(outputFolder+'/importation_Dvar.csv').groupby(['YEAR_op', 'RESOURCES']).sum().drop(columns=['Unnamed: 0','TIMESTAMP']).loc[(slice(None), ['gazBio', 'gazNat']), 'importation_Dvar']).fillna(0).groupby('YEAR_op').sum())
    gazPrice['gazPrice'] = (gazPrice['importCosts_Pvar'] / gazPrice['importation_Dvar']).fillna(0)
    for yr in YEAR: ResParameters.loc[(yr, slice(None), ['gaz']), 'importCost'] = gazPrice.loc[yr]['gazPrice']

    model = GetElectricPriceModel(elecProd, marketPrice, ResParameters, inputDict['techParameters'], capaCosts, carbonContent,inputDict['conversionFactor'],inputDict['carbonTax'], isAbstract=False)
    opt = SolverFactory(solver)
    results = opt.solve(model)
    elec_var = getVariables_panda(model)

    AjustFac = elec_var['AjustFac'].set_index(['YEAR_op', 'TECHNOLOGIES']).fillna(0)
    AjustFac.loc[AjustFac['AjustFac'] < 0] = 0
    NewPrice = []
    for i in marketPrice.index:
        Ajustement = AjustFac.loc[i[0], marketPrice.LastCalled.loc[i]]
        NewPrice.append(marketPrice.loc[i]['energyCtr'] + Ajustement['AjustFac'])

    marketPrice['NewPrice'] = NewPrice
    test = marketPrice.NewPrice == marketPrice.energyCtr
    print(test.loc[test == False])

    # test
    test='energyCtr'
    #test = 'NewPrice'

    TECHNO = list(elecProd.index.get_level_values('TECHNOLOGIES').unique())
    TIMESTAMP = list(elecProd.index.get_level_values('TIMESTAMP').unique())
    RES = list(ResParameters.index.get_level_values('RESOURCES').unique())
    RES.remove('hydrogen')
    RES.remove('electricity')
    elecProd['Revenus'] = elecProd['power_Dvar'] * marketPrice[test]
    Revenus = elecProd.Revenus.groupby(['YEAR_op', 'TECHNOLOGIES']).sum()
    TotalCosts = elecProd.groupby(['YEAR_op', 'TECHNOLOGIES']).sum().drop(columns=['power_Dvar', 'Revenus'])

    for tech in TECHNO:
        id_year=[]
        id_time=[]
        for i in np.arange(len(YEAR)):
            id_year=id_year+[YEAR[i]]*8760
            id_time=id_time+TIMESTAMP
        df = pd.DataFrame({'YEAR_op': id_year,'TIMESTAMP': id_time}).set_index(['YEAR_op', 'TIMESTAMP'])
        for res in RES:
            df[res] = elecProd['power_Dvar'].loc[(slice(None), slice(None), tech)] * ResParameters['importCost'].loc[
                (slice(None), slice(None), res)] * (-inputDict['conversionFactor']['conversionFactor'].loc[(res, tech)])
        for y in YEAR:
            TotalCosts.loc[(y, tech), 'import'] = df.groupby('YEAR_op').sum().sum(axis=1).loc[y]

    for y in YEAR:
        for tech in TECHNO:
            TotalCosts.loc[(y, tech), 'variable'] = elecProd['power_Dvar'].groupby(['YEAR_op', 'TECHNOLOGIES']).sum().loc[(y, tech)] * inputDict['techParameters']['powerCost'].loc[(y - dy, tech)]
            TotalCosts.loc[(y, tech), 'carbon'] = elecProd['power_Dvar'].groupby(['YEAR_op', 'TECHNOLOGIES']).sum().loc[(y, tech)] * inputDict['carbonTax']['carbonTax'].loc[y]

    TotalCosts['capacity'] = capaCosts['capacityCosts_Pvar']
    TotalCosts['total'] = TotalCosts['import'] + TotalCosts['variable'] + TotalCosts['carbon'] + TotalCosts['capacity']
    Difference = TotalCosts['total'] - Revenus
    delta = Difference.loc[(Difference != 0)]
    MW = elecProd.groupby(['YEAR_op', 'TECHNOLOGIES']).sum()['power_Dvar']
    MW = MW.loc[(MW != 0)]
    raport = -delta / MW
    print(elec_var['AjustFac'].loc[elec_var['AjustFac']['AjustFac'] > 0])
    print(raport)

    marketPrice.loc[marketPrice['NewPrice'] < 0] = 0

    for yr in YEAR :
        marketPrice.loc[(yr, slice(None)),'OldPrice_NonAct'] = marketPrice.loc[(yr, slice(None)),'energyCtr'] / ((1 + scenario['economicParameters']['discountRate'].loc[0]) ** (-(yr-y0)))
        marketPrice.loc[(yr,slice(None)),'NewPrice_NonAct']=marketPrice.loc[(yr,slice(None)),'NewPrice']/ ((1 + scenario['economicParameters']['discountRate'].loc[0]) ** (-(yr-y0)))

    marketPrice = round(marketPrice.reset_index().set_index(['YEAR_op','TIMESTAMP']),2)

    AjustFac.to_csv(outputFolder+'/priceCorrection.csv')
    marketPrice.to_csv(outputFolder+'/marketPrice.csv')

    return marketPrice
