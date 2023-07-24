import numpy as np
import pandas as pd
import csv
import copy
from Functions.f_multiResourceModels import loadScenario

def extract_costs(scenario,outputFolder='../Data/output/'):

    v_list = ['capacityInvest_Dvar', 'transInvest_Dvar', 'capacity_Pvar', 'capacityDel_Pvar', 'capacityDem_Dvar',
              'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
              'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar', 'carbon_Pvar', 'powerCosts_Pvar', 'capacityCosts_Pvar',
              'importCosts_Pvar', 'storageCosts_Pvar', 'turpeCosts_Pvar', 'Pmax_Pvar', 'max_PS_Dvar', 'carbonCosts_Pvar']
    Variables = {v: pd.read_csv(outputFolder + '/' + v + '.csv').drop(columns='Unnamed: 0') for v in v_list}

    inputDict = loadScenario(scenario)

    YEAR=Variables['power_Dvar'].set_index('YEAR_op').index.unique().values
    YEAR.sort()
    dy=YEAR[1]-YEAR[0]
    y0=YEAR[0]-dy

    convFac=inputDict['conversionFactor']
    Tech=inputDict['techParameters'].rename(index={2010:2020,2020:2030,2030:2040,2040:2050,2050:2060})
    Tech.sort_index(inplace=True)
    TaxC=inputDict['carbonTax']


    Grid_car=inputDict['resourceImportCO2eq'].set_index(['YEAR','TIMESTAMP'])['electricity'].reset_index().rename(columns={'electricity':'carbonContent'}).set_index(['YEAR','TIMESTAMP'])
    df1=Variables['powerCosts_Pvar'].rename(columns={'YEAR_op':'YEAR','powerCosts_Pvar':'powerCosts'}).set_index(['YEAR','TECHNOLOGIES'])
    df1['capacityCosts']=Variables['capacityCosts_Pvar'].rename(columns={'YEAR_op':'YEAR'}).set_index(['YEAR','TECHNOLOGIES'])
    df1['Prod']=Variables['power_Dvar'].rename(columns={'YEAR_op':'YEAR'}).groupby(['YEAR','TECHNOLOGIES']).sum().drop(columns=['TIMESTAMP'])
    df2=Variables['importCosts_Pvar'].rename(columns={'YEAR_op':'YEAR','importCosts_Pvar':'importCosts'}).set_index(['YEAR','RESOURCES'])
    df2['TURPE']=Variables['turpeCosts_Pvar'].rename(columns={'YEAR_op':'YEAR'}).set_index(['YEAR','RESOURCES'])
    df3=Variables['capacityCosts_Pvar'].rename(columns={'YEAR_op':'YEAR'}).set_index(['YEAR','TECHNOLOGIES'])
    df4=Variables['storageCosts_Pvar'].rename(columns={'YEAR_op':'YEAR','storageCosts_Pvar':'storageCosts'}).set_index(['YEAR','STOCK_TECHNO'])
    df5=Variables['carbonCosts_Pvar'].rename(columns={'YEAR_op':'YEAR','carbonCosts_Pvar':'carbon'}).set_index('YEAR')
    df1.sort_index(inplace=True)
    df2.sort_index(inplace=True)
    df3.sort_index(inplace=True)
    df4.sort_index(inplace=True)
    df5.sort_index(inplace=True)

    for y in YEAR:
        for tech in ['WindOnShore','WindOffShore_flot','Solar']: #'CCS1','CCS2',
            df1.drop((y,tech),inplace=True)

    # Energy use
    TECHNO = list(df1.index.get_level_values('TECHNOLOGIES').unique())
    TIMESTAMP=list(Variables['power_Dvar'].set_index('TIMESTAMP').index.get_level_values('TIMESTAMP').unique())

    df1['elecUse'] = 0
    df1['gasUse'] = 0
    df1['carbon'] = 0

    for tech in TECHNO:
        df1.loc[(slice(None),tech),'elecUse']=df1.loc[(slice(None),tech),'Prod']*(-convFac.loc[('electricity',tech),'conversionFactor'])
        df1.loc[(slice(None), tech), 'gasUse'] = df1.loc[(slice(None), tech), 'Prod'] * (-convFac.loc[('gaz', tech), 'conversionFactor'])

    Elecfac=pd.DataFrame(YEAR,columns=['YEAR']).set_index('YEAR')
    imp=Variables['importation_Dvar'].rename(columns={'YEAR_op':'YEAR'}).set_index(['YEAR','TIMESTAMP','RESOURCES']).loc[(slice(None),slice(None),'electricity')].groupby('YEAR').sum()

    for y in YEAR:
        if df1['elecUse'].groupby('YEAR').sum().loc[y]==0:
            Elecfac.loc[y, 'ElecFac'] =0
        else :
            Elecfac.loc[y,'ElecFac']=imp.loc[y,'importation_Dvar']/df1['elecUse'].groupby('YEAR').sum().loc[y]

    df_biogas=Variables['importation_Dvar'].groupby(['YEAR_op','RESOURCES']).sum().loc[(slice(None),'gazBio'),'importation_Dvar'].reset_index().rename(columns={'YEAR_op':'YEAR'}).set_index('YEAR').drop(columns='RESOURCES')
    df_natgas=Variables['importation_Dvar'].groupby(['YEAR_op','RESOURCES']).sum().loc[(slice(None),'gazNat'),'importation_Dvar'].reset_index().rename(columns={'YEAR_op':'YEAR'}).set_index('YEAR').drop(columns='RESOURCES')
    natgasFac=df_natgas['importation_Dvar']/(df_natgas['importation_Dvar']+df_biogas['importation_Dvar'])
    natgasFac=natgasFac.fillna(0)

    for tech in TECHNO:
        Grid_car[tech]=Variables['power_Dvar'].rename(columns={'YEAR_op':'YEAR'}).set_index(['YEAR','TIMESTAMP','TECHNOLOGIES']).loc[(slice(None),slice(None),tech)]*(-convFac.loc[('electricity',tech),'conversionFactor'])
        Grid_car[tech]=Grid_car[tech]*Grid_car['carbonContent']
        df1.loc[(slice(None), tech), 'carbon'] = (df1.loc[(slice(None), tech), 'Prod'] * ((-convFac.loc[('gaz', tech), 'conversionFactor']) * 203.5 * natgasFac + Tech.loc[(slice(None),tech),'EmissionCO2'].reset_index().drop(columns='TECHNOLOGIES').set_index('YEAR')['EmissionCO2']) + Grid_car[tech].groupby('YEAR').sum()*Elecfac['ElecFac'])*TaxC['carbonTax']

    df1['prodPercent']=0
    for y in YEAR:
        if df1['elecUse'].groupby('YEAR').sum().loc[y] == 0 : df1.loc[(y, slice(None)), 'elecPercent']=0
        else : df1.loc[(y,slice(None)),'elecPercent']=df1.loc[(y,slice(None)),'elecUse']/df1['elecUse'].groupby('YEAR').sum().loc[y]
        if df1['gasUse'].groupby('YEAR').sum().loc[y]==0 : df1.loc[(y, slice(None)), 'gasPercent']=0
        else : df1.loc[(y, slice(None)), 'gasPercent'] = df1.loc[(y, slice(None)), 'gasUse']/df1['gasUse'].groupby('YEAR').sum().loc[y]
        if (df1.loc[(y, 'electrolysis_AEL'), 'Prod']+df1.loc[(y, 'electrolysis_PEMEL'), 'Prod'])==0 : df1.loc[(y, 'electrolysis_AEL'), 'prodPercent'] =0
        else : df1.loc[(y, 'electrolysis_AEL'), 'prodPercent'] = df1.loc[(y, 'electrolysis_AEL'), 'Prod'] / (df1.loc[(y, 'electrolysis_AEL'), 'Prod']+df1.loc[(y, 'electrolysis_PEMEL'), 'Prod'])
        if (df1.loc[(y, 'electrolysis_AEL'), 'Prod'] + df1.loc[(y, 'electrolysis_PEMEL'), 'Prod'])==0: df1.loc[(y, 'electrolysis_PEMEL'), 'prodPercent'] =0
        else : df1.loc[(y, 'electrolysis_PEMEL'), 'prodPercent'] = df1.loc[(y, 'electrolysis_PEMEL'), 'Prod'] / (df1.loc[(y, 'electrolysis_AEL'), 'Prod'] + df1.loc[(y, 'electrolysis_PEMEL'), 'Prod'])

    #regroupement
    df1['type'] = 'None'
    df1.loc[(slice(None), 'SMR + CCS1'), 'type']='SMR'
    df1.loc[(slice(None), 'SMR + CCS2'), 'type']='SMR'
    df1.loc[(slice(None), 'SMR'), 'type']='SMR'
    df1.loc[(slice(None), 'SMR_elec'), 'type']='eSMR'
    df1.loc[(slice(None), 'SMR_elecCCS1'), 'type']='eSMR'
    df1.loc[(slice(None), 'electrolysis_AEL'), 'type']='AEL'
    df1.loc[(slice(None), 'electrolysis_PEMEL'), 'type']='AEL'

    # Repartition coût and Removing actualisation
    def actualisationFactor(r,y):
        return (1 + r) ** (-(y+dy/2 - y0))

    r=inputDict['economics'].loc['discountRate'].value

    for y in YEAR:
        df1.loc[(y,slice(None)),'importElec']=df1.loc[(y,slice(None)),'elecPercent']*df2.loc[(y,'electricity')]['importCosts']/actualisationFactor(r,y)
        df1.loc[(y, slice(None)), 'TURPE'] = df1.loc[(y, slice(None)), 'elecPercent'] * df2.loc[(y, 'electricity')]['TURPE']/actualisationFactor(r,y)
        df1.loc[(y,slice(None)),'capexElec']=df1.loc[(y,slice(None)),'elecPercent']*(df3.loc[(y,'WindOnShore')]['capacityCosts_Pvar']+df3.loc[(y,'WindOffShore_flot')]['capacityCosts_Pvar']+df3.loc[(y,'Solar')]['capacityCosts_Pvar'])/actualisationFactor(r,y)
        df1.loc[(y, slice(None)), 'importGas'] = df1.loc[(y,slice(None)),'gasPercent']*(df2.loc[(y, 'gazNat')]['importCosts']+df2.loc[(y, 'gazBio')]['importCosts'])/actualisationFactor(r,y)
        df1.loc[(y,slice(None)),'storageElec']=df1.loc[(y,slice(None)),'elecPercent']*df4.loc[(y,'Battery')]['storageCosts']/actualisationFactor(r,y)
        df1.loc[(y, slice(None)), 'storageH2'] = df1.loc[(y, slice(None)), 'prodPercent'] *( df4.loc[(y, 'tankH2_G')]['storageCosts']+ df4.loc[(y, 'saltCavernH2_G')]['storageCosts'])/actualisationFactor(r,y)
        df1.loc[(y,slice(None)),'carbon']=df1.loc[(y,slice(None)),'carbon']
        df1.loc[(y, slice(None)), 'powerCosts'] = df1.loc[(y, slice(None)), 'powerCosts'] / actualisationFactor(r, y)
        df1.loc[(y, slice(None)), 'capacityCosts'] = df1.loc[(y, slice(None)), 'capacityCosts'] / actualisationFactor(r, y)


    df1['Prod'].loc[df1['Prod']<0.0001]=0


    TECH=['AEL','SMR','eSMR']
    df={tech:df1.loc[df1['type']==tech].groupby('YEAR').sum() for tech in TECH}
    df_cocon=pd.DataFrame(index=YEAR)

    for tech in TECH:
        if df[tech]['Prod'].sum()==0:
            df.pop(tech)
        # else :
        #     for y in YEAR:
        #         if df[tech].loc[y]['Prod']==0:
        #             df_cocon.loc[y,tech] = df[tech]['capacityCosts'].loc[y]+df[tech]['storageElec'].loc[y]+df[tech]['storageH2'].loc[y]+df[tech]['capexElec'].loc[y]
        #             df[tech].loc[y]=0
        #             df[tech].loc[y]['Prod']=1

    return df

def extract_energy(scenario,outputFolder='../Data/output'):
    v_list = [ 'capacity_Pvar','energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar','importation_Dvar','carbon_Pvar',
             'powerCosts_Pvar', 'capacityCosts_Pvar','importCosts_Pvar', 'storageCosts_Pvar', 'turpeCosts_Pvar','carbonCosts_Pvar', 'exportation_Dvar']
    Variables = {v: pd.read_csv(outputFolder + '/' + v + '.csv').drop(columns='Unnamed: 0') for v in v_list}

    inputDict = loadScenario(scenario)

    YEAR = Variables['power_Dvar'].set_index('YEAR_op').index.unique().values
    YEAR.sort()
    dy=YEAR[1]-YEAR[0]
    y0=YEAR[0]-dy

    df = Variables['power_Dvar'].groupby(['YEAR_op', 'TECHNOLOGIES']).sum().drop(columns='TIMESTAMP').reset_index()
    df = df.pivot(columns='TECHNOLOGIES', values='power_Dvar', index='YEAR_op').rename(columns={
        "electrolysis_AEL": "Alkaline electrolysis",
        "electrolysis_PEMEL": "PEM electrolysis",
        'SMR': "SMR w/o CCUS",
        'SMR + CCS1': 'SMR + CCUS 50%',
        'SMR + CCS2': 'SMR + CCUS 90%',
        'SMR_elec': 'eSMR w/o CCUS',
        'SMR_elecCCS1': 'eSMR + CCUS 50%',
        'cracking': 'Methane cracking'
    }).fillna(0)

    df = df / 1000

    df_capa = Variables['capacity_Pvar']
    df_capa = df_capa.pivot(columns='TECHNOLOGIES', values='capacity_Pvar', index='YEAR_op').rename(columns={
        "electrolysis_AEL": "Alkaline electrolysis",
        "electrolysis_PEMEL": "PEM electrolysis",
        'SMR': "SMR w/o CCUS",
        'SMR + CCS1': 'SMR + CCUS 50%',
        'SMR + CCS2': 'SMR + CCUS 90%',
        'SMR_elec': 'eSMR w/o CCUS',
        'SMR_elecCCS1': 'eSMR + CCUS 50%',
        'cracking': 'Methane cracking'
    }).fillna(0)

    df_capa=df_capa*8760/1000

    df_carbon = Variables['carbon_Pvar'].groupby('YEAR_op').sum().drop(columns='TIMESTAMP')
    df_costs=Variables['powerCosts_Pvar'].groupby('YEAR_op').sum().rename(columns={'powerCosts_Pvar':'power'})
    df_costs['capacity']=Variables['capacityCosts_Pvar'].groupby('YEAR_op').sum()
    df_costs['TURPE']=Variables['turpeCosts_Pvar'].groupby('YEAR_op').sum()
    df_costs['import'] = Variables['importCosts_Pvar'].groupby('YEAR_op').sum()
    df_costs['storage'] = Variables['storageCosts_Pvar'].groupby('YEAR_op').sum()
    df_costs['carbon'] = Variables['carbonCosts_Pvar'].groupby('YEAR_op').sum()
    df_costs['total']=df_costs.sum(axis=1)


    df_loadFac=(df/df_capa).fillna(0)
    for l in df_loadFac.columns:df_loadFac[l]=df_loadFac[l].apply(lambda x:0 if x<0 else x)
    for l in df_loadFac.columns: df_loadFac[l] = df_loadFac[l].apply(lambda x: 0 if x > 1 else x)

    df_renewables = Variables['power_Dvar'].pivot(index=['YEAR_op', 'TIMESTAMP'], columns='TECHNOLOGIES', values='power_Dvar')[
        ['WindOnShore', 'WindOffShore_flot', 'Solar']].reset_index().groupby('YEAR_op').sum().drop(
        columns='TIMESTAMP').sum(axis=1)
    df_export = Variables['exportation_Dvar'].groupby(['YEAR_op', 'RESOURCES']).sum().loc[
        (slice(None), 'electricity'), 'exportation_Dvar'].reset_index().drop(columns='RESOURCES').set_index('YEAR_op')
    df_feedRE = (df_renewables) / 1.54 / 1000 - df_export['exportation_Dvar']

    df_biogas = Variables['importation_Dvar'].groupby(['YEAR_op', 'RESOURCES']).sum().loc[
        (slice(None), 'gazBio'), 'importation_Dvar'].reset_index().set_index('YEAR_op').drop(columns='RESOURCES')
    for y in YEAR:
        fugitives = 0.03 * (1 - (y - YEAR[0]) / (2050 - YEAR[0])) * df_biogas.loc[y]['importation_Dvar']
        temp = df_biogas.loc[y]['importation_Dvar'] - fugitives
        if temp / 1.28 / 1000 < df.loc[y]['SMR w/o CCUS']:
            df_biogas.loc[y]['importation_Dvar'] = temp / 1.28 / 1000
        else:
            temp2 = temp - df.loc[y]['SMR w/o CCUS'] * 1.28 * 1000
            if temp2 / 1.32 / 1000 < df.loc[y]['SMR + CCUS 50%']:
                df_biogas.loc[y]['importation_Dvar'] = df.loc[y]['SMR w/o CCUS'] + temp2 / 1.32 / 1000
            else:
                temp3 = temp - df.loc[y]['SMR w/o CCUS'] * 1.28 * 1000 - df.loc[y]['SMR + CCUS 50%'] * 1.32 * 1000
                if temp3 / 1.45 / 1000 < df.loc[y]['SMR + CCUS 90%']:
                    df_biogas.loc[y]['importation_Dvar'] = df.loc[y]['SMR w/o CCUS'] + df.loc[y][
                        'SMR + CCUS 50%'] + temp3 / 1.45 / 1000
                else:
                    df_biogas.loc[y]['importation_Dvar'] = df.loc[y]['SMR w/o CCUS'] + df.loc[y]['SMR + CCUS 50%'] + \
                                                           df.loc[y]['SMR + CCUS 90%']
    df['feedBiogas']=df_biogas['importation_Dvar']
    df['feedRE']=df_feedRE
    df['loadFac_elec'] = df_loadFac['Alkaline electrolysis']
    df['loadFac_SMR'] = df_loadFac['SMR w/o CCUS']
    df['loadFac_SMR+CCS50'] =df_loadFac['SMR + CCUS 50%']
    df['loadFac_SMR+CCS90']=df_loadFac['SMR + CCUS 90%']
    df['carbon']=df_carbon['carbon_Pvar']/1000/(df[['SMR w/o CCUS','SMR + CCUS 50%','SMR + CCUS 90%','Alkaline electrolysis',"PEM electrolysis"]].sum(axis=1)*30)
    df['carbon'].loc[df['carbon']<0]=0

    def actualisationFactor(r,y):
        return (1 + r) ** (-(y+dy/2 - y0))

    r=inputDict['economics'].loc['discountRate'].value
    for y in YEAR:
        df_costs.loc[y,'total_nonAct']=df_costs.loc[y,'total']/actualisationFactor(r,y)

    df['costs']=df_costs['total_nonAct']/(df[['SMR w/o CCUS','SMR + CCUS 50%','SMR + CCUS 90%','Alkaline electrolysis',"PEM electrolysis"]].sum(axis=1)*30)/1000

    return df

def extract_capa(scenario,outputFolder='../Data/output'):
    v_list = [ 'capacity_Pvar','energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar','importation_Dvar','carbon_Pvar',
             'powerCosts_Pvar', 'capacityCosts_Pvar','importCosts_Pvar', 'storageCosts_Pvar', 'turpeCosts_Pvar','carbonCosts_Pvar','exportation_Dvar']
    Variables = {v: pd.read_csv(outputFolder + '/' + v + '.csv').drop(columns='Unnamed: 0') for v in v_list}

    YEAR = Variables['power_Dvar'].set_index('YEAR_op').index.unique().values
    YEAR.sort()
    dy=YEAR[1]-YEAR[0]
    y0=YEAR[0]-dy

    df = Variables['capacity_Pvar']
    df = df.pivot(columns='TECHNOLOGIES', values='capacity_Pvar', index='YEAR_op').rename(columns={
        "electrolysis_AEL": "Alkaline electrolysis",
        "electrolysis_PEMEL": "PEM electrolysis",
        'SMR': "SMR w/o CCUS",
        'SMR + CCS1': 'SMR + CCUS 50%',
        'SMR + CCS2': 'SMR + CCUS 90%',
        'SMR_elec': 'eSMR w/o CCUS',
        'SMR_elecCCS1': 'eSMR + CCUS 50%',
        'cracking': 'Methane cracking'
    }).fillna(0)

    return df