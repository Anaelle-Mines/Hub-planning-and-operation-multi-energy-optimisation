#region Importation of modules
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
from matplotlib.gridspec import GridSpec
from sklearn import linear_model
import sys
import time
import datetime
import seaborn as sb

from Functions.f_multiResourceModels import *
from Functions.f_optimization import *
from Functions.f_optimModel_elec import *
from Functions.f_InputScenario import *
from Basic_functionalities.scenarios_ref_Fr import scenarioFr

#endregion

def plot_mixProdElec(outputFolder='Data/output/'):

    v_list = ['capacityInvest_Dvar','transInvest_Dvar','capacity_Pvar','capacityDel_Pvar','capacityDem_Dvar', 'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
              'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar','carbon_Pvar','powerCosts_Pvar','capacityCosts_Pvar','importCosts_Pvar','storageCosts_Pvar','turpeCosts_Pvar','Pmax_Pvar','max_PS_Dvar','carbonCosts_Pvar']
    Variables = {v : pd.read_csv(outputFolder+'/'+v+'.csv').drop(columns='Unnamed: 0') for v in v_list}

    YEAR=list(Variables['power_Dvar'].set_index('YEAR_op').index.unique())
    elecProd=Variables['power_Dvar'].set_index(['YEAR_op','TIMESTAMP','TECHNOLOGIES'])

    Prod=elecProd.groupby(['YEAR_op','TECHNOLOGIES']).sum()
    Prod.loc[(slice(None),'IntercoOut'),'power_Dvar']=-Prod.loc[(slice(None),'IntercoOut'),'power_Dvar']
    Capa=Variables['capacity_Pvar'].set_index(['YEAR_op','TECHNOLOGIES'])
    Capa.sort_index(axis = 0,inplace=True)

    TECHNO=list(elecProd.index.get_level_values('TECHNOLOGIES').unique())
    l_tech=len(TECHNO)
    l_year=len(YEAR)

    Interco={y:(Prod.loc[(y,'IntercoIn')]+Prod.loc[(y,'IntercoOut')]) for y in YEAR}
    Fossils={y:(Prod.loc[(y,'CCG')]+Prod.loc[(y,'Coal_p')]+Prod.loc[(y,'NewNuke')]+Prod.loc[(y,'OldNuke')]+Prod.loc[(y,'TAC')])['power_Dvar']/(Prod.loc[(y,slice(None))].sum()['power_Dvar']-Interco[y]['power_Dvar']) for y in YEAR}
    EnR={y:(Prod.loc[(y,'Solar')]+Prod.loc[(y,'WindOnShore')]+Prod.loc[(y,'WindOffShore')]+Prod.loc[(y,'HydroRiver')]+Prod.loc[(y,'HydroReservoir')])['power_Dvar']/(Prod.loc[(y,slice(None))].sum()['power_Dvar']-Interco[y]['power_Dvar']) for y in YEAR}
    test={y:Fossils[y]+EnR[y] for y in YEAR}
    print('EnR+Fossils = ',test)

    sb.set_palette('muted')

    fig, ax = plt.subplots()
    width= 0.60
    x = np.arange(l_year)
    cpt=1
    for tech in TECHNO :
        l=list(Prod.loc[(slice(None),tech),'power_Dvar']/1000000)
        ax.bar(x + cpt*width/l_tech, l, width/l_tech, label=tech)
        cpt=cpt+1

    plt.xticks(x,['2020','2030','2040','2050'])#,'2060'])
    plt.title('Electricity production')
    plt.ylabel('TWh/an')
    plt.legend()

    plt.savefig(outputFolder+'/Electricity production.png')

    plt.show()

    fig, ax = plt.subplots()
    width = 0.60
    x = np.arange(l_year)
    cpt = 1
    for tech in TECHNO:
        l = list(Capa.loc[(slice(None), tech), 'capacity_Pvar'] / 1000)
        ax.bar(x + cpt * width / l_tech, l, width / l_tech, label=tech)
        cpt = cpt + 1

    plt.xticks(x, ['2020', '2030', '2040', '2050'])#,'2060'])
    plt.title('Installed capacity')
    plt.ylabel('GW')
    plt.legend()

    plt.savefig(outputFolder+'/Installed capacity.png')

    plt.show()

    return EnR, Fossils

def plot_hourlyProduction(year,timeRange=range(1,8761),outputFolder='Data/output/',figName='test'):

    v_list = ['capacityInvest_Dvar','transInvest_Dvar','capacity_Pvar','capacityDel_Pvar','capacityDem_Dvar', 'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
              'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar','carbon_Pvar','powerCosts_Pvar','capacityCosts_Pvar','importCosts_Pvar','storageCosts_Pvar','turpeCosts_Pvar','Pmax_Pvar','max_PS_Dvar','carbonCosts_Pvar']
    Variables = {v : pd.read_csv(outputFolder+'/'+v+'.csv').drop(columns='Unnamed: 0') for v in v_list}

    elecProd=Variables['power_Dvar'].set_index(['YEAR_op','TIMESTAMP','TECHNOLOGIES'])

    TECHNO=list(elecProd.index.get_level_values('TECHNOLOGIES').unique())
    TECHNO.remove('IntercoOut')
    TECHNO.remove('curtailment')
    TECHNO.remove('IntercoIn')

    fig, ax = plt.subplots()
    col=sb.color_palette('muted')
    cpt=1
    l={0:0*elecProd.loc[(2030,timeRange,'Solar'),'power_Dvar'].values}

    for tech in TECHNO:
        l[cpt]=l[cpt-1]+elecProd.loc[(year,timeRange,tech),'power_Dvar'].values
        ax.fill_between(timeRange, l[cpt-1], l[cpt],label=tech)
        cpt+=1

    ax.legend()
    plt.savefig(outputFolder+'/'+figName+'.png')
    plt.show()

    return

def plot_monotone(outputFolder='Data/output/'):

    # marketPrice_ref=pd.read_csv('Data/output/Ref_wH2_Fr/marketPrice.csv').set_index(['YEAR_op', 'TIMESTAMP'])
    # marketPrice_ref['OldPrice_NonAct'].loc[marketPrice_ref['OldPrice_NonAct'] > 400] = 400

    marketPrice = pd.read_csv(outputFolder+'/marketPrice.csv').set_index(['YEAR_op', 'TIMESTAMP'])
    marketPrice['OldPrice_NonAct'].loc[marketPrice['OldPrice_NonAct'] > 400] = 400
    prices2013=pd.read_csv('Data/Raw_Ana/electricity-grid-price-2013.csv').set_index('TIMESTAMP').fillna(0)

    YEAR=marketPrice.index.get_level_values('YEAR_op').unique().values
    YEAR.sort()

    col = plt.cm.tab20c
    plt.figure(figsize=(6, 4))

    for k, yr in enumerate(YEAR) :
        MonotoneNew = marketPrice.OldPrice_NonAct.loc[(yr, slice(None))].value_counts(bins=100)
        MonotoneNew.sort_index(inplace=True, ascending=False)
        NbVal = MonotoneNew.sum()
        MonotoneNew_Cumul = []
        MonotoneNew_Price = []
        val = 0
        for i in MonotoneNew.index:
            val = val + MonotoneNew.loc[i]
            MonotoneNew_Cumul.append(val / NbVal * 100)
            MonotoneNew_Price.append(i.right)

        # MonotoneOld = marketPrice_ref.OldPrice_NonAct.loc[(yr, slice(None))].value_counts(bins=100)
        # MonotoneOld.sort_index(inplace=True, ascending=False)
        # NbVal = MonotoneOld.sum()
        # MonotoneOld_Cumul = []
        # MonotoneOld_Price = []
        # val = 0
        # for i in MonotoneOld.index:
        #     val = val + MonotoneOld.loc[i]
        #     MonotoneOld_Cumul.append(val / NbVal * 100)
        #     MonotoneOld_Price.append(i.right)

        plt.plot(MonotoneNew_Cumul, MonotoneNew_Price, color=col(k*4), label='Prices '+ str(yr))
        # plt.plot(MonotoneOld_Cumul, MonotoneOld_Price, 'C{}--'.format(k),label='N1 (75%) Prices '+ str(yr))

    MonotoneReal = prices2013.Prices.value_counts(bins=100)
    MonotoneReal.sort_index(inplace=True, ascending=False)
    NbVal = MonotoneReal.sum()
    MonotoneReal_Cumul = []
    MonotoneReal_Price = []
    val = 0
    for i in MonotoneReal.index:
        val = val + MonotoneReal.loc[i]
        MonotoneReal_Cumul.append(val / NbVal * 100)
        MonotoneReal_Price.append(i.right)
    plt.plot(MonotoneReal_Cumul, MonotoneReal_Price,'--',color='black', label='Reals prices 2013 ')

    plt.legend()
    plt.xlabel('% of time')
    plt.ylabel('Electricity price (€/MWh)')
    # plt.title('Electricity prices monotone')
    plt.savefig(outputFolder+'/Monotone de prix elec _ wo high prices.png')
    plt.show()

    return

def plot_capacity(outputFolder='Data/output/'):
    v_list = ['capacityInvest_Dvar', 'transInvest_Dvar', 'capacity_Pvar', 'capacityDel_Pvar', 'capacityDem_Dvar',
              'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
              'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar', 'carbon_Pvar', 'powerCosts_Pvar', 'capacityCosts_Pvar',
              'importCosts_Pvar', 'storageCosts_Pvar', 'turpeCosts_Pvar', 'Pmax_Pvar', 'max_PS_Dvar', 'carbonCosts_Pvar']
    Variables = {v: pd.read_csv(outputFolder + '/' + v + '.csv').drop(columns='Unnamed: 0') for v in v_list}

    YEAR=Variables['power_Dvar'].set_index('YEAR_op').index.unique().values
    TECHNO = Variables['power_Dvar'].set_index('TECHNOLOGIES').index.unique().values
    TIMESTAMP=Variables['power_Dvar'].set_index('TIMESTAMP').index.unique().values
    YEAR.sort()

    #region Tracé mix prod H2 et EnR
    df=Variables['capacity_Pvar']
    df=df.pivot(columns='TECHNOLOGIES',values='capacity_Pvar', index='YEAR_op').rename(columns={
        "electrolysis_AEL": "Alkaline electrolysis",
        "electrolysis_PEMEL": "PEM electrolysis",
        'SMR': "SMR w/o CCUS",
        'SMR + CCS1':  'SMR + CCUS 50%',
        'SMR + CCS2':  'SMR + CCUS 90%',
        'SMR_elec': 'eSMR w/o CCUS',
        'SMR_elecCCS1': 'eSMR + CCUS 50%',
        'cracking': 'Methane cracking'
    }).fillna(0)

    capa=Variables['capacity_Pvar'].set_index(['YEAR_op','TECHNOLOGIES'])

    #LoadFactors
    EnR_loadFactor={y : (Variables['power_Dvar'].groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']/(Variables['capacity_Pvar'].set_index(['YEAR_op','TECHNOLOGIES'])['capacity_Pvar']*8760)).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values=0).loc[y,['WindOnShore','Solar','WindOffShore_flot']].fillna(0)  for y in YEAR}
    H2_loadFactor={y : (Variables['power_Dvar'].groupby(['YEAR_op','TECHNOLOGIES']).sum().drop(columns='TIMESTAMP')['power_Dvar']/(Variables['capacity_Pvar'].set_index(['YEAR_op','TECHNOLOGIES'])['capacity_Pvar']*8760)).reset_index().pivot(index='YEAR_op',columns='TECHNOLOGIES',values=0).loc[y,['electrolysis_PEMEL','electrolysis_AEL','SMR','SMR + CCS1','SMR + CCS2','SMR_elec','SMR_elecCCS1']].fillna(0) for y in YEAR}
    for y in YEAR : H2_loadFactor[y].loc[H2_loadFactor[y]<-0.0001]=0
    for y in YEAR : H2_loadFactor[y].loc[H2_loadFactor[y]>1.0001]=0
    for y in YEAR : EnR_loadFactor[y].loc[EnR_loadFactor[y]<-0.0001]=0
    for y in YEAR : EnR_loadFactor[y].loc[EnR_loadFactor[y]>1.0001]=0

    fig, ax = plt.subplots(2,1,sharex=True,figsize=(6.2,4))
    width= 0.40
    labels=list(df.index)
    x = np.arange(len(labels))
    col = plt.cm.tab20c

    # Create dark grey Bar
    l1=list(df['SMR w/o CCUS'])
    ax[0].bar(x - width/2, l1,width, color=col(17), label="SMR w/o CCUS",zorder=2)
    # Create dark bleu Bar
    l2=list(df['SMR + CCUS 50%'])
    ax[0].bar(x - width/2,l2,width, bottom=l1,color=col(0), label="SMR + CCUS 50%",zorder=2)
    #Create turquoise bleu Bar
    l3=list(df['SMR + CCUS 90%'])
    ax[0].bar(x - width/2,l3,width, bottom=[i+j for i,j in zip(l1,l2)], color=col(1) ,label="SMR + CCUS 90%",zorder=2)
    #Create orange Bar
    # l4=list(df['eSMR w/o CCUS'])
    # ax[0].bar(x - width/2,l4,width, bottom=[i+j+k for i,j,k in zip(l1,l2,l3)], color=col[1],label="eSMR w/o CCUS")
    # # Create yellow Bars
    # l5=list(df['eSMR + CCUS 50%'])
    # ax[0].bar(x - width/2,l5,width, bottom=[i+j+k+l for i,j,k,l in zip(l1,l2,l3,l4)], color='#F8B740',label="eSMR + CCUS 50%")
    # Create pink bar
    #l6=list(df['Methane cracking'])
    #ax[0].bar(x - width/2,l6,width, bottom=[i+j+k+l+m for i,j,k,l,m in zip(l1,l2,l3,l4,l5)], color=col[6],label="Methane cracking")
    # Create green Bars
    l7=list(df['Alkaline electrolysis']+df['PEM electrolysis'])
    ax[0].bar(x + width/2,l7,width, color=col(9),label="Water electrolysis",zorder=2)

    # Create red bar
    l8=list(df['Solar'])
    ax[1].bar(x ,l8,width, color=col(5),label="Solar",zorder=2)
    # Create violet bar
    l9=list(df['WindOnShore'])
    ax[1].bar(x,l9,width,  bottom=l8,color=col(13),label="Onshore wind",zorder=2)
    # Create pink bar
    l10=list(df['WindOffShore_flot'])
    ax[1].bar(x,l10,width,  bottom=[i+j for i,j in zip(l8,l9)],color=col(14),label="Offshore wind",zorder=2)
    #
    # # Create grey line
    # ax[2].plot(x,list((round(H2_loadFactor[y]['SMR']*100)for y in YEAR)),color=col(17),label='SMR w/o CCUS',zorder=2)
    # # Create dark blue line
    # ax[2].plot(x, list((round(H2_loadFactor[y]['SMR + CCS1'] * 100) for y in YEAR)), color=col(0), label='SMR + CCUS 50%',zorder=2)
    # # Create light blue line
    # ax[2].plot(x, list((round(H2_loadFactor[y]['electrolysis_AEL'] * 100)) for y in YEAR), color=col(9), label='Water electrolysis',zorder=2)
    # Create green line
    # ax[2].plot(x, list((round(H2_loadFactor[y]['SMR + CCS2'] * 100)) for y in YEAR), color=col(1), label='SMR + CCUS 90%',zorder=2)
    # Create WindOnshore line
    # ax[2].plot(x, list((round(EnR_loadFactor[y]['WindOnShore'] * 100)) for y in YEAR),linestyle='--' ,color=col(13), label='Wind Onshore',zorder=2)
    # # Create Solar line
    # ax[2].plot(x, list((round(EnR_loadFactor[y]['Solar'] * 100) for y in YEAR)),linestyle='--',color=col(5), label='Solar',zorder=2)

    #add Load factors
    # for i,y in enumerate(YEAR):
    #     if capa.loc[(y,'electrolysis_AEL'),'capacity_Pvar'] > 100:
    #         ax[0].text((x + width/2)[i], l7[i]/2, str(round(H2_loadFactor[y]['electrolysis_AEL']*100)) +'%',ha='center')
    #     if capa.loc[(y,'SMR'),'capacity_Pvar'] > 100:
    #         ax[0].text((x - width / 2)[i], l1[i] / 2, str(round(H2_loadFactor[y]['SMR'] * 100)) + '%',ha='center',color='white')
    #     if capa.loc[(y,'SMR + CCS1'),'capacity_Pvar'] > 100:
    #         ax[0].text((x - width / 2)[i], l1[i]+l2[i] / 2, str(round(H2_loadFactor[y]['SMR + CCS1'] * 100)) + '%',ha='center',color='white')
    #     if capa.loc[(y, 'Solar'), 'capacity_Pvar'] > 10:
    #         ax[1].text((x)[i], l8[i] / 2, str(round(EnR_loadFactor[y]['Solar'] * 100)) + '%', ha='center')
    #     if capa.loc[(y,'Solar'),'capacity_Pvar'] > 100:
    #         ax[1].text((x)[i], l8[i]/2, str(round(EnR_loadFactor[y]['Solar'] * 100)) + '%', ha='center',color='white')
    #     if capa.loc[(y,'WindOnShore'),'capacity_Pvar'] > 100:
    #         ax[1].text((x)[i], l8[i]+l9[i]/2, str(round(EnR_loadFactor[y]['WindOnShore'] * 100)) + '%', ha='center',color='white')
    #     if capa.loc[(y,'WindOffShore_flot'),'capacity_Pvar'] > 100:
    #         ax[1].text((x)[i], l8[i]+l9[i]+l10[i]/2, str(round(EnR_loadFactor[y]['WindOffShore_flot'] * 100)) + '%', ha='center',color='white')

    ax[0].set_ylim([0,max(max([(n1,n2) for n1,n2 in zip([i+j+k for i,j,k in zip(l2,l2,l3)],l7)]))+100])
    ax[0].grid(axis='y',alpha=0.5,zorder=1)
    ax[1].set_ylim([0,max([i+j+k for i,j,k in zip(l8,l9,l10)])+100])
    ax[1].grid(axis='y',alpha=0.5,zorder=1)
    # ax[2].grid(axis='y', alpha=0.5,zorder=1)
    ax[0].set_ylabel('Installed capacity (MW)')
    ax[1].set_ylabel('Installed capacity (MW)')
    # ax[2].set_ylabel('Load factors (%)')
    ax[0].set_title("Evolution of H2 production assets")
    ax[1].set_title("Evolution of local RE assets")
    # ax[2].set_title("Evolution of load factors")
    plt.xticks(x, ['2010-2020','2020-2030','2030-2040', '2040-2050'])#['2010-2020','2020-2030','2030-2040', '2040-2050']'2050-2060'])
    # Shrink current axis by 20%
    box = ax[0].get_position()
    ax[0].set_position([box.x0, box.y0, box.width * 0.73, box.height*0.95])
    # Put a legend to the right of the current axis
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # Shrink current axis by 20%
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.73, box.height*0.95])
    # Put a legend to the right of the current axis
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # Shrink current axis by 20%
    # box = ax[2].get_position()
    # ax[2].set_position([box.x0, box.y0, box.width * 0.73, box.height*0.95])
    # Put a legend to the right of the current axis
    # ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(outputFolder+'/Evolution mix prod.png')
    plt.show()

    def monthly_average(df):
        df['month'] = df.index // 730 + 1
        df.loc[8760,'month']=12
        return df.groupby('month').mean()

    loadFactors_df=Variables['power_Dvar'].copy().pivot(index=['YEAR_op','TIMESTAMP'],columns='TECHNOLOGIES',values='power_Dvar')
    for y in YEAR :
        for tech in TECHNO:
            loadFactors_df.loc[y,slice(None)][tech]=(Variables['power_Dvar'].set_index(['YEAR_op','TIMESTAMP','TECHNOLOGIES']).loc[(y,slice(None),tech),'power_Dvar']/Variables['capacity_Pvar'].set_index(['YEAR_op','TECHNOLOGIES']).loc[(y,tech),'capacity_Pvar']).reset_index().drop(columns=['TECHNOLOGIES','YEAR_op']).set_index('TIMESTAMP')['power_Dvar']

    month=np.unique(TIMESTAMP//730+1)[:-1]

    fig, ax = plt.subplots()

    for k,y in enumerate(YEAR):
        #Create electrolysis graph
        l1=list(monthly_average(loadFactors_df.loc[(y,slice(None))])['electrolysis_AEL']*100)
        plt.plot(month,l1,color=col(8+k),label=y,zorder=2)

    plt.grid(axis='y',alpha=0.5,zorder=1)
    plt.ylabel('Load factor (%)')
    plt.xlabel('Months')
    plt.xticks(month,['January','February','March','April','May','June','July','August','September','October','November','December'],rotation=45)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0+0.1, box.width * 0.90, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(outputFolder+'/elec_LoadFactor.png')
    plt.show()

    return df

def plot_energy(outputFolder='Data/output/'):
    v_list = ['capacityInvest_Dvar', 'transInvest_Dvar', 'capacity_Pvar', 'capacityDel_Pvar', 'capacityDem_Dvar',
              'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
              'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar', 'carbon_Pvar', 'powerCosts_Pvar', 'capacityCosts_Pvar',
              'importCosts_Pvar', 'storageCosts_Pvar', 'turpeCosts_Pvar', 'Pmax_Pvar', 'max_PS_Dvar', 'carbonCosts_Pvar','exportation_Dvar']
    Variables = {v: pd.read_csv(outputFolder + '/' + v + '.csv').drop(columns='Unnamed: 0') for v in v_list}

    YEAR=Variables['power_Dvar'].set_index('YEAR_op').index.unique().values
    YEAR.sort()

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

    df = df / 1000000

    df_renewables=Variables['power_Dvar'].pivot(index=['YEAR_op','TIMESTAMP'],columns='TECHNOLOGIES',values='power_Dvar')[['WindOnShore','WindOffShore_flot','Solar']].reset_index().groupby('YEAR_op').sum().drop(columns='TIMESTAMP').sum(axis=1)
    df_export=Variables['exportation_Dvar'].groupby(['YEAR_op','RESOURCES']).sum().loc[(slice(None),'electricity'),'exportation_Dvar'].reset_index().drop(columns='RESOURCES').set_index('YEAR_op')
    df_feedRE=(df_renewables-df_export['exportation_Dvar'])/1.54/1000000#

    df_biogas=Variables['importation_Dvar'].groupby(['YEAR_op','RESOURCES']).sum().loc[(slice(None),'gazBio'),'importation_Dvar'].reset_index().set_index('YEAR_op').drop(columns='RESOURCES')
    for y in YEAR:
        fugitives = 0.03 * (1 - (y - YEAR[0]) / (2050 - YEAR[0]))*df_biogas.loc[y]['importation_Dvar']
        temp=df_biogas.loc[y]['importation_Dvar']-fugitives
        if temp/1.28/1000000<df.loc[y]['SMR w/o CCUS']:
            df_biogas.loc[y]['importation_Dvar']=temp/1.28/1000000
        else:
            temp2=temp-df.loc[y]['SMR w/o CCUS']*1.28*1000000
            if temp2/1.32/1000000<df.loc[y]['SMR + CCUS 50%']:
                df_biogas.loc[y]['importation_Dvar']=df.loc[y]['SMR w/o CCUS']+temp2/1.32/1000000
            else:
                temp3=temp-df.loc[y]['SMR w/o CCUS']*1.28*1000000-df.loc[y]['SMR + CCUS 50%']*1.32*1000000
                if temp3/1.45/1000000<df.loc[y]['SMR + CCUS 90%']:
                    df_biogas.loc[y]['importation_Dvar']=df.loc[y]['SMR w/o CCUS']+df.loc[y]['SMR + CCUS 50%']+temp3/1.45/1000000
                else :
                    df_biogas.loc[y]['importation_Dvar'] = df.loc[y]['SMR w/o CCUS']+df.loc[y]['SMR + CCUS 50%']+df.loc[y]['SMR + CCUS 90%']

    fig, ax = plt.subplots(figsize=(6,4))
    width = 0.35
    col=plt.cm.tab20c
    labels = list(df.index)
    x = np.arange(len(labels))

    # Create dark grey Bar
    l1 = list(df['SMR w/o CCUS'])
    ax.bar(x - width / 2, l1, width, color=col(17), label="SMR w/o CCUS",zorder=2)
    # Create dark bleu Bar
    l2 = list(df['SMR + CCUS 50%'])
    ax.bar(x - width / 2, l2, width, bottom=l1, color=col(0), label="SMR + CCUS 50%",zorder=2)
    # Create turquoise bleu Bar
    l3 = list(df['SMR + CCUS 90%'])
    ax.bar(x - width / 2, l3, width, bottom=[i + j for i, j in zip(l1, l2)], color=col(1), label="SMR + CCUS 90%",zorder=2)
    # Create biogas Bars
    l8=list(df_biogas['importation_Dvar'])
    plt.rcParams['hatch.linewidth']=8
    plt.rcParams['hatch.color'] = col(3)
    ax.bar(x - width / 2,l8,width,color='none',hatch='/',edgecolor=col(3),linewidth=0.5,label="Biomethane feed",alpha=0.8,zorder=3)
    # # Create orange Bar
    # l4 = list(df['eSMR w/o CCUS'])
    # ax.bar(x - width / 2, l4, width, bottom=[i + j + k for i, j, k in zip(l1, l2, l3)], color=col[1],
    #        label="eSMR w/o CCUS")
    # # Create yellow Bars
    # l5 = list(df['eSMR + CCUS 50%'])
    # ax.bar(x - width / 2, l5, width, bottom=[i + j + k + l for i, j, k, l in zip(l1, l2, l3, l4)], color=col[8],
    #        label="eSMR + CCUS 50%")
    # Create pink bar
    # l6 = list(df['Methane cracking'])
    # ax.bar(x - width / 2, l6, width, bottom=[i + j + k + l + m for i, j, k, l, m in zip(l1, l2, l3, l4, l5)],
    #        color=col[6], label="Methane cracking")
    # Create light green Bars
    l7 = list(df['Alkaline electrolysis']+ df['PEM electrolysis'])
    ax.bar(x + width / 2, l7, width, color=col(8), label="AEL grid feed",zorder=2)
    # Create dark green bar
    l9=list(df_feedRE)
    ax.bar(x + width / 2,l9,width,color=col(9),label="AEL local feed",zorder=3)

    plt.grid(axis='y',alpha=0.5,zorder=1)
    ax.set_ylabel('H2 production (TWh/yr)')
    # ax.set_title("Use of assets")
    plt.xticks(x,['2020','2030']) #['2020-2030', '2030-2040', '2040-2050', '2050-2060'])#,'2060'])
    m=max(max(l7),max([l1[i]+l2[i]+l3[i] for i in np.arange(len(l1))]))
    ax.set_ylim([0,int(m)+0.5])
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.72, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(outputFolder+'/H2 production.png')
    plt.show()

    return df

def plot_evolve(outputFolder='Data/output/'):

    v_list = ['capacityInvest_Dvar', 'transInvest_Dvar', 'capacity_Pvar', 'capacityDel_Pvar', 'capacityDem_Dvar',
              'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
              'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar', 'carbon_Pvar', 'powerCosts_Pvar', 'capacityCosts_Pvar',
              'importCosts_Pvar', 'storageCosts_Pvar', 'turpeCosts_Pvar', 'Pmax_Pvar', 'max_PS_Dvar', 'carbonCosts_Pvar']
    Variables = {v: pd.read_csv(outputFolder + '/' + v + '.csv').drop(columns='Unnamed: 0') for v in v_list}

    YEAR=Variables['power_Dvar'].set_index('YEAR_op').index.unique().values
    YEAR.sort()
    dy=YEAR[1]-YEAR[0]
    ind_rename={y-dy:y for y in YEAR}

    df0=Variables['transInvest_Dvar'].set_index('YEAR_invest').rename(index=ind_rename).set_index(['TECHNOLOGIES','TECHNOLOGIES.1'],append=True)
    df=pd.DataFrame(index=YEAR,columns=['SMR w/o CCUS','eSMR w/o CCUS','+ CCUS 50%','+ CCUS 75%','+ CCUS 25%','+ eCCUS 50%'])
    df['+ CCUS 50%']=df0.loc[(slice(None),'SMR','SMR + CCS1')]+df0.loc[(slice(None),'SMR','SMR + CCS1')]
    df['+ CCUS 75%']=df0.loc[(slice(None),'SMR','SMR + CCS2')]+df0.loc[(slice(None),'SMR','SMR + CCS2')]
    df['+ CCUS 25%']=df0.loc[(slice(None),'SMR + CCS1','SMR + CCS2')]
    df['+ CCUS 50%']-=df['+ CCUS 25%']
    df['+ eCCUS 50%']=df0.loc[(slice(None),'SMR_elec','SMR_elecCCS1')]
    df=df.sort_index()

    for y in YEAR[:-1] :
        df.loc[y+dy]+=df.loc[y]

    df['SMR w/o CCUS']=list(Variables['capacity_Pvar'].set_index(['YEAR_op','TECHNOLOGIES']).loc[(slice(None),'SMR'),'capacity_Pvar'].sort_index())
    df['eSMR w/o CCUS']=list(Variables['capacity_Pvar'].set_index(['YEAR_op','TECHNOLOGIES']).loc[(slice(None),'SMR_elec'),'capacity_Pvar'].sort_index())


    fig, ax = plt.subplots()
    width= 0.35
    col=sb.color_palette('muted')
    labels=list(df.index)
    x = np.arange(len(labels))

    #Create Ref Bars
    l1=list(df['SMR w/o CCUS'])
    ax.bar(x - width/2, l1,width, color=col[7], label="SMR w/o CCUS")
    l2=list(df['eSMR w/o CCUS'])
    ax.bar(x + width/2, l2,width, color=col[8], label="eSMR w/o CCUS")

    #Create Transfo Bars
    l3=list(df['+ CCUS 50%'])
    ax.bar(x - width/2, l3,width,bottom=l1, color=col[0], label="SMR + CCUS 50%")
    l4=list(df['+ CCUS 75%'])
    ax.bar(x - width/2, l4,width, bottom=[i+j for i,j in zip(l1,l3)], color=col[9], label="SMR + CCUS 75%")
    l5=list(df['+ CCUS 25%'])
    ax.bar(x - width/2, l5,width, bottom=[i+j+k for i,j,k in zip(l1,l3,l4)], color=col[9])
    l6=list(df['+ eCCUS 50%'])
    ax.bar(x + width/2, l6,width, bottom=l2, color=col[1], label="eSMR + CCUS 50%")

    ax.set_ylabel('Capacité (MW)')
    ax.set_title("Evolution des technologies SMR")
    plt.xticks(x, ['2010-2020','2020-2030','2030-2040', '2040-2050'])
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.74, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(outputFolder+'/Evolution SMR.png')
    plt.show()

    return

def plot_elecMean(scenario,outputFolder='Data/output/'):

    v_list = ['capacityInvest_Dvar', 'transInvest_Dvar', 'capacity_Pvar', 'capacityDel_Pvar', 'capacityDem_Dvar',
              'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
              'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar', 'carbon_Pvar', 'powerCosts_Pvar', 'capacityCosts_Pvar',
              'importCosts_Pvar', 'storageCosts_Pvar', 'turpeCosts_Pvar', 'Pmax_Pvar', 'max_PS_Dvar', 'carbonCosts_Pvar']
    Variables = {v: pd.read_csv(outputFolder + '/' + v + '.csv').drop(columns='Unnamed: 0') for v in v_list}

    inputDict = loadScenario(scenario)

    YEAR=Variables['power_Dvar'].set_index('YEAR_op').index.unique().values
    YEAR.sort()

    def weekly_average(df):
        df['week'] = df.index // 168 + 1
        df.loc[8736:8760,'week']=52
        return df.groupby('week').sum() / 1000

    convFac = inputDict[ 'conversionFactor']

    v = 'power_Dvar'
    Pel = {y: Variables[v].loc[Variables[v]['YEAR_op'] == y].pivot(columns='TECHNOLOGIES', values='power_Dvar', index='TIMESTAMP').drop(columns=['CCS1','CCS2']) for y in YEAR}
    for y in YEAR :
        for tech in list(Pel[y].columns):
            Pel[y][tech]=Pel[y][tech]*convFac.loc[('electricity',tech)].conversionFactor
    v = 'storageOut_Pvar'
    Pel_stock_out = {y: Variables[v][np.logical_and(Variables[v]['YEAR_op'] == y, Variables[v]['RESOURCES'] == 'electricity')].pivot(columns='STOCK_TECHNO',values=v,index='TIMESTAMP') for y in YEAR}
    v = 'storageIn_Pvar'
    Pel_stock_in = {y: Variables[v][np.logical_and(Variables[v]['YEAR_op'] == y, Variables[v]['RESOURCES'] == 'electricity')].pivot(columns='STOCK_TECHNO',values=v,index='TIMESTAMP') for y in YEAR}
    v = 'importation_Dvar'
    Pel_imp = {y: Variables[v][np.logical_and(Variables[v]['YEAR_op'] == y, Variables[v]['RESOURCES'] == 'electricity')].pivot(columns='RESOURCES',values=v,index='TIMESTAMP') for y in YEAR}

    Pel_exp = {y: -np.minimum(Pel_imp[y], 0) for y in Pel_imp.keys()}
    Pel_imp = {y: np.maximum(Pel_imp[y], 0) for y in Pel_imp.keys()}


    fig, ax = plt.subplots(4, 1, figsize=(6, 10), sharex=True)
    col=plt.cm.tab20c
    colBis=plt.cm.tab20b

    for k, yr in enumerate(YEAR):
        ax[k].yaxis.grid(linestyle='--', linewidth=0.5,zorder=-6)

        # power_Dvar
        Pel[yr] = weekly_average(Pel[yr])
        # storageOut_Pvar
        Pel_stock_out[yr] = weekly_average(Pel_stock_out[yr])
        # storageIn_Pvar
        Pel_stock_in[yr] = weekly_average(Pel_stock_in[yr])
        # importation_Dvar
        Pel_imp[yr] = weekly_average(Pel_imp[yr])
        Pel_exp[yr] = weekly_average(Pel_exp[yr])

        # Elec production
        ax[k].bar(Pel[yr].index, Pel[yr]['Solar'], label='Solar - PV',color=col(5), zorder=-1)
        ax[k].bar(Pel[yr].index, Pel[yr]['Solar'] + Pel[yr]['WindOnShore'], label='Wind - Onshore', color=col(13),zorder=-2)
        ax[k].bar(Pel[yr].index, Pel[yr]['Solar'] + Pel[yr]['WindOffShore_flot'] + Pel[yr]['WindOnShore'], label='Wind - Offshore',color=col(14), zorder=-3)
        ax[k].bar(Pel_stock_out[yr].index, Pel_stock_out[yr]['Battery'] + Pel[yr]['WindOnShore'] + Pel[yr]['Solar'],label='Battery - Out',color=colBis(9), zorder=-4)
        ax[k].bar(Pel_stock_out[yr].index,Pel_stock_out[yr]['Battery'] + Pel[yr]['WindOnShore'] + Pel[yr]['Solar'] + Pel_imp[yr]['electricity'],label='Imports',color=colBis(14),  zorder=-5)

        # Elec consumption
        ax[k].bar(Pel[yr].index, Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL'], label='Electrolysis',color=col(9), zorder=-1)
        # ax[k].bar(Pel[yr].index, Pel[yr]['SMR_elec'] + Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL'], label='eSMR',color=col(0), zorder=-2)
        # ax[k].bar(Pel[yr].index, Pel[yr]['SMR + CCS1'] + Pel[yr]['SMR + CCS2'] + Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL'] + Pel[yr]['SMR_elec'] , label='CCUS',color=col(0), zorder=-3)
        ax[k].bar(Pel_stock_in[yr].index, -Pel_stock_in[yr]['Battery'] + Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL'] + Pel[yr]['SMR_elec'] + Pel[yr]['SMR + CCS1'] + Pel[yr]['SMR + CCS2']  , label='Battery - In', color=colBis(8) ,zorder=-4)
        ax[k].bar(Pel_stock_in[yr].index,-Pel_stock_in[yr]['Battery'] + Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL'] + Pel[yr]['SMR_elec'] - Pel_exp[yr]['electricity'] + Pel[yr]['SMR + CCS1'] + Pel[yr]['SMR + CCS2'], label='Exports',color=colBis(15),zorder=-5)

        ax[k].set_ylabel('Weakly production (GWh)')
        m=(Pel_stock_out[yr]['Battery'] + Pel[yr]['WindOnShore'] + Pel[yr]['WindOffShore_flot']  + Pel[yr]['Solar'] + Pel_imp[yr]['electricity']).max()+10
        ax[k].set_ylim([-m, m])
        ax[k].set_title(YEAR[k])
        # Shrink all axis by 20%
        box = ax[k].get_position()
        ax[k].set_position([box.x0, box.y0, box.width * 0.74, box.height])

    ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax[-1].set_xlabel('Week')
    plt.savefig(outputFolder+'/Gestion elec.png')
    plt.show()

    return

def plot_H2Mean(scenario,outputFolder='Data/output/'):

    def weekly_average(df):
        df['week'] = df.index // 168 + 1
        df.loc[8736:8760,'week']=52
        return df.groupby('week').sum() / 1000

    v_list = ['capacityInvest_Dvar', 'transInvest_Dvar', 'capacity_Pvar', 'capacityDel_Pvar', 'capacityDem_Dvar',
              'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
              'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar', 'carbon_Pvar', 'powerCosts_Pvar', 'capacityCosts_Pvar',
              'importCosts_Pvar', 'storageCosts_Pvar', 'turpeCosts_Pvar', 'Pmax_Pvar', 'max_PS_Dvar', 'carbonCosts_Pvar']
    Variables = {v: pd.read_csv(outputFolder + '/' + v + '.csv').drop(columns='Unnamed: 0') for v in v_list}

    inputDict = loadScenario(scenario)

    YEAR=Variables['power_Dvar'].set_index('YEAR_op').index.unique().values
    YEAR.sort()

    convFac = inputDict['conversionFactor']
    areaConsumption = inputDict['areaConsumption'].reset_index()
    Conso={y: areaConsumption.loc[areaConsumption['YEAR']==y].pivot(columns='RESOURCES',values='areaConsumption',index='TIMESTAMP') for y in YEAR }

    v = 'power_Dvar'
    Pel = {y: Variables[v].loc[Variables[v]['YEAR_op'] == y].pivot(columns='TECHNOLOGIES', values='power_Dvar', index='TIMESTAMP').drop(columns=['CCS1','CCS2']) for y in YEAR}
    for y in YEAR :
        for tech in list(Pel[y].columns):
            Pel[y][tech]=Pel[y][tech]*convFac.loc[('hydrogen',tech)].conversionFactor
    v = 'storageOut_Pvar'
    Pel_stock_out = {y: Variables[v][np.logical_and(Variables[v]['YEAR_op'] == y, Variables[v]['RESOURCES'] == 'hydrogen')].pivot(columns='STOCK_TECHNO',values=v,index='TIMESTAMP') for y in YEAR}
    v = 'storageIn_Pvar'
    Pel_stock_in = {y: Variables[v][np.logical_and(Variables[v]['YEAR_op'] == y, Variables[v]['RESOURCES'] == 'hydrogen')].pivot(columns='STOCK_TECHNO',values=v,index='TIMESTAMP') for y in YEAR}
    v = 'importation_Dvar'
    Pel_imp = {y: Variables[v][np.logical_and(Variables[v]['YEAR_op'] == y, Variables[v]['RESOURCES'] == 'hydrogen')].pivot(columns='RESOURCES',values=v,index='TIMESTAMP') for y in YEAR}

    Pel_exp = {y: -np.minimum(Pel_imp[y], 0) for y in Pel_imp.keys()}
    Pel_imp = {y: np.maximum(Pel_imp[y], 0) for y in Pel_imp.keys()}


    fig, ax = plt.subplots(4, 1, figsize=(6, 10), sharex=True)
    col=plt.cm.tab20c
    colBis=plt.cm.tab20b

    for k, yr in enumerate(YEAR):
        ax[k].yaxis.grid(linestyle='--', linewidth=0.5,zorder=-6)

        # power_Dvar
        Pel[yr] = weekly_average(Pel[yr])
        # storageOut_Pvar
        Pel_stock_out[yr] = weekly_average(Pel_stock_out[yr])
        # storageIn_Pvar
        Pel_stock_in[yr] = weekly_average(Pel_stock_in[yr])
        #Demand H2
        Conso[yr] = weekly_average(Conso[yr])
        # importation_Dvar
        Pel_imp[yr] = weekly_average(Pel_imp[yr])
        Pel_exp[yr] = weekly_average(Pel_exp[yr])

        # H2 production
        ax[k].bar(Pel[yr].index, Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL'], label='Electrolysis',color=col(9), zorder=-1)
        # ax[k].bar(Pel[yr].index, Pel[yr]['SMR_elec'] + Pel[yr]['SMR_elecCCS1'] + Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL'], label='eSMR',color=col(0), zorder=-2)
        ax[k].bar(Pel[yr].index, Pel[yr]['SMR'] + Pel[yr]['SMR_elec'] + Pel[yr]['SMR_elecCCS1'] + Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL'], label='SMR w/o CCUS',color=col(17), zorder=-3)
        ax[k].bar(Pel[yr].index,Pel[yr]['SMR'] + Pel[yr]['SMR + CCS1'] + Pel[yr]['SMR + CCS2'] + Pel[yr]['SMR_elec'] + Pel[yr]['SMR_elecCCS1'] + Pel[yr]['electrolysis_AEL'] + Pel[yr]['electrolysis_PEMEL'], label='SMR w CCUS',color=col(0), zorder=-4)
        #ax[k].bar(Pel[yr].index, Pel[yr]['SMR']  + Pel[yr]['SMR + CCS1'] + Pel[yr]['SMR + CCS2'] + Pel[yr]['SMR_elec'] + Pel[yr]['SMR_elecCCS1'] + Pel[yr]['electrolysis_AEL'] + Pel[yr]['electrolysis_PEMEL']+ Pel[yr]['cracking'],label='Methane cracking', color='#33caff', zorder=-5)
        ax[k].bar(Pel_stock_out[yr].index, Pel_stock_out[yr]['tankH2_G']+Pel_stock_in[yr]['saltCavernH2_G'] + Pel[yr]['SMR']  + Pel[yr]['SMR + CCS1'] + Pel[yr]['SMR + CCS2'] + Pel[yr]['SMR_elec'] + Pel[yr]['SMR_elecCCS1'] + Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL'],label='Stock - Out',color=colBis(18), zorder=-6)
        # ax[k].bar(Pel_stock_out[yr].index,Pel_stock_out[yr]['tankH2_G']+Pel_stock_in[yr]['saltCavernH2_G'] + Pel[yr]['SMR']  + Pel[yr]['SMR + CCS1'] + Pel[yr]['SMR + CCS2'] + Pel[yr]['SMR_elec'] + Pel[yr]['SMR_elecCCS1'] + Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL']+ Pel_imp[yr]['hydrogen'],label='Imports',color='#f74242',  zorder=-7)

        # H2 concumption
        ax[k].bar(Pel[yr].index, -Conso[yr]['hydrogen'], label='Consumption',color=colBis(10), zorder=-1)
        ax[k].bar(Pel_stock_in[yr].index,-Pel_stock_in[yr]['tankH2_G']-Pel_stock_in[yr]['saltCavernH2_G'] - Conso[yr]['hydrogen'], label='Stock - In',color=colBis(17),zorder=-2)

        ax[k].set_ylabel('Weakly production (GWh)')
        m=max((Pel_stock_in[yr]['tankH2_G']+Pel_stock_in[yr]['saltCavernH2_G'] + Conso[yr]['hydrogen']).max()+10,(Pel_stock_out[yr]['tankH2_G']+Pel_stock_in[yr]['saltCavernH2_G'] + Pel[yr]['SMR']  + Pel[yr]['SMR + CCS1'] + Pel[yr]['SMR + CCS2'] + Pel[yr]['SMR_elec'] + Pel[yr]['SMR_elecCCS1'] + Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL']+ Pel_imp[yr]['hydrogen']).max()+10)
        ax[k].set_ylim([-m, m])
        ax[k].set_title(YEAR[k])
        # Shrink all axis by 20%
        box = ax[k].get_position()
        ax[k].set_position([box.x0, box.y0, box.width * 0.73, box.height])

    ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax[-1].set_xlabel('Week')

    plt.savefig(outputFolder+'/Gestion H2.png')
    plt.show()

    return

def plot_H2Mean2050(scenario,outputFolder='Data/output/'):

    def weekly_average(df):
        df['week'] = df.index // 168 + 1
        df.loc[8736:8760,'week']=52
        return df.groupby('week').sum() / 1000

    v_list = ['capacityInvest_Dvar', 'transInvest_Dvar', 'capacity_Pvar', 'capacityDel_Pvar', 'capacityDem_Dvar',
              'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
              'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar', 'carbon_Pvar', 'powerCosts_Pvar', 'capacityCosts_Pvar',
              'importCosts_Pvar', 'storageCosts_Pvar', 'turpeCosts_Pvar', 'Pmax_Pvar', 'max_PS_Dvar', 'carbonCosts_Pvar']
    Variables = {v: pd.read_csv(outputFolder + '/' + v + '.csv').drop(columns='Unnamed: 0') for v in v_list}

    inputDict = loadScenario(scenario)

    YEAR=Variables['power_Dvar'].set_index('YEAR_op').index.unique().values
    YEAR.sort()

    convFac = inputDict['conversionFactor']
    areaConsumption = inputDict['areaConsumption'].reset_index()
    Conso={y: areaConsumption.loc[areaConsumption['YEAR']==y].pivot(columns='RESOURCES',values='areaConsumption',index='TIMESTAMP') for y in YEAR }

    v = 'power_Dvar'
    Pel = {y: Variables[v].loc[Variables[v]['YEAR_op'] == y].pivot(columns='TECHNOLOGIES', values='power_Dvar', index='TIMESTAMP').drop(columns=['CCS1','CCS2']) for y in YEAR}
    for y in YEAR :
        for tech in list(Pel[y].columns):
            Pel[y][tech]=Pel[y][tech]*convFac.loc[('hydrogen',tech)].conversionFactor
    v = 'storageOut_Pvar'
    Pel_stock_out = {y: Variables[v][np.logical_and(Variables[v]['YEAR_op'] == y, Variables[v]['RESOURCES'] == 'hydrogen')].pivot(columns='STOCK_TECHNO',values=v,index='TIMESTAMP') for y in YEAR}
    v = 'storageIn_Pvar'
    Pel_stock_in = {y: Variables[v][np.logical_and(Variables[v]['YEAR_op'] == y, Variables[v]['RESOURCES'] == 'hydrogen')].pivot(columns='STOCK_TECHNO',values=v,index='TIMESTAMP') for y in YEAR}
    v = 'importation_Dvar'
    Pel_imp = {y: Variables[v][np.logical_and(Variables[v]['YEAR_op'] == y, Variables[v]['RESOURCES'] == 'hydrogen')].pivot(columns='RESOURCES',values=v,index='TIMESTAMP') for y in YEAR}

    Pel_exp = {y: -np.minimum(Pel_imp[y], 0) for y in Pel_imp.keys()}
    Pel_imp = {y: np.maximum(Pel_imp[y], 0) for y in Pel_imp.keys()}


    fig, ax = plt.subplots(figsize=(6, 3.3))
    col=plt.cm.tab20c
    colBis=plt.cm.tab20b

    yr=2050

    ax.yaxis.grid(linestyle='--', linewidth=0.5,zorder=-6)

    # power_Dvar
    Pel[yr] = weekly_average(Pel[yr])
    # storageOut_Pvar
    Pel_stock_out[yr] = weekly_average(Pel_stock_out[yr])
    # storageIn_Pvar
    Pel_stock_in[yr] = weekly_average(Pel_stock_in[yr])
    #Demand H2
    Conso[yr] = weekly_average(Conso[yr])
    # importation_Dvar
    Pel_imp[yr] = weekly_average(Pel_imp[yr])
    Pel_exp[yr] = weekly_average(Pel_exp[yr])

    # H2 production
    ax.bar(Pel[yr].index, Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL'], label='Electrolysis',color=col(9), zorder=-1)
    # ax.bar(Pel[yr].index, Pel[yr]['SMR_elec'] + Pel[yr]['SMR_elecCCS1'] + Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL'], label='eSMR',color=col(0), zorder=-2)
    ax.bar(Pel[yr].index, Pel[yr]['SMR'] + Pel[yr]['SMR_elec'] + Pel[yr]['SMR_elecCCS1'] + Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL'], label='SMR w/o CCUS',color=col(17), zorder=-3)
    ax.bar(Pel[yr].index,Pel[yr]['SMR'] + Pel[yr]['SMR + CCS1'] + Pel[yr]['SMR + CCS2'] + Pel[yr]['SMR_elec'] + Pel[yr]['SMR_elecCCS1'] + Pel[yr]['electrolysis_AEL'] + Pel[yr]['electrolysis_PEMEL'], label='SMR w CCUS',color=col(0), zorder=-4)
    #ax.bar(Pel[yr].index, Pel[yr]['SMR']  + Pel[yr]['SMR + CCS1'] + Pel[yr]['SMR + CCS2'] + Pel[yr]['SMR_elec'] + Pel[yr]['SMR_elecCCS1'] + Pel[yr]['electrolysis_AEL'] + Pel[yr]['electrolysis_PEMEL']+ Pel[yr]['cracking'],label='Methane cracking', color='#33caff', zorder=-5)
    ax.bar(Pel_stock_out[yr].index, Pel_stock_out[yr]['tankH2_G']+Pel_stock_in[yr]['saltCavernH2_G'] + Pel[yr]['SMR']  + Pel[yr]['SMR + CCS1'] + Pel[yr]['SMR + CCS2'] + Pel[yr]['SMR_elec'] + Pel[yr]['SMR_elecCCS1'] + Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL'],label='Stock - Out',color=colBis(18), zorder=-6)
    # ax.bar(Pel_stock_out[yr].index,Pel_stock_out[yr]['tankH2_G']+Pel_stock_in[yr]['saltCavernH2_G'] + Pel[yr]['SMR']  + Pel[yr]['SMR + CCS1'] + Pel[yr]['SMR + CCS2'] + Pel[yr]['SMR_elec'] + Pel[yr]['SMR_elecCCS1'] + Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL']+ Pel_imp[yr]['hydrogen'],label='Imports',color='#f74242',  zorder=-7)

    # H2 concumption
    ax.bar(Pel[yr].index, -Conso[yr]['hydrogen'], label='Consumption',color=colBis(10), zorder=-1)
    ax.bar(Pel_stock_in[yr].index,-Pel_stock_in[yr]['tankH2_G'] - Conso[yr]['hydrogen'], label='Stock - In',color=colBis(17),zorder=-2)

    ax.set_ylabel('H$_2$ weekly production (GWh)')
    m=max((Pel_stock_in[yr]['tankH2_G']+Pel_stock_in[yr]['saltCavernH2_G'] + Conso[yr]['hydrogen']).max()+10,(Pel_stock_out[yr]['tankH2_G']+Pel_stock_in[yr]['saltCavernH2_G'] + Pel[yr]['SMR']  + Pel[yr]['SMR + CCS1'] + Pel[yr]['SMR + CCS2'] + Pel[yr]['SMR_elec'] + Pel[yr]['SMR_elecCCS1'] + Pel[yr]['electrolysis_AEL']+Pel[yr]['electrolysis_PEMEL']+ Pel_imp[yr]['hydrogen']).max()+10)
    ax.set_ylim([-m, m])
    # Shrink all axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0+0.03, box.width * 0.73, box.height])

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xlabel('Week')

    plt.savefig(outputFolder+'/Gestion H2 2050.png')
    plt.show()

    return

def plot_stock(outputFolder='Data/output/'):

    v_list = ['capacityInvest_Dvar', 'transInvest_Dvar', 'capacity_Pvar', 'capacityDel_Pvar', 'capacityDem_Dvar',
              'energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar',
              'stockLevel_Pvar', 'importation_Dvar', 'Cmax_Pvar', 'carbon_Pvar', 'powerCosts_Pvar', 'capacityCosts_Pvar',
              'importCosts_Pvar', 'storageCosts_Pvar', 'turpeCosts_Pvar', 'Pmax_Pvar', 'max_PS_Dvar', 'carbonCosts_Pvar']
    Variables = {v: pd.read_csv(outputFolder + '/' + v + '.csv').drop(columns='Unnamed: 0') for v in v_list}

    YEAR=Variables['power_Dvar'].set_index('YEAR_op').index.unique().values
    YEAR.sort()

    stock={y:Variables['stockLevel_Pvar'].loc[Variables['stockLevel_Pvar']['YEAR_op']==y].pivot(index='TIMESTAMP',columns='STOCK_TECHNO',values='stockLevel_Pvar') for y in YEAR}

    # hourly
    fig, ax = plt.subplots(4, 1, figsize=(6, 10), sharex=True,sharey=True)
    col=plt.cm.tab20c
    colBis=plt.cm.tab20b
    for k,yr in enumerate(YEAR):
        ax[k].plot(stock[yr].index,stock[yr]['tankH2_G']/1000,color=colBis(6),label='Stock hydrogen tank')
        ax[k].plot(stock[yr].index, stock[yr]['saltCavernH2_G'] / 1000,color=colBis(17), label='Stock hydrogen cavern')
        # ax[k].plot(stock[yr].index, stock[yr]['Battery']/1000, color=colBis(8),label='Stock electricity')
        ax[k].set_ylabel('Stock level (GWh)')
        # Shrink all axis by 20%
        box = ax[k].get_position()
        ax[k].set_position([box.x0, box.y0, box.width * 0.74, box.height])
    ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax[-1].set_xlabel('Hour')
    plt.savefig(outputFolder+'/Gestion stockage.png')
    plt.show()

    fig, ax = plt.subplots(figsize=(6,4))
    ax1=ax.twinx()
    l1=ax.plot(stock[2050].index,stock[2050]['tankH2_G']/1000,color=colBis(6),label='Stock tank')
    # l2=ax1.plot(stock[2050].index, stock[2050]['saltCavernH2_G'] / 1000,color=colBis(17), label='Stock cavern')
    l3=ax1.plot(stock[yr].index, stock[yr]['Battery']/1000, color=colBis(8),label='Stock battery')
    legendLabels=[l.get_label() for l in l1+l3]
    ax.set_ylabel('Tank stock level (GWh)')
    ax1.set_ylabel('Battery stock level (GWh)')
    # Shrink all axis by 20%
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.72, box.height])
    plt.legend(l1+l3,legendLabels,loc='upper left', bbox_to_anchor=(1.1, 1))
    ax.set_xlabel('Hour')
    plt.savefig(outputFolder+'/Stock 2050.png')
    plt.show()

    return

def extract_costs(scenario,outputFolder='Data/output/'):

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
    #
    # for y in YEAR : print((df1.loc[(y, 'electrolysis_AEL'), 'Prod'] + df1.loc[(y, 'electrolysis_PEMEL'), 'Prod']))

    #regroupement
    df1['type'] = 'None'
    df1.loc[(slice(None), 'SMR + CCS1'), 'type']='SMR'
    df1.loc[(slice(None), 'SMR + CCS2'), 'type']='SMR'
    df1.loc[(slice(None), 'SMR'), 'type']='SMR'
    df1.loc[(slice(None), 'SMR_elec'), 'type']='eSMR'
    df1.loc[(slice(None), 'SMR_elecCCS1'), 'type']='eSMR'
    df1.loc[(slice(None), 'electrolysis_AEL'), 'type']='AEL'
    df1.loc[(slice(None), 'electrolysis_PEMEL'), 'type']='AEL'
    #df1.loc[(slice(None), 'cracking'), 'type']='Cracking'

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

def plot_costs(df,outputFolder='Data/output/',comparaison=False):

    # caseNames=['Ref','Var_woSMR_2030']

    YEAR=df[list(df.keys())[0]].index.values
    YEAR.sort()
    dy=YEAR[1]-YEAR[0]
    y0=YEAR[0]-dy

    fig, ax = plt.subplots(figsize=(7,4.3))
    width= 0.30
    labels=list(df['SMR'].index)
    x = np.arange(len(labels))
    col=plt.cm.tab20c
    colBis=plt.cm.tab20b
    #code couleur Mines
    # dbl='#005E9E'
    # lbl='#2BA9FF'
    # rd='#fca592'
    # ye='#F8B740'
    # br='#e1a673'
    # gr='#cccccc'
    parameters={'axes.labelsize': 12,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
              'figure.titlesize': 15,
                'legend.fontsize':12}
    plt.rcParams.update(parameters)

    B=list(df.keys())
    B_nb=len(B)
    if B_nb%2>0:
        n=B_nb//2
        X=np.sort([-i*(width+0.05)  for i in np.arange(1,n+1)]+[0]+[i*(width+0.05)  for i in np.arange(1,n+1)])
    else:
        n=B_nb/2
        X=np.sort([-(width/2+0.025)-i*(width+0.05) for i in np.arange(n)]+[(width/2+0.025)+i*(width+0.05) for i in np.arange(n)])
        M=[X[i:i+2].mean() for i in np.arange(0,int(n+1),2)]

    meanCosts=[]
    horizonMean=[]
    c=0
    if comparaison==False:
        meanCosts = sum(df[k][['powerCosts','capacityCosts','capexElec','importElec','importGas','storageElec','storageH2','carbon','TURPE']].sum(axis=1) for k in B)/sum((df[k]['Prod']*30) for k in B)
        horizonMean=sum(df[k][['powerCosts','capacityCosts','capexElec','importElec','importGas','storageElec','storageH2','carbon','TURPE']].sum(axis=1) for k in B).sum()/(sum((df[k]['Prod']*30) for k in B).sum())
    else :
        if B_nb % 2 > 0:
            meanCosts=sum(df[k][['powerCosts','capacityCosts','capexElec','importElec','importGas','storageElec','storageH2','carbon','TURPE']].sum(axis=1) for k in B[0:2])/sum((df[k]['Prod']*30) for k in B[0:2])
            horizonMean.append(sum(df[k][['powerCosts','capacityCosts','capexElec','importElec','importGas','storageElec','storageH2','carbon','TURPE']].sum(axis=1) for k in B[0:2]).sum()/(sum((df[k]['Prod']*30) for k in B[0:2]).sum()))
            horizonMean.append(df[B[-1]][['powerCosts','capacityCosts','capexElec','importElec','importGas','storageElec','storageH2','carbon','TURPE']].sum(axis=1).sum()/(df[B[-1]]['Prod']*30).sum())
        else:
            for i in np.arange(0,int(n+1),2):
                meanCosts.append(sum(df[k][['powerCosts','capacityCosts','capexElec','importElec','importGas','storageElec','storageH2','carbon','TURPE']].sum(axis=1) for k in B[i:i+2])/sum((df[k]['Prod']*30) for k in B[i:i+2]))
                horizonMean.append(sum(df[k][['powerCosts', 'capacityCosts', 'capexElec', 'importElec','importGas', 'storageElec', 'storageH2', 'carbon','TURPE']].sum(axis=1) for k in B[i:i+2]).sum() / (sum((df[k]['Prod'] * 30) for k in B[i:i+2]).sum()))
                c=c+1


    # Create light blue Bars
    a={}
    for i in np.arange(B_nb):
        a[i]=list((df[B[i]]['capacityCosts']/(df[B[i]]['Prod']*30)).fillna(0))
        plt.bar(x + X[i], a[i], width, color=col(1),label="Fixed Costs" if i==0 else "",zorder=2)

    # Create dark blue Bars
    aa={}
    for i in np.arange(B_nb):
        aa[i]=list((df[B[i]]['powerCosts']/(df[B[i]]['Prod']*30)).fillna(0))
        plt.bar(x + X[i], aa[i], width,bottom=a[i], color=col(0),label="Variable Costs" if i==0 else "",zorder=2)

    # Create brown Bars
    b={}
    for i in np.arange(B_nb):
        b[i]=list((df[B[i]]['importGas']/(df[B[i]]['Prod']*30)).fillna(0))
        plt.bar(x + X[i], b[i], width, bottom=[i + j for i, j in zip(a[i],aa[i])], color=colBis(9),label="Gas" if i==0 else "",zorder=2)

    # Create green Bars
    c={}
    for i in np.arange(B_nb):
        c[i]=list((df[B[i]]['capexElec']/(df[B[i]]['Prod']*30)).fillna(0))
        plt.bar(x + X[i], c[i], width, bottom=[i + j + k for i, j, k in zip(a[i],aa[i],b[i])], color=col(9),label="Local RE capa" if i==0 else "",zorder=2)

    # Create dark red Bars
    d={}
    for i in np.arange(B_nb):
        d[i]=list((df[B[i]]['importElec']/(df[B[i]]['Prod']*30)).fillna(0))
        plt.bar(x + X[i], d[i], width, bottom=[i + j + k + l for i, j, k, l in zip(a[i],aa[i],b[i],c[i])], color=colBis(14),label="Grid electricity" if i==0 else "",zorder=2)

    # Create light red Bars
    e={}
    for i in np.arange(B_nb):
        e[i]=list((df[B[i]]['TURPE']/(df[B[i]]['Prod']*30)).fillna(0))
        plt.bar(x + X[i], e[i], width,  bottom=[i + j + k + l + m for i, j, k, l, m in zip(a[i],aa[i],b[i],c[i],d[i])], color=colBis(15),label="Network taxes" if i==0 else "",zorder=2)

    # Create purple Bars
    f={}
    for i in np.arange(B_nb):
        f[i]=list((df[B[i]]['storageH2']/(df[B[i]]['Prod']*30)).fillna(0))
        plt.bar(x + X[i], f[i], width,   bottom=[i + j + k + l + m + n for i, j, k, l, m, n in  zip(a[i],aa[i],b[i],c[i],d[i],e[i])], color=colBis(17),label="H2 storage capa" if i==0 else "",zorder=2)

    # Create yellow Bars
    g={}
    for i in np.arange(B_nb):
        g[i]=list((df[B[i]]['storageElec']/(df[B[i]]['Prod']*30)).fillna(0))
        plt.bar(x + X[i], g[i], width,   bottom=[i + j + k + l + m + n +o for i, j, k, l, m, n, o in zip(a[i],aa[i],b[i],c[i],d[i],e[i],f[i])], color=col(5),label="Elec storage capa" if i==0 else "",zorder=2)

    # Create grey Bars
    h={}
    for i in np.arange(B_nb):
        h[i]=list((df[B[i]]['carbon']/(df[B[i]]['Prod']*30)).fillna(0))
        plt.bar(x + X[i], h[i], width,   bottom=[i + j + k + l + m + n + o +p for i, j, k, l, m, n, o, p in zip(a[i],aa[i],b[i],c[i],d[i],e[i],f[i],g[i])], color=col(18),label="Carbon tax" if i==0 else "",zorder=2)

    s= {}
    maxi=[]
    for i in np.arange(B_nb):
        for j in x:
            ax.text((x+X[i])[j],[k + l + m + n + o + p + q + r + t + 0.05 for k, l, m, n, o, p, q, r,t in zip(a[i],aa[i],b[i],c[i],d[i],e[i],f[i],g[i],h[i])][j],B[i],ha='center',rotation=65)
        s[i]=[k + l + m + n + o + p + q + r + t for k, l, m, n, o, p, q, r, t in zip(a[i],aa[i],b[i],c[i],d[i],e[i],f[i],g[i],h[i])]
        print (B[i],'=',s[i])
        maxi.append(np.max(s[i]))

    for i,j in enumerate(maxi) :
        if j == np.inf: maxi.pop(i)

    print("H2 mean Cost =\n",meanCosts)
    print("H2 mean cost over horizon = ", meanCosts.mean())

    if comparaison==False:
        plt.plot(x,meanCosts,marker='D',color='none',markerfacecolor='None',markeredgecolor='black',markersize=6,markeredgewidth=1.5,label='H2 mean Price',zorder=3)
        plt.axhline(y=horizonMean,color='gray',linestyle='--',alpha=0.3,label='Weighted mean price',zorder=2)
    else:
        if n==1:
            plt.plot(x-0.025-width/2, meanCosts, marker='D', color='none', markerfacecolor='None', markeredgecolor='black',markersize=6, markeredgewidth=1.5, label='H2 mean Price',zorder=2)
            # plt.axhline(y=horizonMean[0],color='gray',linestyle='--',label='Mean price over horizon',alpha=0.3,zorder=2)
            # plt.text(-(width+0.05)*n,horizonMean[0], 'Base')
            # plt.axhline(y=horizonMean[1],color='gray',linestyle='--',alpha=0.3,zorder=2)
            # plt.text(-(width+0.05)*n, horizonMean[1], 'AEL Only')
        else :
            for i in np.arange(len(meanCosts)):
                plt.plot(x+M[i],meanCosts[i],marker='D',color='none',markerfacecolor='None',markeredgecolor='black',markersize=6,markeredgewidth=1.5,label='H2 mean Price' if i==0 else "",zorder=2)
                # plt.axhline(y=horizonMean[i],color='gray',linestyle='--',alpha=0.3, label='Mean over horizon' if i==0 else "",zorder=2)
                # plt.text(-(width+0.05)*n, horizonMean[i]-0.3 if caseNames[i]=='Base' else horizonMean[i]+0.1, caseNames[i],zorder=2)

    ax.set_ylabel('LCOH (€/kgH$_2$)')
    x=list(x)
    plt.xticks(x, ['2020-2030','2030-2040'])#,'2040-2050','2050-2060'

    ax.set_ylim([0,np.max(maxi)+1])
    ax.set_title("Hydrogen production costs")
    plt.grid(axis='y',alpha=0.5,zorder=1)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(outputFolder+'/H2 costs.png')
    plt.show()

    return

def  plot_costs2050(df,outputFolder='Data/output/',comparaison=False):

    caseNames=['BM=80€','BM=90€','BM=100€']

    YEAR=df[list(df.keys())[0]].index.values
    YEAR.sort()
    # dy=YEAR[1]-YEAR[0]
    # y0=YEAR[0]-dy

    fig, ax = plt.subplots(figsize=(7.5,5))
    width= 0.2
    labels=list(df['SMR BM=90€'].index)
    x = np.arange(len(labels))
    col=plt.cm.tab20c
    colBis=plt.cm.tab20b
    #code couleur Mines
    # dbl='#005E9E'
    # lbl='#2BA9FF'
    # rd='#fca592'
    # ye='#F8B740'
    # br='#e1a673'
    # gr='#cccccc'
    parameters={'axes.labelsize': 13,
                'xtick.labelsize': 13,
                'ytick.labelsize': 13,
                'legend.fontsize':13}
    plt.rcParams.update(parameters)

    B=list(df.keys())
    B_nb=len(B)
    if B_nb%2>0:
        n=B_nb//2
        X=np.sort([-i*(width+0.05)  for i in np.arange(1,n+1)]+[0]+[i*(width+0.05) for i in np.arange(1,n+1)])
    else:
        n=B_nb/2
        X=np.sort([-(width/2+0.025)-i*(width+0.05) for i in np.arange(n)]+[(width/2+0.025)+i*(width+0.05) for i in np.arange(n)])
        M=[X[i:i+2].mean() for i in np.arange(0,int(n+2),2)]

    meanCosts=[]
    horizonMean=[]
    c=0
    if comparaison==False:
        meanCosts = sum(df[k][['powerCosts','capacityCosts','capexElec','importElec','importGas','storageElec','storageH2','carbon','TURPE']].sum(axis=1) for k in B)/sum((df[k]['Prod']*30) for k in B)
        horizonMean=sum(df[k][['powerCosts','capacityCosts','capexElec','importElec','importGas','storageElec','storageH2','carbon','TURPE']].sum(axis=1) for k in B).sum()/(sum((df[k]['Prod']*30) for k in B).sum())
    else :
        if B_nb % 2 > 0:
            meanCosts=sum(df[k][['powerCosts','capacityCosts','capexElec','importElec','importGas','storageElec','storageH2','carbon','TURPE']].sum(axis=1) for k in B[0:2])/sum((df[k]['Prod']*30) for k in B[0:2])
            horizonMean.append(sum(df[k][['powerCosts','capacityCosts','capexElec','importElec','importGas','storageElec','storageH2','carbon','TURPE']].sum(axis=1) for k in B[0:2]).sum()/(sum((df[k]['Prod']*30) for k in B[0:2]).sum()))
            horizonMean.append(df[B[-1]][['powerCosts','capacityCosts','capexElec','importElec','importGas','storageElec','storageH2','carbon','TURPE']].sum(axis=1).sum()/(df[B[-1]]['Prod']*30).sum())
        else:
            for i in np.arange(0,int(n+2),2):
                meanCosts.append(sum(df[k][['powerCosts','capacityCosts','capexElec','importElec','importGas','storageElec','storageH2','carbon','TURPE']].sum(axis=1) for k in B[i:i+2])/sum((df[k]['Prod']*30) for k in B[i:i+2]))
                horizonMean.append(sum(df[k][['powerCosts', 'capacityCosts', 'capexElec', 'importElec','importGas', 'storageElec', 'storageH2', 'carbon','TURPE']].sum(axis=1) for k in B[i:i+2]).sum() / (sum((df[k]['Prod'] * 30) for k in B[i:i+2]).sum()))
                c=c+1

    # Create light blue Bars
    a={}
    for i in np.arange(B_nb):
        a[i]=list(df[B[i]]['capacityCosts']/(df[B[i]]['Prod']*30))
        plt.bar(x + X[i], a[i], width, color=col(1),label="Fixed Costs" if i==0 else "",zorder=2)

    # Create dark blue Bars
    aa={}
    for i in np.arange(B_nb):
        aa[i]=list(df[B[i]]['powerCosts']/(df[B[i]]['Prod']*30))
        plt.bar(x + X[i], aa[i], width,bottom=a[i], color=col(0),label="Variable Costs" if i==0 else "",zorder=2)

    # Create brown Bars
    b={}
    for i in np.arange(B_nb):
        b[i]=list(df[B[i]]['importGas']/(df[B[i]]['Prod']*30))
        plt.bar(x + X[i], b[i], width, bottom=[i + j for i, j in zip(a[i],aa[i])], color=colBis(9),label="Gas" if i==0 else "",zorder=2)

    # Create green Bars
    c={}
    for i in np.arange(B_nb):
        c[i]=list(df[B[i]]['capexElec']/(df[B[i]]['Prod']*30))
        plt.bar(x + X[i], c[i], width, bottom=[i + j + k for i, j, k in zip(a[i],aa[i],b[i])], color=col(9),label="Local RE capa" if i==0 else "",zorder=2)

    # Create dark red Bars
    d={}
    for i in np.arange(B_nb):
        d[i]=list(df[B[i]]['importElec']/(df[B[i]]['Prod']*30))
        plt.bar(x + X[i], d[i], width, bottom=[i + j + k + l for i, j, k, l in zip(a[i],aa[i],b[i],c[i])], color=colBis(14),label="Grid electricity" if i==0 else "",zorder=2)

    # Create light red Bars
    e={}
    for i in np.arange(B_nb):
        e[i]=list(df[B[i]]['TURPE']/(df[B[i]]['Prod']*30))
        plt.bar(x + X[i], e[i], width,  bottom=[i + j + k + l + m for i, j, k, l, m in zip(a[i],aa[i],b[i],c[i],d[i])], color=colBis(15),label="Network taxes" if i==0 else "",zorder=2)

    # Create purple Bars
    f={}
    for i in np.arange(B_nb):
        f[i]=list(df[B[i]]['storageH2']/(df[B[i]]['Prod']*30))
        plt.bar(x + X[i], f[i], width,   bottom=[i + j + k + l + m + n for i, j, k, l, m, n in  zip(a[i],aa[i],b[i],c[i],d[i],e[i])], color=colBis(17),label="H2 storage capa" if i==0 else "",zorder=2)

    # Create yellow Bars
    g={}
    for i in np.arange(B_nb):
        g=list(df[B[i]]['storageElec']/(df[B[i]]['Prod']*30))
        plt.bar(x + X[i], g, width,   bottom=[i + j + k + l + m + n +o for i, j, k, l, m, n, o in zip(a[i],aa[i],b[i],c[i],d[i],e[i],f[i])], color=col(5),label="Elec storage capa" if i==0 else "",zorder=2)

    # Create grey Bars
    h={}
    for i in np.arange(B_nb):
        h[i]=list(df[B[i]]['carbon']/(df[B[i]]['Prod']*30))
        plt.bar(x + X[i], h[i], width,   bottom=[i + j + k + l + m + n + o +p for i, j, k, l, m, n, o, p in zip(a[i],aa[i],b[i],c[i],d[i],e[i],f[i],g[i])], color=col(18),label="Carbon tax" if i==0 else "",zorder=2)

    s= {}
    for i in np.arange(B_nb):
        for j in x:
            ax.text((x+X[i])[j],[k + l + m + n + o + p + q + r + t + 0.05 for k, l, m, n, o, p, q, r,t in zip(a[i],aa[i],b[i],c[i],d[i],e[i],f[i],g[i],h[i])][j],B[i][:3]+' '+caseNames[i//2],ha='center',rotation=60,fontsize=11)
        s[i]=[k + l + m + n + o + p + q + r + t for k, l, m, n, o, p, q, r, t in zip(a[i],aa[i],b[i],c[i],d[i],e[i],f[i],g[i],h[i])]
        print (B[i],'=',s[i])

    print("H2 mean Cost =\n",meanCosts)
    # print("H2 mean cost over horizon = ", meanCosts.mean())

    if comparaison==False:
        plt.plot(x,meanCosts,marker='D',color='none',markerfacecolor='None',markeredgecolor='black',markersize=6,markeredgewidth=1.5,label='H2 mean Price',zorder=3)
        plt.axhline(y=horizonMean,color='gray',linestyle='--',alpha=0.3,label='Weighted mean price',zorder=2)
    else:
        if n==1:
            plt.plot(x-0.025-width/2, meanCosts, marker='D', color='none', markerfacecolor='None', markeredgecolor='black',markersize=6, markeredgewidth=1.5, label='H2 mean Price',zorder=2)
            # plt.axhline(y=horizonMean[0],color='gray',linestyle='--',label='Mean price over horizon',alpha=0.3,zorder=2)
            # plt.text(-(width+0.05)*n,horizonMean[0], 'Base')
            # plt.axhline(y=horizonMean[1],color='gray',linestyle='--',alpha=0.3,zorder=2)
            # plt.text(-(width+0.05)*n, horizonMean[1], 'AEL Only')
        else :
            for i in np.arange(len(meanCosts)):
                plt.plot(x+M[i],meanCosts[i],marker='D',color='none',markerfacecolor='None',markeredgecolor='black',markersize=6,markeredgewidth=1.5,label='H2 mean Price' if i==0 else "",zorder=2)
                # plt.axhline(y=horizonMean[i],color='gray',linestyle='--',alpha=0.3, label='Mean over horizon' if i==0 else "",zorder=2)
                # plt.text(-(width+0.05)*n, horizonMean[i]-0.3 if caseNames[i]=='Base' else horizonMean[i]+0.1, caseNames[i],zorder=2)

    ax.set_ylabel('Costs (€/kgH$_2$)')
    x=list(x)
    plt.xticks(x, ['2050-2060'])
    m=max(s.values())
    ax.set_ylim([0,np.round(m[0])+3])
    # ax.set_title("Hydrogen production costs")
    plt.grid(axis='y',alpha=0.5,zorder=1)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.68 , box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(outputFolder+'/H2 costs.png')
    plt.show()

    return

def  plot_costs2030(df,outputFolder='Data/output/'):

    fig, ax = plt.subplots(figsize=(7.5,5))
    width= 0.3
    labels=[2030]
    x = np.arange(len(labels))
    col=plt.cm.tab20c
    colBis=plt.cm.tab20b

    parameters={'axes.labelsize': 13,
                'xtick.labelsize': 13,
                'ytick.labelsize': 13,
                'legend.fontsize':13}
    plt.rcParams.update(parameters)

    horizonMean=(df[['powerCosts','capacityCosts','capexElec','importElec','importGas','storageElec','storageH2','carbon','TURPE']]).sum()/(df['Prod']*30)

    # Create light blue Bars
    a=df['capacityCosts']/(df['Prod']*30)
    plt.bar(x, a, width, color=col(1),label="Fixed Costs",zorder=2)

    # Create dark blue Bars
    aa=df['powerCosts']/(df['Prod']*30)
    plt.bar(x, aa, width,bottom=a, color=col(0),label="Variable Costs" ,zorder=2)

    # Create brown Bars
    b=df['importGas']/(df['Prod']*30)
    plt.bar(x, b, width, bottom=a+aa, color=colBis(9),label="Gas" ,zorder=2)

    # Create green Bars
    c=df['capexElec']/(df['Prod']*30)
    plt.bar(x, c, width, bottom=a+aa+b, color=col(9),label="Local RE capa" ,zorder=2)

    # Create dark red Bars
    d=df['importElec']/(df['Prod']*30)
    plt.bar(x, d, width, bottom=a+aa+b+c, color=colBis(14),label="Grid electricity" ,zorder=2)

    # Create light red Bars
    e=df['TURPE']/(df['Prod']*30)
    plt.bar(x, e, width,  bottom=a+aa+b+c+d, color=colBis(15),label="Network taxes" ,zorder=2)

    # Create purple Bars
    f=df['storageH2']/(df['Prod']*30)
    plt.bar(x, f, width, bottom=a+aa+b+c+d+e, color=colBis(17),label="H2 storage capa" ,zorder=2)

    # Create yellow Bars
    g=df['storageElec']/(df['Prod']*30)
    plt.bar(x, g, width,   bottom=a+aa+b+c+d+e+f, color=col(5),label="Elec storage capa" ,zorder=2)

    # Create grey Bars
    h=df['carbon']/(df['Prod']*30)
    plt.bar(x, h, width,   bottom=a+aa+b+c+d+e+f+g, color=col(18),label="Carbon tax" ,zorder=2)

    plt.axhline(y=horizonMean,color='gray',linestyle='--',alpha=0.3,label='Weighted mean price',zorder=2)

    ax.set_ylabel('Costs (€/kgH$_2$)')
    plt.xticks(x, ['2030'])
    m=a+aa+b+c+d+e+f+g+h
    ax.set_ylim([0,np.round(m)+1])
    # ax.set_title("Hydrogen production costs")
    plt.grid(axis='y',alpha=0.5,zorder=1)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.68 , box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(outputFolder+'/H2 costs.png')
    plt.show()

    return

def plot_carbon(outputFolder='Data/output/'):

    carbon=pd.read_csv(outputFolder+'/carbon_Pvar.csv').groupby('YEAR_op').sum().drop(columns=['TIMESTAMP','Unnamed: 0'])/1000000
    carbon=carbon.sort_index()
    Prod=pd.read_csv(outputFolder+'/energy_Pvar.csv').groupby(['YEAR_op','RESOURCES']).sum().drop(columns=['TIMESTAMP','Unnamed: 0']).loc[(slice(None),'hydrogen'),'energy_Pvar'].reset_index().set_index('YEAR_op').drop(columns='RESOURCES')
    carbonContent=carbon['carbon_Pvar']*1000000/(Prod['energy_Pvar']*30)

    # carbon_ref=pd.read_csv('Data/output\SmrOnly_var4bis_PACA/carbon_Pvar.csv').groupby('YEAR_op').sum().drop(columns=['TIMESTAMP','Unnamed: 0'])/1000000
    # carbon_ref=carbon_ref.sort_index()
    # Prod_ref=pd.read_csv('Data/output\SmrOnly_var4bis_PACA/energy_Pvar.csv').groupby(['YEAR_op','RESOURCES']).sum().drop(columns=['TIMESTAMP','Unnamed: 0']).loc[(slice(None),'hydrogen'),'energy_Pvar'].reset_index().set_index('YEAR_op').drop(columns='RESOURCES')
    # carbonContent_ref = carbon_ref['carbon_Pvar'] * 1000000 / (Prod_ref['energy_Pvar'] * 30)

    # test=pd.read_csv('Data/output\Ref_Base_PACA/carbon_Pvar.csv').groupby('YEAR_op').sum().drop(columns=['TIMESTAMP','Unnamed: 0'])/1000000
    # test=test.sort_index()
    # Prod_test=pd.read_csv('Data/output\Ref_Base_PACA/energy_Pvar.csv').groupby(['YEAR_op','RESOURCES']).sum().drop(columns=['TIMESTAMP','Unnamed: 0']).loc[(slice(None),'hydrogen'),'energy_Pvar'].reset_index().set_index('YEAR_op').drop(columns='RESOURCES')
    # carbonContent_test= test['carbon_Pvar'] * 1000000 / (Prod_test['energy_Pvar'] * 30)

    # avoided=[carbon_ref.carbon_Pvar.loc[2020]-carbon.carbon_Pvar.loc[2050],carbon_ref.carbon_Pvar.loc[2050]-carbon.carbon_Pvar.loc[2050]]

    YEAR=carbon.index.unique().values

    # plt.plot(YEAR,carbon_ref.carbon_Pvar,label='Reference CO2 emission')
    plt.plot(YEAR,carbon.carbon_Pvar,label='CO2 emissions',color='g')
    # plt.plot(YEAR, test.carbon_Pvar,linestyle='--',label='Base CO2 emissions',color='g')

    # plt.fill_between(YEAR,carbon_ref.carbon_Pvar,carbon.carbon_Pvar,color='none',edgecolor='#cccccc',hatch='//')
    plt.title('CO2 Avoided emissions')
    plt.legend()
    plt.ylabel('kt/yr')
    plt.xlabel('year')
    plt.savefig(outputFolder+'/Emissions.png')
    plt.show()

    # plt.plot(YEAR,carbonContent_ref,label='Reference CO2 content')
    plt.plot(YEAR,carbonContent,label='CO2 content',color='g')
    # plt.plot(YEAR, carbonContent_test,linestyle='--',label='Base CO2 content',color='g')

    # plt.fill_between(YEAR,carbonContent_ref,carbonContent,color='none',edgecolor='#cccccc',hatch='//')
    plt.title('Carbon content of hydrogen')
    plt.legend()
    plt.ylabel('kgCO2/kgH2')
    plt.xlabel('year')
    plt.savefig(outputFolder+'/Carbon content.png')
    plt.show()

    return

def plot_carbonCosts(dico,scenarioNames,outputPath='Data/output/'):

    YEAR=list(list(dico.items())[0][1].items())[0][1].index.values
    YEAR.sort()

    carbonContent = {}
    meanPrice = {}
    horizonMean={}
    horizonContent={}
    for s in list(dico.keys()):
        meanPrice[s]=sum(dico[s][k][['powerCosts','capacityCosts','capexElec','importElec','importGas','storageElec','storageH2','carbon','TURPE']].sum(axis=1) for k in list(dico[s].keys()))/(sum((dico[s][k]['Prod']*30) for k in list(dico[s].keys())))
        horizonMean[s]=sum(dico[s][k][['powerCosts','capacityCosts','capexElec','importElec','importGas','storageElec','storageH2','carbon','TURPE']].sum(axis=1) for k in list(dico[s].keys())).sum()/(sum((dico[s][k]['Prod']*30) for k in list(dico[s].keys())).sum())
        carbon=pd.read_csv(outputPath+s+'_PACA/carbon_Pvar.csv').groupby('YEAR_op').sum().drop(columns=['TIMESTAMP','Unnamed: 0'])
        carbon=carbon.sort_index()
        carbonContent[s]=carbon['carbon_Pvar']/(sum((dico[s][k]['Prod']*30) for k in list(dico[s].keys())))
        horizonContent[s]=carbon['carbon_Pvar'].sum()/(sum((dico[s][k]['Prod']*30) for k in list(dico[s].keys())).sum())
        # plt.scatter(horizonContent[s],horizonMean[s],label=s)

    # plt.title('Cost and carbon content of hydrogen horizon mean')
    # plt.legend()
    # plt.ylabel('€/kgH2')
    # plt.xlabel('kgCo2/kgH2')
    # plt.savefig(outputPath + '/Comparaison carbon horizon mean.png')
    # plt.show()
    #
    # fig,ax=plt.subplots(1,3,sharey=True,sharex=True)
    # for k,y in enumerate(YEAR[1:]):
    #     for s in list(dico.keys()):
    #         ax[k].scatter(carbonContent[s].loc[y],meanPrice[s].loc[y],label=s)
    #         ax[k].set_title(str(y))
    #
    # ax[0].set_ylabel('€/kgH2')
    # ax[1].set_xlabel('kgCo2/kgH2')
    # ax[-1].legend()
    # plt.savefig(outputPath + '/Comparaison carbon subplot.png')
    # plt.show()

    fig,ax=plt.subplots()
    col = plt.cm.tab20c
    colBis = plt.cm.tab20b
    dico_color={'Ref':(colBis,0),'BM_':(col,0),'woSMR_':(colBis,16),'CO2_':(colBis,8),'Re_':(col,8)}
    colNumber=[]
    variable=[]
    n=0
    for l,s in enumerate(list(dico.keys())):
        for var in list(dico_color.keys()):
            if var in s:
                variable.append(var)
                if variable[l-1]==variable[l]:
                    n=n+1
                else :
                    n=0
                colNumber.append((dico_color[var][0],dico_color[var][1]+n))
    mark=['s','D','o']

    n=0
    for k,y in enumerate(YEAR[1:]):
        for l,s in enumerate(list(dico.keys())):
            ax.scatter(carbonContent[s].loc[y],meanPrice[s].loc[y],marker=mark[k],color=col(l*4),zorder=2) #colNumber[l][0+l*4](colNumber[l][1])
        ax.plot([],[],marker=mark[k],linestyle='',color='grey',label=str(y+5))
    for l,s in enumerate(list(dico.keys())):
        ax.plot(carbonContent[s].iloc[1:].values,meanPrice[s].iloc[1:].values,marker='',color=col(l*4),label=scenarioNames[n],linestyle='--',alpha=0.5,zorder=2)
        n+=1

    plt.title('')
    plt.ylabel('€/kgH$_2$')
    plt.xlabel('kgCO$_2$/kgH$_2$')
    # plt.title('LCOH and carbon content evolution')
    plt.grid(axis='y',alpha=0.5,zorder=1)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.72, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(outputPath + '/Comparaison carbon.png')
    plt.show()

    return

def plot_carbonCosts2050(dico,scenarioNames,outputPath='Data/output/'):

    YEAR=list(list(dico.items())[0][1].items())[0][1].index.values
    YEAR.sort()

    carbonContent = {}
    meanPrice = {}
    horizonMean={}
    horizonContent={}
    for s in list(dico.keys()):
        meanPrice[s]=sum(dico[s][k][['powerCosts','capacityCosts','capexElec','importElec','importGas','storageElec','storageH2','carbon','TURPE']].sum(axis=1) for k in list(dico[s].keys()))/(sum((dico[s][k]['Prod']*30) for k in list(dico[s].keys())))
        horizonMean[s]=sum(dico[s][k][['powerCosts','capacityCosts','capexElec','importElec','importGas','storageElec','storageH2','carbon','TURPE']].sum(axis=1) for k in list(dico[s].keys())).sum()/(sum((dico[s][k]['Prod']*30) for k in list(dico[s].keys())).sum())
        carbon=pd.read_csv(outputPath+s+'_PACA/carbon_Pvar.csv').groupby('YEAR_op').sum().drop(columns=['TIMESTAMP','Unnamed: 0'])
        carbon=carbon.sort_index()
        carbonContent[s]=carbon['carbon_Pvar']/(sum((dico[s][k]['Prod']*30) for k in list(dico[s].keys())))
        horizonContent[s]=carbon['carbon_Pvar'].sum()/(sum((dico[s][k]['Prod']*30) for k in list(dico[s].keys())).sum())


    fig,ax=plt.subplots(figsize=(6,4))
    col = plt.cm.tab20c
    colBis = plt.cm.tab20b
    dico_color={'Ref':(colBis,0),'BM_':(col,0),'woSMR_':(colBis,16),'CO2_':(colBis,8),'Re_':(col,8)}
    dico_mark=['o','D','s','v','^']
    colNumber=[]
    markNumber=[]
    variable=[]
    n=0
    c=0
    for l,s in enumerate(list(dico.keys())):
        for var in list(dico_color.keys()):
            if var in s:
                variable.append(var)
                if variable[l-1]==variable[l]:
                    n=n+1
                else :
                    n=0
                    c=c+1
                colNumber.append((dico_color[var][0],dico_color[var][1]+n))
                markNumber.append(dico_mark[c])


    # n=0
    y=2050
    for l,s in enumerate(list(dico.keys())):
        ax.scatter(carbonContent[s].loc[y],meanPrice[s].loc[y],marker=markNumber[l],markersize=4,color=colNumber[l][0](colNumber[l][1]),zorder=2,label=scenarioNames[l])
    # for l,s in enumerate(list(dico.keys())):
    #     ax.plot(carbonContent[s].iloc[1:].values,meanPrice[s].iloc[1:].values,marker='',color=colNumber[l][0](colNumber[l][1]),label=scenarioNames[n],linestyle='--',alpha=0.5,zorder=2)
    #     n+=1

    plt.title('')
    plt.ylabel('€/kgH$_2$')
    plt.xlabel('kgCO$_2$/kgH$_2$')
    # plt.title('LCOH and carbon content evolution')
    plt.grid(axis='y',alpha=0.5,zorder=1)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0+0.05, box.width * 0.7, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(outputPath + '/Comparaison carbon 2050.png')
    plt.show()

    return

def extract_energy(scenario,outputFolder='Data/output'):
    v_list = [ 'capacity_Pvar','energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar','importation_Dvar','carbon_Pvar',
             'powerCosts_Pvar', 'capacityCosts_Pvar','importCosts_Pvar', 'storageCosts_Pvar', 'turpeCosts_Pvar','carbonCosts_Pvar']#, 'exportation_Dvar']
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
    # df_export = Variables['exportation_Dvar'].groupby(['YEAR_op', 'RESOURCES']).sum().loc[
    #     (slice(None), 'electricity'), 'exportation_Dvar'].reset_index().drop(columns='RESOURCES').set_index('YEAR_op')
    df_feedRE = (df_renewables) / 1.54 / 1000 #- df_export['exportation_Dvar']

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

def extract_capa(scenario,outputFolder='Data/output'):
    v_list = [ 'capacity_Pvar','energy_Pvar', 'power_Dvar', 'storageConsumption_Pvar', 'storageIn_Pvar', 'storageOut_Pvar','importation_Dvar','carbon_Pvar',
             'powerCosts_Pvar', 'capacityCosts_Pvar','importCosts_Pvar', 'storageCosts_Pvar', 'turpeCosts_Pvar','carbonCosts_Pvar']#, 'exportation_Dvar']
    Variables = {v: pd.read_csv(outputFolder + '/' + v + '.csv').drop(columns='Unnamed: 0') for v in v_list}

    #inputDict = loadScenario(scenario)

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

def plot_compare_energy_carbon(dico_ener,scenarioNames,outputPath='Data/output/'):
    YEAR=list(list(dico_ener.items())[0][1].items())[0][1].index.values
    YEAR.sort()
    L = list(dico_ener.keys())

    for l in L :
        dico_ener[l]=dico_ener[l].loc[2030:2050]

    fig,ax = plt.subplots(2,1,sharex=True)
    width = 0.15
    col = sb.color_palette('muted')
    style={'color':[[col[7],'#969696','#AAAAAA'],['#005E9E','#1472B2','#2886C6'],[col[2],'#7EE078','#92F48C']],'hatch':['','xx','++'],'alpha':[1,1,1],'marker':['.','x','+']}
    labels = list(YEAR[1:])
    x = np.arange(len(labels))

    for k,l in enumerate(L):
        # Create grey Bars
        l1 = list(dico_ener[l]['SMR w/o CCUS']/1000)
        ax[0].bar(x-(len(L)/2+1)*width + 2*width*k, l1, width, color=style['color'][0][k],edgecolor=style['color'][0][0],alpha=style['alpha'][k],hatch=style['hatch'][k], label="SMR w/o CCUS" if k==0 else '')
        # Create blue Bars
        l2 = list((dico_ener[l]['SMR + CCUS 50%']+dico_ener[l]['SMR + CCUS 90%'])/1000)
        ax[0].bar(x-(len(L)/2+1)*width + 2*width*k, l2, width, bottom=l1,color=style['color'][1][k],edgecolor=style['color'][1][0],alpha=style['alpha'][k],hatch=style['hatch'][k], label="SMR + CCUS" if k==0 else '')
        # Create light Bars
        l7 = list((dico_ener[l]['Alkaline electrolysis'] + dico_ener[l]['PEM electrolysis'])/1000)
        ax[0].bar(x-len(L)/2*width + 2*width*k, l7, width, color=style['color'][2][k],edgecolor=style['color'][2][0],alpha=style['alpha'][k],hatch=style['hatch'][k], label="AEL grid feed" if k==0 else '')
        # Create biogas Bars
        l8=list(dico_ener[l]['feedBiogas']/1000)
        ax[0].bar(x-(len(L)/2+1)*width + 2*width*k,l8,width,color='none',linewidth=2,edgecolor=col[6],label="Biomethane feed" if k==0 else '')
        # Create Local renewables bars
        l9=list(dico_ener[l]['feedRE']/1000)
        ax[0].bar(x-len(L)/2*width + 2*width*k,l9,width,color='none',linewidth=2,edgecolor=col[3],label="AEL local feed" if k==0 else '')
        # add carbon emission
        l10=list(dico_ener[l]['carbon'])
        ax[1].plot(l10,marker=style['marker'][k],color=style['color'][1][k],label=scenarioNames[k])

    ax[0].set_ylabel('H2 production (TWh)')
    ax[1].set_ylabel('Carbon content (kgCO2/kgH2)')
    ax[0].set_title("H2 production and carbon content")
    plt.xticks(x, ['2030', '2040', '2050'])#,'2060'])
    # Shrink current axis by 20%
    box = ax[0].get_position()
    ax[0].set_position([box.x0, box.y0, box.width * 0.72, box.height])
    # Put a legend to the right of the current axis
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.72, box.height])
    # Put a legend to the right of the current axis
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(outputPath+'/Comparison energy.png')
    plt.show()

    return

def plot_compare_energy_costs(dico_ener,scenarioNames,outputPath='Data/output/'):
    YEAR=list(list(dico_ener.items())[0][1].items())[0][1].index.values
    YEAR.sort()
    L = list(dico_ener.keys())

    for l in L :
        dico_ener[l]=dico_ener[l].loc[2030:2050]

    fig, ax = plt.subplots(2,1,sharex=True)
    width = 0.15
    col = sb.color_palette('muted')
    style={'color':[[col[7],'#969696','#AAAAAA'],['#005E9E','#1472B2','#2886C6'],[col[2],'#7EE078','#92F48C']],'hatch':['','xx','++'],'alpha':[1,1,1],'marker':['.','x','+']}
    labels = list(YEAR[1:])
    x = np.arange(len(labels))

    for k,l in enumerate(L):
        # Create grey Bars
        l1 = list(dico_ener[l]['SMR w/o CCUS']/1000)
        ax[0].bar(x-(len(L)/2+1)*width + 2*width*k, l1, width, color=style['color'][0][k],edgecolor=style['color'][0][0],alpha=style['alpha'][k],hatch=style['hatch'][k], label="SMR w/o CCUS" if k==0 else '')
        # Create blue Bars
        l2 = list((dico_ener[l]['SMR + CCUS 50%']+dico_ener[l]['SMR + CCUS 90%'])/1000)
        ax[0].bar(x-(len(L)/2+1)*width + 2*width*k, l2, width, bottom=l1,color=style['color'][1][k],edgecolor=style['color'][1][0],alpha=style['alpha'][k],hatch=style['hatch'][k], label="SMR + CCUS" if k==0 else '')
        # Create light Bars
        l7 = list((dico_ener[l]['Alkaline electrolysis'] + dico_ener[l]['PEM electrolysis'])/1000)
        ax[0].bar(x-len(L)/2*width + 2*width*k, l7, width, color=style['color'][2][k],edgecolor=style['color'][2][0],alpha=style['alpha'][k],hatch=style['hatch'][k], label="AEL grid feed" if k==0 else '')
        # Create biogas Bars
        l8=list(dico_ener[l]['feedBiogas']/1000)
        ax[0].bar(x-(len(L)/2+1)*width + 2*width*k,l8,width,color='none',linewidth=2,edgecolor=col[6],label="Biomethane feed" if k==0 else '')
        # Create Local renewables bars
        l9=list(dico_ener[l]['feedRE']/1000)
        ax[0].bar(x-len(L)/2*width + 2*width*k,l9,width,color='none',linewidth=2,edgecolor=col[3],label="AEL local feed" if k==0 else '')
        # add carbon emission
        l10=list(dico_ener[l]['costs'])
        ax[1].plot(l10,marker=style['marker'][k],color=style['color'][1][k],label=scenarioNames[k])

    ax[0].set_ylabel('H2 production (TWh)')
    ax[1].set_ylabel('H2 costs (€/kgH2)')
    ax[0].set_title("H2 production and H2 costs")
    plt.xticks(x, ['2030', '2040', '2050'])#,'2060'])
    # Shrink current axis by 20%
    box = ax[0].get_position()
    ax[0].set_position([box.x0, box.y0, box.width * 0.72, box.height])
    # Put a legend to the right of the current axis
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.72, box.height])
    # Put a legend to the right of the current axis
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(outputPath+'/Comparison energy - EnR.png')
    plt.show()

    return

def plot_compare_elecPrice(dico,scenarioNames,outputPath='Data/output/'):

    YEAR=list(list(dico.items())[0][1].items())[0][1].index.values
    YEAR.sort()

    carbonContent = {}
    meanPrice = {}
    for s in list(dico.keys()):
        meanPrice[s]=dico[s]['AEL'][['powerCosts','capacityCosts','capexElec','importElec','importGas','storageElec','storageH2','carbon','TURPE']].sum(axis=1)/(dico[s]['AEL']['Prod']*30)
        carbon=pd.read_csv(outputPath+s+'_PACA/carbon_Pvar.csv').groupby('YEAR_op').sum().drop(columns=['TIMESTAMP','Unnamed: 0'])
        carbon=carbon.sort_index()
        carbonContent[s]=carbon['carbon_Pvar']/(sum((dico[s][k]['Prod']*30) for k in list(dico[s].keys())))

    fig, ax = plt.subplots(figsize=(6, 4))
    col = plt.cm.tab20c
    mark=['s','D','o']
    n=0
    for k,y in enumerate(YEAR[1:]):
        for l,s in enumerate(list(dico.keys())):
            ax.scatter(carbonContent[s].loc[y],meanPrice[s].loc[y],marker=mark[k],color=col(l*4),zorder=2)
            print(carbonContent[s].loc[y],meanPrice[s].loc[y])
        ax.plot([],[],marker=mark[k],linestyle='',color='grey',label=str(y),zorder=2)
    for l,s in enumerate(list(dico.keys())):
        ax.plot(carbonContent[s].iloc[1:].values,meanPrice[s].iloc[1:].values,marker='',color=col(l*4),label=scenarioNames[n],linestyle='--',alpha=0.5,zorder=2)
        n+=1

    plt.ylabel('€/kgH$_2$')
    plt.xlabel('kgCO$_2$/kgH$_2$')
    # plt.title('Electrolysis LCOH and carbon content evolution')
    plt.grid(axis='y',alpha=0.5,zorder=1)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0+0.05, box.width * 0.71, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(outputPath + '/Comparaison electrolysis.png')
    plt.show()

    return

def plot_compare_energy_and_carbon(dico_ener, scenarioNames, outputPath='Data/output/'):
    YEAR = list(list(dico_ener.items())[0][1].items())[0][1].index.values
    YEAR.sort()
    L = list(dico_ener.keys())

    for l in L:
        dico_ener[l] = dico_ener[l].loc[2030:2050]

    # fig=plt.figure(figsize=(8,6))
    # F = GridSpec(2,3,figure=fig)
    fig,ax=plt.subplots(1,3,sharey=True,figsize=(10,3))
    # ax1 = fig.add_subplot(F[0,0])
    # ax2 = fig.add_subplot(F[0,1],sharey=ax1)
    # ax3 = fig.add_subplot(F[0,2],sharey=ax1)
    # ax = [ax1, ax2, ax3]
    # ay = fig.add_subplot(F[1,:])
    width = 0.35
    col = plt.cm.tab20c
    labels = list(YEAR[1:])
    x = np.arange(len(labels))

    for k, l in enumerate(L):
        ax[k].grid(axis='y',alpha=0.5,color=col(19),zorder=1)

    for k, l in enumerate(L):
        # Create grey Bars
        l1 = list(dico_ener[l]['SMR w/o CCUS'] / 1000)
        ax[k].bar(x - width/2, l1, width, color=col(17),label="SMR w/o CCUS" if k == 2 else '',zorder=2)
        # Create blue Bars
        l2 = list((dico_ener[l]['SMR + CCUS 50%'] + dico_ener[l]['SMR + CCUS 90%']) / 1000)
        ax[k].bar(x - width/2 , l2, width, bottom=l1, color=col(0),label="SMR + CCUS" if k == 2 else '',zorder=2)
        # Create biogas Bars
        plt.rcParams['hatch.linewidth'] = 8
        plt.rcParams['hatch.color'] = col(3)
        l8 = list(dico_ener[l]['feedBiogas'] / 1000)
        ax[k].bar(x - width/2, l8, width, color='none', hatch='/',linewidth=0.5, edgecolor=col(3),alpha=0.8,label="Biomethane feed" if k == 2 else '',zorder=3)
        # Create green Bars
        l7 = list((dico_ener[l]['Alkaline electrolysis']+ dico_ener[l]['PEM electrolysis']) / 1000)
        ax[k].bar(x + width/2, l7, width, color=col(8),label="AEL grid feed" if k == 2 else '',zorder=2)
        # Create Local renewables bars
        l9 = list(dico_ener[l]['feedRE'] / 1000)
        ax[k].bar(x + width/2, l9, width,color=col(9),label="AEL local feed" if k == 2 else '',zorder=3)

    ax[0].set_ylabel('H$_2$ production (TWh/an)')
    for k,l in enumerate(L):
        ax[k].set_title(scenarioNames[k])
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(['2035', '2045', '2055'])# ,'2060'])
    # ay.set_xticks(x)
    # ay.set_xticklabels(['2030', '2040', '2050'])  # ,'2060'])
    # ay.set_ylabel('kgCO2/kgH2')
    # ay.set_title('Carbon content')
    # Shrink current axis by 20%
    box = ax[0].get_position()
    ax[0].set_position([box.x0, box.y0, box.width * 0.9, box.height])
    box = ax[1].get_position()
    ax[1].set_position([box.x0-0.05, box.y0, box.width * 0.9, box.height])
    box = ax[2].get_position()
    ax[2].set_position([box.x0-0.1, box.y0, box.width * 0.9, box.height])
    # Put a legend to the right of the current axis
    ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # box = ay.get_position()
    # ay.set_position([box.x0, box.y0, box.width * 0.815, box.height*0.8])
    # Put a legend to the right of the current axis
    # ay.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(outputPath + '/Comparison energy.png')
    plt.show()

    fig,ax=plt.subplots(1,1,figsize=(4,2.5))

    scenarioColors=[col(1),col(5),col(9)]
    scenarioMarkers=['o','v','s']

    for k, l in enumerate(L):
    # add carbon emission
        l10 = list(dico_ener[l]['carbon'])
        print(l10)
        ax.plot(l10,marker=scenarioMarkers[k],color=scenarioColors[k], label=scenarioNames[k],zorder=2)

    plt.xticks(x,['2030-2040', '2040-2050', '2050-2060'])  # ,'2060'])
    plt.ylabel('kgCO$_2$/kgH$_2$')
    plt.grid(axis='y',alpha=0.5,zorder=1)
    # ay.set_title('Carbon content')
    # Shrink current axis by 20%
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend()
    plt.savefig(outputPath + '/Comparison carbon.png')
    plt.show()

    return

def plot_compare_energy_and_costs(dico_ener, scenarioNames, outputPath='Data/output/'):
    YEAR = list(list(dico_ener.items())[0][1].items())[0][1].index.values
    YEAR.sort()
    L = list(dico_ener.keys())

    for l in L:
        dico_ener[l] = dico_ener[l].loc[2030:2050]

    # fig=plt.figure(figsize=(8,6))
    # F = GridSpec(2,3,figure=fig)
    fig,ax=plt.subplots(1,3,sharey=True,figsize=(10,3))
    parameters={'axes.labelsize': 10,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
              'figure.titlesize': 12,
                'legend.fontsize':10}
    plt.rcParams.update(parameters)
    # ax1 = fig.add_subplot(F[0,0])
    # ax2 = fig.add_subplot(F[0,1],sharey=ax1)
    # ax3 = fig.add_subplot(F[0,2],sharey=ax1)
    # ax = [ax1, ax2, ax3]
    # ay = fig.add_subplot(F[1,:])
    width = 0.35
    col = plt.cm.tab20c
    labels = list(YEAR[1:])
    x = np.arange(len(labels))

    for k, l in enumerate(L):
        ax[k].grid(axis='y',alpha=0.5,color=col(19),zorder=1)

    for k, l in enumerate(L):
        # Create grey Bars
        l1 = list(dico_ener[l]['SMR w/o CCUS'] / 1000)
        ax[k].bar(x - width/2, l1, width, color=col(17),label="SMR w/o CCUS" if k == 2 else '',zorder=2)
        # Create blue Bars
        l2 = list((dico_ener[l]['SMR + CCUS 50%'] + dico_ener[l]['SMR + CCUS 90%']) / 1000)
        ax[k].bar(x - width/2 , l2, width, bottom=l1, color=col(0),label="SMR + CCUS" if k == 2 else '',zorder=2)
        # Create biogas Bars
        plt.rcParams['hatch.linewidth'] = 8
        plt.rcParams['hatch.color'] = col(3)
        l8 = list(dico_ener[l]['feedBiogas'] / 1000)
        ax[k].bar(x - width/2, l8, width, color='none', hatch='/',linewidth=0.5, edgecolor=col(3),alpha=0.8,label="Biomethane feed" if k == 2 else '',zorder=3)
        # Create green Bars
        l7 = list((dico_ener[l]['Alkaline electrolysis']+ dico_ener[l]['PEM electrolysis']) / 1000)
        ax[k].bar(x + width/2, l7, width, color=col(8),label="AEL grid feed" if k == 2 else '',zorder=2)
        # Create Local renewables bars
        l9 = list(dico_ener[l]['feedRE'] / 1000)
        ax[k].bar(x + width/2, l9, width,color=col(9),label="AEL local feed" if k == 2 else '',zorder=3)

    ax[0].set_ylabel('H$_2$ production (TWh/an)')
    for k,l in enumerate(L):
        ax[k].set_title(scenarioNames[k])
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(['2035', '2045', '2055'])# ,'2060'])
    # ay.set_xticks(x)
    # ay.set_xticklabels(['2030', '2040', '2050'])  # ,'2060'])
    # ay.set_ylabel('kgCO2/kgH2')
    # ay.set_title('Carbon content')
    # Shrink current axis by 20%
    box = ax[0].get_position()
    ax[0].set_position([box.x0, box.y0-0.025, box.width * 0.9, box.height])
    box = ax[1].get_position()
    ax[1].set_position([box.x0-0.05, box.y0-0.025, box.width * 0.9, box.height])
    box = ax[2].get_position()
    ax[2].set_position([box.x0-0.1, box.y0-0.025, box.width * 0.9, box.height])
    # Put a legend to the right of the current axis
    ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # box = ay.get_position()
    # ay.set_position([box.x0, box.y0, box.width * 0.815, box.height*0.8])
    # Put a legend to the right of the current axis
    # ay.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(outputPath + '/Comparison energy.png')
    plt.show()

    fig,ax=plt.subplots(1,1,figsize=(5,2.5))
    parameters={'axes.labelsize': 10,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
              'figure.titlesize': 12,
                'legend.fontsize':10}
    plt.rcParams.update(parameters)
    scenarioColors=[col(1),col(5),col(9)]
    scenarioMarkers=['o','v','s']

    for k, l in enumerate(L):
    # add carbon emission
        l10 = list(dico_ener[l]['costs'])
        print(l10)
        ax.plot(l10,marker=scenarioMarkers[k],color=scenarioColors[k], label=scenarioNames[k],zorder=2)

    plt.xticks(x,['2030-2040', '2040-2050', '2050-2060'])  # ,'2060'])
    plt.ylabel('€/kgH$_2$')
    plt.grid(axis='y', alpha=0.5,zorder=1)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.68, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.legend()
    plt.savefig(outputPath + '/Comparison costs.png')
    plt.show()

    return

def plot_annual_co2_emissions(scenarios, labels, legend_title=None, outputPath='Data/output/'):

    col = plt.cm.tab20c
    colBis = plt.cm.tab20b
    dico_color={'Ref':(colBis,0),'EnR':(colBis,1),'Grid':(colBis,2),'BM_':(col,0),'woSMR_':(colBis,16),'Only':(colBis,19),'CO2_':(colBis,8),'Re_':(col,8),'Man':(col,10)}
    dico_mark= {'Ref':'o','EnR':'o','Grid':'o','BM_': 'D', 'woSMR_':'s','Only':'s', 'CO2_':'v', 'Re_':'^','Man':'^'}
    colNumber=[]
    markNumber=[]
    variable=[]
    n=0
    for l,s in enumerate(scenarios):
        for var in list(dico_color.keys()):
            if var in s:
                variable.append(var)
                if variable[l-1]==variable[l]:
                    n=n+1
                else :
                    n=0
                colNumber.append((dico_color[var][0],dico_color[var][1]+n))
                markNumber.append(dico_mark[var])

    fig, ax = plt.subplots(2, 1,figsize=(7,6))
    for k, scenario in enumerate(scenarios):

        outputFolder = outputPath + scenario + '_PACA'

        co2_emissions = pd.read_csv(outputFolder + '/carbon_Pvar.csv').groupby('YEAR_op').sum().drop(columns=['TIMESTAMP', 'Unnamed: 0']).sort_index()

        year = co2_emissions.index.unique().values
        dy = year[1] - year[0]
        y0 = year[0]
        year_ex = np.append(year, [year[-1] + dy])

        r = 0.04
        def actualisationFactor(r, y):
            return (1 + r) ** (-(y + dy / 2 - y0))

        actualisation=pd.DataFrame(columns=['YEAR_op','actualisationFactor'])
        actualisation['YEAR_op']=year
        actualisation['actualisationFactor']=[actualisationFactor(r, yr) for yr in year]
        actualisation=actualisation.set_index('YEAR_op')

        capacityC=pd.read_csv(outputFolder + '/capacityCosts_Pvar.csv').groupby('YEAR_op').sum().drop(columns=['Unnamed: 0']).sort_index()
        variableC=pd.read_csv(outputFolder + '/powerCosts_Pvar.csv').groupby('YEAR_op').sum().drop(columns=['Unnamed: 0']).sort_index()
        carbonC=pd.read_csv(outputFolder + '/carbonCosts_Pvar.csv').groupby('YEAR_op').sum().drop(columns=['Unnamed: 0']).sort_index()
        turpeC=pd.read_csv(outputFolder + '/turpeCosts_Pvar.csv').groupby('YEAR_op').sum().drop(columns=['Unnamed: 0']).sort_index()
        storageC=pd.read_csv(outputFolder + '/storageCosts_Pvar.csv').groupby('YEAR_op').sum().drop(columns=['Unnamed: 0']).sort_index()
        importC=pd.read_csv(outputFolder + '/importCosts_Pvar.csv').groupby('YEAR_op').sum().drop(columns=['Unnamed: 0']).sort_index()

        costs =(capacityC['capacityCosts_Pvar']+variableC['powerCosts_Pvar']+carbonC['carbonCosts_Pvar']+turpeC['turpeCosts_Pvar']+storageC['storageCosts_Pvar']+importC['importCosts_Pvar'])*actualisation['actualisationFactor']

        # h2_prod = pd.read_csv(outputFolder + '/energy_Pvar.csv').groupby(['YEAR_op', 'RESOURCES']).sum().drop(
        #     columns=['TIMESTAMP', 'Unnamed: 0']).loc[(slice(None), 'hydrogen'), 'energy_Pvar'].reset_index().set_index(
        #     'YEAR_op').drop(columns='RESOURCES')

        # co2_content = co2_emissions / (h2_prod / 33)
        co2_total_emissions = co2_emissions.cumsum() * 10
        costs_total=costs.cumsum() *10

        # # Annual CO2 emissions
        # ax[0].plot(year + dy / 2, co2_emissions.values / 1e6, '-s', label=labels[k], color=cm(k * 4))
        # ax[0].legend(title=legend_title)
        # ax[0].set_ylabel('ktCO$_2$/yr')
        # ax[0].set_ylim([0, 1200])
        # ax[0].set_xticks(year + dy / 2)
        # ax[0].set_xticklabels(['{}-{}'.format(year[ny], year[ny] + (dy - 1)) for ny in range(len(year))])
        # ax[0].set_xlim([year[0], year[-1] + dy])
        # ax[0].grid(axis='y',alpha=0.5)

        # Total Costs
        ax[0].plot(year_ex[1:], costs_total.values / 1e9, marker=markNumber[k], label=labels[k], color=colNumber[k][0](colNumber[k][1]),zorder=2)
        ax[0].set_ylabel('bn €')
        # ax[0].set_ylim([0, 24])
        ax[0].set_xticks(year_ex[1:])
        ax[0].set_xlim([year[1], year[-1] + dy])

        # Total CO2 emissions
        ax[1].plot(year_ex[1:], co2_total_emissions.values / 1e9,marker=markNumber[k], label=labels[k], color=colNumber[k][0](colNumber[k][1]),zorder=2)
        ax[1].set_ylabel('MtCO$_2$')
        # ax[1].set_ylim([0, 24])
        ax[1].set_xticks(year_ex[1:])
        ax[1].set_xlim([year[1], year[-1] + dy])

    # Shrink current axis by 20%
    box = ax[0].get_position()
    ax[0].set_position([box.x0, box.y0, box.width * 0.65, box.height])
    # Put a legend to the right of the current axis
    ax[0].legend(loc='center left', bbox_to_anchor=(1.05, 0.3))
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.65, box.height])

    ax[0].yaxis.grid(alpha=0.5, zorder=1)
    ax[1].yaxis.grid(alpha=0.5, zorder=1)
    plt.savefig(outputPath+'/Cumul costs and carbon.png')
    plt.show()


    fig, ax = plt.subplots(figsize=(7,6))
    for k, scenario in enumerate(scenarios):

        outputFolder = outputPath + scenario + '_PACA'

        co2_emissions = pd.read_csv(outputFolder + '/carbon_Pvar.csv').groupby('YEAR_op').sum().drop(columns=['TIMESTAMP', 'Unnamed: 0']).sort_index()

        year = co2_emissions.index.unique().values
        dy = year[1] - year[0]
        y0 = year[0]
        year_ex = np.append(year, [year[-1] + dy])

        r = 0.04
        def actualisationFactor(r, y):
            return (1 + r) ** (-(y + dy / 2 - y0))

        actualisation=pd.DataFrame(columns=['YEAR_op','actualisationFactor'])
        actualisation['YEAR_op']=year
        actualisation['actualisationFactor']=[actualisationFactor(r, yr) for yr in year]
        actualisation=actualisation.set_index('YEAR_op')

        capacityC=pd.read_csv(outputFolder + '/capacityCosts_Pvar.csv').groupby('YEAR_op').sum().drop(columns=['Unnamed: 0']).sort_index()
        variableC=pd.read_csv(outputFolder + '/powerCosts_Pvar.csv').groupby('YEAR_op').sum().drop(columns=['Unnamed: 0']).sort_index()
        carbonC=pd.read_csv(outputFolder + '/carbonCosts_Pvar.csv').groupby('YEAR_op').sum().drop(columns=['Unnamed: 0']).sort_index()
        turpeC=pd.read_csv(outputFolder + '/turpeCosts_Pvar.csv').groupby('YEAR_op').sum().drop(columns=['Unnamed: 0']).sort_index()
        storageC=pd.read_csv(outputFolder + '/storageCosts_Pvar.csv').groupby('YEAR_op').sum().drop(columns=['Unnamed: 0']).sort_index()
        importC=pd.read_csv(outputFolder + '/importCosts_Pvar.csv').groupby('YEAR_op').sum().drop(columns=['Unnamed: 0']).sort_index()

        h2_prod = pd.read_csv(outputFolder + '/energy_Pvar.csv').groupby(['YEAR_op', 'RESOURCES']).sum().drop(columns=['TIMESTAMP', 'Unnamed: 0']).loc[(slice(None), 'hydrogen'), 'energy_Pvar'].reset_index().set_index('YEAR_op').drop(columns='RESOURCES')

        costs =(capacityC['capacityCosts_Pvar']+variableC['powerCosts_Pvar']+carbonC['carbonCosts_Pvar']+turpeC['turpeCosts_Pvar']+storageC['storageCosts_Pvar']+importC['importCosts_Pvar'])/actualisation['actualisationFactor']/(h2_prod['energy_Pvar']*30)


        # co2_content = co2_emissions / (h2_prod / 33)
        co2_total_emissions = co2_emissions.cumsum() * 10
        costs_total=costs.cumsum() *10

        # Total Costs
        ax.plot(year_ex[1:], costs_total.values / 1e9, marker=markNumber[k], label=labels[k], color=colNumber[k][0](colNumber[k][1]),zorder=2)
        ax.set_ylabel('bn €')
        # ax[0].set_ylim([0, 24])


    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.3))
    plt.grid(alpha=0.5, zorder=1)

    plt.savefig(outputPath+'/Cumul costs and carbon.png')
    plt.show()



    return

def plot_total_co2_emissions_and_costs(dico_costs, labels, legend_title=None, outputPath='Data/output/'):

    col = plt.cm.tab20c
    colBis = plt.cm.tab20b
    # dico_color={'Ref':(col,8),'EnR':(col,5),'Grid':(col,6),'BM_':(col,4),'woSMR_':(colBis,16),'Only':(colBis,12),'CO2_':(colBis,8),'Re_':(col,0),'Cav':(col,12)}
    # dico_mark= {'Ref':'d','EnR':'o','Grid':'o','BM_': 'o', 'woSMR_':'s','Only':'s', 'CO2_':'v', 'Re_':'^','Cav':'p'}
    #
    dico_color={'Re_inf':(col,8),'Caverns':(colBis,16),'CavernRE':(col,0)}
    dico_mark= {'Re_inf':'d', 'Caverns':'s', 'CavernRE':'^'}

    colNumber=[]
    markNumber=[]
    variable=[]
    n=0
    for l,s in enumerate(list(dico_costs.keys())):
        for var in list(dico_color.keys()):
            if var in s:
                variable.append(var)
                if variable[l-1]==variable[l]:
                    n=n+1
                else :
                    n=0
                colNumber.append((dico_color[var][0],dico_color[var][1]+n))
                markNumber.append(dico_mark[var])

    carbonCumul = {}
    meanPrice = {}
    horizonMean={}
    for s in list(dico_costs.keys()):
        meanPrice[s]=sum(dico_costs[s][k][['powerCosts','capacityCosts','capexElec','importElec','importGas','storageElec','storageH2','carbon','TURPE']].sum(axis=1) for k in list(dico_costs[s].keys()))/(sum((dico_costs[s][k]['Prod']*30) for k in list(dico_costs[s].keys())))
        horizonMean[s]=sum(dico_costs[s][k][['powerCosts','capacityCosts','capexElec','importElec','importGas','storageElec','storageH2','carbon','TURPE']].sum(axis=1) for k in list(dico_costs[s].keys())).sum()/(sum((dico_costs[s][k]['Prod']*30) for k in list(dico_costs[s].keys())).sum())
        carbon=pd.read_csv(outputPath+s+'_PACA/carbon_Pvar.csv').groupby('YEAR_op').sum().drop(columns=['TIMESTAMP','Unnamed: 0'])
        carbon=carbon.sort_index()
        carbonCumul[s]=(carbon['carbon_Pvar'].cumsum() * 10).sum()


    fig, ax = plt.subplots(figsize=(7,4))
    for k,s in enumerate(list(dico_costs.keys())):
        ax.plot(carbonCumul[s]/1e9,horizonMean[s], linestyle='',marker=markNumber[k],markersize=12, label=labels[k], color=colNumber[k][0](colNumber[k][1]),zorder=2)


    ax.set_ylabel('Mean price from 2020 to 2060 (€/kgH$_2$)')
    ax.set_xlabel('Emission from 2020 to 2060 (MtCO$_2$)')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0+0.05, box.width * 0.6, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.grid(axis='y',alpha=0.5, zorder=1)

    plt.savefig(outputPath+'/Cumul carbon and costs.png')
    plt.show()



    return

def plot_total_co2_emissions_and_flexSMR(dico_costs,scenarioNames,labels, legend_title=None, outputPath='Data/output/'):

    YEAR = list(list(dico_costs.items())[0][1].items())[0][1].index.values
    YEAR.sort()
    x=['2030-2040','2040-2050','2050-2060']

    col = plt.cm.tab20c
    colBis = plt.cm.tab20b
    dico_color={'Re_inf':(col,8),'Caverns':(colBis,16),'CavernRE':(col,0)}
    dico_mark= {'Re_inf':'d', 'Caverns':'s', 'CavernRE':'^'}
    colNumber=[]
    markNumber=[]
    variable=[]
    n=0
    for l,s in enumerate(scenarioNames):
        for var in list(dico_color.keys()):
            if var in s:
                variable.append(var)
                if l>0:
                    if variable[l-1]==variable[l]:
                        n=n+1
                    else :
                        n=0
                colNumber.append((dico_color[var][0],dico_color[var][1]+n))
                markNumber.append(dico_mark[var])

    carbonCumul = {}
    carbonYear = {}
    flexSMR = {}
    for s in scenarioNames:
        carbon = pd.read_csv(outputPath + s + '_PACA/carbon_Pvar.csv').groupby('YEAR_op').sum().drop(columns=['TIMESTAMP', 'Unnamed: 0'])
        carbon=carbon.sort_index()
        carbonYear[s]=carbon*10
        carbonCumul[s]=(carbon['carbon_Pvar'].cumsum() * 10).sum()
        cost1=(sum(dico_costs[s][k][['powerCosts','capacityCosts','capexElec','importElec','importGas','storageElec','storageH2','carbon','TURPE']]for k in dico_costs[s].keys()).sum(axis=1)/(sum(dico_costs[s][k]['Prod']*30 for k in dico_costs[s].keys()))).fillna(0)
        cost2=(sum(dico_costs[s+'_woSMR'][k][['powerCosts','capacityCosts','capexElec','importElec','importGas','storageElec','storageH2','carbon','TURPE']] for k in dico_costs[s].keys()).sum(axis=1)/(sum(dico_costs[s+'_woSMR'][k]['Prod']*30 for k in dico_costs[s].keys()))).fillna(0)
        flexSMR[s]=cost2-cost1


    fig, ax = plt.subplots(figsize=(7,4))
    for k,s in enumerate(scenarioNames):
        ax.plot(carbonCumul[s]/1e9,flexSMR[s].mean(), linestyle='',marker=markNumber[k],markersize=12, label=labels[k], color=colNumber[k][0](colNumber[k][1]),zorder=2)

    ax.set_ylabel('SMR flexibility value \nfrom 2020 to 2060 (€/kgH$_2$)')
    ax.set_xlabel('Emission from 2020 to 2060 (MtCO$_2$)')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0+0.05, box.width * 0.6, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.grid(axis='y',alpha=0.5, zorder=1)

    plt.savefig(outputPath+'/Cumul carbon and flex.png')
    plt.show()

    parameters={'axes.labelsize': 11,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
              'figure.titlesize': 15,
                'legend.fontsize':10}
    plt.rcParams.update(parameters)

    fig, ax = plt.subplots(1,3,figsize=(7.5,4), sharey=True,sharex=True)
    for l,yr in enumerate(YEAR[1:]):
        for k,s in enumerate(scenarioNames):
            ax[l].plot(carbonYear[s].loc[yr,'carbon_Pvar']/1e9,flexSMR[s].loc[yr], linestyle='',marker=markNumber[k],markersize=12, label=labels[k], color=colNumber[k][0](colNumber[k][1]),zorder=2)
        ax[l].set_title(x[l])
        # Shrink current axis by 20%
        box = ax[l].get_position()
        ax[l].set_position([box.x0-(l*box.width*0.3), box.y0+0.01, box.width * 0.8, box.height])
        ax[l].yaxis.grid(alpha=0.5, zorder=1)
        # plt.grid(axis='y', alpha=0.5, zorder=1)

    ax[0].set_ylabel('SMR flexibility value (€/kgH$_2$)')
    ax[1].set_xlabel('Emissions for the period (MtCO$_2$)')
    # Put a legend to the right of the current axis
    ax[2].legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.savefig(outputPath+'/Carbon and flex.png')
    plt.show()

    return flexSMR

