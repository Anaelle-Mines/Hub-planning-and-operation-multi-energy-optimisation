import os
os.sys.path.append(r'../')
import numpy as np
import pandas as pd
import csv
import copy
import matplotlib.pyplot as plt
import seaborn as sb
from Functions.f_multiResourceModels import loadScenario


def plot_costs(df,outputFolder='../Data/output/',comparaison=False):

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
        s[i] = [0 if np.isnan(item) else item for item in s[i]]
        s[i] = [0 if item==np.inf else item for item in s[i]]
        s[i] = [0 if item==-np.inf else item for item in s[i]]
        print (B[i],'=',s[i])
        maxi.append(np.max(s[i]))

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
    plt.xticks(x, ['2020-2030','2030-2040','2040-2050','2050-2060'])

    ax.set_ylim([0,np.max(maxi)+1])
    ax.set_title("Hydrogen production costs")
    plt.grid(axis='y',alpha=0.5,zorder=1)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(outputFolder+'/H2 costs.png')
    plt.savefig('../Plots'+'/H2 costs.png')
    plt.show()

    return

def plot_capacity(outputFolder='../Data/output/',LoadFac=False):
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


    # add Load factors
    if LoadFac==True:
        for i,y in enumerate(YEAR):
            if capa.loc[(y,'electrolysis_AEL'),'capacity_Pvar'] > 100:
                ax[0].text((x + width/2)[i], l7[i]/2, str(round(H2_loadFactor[y]['electrolysis_AEL']*100)) +'%',ha='center')
            if capa.loc[(y,'SMR'),'capacity_Pvar'] > 100:
                ax[0].text((x - width / 2)[i], l1[i] / 2, str(round(H2_loadFactor[y]['SMR'] * 100)) + '%',ha='center')
            if capa.loc[(y,'SMR + CCS1'),'capacity_Pvar'] > 100:
                ax[0].text((x - width / 2)[i], l1[i]+l2[i] / 2, str(round(H2_loadFactor[y]['SMR + CCS1'] * 100)) + '%',ha='center')
            if capa.loc[(y, 'Solar'), 'capacity_Pvar'] > 10:
                ax[1].text((x)[i], l8[i] / 2, str(round(EnR_loadFactor[y]['Solar'] * 100)) + '%', ha='center')
            if capa.loc[(y,'Solar'),'capacity_Pvar'] > 100:
                ax[1].text((x)[i], l8[i]/2, str(round(EnR_loadFactor[y]['Solar'] * 100)) + '%', ha='center')
            if capa.loc[(y,'WindOnShore'),'capacity_Pvar'] > 100:
                ax[1].text((x)[i], l8[i]+l9[i]/2, str(round(EnR_loadFactor[y]['WindOnShore'] * 100)) + '%', ha='center')
            if capa.loc[(y,'WindOffShore_flot'),'capacity_Pvar'] > 100:
                ax[1].text((x)[i], l8[i]+l9[i]+l10[i]/2, str(round(EnR_loadFactor[y]['WindOffShore_flot'] * 100)) + '%', ha='center')

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

def plot_energy(outputFolder='../Data/output/'):
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
    # Create light green Bars
    l7 = list(df['Alkaline electrolysis']+ df['PEM electrolysis'])
    ax.bar(x + width / 2, l7, width, color=col(8), label="AEL grid feed",zorder=2)
    # Create dark green bar
    l9=list(df_feedRE)
    ax.bar(x + width / 2,l9,width,color=col(9),label="AEL local feed",zorder=3)

    plt.grid(axis='y',alpha=0.5,zorder=1)
    ax.set_ylabel('H2 production (TWh/yr)')
    # ax.set_title("Use of assets")
    plt.xticks(x,['2020-2030', '2030-2040', '2040-2050', '2050-2060'])
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

def plot_compare_energy_and_carbon(dico_ener, scenarioNames, outputPath='../Data/output/'):
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

def  plot_costs2050(df,outputFolder='../Data/output/',comparaison=False):

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
        g[i]=list(df[B[i]]['storageElec']/(df[B[i]]['Prod']*30))
        plt.bar(x + X[i], g[i], width,   bottom=[i + j + k + l + m + n +o for i, j, k, l, m, n, o in zip(a[i],aa[i],b[i],c[i],d[i],e[i],f[i])], color=col(5),label="Elec storage capa" if i==0 else "",zorder=2)

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

def plot_carbonCosts(dico,scenarioNames,outputPath='../Data/output/'):

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

def plot_H2Mean2050(scenario,outputFolder='../Data/output/'):

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