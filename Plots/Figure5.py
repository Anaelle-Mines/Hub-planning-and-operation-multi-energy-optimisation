#region Importation of modules
import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import sys
 
sys.path.append('C:/Users/anaelle.jodry/Documents/optim-multienergy_public')

from Functions.f_extract_data import extract_costs
from scenario_creation import scenarioDict
#endregion

#First : execute ModelPACA_ref.py

outputPath='Data/output/'

ScenarioName='Ref'
outputFolder=outputPath+ScenarioName
scenario=scenarioDict[ScenarioName]

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
    plt.show()

    return

df=extract_costs(scenario,outputFolder)

plot_costs(df,outputFolder)