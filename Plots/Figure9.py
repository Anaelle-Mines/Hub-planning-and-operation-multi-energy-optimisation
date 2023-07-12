#region Importation of modules
import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

#endregion

#First : execute ModelFrance.py

outputPath='../Data/output/'

ScenarioName='Ref'
outputFolderFr=outputPath+ScenarioName+'_Fr'

def plot_monotone(outputFolder='Data/output/'):

    marketPrice = pd.read_csv(outputFolder+'/marketPrice.csv').set_index(['YEAR_op', 'TIMESTAMP'])
    marketPrice['OldPrice_NonAct'].loc[marketPrice['OldPrice_NonAct'] > 400] = 400
    prices2013=pd.read_csv('../Data/Raw/electricity-grid-price-2013.csv').set_index('TIMESTAMP').fillna(0)

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

        plt.plot(MonotoneNew_Cumul, MonotoneNew_Price, color=col(k*4), label='Prices '+ str(yr))

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


plot_monotone(outputFolderFr)
