import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd

from Functions.f_graphicalTools import *

def PlotCapacityAndImportationCosts(Variables) :
    df = Variables['capacityCosts_var']
    df.drop(df.loc[df['capacityCosts_var'] < 1000000].index, inplace=True)
    df = pd.DataFrame([list(Variables['capacityCosts_var']['capacityCosts_var'] / 1000000)],
                      columns=list(Variables['capacityCosts_var']['TECHNOLOGIES']))
    fig1 = px.bar(df, title="Capacity costs", width=600)
    fig1 = fig1.update_layout(xaxis_title="Scenario", yaxis_title="M€")
    plotly.offline.plot(fig1, filename='Capacity costs.html')
    df = Variables['importCosts_var']
    df.drop(df.loc[df['importCosts_var'] < 1000000].index, inplace=True)
    df = pd.DataFrame([list(Variables['importCosts_var']['importCosts_var'] / 1000000)],
                      columns=list(Variables['importCosts_var']['RESSOURCES']))
    fig2 = px.bar(df, title="Importation costs", width=600)
    fig2 = fig2.update_layout(xaxis_title="Scenario", yaxis_title="M€")
    plotly.offline.plot(fig2, filename='Importation costs.html')
    return

def PlotCarbonCosts(Variables):
    df = Variables['carbonCosts_var']
    df.drop(df.loc[df['carbonCosts_var'] < 1].index, inplace=True)
    df = pd.DataFrame([list(Variables['carbonCosts_var']['carbonCosts_var'] / 1000000)],
                      columns=list(Variables['carbonCosts']['TECHNOLOGIES']))
    fig6 = px.bar(df, title="Emission costs", width=600)
    fig6 = fig6.update_layout(xaxis_title="Scenario", yaxis_title="M€")
    plotly.offline.plot(fig6, filename='Emissioncosts.html')
    return

def PlotCapacityAndEnegyProduction(Variables):
    df1 = Variables['capacity_var']
    df1.drop(df1.loc[df1['capacity_var'] < 0.1].index, inplace=True)
    df1 = df1.set_index('TECHNOLOGIES')
    df2 = Variables['power_var'].pivot(index="TIMESTAMP", columns='TECHNOLOGIES', values='power_var')
    df2 = df2.sum(axis=0)
    df2 = pd.DataFrame(df2, columns={'energy'})
    df2.drop(df2.loc[df2['energy'] < 100].index, inplace=True)
    df = pd.concat([df1, df2], axis=1)
    df = df.reset_index()
    df1 = pd.DataFrame([list(df['capacity_var'] / 1000)], columns=list(df['TECHNOLOGIES']))
    df2 = pd.DataFrame([list(df['energy'] / 1000000)], columns=list(df['TECHNOLOGIES']))
    fig3 = px.bar(df1, barmode='stack', title='Installed capacity')
    fig3 = fig3.update_layout(xaxis_title="Scenario", yaxis_title="GW", width=600)
    plotly.offline.plot(fig3, filename='Installed capacity.html')
    fig4 = px.bar(df2, barmode='stack', title='Produced energy', width=600)
    fig4 = fig4.update_layout(xaxis_title="Scenario", yaxis_title="TWh")
    plotly.offline.plot(fig4, filename='Produced energy.html')
    return

def PlotRessourceVariation(Variables,year='2013'):
    energy_variation = Variables['energy_var'].pivot(index="TIMESTAMP", columns='RESSOURCES', values='energy_var')
    TIMESTAMP_d = pd.date_range(start=str(year) + "-01-01 00:00:00", end=str(year) + "-12-31 23:00:00", freq="1H")
    energy_variation.index = TIMESTAMP_d;
    del energy_variation['uranium']
    fig5 = MyStackedPlotly(y_df=energy_variation)
    fig5 = fig5.update_layout(title_text="Variation par énergie (production nette + importation) (en MWh)",
                              xaxis_title="heures de l'année")
    plotly.offline.plot(fig5, filename='Energy variation.html')
    return

def PlotElectricityProduction(Variables,year='2013') :
    power_use = Variables['power_var'].pivot(index="TIMESTAMP", columns='TECHNOLOGIES', values='power_var')
    TIMESTAMP_d = pd.date_range(start=str(year) + "-01-01 00:00:00", end=str(year) + "-12-31 23:00:00", freq="1H")
    power_use.index = TIMESTAMP_d;
    del power_use['electrolysis']
    fig6 = MyStackedPlotly(y_df=power_use)
    fig6 = fig6.update_layout(title_text="Production électricité(en MW)",
                              xaxis_title="heures de l'année")
    plotly.offline.plot(fig6, filename='Power.html')
    return

def PlotH2Production(Variables,year='2013') :
    power_H2 = Variables['power_var'].pivot(index="TIMESTAMP", columns='TECHNOLOGIES', values='power_var')
    power_H2=power_H2['electrolysis']
    TIMESTAMP_d = pd.date_range(start=str(year) + "-01-01 00:00:00", end=str(year) + "-12-31 23:00:00", freq="1H")
    power_H2.index = TIMESTAMP_d;
    fig7 = px.area(power_H2)
    fig7 = fig7.update_layout(title_text="Production H2 (en MW)",
                              xaxis_title="heures de l'année")
    plotly.offline.plot(fig7, filename='ProductionH2.html')
    return

def PlotHeatmapAlpha(alpha_df,variation_prix_GazNat,variation_CAPEX_H2):
    alpha_matrice_df=alpha_df.pivot(values='value', index='PrixGaz', columns='Capex')
    fig = go.Figure(data=go.Heatmap(z=alpha_matrice_df, y=variation_prix_GazNat,x=variation_CAPEX_H2))
    fig.update_layout(title='Proportion NRJ PAC / NRJ TAC + CCG', yaxis_title='Prix du gaz €/MWh',xaxis_title='Variation en % des CAPEX électrolyseurs et PAC par rapport à la référence')
    plotly.offline.plot(fig, filename='Abaque aplha.html')
    return

def PlotScatterAlphe(alpha_df,variation_CAPEX_H2):
    alpha_matrice_df=alpha_df.pivot(values='value', index='PrixGaz', columns='Capex')
    alpha_matrice_df.columns=variation_CAPEX_H2
    fig1 = px.line(alpha_matrice_df)
    fig1.update_layout(title='Proportion NRJ PAC / NRJ TAC + CCG en fonction du prix du gaz pour différentes valeur de CAPEX H2',yaxis_title='Proportion NRJ PAC / NRJ TAC', xaxis_title='Prix du gaz (€/MWh')
    plotly.offline.plot(fig1, filename='Scatter aplha.html')
    return
