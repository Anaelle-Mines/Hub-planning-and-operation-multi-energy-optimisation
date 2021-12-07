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

def MyStackedPlotly(y_df, Conso=-1,isModifyOrder=True,Names=-1):
    '''
    :param x:
    :param y:
    :param Names:
    :return:
    '''

    if isModifyOrder: y_df=ModifyOrder_df(y_df) ### set Nuke first column
    if (Names.__class__ == int): Names=y_df.columns.unique().tolist()
    x_df=y_df.index
    fig = go.Figure()
    i = 0
    for col in y_df.columns:
        if i == 0:
            fig.add_trace(go.Scatter(x=x_df, y=y_df[col], fill='tozeroy',
                                     mode='none', name=Names[i]))  # fill down to xaxis
            colNames = [col]
        else:
            colNames.append(col)
            fig.add_trace(go.Scatter(x=x_df, y=y_df.loc[:, y_df.columns.isin(colNames)].sum(axis=1), fill='tonexty',
                                     mode='none', name=Names[i]))  # fill to trace0 y
        i = i + 1

    if (Conso.__class__ != int):
        fig.add_trace(go.Scatter(x=Conso.index,
                                 y=Conso["areaConsumption"], name="Conso",
                                 line=dict(color='red', width=0.4)))  # fill down to xaxis
        if "NewConsumption" in Conso.keys():
            fig.add_trace(go.Scatter(x=Conso.index,
                                     y=Conso["NewConsumption"], name="Conso+stockage",
                                     line=dict(color='black', width=0.4)))  # fill down to xaxis

    fig.update_xaxes(rangeslider_visible=True)
    return(fig)

def MyPlotly(x_df,y_df,Names="",fill=True):
    '''
    :param x:
    :param y:
    :param Names:
    :return:
    '''
    if Names=="" : Names=y_df.columns.values.tolist()
    fig = go.Figure()
    i=0
    for col in y_df.columns:
        if i==0:
            if fill :
                fig.add_trace(go.Scatter(x=x_df, y=y_df[col] , fill='tozeroy',
                             mode='none' ,name=Names[i])) # fill down to xaxis
            else :
                fig.add_trace(go.Scatter(x=x_df, y=y_df[col],mode='lines', name=Names[i]))  # fill down to xaxis
            colNames=[col]
        else:
            colNames.append(col)
            if fill :
                fig.add_trace(go.Scatter(x=x_df, y=y_df[col], fill='tozeroy',
                                     mode='none', name=Names[i]))  # fill to trace0 y
            else :
                fig.add_trace(go.Scatter(x=x_df, y=y_df[col],
                                     mode='lines', name=Names[i]))  # fill to trace0 y
        i=i+1

    fig.update_xaxes(rangeslider_visible=True)
    return(fig)

def MyAreaStackedPlot_tidy(df,Selected_TECHNOLOGIES=-1,AREA_name="AREAS",TechName='TECHNOLOGIES'):
    if (Selected_TECHNOLOGIES==-1):
        Selected_TECHNOLOGIES=df[TechName].unique().tolist()
    AREAS=df[AREA_name].unique().tolist()

    visible={}
    for AREA in AREAS: visible[AREA] = []
    for AREA in AREAS:
        for AREA2 in AREAS:
            if AREA2==AREA:
                for TECH in Selected_TECHNOLOGIES:
                    visible[AREA2].append(True)
            else :
                for TECH in Selected_TECHNOLOGIES:
                    visible[AREA2].append(False)

    fig = go.Figure()
    dicts=[]
    for AREA in AREAS:
        production_df = df[df[AREA_name] == AREA].pivot(index="TIMESTAMP",columns='TECHNOLOGIES',values='energy')
        fig = AppendMyStackedPlotly(fig,x_df=production_df.index,
                            y_df=production_df[list(Selected_TECHNOLOGIES)],
                            Names=list(Selected_TECHNOLOGIES))
        dicts.append(dict(label=AREA,
             method="update",
             args=[{"visible": visible[AREA]},
                   {"title": AREA }]))

    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=list(dicts),
            )
        ])

    return(fig)

def ModifyOrder(Names):
    if "OldNuke" in Names:
        Names.remove("OldNuke")
        Names.insert(0, "OldNuke")
    if "NewNuke" in Names:
        Names.remove("NewNuke")
        Names.insert(0, "NewNuke")
    if "NukeCarrene" in Names:
        Names.remove("NukeCarrene")
        Names.insert(0, "NukeCarrene")

    return(Names)

def ModifyOrder_df(df):
    if "OldNuke" in df.columns:
        Nuke=df.pop("OldNuke")
        df.insert(0, "OldNuke", Nuke)
    if "NewNuke" in df.columns:
        Nuke=df.pop("NewNuke")
        df.insert(0, "NewNuke", Nuke)
    if "NukeCarrene" in df.columns:
        Nuke=df.pop("NukeCarrene")
        df.insert(0, "NukeCarrene", Nuke)
    return(df);
