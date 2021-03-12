import statsmodels.api as sm
import numpy as np
import pandas as pd

def regression(MyData):

    MyData.loc[:, "Capex"] = MyData["Capex"].astype(float)
    MyData.loc[:, "PrixGaz"] = MyData["PrixGaz"].astype(float)
    MyData.loc[:, "PrixGazCarre"] = MyData["PrixGaz"] ** 2
    MyData.loc[:, "CapexCarre"] = MyData["Capex"] ** 2
    MyData.loc[:, "PrixGazdemi"] = MyData["PrixGaz"] ** (1 / 2)
    MyData.loc[:, "Capexdemi"] = MyData["Capex"] ** (1 / 2)
    MyData.loc[:, "Cross"] = MyData["Capex"] * MyData["PrixGaz"]
    MyData.loc[:, "Ratio1"] = MyData["Capex"] / MyData["PrixGaz"]
    MyData.loc[:, "Ratio2"] = MyData["PrixGaz"] / MyData["Capex"]
    MyData.loc[:, "Ratio1Carre"] = (MyData["Capex"] / MyData["PrixGaz"]) ** 2
    MyData.loc[:, "Ratio3"] = MyData["Capex"] * MyData["Ratio1"]
    MyData.loc[:, "logCapex"] = np.log(MyData["Capex"])
    MyData.loc[:, "logPrixGaz"] = np.log(MyData["PrixGaz"])
    MyData.loc[:, "logCross"] = np.log(MyData["Capex"]) * np.log(MyData["PrixGaz"])
    MyData.loc[:, "logAlpha"] = -np.log(MyData["alpha"])

    Models = {"simple": ["PrixGaz", "Capex"],
              "RatioCapexPrixGaz": ["Ratio1"],
              "RatioCapexPrixGazCross": ["Ratio1", "Cross"],
              "RatioCapexPrixGazCarre": ["Ratio1Carre"],
              "RatioCapexPrixGazCarreSimple": ["Ratio1Carre", "Ratio1"],
              "RatioCapexPrixGazCarreSimpleCross": ["Ratio1Carre", "Ratio1", "Cross"],
              "Ratio2": ["Ratio2"],
              "Ratio2Cross": ["Ratio2", "Cross"],
              "Cross": ["PrixGaz", "Capex", "Cross"],
              "PrixGazcarre": ["PrixGaz", "Capex", "PrixGazCarre"],
              "PrixGazdemi": ["PrixGaz", "Capex", "PrixGazdemi"],
              "PrixCapexCarre": ["PrixGaz", "Capex", "CapexCarre"],
              "PrixCapexdemi": ["PrixGaz", "Capex", "Capexdemi"],
              "PrixCapexGazCarre": ["PrixGaz", "Capex", "CapexCarre", "PrixGazCarre"],
              "PrixCapexGazCarreCross": ["PrixGaz", "Capex", "CapexCarre", "PrixGazCarre", "Cross"],
              "CapexcarreCross": ["CapexCarre", "Cross"],
              "PrixGazCarreCross": ["PrixGazCarre", "Cross"],
              "CapexCarrePrixGazCarreCross": ["CapexCarre", "PrixGazCarre", "Cross"],
              "CapexRatio1Cross": ["Capex", "Ratio1", "Cross"],
              "CapexPrixGazRatio1Cross": ["Capex", "PrixGaz", "Ratio1", "Cross"],
              "CapexCarreRatio1Cross": ["CapexCarre", "Ratio1", "Cross"],
              "Ratio3Cross": ["Ratio3", "Cross"],
              "logSimple": ["logCapex", "logPrixGaz"],
              "logCross": ["logCapex", "logPrixGaz", "logCross"]
              }

    Predictions = {}
    Rdeux = {}
    Parameters = {}

    for model in Models.keys():
        My_model = sm.OLS(MyData["alpha"], sm.add_constant(MyData[Models[model]]))
        results = My_model.fit()
        Rdeux[model] = results.rsquared
        Predictions[model] = results.predict()
        Parameters[model] = results.params

    # log/log
    My_model = sm.OLS(MyData["logAlpha"], sm.add_constant(MyData[Models["logSimple"]]))
    results = My_model.fit()
    Rdeux['log_log'] = results.rsquared
    Predictions['log_log'] = results.predict()
    Parameters['log_log'] = results.params

    return Rdeux,Predictions,Parameters

def get_Clean_Prices(year=2013,sorted=True,InputFolder='Data/input/'):
    Market_Prices = pd.read_csv(InputFolder + 'EuropeanMarket_Prices_UTC_2007_2017.csv', sep=',', decimal='.',
                                skiprows=0)
    Market_Prices.Dates = pd.to_datetime(Market_Prices.Dates)
    Market_Prices.set_index(["Dates"], inplace=True)
    #### remove nan values at time t by using value at time t - 1
    Market_Prices_year = Market_Prices[Market_Prices.index.year == year]
    prectime = Market_Prices_year.index[0]
    for t in Market_Prices_year.index:
        if np.isnan(Market_Prices_year.loc[t, "Prices"]): Market_Prices_year.loc[t, "Prices"] = Market_Prices_year.loc[
            prectime, "Prices"]
        prectime = t
    if sorted: Market_Prices_year= Market_Prices_year.sort_values(by="Prices")
    return Market_Prices_year