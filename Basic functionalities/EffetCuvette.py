#region importation modules
import os

import numpy as np
import pandas as pd
import csv
from Functions.f_AnalyseToolsAna import get_Clean_Prices

pd.options.display.width = 0
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import plotly
import plotly.express as px


#endregion

PCI_Massique_Hydrogene = 33.3	#kWh/kg H2
PCI_Volumique_Hydrogene = 2.99	#kWh/Nm3
Densite_hydrogene =	0.0899 #	kg/Nm3
Efficacite = 0.6
Cout_Energetique = PCI_Massique_Hydrogene / Efficacite # kWh Elec/kg H2

Facteur_Emission_kg = 0.1 #kgCO2/kg H2
Facteur_Emission_MWh = Facteur_Emission_kg / PCI_Massique_Hydrogene * 1000 #kgCO2/MWh
Facteur_Emission_Elec = 40 #kgCO2/MWh

DureeFonctionnement = 85000 # nombre heures

Market_Prices_year = get_Clean_Prices(year=2013,sorted=True)
Market_Prices_year.insert(0, 'NbHours', range(1, len(Market_Prices_year)+1))
Resultats = pd.DataFrame(None)
AverageShiftVals =[0]
HomotheticVals = [1/2,1,3/2,4/2]
AcutalisationVals= [0.04,0.06,0.08]
DureeVieMaxVals = [15] # [15,20,25]
CAPEXVals = [1000,600,200]
for Actualisation in AcutalisationVals:
    for DureeVieMax in DureeVieMaxVals:
        for CAPEX in CAPEXVals:
            for AverageShift in AverageShiftVals:
                for Homothetic in HomotheticVals:
                    Market_Prices_year_TMP=Market_Prices_year.copy()
                    Market_Prices_year_TMP.Prices=Market_Prices_year.Prices+AverageShift
                    Average= Market_Prices_year_TMP.Prices.mean()
                    Market_Prices_year_TMP.Prices=(Market_Prices_year_TMP.Prices-Average)*Homothetic+Average
                    #Actualisation = 0.05;
                    #DureeVieMax = 30 # ans
                    OPEX = CAPEX * 7 / 100  # €/kW/an
                    FacteurAnnuite = (1-1/(1+Actualisation)**DureeVieMax)/Actualisation
                    Market_Prices_year_TMP["CoutMarginal"]=Market_Prices_year_TMP.Prices.cumsum()/Market_Prices_year_TMP.NbHours # € / MWh Elec
                    Market_Prices_year_TMP["EnergieProduite"] = Market_Prices_year_TMP.NbHours * Efficacite
                    Market_Prices_year_TMP["CoutOperationFixe"] = OPEX / Market_Prices_year_TMP["EnergieProduite"] * 1000
                    Market_Prices_year_TMP["CoutInvFixe"] = CAPEX/FacteurAnnuite / Market_Prices_year_TMP["EnergieProduite"] * 1000
                    Market_Prices_year_TMP["CoutFixe"] = Market_Prices_year_TMP["CoutInvFixe"] + Market_Prices_year_TMP["CoutOperationFixe"]
                    Market_Prices_year_TMP["CoutTotal"] = Market_Prices_year_TMP["CoutFixe"] +Market_Prices_year_TMP["CoutMarginal"]
                    Market_Prices_year_TMP["Actualisation"] = Actualisation
                    Market_Prices_year_TMP["DureeVieMax"] = DureeVieMax
                    Market_Prices_year_TMP["CAPEX"] = CAPEX
                    Market_Prices_year_TMP["EcartTypePrixElec"] = Market_Prices_year_TMP.Prices.std().round(2)
                    Market_Prices_year_TMP["AverageShift"] = AverageShift
                    Resultats=pd.concat([Resultats,Market_Prices_year_TMP])


indexes = Resultats.CoutTotal<250
fig = px.line(Resultats[indexes], x="NbHours", y=["CoutTotal","CoutMarginal"], color='Actualisation',facet_col="EcartTypePrixElec",facet_row="CAPEX",
              labels={
                    "value": "Prix H2 [€/MWh]",
                    "NbHours" : "Nb d'heures de fonctionnement"})

plotly.offline.plot(fig, filename='tmp.html')


