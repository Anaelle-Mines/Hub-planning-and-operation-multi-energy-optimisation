import os
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
os.sys.path.append(r'../')
from Data.Raw import tech_eco_data

inputPath='../Data/Raw/'

nHours = 8760
t = np.arange(1,nHours + 1)

zones = ['Fr']

yearZero = 2010
yearFinal = 2050
yearStep = 10
yearList = [yr for yr in range(yearZero, yearFinal+yearStep, yearStep)] # +1 to include the final year
nYears=len(yearList)


def interpolate(dic,year):
    years=list(dic.keys())
    val=list(dic.values())
    fill_ub=val[-1]
    fill_lb=val[0]
    return float(interp1d(years,val,fill_value=(fill_lb,fill_ub),bounds_error=False)(year))


scenarioFr = {}

elec_demand=pd.read_csv(inputPath+'areaConsumption2020_Fr_TIMExRES.csv').set_index(['TIMESTAMP','RESOURCES'])
anualElec={2020:492e6,2030:562.4e6,2040:619.8e6,2050:659.2e6}
hourlyH2={2020:0,2030:1825,2040:2400,2050:3710}
scenarioFr['resourceDemand'] =  pd.concat(
    (
        pd.DataFrame(data = {
              'YEAR': year,
              'TIMESTAMP': t, # We add the TIMESTAMP so that it can be used as an index later.
              'electricity': np.array(elec_demand.loc[(slice(None),'electricity'),'areaConsumption']*interpolate(anualElec,year)/anualElec[2020]), # incrising demand of electricity (hypothesis : ADEME)
              'hydrogen': interpolate(hourlyH2,year)*np.ones(nHours), # base-load consumption of H2 (hypothesis : ADEME)
              'gaz': np.zeros(nHours),
              'uranium': np.zeros(nHours)
             }
        ) for k, year in enumerate(yearList[1:])
    )
) 

scenarioFr['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    max_install_capacity = {2010:100000,2040:100000}
    min_install_capacity={2010:0,2040:0}
    max_cumul_capacity= {2010:0,2020:5200,2030:20900,2040:45000}
    min_cumul_capacity = {2010:0,2020:5200,2030:20900,2040:45000}
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':interpolate(min_install_capacity,year),'maxInstallCapacity': interpolate(max_install_capacity,year),
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': interpolate(min_cumul_capacity,year),'maxCumulCapacity': interpolate(max_cumul_capacity,year)}
            }
         )
    )

    tech = "WindOnShore"
    max_install_capacity = {2010:100000,2040:100000}
    min_install_capacity={2010:0,2040:0}
    max_cumul_capacity= {2010:8000,2020: 33200 ,2030: 47200, 2040:58000}
    min_cumul_capacity = {2010:8000,2020: 33200 ,2030: 47200,2040: 58000}
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2) 
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech: 
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex, 
                'minInstallCapacity': interpolate(min_install_capacity,year),'maxInstallCapacity': interpolate(max_install_capacity,year),
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': interpolate(min_cumul_capacity,year),'maxCumulCapacity': interpolate(max_cumul_capacity,year)}
            }
         )
    )

    tech = "Solar"
    max_install_capacity = {2010:100000,2040:100000}
    min_install_capacity={2010:0,2040:0}
    max_cumul_capacity={2010:4000,2020:35100,2030:79600,2040:118000}
    min_cumul_capacity ={2010:4000,2020:35100,2030:79600,2040:118000}
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2) 
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': interpolate(min_install_capacity,year),'maxInstallCapacity': interpolate(max_install_capacity,year),
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': interpolate(min_cumul_capacity,year),'maxCumulCapacity': interpolate(max_cumul_capacity,year)}
            }
         )
    )

    tech = "HydroReservoir"
    max_install_capacity = {2010:100000,2040:100000}
    min_install_capacity={2010:0,2040:0}
    max_cumul_capacity={2010:15000,2020:15000,2030:16000,2040:17000}
    min_cumul_capacity = {2010:15000,2020:15000,2030:16000,2040:17000}
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':interpolate(min_install_capacity,year),'maxInstallCapacity': interpolate(max_install_capacity,year),
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'EnergyNbhourCap': 2100, # used for hydroelectricity
                'minCumulCapacity': interpolate(min_cumul_capacity,year),'maxCumulCapacity': interpolate(max_cumul_capacity,year)}
            }
         )
    )

    tech = "HydroRiver"
    max_install_capacityacity = {2010:100000,2040:100000}
    min_install_capacity={2010:0,2040:0}
    max_cumul_capacity= {2010:11000,2020:11000,2030:12000,2040:13000}
    min_cumul_capacity = {2010:11000,2020:11000,2030:12000,2040:13000}
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':interpolate(min_install_capacity,year),'maxInstallCapacity': interpolate(max_install_capacity,year),
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': interpolate(min_cumul_capacity,year),'maxCumulCapacity': interpolate(max_cumul_capacity,year)}
            }
         )
    )

    tech = "OldNuke"
    max_install_capacity = {2010:100000,2040:100000}
    min_install_capacity={2010:0,2040:0}
    max_cumul_capacity= {2010:63100,2020:59400,2030:43100,2040:15500}
    min_cumul_capacity = {2010:63100,2020:59400,2030:43100,2040:15500}
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost':30, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':interpolate(min_install_capacity,year),'maxInstallCapacity': interpolate(max_install_capacity,year),
                'EmissionCO2': 0, 'Conversion': {'electricity': 1, 'uranium':-3.03},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': interpolate(min_cumul_capacity,year),'maxCumulCapacity': interpolate(max_cumul_capacity,year),'RampConstraintPlus':0.04,'RampConstraintMoins':0.04}
            }
         )
    )

    tech = "NewNuke"
    max_install_capacity = {2010:100000,2040:100000}
    min_install_capacity={2010:0,2040:0}
    max_cumul_capacity= {2010:0,2020:0,2030:6600,2040:13200}
    min_cumul_capacity = {2010:0,2020:0,2030:6600,2040:13200}
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':interpolate(min_install_capacity,year),'maxInstallCapacity': interpolate(max_install_capacity,year),
                'EmissionCO2': 0, 'Conversion': {'electricity': 1, 'uranium':-3.03},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': interpolate(min_cumul_capacity,year),'maxCumulCapacity': interpolate(max_cumul_capacity,year),'RampConstraintPlus':0.04,'RampConstraintMoins':0.04}
            }
         )
    )

    tech = "Coal_p"
    max_install_capacity = {2010:100000,2040:100000}
    min_install_capacity={2010:0,2040:0}
    max_cumul_capacity=  {2010:6000,2020:1000,2030:0,2040:0}
    min_cumul_capacity = {2010:6000,2020:1000,2030:0,2040:0}
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 50, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':interpolate(min_install_capacity,year),'maxInstallCapacity': interpolate(max_install_capacity,year),
                'EmissionCO2': 1000, 'Conversion': {'electricity': 1},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': interpolate(min_cumul_capacity,year),'maxCumulCapacity': interpolate(max_cumul_capacity,year),'RampConstraintPlus':0.06,'RampConstraintMoins':0.06}
            }
         )
    )

    tech = "TAC"
    max_install_capacity = {2010:100000,2040:100000}
    min_install_capacity={2010:0,2040:0}
    max_cumul_capacity= {2010:7100,2020:6500,2030:0,2040:0}
    min_cumul_capacity = {2010:7100,2020:6500,2030:0,2040:0}
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost':0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':interpolate(min_install_capacity,year),'maxInstallCapacity': interpolate(max_install_capacity,year),
                'EmissionCO2': 0, 'Conversion': {'electricity': 1, 'gaz':-2.7},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': interpolate(min_cumul_capacity,year),'maxCumulCapacity': interpolate(max_cumul_capacity,year) }
            }
         )
    )

    tech = "TAC_H2"
    max_install_capacity = {2010:100000,2040:100000}
    min_install_capacity={2010:0,2040:0}
    max_cumul_capacity= {2010:100000,2040:100000}
    min_cumul_capacity = {2010:0,2040:0}
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost':0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':interpolate(min_install_capacity,year),'maxInstallCapacity': interpolate(max_install_capacity,year),
                'EmissionCO2': 0, 'Conversion': {'electricity': 1, 'hydrogen':-2.7},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': interpolate(min_cumul_capacity,year),'maxCumulCapacity': interpolate(max_cumul_capacity,year) }
            }
         )
    )

    tech = "CCG"
    max_install_capacity = {2010:100000,2040:100000}
    min_install_capacity={2010:0,2040:0}
    max_cumul_capacity= {2010:5000,2040:5000}
    min_cumul_capacity ={2010:5000,2020:0,2040:0}
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':interpolate(min_install_capacity,year),'maxInstallCapacity': interpolate(max_install_capacity,year),
                'EmissionCO2': 0, 'Conversion': {'electricity': 1, 'gaz':-1.72},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': interpolate(min_cumul_capacity,year),'maxCumulCapacity': interpolate(max_cumul_capacity,year),'RampConstraintPlus':0.06,'RampConstraintMoins':0.06 }
            }
         )
    )

    tech = "CCG_H2"
    max_install_capacity = {2010:100000,2040:100000}
    min_install_capacity={2010:0,2040:0}
    max_cumul_capacity= {2010:100000,2040:100000}
    min_cumul_capacity ={2010:0,2040:0}
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':interpolate(min_install_capacity,year),'maxInstallCapacity': interpolate(max_install_capacity,year),
                'EmissionCO2': 0, 'Conversion': {'electricity': 1, 'hydrogen':-1.72},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': interpolate(min_cumul_capacity,year),'maxCumulCapacity': interpolate(max_cumul_capacity,year),'RampConstraintPlus':0.06,'RampConstraintMoins':0.06 }
            }
         )
    )

    tech = "electrolysis_AEL"
    max_cumul_capacity= {2010:0,2020:6500,2030:7500,2040:13400}
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Hydrogen production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.54},
                'minCumulCapacity': 0,'maxCumulCapacity': interpolate(max_cumul_capacity,year)}
            }
         )
    )

    tech = "IntercoIn"
    max_install_capacity = {2010:11000,2020:22300,2030:29700,2040:39400}
    min_install_capacity={2010:0,2040:0}
    max_cumul_capacity= {2010:11000,2020:22300,2030:29700,2040:39400}
    min_cumul_capacity ={2010:0,2040:0}
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 150, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':interpolate(min_install_capacity,year),'maxInstallCapacity': interpolate(max_install_capacity,year),
                'EmissionCO2': 290, 'Conversion': {'electricity': 1},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': interpolate(min_cumul_capacity,year),'maxCumulCapacity': interpolate(max_cumul_capacity,year)}
            }
         )
    )

    tech = "IntercoOut"
    max_install_capacity = {2010:0,2040:0}
    min_install_capacity={2010:0,2040:0}
    max_cumul_capacity= {2010:11000,2020:22300,2030:29700,2040:39400}
    min_cumul_capacity ={2010:0,2040:0}
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': -50, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':interpolate(min_install_capacity,year),'maxInstallCapacity': interpolate(max_install_capacity,year),
                'EmissionCO2': 0, 'Conversion': {'electricity': -1},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': interpolate(min_cumul_capacity,year),'maxCumulCapacity': interpolate(max_cumul_capacity,year) }
            }
         )
    )

    tech = "curtailment"
    max_install_capacity = {2010:100000,2040:100000}
    min_install_capacity={2010:0,2040:0}
    max_cumul_capacity= {2010:100000,2040:100000}
    min_cumul_capacity ={2010:0,2040:0}
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':interpolate(min_install_capacity,year),'maxInstallCapacity': interpolate(max_install_capacity,year),
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': interpolate(min_cumul_capacity,year),'maxCumulCapacity': interpolate(max_cumul_capacity,year)}
            }
         )
    )

scenarioFr['conversionTechs'] =  pd.concat(scenarioFr['conversionTechs'], axis=1)

scenarioFr['storageTechs'] = []
for k, year in enumerate(yearList[:-1]):
    tech = "Battery"
    max_install_capacity = {2010:0,2020:5000,2030:10000,2040:77000}
    max_install_power={2010:0,2020:500,2030:1000,2040:7700}
    capex1, opex1, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech + ' - 1h', hyp='ref', year=year+yearStep/2)
    capex4, opex4, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech + ' - 4h', hyp='ref', year=year+yearStep/2)
    capex_per_kWh = (capex4 - capex1) / 3
    capex_per_kW = capex1 - capex_per_kWh

    scenarioFr['storageTechs'].append(
        pd.DataFrame(data={tech: 
                { 'YEAR': year, 'resource': 'electricity',
                'storagelifeSpan': lifespan, 
                'storagePowerCost': capex_per_kW,
                'storageEnergyCost': capex_per_kWh,
                'storageOperationCost': opex1, # TODO: according to RTE OPEX seems to vary with energy rather than power
                'p_max': interpolate(max_install_power,year),
                'c_max': interpolate(max_install_capacity,year),
                'chargeFactors': {'electricity': 0.9200},
                'dischargeFactors': {'electricity': 1.09},
                'dissipation': 0.0085,
                }, 
            }
         )
    )

    # tech = "STEP"
    # max_install_capacity = {2010:30000,2020:10000,2030:20000,2040:20000}
    # max_install_power={2010:3000,2020:1000,2030:2000,2040:2000}
    # capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    # scenarioFr['storageTechs'].append(
    #     pd.DataFrame(data={tech:
    #             { 'YEAR': year,
    #            'resource': 'electricity',
    #            'storagelifeSpan': lifespan,
    #             'storagePowerCost': capex*0.5,
    #             'storageEnergyCost': capex*0.5/10,
    #             'storageOperationCost': opex,
    #             'p_max': interpolate(max_install_power,year),
    #             'c_max': interpolate(max_install_capacity,year),
    #             'chargeFactors': {'electricity': 0.9},
    #             'dischargeFactors': {'electricity': 1.11},
    #             'dissipation': 0,
    #             },
    #         }
    #      )
    # )

    tech = "saltCavernH2_G"
    max_install_capacity = {2010:100000,2040:100000}
    max_install_power={2010:100000,2040:100000}
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFr['storageTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year,
               'resource': 'hydrogen',
               'storagelifeSpan': lifespan,
                'storagePowerCost': 1000,
                'storageEnergyCost': capex,
                'storageOperationCost': opex,
                'p_max': interpolate(max_install_power,year),
                'c_max': interpolate(max_install_capacity,year),
                'chargeFactors': {'electricity': 0.0168,'hydrogen':1},
                'dischargeFactors': {'hydrogen': 1},
                'dissipation': 0,
                },
            }
         )
    )

scenarioFr['storageTechs'] =  pd.concat(scenarioFr['storageTechs'], axis=1)

scenarioFr['carbonTax'] = pd.DataFrame(data=np.linspace(0.0675,0.165, nYears),
    index=yearList, columns=('carbonTax',))

scenarioFr['carbonGoals'] = pd.DataFrame(data=np.linspace(974e6, 205e6, nYears),
    index=yearList, columns=('carbonGoals',))


scenarioFr['turpeFactorsHTB']=pd.DataFrame(columns=['HORAIRE','fixeTurpeHTB'],data={'P':5880,'HPH':5640,'HCH':5640,'HPE':5280,'HCE':4920}.items()).set_index('HORAIRE') # en €/MW/an part abonnement


scenarioFr['gridConnection'] = pd.read_csv(inputPath+'CalendrierHTB_TIME.csv', sep=',', decimal='.', skiprows=0,
                                comment="#").set_index(["TIMESTAMP"])


scenarioFr['maxImportCap'] = pd.concat(
    (
        pd.DataFrame(index=[year],data={
            'electricity': 0,
            'gazNat': 10e10,
            'gazBio': 310e6,
            'hydrogen': 0,
            'gaz': 0,
            'uranium':10e10
        }) for k, year in enumerate(yearList[1:])
    )
)

scenarioFr['maxExportCap'] = pd.concat(
    (
        pd.DataFrame(index=[year],data={
            'electricity': 0,
            'gazNat': 0,
            'gazBio': 0,
            'hydrogen': 0,
            'gaz': 0,
            'uranium':0
        }) for k, year in enumerate(yearList[1:])
    )
)

scenarioFr['economicParameters'] = pd.DataFrame({
    'discountRate':[0.04], 
    'financeRate': [0.04]
    }
)

df_res_ref = pd.read_csv(inputPath+'set2020_horaire_TIMExRES.csv',
    sep=',', decimal='.', skiprows=0,comment="#").set_index(["TIMESTAMP",'RESOURCES'])

gasPriceFactor = {2020:1,2050:2} # First term : factor for 2020 (price same as 2019) and second term : multiplicative factor in 2050 compare to 2019 prices
biogasPrice = {2020:120,2030: 110,2040: 100,2050: 90}
scenarioFr["resourceImportPrices"] = pd.concat(
    (
        pd.DataFrame(
            data={
                "YEAR": year,
                "TIMESTAMP": t,
                "electricity": 100000 * np.ones(nHours),
                "gazNat": df_res_ref.loc[(slice(None), "gazNat"), "importCost"].values
                *interpolate(gasPriceFactor,year),
                "gazBio": interpolate(biogasPrice,year) * np.ones(nHours),
                "hydrogen": 100000 * np.ones(nHours),
                "gaz": 100000 * np.ones(nHours),
                "uranium":3.3*np.ones(nHours)
            }
        )
        for k, year in enumerate(yearList[1:])
    )
)

scenarioFr['resourceImportCO2eq'] = pd.concat(
    (
        pd.DataFrame(data={
            'YEAR': year, 
            'TIMESTAMP': t, 
            'electricity': 0*np.ones(nHours),
            'gaz': max(0, 0.03 * (1 - (year - yearZero)/(2050 - yearZero))) * 29 / 13.1 + 203.5  * (1 - tech_eco_data.get_biogas_share_in_network_RTE(year)), # Taking 100 yr GWP of methane and 3% losses due to upstream leaks. Losses drop to zero in 2050.
            'gazNat': max(0, 0.03 * (1 - (year - yearZero)/(2050 - yearZero))) * 29 / 13.1 + 203.5  * (1 - tech_eco_data.get_biogas_share_in_network_RTE(year)), # Taking 100 yr GWP of methane and 3% losses due to upstream leaks. Losses drop to zero in 2050.
            'gazBio': max(0, 0.03 * (1 - (year - yearZero)/(2050 - yearZero))) * 29 / 13.1,
            'uranium': 0 * np.ones(nHours),
            'hydrogen': max(0, 0.05  - .03 * (year - yearZero)/(2050 - yearZero)) * 11 / 33, # Taking 100 yr GWP of H2 and 5% losses due to upstream leaks. Leaks fall to 2% in 2050 See: https://www.energypolicy.columbia.edu/research/commentary/hydrogen-leakage-potential-risk-hydrogen-economy
        }) for k, year in enumerate(yearList[1:])
    )
)

scenarioFr['convTechList'] = ["WindOnShore", "WindOffShore", "Solar", "CCG", "TAC","Coal_p", "OldNuke","NewNuke","IntercoIn","IntercoOut","curtailment","HydroReservoir","HydroRiver","CCG_H2","TAC_H2","electrolysis_AEL"]
ctechs = scenarioFr['convTechList']
availabilityFactor = pd.read_csv(inputPath+'availabilityFactor2020_Fr_TIMExTECH.csv',sep=',', decimal='.', skiprows=0).set_index(["TIMESTAMP", "TECHNOLOGIES"])
techs=list(availabilityFactor.index.get_level_values('TECHNOLOGIES').unique())

scenarioFr['availability']=[]
for year in yearList[1:]:
    for tech in techs:
        scenarioFr['availability'].append(
            pd.DataFrame(data={
                'YEAR':year,
                'availabilityFactor': availabilityFactor.loc[(slice(None), tech),'availabilityFactor']
                }
                ).set_index('YEAR', append=True)
        )

scenarioFr['availability']=pd.concat(scenarioFr['availability'], axis=0).reorder_levels(['YEAR','TIMESTAMP','TECHNOLOGIES'])

itechs = scenarioFr['availability'].index.isin(ctechs, level=2)
scenarioFr["availability"] = scenarioFr["availability"].loc[(slice(None),slice(None), itechs)]


scenarioFr["yearList"] = yearList
scenarioFr["transitionFactors"] =pd.DataFrame(
    {'TECHNO1':[],
    'TECHNO2':[],
    'TransFactor': 1}).set_index(['TECHNO1','TECHNO2'])

# pd.set_option('display.max_columns', 500)
# print(scenarioFr.keys())
# print(scenarioFr['conversionTechs'])
