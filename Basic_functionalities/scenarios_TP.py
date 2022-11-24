import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

def getTechnoPrice(tech,year):
    #capex en €/kW et opex en €/kW/an
    years=[2020,2030,2040,2050]
    capex={
        'WindOnShore': interp1d(years, [1300, 710, 620, 530],fill_value=(1300,530),bounds_error=False),
        'Solar': interp1d(years, [747, 557, 497, 427],fill_value=(747,427),bounds_error=False),
        'Boiler_elec': interp1d(years, [1000, 1000, 1000, 1000],fill_value=(1000,1000),bounds_error=False),
        'Boiler_gas': interp1d(years, [1000, 1000, 1000, 1000],fill_value=(1000,1000),bounds_error=False),
        'PAC': interp1d(years, [1000, 1000, 1000, 1000],fill_value=(1000,1000),bounds_error=False),
    }
    opex={
        'WindOnShore': interp1d(years, [40, 22, 18, 16],fill_value=(40,16),bounds_error=False),
        'Solar': interp1d(years,  [11, 9, 8, 7],fill_value=(11,7),bounds_error=False),
        'Boiler_elec': interp1d(years,  [10, 10, 10, 10],fill_value=(10,10),bounds_error=False),
        'Boiler_gas': interp1d(years,  [10, 10, 10, 10],fill_value=(10,10),bounds_error=False),
        'PAC': interp1d(years,  [10, 10, 10, 10],fill_value=(10,10),bounds_error=False),
    }
    return [capex[tech](year)*1,opex[tech](year)*1]

def Scenario_Heat(year,inputPath='Data/Raw_TP/'):

    nHours = 8760
    t = np.arange(1, nHours + 1)

    scenario = {}

    demand=pd.read_csv(inputPath+'heatDemand_TIME.csv').set_index(['TIMESTAMP'])

    scenario['resourceDemand'] =pd.DataFrame(data = {
                'TIMESTAMP': t, # We add the TIMESTAMP so that it can be used as an index later.
                'heat': np.array(demand['areaConsumption']), # incrising demand of electricity (hypothesis : ADEME)
                'electricity': np.zeros(nHours),
                'gas': np.zeros(nHours),
            })

    scenario['conversionTechs'] = []

    tech = "WindOnShore"
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech:
                                { 'Category': 'Electricity production',
                                 'lifeSpan': 30, 'powerCost': 0, 'investCost': getTechnoPrice(tech,year)[0], 'operationCost': getTechnoPrice(tech,year)[1],
                                 'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                                 'EnergyNbhourCap': 0, # used for hydroelectricity
                                 'minCapacity':0 ,'maxCapacity':1000 }
                            }
                         )
        )


    tech = "Solar"
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech:
                               {'Category': 'Electricity production',
                                'lifeSpan': 20, 'powerCost': 0, 'investCost': getTechnoPrice(tech, year)[0],
                                'operationCost': getTechnoPrice(tech, year)[1],
                                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'minCapacity': 0, 'maxCapacity': 1000}
                           }
                     )
    )

    tech = "Boiler_elec"
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech:
                               {'Category': 'Heat production',
                                'lifeSpan': 50, 'powerCost': 0, 'investCost': getTechnoPrice(tech, year)[0],
                                'operationCost': getTechnoPrice(tech, year)[1],
                                'EmissionCO2': 0, 'Conversion': {'heat': 1,'electricity':-1},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'minCapacity': 0, 'maxCapacity': 1000}
                           }
                     )
    )

    tech = "Boiler_gas"
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech:
                               {'Category': 'Heat production',
                                'lifeSpan': 50, 'powerCost': 0, 'investCost': getTechnoPrice(tech, year)[0],
                                'operationCost': getTechnoPrice(tech, year)[1],
                                'EmissionCO2': 0, 'Conversion': {'heat': 1,'gas':-1},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'minCapacity': 0, 'maxCapacity': 1000}
                           }
                     )
    )

    tech = "PAC"
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech:
                               {'Category': 'Heat production',
                                'lifeSpan': 20, 'powerCost': 0, 'investCost': getTechnoPrice(tech, year)[0],
                                'operationCost': getTechnoPrice(tech, year)[1],
                                'EmissionCO2': 0, 'Conversion': {'heat': 1,'electricity':-0.5},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'minCapacity': 0, 'maxCapacity': 1000}
                           }
                     )
    )

    tech = "curtailment"
    scenario['conversionTechs'].append(
        pd.DataFrame(data={tech:
                               {'Category': 'Heat production',
                                'lifeSpan': 100, 'powerCost': 3000, 'investCost': 0,
                                'operationCost': 0,
                                'EmissionCO2': 0, 'Conversion': {'heat': 1},
                                'EnergyNbhourCap': 0,  # used for hydroelectricity
                                'minCapacity': 0, 'maxCapacity': 1000}
                           }
                     )
    )

    scenario['conversionTechs'] = pd.concat(scenario['conversionTechs'], axis=1)

    scenario['storageTechs'] = []

    tech = "Tank"
    scenario['storageTechs'].append(
        pd.DataFrame(data={tech:
                               { 'resource': 'heat',
                                 'storagelifeSpan': 30,
                                 'storagePowerCost': 1000,
                                 'storageEnergyCost':10000,
                                 'storageOperationCost': 10,
                                 'p_max': 1000,
                                 'c_max':10000,
                                 'chargeFactors': {'heat': 1},
                                 'dischargeFactors': {'heat': 1},
                                 'dissipation': 0.001,
                                 }
                           }
                     )
    )

    tech = "battery"
    scenario['storageTechs'].append(
        pd.DataFrame(data={tech:
                               { 'resource': 'electricity',
                                 'storagelifeSpan': 15,
                                 'storagePowerCost': 1000,
                                 'storageEnergyCost':10000,
                                 'storageOperationCost': 10,
                                 'p_max': 1000,
                                 'c_max':10000,
                                 'chargeFactors': {'electricity': 1},
                                 'dischargeFactors': {'electricity': 1},
                                 'dissipation': 0.0085,
                                 }
                           }
                     )
    )

    scenario['storageTechs'] = pd.concat(scenario['storageTechs'], axis=1)

    scenario['carbonTax'] = 0.13

    scenario['carbonGoals'] = 500

    scenario['maxBiogasCap'] = 1000

    scenario['gridConnection'] = pd.read_csv(inputPath+'CalendrierHPHC_TIME.csv', sep=',', decimal='.', skiprows=0,comment="#").set_index(["TIMESTAMP"])

    scenario['economicParameters'] = pd.DataFrame({
        'discountRate':[0.04],
        'financeRate': [0.04]
    }
    )

    df_res_ref = pd.read_csv(inputPath+'resPrice_YEARxTIMExRES.csv', sep=',', decimal='.', skiprows=0,comment="#").set_index(["YEAR", "TIMESTAMP",'RESOURCES'])
    gasBioPrice=interp1d([2020,2030,2040,2050], [150, 120, 100, 80],fill_value=(150,80),bounds_error=False)

    scenario['resourceImportPrices'] =pd.DataFrame(data={
        'TIMESTAMP': t,
        'electricity': df_res_ref.loc[(year, slice(None), 'electricity'),'importCost'].values,
        'gasNatural': df_res_ref.loc[(year, slice(None), 'gasNatural'),'importCost'].values,
        'gasBio': gasBioPrice(year)*1,
        'heat': 1000000,
        'gas': 1000000
    })

    scenario['resourceImportCO2eq'] =pd.DataFrame(data={
        'TIMESTAMP': t,
        'electricity': df_res_ref.loc[(year, slice(None), 'electricity'),'emission'].values,
        'gasNatural': max(0, 0.03 * 29 / 13.1 + 203.5), # Taking 100 yr GWP of methane and 3% losses due to upstream leaks. Losses drop to zero in 2050.
        'gasBio': max(0, 0.03 * 29 / 13.1),
        'heat': 0 * np.ones(nHours),
        'gas': max(0, 0.03 * 29 / 13.1 + 203.5) # Taking 100 yr GWP of methane and 3% losses due to upstream leaks. Losses drop to zero in 2050.
    })

    scenario['convTechList'] = ["WindOnShore",  "Solar", "PAC", 'Boiler_elec','Boiler_gas','curtailment']
    ctechs = scenario['convTechList']
    availabilityFactor = pd.read_csv(inputPath+'availabilityFactorTIMExTECH.csv',sep=',', decimal='.', skiprows=0).set_index([ "TIMESTAMP", "TECHNOLOGIES"])
    itechs = availabilityFactor.index.isin(ctechs,level='TECHNOLOGIES')
    scenario['availability'] = availabilityFactor.loc[(itechs,slice(None))]

    return scenario