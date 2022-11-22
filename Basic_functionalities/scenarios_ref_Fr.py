import numpy as np
import pandas as pd
from Basic_functionalities import tech_eco_data

inputPath='Data/Raw_Ana/'

nHours = 8760
t = np.arange(1,nHours + 1)

zones = ['Fr']

yearZero = 2010
yearFinal = 2050
yearStep = 10
yearList = [yr for yr in range(yearZero, yearFinal+yearStep, yearStep)] # +1 to include the final year
nYears = len(yearList)

scenarioFr = {}

elec_demand=pd.read_csv(inputPath+'areaConsumption2010-2050_Fr_TIMExRESxYEAR.csv').set_index(['YEAR','TIMESTAMP','RESOURCES'])
scenarioFr['resourceDemand'] =  pd.concat(
    (
        pd.DataFrame(data = { 
              'YEAR': year, 
              'TIMESTAMP': t, # We add the TIMESTAMP so that it can be used as an index later. 
              'electricity': np.array(elec_demand.loc[(year,slice(None),'electricity'),'areaConsumption']), # incrising demand of electricity (hypothesis : ADEME)
              'hydrogen': np.zeros(nHours),
              'gaz': np.zeros(nHours),
              'uranium': np.zeros(nHours)
             } 
        ) for k, year in enumerate(yearList[1:])
    ) 
) 

scenarioFr['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    max_install_capacity = [100000,100000,100000,100000]
    min_install_capacity=[0,0,0,0]
    max_cumul_capacity= [0,6000,15000,45000]
    min_cumul_capacity = [0,6000,15000,45000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year)
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': min_install_capacity[k],'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': min_cumul_capacity[k],'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "WindOnShore"
    max_install_capacity = [100000,100000,100000,100000]
    min_install_capacity=[0,0,0,0]
    max_cumul_capacity= [8000, 40000 , 52000, 70000]
    min_cumul_capacity = [8000, 40000 , 52000, 70000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year) 
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech: 
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex, 
                'minInstallCapacity': min_install_capacity[k],'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': min_cumul_capacity[k],'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "Solar"
    max_install_capacity = [100000,100000,100000,100000]
    min_install_capacity=[0,0,0,0]
    max_cumul_capacity=[4000,45000,55000,75000]
    min_cumul_capacity = [4000,45000,55000,75000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year) 
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': min_install_capacity[k],'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': min_cumul_capacity[k],'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "HydroReservoir"
    max_install_capacity = [100000,100000,100000,100000]
    min_install_capacity=[0,0,0,0]
    max_cumul_capacity=[15000,15000,15000,15000]
    min_cumul_capacity = [15000,15000,15000,15000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year)
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': min_install_capacity[k],'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'EnergyNbhourCap': 2100, # used for hydroelectricity
                'minCumulCapacity': min_cumul_capacity[k],'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "HydroRiver"
    max_install_capacityacity = [100000,100000,100000,100000]
    min_install_capacity=[0,0,0,0]
    max_cumul_capacity= [11000,11000,11000,11000]
    min_cumul_capacity = [11000,11000,11000,11000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year)
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': min_install_capacity[k],'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': min_cumul_capacity[k],'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "OldNuke"
    max_install_capacity = [100000,100000,100000,100000]
    min_install_capacity=[0,0,0,0]
    max_cumul_capacity= [63000,54000,45000,15000]
    min_cumul_capacity = [63000,54000,45000,15000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year)
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': min_install_capacity[k],'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1, 'uranium':-3.03},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': min_cumul_capacity[k],'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.04,'RampConstraintMoins':0.04}
            }
         )
    )

    tech = "NewNuke"
    max_install_capacity = [100000,100000,100000,100000]
    min_install_capacity=[0,0,0,0]
    max_cumul_capacity= [0,0,5000,13000]
    min_cumul_capacity = [0,0,5000,13000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year)
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': min_install_capacity[k],'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1, 'uranium':-3.03},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': min_cumul_capacity[k],'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.04,'RampConstraintMoins':0.04}
            }
         )
    )

    tech = "Coal_p"
    max_install_capacity = [100000,100000,100000,100000]
    min_install_capacity=[0,0,0,0]
    max_cumul_capacity= [10000,6000,0,0]
    min_cumul_capacity = [10000,6000,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year)
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 18, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': min_install_capacity[k],'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 1000, 'Conversion': {'electricity': 1},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': min_cumul_capacity[k],'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.06,'RampConstraintMoins':0.06}
            }
         )
    )

    tech = "TAC"
    max_install_capacity = [100000,100000,100000,100000]
    min_install_capacity=[0,0,0,0]
    max_cumul_capacity= [10000,10000,5000,0]
    min_cumul_capacity = [10000,10000,5000,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year)
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost':0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': min_install_capacity[k],'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1, 'gaz':-2.7},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': min_cumul_capacity[k],'maxCumulCapacity': max_cumul_capacity[k] }
            }
         )
    )

    tech = "CCG"
    max_install_capacity = [100000,100000,100000,100000]
    min_install_capacity=[0,0,0,0]
    max_cumul_capacity= [5000,7000,10000,17000]
    min_cumul_capacity = [5000,7000,10000,17000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year)
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': min_install_capacity[k],'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1, 'gaz':-1.72},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': min_cumul_capacity[k],'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.06,'RampConstraintMoins':0.06 }
            }
         )
    )

    tech = "Interco"
    max_install_capacity = [100000,100000,100000,100000]
    min_install_capacity=[0,0,0,0]
    max_cumul_capacity= [10000,13000,26000,39000]
    min_cumul_capacity = [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year)
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 15, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': min_install_capacity[k],'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 290, 'Conversion': {'electricity': 1},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': min_cumul_capacity[k],'maxCumulCapacity': max_cumul_capacity[k] }
            }
         )
    )

    tech = "curtailment"
    max_install_capacity = [100000,100000,100000,100000]
    min_install_capacity=[0,0,0,0]
    max_cumul_capacity= [100000,100000,100000,100000]
    min_cumul_capacity = [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year)
    scenarioFr['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': min_install_capacity[k],'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': min_cumul_capacity[k],'maxCumulCapacity': max_cumul_capacity[k] }
            }
         )
    )

scenarioFr['conversionTechs'] =  pd.concat(scenarioFr['conversionTechs'], axis=1)

scenarioFr['storageTechs'] = []
for k, year in enumerate(yearList[:-1]):
    tech = "Battery"
    max_install_capacity = [100000,100000,100000,100000]
    max_install_power=[10000,10000,10000,10000]
    capex1, opex1, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech + ' - 1h', hyp='ref', year=year)
    capex4, opex4, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech + ' - 4h', hyp='ref', year=year)
    capex_per_kWh = (capex4 - capex1) / 3
    capex_per_kW = capex1 - capex_per_kWh

    scenarioFr['storageTechs'].append(
        pd.DataFrame(data={tech: 
                { 'YEAR': year, 'resource': 'electricity',
                'storagelifeSpan': lifespan, 
                'storagePowerCost': capex_per_kW/10,
                'storageEnergyCost': capex_per_kWh/10,
                'storageOperationCost': opex1, # TODO: according to RTE OPEX seems to vary with energy rather than power
                'p_max': max_install_power[k],
                'c_max': max_install_capacity[k],
                'chargeFactors': {'electricity': 0.9200},
                'dischargeFactors': {'electricity': 1.09},
                'dissipation': 0.0085,
                }, 
            }
         )
    )

    tech = "STEP"
    max_install_capacity = [30000,20000,20000,20000]
    max_install_power=[3000,2000,2000,2000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year)
    scenarioFr['storageTechs'].append(
        pd.DataFrame(data={tech: 
                { 'YEAR': year, 
               'resource': 'electricity',
               'storagelifeSpan': lifespan,
                'storagePowerCost': capex,
                'storageEnergyCost': 0,
                'storageOperationCost': opex,
                'p_max': max_install_power[k],
                'c_max': max_install_capacity[k],
                'chargeFactors': {'electricity': 0.9},
                'dischargeFactors': {'electricity': 1.11},
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

scenarioFr['maxBiogasCap'] = pd.DataFrame(data=np.linspace(0, 310e6, nYears),
    index=yearList, columns=('maxBiogasCap',))

scenarioFr['gridConnection'] = pd.read_csv(inputPath+'CalendrierHPHC_TIME.csv', sep=',', decimal='.', skiprows=0,
                                comment="#").set_index(["TIMESTAMP"])

scenarioFr['economicParameters'] = pd.DataFrame({
    'discountRate':[0.04], 
    'financeRate': [0.04]
    }
)

df_res_ref = pd.read_csv(inputPath+'set2020-2050_horaire_TIMExRESxYEAR.csv',
    sep=',', decimal='.', skiprows=0,comment="#").set_index(["YEAR", "TIMESTAMP",'RESOURCES'])

scenarioFr['resourceImportPrices'] = pd.concat(
    (
        pd.DataFrame(data={
            'YEAR': year, 
            'TIMESTAMP': t, 
            'electricity': df_res_ref.loc[(year, slice(None), 'electricity'),'importCost'].values,
            'gazNat': 2 * df_res_ref.loc[(year, slice(None), 'gazNat'),'importCost'].values,
            'gazBio': df_res_ref.loc[(year, slice(None), 'gazBio'),'importCost'].values,
            'uranium': df_res_ref.loc[(year, slice(None), 'uranium'),'importCost'].values,
            'hydrogen': df_res_ref.loc[(year, slice(None), 'hydrogen'),'importCost'].values,
            'gaz': df_res_ref.loc[(year, slice(None), 'gaz'),'importCost'].values
        }) for k, year in enumerate(yearList[1:])
    )
)

scenarioFr['resourceImportCO2eq'] = pd.concat(
    (
        pd.DataFrame(data={
            'YEAR': year, 
            'TIMESTAMP': t, 
            'electricity': df_res_ref.loc[(year, slice(None), 'electricity'),'emission'].values,
            'gaz': max(0, 0.03 * (1 - (year - yearZero)/(2050 - yearZero))) * 29 / 13.1 + 203.5  * (1 - tech_eco_data.get_biogas_share_in_network_RTE(year)), # Taking 100 yr GWP of methane and 3% losses due to upstream leaks. Losses drop to zero in 2050.
            'gazNat': max(0, 0.03 * (1 - (year - yearZero)/(2050 - yearZero))) * 29 / 13.1 + 203.5  * (1 - tech_eco_data.get_biogas_share_in_network_RTE(year)), # Taking 100 yr GWP of methane and 3% losses due to upstream leaks. Losses drop to zero in 2050.
            'gazBio': max(0, 0.03 * (1 - (year - yearZero)/(2050 - yearZero))) * 29 / 13.1,
            'uranium': 0 * np.ones(nHours),
            'hydrogen': max(0, 0.05  - .03 * (year - yearZero)/(2050 - yearZero)) * 11 / 33, # Taking 100 yr GWP of H2 and 5% losses due to upstream leaks. Leaks fall to 2% in 2050 See: https://www.energypolicy.columbia.edu/research/commentary/hydrogen-leakage-potential-risk-hydrogen-economy
        }) for k, year in enumerate(yearList[1:])
    )
)

scenarioFr['convTechList'] = ["WindOnShore", "WindOffShore", "Solar", "CCG", "TAC","Coal_p", "OldNuke","NewNuke","Interco","curtailment","HydroReservoir","HydroRiver"]
ctechs = scenarioFr['convTechList']
availabilityFactor = pd.read_csv(inputPath+'availabilityFactor2010-2050_Fr_TIMExTECHxYEAR.csv',
                                 sep=',', decimal='.', skiprows=0).set_index(["YEAR", "TIMESTAMP", "TECHNOLOGIES"])
itechs = availabilityFactor.index.isin(ctechs, level=2)
scenarioFr['availability'] = availabilityFactor.loc[(slice(None), slice(None), itechs)]

scenarioFr["yearList"] = yearList
scenarioFr["transitionFactors"] =pd.DataFrame(
    {'TECHNO1':[],
    'TECHNO2':[],
    'TransFactor': 1}).set_index(['TECHNO1','TECHNO2'])