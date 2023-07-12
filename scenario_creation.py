
import os

import numpy as np
import pandas as pd
import csv
import datetime
import copy
import time

from scenarios_ref_PACA import scenarioPACA
from Data.Raw import tech_eco_data

outputPath='Data/output/'
inputPath='Data/Raw/'
outputFolderFr='Data/output/Ref_Fr'

scenarioDict={'Ref':scenarioPACA}

nHours = 8760
t = np.arange(1,nHours + 1)


yearZero = 2010
yearFinal = 2050
yearStep = 10
yearList = [yr for yr in range(yearZero, yearFinal+yearStep, yearStep)] # +1 to include the final year
nYears = len(yearList)

#region Scenario EnR
scenarioEnR={k: v.copy() for (k, v) in scenarioPACA.items()}

scenarioEnR['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='low', year=year+yearStep/2)
    scenarioEnR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity':0,
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 0}
            }
         )
    )

    tech = "WindOffShore_flot"
    max_install_capacity = [0,1000,1000,1000]
    max_cumul_capacity= [0,1000,1500,2000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='low', year=year+yearStep/2)
    scenarioEnR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "WindOnShore"
    max_install_capacity = [0,200,200,200]
    max_cumul_capacity=[0,300,300,300]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='low', year=year+yearStep/2)
    scenarioEnR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "Solar"
    max_install_capacity = [0,200,200,200]
    max_cumul_capacity=[0,300,300,300]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='low', year=year+yearStep/2)
    scenarioEnR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR"
    max_install_capacity = [411,1000,1000,1000]
    min_install_capacity=[411,0,0,0]
    max_cumul_capacity= [411,10000,10000,10000]
    min_cumul_capacity =[411,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioEnR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Hydrogen production',
                'lifeSpan': lifespan, 'powerCost': 0.21, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': min_install_capacity[k],'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'hydrogen':1,'gaz':-1.28},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': min_cumul_capacity[k],'maxCumulCapacity': max_cumul_capacity[k] ,'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS1"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioEnR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 7.71, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': -150, 'Conversion': {'hydrogen': 1,'gaz':-1.32},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS2"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioEnR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 13.7, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': -270, 'Conversion': {'hydrogen': 1,'gaz':-1.45},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "CCS1"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioEnR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "CCS2"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioEnR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR_elec"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioEnR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.4},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR_elecCCS1"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioEnR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': -156, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.57},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "electrolysis_AEL"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioEnR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.54},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "electrolysis_PEMEL"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioEnR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.67},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "curtailment"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioEnR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 100000}
            }
         )
    )

scenarioEnR['conversionTechs'] = pd.concat(scenarioEnR['conversionTechs'], axis=1)

scenarioDict['EnR']=scenarioEnR

#endregion

#region Scenario conv_SmrOnly

scenarioSmrOnlyConv={k: v.copy() for (k, v) in scenarioPACA.items()}

scenarioSmrOnlyConv['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnlyConv['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity':0,
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 0}
            }
         )
    )

    tech = "WindOffShore_flot"
    max_install_capacity = [0,500,500,500]
    max_cumul_capacity= [0,500,750,1000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnlyConv['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "WindOnShore"
    max_install_capacity = [0,100,100,100]
    max_cumul_capacity= [0, 150 , 150, 150]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnlyConv['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "Solar"
    max_install_capacity = [0,100,100,100]
    max_cumul_capacity=[0,150,150,150]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnlyConv['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR"
    max_install_capacity = [411,10000,10000,10000]
    min_install_capacity=[411,0,0,0]
    max_cumul_capacity= [411,10000,10000,10000]
    min_cumul_capacity =[411,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnlyConv['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Hydrogen production',
                'lifeSpan': lifespan, 'powerCost': 0.21, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': min_install_capacity[k],'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'hydrogen':1,'gaz':-1.28},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': min_cumul_capacity[k],'maxCumulCapacity': max_cumul_capacity[k] ,'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS1"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnlyConv['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 7.71, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': -150, 'Conversion': {'hydrogen': 1,'gaz':-1.32},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS2"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnlyConv['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 13.7, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': -270, 'Conversion': {'hydrogen': 1,'gaz':-1.45},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "CCS1"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnlyConv['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "CCS2"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnlyConv['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR_elec"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnlyConv['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.4},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR_elecCCS1"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnlyConv['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': -156, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.57},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "electrolysis_AEL"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnlyConv['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.54},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "electrolysis_PEMEL"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnlyConv['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.67},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "curtailment"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnlyConv['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 100000}
            }
         )
    )

scenarioSmrOnlyConv['conversionTechs'] = pd.concat(scenarioSmrOnlyConv['conversionTechs'], axis=1)

scenarioSmrOnlyConv['carbonTax'] = pd.DataFrame(data=np.linspace(0.0675,0.0675, nYears),
    index=yearList, columns=('carbonTax',))

scenarioSmrOnlyConv['maxBiogasCap'] = pd.DataFrame(data=np.linspace(0, 0, nYears),
    index=yearList, columns=('maxBiogasCap',))

scenarioSmrOnlyConv["transitionFactors"] =pd.DataFrame(
    {'TECHNO1':[],
    'TECHNO2':[],
    'TransFactor':[] }).set_index(['TECHNO1','TECHNO2'])

scenarioDict['conv_SmrOnly']=scenarioSmrOnlyConv

#endregion

#region Scenario SmrOnly

scenarioSmrOnly={k: v.copy() for (k, v) in scenarioPACA.items()}

scenarioSmrOnly['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnly['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity':0,
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 0}
            }
         )
    )

    tech = "WindOffShore_flot"
    max_install_capacity = [0,500,500,500]
    max_cumul_capacity= [0,500,750,1000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnly['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "WindOnShore"
    max_install_capacity = [0,100,100,100]
    max_cumul_capacity= [0, 150 , 150, 150]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnly['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "Solar"
    max_install_capacity = [0,100,100,100]
    max_cumul_capacity=[0,150,150,150]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnly['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR"
    max_install_capacity = [411,10000,10000,10000]
    min_install_capacity=[411,0,0,0]
    max_cumul_capacity= [411,10000,10000,10000]
    min_cumul_capacity =[411,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnly['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Hydrogen production',
                'lifeSpan': lifespan, 'powerCost': 0.21, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': min_install_capacity[k],'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'hydrogen':1,'gaz':-1.28},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': min_cumul_capacity[k],'maxCumulCapacity': max_cumul_capacity[k] ,'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS1"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnly['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 7.71, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': -150, 'Conversion': {'hydrogen': 1,'gaz':-1.32},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS2"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnly['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 13.7, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': -270, 'Conversion': {'hydrogen': 1,'gaz':-1.45},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "CCS1"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnly['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "CCS2"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnly['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR_elec"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnly['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.4},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR_elecCCS1"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnly['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': -156, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.57},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "electrolysis_AEL"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnly['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.54},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "electrolysis_PEMEL"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnly['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.67},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "curtailment"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnly['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 100000}
            }
         )
    )

scenarioSmrOnly['conversionTechs'] = pd.concat(scenarioSmrOnly['conversionTechs'], axis=1)


scenarioDict['SmrOnly']=scenarioSmrOnly

#endregion

#region Scenario Grid
scenarioGrid={k: v.copy() for (k, v) in scenarioPACA.items()}

scenarioGrid['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='high', year=year+yearStep/2)
    scenarioGrid['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity':0,
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 0}
            }
         )
    )

    tech = "WindOffShore_flot"
    max_install_capacity = [0,1000,1000,1000]
    max_cumul_capacity= [0,500,750,1000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='high', year=year+yearStep/2)
    scenarioGrid['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "WindOnShore"
    max_install_capacity = [0,100,100,100]
    max_cumul_capacity=[0,150,150,150]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='high', year=year+yearStep/2)
    scenarioGrid['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "Solar"
    max_install_capacity = [0,100,100,100]
    max_cumul_capacity=[0,150,150,150]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='high', year=year+yearStep/2)
    scenarioGrid['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR"
    max_install_capacity = [411,1000,1000,1000]
    min_install_capacity=[411,0,0,0]
    max_cumul_capacity= [411,10000,10000,10000]
    min_cumul_capacity =[411,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioGrid['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Hydrogen production',
                'lifeSpan': lifespan, 'powerCost': 0.21, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': min_install_capacity[k],'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'hydrogen':1,'gaz':-1.28},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': min_cumul_capacity[k],'maxCumulCapacity': max_cumul_capacity[k] ,'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS1"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioGrid['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 7.71, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': -150, 'Conversion': {'hydrogen': 1,'gaz':-1.32},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS2"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioGrid['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 13.7, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': -270, 'Conversion': {'hydrogen': 1,'gaz':-1.45},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "CCS1"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioGrid['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "CCS2"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioGrid['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR_elec"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioGrid['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.4},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR_elecCCS1"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioGrid['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': -156, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.57},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "electrolysis_AEL"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioGrid['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.54},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "electrolysis_PEMEL"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioGrid['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.67},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "curtailment"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioGrid['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 100000}
            }
         )
    )

scenarioGrid['conversionTechs'] = pd.concat(scenarioGrid['conversionTechs'], axis=1)

scenarioDict['Grid']=scenarioGrid
#endregion

#region Scenario EnR_x2
scenarioREx2={k: v.copy() for (k, v) in scenarioPACA.items()}


scenarioREx2['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREx2['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity':0,
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 0}
            }
         )
    )

    tech = "WindOffShore_flot"
    max_install_capacity = [0,500,500,500]
    max_install_capacity = [round(i * 2, 0) for i in max_install_capacity]
    max_cumul_capacity= [0,500,750,1000]
    max_cumul_capacity = [round(i * 2, 0) for i in max_cumul_capacity]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREx2['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "WindOnShore"
    max_install_capacity = [0,100,100,100]
    max_install_capacity = [round(i * 2, 0) for i in max_install_capacity]
    max_cumul_capacity= [0, 150 , 150, 150]
    max_cumul_capacity = [round(i * 2, 0) for i in max_cumul_capacity]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2) 
    scenarioREx2['conversionTechs'].append(
        pd.DataFrame(data={tech: 
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex, 
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "Solar"
    max_install_capacity = [0,100,100,100]
    max_install_capacity = [round(i*2,0) for i in max_install_capacity]
    max_cumul_capacity=[0,150,150,150]
    max_cumul_capacity = [round(i * 2, 0) for i in max_cumul_capacity]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2) 
    scenarioREx2['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR"
    max_install_capacity = [411,10000,10000,10000]
    min_install_capacity=[411,0,0,0]
    max_cumul_capacity= [411,10000,10000,10000]
    min_cumul_capacity =[411,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREx2['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Hydrogen production',
                'lifeSpan': lifespan, 'powerCost': 0.21, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': min_install_capacity[k],'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'hydrogen':1,'gaz':-1.28},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': min_cumul_capacity[k],'maxCumulCapacity': max_cumul_capacity[k] ,'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS1"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREx2['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost':7.71, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': -150, 'Conversion': {'hydrogen': 1,'gaz':-1.32},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS2"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREx2['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 13.7, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': -270, 'Conversion': {'hydrogen': 1,'gaz':-1.45},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "CCS1"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREx2['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "CCS2"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREx2['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR_elec"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREx2['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.4},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR_elecCCS1"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREx2['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': -156, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.57},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "electrolysis_AEL"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREx2['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.54},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "electrolysis_PEMEL"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREx2['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.67},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "curtailment"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREx2['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 100000}
            }
         )
    )

scenarioREx2['conversionTechs'] = pd.concat(scenarioREx2['conversionTechs'], axis=1)

scenarioDict['Re_x2']=scenarioREx2

#endregion

#region Scenario EnR_inf
scenarioREinf={k: v.copy() for (k, v) in scenarioPACA.items()}

scenarioREinf['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity':0,
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 0}
            }
         )
    )

    tech = "WindOffShore_flot"
    max_install_capacity = [0, 10000, 10000, 10000]
    max_cumul_capacity = [0, 10000, 10000, 10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "WindOnShore"
    max_install_capacity = [0, 10000, 10000, 10000]
    max_cumul_capacity = [0, 10000, 10000, 10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2) 
    scenarioREinf['conversionTechs'].append(
        pd.DataFrame(data={tech: 
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex, 
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "Solar"
    max_install_capacity = [0, 10000, 10000, 10000]
    max_cumul_capacity = [0, 10000, 10000, 10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2) 
    scenarioREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR"
    max_install_capacity = [411,10000,10000,10000]
    min_install_capacity=[411,0,0,0]
    max_cumul_capacity= [411,10000,10000,10000]
    min_cumul_capacity =[411,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Hydrogen production',
                'lifeSpan': lifespan, 'powerCost': 0.21, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': min_install_capacity[k],'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'hydrogen':1,'gaz':-1.28},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': min_cumul_capacity[k],'maxCumulCapacity': max_cumul_capacity[k] ,'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS1"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost':7.71, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': -150, 'Conversion': {'hydrogen': 1,'gaz':-1.32},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS2"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 13.7, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': -270, 'Conversion': {'hydrogen': 1,'gaz':-1.45},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "CCS1"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "CCS2"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR_elec"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.4},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR_elecCCS1"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': -156, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.57},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "electrolysis_AEL"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.54},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "electrolysis_PEMEL"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.67},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "curtailment"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 100000}
            }
         )
    )

scenarioREinf['conversionTechs'] = pd.concat(scenarioREinf['conversionTechs'], axis=1)

scenarioDict['Re_inf']=scenarioREinf
#endregion

#region Scenario BM_80
scenarioBM80={k: v.copy() for (k, v) in scenarioPACA.items()}

df_res_ref = pd.read_csv(inputPath+'set2020-2050_horaire_TIMExRESxYEAR.csv',sep=',', decimal='.', skiprows=0,comment="#").set_index(["YEAR", "TIMESTAMP",'RESOURCES'])
df_elecPrice=pd.read_csv(outputFolderFr+'/marketPrice.csv').set_index(['YEAR_op','TIMESTAMP'])
df_elecCarbon=pd.read_csv(outputFolderFr+'/carbon.csv').set_index(['YEAR_op','TIMESTAMP'])

gasPriceFactor=[1,2,2,2]
bioGasPrice=[120,100,90,80]
scenarioBM80['resourceImportPrices'] = pd.concat(
    (
        pd.DataFrame(data={
            'YEAR': year,
            'TIMESTAMP': t,
            'electricity': df_elecPrice.loc[(year, slice(None)),'OldPrice_NonAct'].values,
            'gazNat': df_res_ref.loc[(year, slice(None), 'gazNat'),'importCost'].values*gasPriceFactor[k],
            'gazBio':bioGasPrice[k]*np.ones(nHours),
            'hydrogen': 100000*np.ones(nHours),
            'gaz': 100000*np.ones(nHours)
        }) for k, year in enumerate(yearList[1:])
    )
)


scenarioDict['BM_80']=scenarioBM80
#endregion

#region Scenario BM_100
scenarioBM100={k: v.copy() for (k, v) in scenarioPACA.items()}

df_res_ref = pd.read_csv(inputPath+'set2020-2050_horaire_TIMExRESxYEAR.csv',sep=',', decimal='.', skiprows=0,comment="#").set_index(["YEAR", "TIMESTAMP",'RESOURCES'])
df_elecPrice=pd.read_csv(outputFolderFr+'/marketPrice.csv').set_index(['YEAR_op','TIMESTAMP'])
df_elecCarbon=pd.read_csv(outputFolderFr+'/carbon.csv').set_index(['YEAR_op','TIMESTAMP'])

gasPriceFactor=[1,2,2,2]
bioGasPrice=[120,120,110,100]
scenarioBM100['resourceImportPrices'] = pd.concat(
    (
        pd.DataFrame(data={
            'YEAR': year,
            'TIMESTAMP': t,
            'electricity': df_elecPrice.loc[(year, slice(None)),'OldPrice_NonAct'].values,
            'gazNat': df_res_ref.loc[(year, slice(None), 'gazNat'),'importCost'].values*gasPriceFactor[k],
            'gazBio':bioGasPrice[k]*np.ones(nHours),
            'hydrogen': 100000*np.ones(nHours),
            'gaz': 100000*np.ones(nHours)
        }) for k, year in enumerate(yearList[1:])
    )
)

scenarioDict['BM_100']=scenarioBM100
#endregion

#region Scenario woSMR_2030
scenarioWoSMR2030={k: v.copy() for (k, v) in scenarioPACA.items()}

scenarioWoSMR2030['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2030['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity':0,
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 0}
            }
         )
    )

    tech = "WindOffShore_flot"
    max_install_capacity = [0,500,500,500]
    max_cumul_capacity= [0,500,750,1000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2030['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "WindOnShore"
    max_install_capacity = [0,100,100,100]
    max_cumul_capacity= [0, 150 , 150, 150]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2030['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "Solar"
    max_install_capacity = [0,100,100,100]
    max_cumul_capacity=[0,150,150,150]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2030['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR"
    max_install_capacity = [411,0,0,0]
    min_install_capacity=[411,0,0,0]
    max_cumul_capacity= [411,0,0,0]
    min_cumul_capacity =[411,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2030['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Hydrogen production',
                'lifeSpan': lifespan, 'powerCost': 0.21, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': min_install_capacity[k],'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'hydrogen':1,'gaz':-1.28},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': min_cumul_capacity[k],'maxCumulCapacity': max_cumul_capacity[k] ,'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS1"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2030['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 7.71, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': -150, 'Conversion': {'hydrogen': 1,'gaz':-1.32},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS2"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2030['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 13.7, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': -270, 'Conversion': {'hydrogen': 1,'gaz':-1.45},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "CCS1"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2030['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "CCS2"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2030['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR_elec"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2030['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.4},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR_elecCCS1"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2030['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': -156, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.57},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "electrolysis_AEL"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2030['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.54},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "electrolysis_PEMEL"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2030['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.67},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "curtailment"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2030['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 100000}
            }
         )
    )

scenarioWoSMR2030['conversionTechs'] = pd.concat(scenarioWoSMR2030['conversionTechs'], axis=1)

scenarioWoSMR2030["transitionFactors"] =pd.DataFrame(
    {'TECHNO1':[],
    'TECHNO2':[],
    'TransFactor':[] }).set_index(['TECHNO1','TECHNO2'])

scenarioDict['woSMR_2030']=scenarioWoSMR2030

#endregion

#region Scenario woSMR_2040
scenarioWoSMR2040={k: v.copy() for (k, v) in scenarioPACA.items()}

scenarioWoSMR2040['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2040['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity':0,
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 0}
            }
         )
    )

    tech = "WindOffShore_flot"
    max_install_capacity = [0,500,500,500]
    max_cumul_capacity= [0,500,750,1000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2040['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "WindOnShore"
    max_install_capacity = [0,100,100,100]
    max_cumul_capacity= [0, 150 , 150, 150]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2040['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "Solar"
    max_install_capacity = [0,100,100,100]
    max_cumul_capacity=[0,150,150,150]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2040['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR"
    max_install_capacity = [411,10000,0,0]
    min_install_capacity=[411,0,0,0]
    max_cumul_capacity= [411,10000,0,0]
    min_cumul_capacity =[411,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2040['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Hydrogen production',
                'lifeSpan': lifespan, 'powerCost': 0.21, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': min_install_capacity[k],'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'hydrogen':1,'gaz':-1.28},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': min_cumul_capacity[k],'maxCumulCapacity': max_cumul_capacity[k] ,'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS1"
    max_cumul_capacity= [0,10000,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2040['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 7.71, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': -150, 'Conversion': {'hydrogen': 1,'gaz':-1.32},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS2"
    max_cumul_capacity= [0,10000,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2040['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 13.7, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': -270, 'Conversion': {'hydrogen': 1,'gaz':-1.45},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "CCS1"
    max_cumul_capacity= [0,10000,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2040['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "CCS2"
    max_cumul_capacity= [0,10000,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2040['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR_elec"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2040['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.4},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR_elecCCS1"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2040['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': -156, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.57},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "electrolysis_AEL"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2040['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.54},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "electrolysis_PEMEL"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2040['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.67},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "curtailment"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2040['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 100000}
            }
         )
    )

scenarioWoSMR2040['conversionTechs'] = pd.concat(scenarioWoSMR2040['conversionTechs'], axis=1)

scenarioDict['woSMR_2040']=scenarioWoSMR2040
#endregion

#region Scenario woSMR_2050
scenarioWoSMR2050={k: v.copy() for (k, v) in scenarioPACA.items()}

scenarioWoSMR2050['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2050['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity':0,
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 0}
            }
         )
    )

    tech = "WindOffShore_flot"
    max_install_capacity = [0,500,500,500]
    max_cumul_capacity= [0,500,750,1000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2050['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "WindOnShore"
    max_install_capacity = [0,100,100,100]
    max_cumul_capacity= [0, 150 , 150, 150]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2050['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "Solar"
    max_install_capacity = [0,100,100,100]
    max_cumul_capacity=[0,150,150,150]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2050['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR"
    max_install_capacity = [411,10000,10000,0]
    min_install_capacity=[411,0,0,0]
    max_cumul_capacity= [411,10000,10000,0]
    min_cumul_capacity =[411,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2050['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Hydrogen production',
                'lifeSpan': lifespan, 'powerCost': 0.21, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': min_install_capacity[k],'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'hydrogen':1,'gaz':-1.28},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': min_cumul_capacity[k],'maxCumulCapacity': max_cumul_capacity[k] ,'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS1"
    max_cumul_capacity= [0,10000,10000,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2050['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 7.71, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': -150, 'Conversion': {'hydrogen': 1,'gaz':-1.32},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS2"
    max_cumul_capacity= [0,10000,10000,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2050['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 13.7, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': -270, 'Conversion': {'hydrogen': 1,'gaz':-1.45},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "CCS1"
    max_cumul_capacity= [0,10000,10000,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2050['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "CCS2"
    max_cumul_capacity= [0,10000,10000,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2050['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR_elec"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2050['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.4},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR_elecCCS1"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2050['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': -156, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.57},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "electrolysis_AEL"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2050['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.54},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "electrolysis_PEMEL"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2050['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.67},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "curtailment"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioWoSMR2050['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 100000}
            }
         )
    )

scenarioWoSMR2050['conversionTechs'] = pd.concat(scenarioWoSMR2050['conversionTechs'], axis=1)

scenarioDict['woSMR_2050']=scenarioWoSMR2050
#endregion

#region Scenario CO2_10
scenarioCO210={k: v.copy() for (k, v) in scenarioPACA.items()}

scenarioCO210['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO210['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity':0,
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 0}
            }
         )
    )

    tech = "WindOffShore_flot"
    max_install_capacity = [0,500,500,500]
    max_cumul_capacity= [0,500,750,1000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO210['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "WindOnShore"
    max_install_capacity = [0,100,100,100]
    max_cumul_capacity= [0, 150 , 150, 150]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO210['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "Solar"
    max_install_capacity = [0,100,100,100]
    max_cumul_capacity=[0,150,150,150]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO210['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR"
    max_install_capacity = [411,1000,1000,1000]
    min_install_capacity=[411,0,0,0]
    max_cumul_capacity= [411,1000,1000,1000]
    min_cumul_capacity =[411,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO210['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Hydrogen production',
                'lifeSpan': lifespan, 'powerCost': 0.21, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': min_install_capacity[k],'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'hydrogen':1,'gaz':-1.28},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': min_cumul_capacity[k],'maxCumulCapacity': max_cumul_capacity[k] ,'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS1"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO210['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 1.71, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': -150, 'Conversion': {'hydrogen': 1,'gaz':-1.32},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS2"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO210['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 2.9, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': -270, 'Conversion': {'hydrogen': 1,'gaz':-1.45},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "CCS1"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO210['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "CCS2"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO210['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR_elec"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO210['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.4},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR_elecCCS1"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO210['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': -156, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.57},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "electrolysis_AEL"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO210['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.54},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "electrolysis_PEMEL"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO210['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.67},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "curtailment"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO210['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 100000}
            }
         )
    )

scenarioCO210['conversionTechs'] = pd.concat(scenarioCO210['conversionTechs'], axis=1)

scenarioDict['CO2_10']=scenarioCO210
#endregion

#region Scenario CO2_100
scenarioCO2100={k: v.copy() for (k, v) in scenarioPACA.items()}

scenarioCO2100['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO2100['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity':0,
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 0}
            }
         )
    )

    tech = "WindOffShore_flot"
    max_install_capacity = [0,500,500,500]
    max_cumul_capacity= [0,500,750,1000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO2100['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "WindOnShore"
    max_install_capacity = [0,100,100,100]
    max_cumul_capacity= [0, 150 , 150, 150]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO2100['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "Solar"
    max_install_capacity = [0,100,100,100]
    max_cumul_capacity=[0,150,150,150]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO2100['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR"
    max_install_capacity = [411,10000,10000,10000]
    min_install_capacity=[411,0,0,0]
    max_cumul_capacity= [411,10000,10000,10000]
    min_cumul_capacity =[411,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO2100['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Hydrogen production',
                'lifeSpan': lifespan, 'powerCost': 0.21, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': min_install_capacity[k],'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'hydrogen':1,'gaz':-1.28},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': min_cumul_capacity[k],'maxCumulCapacity': max_cumul_capacity[k] ,'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS1"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO2100['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 15.21, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': -150, 'Conversion': {'hydrogen': 1,'gaz':-1.32},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS2"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO2100['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 27.2, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity':10000,
                'EmissionCO2': -270, 'Conversion': {'hydrogen': 1,'gaz':-1.45},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "CCS1"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO2100['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "CCS2"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO2100['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR_elec"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO2100['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.4},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR_elecCCS1"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO2100['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': -156, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.57},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "electrolysis_AEL"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO2100['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.54},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "electrolysis_PEMEL"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO2100['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.67},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "curtailment"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCO2100['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 100000}
            }
         )
    )

scenarioCO2100['conversionTechs'] = pd.concat(scenarioCO2100['conversionTechs'], axis=1)

scenarioDict['CO2_100']=scenarioCO2100
#endregion

#region Scenario Manosque
scenarioCavern={k: v.copy() for (k, v) in scenarioPACA.items()}


scenarioCavern['storageTechs'] = []
for k, year in enumerate(yearList[:-1]):
    tech = "Battery"
    max_install_capacity = [0, 0, 0, 0]
    max_install_power = [0, 0, 0, 0]
    capex1, opex1, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech + ' - 1h', hyp='ref', year=year+yearStep/2)
    capex4, opex4, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech + ' - 4h', hyp='ref', year=year+yearStep/2)
    capex_per_kWh = (capex4 - capex1) / 3
    capex_per_kW = capex1 - capex_per_kWh

    scenarioCavern['storageTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'resource': 'electricity',
                'storagelifeSpan': lifespan,
                'storagePowerCost': capex_per_kW,
                'storageEnergyCost': capex_per_kWh,
                'storageOperationCost': opex1,
                'p_max': max_install_power[k],
                'c_max': max_install_capacity[k],
                'chargeFactors': {'electricity': 0.9200},
                'dischargeFactors': {'electricity': 1.09},
                'dissipation': 0.0085,
                },
            }
         )
    )

    tech = "tankH2_G"
    max_install_capacity = [0,10000,20000,30000]
    max_install_power=[0,1000,2000,3000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavern['storageTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year,
               'resource': 'hydrogen',
               'storagelifeSpan': lifespan,
                'storagePowerCost': capex*0.7,
                'storageEnergyCost': capex*0.3,
                'storageOperationCost': opex,
                'p_max': max_install_power[k],
                'c_max': max_install_capacity[k],
                'chargeFactors': {'electricity': 0.0168,'hydrogen':1},
                'dischargeFactors': {'hydrogen': 1},
                'dissipation': 0,
                },
            }
         )
    )

    tech = "saltCavernH2_G"
    max_install_capacity = [0,130000,130000,130000]
    max_install_power=[0,13000,13000,13000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavern['storageTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year,
               'resource': 'hydrogen',
               'storagelifeSpan': lifespan,
                'storagePowerCost': capex,
                'storageEnergyCost': 280,
                'storageOperationCost': opex,
                'p_max': max_install_power[k],
                'c_max': max_install_capacity[k],
                'chargeFactors': {'electricity': 0.0168,'hydrogen':1},
                'dischargeFactors': {'hydrogen': 1},
                'dissipation': 0,
                },
            }
         )
    )

scenarioCavern['storageTechs'] =  pd.concat(scenarioCavern['storageTechs'], axis=1)

scenarioDict['Caverns']=scenarioCavern
#endregion

#region Scenario Free
scenarioFree={k: v.copy() for (k, v) in scenarioPACA.items()}

scenarioFree['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFree['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity':0,
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 0}
            }
         )
    )

    tech = "WindOffShore_flot"
    max_install_capacity = [500,500,500,500]
    max_cumul_capacity= [500,500,750,1000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFree['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "WindOnShore"
    max_install_capacity = [100,100,100,100]
    max_cumul_capacity= [150, 150 , 150, 150]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFree['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "Solar"
    max_install_capacity = [100,100,100,100]
    max_cumul_capacity=[150,150,150,150]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFree['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR"
    max_cumul_capacity = [411, 10000, 10000, 10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFree['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Hydrogen production',
                'lifeSpan': lifespan, 'powerCost': 0.21, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen':1,'gaz':-1.28},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k] ,'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS1"
    max_cumul_capacity= [10000,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFree['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 7.71, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': -150, 'Conversion': {'hydrogen': 1,'gaz':-1.32},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS2"
    max_cumul_capacity= [10000,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFree['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 13.7, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': -270, 'Conversion': {'hydrogen': 1,'gaz':-1.45},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "CCS1"
    max_cumul_capacity= [10000,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFree['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "CCS2"
    max_cumul_capacity= [10000,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFree['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR_elec"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFree['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.4},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR_elecCCS1"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFree['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': -156, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.57},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "electrolysis_AEL"
    max_cumul_capacity= [10000,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFree['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.54},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "electrolysis_PEMEL"
    max_cumul_capacity= [10000,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFree['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.67},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "curtailment"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioFree['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 100000}
            }
         )
    )

scenarioFree['conversionTechs'] = pd.concat(scenarioFree['conversionTechs'], axis=1)

scenarioDict['Free']=scenarioFree
#endregion

#region Scenario TC * 1.2
scenarioTC120={k: v.copy() for (k, v) in scenarioPACA.items()}

scenarioTC120['carbonTax'] = pd.DataFrame(data=np.linspace(0.0675,0.2, nYears),
    index=yearList, columns=('carbonTax',))

scenarioDict['TC120']=scenarioTC120
#endregion

#region Scenario Manosque + RE_inf
scenarioCavernREinf={k: v.copy() for (k, v) in scenarioPACA.items()}


scenarioCavernREinf['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavernREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity':0,
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 0}
            }
         )
    )

    tech = "WindOffShore_flot"
    max_install_capacity = [0, 10000, 10000, 10000]
    max_cumul_capacity = [0, 10000, 10000, 10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavernREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity':0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "WindOnShore"
    max_install_capacity = [0, 10000, 10000, 10000]
    max_cumul_capacity = [0, 10000, 10000, 10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavernREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "Solar"
    max_install_capacity = [0, 10000, 10000, 10000]
    max_cumul_capacity = [0, 10000, 10000, 10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavernREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'electricity': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR"
    max_install_capacity = [411,10000,10000,10000]
    min_install_capacity=[411,0,0,0]
    max_cumul_capacity= [411,10000,10000,10000]
    min_cumul_capacity =[411,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavernREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Hydrogen production',
                'lifeSpan': lifespan, 'powerCost': 0.21, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': min_install_capacity[k],'maxInstallCapacity': max_install_capacity[k],
                'EmissionCO2': 0, 'Conversion': {'hydrogen':1,'gaz':-1.28},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': min_cumul_capacity[k],'maxCumulCapacity': max_cumul_capacity[k] ,'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS1"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavernREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost':7.71, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': -150, 'Conversion': {'hydrogen': 1,'gaz':-1.32},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS2"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavernREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 13.7, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': -270, 'Conversion': {'hydrogen': 1,'gaz':-1.45},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "CCS1"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavernREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "CCS2"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavernREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 10000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "SMR_elec"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavernREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.4},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR_elecCCS1"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavernREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': -156, 'Conversion': {'hydrogen': 1,'gaz':-0.91,'electricity':-0.57},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "electrolysis_AEL"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavernREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.54},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "electrolysis_PEMEL"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavernREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1,'electricity':-1.67},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k]}
            }
         )
    )

    tech = "curtailment"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavernREinf['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 100000}
            }
         )
    )

scenarioCavernREinf['conversionTechs'] = pd.concat(scenarioCavernREinf['conversionTechs'], axis=1)

scenarioCavernREinf['storageTechs'] = []
for k, year in enumerate(yearList[:-1]):
    tech = "Battery"
    max_install_capacity = [0, 0, 0, 0]
    max_install_power = [0, 0, 0, 0]
    capex1, opex1, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech + ' - 1h', hyp='ref', year=year+yearStep/2)
    capex4, opex4, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech + ' - 4h', hyp='ref', year=year+yearStep/2)
    capex_per_kWh = (capex4 - capex1) / 3
    capex_per_kW = capex1 - capex_per_kWh

    scenarioCavernREinf['storageTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'resource': 'electricity',
                'storagelifeSpan': lifespan,
                'storagePowerCost': capex_per_kW,
                'storageEnergyCost': capex_per_kWh,
                'storageOperationCost': opex1,
                'p_max': max_install_power[k],
                'c_max': max_install_capacity[k],
                'chargeFactors': {'electricity': 0.9200},
                'dischargeFactors': {'electricity': 1.09},
                'dissipation': 0.0085,
                },
            }
         )
    )

    tech = "tankH2_G"
    max_install_capacity = [0,10000,20000,30000]
    max_install_power=[0,1000,2000,3000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavernREinf['storageTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year,
               'resource': 'hydrogen',
               'storagelifeSpan': lifespan,
                'storagePowerCost': capex*0.7,
                'storageEnergyCost': capex*0.3,
                'storageOperationCost': opex,
                'p_max': max_install_power[k],
                'c_max': max_install_capacity[k],
                'chargeFactors': {'electricity': 0.0168,'hydrogen':1},
                'dischargeFactors': {'hydrogen': 1},
                'dissipation': 0,
                },
            }
         )
    )

    tech = "saltCavernH2_G"
    max_install_capacity = [0,130000,130000,130000]
    max_install_power=[0,13000,13000,13000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavernREinf['storageTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year,
               'resource': 'hydrogen',
               'storagelifeSpan': lifespan,
                'storagePowerCost': capex,
                'storageEnergyCost': 280,
                'storageOperationCost': opex,
                'p_max': max_install_power[k],
                'c_max': max_install_capacity[k],
                'chargeFactors': {'electricity': 0.0168,'hydrogen':1},
                'dischargeFactors': {'hydrogen': 1},
                'dissipation': 0,
                },
            }
         )
    )

scenarioCavernREinf['storageTechs'] =  pd.concat(scenarioCavernREinf['storageTechs'], axis=1)

scenarioDict['CavernREinf']=scenarioCavernREinf
#endregion

#region Scenario export
scenarioExport={k: v.copy() for (k, v) in scenarioPACA.items()}

scenarioExport['storageTechs'] = []
for k, year in enumerate(yearList[:-1]):
    tech = "Battery"
    max_install_capacity = [0, 0, 0, 0]
    max_install_power = [0, 0, 0, 0]
    capex1, opex1, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech + ' - 1h', hyp='ref', year=year+yearStep/2)
    capex4, opex4, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech + ' - 4h', hyp='ref', year=year+yearStep/2)
    capex_per_kWh = (capex4 - capex1) / 3
    capex_per_kW = capex1 - capex_per_kWh

    scenarioExport['storageTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'resource': 'electricity',
                'storagelifeSpan': lifespan,
                'storagePowerCost': capex_per_kW,
                'storageEnergyCost': capex_per_kWh,
                'storageOperationCost': opex1,
                'p_max': max_install_power[k],
                'c_max': max_install_capacity[k],
                'chargeFactors': {'electricity': 0.9200},
                'dischargeFactors': {'electricity': 1.09},
                'dissipation': 0.0085,
                },
            }
         )
    )

    tech = "tankH2_G"
    max_install_capacity = [0,10000,20000,30000]
    max_install_power=[0,1000,2000,3000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioExport['storageTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year,
               'resource': 'hydrogen',
               'storagelifeSpan': lifespan,
                'storagePowerCost': capex*0.7,
                'storageEnergyCost': capex*0.3,
                'storageOperationCost': opex,
                'p_max': max_install_power[k],
                'c_max': max_install_capacity[k],
                'chargeFactors': {'electricity': 0.0168,'hydrogen':1},
                'dischargeFactors': {'hydrogen': 1},
                'dissipation': 0,
                },
            }
         )
    )

    tech = "saltCavernH2_G"
    max_install_capacity = [0,0,0,0]
    max_install_power=[0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioExport['storageTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year,
               'resource': 'hydrogen',
               'storagelifeSpan': lifespan,
                'storagePowerCost': capex,
                'storageEnergyCost': 280,
                'storageOperationCost': opex,
                'p_max': max_install_power[k],
                'c_max': max_install_capacity[k],
                'chargeFactors': {'electricity': 0.0168,'hydrogen':1},
                'dischargeFactors': {'hydrogen': 1},
                'dissipation': 0,
                },
            }
         )
    )

scenarioExport['storageTechs'] =  pd.concat(scenarioExport['storageTechs'], axis=1)

# expH2Cap=np.linspace(0, 30e6, nYears)
scenarioExport['maxExportCap'] = pd.concat(
    (
        pd.DataFrame(index=[year],data={
            'electricity': 10e6,
            'gazNat': 0,
            'gazBio': 0,
            'hydrogen': 0,
            'gaz': 0
        }) for k, year in enumerate(yearList[1:])
    )
)

scenarioDict['Export']=scenarioExport
#endregion


