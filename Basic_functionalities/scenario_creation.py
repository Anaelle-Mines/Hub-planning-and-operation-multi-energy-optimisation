
import os

import numpy as np
import pandas as pd
import csv
import datetime
import copy
import time

from Basic_functionalities.scenarios_ref_PACA import scenarioPACA
from Basic_functionalities.scenarios_2060_PACA import scenarioPACA2060
from Basic_functionalities.scenarios_testPrix_PACA import scenarioPACAtest
from Basic_functionalities import tech_eco_data

outputPath='Data/output/'

scenarioDict={'Ref':scenarioPACA,'test2060':scenarioPACA2060,'test':scenarioPACAtest}


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
    max_install_capacity = [360,1000,1000,1000]
    min_install_capacity=[360,0,0,0]
    max_cumul_capacity= [360,10000,10000,10000]
    min_cumul_capacity =[360,0,0,0]
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

#region Scenario ElecOnly

scenarioElecOnly={k: v.copy() for (k, v) in scenarioPACA.items()}

scenarioElecOnly['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioElecOnly['conversionTechs'].append(
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
    scenarioElecOnly['conversionTechs'].append(
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
    scenarioElecOnly['conversionTechs'].append(
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
    scenarioElecOnly['conversionTechs'].append(
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
    max_install_capacity = [360,0,0,0]
    min_install_capacity=[360,0,0,0]
    max_cumul_capacity= [360,0,0,0]
    min_cumul_capacity =[360,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioElecOnly['conversionTechs'].append(
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
    scenarioElecOnly['conversionTechs'].append(
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
    scenarioElecOnly['conversionTechs'].append(
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
    scenarioElecOnly['conversionTechs'].append(
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
    scenarioElecOnly['conversionTechs'].append(
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
    scenarioElecOnly['conversionTechs'].append(
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
    scenarioElecOnly['conversionTechs'].append(
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
    scenarioElecOnly['conversionTechs'].append(
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
    scenarioElecOnly['conversionTechs'].append(
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
    scenarioElecOnly['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 100000}
            }
         )
    )

scenarioElecOnly['conversionTechs'] = pd.concat(scenarioElecOnly['conversionTechs'], axis=1)

scenarioElecOnly["transitionFactors"] =pd.DataFrame(
    {'TECHNO1':[],
    'TECHNO2':[],
    'TransFactor':[] }).set_index(['TECHNO1','TECHNO2'])

scenarioDict['ElecOnly']=scenarioElecOnly

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
    max_install_capacity = [360,1000,1000,1000]
    min_install_capacity=[360,0,0,0]
    max_cumul_capacity= [360,10000,10000,10000]
    min_cumul_capacity =[360,0,0,0]
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
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioSmrOnly['conversionTechs'].append(
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
    scenarioSmrOnly['conversionTechs'].append(
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
    scenarioSmrOnly['conversionTechs'].append(
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
    scenarioSmrOnly['conversionTechs'].append(
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

scenarioSmrOnly['carbonTax'] = pd.DataFrame(data=np.linspace(0.0675,0.0675, nYears),
    index=yearList, columns=('carbonTax',))

scenarioSmrOnly['maxBiogasCap'] = pd.DataFrame(data=np.linspace(0, 0, nYears),
    index=yearList, columns=('maxBiogasCap',))

scenarioSmrOnly["transitionFactors"] =pd.DataFrame(
    {'TECHNO1':[],
    'TECHNO2':[],
    'TransFactor':[] }).set_index(['TECHNO1','TECHNO2'])

scenarioDict['SmrOnly']=scenarioSmrOnly

#endregion

#region Scenatio Grid
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
    max_install_capacity = [360,1000,1000,1000]
    min_install_capacity=[360,0,0,0]
    max_cumul_capacity= [360,10000,10000,10000]
    min_cumul_capacity =[360,0,0,0]
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

