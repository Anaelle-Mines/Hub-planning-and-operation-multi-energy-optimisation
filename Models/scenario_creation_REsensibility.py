
import os

import numpy as np
import pandas as pd
import csv
import datetime
import copy

from Models.scenarios_ref_PACA import scenarioPACA
from Data.Raw import tech_eco_data

outputPath='../Data/output/'
inputPath='../Data/Raw/'
outputFolderFr='../Data/output/Ref_Fr'

scenarioDict_RE={'Ref':scenarioPACA}

nHours = 8760
t = np.arange(1,nHours + 1)

yearZero = 2010
yearFinal = 2050
yearStep = 10
yearList = [yr for yr in range(yearZero, yearFinal+yearStep, yearStep)] # +1 to include the final year
nYears = len(yearList)

#region Scenario EnR_inf / TC 165 / CO2 50
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

scenarioDict_RE['Re_inf']=scenarioREinf
#endregion

#region Scenario EnR_inf / TC 165 / CO2 50 / woSMR
scenarioREinf_woSMR={k: v.copy() for (k, v) in scenarioPACA.items()}

scenarioREinf_woSMR['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinf_woSMR['conversionTechs'].append(
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
    scenarioREinf_woSMR['conversionTechs'].append(
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
    scenarioREinf_woSMR['conversionTechs'].append(
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
    scenarioREinf_woSMR['conversionTechs'].append(
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
    scenarioREinf_woSMR['conversionTechs'].append(
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
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinf_woSMR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost':7.71, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': -150, 'Conversion': {'hydrogen': 1,'gaz':-1.32},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': 0,'maxCumulCapacity':0,'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS2"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinf_woSMR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 13.7, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': -270, 'Conversion': {'hydrogen': 1,'gaz':-1.45},
                'minCumulCapacity': 0,'maxCumulCapacity':0,'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "CCS1"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinf_woSMR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': 0}
            }
         )
    )

    tech = "CCS2"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinf_woSMR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': 0}
            }
         )
    )

    tech = "SMR_elec"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinf_woSMR['conversionTechs'].append(
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
    scenarioREinf_woSMR['conversionTechs'].append(
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
    scenarioREinf_woSMR['conversionTechs'].append(
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
    scenarioREinf_woSMR['conversionTechs'].append(
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
    scenarioREinf_woSMR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 100000}
            }
         )
    )

scenarioREinf_woSMR['conversionTechs'] = pd.concat(scenarioREinf_woSMR['conversionTechs'], axis=1)

scenarioDict_RE['Re_inf_woSMR']=scenarioREinf_woSMR
#endregion

#region Scenario EnR_inf / TC 165 / CO2 10
scenarioREinfCCS10={k: v.copy() for (k, v) in scenarioREinf.items()}

scenarioREinfCCS10['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinfCCS10['conversionTechs'].append(
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
    scenarioREinfCCS10['conversionTechs'].append(
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
    scenarioREinfCCS10['conversionTechs'].append(
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
    scenarioREinfCCS10['conversionTechs'].append(
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
    scenarioREinfCCS10['conversionTechs'].append(
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
    scenarioREinfCCS10['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost':1.71, 'investCost': capex, 'operationCost': opex,
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
    scenarioREinfCCS10['conversionTechs'].append(
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
    scenarioREinfCCS10['conversionTechs'].append(
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
    scenarioREinfCCS10['conversionTechs'].append(
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
    scenarioREinfCCS10['conversionTechs'].append(
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
    scenarioREinfCCS10['conversionTechs'].append(
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
    scenarioREinfCCS10['conversionTechs'].append(
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
    scenarioREinfCCS10['conversionTechs'].append(
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
    scenarioREinfCCS10['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 100000}
            }
         )
    )

scenarioREinfCCS10['conversionTechs'] = pd.concat(scenarioREinfCCS10['conversionTechs'], axis=1)

scenarioDict_RE['Re_inf_CCS10']=scenarioREinfCCS10
#endregion

#region Scenario EnR_inf / TC 165 / CO2 10 / woSMR
scenarioREinfCCS10_woSMR={k: v.copy() for (k, v) in scenarioREinf_woSMR.items()}

scenarioREinfCCS10_woSMR['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinfCCS10_woSMR['conversionTechs'].append(
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
    scenarioREinfCCS10_woSMR['conversionTechs'].append(
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
    scenarioREinfCCS10_woSMR['conversionTechs'].append(
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
    scenarioREinfCCS10_woSMR['conversionTechs'].append(
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
    scenarioREinfCCS10_woSMR['conversionTechs'].append(
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
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinfCCS10_woSMR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost':1.71, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': -150, 'Conversion': {'hydrogen': 1,'gaz':-1.32},
                'EnergyNbhourCap': 0, # used for hydroelectricity
                'minCumulCapacity': 0,'maxCumulCapacity':0,'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "SMR + CCS2"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinfCCS10_woSMR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 2.9, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': -270, 'Conversion': {'hydrogen': 1,'gaz':-1.45},
                'minCumulCapacity': 0,'maxCumulCapacity':0,'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "CCS1"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinfCCS10_woSMR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': 0}
            }
         )
    )

    tech = "CCS2"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinfCCS10_woSMR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 0, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 0},
                'minCumulCapacity': 0,'maxCumulCapacity': 0}
            }
         )
    )

    tech = "SMR_elec"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioREinfCCS10_woSMR['conversionTechs'].append(
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
    scenarioREinfCCS10_woSMR['conversionTechs'].append(
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
    scenarioREinfCCS10_woSMR['conversionTechs'].append(
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
    scenarioREinfCCS10_woSMR['conversionTechs'].append(
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
    scenarioREinfCCS10_woSMR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 100000}
            }
         )
    )

scenarioREinfCCS10_woSMR['conversionTechs'] = pd.concat(scenarioREinfCCS10_woSMR['conversionTechs'], axis=1)

scenarioDict_RE['Re_inf_CCS10_woSMR']=scenarioREinfCCS10_woSMR
#endregion

#region Scenario EnR_inf / TC 200 / CO2 50
scenarioREinfTC200={k: v.copy() for (k, v) in scenarioDict_RE['Re_inf'].items()}

scenarioREinfTC200['carbonTax'] = pd.DataFrame(data=np.linspace(0.0675,0.2, nYears),index=yearList, columns=('carbonTax',))

scenarioDict_RE['Re_inf_TC200']=scenarioREinfTC200
#endregion

#region Scenario EnR_inf / TC 200 / CO2 50 / woSMR
scenarioREinfTC200_woSMR={k: v.copy() for (k, v) in scenarioDict_RE['Re_inf_woSMR'].items()}

scenarioREinfTC200_woSMR['carbonTax'] = pd.DataFrame(data=np.linspace(0.0675,0.2, nYears),index=yearList, columns=('carbonTax',))

scenarioDict_RE['Re_inf_TC200_woSMR']=scenarioREinfTC200_woSMR
#endregion

#region Scenario Cavern / TC 165 / CO2 50
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

scenarioDict_RE['Caverns']=scenarioCavern
#endregion

#region Scenario Cavern / TC 165 / CO2 50 / woSMR
scenarioCavern_woSMR={k: v.copy() for (k, v) in scenarioPACA.items()}

scenarioCavern_woSMR['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavern_woSMR['conversionTechs'].append(
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
    scenarioCavern_woSMR['conversionTechs'].append(
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
    scenarioCavern_woSMR['conversionTechs'].append(
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
    scenarioCavern_woSMR['conversionTechs'].append(
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
    scenarioCavern_woSMR['conversionTechs'].append(
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
    scenarioCavern_woSMR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost':7.71, 'investCost': capex, 'operationCost': opex,
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
    scenarioCavern_woSMR['conversionTechs'].append(
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
    scenarioCavern_woSMR['conversionTechs'].append(
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
    scenarioCavern_woSMR['conversionTechs'].append(
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
    scenarioCavern_woSMR['conversionTechs'].append(
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
    scenarioCavern_woSMR['conversionTechs'].append(
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
    scenarioCavern_woSMR['conversionTechs'].append(
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
    scenarioCavern_woSMR['conversionTechs'].append(
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
    scenarioCavern_woSMR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 100000}
            }
         )
    )

scenarioCavern_woSMR['conversionTechs'] = pd.concat(scenarioCavern_woSMR['conversionTechs'], axis=1)

scenarioCavern_woSMR['storageTechs'] = []
for k, year in enumerate(yearList[:-1]):
    tech = "Battery"
    max_install_capacity = [0, 0, 0, 0]
    max_install_power = [0, 0, 0, 0]
    capex1, opex1, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech + ' - 1h', hyp='ref', year=year+yearStep/2)
    capex4, opex4, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech + ' - 4h', hyp='ref', year=year+yearStep/2)
    capex_per_kWh = (capex4 - capex1) / 3
    capex_per_kW = capex1 - capex_per_kWh

    scenarioCavern_woSMR['storageTechs'].append(
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
    scenarioCavern_woSMR['storageTechs'].append(
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
    scenarioCavern_woSMR['storageTechs'].append(
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

scenarioCavern_woSMR['storageTechs'] =  pd.concat(scenarioCavern_woSMR['storageTechs'], axis=1)

scenarioDict_RE['Caverns_woSMR']=scenarioCavern_woSMR
#endregion

#region Scenario Cavern / TC 165 / CO2 10
scenarioCavernCCS10={k: v.copy() for (k, v) in scenarioCavern.items()}

scenarioCavernCCS10['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavernCCS10['conversionTechs'].append(
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
    scenarioCavernCCS10['conversionTechs'].append(
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
    scenarioCavernCCS10['conversionTechs'].append(
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
    scenarioCavernCCS10['conversionTechs'].append(
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
    scenarioCavernCCS10['conversionTechs'].append(
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
    scenarioCavernCCS10['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost':1.71, 'investCost': capex, 'operationCost': opex,
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
    scenarioCavernCCS10['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 2.9, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': -270, 'Conversion': {'hydrogen': 1,'gaz':-1.45},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "CCS1"
    max_cumul_capacity= [0,10000,10000,10000]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavernCCS10['conversionTechs'].append(
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
    scenarioCavernCCS10['conversionTechs'].append(
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
    scenarioCavernCCS10['conversionTechs'].append(
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
    scenarioCavernCCS10['conversionTechs'].append(
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
    scenarioCavernCCS10['conversionTechs'].append(
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
    scenarioCavernCCS10['conversionTechs'].append(
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
    scenarioCavernCCS10['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 100000}
            }
         )
    )

scenarioCavernCCS10['conversionTechs'] = pd.concat(scenarioCavernCCS10['conversionTechs'], axis=1)


scenarioDict_RE['Caverns_CCS10']=scenarioCavernCCS10
#endregion

#region Scenario Cavern / TC 165 / CO2 10 / woSMR
scenarioCavernCCS10_woSMR={k: v.copy() for (k, v) in scenarioCavern_woSMR.items()}

scenarioCavernCCS10_woSMR['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavernCCS10_woSMR['conversionTechs'].append(
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
    scenarioCavernCCS10_woSMR['conversionTechs'].append(
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
    scenarioCavernCCS10_woSMR['conversionTechs'].append(
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
    scenarioCavernCCS10_woSMR['conversionTechs'].append(
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
    scenarioCavernCCS10_woSMR['conversionTechs'].append(
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
    scenarioCavernCCS10_woSMR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost':1.71, 'investCost': capex, 'operationCost': opex,
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
    scenarioCavernCCS10_woSMR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 2.9, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': -270, 'Conversion': {'hydrogen': 1,'gaz':-1.45},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "CCS1"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavernCCS10_woSMR['conversionTechs'].append(
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
    scenarioCavernCCS10_woSMR['conversionTechs'].append(
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
    scenarioCavernCCS10_woSMR['conversionTechs'].append(
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
    scenarioCavernCCS10_woSMR['conversionTechs'].append(
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
    scenarioCavernCCS10_woSMR['conversionTechs'].append(
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
    scenarioCavernCCS10_woSMR['conversionTechs'].append(
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
    scenarioCavernCCS10_woSMR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 100000}
            }
         )
    )

scenarioCavernCCS10_woSMR['conversionTechs'] = pd.concat(scenarioCavernCCS10_woSMR['conversionTechs'], axis=1)

scenarioDict_RE['Caverns_CCS10_woSMR']=scenarioCavernCCS10_woSMR
#endregion

#region Scenario Cavern / TC 200 / CO2 50
scenarioCavernTC200={k: v.copy() for (k, v) in scenarioDict_RE['Caverns'].items()}

scenarioCavernTC200['carbonTax'] = pd.DataFrame(data=np.linspace(0.0675,0.2, nYears),index=yearList, columns=('carbonTax',))

scenarioDict_RE['Caverns_TC200']=scenarioCavernTC200
#endregion

#region Scenario Cavern / TC 200 / CO2 50 / woSMR
scenarioCavernTC200_woSMR={k: v.copy() for (k, v) in scenarioDict_RE['Caverns_woSMR'].items()}

scenarioCavernTC200_woSMR['carbonTax'] = pd.DataFrame(data=np.linspace(0.0675,0.2, nYears),index=yearList, columns=('carbonTax',))

scenarioDict_RE['Caverns_TC200_woSMR']=scenarioCavernTC200_woSMR
#endregion

#region Scenario Cavern + RE_inf / TC 165 / CO2 50
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

scenarioDict_RE['CavernREinf']=scenarioCavernREinf
#endregion

#region Scenario Cavern + RE_inf / TC 165 / CO2 50 / woSMR
scenarioCavernREinf_woSMR={k: v.copy() for (k, v) in scenarioPACA.items()}


scenarioCavernREinf_woSMR['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavernREinf_woSMR['conversionTechs'].append(
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
    scenarioCavernREinf_woSMR['conversionTechs'].append(
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
    scenarioCavernREinf_woSMR['conversionTechs'].append(
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
    scenarioCavernREinf_woSMR['conversionTechs'].append(
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
    scenarioCavernREinf_woSMR['conversionTechs'].append(
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
    scenarioCavernREinf_woSMR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost':7.71, 'investCost': capex, 'operationCost': opex,
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
    scenarioCavernREinf_woSMR['conversionTechs'].append(
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
    scenarioCavernREinf_woSMR['conversionTechs'].append(
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
    scenarioCavernREinf_woSMR['conversionTechs'].append(
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
    scenarioCavernREinf_woSMR['conversionTechs'].append(
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
    scenarioCavernREinf_woSMR['conversionTechs'].append(
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
    scenarioCavernREinf_woSMR['conversionTechs'].append(
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
    scenarioCavernREinf_woSMR['conversionTechs'].append(
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
    scenarioCavernREinf_woSMR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 100000}
            }
         )
    )

scenarioCavernREinf_woSMR['conversionTechs'] = pd.concat(scenarioCavernREinf_woSMR['conversionTechs'], axis=1)

scenarioCavernREinf_woSMR['storageTechs'] = []
for k, year in enumerate(yearList[:-1]):
    tech = "Battery"
    max_install_capacity = [0, 0, 0, 0]
    max_install_power = [0, 0, 0, 0]
    capex1, opex1, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech + ' - 1h', hyp='ref', year=year+yearStep/2)
    capex4, opex4, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech + ' - 4h', hyp='ref', year=year+yearStep/2)
    capex_per_kWh = (capex4 - capex1) / 3
    capex_per_kW = capex1 - capex_per_kWh

    scenarioCavernREinf_woSMR['storageTechs'].append(
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
    scenarioCavernREinf_woSMR['storageTechs'].append(
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
    scenarioCavernREinf_woSMR['storageTechs'].append(
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

scenarioCavernREinf_woSMR['storageTechs'] =  pd.concat(scenarioCavernREinf_woSMR['storageTechs'], axis=1)

scenarioDict_RE['CavernREinf_woSMR']=scenarioCavernREinf_woSMR
#endregion

#region Scenario Cavern + RE_inf / TC 165 / CO2 10
scenarioCavernREinfCCS10={k: v.copy() for (k, v) in scenarioCavernREinf.items()}


scenarioCavernREinfCCS10['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavernREinfCCS10['conversionTechs'].append(
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
    scenarioCavernREinfCCS10['conversionTechs'].append(
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
    scenarioCavernREinfCCS10['conversionTechs'].append(
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
    scenarioCavernREinfCCS10['conversionTechs'].append(
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
    scenarioCavernREinfCCS10['conversionTechs'].append(
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
    scenarioCavernREinfCCS10['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost':1.71, 'investCost': capex, 'operationCost': opex,
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
    scenarioCavernREinfCCS10['conversionTechs'].append(
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
    scenarioCavernREinfCCS10['conversionTechs'].append(
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
    scenarioCavernREinfCCS10['conversionTechs'].append(
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
    scenarioCavernREinfCCS10['conversionTechs'].append(
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
    scenarioCavernREinfCCS10['conversionTechs'].append(
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
    scenarioCavernREinfCCS10['conversionTechs'].append(
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
    scenarioCavernREinfCCS10['conversionTechs'].append(
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
    scenarioCavernREinfCCS10['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 100000}
            }
         )
    )

scenarioCavernREinfCCS10['conversionTechs'] = pd.concat(scenarioCavernREinfCCS10['conversionTechs'], axis=1)

scenarioDict_RE['CavernREinf_CCS10']=scenarioCavernREinfCCS10
#endregion

#region Scenario Cavern + RE_inf / TC 165 / CO2 50 / woSMR
scenarioCavernREinfCCS10_woSMR={k: v.copy() for (k, v) in scenarioCavernREinf_woSMR.items()}


scenarioCavernREinfCCS10_woSMR['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavernREinfCCS10_woSMR['conversionTechs'].append(
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
    scenarioCavernREinfCCS10_woSMR['conversionTechs'].append(
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
    scenarioCavernREinfCCS10_woSMR['conversionTechs'].append(
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
    scenarioCavernREinfCCS10_woSMR['conversionTechs'].append(
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
    scenarioCavernREinfCCS10_woSMR['conversionTechs'].append(
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
    scenarioCavernREinfCCS10_woSMR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost':1.71, 'investCost': capex, 'operationCost': opex,
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
    scenarioCavernREinfCCS10_woSMR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 2.9, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 0,
                'EmissionCO2': -270, 'Conversion': {'hydrogen': 1,'gaz':-1.45},
                'minCumulCapacity': 0,'maxCumulCapacity': max_cumul_capacity[k],'RampConstraintPlus':0.3}
            }
         )
    )

    tech = "CCS1"
    max_cumul_capacity= [0,0,0,0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioCavernREinfCCS10_woSMR['conversionTechs'].append(
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
    scenarioCavernREinfCCS10_woSMR['conversionTechs'].append(
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
    scenarioCavernREinfCCS10_woSMR['conversionTechs'].append(
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
    scenarioCavernREinfCCS10_woSMR['conversionTechs'].append(
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
    scenarioCavernREinfCCS10_woSMR['conversionTechs'].append(
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
    scenarioCavernREinfCCS10_woSMR['conversionTechs'].append(
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
    scenarioCavernREinfCCS10_woSMR['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 100000}
            }
         )
    )

scenarioCavernREinfCCS10_woSMR['conversionTechs'] = pd.concat(scenarioCavernREinfCCS10_woSMR['conversionTechs'], axis=1)

scenarioDict_RE['CavernREinf_CCS10_woSMR']=scenarioCavernREinfCCS10_woSMR
#endregion

#region Scenario Cavern + RE_inf / TC 200 / CO2 50
scenarioCavernREinfTC200={k: v.copy() for (k, v) in scenarioDict_RE['CavernREinf'].items()}

scenarioCavernREinfTC200['carbonTax'] = pd.DataFrame(data=np.linspace(0.0675,0.2, nYears),index=yearList, columns=('carbonTax',))

scenarioDict_RE['CavernREinf_TC200']=scenarioCavernREinfTC200
#endregion

#region Scenario Cavern + RE_inf / TC 200 / CO2 50 / woSMR
scenarioCavernREinfTC200_woSMR={k: v.copy() for (k, v) in scenarioDict_RE['CavernREinf_woSMR'].items()}

scenarioCavernREinfTC200_woSMR['carbonTax'] = pd.DataFrame(data=np.linspace(0.0675,0.2, nYears),index=yearList, columns=('carbonTax',))

scenarioDict_RE['CavernREinf_TC200_woSMR']=scenarioCavernREinfTC200_woSMR
#endregion





