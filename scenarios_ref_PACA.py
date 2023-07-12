import numpy as np
from scipy.interpolate import interp1d
from Basic_functionalities import tech_eco_data
import pandas as pd

inputPath='Data/Raw_Ana/'
outputFolderFr='Data/output/Ref_wH2_Fr'

nHours = 8760
t = np.arange(1,nHours + 1)


yearZero = 2010
yearFinal = 2050
yearStep = 10
yearList = [yr for yr in range(yearZero, yearFinal+yearStep, yearStep)] # +1 to include the final year
nYears = len(yearList)

scenarioPACA = {}

#hourlyDemand_H2=[360,390,460,755]
hourlyDemand_H2=interp1d(yearList[1:], [360 * (1 + 0.025) ** (k * yearStep) for k in np.arange(len(yearList[1:]))], fill_value=(360,755),bounds_error=False)

scenarioPACA['resourceDemand'] =  pd.concat(
    (
        pd.DataFrame(data = { 
              'YEAR': year, 
              'TIMESTAMP': t, # We add the TIMESTAMP so that it can be used as an index later. 
              'electricity': np.zeros(nHours), # Only considering H2 final demand
              'hydrogen': hourlyDemand_H2(year+yearStep/2)*np.ones(nHours),
              'gaz': np.zeros(nHours),
             } 
        ) for k, year in enumerate(yearList[1:])
    ) 
)

scenarioPACA['conversionTechs'] = []
for k, year in enumerate(yearList[:-1]):

    tech = "WindOffShore"
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioPACA['conversionTechs'].append(
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
    scenarioPACA['conversionTechs'].append(
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
    scenarioPACA['conversionTechs'].append(
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
    scenarioPACA['conversionTechs'].append(
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
    scenarioPACA['conversionTechs'].append(
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
    scenarioPACA['conversionTechs'].append(
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
    scenarioPACA['conversionTechs'].append(
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
    scenarioPACA['conversionTechs'].append(
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
    scenarioPACA['conversionTechs'].append(
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
    scenarioPACA['conversionTechs'].append(
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
    scenarioPACA['conversionTechs'].append(
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
    scenarioPACA['conversionTechs'].append(
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
    scenarioPACA['conversionTechs'].append(
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
    scenarioPACA['conversionTechs'].append(
        pd.DataFrame(data={tech:
                { 'YEAR': year, 'Category': 'Electricity production',
                'lifeSpan': lifespan, 'powerCost': 3000, 'investCost': capex, 'operationCost': opex,
                'minInstallCapacity': 0,'maxInstallCapacity': 100000,
                'EmissionCO2': 0, 'Conversion': {'hydrogen': 1},
                'minCumulCapacity': 0,'maxCumulCapacity': 100000}
            }
         )
    )

scenarioPACA['conversionTechs'] = pd.concat(scenarioPACA['conversionTechs'], axis=1)

scenarioPACA['storageTechs'] = []
for k, year in enumerate(yearList[:-1]):
    tech = "Battery"
    max_install_capacity = [0,5000,10000,77000]
    # max_install_capacity = [0, 0, 0, 0]
    max_install_power=[0,500,1000,7700]
    # max_install_power = [0, 0, 0, 0]
    capex1, opex1, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech + ' - 1h', hyp='ref', year=year+yearStep/2)
    capex4, opex4, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech + ' - 4h', hyp='ref', year=year+yearStep/2)
    capex_per_kWh = (capex4 - capex1) / 3
    capex_per_kW = capex1 - capex_per_kWh

    scenarioPACA['storageTechs'].append(
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
    # max_install_capacity = [0,0,0,0]
    max_install_power=[0,1000,2000,3000]
    # max_install_power = [0, 0, 0, 0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioPACA['storageTechs'].append(
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
    # max_install_capacity = [0,130000,130000,130000]
    max_install_capacity = [0, 0, 0, 0]
    # max_install_power=[0,13000,13000,13000]
    max_install_power = [0, 0, 0, 0]
    capex, opex, lifespan = tech_eco_data.get_capex_new_tech_RTE(tech, hyp='ref', year=year+yearStep/2)
    scenarioPACA['storageTechs'].append(
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

scenarioPACA['storageTechs'] =  pd.concat(scenarioPACA['storageTechs'], axis=1)

scenarioPACA['carbonTax'] = pd.DataFrame(data=np.linspace(0.0675,0.165, nYears),
    index=yearList, columns=('carbonTax',))

scenarioPACA['carbonGoals'] = pd.DataFrame(data=np.linspace(974e6, 205e6, nYears),
    index=yearList, columns=('carbonGoals',))

impBiogasCap=np.linspace(0, 5e6, nYears)
# impH2Cap=np.linspace(0, 30e6, nYears)
scenarioPACA['maxImportCap'] = pd.concat(
    (
        pd.DataFrame(index=[year],data={
            'electricity': 10e10,
            'gazNat': 10e10,
            'gazBio': impBiogasCap[k+1],
            'hydrogen': 0,
            'gaz': 0
        }) for k, year in enumerate(yearList[1:])
    )
)

# expH2Cap=np.linspace(0, 30e6, nYears)
scenarioPACA['maxExportCap'] = pd.concat(
    (
        pd.DataFrame(index=[year],data={
            'electricity': 0,#10e6,
            'gazNat': 0,
            'gazBio': 0,
            'hydrogen': 0,
            'gaz': 0
        }) for k, year in enumerate(yearList[1:])
    )
)

scenarioPACA['gridConnection'] = pd.read_csv(inputPath+'CalendrierHTB_TIME.csv', sep=',', decimal='.', skiprows=0,
                                comment="#").set_index(["TIMESTAMP"])

scenarioPACA['economicParameters'] = pd.DataFrame({
    'discountRate':[0.04], 
    'financeRate': [0.04]
    }
)

df_res_ref = pd.read_csv(inputPath+'set2020-2050_horaire_TIMExRESxYEAR.csv',
    sep=',', decimal='.', skiprows=0,comment="#").set_index(["YEAR", "TIMESTAMP",'RESOURCES'])
df_elecPrice=pd.read_csv(outputFolderFr+'/marketPrice.csv').set_index(['YEAR_op','TIMESTAMP'])
df_elecCarbon=pd.read_csv(outputFolderFr+'/carbon.csv').set_index(['YEAR_op','TIMESTAMP'])

gasPriceFactor=[1,2,2,2]
bioGasPrice=[120,110,100,90]
scenarioPACA['turpeFactorsHTB']=pd.DataFrame(columns=['HORAIRE','fixeTurpeHTB'],data={'P':5880,'HPH':5640,'HCH':5640,'HPE':5280,'HCE':4920}.items()).set_index('HORAIRE') # en €/MW/an part abonnement
scenarioPACA['resourceImportPrices'] = pd.concat(
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

scenarioPACA['resourceImportCO2eq'] = pd.concat(
    (
        pd.DataFrame(data={
            'YEAR': year, 
            'TIMESTAMP': t, 
            'electricity':  df_elecCarbon.loc[(year, slice(None)),'carbonContent'].values,
            'gaz': max(0, 0.03 * (1 - (year - yearZero)/(2050 - yearZero))) * 29 / 13.1 + 203.5  * (1 - tech_eco_data.get_biogas_share_in_network_RTE(year)), # Taking 100 yr GWP of methane and 3% losses due to upstream leaks. Losses drop to zero in 2050.
            'gazNat': max(0, 0.03 * (1 - (year - yearZero)/(2050 - yearZero))) * 29 / 13.1 + 203.5  * (1 - tech_eco_data.get_biogas_share_in_network_RTE(year)), # Taking 100 yr GWP of methane and 3% losses due to upstream leaks. Losses drop to zero in 2050.
            'gazBio': max(0, 0.03 * (1 - (year - yearZero)/(2050 - yearZero))) * 29 / 13.1,
            'hydrogen': max(0, 0.05  - .03 * (year - yearZero)/(2050 - yearZero)) * 11 / 33, # Taking 100 yr GWP of H2 and 5% losses due to upstream leaks. Leaks fall to 2% in 2050 See: https://www.energypolicy.columbia.edu/research/commentary/hydrogen-leakage-potential-risk-hydrogen-economy
        }) for k, year in enumerate(yearList[1:])
    )
)

scenarioPACA['convTechList'] = ["WindOnShore", "WindOffShore_flot", "Solar", 'SMR','SMR + CCS1','SMR + CCS2','CCS1','CCS2','electrolysis_PEMEL','electrolysis_AEL' ,"curtailment",'SMR_elec','SMR_elecCCS1']
ctechs = scenarioPACA['convTechList']
availabilityFactor = pd.read_csv(inputPath+'availabilityFactor2010-2050_PACA_TIMExTECHxYEAR.csv',
                                 sep=',', decimal='.', skiprows=0).set_index(["YEAR", "TIMESTAMP", "TECHNOLOGIES"])
itechs = availabilityFactor.index.isin(ctechs, level=2)
scenarioPACA['availability'] = availabilityFactor.loc[(slice(None), slice(None), itechs)]

scenarioPACA["yearList"] = yearList
scenarioPACA["transitionFactors"] =pd.DataFrame(
    {'TECHNO1':['SMR','SMR'],
    'TECHNO2':['SMR + CCS1','SMR + CCS2'],
    'TransFactor': 1}).set_index(['TECHNO1','TECHNO2'])