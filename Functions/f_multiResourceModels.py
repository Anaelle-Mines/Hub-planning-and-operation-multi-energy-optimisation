from pyomo.environ import *
from pyomo.core import *
import xarray as xr
import numpy as np
import pandas as pd
#from Functions.f_optimization import *

def loadScenario(scenario, printTables=False):
    yearZero = scenario["yearList"][0]
    dy = scenario["yearList"][1] - yearZero

    areaConsumption = scenario['resourceDemand'].melt(id_vars=['TIMESTAMP', 'YEAR'], var_name=['RESOURCES'],
                                                      value_name='areaConsumption').set_index(
        ['YEAR', 'TIMESTAMP', 'RESOURCES'])

    TechParameters = scenario['conversionTechs'].transpose().fillna(0)
    TechParameters.index.name = 'TECHNOLOGIES'
    TechParametersList = ['powerCost', 'operationCost', 'investCost', 'EnergyNbhourCap', 'minInstallCapacity',
                          'maxInstallCapacity', 'RampConstraintPlus', 'RampConstraintMoins', 'EmissionCO2', 'minCumulCapacity', 'maxCumulCapacity']
    for k in TechParametersList:
        if k not in TechParameters:
            TechParameters[k] = 0
    TechParameters.drop(columns=['Conversion', 'Category'], inplace=True)
    TechParameters['yearStart'] = TechParameters['YEAR'] - TechParameters['lifeSpan'] // dy * dy
    TechParameters.loc[TechParameters['yearStart'] < yearZero, 'yearStart'] = 0
    TechParameters.set_index(['YEAR', TechParameters.index], inplace=True)

    StorageParameters = scenario['storageTechs'].transpose().fillna(0)
    StorageParameters.index.name = 'STOCK_TECHNO'
    StorageParametersList = ['resource', 'storagePowerCost', 'storageEnergyCost', 'p_max', 'c_max']
    for k in StorageParametersList:
        if k not in StorageParameters:
            StorageParameters[k] = 0
    StorageParameters.drop(columns=['chargeFactors', 'dischargeFactors', 'dissipation'], inplace=True)
    StorageParameters['storageYearStart'] = StorageParameters['YEAR'] - round(
        StorageParameters['storagelifeSpan'] / dy) * dy
    StorageParameters.loc[StorageParameters['storageYearStart'] < yearZero, 'storageYearStart'] = 0
    StorageParameters.set_index(['YEAR', StorageParameters.index], inplace=True)

    CarbonTax = scenario['carbonTax'].copy()
    CarbonTax.index.name = 'YEAR'

    maxImportCap=scenario['maxImportCap'].copy()
    maxImportCap.index.name='YEAR'
    maxImportCap=maxImportCap.reset_index().melt(id_vars=['YEAR'], var_name=['RESOURCES'],value_name='maxImportCap').set_index(['YEAR', 'RESOURCES'])

    maxExportCap=scenario['maxExportCap'].copy()
    maxExportCap.index.name='YEAR'
    maxExportCap=maxExportCap.reset_index().melt(id_vars=['YEAR'], var_name=['RESOURCES'],value_name='maxExportCap').set_index(['YEAR', 'RESOURCES'])

    df_conv = scenario['conversionTechs'].transpose().set_index('YEAR', append=True)['Conversion']
    conversionFactor = pd.DataFrame(
        data={tech: df_conv.loc[(tech, 2020)] for tech in scenario['conversionTechs'].columns}).fillna(
        0)  # TODO: Take into account evolving conversion factors (for electrolysis improvement, for instance)
    conversionFactor.index.name = 'RESOURCES'
    conversionFactor = conversionFactor.reset_index('RESOURCES').melt(id_vars=['RESOURCES'], var_name='TECHNOLOGIES',
                                                                      value_name='conversionFactor').set_index(
        ['RESOURCES', 'TECHNOLOGIES'])

    df_sconv = scenario['storageTechs'].transpose().set_index('YEAR', append=True)
    stechSet = set([k[0] for k in df_sconv.index.values])
    df = {}
    for k1, k2 in (('charge', 'In'), ('discharge', 'Out')):
        df[k1] = pd.DataFrame(data={tech: df_sconv.loc[(tech, 2020), k1 + 'Factors'] for tech in stechSet}).fillna(
            0)  # TODO: Take into account evolving conversion factors
        df[k1].index.name = 'RESOURCES'
        df[k1] = df[k1].reset_index(['RESOURCES']).melt(id_vars=['RESOURCES'], var_name='TECHNOLOGIES',
                                                        value_name='storageFactor' + k2)

    df['dissipation'] = pd.concat(pd.DataFrame(
        data={'dissipation': [df_sconv.loc[(tech, 2020), 'dissipation']],
              'RESOURCES': df_sconv.loc[(tech, 2020), 'resource'],
              'TECHNOLOGIES': tech}) for tech in stechSet
                                  )
    storageFactors = pd.merge(df['charge'], df['discharge'], how='outer').fillna(0)
    storageFactors = pd.merge(storageFactors, df['dissipation'], how='outer').fillna(0).set_index(
        ['RESOURCES', 'TECHNOLOGIES'])

    Calendrier = scenario['gridConnection']
    Economics = scenario['economicParameters'].melt(var_name='Eco').set_index('Eco')

    ResParameters = pd.concat((
        k.melt(id_vars=['TIMESTAMP', 'YEAR'], var_name=['RESOURCES'], value_name=name).set_index(
            ['YEAR', 'TIMESTAMP', 'RESOURCES'])
        for k, name in [(scenario['resourceImportPrices'], 'importCost'), (scenario['resourceImportCO2eq'], 'emission')]
    ), axis=1)

    availabilityFactor = scenario['availability']

    # Return hydrogen annual consumption in kt
    if printTables:
        print(areaConsumption.loc[slice(None), slice(None), 'electricity'].groupby('YEAR').sum() / 33e3)
        print(TechParameters)
        print(CarbonTax)
        print(conversionFactor)
        print(StorageParameters)
        print(storageFactors)
        print(ResParameters)
        print(availabilityFactor)

    inputDict = scenario.copy()
    inputDict["areaConsumption"] = areaConsumption
    inputDict["availabilityFactor"] = availabilityFactor
    inputDict["techParameters"] = TechParameters
    inputDict["resParameters"] = ResParameters
    inputDict["conversionFactor"] = conversionFactor
    inputDict["economics"] = Economics
    inputDict["calendar"] = Calendrier
    inputDict["storageParameters"] = StorageParameters
    inputDict["storageFactors"] = storageFactors
    inputDict["carbonTax"] = CarbonTax
    inputDict["transitionFactors"] = scenario["transitionFactors"]
    inputDict["yearList"] = scenario["yearList"]
    inputDict["maxImportCap"]=maxImportCap
    inputDict["maxExportCap"] = maxExportCap
    inputDict["turpeFactors"] = scenario["turpeFactorsHTB"]

    return inputDict

def systemModel_MultiResource_WithStorage(scenario, zone,isAbstract=False):
    """
    This function creates the pyomo model and initlize the Parameters and (pyomo) Set values
    :param areaConsumption: panda table with consumption
    :param availabilityFactor: panda table
    :param isAbstract: boolean true is the model should be abstract. ConcreteModel otherwise
    :return: pyomo model
    """

    inputDict = loadScenario(scenario, False)

    yearList = np.array(inputDict["yearList"])
    dy = yearList[1] - yearList[0]
    y0 = yearList[0]


    areaConsumption = inputDict["areaConsumption"].loc[(inputDict["yearList"][1:], slice(None), slice(None))]
    availabilityFactor = inputDict["availabilityFactor"].loc[(inputDict["yearList"][1:], slice(None), slice(None))]
    TechParameters = inputDict["techParameters"].loc[(slice(None),inputDict['convTechList']),slice(None)]
    ResParameters = inputDict["resParameters"]
    conversionFactor = inputDict["conversionFactor"].loc[(slice(None),inputDict['convTechList']),slice(None)]
    Economics = inputDict["economics"]
    Calendrier = inputDict["calendar"]
    StorageParameters = inputDict["storageParameters"]
    storageFactors = inputDict["storageFactors"]
    TransFactors = inputDict["transitionFactors"]
    CarbonTax = inputDict["carbonTax"].loc[inputDict["yearList"][1:]]
    carbonGoals = inputDict["carbonGoals"].loc[inputDict["yearList"][1:]]
    maxImportCap = inputDict["maxImportCap"]
    maxExportCap = inputDict["maxExportCap"]
    turpeFactors=inputDict["turpeFactors"]

    isAbstract = False
    availabilityFactor.isna().sum()

    ### Cleaning
    availabilityFactor = availabilityFactor.fillna(method='pad');
    areaConsumption = areaConsumption.fillna(method='pad');
    ResParameters = ResParameters.fillna(0);

    ### obtaining dimensions values
    TECHNOLOGIES = set(TechParameters.index.get_level_values('TECHNOLOGIES').unique())
    STOCK_TECHNO = set(StorageParameters.index.get_level_values('STOCK_TECHNO').unique())
    RESOURCES = set(ResParameters.index.get_level_values('RESOURCES').unique())
    TIMESTAMP = set(areaConsumption.index.get_level_values('TIMESTAMP').unique())
    YEAR = set(yearList)

    TIMESTAMP_list = areaConsumption.index.get_level_values('TIMESTAMP').unique()
    YEAR_list = yearList

    HORAIRE = set(turpeFactors.index.get_level_values('HORAIRE').unique())
    # Subsets
    TIMESTAMP_HCH = set(Calendrier[Calendrier['Calendrier'] == 'HCH'].index.get_level_values('TIMESTAMP').unique())
    TIMESTAMP_HPH = set(Calendrier[Calendrier['Calendrier'] == 'HPH'].index.get_level_values('TIMESTAMP').unique())
    TIMESTAMP_HCE = set(Calendrier[Calendrier['Calendrier'] == 'HCE'].index.get_level_values('TIMESTAMP').unique())
    TIMESTAMP_HPE = set(Calendrier[Calendrier['Calendrier'] == 'HPE'].index.get_level_values('TIMESTAMP').unique())
    TIMESTAMP_P = set(Calendrier[Calendrier['Calendrier'] == 'P'].index.get_level_values('TIMESTAMP').unique())

    #####################
    #    Pyomo model    #
    #####################

    if (isAbstract):
        model = pyomo.environ.AbstractModel()
    else:
        model = pyomo.environ.ConcreteModel()

    ###############
    # Sets       ##
    ###############
    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES, ordered=False)
    model.STOCK_TECHNO = Set(initialize=STOCK_TECHNO, ordered=False)
    model.RESOURCES = Set(initialize=RESOURCES, ordered=False)
    model.TIMESTAMP = Set(initialize=TIMESTAMP, ordered=False)
    model.YEAR = Set(initialize=YEAR, ordered=False)
    model.HORAIRE = Set(initialize=HORAIRE, ordered=False)
    model.YEAR_invest = Set(initialize=YEAR_list[:-1], ordered=False)
    model.YEAR_op = Set(initialize=YEAR_list[1:], ordered=False)
    model.YEAR_invest_TECHNOLOGIES = model.YEAR_invest * model.TECHNOLOGIES
    model.YEAR_invest_STOCKTECHNO = model.YEAR_invest * model.STOCK_TECHNO
    model.YEAR_op_TECHNOLOGIES = model.YEAR_op * model.TECHNOLOGIES
    model.YEAR_op_TIMESTAMP_TECHNOLOGIES = model.YEAR_op * model.TIMESTAMP * model.TECHNOLOGIES
    model.YEAR_op_TIMESTAMP_STOCKTECHNO = model.YEAR_op * model.TIMESTAMP * model.STOCK_TECHNO
    model.RESOURCES_TECHNOLOGIES = model.RESOURCES * model.TECHNOLOGIES
    model.RESOURCES_STOCKTECHNO = model.RESOURCES * model.STOCK_TECHNO
    model.YEAR_op_TIMESTAMP_RESOURCES = model.YEAR_op * model.TIMESTAMP * model.RESOURCES
    model.TECHNOLOGIES_TECHNOLOGIES = model.TECHNOLOGIES * model.TECHNOLOGIES
    model.YEAR_op_RESOURCES = model.YEAR_op * model.RESOURCES

    # Subset of Simple only required if ramp constraint
    model.TIMESTAMP_MinusOne = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 1], ordered=False)
    model.TIMESTAMP_MinusThree = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 3], ordered=False)

    ###############
    # Parameters ##
    ###############
    model.areaConsumption = Param(model.YEAR_op_TIMESTAMP_RESOURCES, default=0,
                                  initialize=areaConsumption.loc[:, "areaConsumption"].squeeze().to_dict(),
                                  domain=Reals)
    model.availabilityFactor = Param(model.YEAR_op_TIMESTAMP_TECHNOLOGIES, domain=PercentFraction, default=1,
                                     initialize=availabilityFactor.loc[:, "availabilityFactor"].squeeze().to_dict())
    model.conversionFactor = Param(model.RESOURCES_TECHNOLOGIES, default=0,
                                   initialize=conversionFactor.loc[:, "conversionFactor"].squeeze().to_dict())
    model.carbon_goal = Param(model.YEAR_op, default=0,
                              initialize=carbonGoals.loc[:, 'carbonGoals'].squeeze().to_dict(), domain=NonNegativeReals)
    model.carbon_taxe = Param(model.YEAR_op, default=0, initialize=CarbonTax.loc[:, 'carbonTax'].squeeze().to_dict(),
                              domain=NonNegativeReals)
    model.import_max = Param(model.YEAR_op_RESOURCES, default=0,
                             initialize=maxImportCap.loc[:, "maxImportCap"].squeeze().to_dict(),
                             domain=NonNegativeReals)
    model.export_max = Param(model.YEAR_op_RESOURCES, default=0,
                             initialize=maxExportCap.loc[:, "maxExportCap"].squeeze().to_dict(),
                             domain=NonNegativeReals)
    model.transFactor = Param(model.TECHNOLOGIES_TECHNOLOGIES, mutable=False, default=0,
                              initialize=TransFactors.loc[:, 'TransFactor'].squeeze().to_dict())
    model.turpeFactors=Param(model.HORAIRE, mutable=False, initialize=turpeFactors.loc[:, 'fixeTurpeHTB'].squeeze().to_dict())

    gasTypes = ['gazBio', 'gazNat']
    # with test of existing columns on TechParameters

    for COLNAME in TechParameters:
        if COLNAME not in ["TECHNOLOGIES", "AREAS", "YEAR"]:  ### each column in TechParameters will be a parameter
            exec("model." + COLNAME + "= Param(model.YEAR_invest, model.TECHNOLOGIES, default=0, domain=Reals," +
                 "initialize=TechParameters." + COLNAME + ".loc[(inputDict['yearList'][:-1], slice(None))].squeeze().to_dict())")

    for COLNAME in ResParameters:
        if COLNAME not in ["TECHNOLOGIES", "AREAS", "YEAR"]:  ### each column in TechParameters will be a parameter
            exec("model." + COLNAME + "= Param(model.YEAR_op_TIMESTAMP_RESOURCES, domain=NonNegativeReals,default=0," +
                 "initialize=ResParameters." + COLNAME + ".squeeze().to_dict())")

    for COLNAME in Calendrier:
        if COLNAME not in ["TIMESTAMP", "AREAS"]:
            exec("model." + COLNAME + " = Param(model.TIMESTAMP, default=0," +
                 "initialize=Calendrier." + COLNAME + ".squeeze().to_dict(),domain=Any)")

    for COLNAME in StorageParameters:
        if COLNAME not in ["STOCK_TECHNO", "AREAS", "YEAR"]:  ### each column in StorageParameters will be a parameter
            exec("model." + COLNAME + " =Param(model.YEAR_invest_STOCKTECHNO,domain=Any,default=0," +
                 "initialize=StorageParameters." + COLNAME + ".loc[(inputDict['yearList'][:-1], slice(None))].squeeze().to_dict())")

    for COLNAME in storageFactors:
        if COLNAME not in ["TECHNOLOGIES", "RESOURCES"]:
            exec("model." + COLNAME + " =Param(model.RESOURCES_STOCKTECHNO,domain=NonNegativeReals,default=0," +
                 "initialize=storageFactors." + COLNAME + ".squeeze().to_dict())")

    ################
    # Variables    #
    ################

    # In this section, variables are separated in two categories : decision variables wich are the reals variables of the otimisation problem (these are noted Dvar), and problem variables which are resulting of calculation and are convenient for the readability and the analyse of results (these are noted Pvar)

    # Objective function
    model.objective_Pvar = Var(model.YEAR_op)

    # Operation
    model.power_Dvar = Var(model.YEAR_op, model.TIMESTAMP, model.TECHNOLOGIES,
                           domain=NonNegativeReals)  ### Power of a conversion mean at time t
    model.importation_Dvar = Var(model.YEAR_op, model.TIMESTAMP, model.RESOURCES,domain=NonNegativeReals,
                                 initialize=0)  ### Importation of a resource at time t
    model.turpeQuantity=Var(model.YEAR_op,model.TIMESTAMP,model.RESOURCES,
                           domain=NonNegativeReals,initialize=0)
    model.exportation_Dvar = Var(model.YEAR_op, model.TIMESTAMP, model.RESOURCES,domain=NonNegativeReals,
                                 initialize=0)  ### Exportation of a resource at time t
    model.energy_Pvar = Var(model.YEAR_op, model.TIMESTAMP, model.RESOURCES)  ### Amount of a resource at time t
    model.max_PS_Dvar = Var(model.YEAR_op, model.HORAIRE,
                            domain=NonNegativeReals)  ### Puissance souscrite max par plage horaire pour l'année d'opération y
    model.carbon_Pvar = Var(model.YEAR_op, model.TIMESTAMP)  ### CO2 emission at each time t


    ### Storage operation variables
    model.stockLevel_Pvar = Var(model.YEAR_op, model.TIMESTAMP, model.STOCK_TECHNO,
                                domain=NonNegativeReals)  ### level of the energy stock in a storage mean at time t
    model.storageIn_Pvar = Var(model.YEAR_op, model.TIMESTAMP, model.RESOURCES, model.STOCK_TECHNO,
                               domain=NonNegativeReals)  ### Energy stored in a storage mean at time t
    model.storageOut_Pvar = Var(model.YEAR_op, model.TIMESTAMP, model.RESOURCES, model.STOCK_TECHNO,
                                domain=NonNegativeReals)  ### Energy taken out of the in a storage mean at time t
    model.storageConsumption_Pvar = Var(model.YEAR_op, model.TIMESTAMP, model.RESOURCES, model.STOCK_TECHNO,
                                        domain=NonNegativeReals)  ### Energy consumed the in a storage mean at time t (other than the one stored)


    # Investment
    model.capacityInvest_Dvar = Var(model.YEAR_invest, model.TECHNOLOGIES, domain=NonNegativeReals,
                                    initialize=0)  ### Capacity of a conversion mean invested in year y
    model.capacityDel_Pvar = Var(model.YEAR_invest, model.YEAR_invest, model.TECHNOLOGIES,
                                 domain=NonNegativeReals)  ### Capacity of a conversion mean that is removed each year y
    model.transInvest_Dvar = Var(model.YEAR_invest, model.TECHNOLOGIES, model.TECHNOLOGIES,
                                 domain=NonNegativeReals)  ### Transformation of technologies 1 into technologies 2
    model.capacityDem_Dvar = Var(model.YEAR_invest, model.YEAR_invest, model.TECHNOLOGIES, domain=NonNegativeReals)
    model.capacity_Pvar = Var(model.YEAR_op, model.TECHNOLOGIES, domain=NonNegativeReals, initialize=0)
    model.CmaxInvest_Dvar = Var(model.YEAR_invest, model.STOCK_TECHNO,
                                domain=NonNegativeReals)  # Maximum capacity of a storage mean
    model.PmaxInvest_Dvar = Var(model.YEAR_invest, model.STOCK_TECHNO,
                                domain=NonNegativeReals)  # Maximum flow of energy in/out of a storage mean
    model.Cmax_Pvar = Var(model.YEAR_op, model.STOCK_TECHNO,
                          domain=NonNegativeReals)  # Maximum capacity of a storage mean
    model.Pmax_Pvar = Var(model.YEAR_op, model.STOCK_TECHNO,
                          domain=NonNegativeReals)  # Maximum flow of energy in/out of a storage mean
    model.CmaxDel_Dvar = Var(model.YEAR_invest, model.STOCK_TECHNO, domain=NonNegativeReals)
    model.PmaxDel_Dvar = Var(model.YEAR_invest, model.STOCK_TECHNO, domain=NonNegativeReals)


    #
    model.powerCosts_Pvar = Var(model.YEAR_op,
                                model.TECHNOLOGIES)  ### Marginal cost for a conversion mean, explicitely defined by definition powerCostsDef
    model.capacityCosts_Pvar = Var(model.YEAR_op,
                                   model.TECHNOLOGIES)  ### Fixed costs for a conversion mean, explicitely defined by definition capacityCostsDef
    model.importCosts_Pvar = Var(model.YEAR_op,model.RESOURCES)  ### Cost of ressource imported, explicitely defined by definition importCostsDef
    model.turpeCosts_Pvar = Var(model.YEAR_op, model.RESOURCES,domain=NonNegativeReals)  ### Coûts TURPE pour électricité
    model.turpeCostsFixe_Pvar = Var(model.YEAR_op, model.RESOURCES,domain=NonNegativeReals)
    model.turpeCostsVar_Pvar = Var(model.YEAR_op, model.RESOURCES,domain=NonNegativeReals)
    model.storageCosts_Pvar = Var(model.YEAR_op,
                                  model.STOCK_TECHNO)  ### Cost of storage for a storage mean, explicitely defined by definition storageCostsDef
    model.carbonCosts_Pvar = Var(model.YEAR_op,domain=NonNegativeReals)

    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    model.slack = Suffix(direction=Suffix.IMPORT)

    ########################
    # Objective Function   #
    ########################

    def ObjectiveFunction_rule(model):  # OBJ
        return sum(model.objective_Pvar[y] for y in model.YEAR_op)

    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)

    def objective_rule(model,y):
        return sum(model.powerCosts_Pvar[y, tech] + model.capacityCosts_Pvar[y, tech] for tech in model.TECHNOLOGIES)\
               + sum(model.importCosts_Pvar[y, res] for res in model.RESOURCES)\
               + sum(model.storageCosts_Pvar[y, s_tech] for s_tech in STOCK_TECHNO)\
               + model.turpeCosts_Pvar[y, 'electricity']\
               + model.carbonCosts_Pvar[y]\
               == model.objective_Pvar[y]

    model.objective_ruleCtr = Constraint(model.YEAR_op,rule=objective_rule)

    #################
    # Constraints   #
    #################
    r = Economics.loc['discountRate'].value
    i = Economics.loc['financeRate'].value
    y_act=Economics.loc['y_act'].value

    def f1(r, n):  # This factor converts the investment costs into n annual repayments
        return r / ((1 + r) * (1 - (1 + r) ** -n))

    def f3(r, y,y_act='middle'):  # This factor discounts a payment to y0 values
        if y_act == 'beginning': 
            return (1 + r) ** (-(y - y0))
        elif y_act == 'middle': 
            return (1 + r) ** (-(y+dy/2 - y0))
        elif y_act == 'end': 
            return (1 + r) ** (-(y+dy - y0))
        else :
            print("Error in the actualisation factor, please select one of the three 'beginning', 'middle', 'end'.")



    # powerCosts definition Constraints
    def powerCostsDef_rule(model, y,tech):  # EQ forall tech in TECHNOLOGIES powerCosts  = sum{t in TIMESTAMP} powerCost[tech]*power[t,tech] / 1E6;
        return sum(model.powerCost[y - dy, tech] * f3(r, y,y_act) * model.power_Dvar[y, t, tech] for t in model.TIMESTAMP) == \
               model.powerCosts_Pvar[y, tech]

    model.powerCostsCtr = Constraint(model.YEAR_op, model.TECHNOLOGIES, rule=powerCostsDef_rule)

    # capacityCosts definition Constraints
    def capacityCostsDef_rule(model, y, tech):  # EQ forall tech in TECHNOLOGIES
        return sum(model.investCost[yi, tech]
                   * f1(i, model.lifeSpan[yi, tech]) * f3(r, y - dy,y_act)
                   * (model.capacityInvest_Dvar[yi, tech] - model.capacityDel_Pvar[yi, y - dy, tech])
                   for yi in yearList[yearList < y]) + model.operationCost[y - dy, tech] * f3(r, y,y_act) * \
               model.capacity_Pvar[y, tech] == model.capacityCosts_Pvar[y, tech]

    model.capacityCostsCtr = Constraint(model.YEAR_op, model.TECHNOLOGIES, rule=capacityCostsDef_rule)

    # importCosts definition Constraints
    def importCostsDef_rule(model, y, res):
        return sum(
            (model.importCost[y, t, res] * f3(r, y,y_act) * (model.importation_Dvar[y, t, res]-model.exportation_Dvar[y,t,res])) for t in model.TIMESTAMP) == \
               model.importCosts_Pvar[y, res]

    model.importCostsCtr = Constraint(model.YEAR_op, model.RESOURCES, rule=importCostsDef_rule)

    # Max importation/exportation definition Constraints
    def importMaxDef_rule(model, y, res):
        return sum(model.importation_Dvar[y, t, res] for t in model.TIMESTAMP) <= model.import_max[y,res]

    model.importMaxCtr = Constraint(model.YEAR_op, model.RESOURCES, rule=importMaxDef_rule)

    def exportMaxDef_rule(model, y, res):
        return sum(model.exportation_Dvar[y, t, res] for t in model.TIMESTAMP) <= model.export_max[y,res]

    model.exportMaxCtr = Constraint(model.YEAR_op, model.RESOURCES, rule=exportMaxDef_rule)

    # Carbon emission definition Constraints
    def CarbonDef_rule(model, y, t):
        return sum(
            (model.power_Dvar[y, t, tech] * model.EmissionCO2[y - dy, tech]) for tech in model.TECHNOLOGIES) + sum(
            model.importation_Dvar[y, t, res] * model.emission[y, t, res] for res in model.RESOURCES) == \
               model.carbon_Pvar[y, t]

    model.CarbonDefCtr = Constraint(model.YEAR_op, model.TIMESTAMP, rule=CarbonDef_rule)

    # def CarbonCtr_rule(model):
    # return sum(model.carbon_Pvar[y,t] for y,t in zip(model.YEAR_op,model.TIMESTAMP)) <= sum(model.carbon_goal[y] for y in model.YEAR_op)
    # model.CarbonCtr = Constraint(rule=CarbonCtr_rule)

    # def CarbonCtr_rule(model,y):
    #     return sum(model.carbon_Pvar[y,t] for t in model.TIMESTAMP) <= model.carbon_goal[y]
    # model.CarbonCtr = Constraint(model.YEAR_op,rule=CarbonCtr_rule)

    # CarbonCosts definition Constraint
    def CarbonCosts_rule(model, y):
        return model.carbonCosts_Pvar[y] == sum(
            model.carbon_Pvar[y, t] * model.carbon_taxe[y] * f3(r, y,y_act) for t in model.TIMESTAMP)

    model.CarbonCostsCtr = Constraint(model.YEAR_op, rule=CarbonCosts_rule)

    # PPA
    def turpePPA_rule(model,y,t,res):
        if res == 'electricity':
            return model.turpeQuantity[y,t,res] == model.importation_Dvar[y,t,res]+ model.exportation_Dvar[y,t,res]
        else :
            return Constraint.Skip

    model.turpePPACtr = Constraint(model.YEAR_op,model.TIMESTAMP,model.RESOURCES,rule=turpePPA_rule)

    # TURPE
    def PuissanceSouscrite_rule(model, y, t, res):
        if res == 'electricity':
            if t in TIMESTAMP_P:
                return model.max_PS_Dvar[y, 'P'] >= model.turpeQuantity[y, t, res]   # en MW
            elif t in TIMESTAMP_HPH:
                return model.max_PS_Dvar[y, 'HPH'] >= model.turpeQuantity[y, t, res] 
            elif t in TIMESTAMP_HCH:
                return model.max_PS_Dvar[y, 'HCH'] >= model.turpeQuantity[y, t, res] 
            elif t in TIMESTAMP_HPE:
                return model.max_PS_Dvar[y, 'HPE'] >= model.turpeQuantity[y, t, res] 
            elif t in TIMESTAMP_HCE:
                return model.max_PS_Dvar[y, 'HCE'] >= model.turpeQuantity[y, t, res]
        else:
            return Constraint.Skip

    model.PuissanceSouscriteCtr = Constraint(model.YEAR_op, model.TIMESTAMP, model.RESOURCES,rule=PuissanceSouscrite_rule)

    def TurpeCtr1_rule(model, y):
        return model.max_PS_Dvar[y, 'P'] <= model.max_PS_Dvar[y, 'HPH']

    model.TurpeCtr1 = Constraint(model.YEAR_op, rule=TurpeCtr1_rule)

    def TurpeCtr2_rule(model, y):
        return model.max_PS_Dvar[y, 'HPH'] <= model.max_PS_Dvar[y, 'HCH']

    model.TurpeCtr2 = Constraint(model.YEAR_op, rule=TurpeCtr2_rule)

    def TurpeCtr3_rule(model, y):
        return model.max_PS_Dvar[y, 'HCH'] <= model.max_PS_Dvar[y, 'HPE']

    model.TurpeCtr3 = Constraint(model.YEAR_op,rule=TurpeCtr3_rule)

    def TurpeCtr4_rule(model, y):
        return model.max_PS_Dvar[y, 'HPE'] <= model.max_PS_Dvar[y, 'HCE']

    model.TurpeCtr4 = Constraint(model.YEAR_op, rule=TurpeCtr4_rule)

    def TurpeCostsFixe_rule(model, y, res):
        if res == 'electricity':
            return model.turpeCostsFixe_Pvar[y, res] == (model.max_PS_Dvar[y, 'P'] * model.turpeFactors['P'] + (model.max_PS_Dvar[y, 'HPH'] - model.max_PS_Dvar[y, 'P']) * model.turpeFactors['HPH'] + (model.max_PS_Dvar[y, 'HCH'] - model.max_PS_Dvar[y, 'HPH']) * model.turpeFactors['HCH'] + (model.max_PS_Dvar[y, 'HPE'] - model.max_PS_Dvar[y, 'HCH']) * model.turpeFactors['HPE'] + (model.max_PS_Dvar[y, 'HCE'] - model.max_PS_Dvar[y, 'HPE']) * model.turpeFactors['HCE']) * f3(r, y,y_act)
        else:
            return model.turpeCostsFixe_Pvar[y, res] == 0
    model.TurpeCostsFixe = Constraint(model.YEAR_op, model.RESOURCES, rule=TurpeCostsFixe_rule)

    def TurpeCostsVar_rule(model, y, res):
        if res == 'electricity':
            return model.turpeCostsVar_Pvar[y, res] == sum(model.HTB[t] * (model.turpeQuantity[y, t, res] + model.exportation_Dvar[y,t,res]) for t in TIMESTAMP) * f3(r, y,y_act)
        else:
            return model.turpeCostsVar_Pvar[y, res] == 0

    model.TurpeCostsVar = Constraint(model.YEAR_op, model.RESOURCES, rule=TurpeCostsVar_rule)

    def TurpeCostsDef_rule(model, y, res):
        return model.turpeCosts_Pvar[y, res] == model.turpeCostsFixe_Pvar[y, res] + model.turpeCostsVar_Pvar[y, res]

    model.TurpeCostsDef = Constraint(model.YEAR_op, model.RESOURCES, rule=TurpeCostsDef_rule)

    # Capacity constraints
    if ('CCS1' and 'CCS2') in model.TECHNOLOGIES:
        def capacityCCS_rule(model, y, tech):
            if tech == 'CCS1':
                return model.capacityInvest_Dvar[y, tech] == sum(model.transInvest_Dvar[y, tech1, tech2]
                                                                 for tech1, tech2 in
                                                                 [('SMR', 'SMR + CCS1'), ('SMR', 'SMR + CCS2'),])
            elif tech == 'CCS2':
                return model.capacityInvest_Dvar[y, tech] == sum(model.transInvest_Dvar[y, tech1, tech2]
                                                                 for tech1, tech2 in
                                                                 [('SMR', 'SMR + CCS2'),('SMR + CCS1', 'SMR + CCS2')])
            else:
                return Constraint.Skip

        model.capacityCCSCtr = Constraint(model.YEAR_invest, model.TECHNOLOGIES, rule=capacityCCS_rule)

    def TransInvest_rule(model, y, tech1, tech2):
        if model.transFactor[tech1, tech2] == 0:
            return model.transInvest_Dvar[y, tech1, tech2] == 0
        else:
            return Constraint.Skip

    model.TransInvestCtr = Constraint(model.YEAR_invest, model.TECHNOLOGIES, model.TECHNOLOGIES, rule=TransInvest_rule)

    def TransCapacity_rule(model, y, tech):
        if y == y0:
            return sum(model.transInvest_Dvar[y, tech, tech2] for tech2 in model.TECHNOLOGIES) == 0
        else:
            return sum(model.transInvest_Dvar[y, tech, tech2] for tech2 in model.TECHNOLOGIES) <= \
                       model.capacity_Pvar[y, tech]

    model.TransCapacityCtr = Constraint(model.YEAR_invest, model.TECHNOLOGIES, rule=TransCapacity_rule)

    def CapacityDemUB_rule(model, yi, y, tech):
        if yi == model.yearStart[y, tech]:
            return sum(model.capacityDem_Dvar[yi, z, tech] for z in yearList[yearList <= y]) == \
                   model.capacityInvest_Dvar[yi, tech]
        elif yi > y:
            return model.capacityDem_Dvar[yi, y, tech] == 0
        else:
            return sum(model.capacityDem_Dvar[yi, yt, tech] for yt in model.YEAR_invest) <= model.capacityInvest_Dvar[
                yi, tech]

    model.CapacityDemUBCtr = Constraint(model.YEAR_invest, model.YEAR_invest, model.TECHNOLOGIES,
                                        rule=CapacityDemUB_rule)

    def CapacityDel_rule(model, yi, y, tech):
        if model.yearStart[y, tech] >= yi:
            return model.capacityDel_Pvar[yi, y, tech] == model.capacityInvest_Dvar[yi, tech]
        else:
            return model.capacityDel_Pvar[yi, y, tech] == 0

    model.CapacityDelCtr = Constraint(model.YEAR_invest, model.YEAR_invest, model.TECHNOLOGIES, rule=CapacityDel_rule)

    def CapacityTot_rule(model, y, tech):
        if y == y0 + dy:
            return model.capacity_Pvar[y, tech] == model.capacityInvest_Dvar[y - dy, tech] - sum(
                model.capacityDem_Dvar[yi, y - dy, tech] for yi in model.YEAR_invest) + sum(
                model.transInvest_Dvar[y - dy, tech1, tech] for tech1 in model.TECHNOLOGIES) - sum(
                model.transInvest_Dvar[y - dy, tech, tech2] for tech2 in model.TECHNOLOGIES)
        else:
            return model.capacity_Pvar[y, tech] == model.capacity_Pvar[y - dy, tech] - sum(
                model.capacityDem_Dvar[yi, y - dy, tech] for yi in model.YEAR_invest) + model.capacityInvest_Dvar[
                       y - dy, tech] + sum(
                model.transInvest_Dvar[y - dy, tech1, tech] for tech1 in model.TECHNOLOGIES) - sum(
                model.transInvest_Dvar[y - dy, tech, tech2] for tech2 in model.TECHNOLOGIES)

    model.CapacityTotCtr = Constraint(model.YEAR_op, model.TECHNOLOGIES, rule=CapacityTot_rule)

    def Capacity_rule(model, y, t, tech):  # INEQ forall t, tech
        return model.capacity_Pvar[y, tech] * model.availabilityFactor[y, t, tech] >= model.power_Dvar[y, t, tech]

    model.CapacityCtr = Constraint(model.YEAR_op, model.TIMESTAMP, model.TECHNOLOGIES, rule=Capacity_rule)

    # Ressource production constraint
    def Production_rule(model, y, t, res):  # EQ forall t, res
        if res == 'gaz':
            return sum(
                model.power_Dvar[y, t, tech] * model.conversionFactor[res, tech] for tech in model.TECHNOLOGIES) + sum(
                model.importation_Dvar[y, t, resource] - model.exportation_Dvar[y,t,res] for resource in gasTypes) + \
                   sum(model.storageOut_Pvar[y, t, res, s_tech] - model.storageIn_Pvar[y, t, res, s_tech] -
                       model.storageConsumption_Pvar[y, t, res, s_tech] for s_tech in STOCK_TECHNO) == \
                   model.energy_Pvar[y, t, res]
        elif res in gasTypes:
            return model.energy_Pvar[y, t, res] == 0
        else:
            return sum(
                model.power_Dvar[y, t, tech] * model.conversionFactor[res, tech] for tech in model.TECHNOLOGIES) + \
                   model.importation_Dvar[y, t, res] - model.exportation_Dvar[y,t,res] + sum(
                model.storageOut_Pvar[y, t, res, s_tech] - model.storageIn_Pvar[y, t, res, s_tech] -
                model.storageConsumption_Pvar[y, t, res, s_tech] for s_tech in STOCK_TECHNO) == model.energy_Pvar[
                       y, t, res]

    model.ProductionCtr = Constraint(model.YEAR_op, model.TIMESTAMP, model.RESOURCES, rule=Production_rule)

    # contrainte d'equilibre offre demande
    def energyCtr_rule(model, y, t, res):  # INEQ forall
        return model.energy_Pvar[y, t, res] == model.areaConsumption[y, t, res]

    model.energyCtr = Constraint(model.YEAR_op, model.TIMESTAMP, model.RESOURCES, rule=energyCtr_rule)

    # Storage power and capacity constraints
    def StorageCmaxTot_rule(model, y, stech):  # INEQ forall t, tech
        if y == y0 + dy:
            return model.Cmax_Pvar[y, stech] == model.CmaxInvest_Dvar[y - dy, stech] - model.CmaxDel_Dvar[y - dy, stech]
        else:
            return model.Cmax_Pvar[y, stech] == model.Cmax_Pvar[y - dy, stech] + model.CmaxInvest_Dvar[y - dy, stech] - \
                   model.CmaxDel_Dvar[y - dy, stech]

    model.StorageCmaxTotCtr = Constraint(model.YEAR_op, model.STOCK_TECHNO, rule=StorageCmaxTot_rule)

    def StoragePmaxTot_rule(model, y, s_tech):  # INEQ forall t, tech
        if y == y0 + dy:
            return model.Pmax_Pvar[y, s_tech] == model.PmaxInvest_Dvar[y - dy, s_tech] - model.PmaxDel_Dvar[
                y - dy, s_tech]
        else:
            return model.Pmax_Pvar[y, s_tech] == model.Pmax_Pvar[y - dy, s_tech] + model.PmaxInvest_Dvar[
                y - dy, s_tech] - model.PmaxDel_Dvar[y - dy, s_tech]

    model.StoragePmaxTotCtr = Constraint(model.YEAR_op, model.STOCK_TECHNO, rule=StoragePmaxTot_rule)

    # storageCosts definition Constraint
    def storageCostsDef_rule(model, y, s_tech):  # EQ forall s_tech in STOCK_TECHNO
        return sum((model.storageEnergyCost[yi, s_tech] * model.Cmax_Pvar[yi + dy, s_tech] +
                    model.storagePowerCost[yi, s_tech] * model.Pmax_Pvar[yi + dy, s_tech]) * f1(i,
                                                                                                model.storagelifeSpan[
                                                                                                    yi, s_tech]) * f3(r,
                                                                                                                      y - dy,y_act)
                   for yi in yearList[yearList < y]) \
               + model.storageOperationCost[y - dy, s_tech] * f3(r, y,y_act) * model.Pmax_Pvar[y, s_tech] == \
               model.storageCosts_Pvar[y, s_tech]

    model.storageCostsCtr = Constraint(model.YEAR_op, model.STOCK_TECHNO, rule=storageCostsDef_rule)

    # Storage max capacity constraint
    def storageCapacity_rule(model, y, s_tech):  # INEQ forall s_tech
        return model.CmaxInvest_Dvar[y, s_tech] <= model.c_max[y, s_tech]

    model.storageCapacityCtr = Constraint(model.YEAR_invest, model.STOCK_TECHNO, rule=storageCapacity_rule)

    def storageCapacityDel_rule(model, y, stech):
        if model.storageYearStart[y, stech] > 0:
            return model.CmaxDel_Dvar[y, stech] == model.CmaxInvest_Dvar[model.storageYearStart[y, stech], stech]
        else:
            return model.CmaxDel_Dvar[y, stech] == 0

    model.storageCapacityDelCtr = Constraint(model.YEAR_invest, model.STOCK_TECHNO, rule=storageCapacityDel_rule)

    # Storage max power constraint
    def storagePower_rule(model, y, s_tech):  # INEQ forall s_tech
        return model.PmaxInvest_Dvar[y, s_tech] <= model.p_max[y, s_tech]

    model.storagePowerCtr = Constraint(model.YEAR_invest, model.STOCK_TECHNO, rule=storagePower_rule)

    # contraintes de stock puissance
    def StoragePowerUB_rule(model, y, t, res, s_tech):  # INEQ forall t
        if res == model.resource[y - dy, s_tech]:
            return model.storageIn_Pvar[y, t, res, s_tech] - model.Pmax_Pvar[y, s_tech] <= 0
        else:
            return model.storageIn_Pvar[y, t, res, s_tech] == 0

    model.StoragePowerUBCtr = Constraint(model.YEAR_op, model.TIMESTAMP, model.RESOURCES, model.STOCK_TECHNO,
                                         rule=StoragePowerUB_rule)

    def StoragePowerLB_rule(model, y, t, res, s_tech, ):  # INEQ forall t
        if res == model.resource[y - dy, s_tech]:
            return model.storageOut_Pvar[y, t, res, s_tech] - model.Pmax_Pvar[y, s_tech] <= 0
        else:
            return model.storageOut_Pvar[y, t, res, s_tech] == 0

    model.StoragePowerLBCtr = Constraint(model.YEAR_op, model.TIMESTAMP, model.RESOURCES, model.STOCK_TECHNO,
                                         rule=StoragePowerLB_rule)

    def storagePowerDel_rule(model, y, stech):
        if model.storageYearStart[y, stech] > 0:
            return model.PmaxDel_Dvar[y, stech] == model.PmaxInvest_Dvar[model.storageYearStart[y, stech], stech]
        else:
            return model.PmaxDel_Dvar[y, stech] == 0

    model.storagePowerDelCtr = Constraint(model.YEAR_invest, model.STOCK_TECHNO, rule=storagePowerDel_rule)

    # contrainte de consommation du stockage (autre que l'énergie stockée)
    def StorageConsumption_rule(model, y, t, res, s_tech):  # EQ forall t
        temp = model.resource[y - dy, s_tech]
        if res == temp:
            return model.storageConsumption_Pvar[y, t, res, s_tech] == 0
        else:
            return model.storageConsumption_Pvar[y, t, res, s_tech] == model.storageFactorIn[res, s_tech] * \
                   model.storageIn_Pvar[y, t, temp, s_tech] + model.storageFactorOut[res, s_tech] * \
                   model.storageOut_Pvar[
                       y, t, temp, s_tech]

    model.StorageConsumptionCtr = Constraint(model.YEAR_op, model.TIMESTAMP, model.RESOURCES, model.STOCK_TECHNO,
                                             rule=StorageConsumption_rule)

    # contraintes de stock capacité
    def StockLevel_rule(model, y, t, s_tech):  # EQ forall t
        res = model.resource[y - dy, s_tech]
        if t > 1:
            return model.stockLevel_Pvar[y, t, s_tech] == model.stockLevel_Pvar[y, t - 1, s_tech] * (
                    1 - model.dissipation[res, s_tech]) + model.storageIn_Pvar[y, t, res, s_tech] * \
                   model.storageFactorIn[res, s_tech] - model.storageOut_Pvar[y, t, res, s_tech] * \
                   model.storageFactorOut[
                       res, s_tech]
        else:
            return model.stockLevel_Pvar[y, t, s_tech] == model.stockLevel_Pvar[y, 8760, s_tech] + model.storageIn_Pvar[
                y, t, res, s_tech] * \
                   model.storageFactorIn[res, s_tech] - model.storageOut_Pvar[y, t, res, s_tech] * \
                   model.storageFactorOut[
                       res, s_tech]

    model.StockLevelCtr = Constraint(model.YEAR_op, model.TIMESTAMP, model.STOCK_TECHNO, rule=StockLevel_rule)

    def StockCapacity_rule(model, y, t, s_tech, ):  # INEQ forall t
        return model.stockLevel_Pvar[y, t, s_tech] <= model.Cmax_Pvar[y, s_tech]

    model.StockCapacityCtr = Constraint(model.YEAR_op, model.TIMESTAMP, model.STOCK_TECHNO, rule=StockCapacity_rule)

    if "maxCumulCapacity" in TechParameters:
        def capacityLimUP_rule(model, y, tech):  # INEQ forall t, tech
            return model.maxCumulCapacity[y, tech] >= model.capacity_Pvar[y + dy, tech]

        model.capacityLimUPCtr = Constraint(model.YEAR_invest, model.TECHNOLOGIES, rule=capacityLimUP_rule)

    if "minCumulCapacity" in TechParameters:
        def capacityLimUB_rule(model, y, tech):  # INEQ forall t, tech
            return model.minCumulCapacity[y, tech] <= model.capacity_Pvar[y + dy, tech]

        model.capacityLimUBCtr = Constraint(model.YEAR_invest, model.TECHNOLOGIES, rule=capacityLimUB_rule)

    if "minInstallCapacity" in TechParameters:
        def maxCapacity_rule(model, y, tech):  # INEQ forall t, tech
            return model.minInstallCapacity[y, tech] <= model.capacityInvest_Dvar[y, tech]

        model.maxCapacityCtr = Constraint(model.YEAR_invest, model.TECHNOLOGIES, rule=maxCapacity_rule)

    if "maxInstallCapacity" in TechParameters:
        def minCapacity_rule(model, y, tech):  # INEQ forall t, tech
            return model.maxInstallCapacity[y, tech] >= model.capacityInvest_Dvar[y, tech]

        model.minCapacityCtr = Constraint(model.YEAR_invest, model.TECHNOLOGIES, rule=minCapacity_rule)

    if "EnergyNbhourCap" in TechParameters:
        def storage_rule(model, y, tech):  # INEQ forall t, tech
            if model.EnergyNbhourCap[y - dy, tech] > 0:
                return model.EnergyNbhourCap[y - dy, tech] * model.capacity_Pvar[y, tech] >= sum(
                    model.power_Dvar[y, t, tech] for t in model.TIMESTAMP)
            else:
                return Constraint.Skip

        model.storageCtr = Constraint(model.YEAR_op, model.TECHNOLOGIES, rule=storage_rule)

    if "RampConstraintPlus" in TechParameters:
        def rampCtrPlus_rule(model, y, t, tech):  # INEQ forall t<
            if model.RampConstraintPlus[y - dy, tech] > 0:
                return model.power_Dvar[y, t + 1, tech] - model.power_Dvar[y, t, tech] <= model.capacity_Pvar[y, tech] * \
                       model.RampConstraintPlus[y - dy, tech];
            else:
                return Constraint.Skip

        model.rampCtrPlus = Constraint(model.YEAR_op, model.TIMESTAMP_MinusOne, model.TECHNOLOGIES,
                                       rule=rampCtrPlus_rule)

    if "RampConstraintMoins" in TechParameters:
        def rampCtrMoins_rule(model, y, t, tech):  # INEQ forall t<
            if model.RampConstraintMoins[y - dy, tech] > 0:
                var = model.power_Dvar[y, t + 1, tech] - model.power_Dvar[y, t, tech]
                return var >= - model.capacity_Pvar[y, tech] * model.RampConstraintMoins[y - dy, tech];
            else:
                return Constraint.Skip

        model.rampCtrMoins = Constraint(model.YEAR_op, model.TIMESTAMP_MinusOne, model.TECHNOLOGIES,
                                        rule=rampCtrMoins_rule)

    if "RampConstraintPlus2" in TechParameters:
        def rampCtrPlus2_rule(model, y, t, tech):  # INEQ forall t<
            if model.RampConstraintPlus2[y - dy, tech] > 0:
                var = (model.power_Dvar[y, t + 2, tech] + model.power_Dvar[y, t + 3, tech]) / 2 - (
                        model.power_Dvar[y, t + 1, tech] + model.power_Dvar[y, t, tech]) / 2;
                return var <= model.capacity_Pvar[y, tech] * model.RampConstraintPlus[y - dy, tech];
            else:
                return Constraint.Skip

        model.rampCtrPlus2 = Constraint(model.YEAR_op, model.TIMESTAMP_MinusThree, model.TECHNOLOGIES,
                                        rule=rampCtrPlus2_rule)

    if "RampConstraintMoins2" in TechParameters:
        def rampCtrMoins2_rule(model, y, t, tech):  # INEQ forall t<
            if model.RampConstraintMoins2[y - dy, tech] > 0:
                var = (model.power_Dvar[y, t + 2, tech] + model.power_Dvar[y, t + 3, tech]) / 2 - (
                        model.power_Dvar[y, t + 1, tech] + model.power_Dvar[y, t, tech]) / 2;
                return var >= - model.capacity_Pvar[y, tech] * model.RampConstraintMoins2[y - dy, tech];
            else:
                return Constraint.Skip

        model.rampCtrMoins2 = Constraint(model.YEAR_op, model.TIMESTAMP_MinusThree, model.TECHNOLOGIES,
                                         rule=rampCtrMoins2_rule)

    return model;

def ElecPrice_optim(scenario,IntercoOut=0,solver='mosek',outputFolder = 'Data/output/'):


    inputDict = loadScenario(scenario, False)

    r = scenario['economicParameters']['discountRate'].loc[0]
    y_act=scenario['economicParameters']['y_act'].loc[0]

    def f3(r, y,y_act='middle'):  # This factor discounts a payment to y0 values
        if y_act == 'beginning': 
            return (1 + r) ** (-(y - y0))
        elif y_act == 'middle': 
            return (1 + r) ** (-(y+dy/2 - y0))
        elif y_act == 'end': 
            return (1 + r) ** (-(y+dy - y0))
        else :
            print("Error in the actualisation factor, please select one of the three 'beginning', 'middle', 'end'.")


    TECH_elec=list(inputDict['conversionTechs'].transpose().loc[inputDict['conversionTechs'].transpose()['Category']=='Electricity production'].index.unique())

    elecProd = pd.read_csv(outputFolder+'/power_Dvar.csv').drop(columns='Unnamed: 0').set_index(['YEAR_op','TIMESTAMP', 'TECHNOLOGIES']).loc[(slice(None),slice(None),TECH_elec)]

    YEAR = sorted(list(elecProd.index.get_level_values('YEAR_op').unique()))
    dy=YEAR[1]-YEAR[0]
    y0=YEAR[0]-dy

    carbon_content = pd.read_csv(outputFolder+'/carbon.csv')
    elec_price = pd.read_csv(outputFolder+'/elecPrice.csv')
    capaCosts = pd.read_csv(outputFolder + '/capacityCosts_Pvar.csv').drop(columns='Unnamed: 0').set_index(['YEAR_op', 'TECHNOLOGIES'])
    carbonContent = carbon_content.set_index(['YEAR_op', 'TIMESTAMP'])
    ResParameters = inputDict['resParameters'].loc[
        (slice(None), slice(None), ['electricity', 'gaz', 'hydrogen', 'uranium'])].reset_index().rename(
        columns={'YEAR': 'YEAR_op'}).set_index(['YEAR_op', 'TIMESTAMP', 'RESOURCES'])
    gazPrice = (pd.DataFrame(pd.read_csv(outputFolder + '/importCosts_Pvar.csv').drop(columns='Unnamed: 0').set_index(
        ['YEAR_op', 'RESOURCES']).loc[(slice(None), ['gazBio', 'gazNat']), 'importCosts_Pvar']).fillna(0).groupby(
        'YEAR_op').sum()).join(pd.DataFrame(
        pd.read_csv(outputFolder + '/importation_Dvar.csv').groupby(['YEAR_op', 'RESOURCES']).sum().drop(
            columns=['Unnamed: 0', 'TIMESTAMP']).loc[(slice(None), ['gazBio', 'gazNat']), 'importation_Dvar']).fillna(
        0).groupby('YEAR_op').sum())
    gazPrice['gazPrice'] = (gazPrice['importCosts_Pvar'] / gazPrice['importation_Dvar']).fillna(0)
    Capacities=pd.read_csv(outputFolder+'/capacity_Pvar.csv').drop(columns='Unnamed: 0').set_index(['YEAR_op', 'TECHNOLOGIES'])

    marketPrice = elec_price.set_index(['YEAR_op','TIMESTAMP'])
    for yr in YEAR:
        marketPrice.loc[(yr, slice(None)), 'OldPrice_NonAct'] = marketPrice.loc[(yr, slice(None)), 'energyCtr'] / f3(r,yr,y_act)
        capaCosts.loc[(yr,slice(None)),'capacityCosts_NonAct']=capaCosts.loc[(yr,slice(None)),'capacityCosts_Pvar']/ f3(r,yr,y_act)
        ResParameters.loc[(yr, slice(None), ['gaz']), 'importCost'] = gazPrice.loc[yr]['gazPrice'] / f3(r,yr,y_act)

    export_TECH=['OldNuke', 'WindOnShore', 'Solar', 'WindOffShore', 'NewNuke','HydroRiver']

    availableCapa=inputDict['availabilityFactor'].reset_index().rename(columns={'YEAR':'YEAR_op'}).set_index(['YEAR_op','TIMESTAMP','TECHNOLOGIES']).loc[(slice(None),slice(None),export_TECH)]
    for yr in YEAR :
        for tech in export_TECH:
            availableCapa.loc[(yr,slice(None),tech),'maxCapa']=availableCapa.loc[(yr,slice(None),tech),'availabilityFactor']*Capacities.loc[(yr,tech)]['capacity_Pvar']
    availableCapa['availableCapa']=availableCapa['maxCapa']-elecProd.loc[(slice(None),slice(None),export_TECH)]['power_Dvar']
    availableCapa.loc[availableCapa['availableCapa'] < 0]=0

    marketPrice['LastCalled'] = ""

    for i in marketPrice.index:
        if elecProd.loc[(i[0], i[1],'IntercoIn')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'IntercoIn'
        elif elecProd.loc[(i[0], i[1], 'Coal_p')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'Coal_p'
        elif elecProd.loc[(i[0], i[1], 'TAC')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'TAC'
        elif elecProd.loc[(i[0], i[1], 'CCG')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'CCG'
        elif elecProd.loc[(i[0], i[1], 'TAC_H2')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'TAC_H2'
        elif elecProd.loc[(i[0], i[1], 'CCG_H2')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'CCG_H2'
        elif elecProd.loc[(i[0], i[1], 'OldNuke')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'OldNuke'
        elif elecProd.loc[(i[0], i[1], 'NewNuke')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'NewNuke'
        elif elecProd.loc[(i[0], i[1], 'HydroRiver')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'HydroRiver'
        elif elecProd.loc[(i[0], i[1], 'WindOffShore')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'WindOffShore'
        elif elecProd.loc[(i[0], i[1], 'WindOnShore')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'WindOnShore'
        elif elecProd.loc[(i[0], i[1], 'Solar')]['power_Dvar'] > 0:
            marketPrice.LastCalled.loc[i] = 'Solar'
        else:
            marketPrice.LastCalled.loc[i] = 'Undetermined'

    marketPrice = round(marketPrice.reset_index().set_index(['YEAR_op', 'TIMESTAMP']), 2)

    marketPrice.to_csv(outputFolder + '/marketPrice.csv')

    return marketPrice,elecProd
