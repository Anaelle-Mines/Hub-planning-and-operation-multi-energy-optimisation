from pyomo.environ import *
from pyomo.core import *
from dynprogstorage.Wrapper_dynprogstorage import Pycplfunction, Pycplfunctionvec
from dynprogstorage.wrappers import *
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mosek
from Functions.f_optimization import *

def loadingParameters(Selected_TECHNOLOGIES = ['OldNuke', 'Solar', 'WindOnShore', 'HydroReservoir', 'HydroRiver', 'TAC', 'CCG', 'pac','electrolysis'],InputFolder = 'Data/Input/',Zones = "FR",year = 2013):

    #### reading CSV files
    areaConsumption = pd.read_csv(InputFolder + 'areaConsumption' + str(year) + '_' + str(Zones) + '_WithH2.csv',
                                  sep=',', decimal='.', skiprows=0).set_index(["TIMESTAMP", "RESSOURCES"])
    availabilityFactor = pd.read_csv(InputFolder + 'availabilityFactor' + str(year) + '_' + str(Zones) + '.csv',
                                     sep=',', decimal='.', skiprows=0).set_index(["TIMESTAMP", "TECHNOLOGIES"])
    TechParameters = pd.read_csv(InputFolder + 'Planing-RAMP2_TECHNOLOGIES.csv', sep=',', decimal='.', skiprows=0,
                                 comment="#").set_index(["TECHNOLOGIES"])
    conversionFactor = pd.read_csv(InputFolder + 'Ressources_conversionFactors.csv', sep=',', decimal='.', skiprows=0,
                                   comment="#").set_index(["RESSOURCES", "TECHNOLOGIES"])
    ResParameters = pd.read_csv(InputFolder + 'Ressources_set.csv', sep=',', decimal='.', skiprows=0,
                                comment="#").set_index(["RESSOURCES"])

    #### Selection of subset
    availabilityFactor = availabilityFactor.loc[(slice(None), Selected_TECHNOLOGIES), :]
    conversionFactor = conversionFactor.loc[(slice(None), Selected_TECHNOLOGIES), :]
    TechParameters = TechParameters.loc[Selected_TECHNOLOGIES, :]
    TechParameters.loc["OldNuke", 'RampConstraintMoins'] = 0.01  ## a bit strong to put in light the effect
    TechParameters.loc["OldNuke", 'RampConstraintPlus'] = 0.02  ## a bit strong to put in light the effect
    return areaConsumption,availabilityFactor,TechParameters,conversionFactor,ResParameters

def My_GetElectricSystemModel_PlaningSingleNode_MultiRessources(areaConsumption, availabilityFactor, TechParameters, ResParameters,
                                                conversionFactor, isAbstract=False):


    availabilityFactor.isna().sum()

    Carbontax=0 # €/kgCO2

    ### Cleaning
    availabilityFactor = availabilityFactor.fillna(method='pad');
    areaConsumption = areaConsumption.fillna(method='pad');
    conversionFactor = conversionFactor.fillna(method='pad');

    ### obtaining dimensions values
    TECHNOLOGIES = set(TechParameters.index.get_level_values('TECHNOLOGIES').unique())
    RESSOURCES = set(ResParameters.index.get_level_values('RESSOURCES').unique())
    TIMESTAMP = set(areaConsumption.index.get_level_values('TIMESTAMP').unique())
    TIMESTAMP_list = areaConsumption.index.get_level_values('TIMESTAMP').unique()

    #####################
    #    Pyomo model    #
    #####################

    if (isAbstract):
        model = AbstractModel()
    else:
        model = ConcreteModel()

    ###############
    # Sets       ##
    ###############
    model.TECHNOLOGIES = Set(initialize=TECHNOLOGIES, ordered=False)
    model.RESSOURCES = Set(initialize=RESSOURCES, ordered=False)
    model.TIMESTAMP = Set(initialize=TIMESTAMP, ordered=False)
    model.TIMESTAMP_TECHNOLOGIES = model.TIMESTAMP * model.TECHNOLOGIES
    model.TIMESTAMP_RESSOURCES = model.TIMESTAMP * model.RESSOURCES
    model.RESSOURCES_TECHNOLOGIES = model.RESSOURCES * model.TECHNOLOGIES

    # Subset of Simple only required if ramp constraint
    model.TIMESTAMP_MinusOne = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 1], ordered=False)
    model.TIMESTAMP_MinusThree = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 3], ordered=False)

    ###############
    # Parameters ##
    ###############

    model.areaConsumption =     Param(model.TIMESTAMP_RESSOURCES, mutable=True,default=0,
                                      initialize=areaConsumption.loc[:,"areaConsumption"].squeeze().to_dict(), domain=Any)
    model.availabilityFactor =  Param( model.TIMESTAMP_TECHNOLOGIES, domain=PercentFraction,default=1,
                                      initialize=availabilityFactor.loc[:,"availabilityFactor"].squeeze().to_dict())
    model.conversionFactor = Param(model.RESSOURCES_TECHNOLOGIES, default=0,
                                   initialize=conversionFactor.loc[:, "conversionFactor"].squeeze().to_dict())

    # with test of existing columns on TechParameters
    for COLNAME in TechParameters:
        if COLNAME not in ["TECHNOLOGIES", "AREAS"]:  ### each column in TechParameters will be a parameter
            exec("model." + COLNAME + " = Param(model.TECHNOLOGIES, default=0," +
                 "initialize=TechParameters." + COLNAME + ".squeeze().to_dict())")
    ## manière générique d'écrire pour toutes les colomnes COL de TechParameters quelque chose comme
    #    model.COLNAME =          Param(model.TECHNOLOGIES, domain=NonNegativeReals,default=0,
    #                                 initialize=TechParameters.COLNAME.squeeze().to_dict())
    for COLNAME in ResParameters:
        if COLNAME not in ["RESSOURCES"]:  ### each column in ResParameters will be a parameter
            exec("model." + COLNAME + " = Param(model.RESSOURCES, default=0," +
                 "initialize=ResParameters." + COLNAME + ".squeeze().to_dict())")

    ################
    # Variables    #
    ################

    model.power = Var(model.TIMESTAMP, model.TECHNOLOGIES,domain=NonNegativeReals)  # Instant power for a conversion mean at t
    model.powerCosts = Var(model.TECHNOLOGIES)  ### Marginal cost for a conversion mean, explicitely defined by definition powerCostsDef
    model.capacityCosts = Var(model.TECHNOLOGIES)  ### Fixed costs for a conversion mean, explicitely defined by definition capacityCostsDef
    model.capacity = Var(model.TECHNOLOGIES, domain=NonNegativeReals)  ### Installed capacity for a conversion mean
    model.importCosts = Var(model.RESSOURCES)  ### Cost of ressource imported, explicitely defined by definition importCostsDef

    model.importation = Var(model.TIMESTAMP, model.RESSOURCES, domain=NonNegativeReals)
    model.energy = Var(model.TIMESTAMP, model.RESSOURCES)  ### Variation of ressource r at time t

    model.carbonCosts = Var(model.TECHNOLOGIES, domain=NonNegativeReals)  ### Carbon emission costs for a conversion mean, explicitly defined by powerCostsDef

    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    model.slack = Suffix(direction=Suffix.IMPORT)

    ########################
    # Objective Function   #
    ########################

    def ObjectiveFunction_rule(model):  # OBJ
        return sum(model.powerCosts[tech] + model.capacityCosts[tech] + model.carbonCosts[tech] for tech in model.TECHNOLOGIES) + sum(model.importCosts[res] for res in model.RESSOURCES)
    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)

    #################
    # Constraints   #
    #################

    # energyCosts definition Constraints
    def powerCostsDef_rule(model,tech):  # EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in TIMESTAMP} energyCost[tech]*energy[t,tech];
        temp1 = model.powerCost[tech]
        return sum(temp1 * model.power[t, tech] for t in model.TIMESTAMP) == model.powerCosts[tech]
    model.powerCostsCtr = Constraint(model.TECHNOLOGIES, rule=powerCostsDef_rule)

    def carbonCostsDef_rule(model,tech):  # EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in TIMESTAMP} energyCost[tech]*energy[t,tech];
        temp2 = model.EmissionCO2[tech]
        return sum(temp2 * model.power[t, tech] * Carbontax for t in model.TIMESTAMP) == model.carbonCosts[tech]
    model.carbonCostsCtr = Constraint(model.TECHNOLOGIES, rule=carbonCostsDef_rule)

    # capacityCosts definition Constraints
    def capacityCostsDef_rule(model,tech):  # EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in TIMESTAMP} energyCost[tech]*energy[t,tech];
        temp = model.capacityCost[tech]
        return temp * model.capacity[tech] == model.capacityCosts[tech]
    model.capacityCostsCtr = Constraint(model.TECHNOLOGIES, rule=capacityCostsDef_rule)

    # importCosts definition Constraints
    def importCostsDef_rule(model,res):  # EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in TIMESTAMP} energyCost[tech]*energy[t,tech];
        temp = model.importCost[res]
        return sum(temp * model.importation[t, res] for t in model.TIMESTAMP) == model.importCosts[res]
    model.importCostsCtr = Constraint(model.RESSOURCES, rule=importCostsDef_rule)

     # gaz volume Constraints
    def Volume_rule(model):  # INEQ for gaz
        return sum(model.importation[t, 'gaz'] for t in model.TIMESTAMP) <= 100000000
    model.VolumeCtr = Constraint(rule=Volume_rule)

    # Capacity constraint
    def Capacity_rule(model, t, tech):  # INEQ forall t, tech
        return model.capacity[tech] * model.availabilityFactor[t, tech] >= model.power[t, tech]
    model.CapacityCtr = Constraint(model.TIMESTAMP, model.TECHNOLOGIES, rule=Capacity_rule)

    # Ressource production constraint
    def Production_rule(model, t, res):  # EQ forall t, res
        return sum(model.power[t, tech] * model.conversionFactor[res, tech] for tech in model.TECHNOLOGIES) + model.importation[t, res] == model.energy[t, res]
    model.ProductionCtr = Constraint(model.TIMESTAMP, model.RESSOURCES, rule=Production_rule)

    # Contrainte d'equilibre offre/demande pour l'électricité
    def energyCtr_rule(model, t, res):  # INEQ forall t,res
        if res == 'electricity':
            return model.energy[t, res] >= model.areaConsumption[t, res]
        else:
            return Constraint.Skip
    model.energyCtr = Constraint(model.TIMESTAMP, model.RESSOURCES, rule=energyCtr_rule)

    # Contrainte d'équilibre offre/demande pour les ressources stockables
    def annualEnergyCtr_rule(model, res):   # INEQ forall res
        if res == 'electricity':
            return Constraint.Skip
        else:
            return sum(model.energy[t, res] for t in TIMESTAMP) >= sum(model.areaConsumption[t, res] for t in TIMESTAMP)
    model.annualEnergyCtr = Constraint(model.RESSOURCES, rule=annualEnergyCtr_rule)

    # Contrainte de capacité de stockage
    def StorageCtr_rule(model,t, res):   # INEQ forall res
        if res == 'electricity' :
            return Constraint.Skip
        else:
            return  model.energy[t, res] - model.areaConsumption[t,res] <= 1000000
    model.StorageCtr = Constraint(model.TIMESTAMP,model.RESSOURCES,rule=StorageCtr_rule)

    if "maxCapacity" in TechParameters:
        def maxCapacity_rule(model, tech):  # INEQ forall t, tech
            if model.maxCapacity[tech] > 0:
                return model.maxCapacity[tech] >= model.capacity[tech]
            else:
                return Constraint.Skip
        model.maxCapacityCtr = Constraint(model.TECHNOLOGIES, rule=maxCapacity_rule)

    if "minCapacity" in TechParameters:
        def minCapacity_rule(model, tech):  # INEQ forall t, tech
            if model.minCapacity[tech] > 0:
                return model.minCapacity[tech] <= model.capacity[tech]
            else:
                return Constraint.Skip

        model.minCapacityCtr = Constraint(model.TECHNOLOGIES, rule=minCapacity_rule)

    if "EnergyNbhourCap" in TechParameters:
        def storage_rule(model, tech):  # INEQ forall t, tech
            if model.EnergyNbhourCap[tech] > 0:
                return model.EnergyNbhourCap[tech] * model.capacity[tech] >= sum(
                    model.power[t, tech] for t in model.TIMESTAMP)
            else:
                return Constraint.Skip

        model.storageCtr = Constraint(model.TECHNOLOGIES, rule=storage_rule)

    if "RampConstraintPlus" in TechParameters:
        def rampCtrPlus_rule(model, t, tech):  # INEQ forall t<
            if model.RampConstraintPlus[tech] > 0:
                return model.power[t + 1, tech] - model.power[t, tech] <= model.capacity[tech] * \
                       model.RampConstraintPlus[tech];
            else:
                return Constraint.Skip

        model.rampCtrPlus = Constraint(model.TIMESTAMP_MinusOne, model.TECHNOLOGIES, rule=rampCtrPlus_rule)

    if "RampConstraintMoins" in TechParameters:
        def rampCtrMoins_rule(model, t, tech):  # INEQ forall t<
            if model.RampConstraintMoins[tech] > 0:
                return model.power[t + 1, tech] - model.power[t, tech] >= - model.capacity[tech] * \
                       model.RampConstraintMoins[tech];
            else:
                return Constraint.Skip

        model.rampCtrMoins = Constraint(model.TIMESTAMP_MinusOne, model.TECHNOLOGIES, rule=rampCtrMoins_rule)

    return model;


def My_GetElectricSystemModel_PlaningSingleNode_MultiRessources_With1Storage(areaConsumption, availabilityFactor, TechParameters, ResParameters,
                                                conversionFactor,StorageParameters,tol,n,solver="mosek"):

    model = My_GetElectricSystemModel_PlaningSingleNode_MultiRessources(areaConsumption, availabilityFactor,TechParameters, ResParameters, conversionFactor)
    opt = SolverFactory(solver)

    ##### Loop
    PrixTotal = {}
    Consommation = {}
    LMultipliers = {}
    DeltaPrix = {}
    Deltazz = {}
    CostFunction = {}
    TotalCols = {}
    zz = {}
    # p_max_out=100.; p_max_in=100.; c_max=10.;

    areaConsumption["NewConsumption"] = areaConsumption["areaConsumption"]

    nbTime = len(areaConsumption["areaConsumption"])
    cpt = 0
    for i in model.areaConsumption:
        if i[1] == 'electricity':
            model.areaConsumption[i] = areaConsumption.NewConsumption[i]

    DeltaPrix_ = tol + 1
    while ((DeltaPrix_ > tol) & (n > cpt)):
        print(cpt)
        if (cpt == 0):
            zz[cpt] = [0] * nbTime
        else:
            zz[cpt] = areaConsumption["Storage"].tolist()

        # if solver=="mosek" :
        #    results = opt.solve(model, options= {"dparam.optimizer_max_time":  100.0, "iparam.outlev" : 2,"iparam.optimizer":     mosek.optimizertype.primal_simplex},tee=True)
        # else :
        # if (solver == 'cplex') | (solver == 'cbc'):
        #   results = opt.solve(model, warmstart=True)
        # else:
        results = opt.solve(model)
        Constraints = getConstraintsDual_panda(model)
        # if solver=='cbc':
        #    Variables = getVariables_panda(model)['energy'].set_index(['TIMESTAMP','TECHNOLOGIES'])
        #    for i in model.energy:  model.energy[i] = Variables.energy[i]

        TotalCols[cpt] = getVariables_panda_indexed(model)['powerCosts'].sum()[1] + \
                         getVariables_panda_indexed(model)['importCosts'].sum()[1]
        Prix = Constraints["energyCtr"].assign(Prix=lambda x: x.energyCtr).Prix.to_numpy()
        Prix[Prix <= 0] = 0.0000000001
        valueAtZero = TotalCols[cpt] - Prix * zz[cpt]
        tmpCost = GenCostFunctionFromMarketPrices_dict(Prix, r_in=StorageParameters['efficiency_in'],
                                                       r_out=StorageParameters['efficiency_out'],
                                                       valueAtZero=valueAtZero)
        if (cpt == 0):
            CostFunction[cpt] = GenCostFunctionFromMarketPrices(Prix, r_in=StorageParameters['efficiency_in'],
                                                                r_out=StorageParameters['efficiency_out'],
                                                                valueAtZero=valueAtZero)
        else:
            tmpCost = GenCostFunctionFromMarketPrices_dict(Prix, r_in=StorageParameters['efficiency_in'],
                                                           r_out=StorageParameters['efficiency_out'],
                                                           valueAtZero=valueAtZero)
            tmpCost2 = CostFunction[cpt - 1]
            if StorageParameters['efficiency_in'] * StorageParameters['efficiency_out'] == 1:
                tmpCost2.Maxf_1Breaks_withO(tmpCost['S1'], tmpCost['B1'], tmpCost[
                    'f0'])
            else:
                tmpCost2.Maxf_2Breaks_withO(tmpCost['S1'], tmpCost['S2'], tmpCost['B1'], tmpCost['B2'], tmpCost[
                    'f0'])  ### etape clé, il faut bien comprendre ici l'utilisation du max de deux fonction de coût
            CostFunction[cpt] = tmpCost2
        LMultipliers[cpt] = Prix
        if cpt > 0:
            DeltaPrix[cpt] = sum(abs(LMultipliers[cpt] - LMultipliers[cpt - 1])) / sum(abs(LMultipliers[cpt]))
            if sum(abs(pd.DataFrame(zz[cpt]))) > 0:
                Deltazz[cpt] = sum(abs(pd.DataFrame(zz[cpt]) - pd.DataFrame(zz[cpt - 1]))) / sum(
                    abs(pd.DataFrame(zz[cpt])))
            else:
                Deltazz[cpt] = 0
            DeltaPrix_ = DeltaPrix[cpt]

        areaConsumption.loc[:, "Storage"] = CostFunction[cpt].OptimMargInt(
            [-StorageParameters['p_max'] / StorageParameters['efficiency_out']] * nbTime,
            [StorageParameters['p_max'] * StorageParameters['efficiency_in']] * nbTime,
            [0] * nbTime,
            [StorageParameters['c_max']] * nbTime)

        areaConsumption.loc[areaConsumption.loc[:, "Storage"] > 0, "Storage"] = areaConsumption.loc[
                                                                                    areaConsumption.loc[:,
                                                                                    "Storage"] > 0, "Storage"] / \
                                                                                StorageParameters['efficiency_in']
        areaConsumption.loc[areaConsumption.loc[:, "Storage"] < 0, "Storage"] = areaConsumption.loc[
                                                                                    areaConsumption.loc[:,
                                                                                    "Storage"] < 0, "Storage"] * \
                                                                                StorageParameters['efficiency_out']
        areaConsumption.loc[:, "NewConsumption"] = areaConsumption.loc[:, "areaConsumption"] + areaConsumption.loc[:,
                                                                                               "Storage"]
        for i in model.areaConsumption:
            if i[1] == 'electricity':
                model.areaConsumption[i] = areaConsumption.NewConsumption[i]
        cpt = cpt + 1

    results = opt.solve(model)
    stats = {"DeltaPrix": DeltaPrix, "Deltazz": Deltazz}
    Variables = getVariables_panda(model)  #### obtain optimized variables in panda form

    return results,stats,Variables

def Boucle_SensibiliteAlphaSimple(areaConsumption, availabilityFactor, TechParameters, ResParameters,
                                                conversionFactor,variation_CAPEX_H2,variation_prix_GazNat) :
    CAPEX_electrolysis = TechParameters['capacityCost']['electrolysis']
    CAPEX_pac = TechParameters['capacityCost']['pac']
    Prix_gaz = ResParameters['importCost']['gaz']
    alpha_list = []
    Prod_EnR_list = []
    Prod_elec_list = []
    Prod_gaz_list = []
    Prod_H2_list = []
    Prod_Nuke_list = []
    Conso_gaz_list = []
    Capa_gaz_list = []
    Capa_EnR_list = []
    Capa_electrolysis_list = []
    Capa_PAC_list = []

    for var2 in variation_prix_GazNat:
        ResParameters['importCost']['gaz'] = var2
        for var1 in variation_CAPEX_H2:
            TechParameters['capacityCost']['electrolysis'] = CAPEX_electrolysis * (1 + var1)
            TechParameters['capacityCost']['pac'] = CAPEX_pac * (1 + var1)
            model = My_GetElectricSystemModel_PlaningSingleNode_MultiRessources(areaConsumption, availabilityFactor,
                                                                                TechParameters, ResParameters,
                                                                                conversionFactor)
            solver = 'mosek'
            opt = SolverFactory(solver)
            results = opt.solve(model)
            Variables = getVariables_panda_indexed(model)
            ener = Variables['power'].pivot(index="TIMESTAMP", columns='TECHNOLOGIES', values='power')
            ener = ener.sum(axis=0)
            ener = pd.DataFrame(ener, columns={'energy'})
            alpha = ener.loc['pac'] / (ener.loc['CCG'] + ener.loc['TAC'] + ener.loc['pac'])
            alpha_list = alpha_list + [alpha['energy']]
            Prod_elec_list = Prod_elec_list + [ener.sum(axis=0)['energy'] / 1000000]
            Prod_EnR_list = Prod_EnR_list + [(ener.loc['Solar']['energy'] + ener.loc['HydroReservoir']['energy'] +
                                              ener.loc['HydroRiver']['energy'] + ener.loc['WindOnShore'][
                                                  'energy']) / 1000000]
            Prod_gaz_list = Prod_gaz_list + [(ener.loc['CCG']['energy'] + ener.loc['TAC']['energy']) / 1000000]
            Prod_H2_list = Prod_H2_list + [ener.loc['pac']['energy'] / 1000000]
            Prod_Nuke_list = Prod_Nuke_list + [ener.loc['OldNuke']['energy'] / 1000000]
            Conso_gaz_list = Conso_gaz_list + [
                Variables['importation'].loc[Variables['importation']['RESSOURCES'] == 'gaz'].sum(axis=0)[
                    'importation'] / 1000000]
            capa = Variables['capacity'].set_index('TECHNOLOGIES')
            Capa_gaz_list = Capa_gaz_list + [(capa.loc['TAC']['capacity'] + capa.loc['CCG']['capacity']) / 1000]
            Capa_EnR_list = Capa_EnR_list + [(capa.loc['Solar']['capacity'] + capa.loc['HydroReservoir']['capacity'] +
                                              capa.loc['HydroRiver']['capacity'] + capa.loc['WindOnShore'][
                                                  'capacity']) / 1000]
            Capa_electrolysis_list = Capa_electrolysis_list + [capa.loc['electrolysis']['capacity'] / 1000]
            Capa_PAC_list = Capa_PAC_list + [capa.loc['pac']['capacity'] / 1000]
            print(alpha_list)

    ### récupérer dataframe à partir de la liste des résultats

    PrixGaz_list = []
    CAPEX_list = []
    CAPEX_H2 = []
    for var1 in variation_CAPEX_H2:
        CAPEX_H2.append(round(CAPEX_electrolysis * (1 + var1) + CAPEX_pac * (1 + var1), 1))

    for i in variation_prix_GazNat:
        for j in CAPEX_H2:
            PrixGaz_list.append(i)
            CAPEX_list.append(j)

    alpha_df = pd.DataFrame()
    alpha_df['PrixGaz'] = PrixGaz_list
    alpha_df['Capex'] = CAPEX_list
    alpha_df['value'] = alpha_list
    alpha_df['Prod_elec'] = Prod_elec_list
    alpha_df['Prod_EnR'] = Prod_EnR_list
    alpha_df['Prod_gaz'] = Prod_gaz_list
    alpha_df['Prod_H2'] = Prod_H2_list
    alpha_df['Prod_Nuke'] = Prod_Nuke_list
    alpha_df['Conso_gaz'] = Conso_gaz_list
    alpha_df['Capa_gaz'] = Capa_gaz_list
    alpha_df['Capa_EnR'] = Capa_EnR_list
    alpha_df['Capa_electrolysis'] = Capa_electrolysis_list
    alpha_df['Capa_PAC'] = Capa_PAC_list

    return alpha_df


def SensibiliteAlphaSimple(Variations, solver = 'mosek') :

    VariationPrixGaz = Variations["variation_prix_GazNat"]
    VariationCAPEX = Variations["variation_CAPEX_H2"]

    areaConsumption,availabilityFactor, TechParameters, conversionFactor, ResParameters = loadingParameters()

    ResParameters.loc['gaz','importCost'] = VariationPrixGaz.squeeze()
    TechParameters.loc['electrolysis','capacityCost'] = TechParameters.loc['electrolysis','capacityCost'] * (1 + VariationCAPEX.squeeze())
    TechParameters.loc['pac','capacityCost'] = TechParameters.loc['pac','capacityCost'] * (1 + VariationCAPEX.squeeze())
    model = My_GetElectricSystemModel_PlaningSingleNode_MultiRessources(areaConsumption, availabilityFactor,
                                                                        TechParameters, ResParameters,
                                                                        conversionFactor)
    #Resultat= pd.DataFrame()
    opt = SolverFactory(solver)
    results = opt.solve(model)
    Variables = getVariables_panda_indexed(model)
    Data=Variables['power'].set_index('TECHNOLOGIES')
    DemRes=areaConsumption.reset_index()[areaConsumption.reset_index().RESSOURCES=='electricity'].set_index('TIMESTAMP').drop('RESSOURCES',axis=1)
    DemRes.loc[:,'areaConsumption']=DemRes.loc[:,'areaConsumption']\
                                    -Data.loc['HydroReservoir'].set_index('TIMESTAMP')['power']\
                                    -Data.loc['WindOnShore'].set_index('TIMESTAMP')['power']\
                                    -Data.loc['OldNuke'].set_index('TIMESTAMP')['power']\
                                    -Data.loc['Solar'].set_index('TIMESTAMP')['power']
    DemResMax=DemRes.max()/(10**6)
    Production = (Variables['power'].groupby('TECHNOLOGIES').agg({"power" : "sum"})/(10**6)).rename_axis(None, axis = 0).transpose()
    Capacity = (Variables['capacity'].set_index('TECHNOLOGIES') / (10 ** 3)).rename_axis(None, axis=0).transpose()
    Importation = Variables['importation'].groupby('RESSOURCES').agg({"importation" : "sum"})/(10**6)
    alpha = Production.pac['power'] / (Production.CCG['power'] + Production.TAC['power'] + Production.pac['power'])
    Capacity.columns=[x+'_Capa' for x in list(Capacity.columns)]
    Capacity.reset_index(drop=True,inplace=True)
    Production.columns = [x + '_Prod' for x in list(Production.columns)]
    Production.reset_index(drop=True, inplace=True)
    Resultat=Capacity.join(Production)
    Resultat[['gaz_Conso','alpha','Capex','PrixGaz','DemResMax']]=[Importation.loc['gaz','importation'],alpha,TechParameters.loc['electrolysis','capacityCost']+TechParameters.loc['pac','capacityCost'],VariationPrixGaz.squeeze(),DemResMax]

    return Resultat



def Boucle_SensibiliteAlphaSimple_With1Storage(areaConsumption, availabilityFactor, TechParameters, ResParameters,
                                                conversionFactor,variation_CAPEX_H2,variation_prix_GazNat,StorageParameters,tol,n) :

    CAPEX_electrolysis = TechParameters['capacityCost']['electrolysis']
    CAPEX_pac = TechParameters['capacityCost']['pac']
    Prix_gaz = ResParameters['importCost']['gaz']
    alpha_list = []
    Prod_EnR_list = []
    Prod_elec_list = []
    Prod_gaz_list = []
    Prod_H2_list = []
    Prod_Nuke_list = []
    Conso_gaz_list = []
    Capa_gaz_list = []
    Capa_EnR_list = []
    Capa_electrolysis_list = []
    Capa_PAC_list = []

    for var2 in variation_prix_GazNat:
        ResParameters['importCost']['gaz'] = var2
        for var1 in variation_CAPEX_H2:
            TechParameters['capacityCost']['electrolysis'] = CAPEX_electrolysis * (1 + var1)
            TechParameters['capacityCost']['pac'] = CAPEX_pac * (1 + var1)
            Variables = My_GetElectricSystemModel_PlaningSingleNode_MultiRessources_With1Storage(areaConsumption, availabilityFactor, TechParameters, ResParameters,
                                                conversionFactor,StorageParameters,tol,n)[2]
            ener = Variables['power'].pivot(index="TIMESTAMP", columns='TECHNOLOGIES', values='power')
            ener = ener.sum(axis=0)
            ener = pd.DataFrame(ener, columns={'energy'})
            alpha = ener.loc['pac'] / (ener.loc['CCG'] + ener.loc['TAC'] + ener.loc['pac'])
            alpha_list = alpha_list + [alpha['energy']]
            Prod_elec_list = Prod_elec_list + [ener.sum(axis=0)['energy'] / 1000000]
            Prod_EnR_list = Prod_EnR_list + [(ener.loc['Solar']['energy'] + ener.loc['HydroReservoir']['energy'] +
                                              ener.loc['HydroRiver']['energy'] + ener.loc['WindOnShore'][
                                                  'energy']) / 1000000]
            Prod_gaz_list = Prod_gaz_list + [(ener.loc['CCG']['energy'] + ener.loc['TAC']['energy']) / 1000000]
            Prod_H2_list = Prod_H2_list + [ener.loc['pac']['energy'] / 1000000]
            Prod_Nuke_list = Prod_Nuke_list + [ener.loc['OldNuke']['energy'] / 1000000]
            Conso_gaz_list = Conso_gaz_list + [
                Variables['importation'].loc[Variables['importation']['RESSOURCES'] == 'gaz'].sum(axis=0)[
                    'importation'] / 1000000]
            capa = Variables['capacity'].set_index('TECHNOLOGIES')
            Capa_gaz_list = Capa_gaz_list + [(capa.loc['TAC']['capacity'] + capa.loc['CCG']['capacity']) / 1000]
            Capa_EnR_list = Capa_EnR_list + [(capa.loc['Solar']['capacity'] + capa.loc['HydroReservoir']['capacity'] +
                                              capa.loc['HydroRiver']['capacity'] + capa.loc['WindOnShore'][
                                                  'capacity']) / 1000]
            Capa_electrolysis_list = Capa_electrolysis_list + [capa.loc['electrolysis']['capacity'] / 1000]
            Capa_PAC_list = Capa_PAC_list + [capa.loc['pac']['capacity'] / 1000]
            print(alpha_list)

    ### récupérer dataframe à partir de la liste des résultats

    PrixGaz_list = []
    CAPEX_list = []
    CAPEX_H2 = []
    for var1 in variation_CAPEX_H2:
        CAPEX_H2.append(round(CAPEX_electrolysis * (1 + var1) + CAPEX_pac * (1 + var1), 1))

    for i in variation_prix_GazNat:
        for j in CAPEX_H2:
            PrixGaz_list.append(i)
            CAPEX_list.append(j)

    alpha_df = pd.DataFrame()
    alpha_df['PrixGaz'] = PrixGaz_list
    alpha_df['Capex'] = CAPEX_list
    alpha_df['value'] = alpha_list
    alpha_df['Prod_elec'] = Prod_elec_list
    alpha_df['Prod_EnR'] = Prod_EnR_list
    alpha_df['Prod_gaz'] = Prod_gaz_list
    alpha_df['Prod_H2'] = Prod_H2_list
    alpha_df['Prod_Nuke'] = Prod_Nuke_list
    alpha_df['Conso_gaz'] = Conso_gaz_list
    alpha_df['Capa_gaz'] = Capa_gaz_list
    alpha_df['Capa_EnR'] = Capa_EnR_list
    alpha_df['Capa_electrolysis'] = Capa_electrolysis_list
    alpha_df['Capa_PAC'] = Capa_PAC_list

    return alpha_df

def SensibiliteAlpha_WithStorage(Variations) :

    VariationPrixGaz = Variations["variation_prix_GazNat"]
    VariationCAPEX = Variations["variation_CAPEX_H2"]

    areaConsumption,availabilityFactor, TechParameters, conversionFactor, ResParameters = loadingParameters()
    StorageParameters = {"p_max": 5000, "c_max": 50000, "efficiency_in": 0.9, "efficiency_out": 0.9}
    tol = exp(-4)
    n = 10

    ResParameters.loc['gaz','importCost'] = VariationPrixGaz.squeeze()
    TechParameters.loc['electrolysis','capacityCost'] = TechParameters.loc['electrolysis','capacityCost'] * (1 + VariationCAPEX.squeeze())
    TechParameters.loc['pac','capacityCost'] = TechParameters.loc['pac','capacityCost'] * (1 + VariationCAPEX.squeeze())
    results,Stats,Variables = My_GetElectricSystemModel_PlaningSingleNode_MultiRessources_With1Storage(areaConsumption, availabilityFactor, TechParameters, ResParameters,
                                                conversionFactor,StorageParameters,tol,n)
    Data=Variables['power'].set_index('TECHNOLOGIES')
    DemRes=areaConsumption.reset_index()[areaConsumption.reset_index().RESSOURCES=='electricity'].set_index('TIMESTAMP').drop('RESSOURCES',axis=1)
    DemRes.loc[:,'areaConsumption']=DemRes.loc[:,'areaConsumption']\
                                    -Data.loc['HydroReservoir'].set_index('TIMESTAMP')['power']\
                                    -Data.loc['WindOnShore'].set_index('TIMESTAMP')['power']\
                                    -Data.loc['OldNuke'].set_index('TIMESTAMP')['power']\
                                    -Data.loc['Solar'].set_index('TIMESTAMP')['power']
    DemResMax=DemRes.max()/(10**6)
    Production = (Variables['power'].groupby('TECHNOLOGIES').agg({"power" : "sum"})/(10**6)).rename_axis(None, axis = 0).transpose()
    Capacity = (Variables['capacity'].set_index('TECHNOLOGIES') / (10 ** 3)).rename_axis(None, axis=0).transpose()
    Importation = Variables['importation'].groupby('RESSOURCES').agg({"importation" : "sum"})/(10**6)
    alpha = Production.pac['power'] / (Production.CCG['power'] + Production.TAC['power'] + Production.pac['power'])

    Capacity.columns=[x+'_Capa' for x in list(Capacity.columns)]
    Capacity.reset_index(drop=True,inplace=True)
    Production.columns = [x + '_Prod' for x in list(Production.columns)]
    Production.reset_index(drop=True, inplace=True)
    Resultat=Capacity.join(Production)
    Resultat[['gaz_Conso','alpha','Capex','PrixGaz','DemResMax']]=[Importation.loc['gaz','importation'],alpha,TechParameters.loc['electrolysis','capacityCost']+TechParameters.loc['pac','capacityCost'],VariationPrixGaz.squeeze(),DemResMax]

    return Resultat