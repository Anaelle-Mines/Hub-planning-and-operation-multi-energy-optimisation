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
    # model.COLNAME =          Param(model.TECHNOLOGIES, domain=NonNegativeReals,default=0,
    #                                  initialize=TechParameters.set_index([ "TECHNOLOGIES"]).COLNAME.squeeze().to_dict())
    for COLNAME in ResParameters:
        if COLNAME not in ["RESSOURCES"]:  ### each column in ResParameters will be a parameter
            exec("model." + COLNAME + " = Param(model.RESSOURCES, domain=Reals,default=0," +
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