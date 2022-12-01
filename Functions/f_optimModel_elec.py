from pyomo.environ import *
from pyomo.core import *
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mosek
from Functions.f_optimization import *

def GetElectricPriceModel(elecProd, marketPrice,ResParameters,TechParameters,capaCosts, carbonContent,conversionFactor,carbonTax,isAbstract=False):
    """
    This function creates the pyomo model and initlize the Parameters and (pyomo) Set values
    :param areaConsumption: panda table with consumption
    :param availabilityFactor: panda table
    :param isAbstract: boolean true is the model should be abstract. ConcreteModel otherwise
    :return: pyomo model
    """
    #TechParameters = inputDict['techParameters']
    #carbonTax=inputDict['carbonTax']
    #conversionFactor=inputDict['conversionFactor']

    isAbstract=False

    ### obtaining dimensions values

    TECHNOLOGIES = set(elecProd.index.get_level_values('TECHNOLOGIES').unique())
    TIMESTAMP = set(marketPrice.index.get_level_values('TIMESTAMP').unique())
    YEAR = set.union(set(TechParameters.index.get_level_values('YEAR').unique()),set(marketPrice.index.get_level_values('YEAR_op').unique()))
    RESOURCES = set(ResParameters.index.get_level_values('RESOURCES').unique())

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
    model.RESOURCES = Set(initialize=RESOURCES, ordered=False)
    model.TIMESTAMP = Set(initialize=TIMESTAMP, ordered=False)
    model.YEAR = Set(initialize=YEAR, ordered=False)
    model.YEAR_TIMESTAMP = model.YEAR * model.TIMESTAMP
    model.YEAR_TIMESTAMP_TECHNOLOGIES = model.YEAR * model.TIMESTAMP * model.TECHNOLOGIES
    model.YEAR_TECHNOLOGIES = model.YEAR  * model.TECHNOLOGIES
    model.YEAR_TIMESTAMP_RESOURCES = model.YEAR * model.TIMESTAMP * model.RESOURCES

    YEAR_list=sorted(list(YEAR))
    dy = YEAR_list[1] - YEAR_list[0]
    model.YEAR_invest = Set(initialize=YEAR_list[: len(YEAR_list) - 1], ordered=False)
    model.YEAR_op = Set(initialize=YEAR_list[len(YEAR_list) - (len(YEAR_list) - 1):], ordered=False)

    ###############
    # Parameters ##
    ###############

    model.marketPrice = Param(model.YEAR_TIMESTAMP, default=0,
                                  initialize=marketPrice.loc[:, "energyCtr"].squeeze().to_dict(), domain=Any)
    model.LastCalled = Param(model.YEAR_TIMESTAMP,
                              initialize=marketPrice.loc[:, "LastCalled"].squeeze().to_dict(), domain=Any)
    model.elecProd = Param(model.YEAR_TIMESTAMP_TECHNOLOGIES, default=0,
                                  initialize=elecProd.loc[:, "power_Dvar"].squeeze().to_dict(), domain=Any)
    model.capaCosts = Param(model.YEAR_TECHNOLOGIES, default=0,
                       initialize=capaCosts.loc[:, "capacityCosts_Pvar"].squeeze().to_dict(), domain=Any)
    model.importCosts = Param(model.YEAR_TIMESTAMP_RESOURCES, default=0,
                       initialize=ResParameters.loc[:, "importCost"].squeeze().to_dict(), domain=Any)
    model.carbonContent = Param(model.YEAR_TIMESTAMP, default=0,
                       initialize=carbonContent.loc[:, "carbonContent"].squeeze().to_dict(), domain=Any)
    model.varCosts = Param(model.YEAR_TECHNOLOGIES, default=0,
                       initialize=TechParameters.loc[:, "powerCost"].squeeze().to_dict(), domain=Any)
    model.carbon_taxe = Param(model.YEAR, default=0,initialize=carbonTax.loc[:,'carbonTax'].squeeze().to_dict(), domain=Any)
    model.conversionFactor = Param(model.RESOURCES,model.TECHNOLOGIES,default=0,initialize=conversionFactor.loc[:,"conversionFactor"].squeeze().to_dict(),domain=Any)

    ################
    # Variables    #
    ################

    model.AjustFac = Var(model.YEAR_op, model.TECHNOLOGIES,domain=NonNegativeReals,initialize=0)
    model.Revenus = Var(model.YEAR_op,model.TECHNOLOGIES,domain=NonNegativeReals)
    model.TotalCosts = Var(model.YEAR_op,model.TECHNOLOGIES,domain=NonNegativeReals)

    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    model.slack = Suffix(direction=Suffix.IMPORT)

    ########################
    # Objective Function   #
    ########################

    def ObjectiveFunction_rule(model):  # OBJ
        return sum(model.Revenus[y,tech] for y,tech in zip(model.YEAR_op,model.TECHNOLOGIES))
    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)

    #################
    # Constraints   #
    #################

    # Revenus definition Constraints
    def RevenusDef_rule(model,y,tech):
        return sum(model.elecProd[y,t,tech]*(model.marketPrice[y,t]+model.AjustFac[y,model.LastCalled[y,t]]) for t in model.TIMESTAMP) == model.Revenus[y,tech]
    model.RevenusCtr = Constraint(model.YEAR_op,model.TECHNOLOGIES, rule=RevenusDef_rule)

    # TotalCosts definition Constraints
    def TotalCostsDef_rule(model,y,tech):
        if tech in ['IntercoOut','IntercoIn'] :
            return model.TotalCosts[y,tech] == 0
        else :
            return sum(sum(model.elecProd[y,t,tech]*(-model.conversionFactor[res,tech]*model.importCosts[y,t,res]) for res in ['uranium','gaz','hydrogen',]) + model.elecProd[y,t,tech]*model.varCosts[y-dy,tech]+ model.elecProd[y,t,tech]*model.carbonContent[y,t]*model.carbon_taxe[y] for t in model.TIMESTAMP) + model.capaCosts[y,tech] == model.TotalCosts[y,tech]
    model.TotalCostsCtr = Constraint(model.YEAR_op,model.TECHNOLOGIES, rule=TotalCostsDef_rule)

    def AjustDef_rule(model,y,tech):
        if tech in ['IntercoOut','IntercoIn','curtailment']:
            return model.AjustFac[y, tech] == 0
        elif tech in 'Coal_p':
            return Constraint.Skip
        elif sum(model.elecProd[y,t,tech] for t in TIMESTAMP) == 0:
            return model.AjustFac[y,tech]==0
        else :
            return model.Revenus[y,tech] >= model.TotalCosts[y,tech]
    model.AjustCtr = Constraint(model.YEAR_op,model.TECHNOLOGIES, rule=AjustDef_rule)

    def meritDef_rule(model,y,tech):
        if tech == 'Solar' :
            return model.varCosts[y-dy,tech] + model.AjustFac[y,tech] <= model.varCosts[y-dy,'WindOnShore'] + model.AjustFac[y,'WindOnShore']
        if tech == 'WindOnShore' :
            return  model.varCosts[y-dy,tech] + model.AjustFac[y,tech] <= model.varCosts[y-dy,'WindOffShore'] + model.AjustFac[y,'WindOffShore']
        if tech == 'WindOffShore' :
            return  model.varCosts[y-dy,tech] + model.AjustFac[y,tech] <= sum((-model.conversionFactor[res, 'NewNuke']) * model.importCosts[y, 1, res] for res in['uranium', 'gaz', 'hydrogen', ]) + model.varCosts[y - dy, 'NewNuke'] + model.AjustFac[y, 'NewNuke']
        if tech == 'NewNuke' :
            return sum((-model.conversionFactor[res, 'NewNuke']) * model.importCosts[y, 1, res] for res in ['uranium','gaz','hydrogen',]) + model.varCosts[y-dy,'NewNuke'] + model.AjustFac[y,'NewNuke'] <= sum((-model.conversionFactor[res, 'OldNuke']) * model.importCosts[y, 1, res] for res in ['uranium','gaz','hydrogen',]) + model.varCosts[y-dy,'OldNuke'] + model.AjustFac[y,'OldNuke']
        if tech == 'OldNuke' :
            return sum((-model.conversionFactor[res, 'OldNuke']) * model.importCosts[y, 1, res] for res in ['uranium','gaz','hydrogen',]) + model.varCosts[y-dy,'OldNuke'] + model.AjustFac[y,'OldNuke'] <= sum((-model.conversionFactor[res, 'CCG']) * model.importCosts[y, 1, res] for res in ['uranium','gaz','hydrogen',]) + model.varCosts[y-dy,'CCG'] + model.AjustFac[y,'CCG']
        if tech == 'CCG' :
            return sum((-model.conversionFactor[res, 'CCG']) * model.importCosts[y, 1, res] for res in ['uranium','gaz','hydrogen',]) + model.varCosts[y-dy,'CCG'] + model.AjustFac[y,'CCG'] <= sum((-model.conversionFactor[res, 'TAC']) * model.importCosts[y, 1, res] for res in ['uranium','gaz','hydrogen',]) + model.varCosts[y-dy,'TAC'] + model.AjustFac[y,'TAC']
        else :
            return Constraint.Skip
    model.meritCtr = Constraint(model.YEAR_op,model.TECHNOLOGIES, rule=meritDef_rule)

    def Lim_rule(model,y,tech):
        return model.AjustFac[y,tech] <= 80
    model.LimCtr = Constraint(model.YEAR_op,model.TECHNOLOGIES, rule=Lim_rule)

    return model
