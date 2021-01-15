# %%

# region Importation of modules
import os

if os.path.basename(os.getcwd()) == "Basic functionalities":
    os.chdir('..')  ## to work at project root  like in any IDE

import numpy as np
import pandas as pd
import csv

import datetime
import copy
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn import linear_model
import sys

from functions.f_operationModels import *
from functions.f_optimization import *
from functions.f_graphicalTools import *

# endregion

# region Solver and data location definition
InputFolder = 'Data/Input/'
if sys.platform != 'win32':
    myhost = os.uname()[1]
else:
    myhost = ""
if (myhost == "jupyter-sop"):
    ## for https://jupyter-sop.mines-paristech.fr/ users, you need to
    #  (1) run the following to loanch the license server
    if (os.system(
            "/opt/mosek/9.2/tools/platform/linux64x86/bin/lmgrd -c /opt/mosek/9.2/tools/platform/linux64x86/bin/mosek.lic -l lmgrd.log") == 0):
        os.system("/opt/mosek/9.2/tools/platform/linux64x86/bin/lmutil lmstat -c 27007@127.0.0.1 -a")
    #  (2) definition of license
    os.environ["MOSEKLM_LICENSE_FILE"] = '@jupyter-sop'

BaseSolverPath = '/Users/robin.girard/Documents/Code/Packages/solvers/ampl_macosx64'  ### change this to the folder with knitro ampl ...
## in order to obtain more solver see see https://ampl.com/products/solvers/open-source/
## for eduction this site provides also several professional solvers, that are more efficient than e.g. cbc
sys.path.append(BaseSolverPath)
solvers = ['gurobi', 'knitro', 'cbc']  # try 'glpk', 'cplex'
solverpath = {}
for solver in solvers: solverpath[solver] = BaseSolverPath + '/' + solver
solver = 'mosek'  ## no need for solverpath with mosek.


# endregion

#region Model definition
def My_GetElectricSystemModel_PlaningSingleNode(areaConsumption, availabilityFactor, TechParameters, ResParameters,
                                                conversionFactor, isAbstract=False):
    import pandas as pd
    import numpy as np
    # isAbstract=False
    availabilityFactor.isna().sum()

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

    model.areaConsumption = Param(model.TIMESTAMP_RESSOURCES, domain=NonNegativeReals, default=0,
                                  initialize=areaConsumption.loc[:, "areaConsumption"].squeeze().to_dict())
    model.availabilityFactor = Param(model.TIMESTAMP_TECHNOLOGIES, domain=PercentFraction, default=1,
                                     initialize=availabilityFactor.loc[:, "availabilityFactor"].squeeze().to_dict())
    model.conversionFactor = Param(model.RESSOURCES_TECHNOLOGIES, default=0,
                                   initialize=conversionFactor.loc[:, "conversionFactor"].squeeze().to_dict())

    # with test of existing columns on TechParameters
    for COLNAME in TechParameters:
        if COLNAME not in ["TECHNOLOGIES", "AREAS"]:  ### each column in TechParameters will be a parameter
            exec("model." + COLNAME + " =          Param(model.TECHNOLOGIES, default=0," +
                 "initialize=TechParameters." + COLNAME + ".squeeze().to_dict())")
    ## manière générique d'écrire pour toutes les colomnes COL de TechParameters quelque chose comme
    # model.COLNAME =          Param(model.TECHNOLOGIES, domain=NonNegativeReals,default=0,
    #                                  initialize=TechParameters.set_index([ "TECHNOLOGIES"]).COLNAME.squeeze().to_dict())
    for COLNAME in ResParameters:
        if COLNAME not in ["RESSOURCES"]:  ### each column in ResParameters will be a parameter
            exec("model." + COLNAME + " =          Param(model.RESSOURCES, domain=Reals,default=0," +
                 "initialize=ResParameters." + COLNAME + ".squeeze().to_dict())")

    ################
    # Variables    #
    ################

    model.power = Var(model.TIMESTAMP, model.TECHNOLOGIES,
                      domain=NonNegativeReals)  # Instant power for a conversion mean at t
    model.powerCosts = Var(
        model.TECHNOLOGIES)  ### Marginal cost for a conversion mean, explicitely defined by definition powerCostsDef
    model.capacityCosts = Var(
        model.TECHNOLOGIES)  ### Fixed costs for a conversion mean, explicitely defined by definition capacityCostsDef
    model.capacity = Var(model.TECHNOLOGIES, domain=NonNegativeReals)  ### Installed capacity for a conversion mean
    model.importCosts = Var(
        model.RESSOURCES)  ### Cost of ressource imported, explicitely defined by definition importCostsDef

    model.importation = Var(model.TIMESTAMP, model.RESSOURCES, domain=NonNegativeReals)
    model.energy = Var(model.TIMESTAMP, model.RESSOURCES)  ### Variation of ressource r at time t

    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    model.slack = Suffix(direction=Suffix.IMPORT)

    ########################
    # Objective Function   #
    ########################

    def ObjectiveFunction_rule(model):  # OBJ
        return sum(model.powerCosts[tech] + model.capacityCosts[tech] for tech in model.TECHNOLOGIES) + sum(
            model.importCosts[res] for res in model.RESSOURCES)

    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)

    #################
    # Constraints   #
    #################

    # energyCosts definition Constraints
    def powerCostsDef_rule(model,
                           tech):  # EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in TIMESTAMP} energyCost[tech]*energy[t,tech];
        temp = model.powerCost[tech]
        return sum(temp * model.power[t, tech] for t in model.TIMESTAMP) == model.powerCosts[tech]

    model.powerCostsCtr = Constraint(model.TECHNOLOGIES, rule=powerCostsDef_rule)

    # capacityCosts definition Constraints
    def capacityCostsDef_rule(model,
                              tech):  # EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in TIMESTAMP} energyCost[tech]*energy[t,tech];
        temp = model.capacityCost[tech]
        return temp * model.capacity[tech] == model.capacityCosts[tech]

    model.capacityCostsCtr = Constraint(model.TECHNOLOGIES, rule=capacityCostsDef_rule)

    # importCosts definition Constraints
    def importCostsDef_rule(model,
                            res):  # EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in TIMESTAMP} energyCost[tech]*energy[t,tech];
        temp = model.importCost[res]
        return sum(temp * model.importation[t, res] for t in model.TIMESTAMP) == model.importCosts[res]

    model.importCostsCtr = Constraint(model.RESSOURCES, rule=importCostsDef_rule)

    # Capacity constraint
    def Capacity_rule(model, t, tech):  # INEQ forall t, tech
        return model.capacity[tech] * model.availabilityFactor[t, tech] >= model.power[t, tech]

    model.CapacityCtr = Constraint(model.TIMESTAMP, model.TECHNOLOGIES, rule=Capacity_rule)

    # Ressource production constraint
    def Production_rule(model, t, res):  # EQ forall t, res
        return sum(model.power[t, tech] * model.conversionFactor[res, tech] for tech in model.TECHNOLOGIES) + \
               model.importation[t, res] == model.energy[t, res]

    model.ProductionCtr = Constraint(model.TIMESTAMP, model.RESSOURCES, rule=Production_rule)

    # contrainte d'equilibre offre demande
    def energyCtr_rule(model, t, res):  # INEQ forall t,res
        if res == 'hydrogen':
            return sum(model.energy[t, res] for t in TIMESTAMP) >= sum(model.areaConsumption[t, res] for t in TIMESTAMP)
        else:
            return model.energy[t, res] >= model.areaConsumption[t, res]

    model.energyCtr = Constraint(model.TIMESTAMP, model.RESSOURCES, rule=energyCtr_rule)

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
# endregion

#region II - Ramp Single area : loading parameters loading parameterscase with ramp constraints
Zones="FR"
year=2013
Selected_TECHNOLOGIES=['OldNuke','pac','electrolysis','Solar','WindOnShore','HydroReservoir','HydroRiver']
#### reading CSV files
areaConsumption = pd.read_csv(InputFolder+'areaConsumption'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0).set_index(["TIMESTAMP","RESSOURCES"])
availabilityFactor = pd.read_csv(InputFolder+'availabilityFactor'+str(year)+'_'+str(Zones)+'.csv',
                                sep=',',decimal='.',skiprows=0).set_index(["TIMESTAMP","TECHNOLOGIES"])
TechParameters = pd.read_csv(InputFolder+'Planing-RAMP2_TECHNOLOGIES.csv',sep=',',decimal='.',skiprows=0,comment="#").set_index(["TECHNOLOGIES"])
conversionFactor = pd.read_csv(InputFolder+'Ressources_conversionFactors.csv',sep=',',decimal='.',skiprows=0,comment="#").set_index(["RESSOURCES","TECHNOLOGIES"])
ResParameters = pd.read_csv(InputFolder+'Ressources_set.csv',sep=',',decimal='.',skiprows=0,comment="#").set_index(["RESSOURCES"])

#### Selection of subset
availabilityFactor=availabilityFactor.loc[(slice(None),Selected_TECHNOLOGIES),:]
conversionFactor=conversionFactor.loc[(slice(None),Selected_TECHNOLOGIES),:]
TechParameters=TechParameters.loc[Selected_TECHNOLOGIES,:]
TechParameters.loc["OldNuke",'RampConstraintMoins']=0.01 ## a bit strong to put in light the effect
TechParameters.loc["OldNuke",'RampConstraintPlus']=0.02 ## a bit strong to put in light the effect
#endregion