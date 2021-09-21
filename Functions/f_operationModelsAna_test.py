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

def loadingParameters(Selected_TECHNOLOGIES = ['OldNuke', 'Solar', 'WindOnShore', 'HydroReservoir', 'HydroRiver', 'TAC', 'CCG', 'pac','electrolysis'],InputFolder = 'Data/Input/',Zones = "FR",year = 2013,other='',PrixRes='fixe'):

    #### reading CSV files
    areaConsumption = pd.read_csv(InputFolder + 'areaConsumption' + str(year) +'_' + str(Zones)+str(other)+'.csv',
                                  sep=',', decimal='.', skiprows=0).set_index(["TIMESTAMP", "RESSOURCES"])
    availabilityFactor = pd.read_csv(InputFolder + 'availabilityFactor' + str(year) + '_' + str(Zones) + '.csv',
                                     sep=',', decimal='.', skiprows=0).set_index(["TIMESTAMP", "TECHNOLOGIES"])
    TechParameters = pd.read_csv(InputFolder + 'Planing-RAMP2_TECHNOLOGIES.csv', sep=',', decimal='.', skiprows=0,
                                 comment="#").set_index(["TECHNOLOGIES"])
    conversionFactor = pd.read_csv(InputFolder + 'Ressources_conversionFactors.csv', sep=',', decimal='.', skiprows=0,
                                   comment="#").set_index(["RESSOURCES", "TECHNOLOGIES"])
    ResParameters = pd.read_csv(InputFolder + 'Ressources_set_'+str(PrixRes)+'.csv', sep=',', decimal='.', skiprows=0,
                                comment="#").set_index(["TIMESTAMP","RESSOURCES"])
    StorageParameters = pd.read_csv(InputFolder + 'Stock_Techno_set.csv', sep=',', decimal='.', skiprows=0,
                                comment="#").set_index(["RESSOURCES"])
    Calendrier = pd.read_csv(InputFolder + 'Calandrier.csv', sep=',', decimal='.', skiprows=0,
                                comment="#").set_index(["TIMESTAMP"])

    #### Selection of subset
    availabilityFactor = availabilityFactor.loc[(slice(None), Selected_TECHNOLOGIES), :]
    conversionFactor = conversionFactor.loc[(slice(None), Selected_TECHNOLOGIES), :]
    TechParameters = TechParameters.loc[Selected_TECHNOLOGIES, :]
    TechParameters.loc["OldNuke", 'RampConstraintMoins'] = 0.01  ## a bit strong to put in light the effect
    TechParameters.loc["OldNuke", 'RampConstraintPlus'] = 0.02  ## a bit strong to put in light the effect
    return areaConsumption,availabilityFactor,TechParameters,conversionFactor,ResParameters,Calendrier,StorageParameters

def My_GetElectricSystemModel_PlaningSingleNode_MultiRessources_WithStorage(areaConsumption, availabilityFactor, TechParameters, ResParameters,
                                                conversionFactor,StorageParameters,Calendrier,isAbstract=False):


    availabilityFactor.isna().sum()

    Carbontax=1 # €/kgCO2, 1 signifie qu'on obtient la quantité de CO2 émise

    ### Cleaning
    availabilityFactor = availabilityFactor.fillna(method='pad')
    areaConsumption = areaConsumption.fillna(method='pad')
    conversionFactor = conversionFactor.fillna(method='pad')

    ### obtaining dimensions values
    TECHNOLOGIES = set(TechParameters.index.get_level_values('TECHNOLOGIES').unique())
    RESSOURCES = set(ResParameters.index.get_level_values('RESSOURCES').unique())
    TIMESTAMP = set(areaConsumption.index.get_level_values('TIMESTAMP').unique())
    TIMESTAMP_list = areaConsumption.index.get_level_values('TIMESTAMP').unique()
    HORAIRE = {'P', 'HPH', 'HCH', 'HPE', 'HCE'}

    #Subsets
    TIMESTAMP_HCH= set(Calendrier[Calendrier['Calendrier']=='HCH'].index.get_level_values('TIMESTAMP').unique())
    TIMESTAMP_HPH = set(Calendrier[Calendrier['Calendrier'] == 'HPH'].index.get_level_values('TIMESTAMP').unique())
    TIMESTAMP_HCE = set(Calendrier[Calendrier['Calendrier'] == 'HCE'].index.get_level_values('TIMESTAMP').unique())
    TIMESTAMP_HPE = set(Calendrier[Calendrier['Calendrier'] == 'HPE'].index.get_level_values('TIMESTAMP').unique())
    TIMESTAMP_P = set(Calendrier[Calendrier['Calendrier'] == 'P'].index.get_level_values('TIMESTAMP').unique())


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
    model.HORAIRE = Set(initialize=HORAIRE,ordered=False)
    model.TIMESTAMP_TECHNOLOGIES = model.TIMESTAMP * model.TECHNOLOGIES
    model.TIMESTAMP_RESSOURCES = model.TIMESTAMP * model.RESSOURCES
    model.RESSOURCES_TECHNOLOGIES = model.RESSOURCES * model.TECHNOLOGIES

    # Subset of Simple only required if ramp constraint
    model.TIMESTAMP_MinusOne = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 1], ordered=False)
    model.TIMESTAMP_MinusThree = Set(initialize=TIMESTAMP_list[: len(TIMESTAMP) - 3], ordered=False)

    ###############
    # Parameters ##
    ###############

    model.areaConsumption = Param(model.TIMESTAMP_RESSOURCES, mutable=True,default=0,initialize=areaConsumption.loc[:,"areaConsumption"].squeeze().to_dict(), domain=Any)
    model.availabilityFactor = Param( model.TIMESTAMP_TECHNOLOGIES,mutable=True, domain=PercentFraction,default=1,initialize=availabilityFactor.loc[:,"availabilityFactor"].squeeze().to_dict())

    for t in TIMESTAMP :
        model.availabilityFactor[t,'Solar_PPA']=model.availabilityFactor[t,'Solar']
        model.availabilityFactor[t,'WindOnShore_PPA']=model.availabilityFactor[t,'WindOnShore']

    model.conversionFactor = Param(model.RESSOURCES_TECHNOLOGIES, default=0,
                                   initialize=conversionFactor.loc[:, "conversionFactor"].squeeze().to_dict())
    model.importCost = Param(model.TIMESTAMP_RESSOURCES, mutable=True,default=0,
                                      initialize=ResParameters.loc[:,"importCost"].squeeze().to_dict(), domain=Any)


   # with test of existing columns on TechParameters
    for COLNAME in TechParameters:
        if COLNAME not in ["TECHNOLOGIES", "AREAS"]:  ### each column in TechParameters will be a parameter
            exec("model." + COLNAME + " = Param(model.TECHNOLOGIES, default=0," +
                 "initialize=TechParameters." + COLNAME + ".squeeze().to_dict())")
    ## manière générique d'écrire pour toutes les colomnes COL de TechParameters quelque chose comme
    #    model.COLNAME =          Param(model.TECHNOLOGIES, domain=NonNegativeReals,default=0,
    #                                 initialize=TechParameters.COLNAME.squeeze().to_dict())
    for COLNAME in Calendrier:
        if COLNAME not in ["TIMESTAMP"]:
            exec("model." + COLNAME + " = Param(model.TIMESTAMP, default=0," +
             "initialize=Calendrier." + COLNAME + ".squeeze().to_dict(),domain=Any)")

    for COLNAME in StorageParameters:
        if COLNAME not in ["RESSOURCES"]:
            exec("model." + COLNAME + " = Param(model.RESSOURCES, default=0,mutable=True," +
             "initialize=StorageParameters." + COLNAME + ".squeeze().to_dict(),domain=Any)")

    ################
    # Variables    #
    ################

    model.power_var = Var(model.TIMESTAMP, model.TECHNOLOGIES,domain=NonNegativeReals)  # Instant power for a conversion mean at t
    model.powerCosts_var = Var(model.TECHNOLOGIES)  ### Marginal cost for a conversion mean, explicitely defined by definition powerCostsDef
    model.capacityCosts_var = Var(model.TECHNOLOGIES)  ### Fixed costs for a conversion mean, explicitely defined by definition capacityCostsDef
    model.capacity_var = Var(model.TECHNOLOGIES, domain=NonNegativeReals)  ### Installed capacity for a conversion mean
    model.importCosts_var = Var(model.RESSOURCES)  ### Cost of ressource imported, explicitely defined by definition importCostsDef
    model.turpeCosts_var = Var(model.RESSOURCES,domain=NonNegativeReals) ### Coûts TURPE pour électricité
    model.injectionGain_var = Var(model.RESSOURCES) ### Gain tirés de l'injection de la ressource r sur le réseau
    model.max_PS_var = Var(model.HORAIRE,domain=NonNegativeReals) ### Puissance souscrite max par plage horaire
    model.storageCosts_var = Var(model.RESSOURCES,domain=NonNegativeReals) ### Coûts de stockage

    model.StorageIn_var = Var(model.TIMESTAMP,model.RESSOURCES,domain=NonNegativeReals)  # Valeur de l'énergie stockée
    model.StorageOut_var = Var(model.TIMESTAMP,model.RESSOURCES,domain=NonNegativeReals)  # Valeur de l'énergie déstockée
    model.StockNiv_var = Var(model.TIMESTAMP,model.RESSOURCES,initialize=0,domain=NonNegativeReals) # Valeur du niveau de stock
    model.Cmax_var = Var(model.RESSOURCES,domain=NonNegativeReals) # Valeur de la capacité de stockage pour une ressource donnée
    model.Pmax_var = Var(model.RESSOURCES,domain=NonNegativeReals)  # Valeur de la capacité de stockage pour une ressource donnée
    model.importation_var  = Var(model.TIMESTAMP, model.RESSOURCES, domain=NonNegativeReals,initialize=0)
    model.injection_var = Var(model.TIMESTAMP, model.RESSOURCES, domain=NonNegativeReals)
    model.PPA_var = Var(model.TIMESTAMP, domain=NonNegativeReals)
    model.energy_var = Var(model.TIMESTAMP, model.RESSOURCES)  ### Variation of ressource r at time t

    model.carbonCosts_var = Var(model.TECHNOLOGIES, domain=NonNegativeReals)  ### Carbon emission costs for a conversion mean, explicitly defined by powerCostsDef

    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    model.slack = Suffix(direction=Suffix.IMPORT)

    ########################
    # Objective Function   #
    ########################

    def ObjectiveFunction_rule(model):  # OBJ
        return (sum(model.powerCosts_var[tech] + model.capacityCosts_var[tech] + model.carbonCosts_var[tech] for tech in model.TECHNOLOGIES) + sum(model.importCosts_var[res]+model.storageCosts_var[res] for res in model.RESSOURCES) + model.turpeCosts_var['electricity'])
    model.OBJ = Objective(rule=ObjectiveFunction_rule, sense=minimize)

    #################
    # Constraints   #
    #################

    # energyCosts definition Constraints
    def powerCostsDef_rule(model,tech):  # EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in TIMESTAMP} energyCost[tech]*energy[t,tech];
        temp1 = model.powerCost[tech]
        return sum(temp1 * model.power_var[t, tech] for t in model.TIMESTAMP) == model.powerCosts_var[tech]
    model.powerCostsCtr = Constraint(model.TECHNOLOGIES, rule=powerCostsDef_rule)

    def carbonCostsDef_rule(model,tech):
        temp2 = model.EmissionCO2[tech]
        return sum(temp2 * model.power_var[t, tech] * Carbontax for t in model.TIMESTAMP) == model.carbonCosts_var[tech]
    model.carbonCostsCtr = Constraint(model.TECHNOLOGIES, rule=carbonCostsDef_rule)

    # capacityCosts definition Constraints
    def capacityCostsDef_rule(model,tech):  # EQ forall tech in TECHNOLOGIES   energyCosts  = sum{t in TIMESTAMP} energyCost[tech]*energy[t,tech];
        temp = model.capacityCost[tech]
        return temp * model.capacity_var[tech] == model.capacityCosts_var[tech]
    model.capacityCostsCtr = Constraint(model.TECHNOLOGIES, rule=capacityCostsDef_rule)

    def storageCostsDef_rule(model,res):
        return model.storageCost[res]*model.Cmax_var[res]  == model.storageCosts_var[res]
    model.storageCostsCtr = Constraint(model.RESSOURCES, rule=storageCostsDef_rule)

    # importCosts definition Constraints
    def importCostsDef_rule(model,res):  # ;
        return sum((model.importCost[t,res] * model.importation_var[t, res]) for t in model.TIMESTAMP) == model.importCosts_var[res]
    model.importCostsCtr = Constraint(model.RESSOURCES, rule=importCostsDef_rule)

     # volume gaz Constraints
    def Volume_rule(model):  # INEQ for gaz
        return sum(model.importation_var[t, 'gaz'] for t in model.TIMESTAMP) <= 100000000
    model.VolumeCtr = Constraint(rule=Volume_rule)

    # Capacity constraint
    def Capacity_rule(model, t, tech):  # INEQ forall t, tech
        return model.capacity_var[tech] * model.availabilityFactor[t, tech] >= model.power_var[t, tech]
    model.CapacityCtr = Constraint(model.TIMESTAMP, model.TECHNOLOGIES, rule=Capacity_rule)

    # Storage max capacity constraint
    def StorageMaxCapacity_rule(model, Res):  # INEQ forall Res
        return model.Cmax_var[Res] <= model.c_max[Res]
    model.StorageMaxCapacityCtr = Constraint(model.RESSOURCES, rule=StorageMaxCapacity_rule)

    # Storage power constraints
    def StoragePowerUB_rule(model, t,res):  # INEQ forall t
        return model.StorageIn_var[t,res]  <= model.Pmax_var[res]
    model.StoragePowerUBCtr = Constraint(model.TIMESTAMP,model.RESSOURCES, rule=StoragePowerUB_rule)

    def StoragePowerLB_rule(model, t,res):  # INEQ forall t
        return  model.StorageOut_var[t,res] <= model.Pmax_var[res]
    model.StoragePowerLBCtr = Constraint(model.TIMESTAMP,model.RESSOURCES, rule=StoragePowerLB_rule)

    # Ressource production constraint
    def Production_rule(model, t, res):  # EQ forall t, res
        return sum(model.power_var[t, tech] * model.conversionFactor[res, tech] for tech in model.TECHNOLOGIES) + model.importation_var[t, res] + model.StorageOut_var[t,res]*model.efficiency_out[res] - model.StorageIn_var[t,res] == model.energy_var[t, res]
    model.ProductionCtr = Constraint(model.TIMESTAMP, model.RESSOURCES, rule=Production_rule)

    # Contrainte d'equilibre offre/demande pour l'électricité et hydrogen
    def energyCtr_rule(model, t, res):  # INEQ forall t,res
        if res == 'electricity':
            return model.energy_var[t,res] - model.areaConsumption[t, res] == model.injection_var[t,res]
        elif res == 'hydrogen':
            return model.energy_var[t, res]  >= model.areaConsumption[t, res]
        else:
            return model.injection_var[t,res]==0
    model.energyCtr = Constraint(model.TIMESTAMP, model.RESSOURCES, rule=energyCtr_rule)

    # Contrainte d'équilibre offre/demande pour les ressources stockables (annuelle)
    def annualEnergyCtr_rule(model, res):   # INEQ forall res
            return sum(model.energy_var[t, res] for t in TIMESTAMP) >= sum(model.areaConsumption[t, res] for t in TIMESTAMP)
    model.annualEnergyCtr = Constraint(model.RESSOURCES, rule=annualEnergyCtr_rule)

    # Contrainte de stock
    def Storage_rule(model, t, res):   # INEQ forall t
        if t == 1 :
            return Constraint.Skip
        else :
            return model.StockNiv_var[t,res] == model.StockNiv_var [t-1,res] + model.StorageIn_var[t,res]*model.efficiency_in[res] - model.StorageOut_var[t,res]
    model.StorageCtr = Constraint(model.TIMESTAMP,model.RESSOURCES, rule=Storage_rule)

    def StorageLoop_rule(model, t, res):   # EQ for t == 0 and t == 8760
        if t == 8760 : return model.StockNiv_var[t,res] == model.StockNiv_var[1,res]
        else : return Constraint.Skip
    model.StorageLoopCtr = Constraint(model.TIMESTAMP,model.RESSOURCES, rule=StorageLoop_rule)

    def StorageCapacity_rule(model, t, res):  # INEQ forall t
        return model.StockNiv_var[t,res] <= model.Cmax_var[res]
    model.StorageCapacityCtr = Constraint(model.TIMESTAMP,model.RESSOURCES, rule=StorageCapacity_rule)

    # PPA constraint
    def PPA_rule(model, t, tech):
        if tech == ('Solar_PPA' or 'WindOnShore_PPA') :
            return model.PPA_var[t] == model.power_var[t, tech] * model.conversionFactor['electricity', tech]
        else :
            return Constraint.Skip
    model.PPACtr = Constraint(model.TIMESTAMP,model.TECHNOLOGIES, rule=PPA_rule)

    # injectionGain constraint
    def injectionGain_rule(model, res):
            return model.injectionGain_var[res] == sum(model.injection_var[t,res]*model.importCost[t,res] for t in TIMESTAMP)
    model.injectionGainCtr = Constraint(model.RESSOURCES, rule=injectionGain_rule)

    # TURPE classique
    def PuissanceSouscrite_rule(model,t,res):
        if res == 'electricity':
            if t in TIMESTAMP_P :
                return model.max_PS_var['P'] >= model.importation_var[t,res] + model.PPA_var[t] # en MW
            elif t in TIMESTAMP_HPH :
                return model.max_PS_var['HPH'] >= model.importation_var[t, res] + model.PPA_var[t]
            elif t in TIMESTAMP_HCH :
                return model.max_PS_var['HCH'] >= model.importation_var[t, res] + model.PPA_var[t]
            elif t in TIMESTAMP_HPE :
                return model.max_PS_var['HPE'] >= model.importation_var[t, res] + model.PPA_var[t]
            elif t in TIMESTAMP_HCE :
                return model.max_PS_var['HCE'] >= model.importation_var[t, res] + model.PPA_var[t]
        else:
            return Constraint.Skip
    model.PuissanceSouscriteCtr = Constraint(model.TIMESTAMP,model.RESSOURCES, rule=PuissanceSouscrite_rule)

    def TurpeCtr_rule(model, res):
        if res == 'electricity':
            return model.turpeCosts_var[res] == sum(model.HTA[t] * (model.importation_var[t,res] + model.PPA_var[t]) for t in TIMESTAMP) + model.max_PS_var['P']*16310+(model.max_PS_var['HPH']-model.max_PS_var['P'])*15760+(model.max_PS_var['HCH']-model.max_PS_var['HPH'])*13290+(model.max_PS_var['HPE']-model.max_PS_var['HCH'])*8750+(model.max_PS_var['HCE']-model.max_PS_var['HPE'])*1670
        else:
            return model.turpeCosts_var[res] == 0
    model.TurpeCtr = Constraint(model.RESSOURCES, rule=TurpeCtr_rule)

    if "maxCapacity" in TechParameters:
        def maxCapacity_rule(model, tech):  # INEQ forall t, tech
            return model.maxCapacity[tech] >= model.capacity_var[tech]
        model.maxCapacityCtr = Constraint(model.TECHNOLOGIES, rule=maxCapacity_rule)

    if "minCapacity" in TechParameters:
        def minCapacity_rule(model, tech):  # INEQ forall t, tech
            if model.minCapacity[tech] > 0:
                return model.minCapacity[tech] <= model.capacity_var[tech]
            else:
                return Constraint.Skip

        model.minCapacityCtr = Constraint(model.TECHNOLOGIES, rule=minCapacity_rule)

    if "EnergyNbhourCap" in TechParameters:
        def storage_rule(model, tech):  # INEQ forall t, tech
            if model.EnergyNbhourCap[tech] > 0:
                return model.EnergyNbhourCap[tech] * model.capacity_var[tech] >= sum(
                    model.power_var[t, tech] for t in model.TIMESTAMP)
            else:
                return Constraint.Skip

        model.storageCtr = Constraint(model.TECHNOLOGIES, rule=storage_rule)

    if "RampConstraintPlus" in TechParameters:
        def rampCtrPlus_rule(model, t, tech):  # INEQ forall t<
            if model.RampConstraintPlus[tech] > 0:
                return model.power_var[t + 1, tech] - model.power_var[t, tech] <= model.capacity_var[tech] * \
                       model.RampConstraintPlus[tech];
            else:
                return Constraint.Skip

        model.rampCtrPlus = Constraint(model.TIMESTAMP_MinusOne, model.TECHNOLOGIES, rule=rampCtrPlus_rule)

    if "RampConstraintMoins" in TechParameters:
        def rampCtrMoins_rule(model, t, tech):  # INEQ forall t<
            if model.RampConstraintMoins[tech] > 0:
                return model.power_var[t + 1, tech] - model.power_var[t, tech] >= - model.capacity_var[tech] * \
                       model.RampConstraintMoins[tech];
            else:
                return Constraint.Skip

        model.rampCtrMoins = Constraint(model.TIMESTAMP_MinusOne, model.TECHNOLOGIES, rule=rampCtrMoins_rule)

    return model
