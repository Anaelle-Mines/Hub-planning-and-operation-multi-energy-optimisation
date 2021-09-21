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

def My_GetElectricSystemModel_PlaningSingleNode_MultiRessources(areaConsumption, availabilityFactor, TechParameters, ResParameters,
                                                conversionFactor,StorageParameters,Calendrier,N,isAbstract=False):


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
    model.N = Param(model.RESSOURCES,mutable=True, domain=NonNegativeIntegers,default=1,initialize=N)

    for t in TIMESTAMP :
        model.availabilityFactor[t,'Solar_PPA']=model.availabilityFactor[t,'Solar']
        model.availabilityFactor[t,'WindOnShore_PPA']=model.availabilityFactor[t,'WindOnShore']

    model.conversionFactor = Param(model.RESSOURCES_TECHNOLOGIES, default=0,
                                   initialize=conversionFactor.loc[:, "conversionFactor"].squeeze().to_dict())
    model.importCost = Param(model.TIMESTAMP_RESSOURCES, mutable=True,default=0,
                                      initialize=ResParameters.loc[:,"importCost"].squeeze().to_dict(), domain=Any)

    model.StorageNorm = Param(model.TIMESTAMP,model.RESSOURCES,mutable=True,initialize=0) # Valeur normalisée de l'énergie stockée (z/Cmax)

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

    model.Cmax_var = Var(model.RESSOURCES,domain=NonNegativeReals) # Valeur de la capacité de stockage pour une ressource donnée
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

    # Storage capacity constraint
    def StorageCapacity_rule(model, Res):  # INEQ forall t, tech
        return model.Cmax_var[Res] <= model.c_max[Res]
    model.StorageCapacityCtr = Constraint(model.RESSOURCES, rule=StorageCapacity_rule)

    # Ressource production constraint
    def Production_rule(model, t, res):  # EQ forall t, res
        return sum(model.power_var[t, tech] * model.conversionFactor[res, tech] for tech in model.TECHNOLOGIES)+model.importation_var[t, res] == model.energy_var[t, res]
    model.ProductionCtr = Constraint(model.TIMESTAMP, model.RESSOURCES, rule=Production_rule)

    # Contrainte d'equilibre offre/demande pour l'électricité et hydrogen
    def energyCtr_rule(model, t, res):  # INEQ forall t,res
        if res == 'electricity':
            return model.energy_var[t,res] - model.areaConsumption[t, res] - model.StorageNorm[t,res]*model.Cmax_var[res]/(model.N[res]) == model.injection_var[t,res]
        elif res == 'hydrogen':
            return model.energy_var[t, res] - model.StorageNorm[t,res]*model.Cmax_var[res]/(model.N[res]) >= model.areaConsumption[t, res]
        else:
            return model.injection_var[t,res]==0
    model.energyCtr = Constraint(model.TIMESTAMP, model.RESSOURCES, rule=energyCtr_rule)

    # Contrainte d'équilibre offre/demande pour les ressources stockables (annuelle)
    def annualEnergyCtr_rule(model, res):   # INEQ forall res
        return sum(model.energy_var[t, res] for t in TIMESTAMP) >= sum(model.areaConsumption[t, res] for t in TIMESTAMP)
    model.annualEnergyCtr = Constraint(model.RESSOURCES, rule=annualEnergyCtr_rule)

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

def My_GetElectricSystemModel_PlaningSingleNode_MultiRessources_WithStorage(areaConsumption, availabilityFactor, TechParameters, ResParameters,
                                                conversionFactor,Calendrier,StorageParameters,tol,n,N,solver="mosek"):

    model = My_GetElectricSystemModel_PlaningSingleNode_MultiRessources(areaConsumption, availabilityFactor,TechParameters, ResParameters, conversionFactor,StorageParameters,Calendrier,N)
    opt = SolverFactory(solver)

    RESSOURCES = ResParameters.index.get_level_values('RESSOURCES').unique().tolist()

    ##### Loop
    Consommation = {}
    for Res in RESSOURCES : Consommation[Res]={}
    LMultipliers = {}
    for Res in RESSOURCES : LMultipliers[Res]={}
    CostFunction = {}
    for Res in RESSOURCES : CostFunction[Res] = {}
    zz = {}
    for Res in RESSOURCES: zz[Res] = {}
    ResCosts = {}
    Capacity = {}
    Cost={}
    Cmax = {}
    TMP_list={}
    for Res in RESSOURCES: TMP_list[Res] = {}
    CostFun={}
    for Res in RESSOURCES: CostFun[Res] = {}

    OptimControl=pd.DataFrame(columns=["Step","RESSOURCES","TotalCols","DeltaPrix","Deltazz"])

    nbTime = len(areaConsumption.index.get_level_values('TIMESTAMP').unique().tolist())
    cpt = 0

    DeltaPrix_ = tol + 1
    while ((DeltaPrix_>tol) & (n > cpt)):
        print(cpt)
        if (cpt == 0):
            for Res in RESSOURCES :
                zz[Res][cpt] = [0] * nbTime
                areaConsumption["Storage"] = 0
        else:
            DeltaPrix_ = 0
            for Res in RESSOURCES :
                zz[Res][cpt] = areaConsumption.loc[(slice(None),Res),"Storage"].to_list()

        results = opt.solve(model)
        Constraints = getConstraintsDual_panda(model)["energyCtr"]
        Constraints = Constraints.set_index(["RESSOURCES", "TIMESTAMP"])
        Variables = getVariables_panda(model)
        TechCosts = Variables['powerCosts_var'].sum()[1] + Variables['capacityCosts_var'].sum()[1]
        Variables['turpeCosts_var'].loc[Variables['turpeCosts_var'].turpeCosts_var.isna(),'turpeCosts_var']=0
        Capacity[cpt] = Variables['capacity_var']
        Cost[cpt]= Variables['capacityCosts_var']
        Cmax[cpt]=Variables['Cmax_var']

        Res='hydrogen'
        ResCosts[Res] = Variables['importCosts_var'].set_index(['RESSOURCES']).loc[Res]['importCosts_var'] + \
                        Variables['turpeCosts_var'].set_index(['RESSOURCES']).loc[Res]['turpeCosts_var']+ \
                        Variables['storageCosts_var'].set_index(['RESSOURCES']).loc[Res]['storageCosts_var']
        TotalCols = TechCosts + ResCosts[Res]
        Prix = (round(Constraints.energyCtr.loc[(Res,slice(None))],0)).to_numpy()
        Prix = Prix + 1000000000
        Prix[Prix <= 0] = 0.0001
        valueAtZero = TotalCols - Prix * zz[Res][cpt]
        CostFun[Res][cpt]=GenCostFunctionFromMarketPrices_dict(Prix,r_in=StorageParameters.efficiency_in.loc[Res].tolist(),r_out=StorageParameters.efficiency_out.loc[Res].tolist(),valueAtZero=valueAtZero)

        if (cpt == 0):
            CostFunction[Res][cpt] = GenCostFunctionFromMarketPrices(Prix,r_in=StorageParameters.efficiency_in.loc[Res].tolist(),r_out=StorageParameters.efficiency_out.loc[Res].tolist(),valueAtZero=valueAtZero)
        else:
            tmpCost = GenCostFunctionFromMarketPrices_dict(Prix,r_in=StorageParameters.efficiency_in.loc[Res].tolist(),r_out=StorageParameters.efficiency_out.loc[Res].tolist(),valueAtZero=valueAtZero)
            tmpCost2 = CostFunction[Res][cpt - 1]
            if StorageParameters.efficiency_in.loc[Res] * StorageParameters.efficiency_out.loc[Res] == 1:
                tmpCost2.Maxf_1Breaks_withO(tmpCost['S1'], tmpCost['B1'], tmpCost['f0'])
            else:
                tmpCost2.Maxf_2Breaks_withO(tmpCost['S1'], tmpCost['S2'], tmpCost['B1'], tmpCost['B2'], tmpCost['f0'])  ### etape clé, il faut bien comprendre ici l'utilisation du max de deux fonction de coût
            CostFunction[Res][cpt] = tmpCost2
        LMultipliers[Res][cpt] = Prix

        if cpt > 0:
            DeltaPrix = sum(abs(LMultipliers[Res][cpt] - LMultipliers[Res][cpt - 1])) / sum(abs(LMultipliers[Res][cpt]))
            # if (sum(abs(zz[Res][cpt])) == 0) & (sum(abs(zz[Res][cpt] - zz[Res][cpt - 1])) == 0):
            #     Deltazz = 0
            # else:
            #     Deltazz = sum(abs(zz[Res][cpt] - zz[Res][cpt - 1])) / sum(abs(zz[Res][cpt]))
            DeltaPrix_ = DeltaPrix_ + DeltaPrix

        #OptimControl_tmp = pd.DataFrame([cpt, Res, TotalCols, DeltaPrix, Deltazz]).transpose()
        # OptimControl_tmp.columns = ["Step", "RESSOURCES", "TotalCols", "DeltaPrix"? "Deltazz"]
        #  OptimControl = pd.concat([OptimControl, OptimControl_tmp], axis=0)

        efficiency_out = StorageParameters.efficiency_out.loc[Res].tolist()
        efficiency_in = StorageParameters.efficiency_in.loc[Res].tolist()
        TMP=CostFunction[Res][cpt].OptimMargInt([-1/efficiency_out] * nbTime,[1*efficiency_in] * nbTime, [0] * nbTime,[N[Res]]* nbTime)
        TMP_df = pd.DataFrame(TMP,columns=['Storage'],index=[np.arange(1,8761,1)])
        TMP_df[TMP_df > 0] = TMP_df[TMP_df > 0] / efficiency_in
        TMP_df[TMP_df < 0] = TMP_df[TMP_df < 0] * efficiency_out
        TMP_list[Res][cpt]=TMP

        if (TMP_df.Storage.max()>(1+tol) or abs(TMP_df.Storage.min())>(1+tol)) :
            print("ATTENTION, la contrainte de puissance n'est pas respectée")
            break
        areaConsumption.loc[(slice(None), Res), "Storage"]=TMP_df.Storage.tolist()
        for i in np.arange(1,nbTime+1,1) : model.StorageNorm[(i,Res)] = TMP_df.Storage.tolist()[i-1]

        cpt = cpt + 1

    results = opt.solve(model)
    stats = {"DeltaPrix": DeltaPrix}#, "Deltazz": Deltazz}
    Variables = getVariables_panda(model)  #### obtain optimized variables in panda form
    Constraints = getConstraintsDual_panda(model) #### obtain lagrange constraints in panda form

    return results,stats,Variables,Constraints,LMultipliers,Capacity,Cost,TMP_list,Cmax,CostFun

# def Boucle_SensibiliteAlphaSimple(areaConsumption, availabilityFactor, TechParameters, ResParameters,
#                                                 conversionFactor,variation_CAPEX_H2,variation_prix_GazNat) :
#     CAPEX_electrolysis = TechParameters['capacityCost']['electrolysis']
#     CAPEX_pac = TechParameters['capacityCost']['pac']
#     Prix_gaz = ResParameters['importCost']['gaz']
#     alpha_list = []
#     Prod_EnR_list = []
#     Prod_elec_list = []
#     Prod_gaz_list = []
#     Prod_H2_list = []
#     Prod_Nuke_list = []
#     Conso_gaz_list = []
#     Capa_gaz_list = []
#     Capa_EnR_list = []
#     Capa_electrolysis_list = []
#     Capa_PAC_list = []
#
#     for var2 in variation_prix_GazNat:
#         ResParameters['importCost']['gaz'] = var2
#         for var1 in variation_CAPEX_H2:
#             TechParameters['capacityCost']['electrolysis'] = CAPEX_electrolysis * (1 + var1)
#             TechParameters['capacityCost']['pac'] = CAPEX_pac * (1 + var1)
#             model = My_GetElectricSystemModel_PlaningSingleNode_MultiRessources(areaConsumption, availabilityFactor,
#                                                                                 TechParameters, ResParameters,
#                                                                                 conversionFactor)
#             solver = 'mosek'
#             opt = SolverFactory(solver)
#             results = opt.solve(model)
#             Variables = getVariables_panda_indexed(model)
#             ener = Variables['power'].pivot(index="TIMESTAMP", columns='TECHNOLOGIES', values='power')
#             ener = ener.sum(axis=0)
#             ener = pd.DataFrame(ener, columns={'energy'})
#             alpha = ener.loc['pac'] / (ener.loc['CCG'] + ener.loc['TAC'] + ener.loc['pac'])
#             alpha_list = alpha_list + [alpha['energy']]
#             Prod_elec_list = Prod_elec_list + [ener.sum(axis=0)['energy'] / 1000000]
#             Prod_EnR_list = Prod_EnR_list + [(ener.loc['Solar']['energy'] + ener.loc['HydroReservoir']['energy'] +
#                                               ener.loc['HydroRiver']['energy'] + ener.loc['WindOnShore'][
#                                                   'energy']) / 1000000]
#             Prod_gaz_list = Prod_gaz_list + [(ener.loc['CCG']['energy'] + ener.loc['TAC']['energy']) / 1000000]
#             Prod_H2_list = Prod_H2_list + [ener.loc['pac']['energy'] / 1000000]
#             Prod_Nuke_list = Prod_Nuke_list + [ener.loc['OldNuke']['energy'] / 1000000]
#             Conso_gaz_list = Conso_gaz_list + [
#                 Variables['importation'].loc[Variables['importation']['RESSOURCES'] == 'gaz'].sum(axis=0)[
#                     'importation'] / 1000000]
#             capa = Variables['capacity'].set_index('TECHNOLOGIES')
#             Capa_gaz_list = Capa_gaz_list + [(capa.loc['TAC']['capacity'] + capa.loc['CCG']['capacity']) / 1000]
#             Capa_EnR_list = Capa_EnR_list + [(capa.loc['Solar']['capacity'] + capa.loc['HydroReservoir']['capacity'] +
#                                               capa.loc['HydroRiver']['capacity'] + capa.loc['WindOnShore'][
#                                                   'capacity']) / 1000]
#             Capa_electrolysis_list = Capa_electrolysis_list + [capa.loc['electrolysis']['capacity'] / 1000]
#             Capa_PAC_list = Capa_PAC_list + [capa.loc['pac']['capacity'] / 1000]
#             print(alpha_list)
#
#     ### récupérer dataframe à partir de la liste des résultats
#
#     PrixGaz_list = []
#     CAPEX_list = []
#     CAPEX_H2 = []
#     for var1 in variation_CAPEX_H2:
#         CAPEX_H2.append(round(CAPEX_electrolysis * (1 + var1) + CAPEX_pac * (1 + var1), 1))
#
#     for i in variation_prix_GazNat:
#         for j in CAPEX_H2:
#             PrixGaz_list.append(i)
#             CAPEX_list.append(j)
#
#     alpha_df = pd.DataFrame()
#     alpha_df['PrixGaz'] = PrixGaz_list
#     alpha_df['Capex'] = CAPEX_list
#     alpha_df['value'] = alpha_list
#     alpha_df['Prod_elec'] = Prod_elec_list
#     alpha_df['Prod_EnR'] = Prod_EnR_list
#     alpha_df['Prod_gaz'] = Prod_gaz_list
#     alpha_df['Prod_H2'] = Prod_H2_list
#     alpha_df['Prod_Nuke'] = Prod_Nuke_list
#     alpha_df['Conso_gaz'] = Conso_gaz_list
#     alpha_df['Capa_gaz'] = Capa_gaz_list
#     alpha_df['Capa_EnR'] = Capa_EnR_list
#     alpha_df['Capa_electrolysis'] = Capa_electrolysis_list
#     alpha_df['Capa_PAC'] = Capa_PAC_list
#
#     return alpha_df
#

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
    Resultat[['gaz_Conso','alpha','Capex','PrixGaz','DemResMax']]=[Importation.loc['gaz','importation'],alpha,TechParameters.loc['electrolysis','capacityCost']+TechParameters.loc['pac','capacityCost'],VariationPrixGaz.squeeze(),DemResMax['areaConsumption']]
    return Resultat


# def Boucle_SensibiliteAlphaSimple_With1Storage(areaConsumption, availabilityFactor, TechParameters, ResParameters,
#                                                 conversionFactor,variation_CAPEX_H2,variation_prix_GazNat,StorageParameters,tol,n) :
#
#     CAPEX_electrolysis = TechParameters['capacityCost']['electrolysis']
#     CAPEX_pac = TechParameters['capacityCost']['pac']
#     Prix_gaz = ResParameters['importCost']['gaz']
#     alpha_list = []
#     Prod_EnR_list = []
#     Prod_elec_list = []
#     Prod_gaz_list = []
#     Prod_H2_list = []
#     Prod_Nuke_list = []
#     Conso_gaz_list = []
#     Capa_gaz_list = []
#     Capa_EnR_list = []
#     Capa_electrolysis_list = []
#     Capa_PAC_list = []
#
#     for var2 in variation_prix_GazNat:
#         ResParameters['importCost']['gaz'] = var2
#         for var1 in variation_CAPEX_H2:
#             TechParameters['capacityCost']['electrolysis'] = CAPEX_electrolysis * (1 + var1)
#             TechParameters['capacityCost']['pac'] = CAPEX_pac * (1 + var1)
#             Variables = My_GetElectricSystemModel_PlaningSingleNode_MultiRessources_With1Storage(areaConsumption, availabilityFactor, TechParameters, ResParameters,
#                                                 conversionFactor,StorageParameters,tol,n)[2]
#             ener = Variables['power'].pivot(index="TIMESTAMP", columns='TECHNOLOGIES', values='power')
#             ener = ener.sum(axis=0)
#             ener = pd.DataFrame(ener, columns={'energy'})
#             alpha = ener.loc['pac'] / (ener.loc['CCG'] + ener.loc['TAC'] + ener.loc['pac'])
#             alpha_list = alpha_list + [alpha['energy']]
#             Prod_elec_list = Prod_elec_list + [ener.sum(axis=0)['energy'] / 1000000]
#             Prod_EnR_list = Prod_EnR_list + [(ener.loc['Solar']['energy'] + ener.loc['HydroReservoir']['energy'] +
#                                               ener.loc['HydroRiver']['energy'] + ener.loc['WindOnShore'][
#                                                   'energy']) / 1000000]
#             Prod_gaz_list = Prod_gaz_list + [(ener.loc['CCG']['energy'] + ener.loc['TAC']['energy']) / 1000000]
#             Prod_H2_list = Prod_H2_list + [ener.loc['pac']['energy'] / 1000000]
#             Prod_Nuke_list = Prod_Nuke_list + [ener.loc['OldNuke']['energy'] / 1000000]
#             Conso_gaz_list = Conso_gaz_list + [
#                 Variables['importation'].loc[Variables['importation']['RESSOURCES'] == 'gaz'].sum(axis=0)[
#                     'importation'] / 1000000]
#             capa = Variables['capacity'].set_index('TECHNOLOGIES')
#             Capa_gaz_list = Capa_gaz_list + [(capa.loc['TAC']['capacity'] + capa.loc['CCG']['capacity']) / 1000]
#             Capa_EnR_list = Capa_EnR_list + [(capa.loc['Solar']['capacity'] + capa.loc['HydroReservoir']['capacity'] +
#                                               capa.loc['HydroRiver']['capacity'] + capa.loc['WindOnShore'][
#                                                   'capacity']) / 1000]
#             Capa_electrolysis_list = Capa_electrolysis_list + [capa.loc['electrolysis']['capacity'] / 1000]
#             Capa_PAC_list = Capa_PAC_list + [capa.loc['pac']['capacity'] / 1000]
#             print(alpha_list)
#
#     ### récupérer dataframe à partir de la liste des résultats
#
#     PrixGaz_list = []
#     CAPEX_list = []
#     CAPEX_H2 = []
#     for var1 in variation_CAPEX_H2:
#         CAPEX_H2.append(round(CAPEX_electrolysis * (1 + var1) + CAPEX_pac * (1 + var1), 1))
#
#     for i in variation_prix_GazNat:
#         for j in CAPEX_H2:
#             PrixGaz_list.append(i)
#             CAPEX_list.append(j)
#
#     alpha_df = pd.DataFrame()
#     alpha_df['PrixGaz'] = PrixGaz_list
#     alpha_df['Capex'] = CAPEX_list
#     alpha_df['value'] = alpha_list
#     alpha_df['Prod_elec'] = Prod_elec_list
#     alpha_df['Prod_EnR'] = Prod_EnR_list
#     alpha_df['Prod_gaz'] = Prod_gaz_list
#     alpha_df['Prod_H2'] = Prod_H2_list
#     alpha_df['Prod_Nuke'] = Prod_Nuke_list
#     alpha_df['Conso_gaz'] = Conso_gaz_list
#     alpha_df['Capa_gaz'] = Capa_gaz_list
#     alpha_df['Capa_EnR'] = Capa_EnR_list
#     alpha_df['Capa_electrolysis'] = Capa_electrolysis_list
#     alpha_df['Capa_PAC'] = Capa_PAC_list
#
#     return alpha_df

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