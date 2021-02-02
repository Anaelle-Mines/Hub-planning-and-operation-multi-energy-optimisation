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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import linear_model
import sys

from Functions.f_operationModels import *
from Functions.f_optimization import *
from Functions.f_graphicalTools import *

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

    model.areaConsumption = Param(model.TIMESTAMP_RESSOURCES, domain=NonNegativeReals, default=0,
                                  initialize=areaConsumption.loc[:, "areaConsumption"].squeeze().to_dict())
    model.availabilityFactor = Param(model.TIMESTAMP_TECHNOLOGIES, domain=PercentFraction, default=1,
                                     initialize=availabilityFactor.loc[:, "availabilityFactor"].squeeze().to_dict())
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
            return  model.energy[t, res] - model.areaConsumption[t,res] <= 500000
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
# endregion

#region Ramp Single area : loading parameters case with ramp constraints
Zones="FR"
year=2013
Selected_TECHNOLOGIES=['OldNuke','Solar','WindOnShore','HydroReservoir','HydroRiver','TAC','CCG']
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

#region Ramp Single area : solving and loading results
model = My_GetElectricSystemModel_PlaningSingleNode(areaConsumption,availabilityFactor,TechParameters,ResParameters,conversionFactor)
opt = SolverFactory(solver)
results=opt.solve(model)
Variables=getVariables_panda_indexed(model)

#pour avoir la production en KWh de chaque moyen de prod chaque heure
#production_df=Variables['power'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='power')
#energy_variation=
### Check sum Prod = Consumption
#Delta=(production_df.sum(axis=1) - areaConsumption.areaConsumption);
#abs(Delta).max()
#endregion

#region Visualisation des résultats

# Capacity Costs
df=Variables['capacityCosts']
df.drop(df.loc[df['capacityCosts']<1000000].index,inplace=True)
df=pd.DataFrame([list(Variables['capacityCosts']['capacityCosts']/1000000)], columns=list(Variables['capacityCosts']['TECHNOLOGIES']))
fig1 = px.bar(df, title="Capacity costs",width=600)
fig1=fig1.update_layout(xaxis_title="Scenario",yaxis_title="M€")
plotly.offline.plot(fig1, filename='Capacity costs.html')

# Importation Costs
df=Variables['importCosts']
df.drop(df.loc[df['importCosts']<1000000].index,inplace=True)
df=pd.DataFrame([list(Variables['importCosts']['importCosts']/1000000)], columns=list(Variables['importCosts']['RESSOURCES']))
fig2 = px.bar(df, title="Importation costs",width=600)
fig2=fig2.update_layout(xaxis_title="Scenario",yaxis_title="M€")
plotly.offline.plot(fig2, filename='Importation costs.html')

# Carbon Costs
df=Variables['carbonCosts']
df.drop(df.loc[df['carbonCosts']<1].index,inplace=True)
df=pd.DataFrame([list(Variables['carbonCosts']['carbonCosts']/1000000)], columns=list(Variables['carbonCosts']['TECHNOLOGIES']))
fig6 = px.bar(df, title="Emission costs",width=600)
fig6=fig6.update_layout(xaxis_title="Scenario",yaxis_title="M€")
plotly.offline.plot(fig6, filename='Emissioncosts.html')

# Installed Capacity and Total energy production
df1=Variables['capacity']
df1.drop(df1.loc[df1['capacity']<0.1].index,inplace=True)
df1=df1.set_index('TECHNOLOGIES')
df2=Variables['power'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='power')
df2=df2.sum(axis=0)
df2=pd.DataFrame(df2,columns={'energy'})
df2.drop(df2.loc[df2['energy']<100].index,inplace=True)
df=pd.concat([df1,df2],axis=1)
df=df.reset_index()
df1=pd.DataFrame([list(df['capacity']/1000)], columns=list(df['TECHNOLOGIES']))
df2=pd.DataFrame([list(df['energy']/1000000)], columns=list(df['TECHNOLOGIES']))
fig3 = px.bar(df1,barmode='stack',title='Installed capacity')
fig3=fig3.update_layout(xaxis_title="Scenario",yaxis_title="GW",width=600)
plotly.offline.plot(fig3, filename='Installed capacity.html')
fig4 = px.bar(df2,barmode='stack',title='Produced energy',width=600)
fig4=fig4.update_layout(xaxis_title="Scenario",yaxis_title="TWh")
plotly.offline.plot(fig4, filename='Produced energy.html')

# test
df1=Variables['capacity']
df1.drop(df1.loc[df1['capacity']<0.1].index,inplace=True)
df1=df1.set_index('TECHNOLOGIES')
df2=Variables['power'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='power')
df2=df2.sum(axis=0)
df2=pd.DataFrame(df2,columns={'energy'})
df2.drop(df2.loc[df2['energy']<100].index,inplace=True)
df=pd.concat([df1,df2])
df=df.reset_index()
df['type']=np.nan
df.loc[df['capacity'] > 0,'type'] = 'capacity'
df.loc[df['energy'] > 0,'type'] = 'energy'
df['capacity']=df['capacity']/1000
df['energy']=df['energy']/1000000

fig1=px.bar(df, x='type', y=['capacity','energy'], color='TECHNOLOGIES')

fig.add_trace(go.Bar(x=df['type'], y=df['capacity'], name="yaxis data"),secondary_y=False)
fig.add_trace(go.Bar(x=df['type'], y=df['energy'], name="yaxis2 data"),secondary_y=True)
#fig.update_yaxes(title_text="<b>primary</b> yaxis title", secondary_y=False)
#fig.update_yaxes(title_text="<b>secondary</b> yaxis title", secondary_y=True)
plotly.offline.plot(fig, filename='Produced energy.html')



# Energy variation
energy_variation=Variables['energy'].pivot(index="TIMESTAMP",columns='RESSOURCES', values='energy')
TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
energy_variation.index=TIMESTAMP_d;
del energy_variation['uranium']
fig5=MyStackedPlotly(y_df=energy_variation)
fig5=fig5.update_layout(title_text="Variation par énergie (production nette + importation) (en MWh)", xaxis_title="heures de l'année")
plotly.offline.plot(fig5, filename='Energy variation.html')

# Use of production mean
power_use=Variables['power'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='power')
TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
power_use.index=TIMESTAMP_d;
fig6=MyStackedPlotly(y_df=power_use)
fig6=fig6.update_layout(title_text="Utilisation des moyens de production (en MW)", xaxis_title="heures de l'année")
plotly.offline.plot(fig6, filename='Power.html')

#TIMESTAMP_d=pd.date_range(start=str(year)+"-01-01 00:00:00",end=str(year)+"-12-31 23:00:00",   freq="1H")
#productionH2_df.index=TIMESTAMP_d;
#fig2=MyStackedPlotly(y_df=productionH2_df)
#fig2=fig2.update_layout(title_text="Production H2 (en KWh)", xaxis_title="heures de l'année")
#plotly.offline.plot(fig, filename='file.html') ## offline
#fig2.show()

#### lagrange multipliers
Constraints= getConstraintsDual_panda(model)

# Analyse energyCtr
#energyCtrDual=Constraints['energyCtr']; energyCtrDual['energyCtr']=energyCtrDual['energyCtr']*1000000
#energyCtrDual
#round(energyCtrDual.energyCtr,2).unique()

# Analyse CapacityCtr
#CapacityCtrDual=Constraints['CapacityCtr'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='CapacityCtr')*1000000;
#round(CapacityCtrDual,2)
#round(CapacityCtrDual.OldNuke,2).unique() ## if you increase by Delta the installed capacity of nuke you decrease by xxx the cost when nuke is not sufficient
#round(CapacityCtrDual.CCG,2).unique() ## increasing the capacity of CCG as no effect on prices
#endregion

#region Chiffres principaux
Capa_totale = sum(Variables['capacity']['capacity'])/1000
power_use=Variables['power'].pivot(index="TIMESTAMP",columns='TECHNOLOGIES', values='power')
Prod_elec = sum(Variables['power']['power'])/1000000#-sum(power_use['electrolysis'])*1.43/1000000
#Prod_H2 = sum(power_use['electrolysis'])/1000000
Capa_cost = sum(Variables['capacityCosts']['capacityCosts'])/1000000
Import_cost = sum(Variables['importCosts']['importCosts'])/1000000
#Emission_costs = sum(Variables['carbonCosts']['carbonCosts'])/1000000

print('Capa_totale = ',Capa_totale, ' GW')
print('Prod_elec = ',Prod_elec,' TWh')
#print('Prod_H2 = ',Prod_H2,' TWh')
print('Capa_cost = ',Capa_cost,' M€')
print('Import_cost = ',Import_cost,' M€')
#print('Emission_costs = ',Emission_costs,' M€')

#endregion

#region Ordres de grandeur, ratio H2/gaz

Zones="FR"
year=2013
Selected_TECHNOLOGIES=['OldNuke','Solar','WindOnShore','HydroReservoir','HydroRiver','TAC','CCG','pac','electrolysis']
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

#### Boucle pour la variation du CAPEX H2 + la variation du prix du gaz

# Prix du gaz borné à 100€/MWh, au delà => biogaz
# Prix du CAPEX H2 = -50% à +50% par rapport à la référence.
variation_CAPEX_H2=[0.5,0.4,0.3,0.2,0.1,0,-0.1,-0.2,-0.3,-0.4,-0.5]
variation_prix_GazNat=[10,20,30,40,50,60,70,80,90,100]
CAPEX_electrolysis=TechParameters['capacityCost']['electrolysis']
CAPEX_pac=TechParameters['capacityCost']['pac']
Prix_gaz=ResParameters['importCost']['gaz']
alpha_list=[]

for var1 in variation_CAPEX_H2 :
    TechParameters['capacityCost']['electrolysis']=CAPEX_electrolysis*(1+var1)
    TechParameters['capacityCost']['pac']=CAPEX_pac*(1+var1)
    for var2 in variation_prix_GazNat :
        ResParameters['importCost']['gaz']=var2
        model = My_GetElectricSystemModel_PlaningSingleNode(areaConsumption, availabilityFactor, TechParameters,ResParameters, conversionFactor)
        opt = SolverFactory(solver)
        results = opt.solve(model)
        Variables = getVariables_panda_indexed(model)
        ener = Variables['power'].pivot(index="TIMESTAMP", columns='TECHNOLOGIES', values='power')
        ener = ener.sum(axis=0)
        ener = pd.DataFrame(ener, columns={'energy'})
        alpha=ener.loc['pac']/(ener.loc['CCG']+ener.loc['TAC']+ener.loc['pac'])
        alpha_list=alpha_list+[alpha['energy']]
        print(alpha_list)

alpha_matrice=np.zeros([11,10])

for i in np.arange(0,11,1) :
    alpha_matrice[i,:]=alpha_list[10*i:10*i+10]

alpha_df=pd.DataFrame(alpha_matrice,index=variation_CAPEX_H2,columns=variation_prix_GazNat)

fig = go.Figure(data=go.Heatmap(z=alpha_df,x=variation_prix_GazNat,y=['-50%','-40%','-30%','-20%','-10%','0','+10%','+20%','+30%','+40%','+50%']))
fig.update_layout(title='Proportion NRJ PAC / NRJ TAC + CCG',xaxis_title='Prix du gaz €/MWh',yaxis_title='Variation en % des CAPEX électroliseurs et PAC par rapport à la référence')
plotly.offline.plot(fig, filename='Abaque aplha.html')

#endregion

