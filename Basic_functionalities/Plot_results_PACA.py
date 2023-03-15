
import os
import numpy as np
import pandas as pd
import csv

from Functions.f_graphicTools import *
from Basic_functionalities.scenario_creation import scenarioDict

outputPath='Data/output/'

ScenarioName='test' # 'Ref', 'ElecOnly', 'SmrOnly','Grid','EnR',
outputFolder=outputPath+ScenarioName+'_Base_PACA'

# plot installed capacity by technologies
capacity=plot_capacity(outputFolder)

# plot H2 production by technologies
energy=plot_energy(outputFolder)

# calculation of charge factors
chargeFactors=(energy/(capacity*8.760)).fillna(0)

# plot evolution of SMR techno
plot_evolve(outputFolder)

# plot electricity weekly management
plot_elecMean(scenarioDict[ScenarioName],outputFolder)

# plot H2 weekly management
plot_H2Mean(scenarioDict[ScenarioName],outputFolder)

# plot stock level
plot_stock(outputFolder)

# plot carbon emissions
plot_carbon(outputFolder)

# plot costs
plot_costs(extract_costs(scenarioDict[ScenarioName],outputFolder),outputFolder)


