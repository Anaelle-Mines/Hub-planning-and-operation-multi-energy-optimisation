
import os
import numpy as np
import pandas as pd
import csv

from Functions.f_graphicTools import *
from Basic_functionalities.scenarios_ref_PACA import scenarioPACA

outputPath='Data/output/'

ScenarioName='Ref_v2' # Possible Choice : 'Ref', 'eSMR', 'EnR', 'Grid', 'GN', 'BG', 'EnR+','Crack'
outputFolder=outputPath+ScenarioName
outputFolderFr=outputPath+ScenarioName+'_Fr'

# plot installed capacity by technologies
capacity=plot_capacity(outputFolder)

# plot H2 production by technologies
energy=plot_energy(outputFolder)

# calculation of charge factors
chargeFactors=(energy/(capacity*8.760)).fillna(0)

# plot evolution of SMR techno
plot_evolve(outputFolder)

# plot electricity weekly management
plot_elecMean(scenarioPACA,outputFolder)

# plot H2 weekly management
plot_H2Mean(scenarioPACA,outputFolder)

# plot stock level
plot_stock(outputFolder)

# plot carbon emissions
plot_carbon(outputFolder)

# plot costs
plot_costs(scenarioPACA,outputFolder)

