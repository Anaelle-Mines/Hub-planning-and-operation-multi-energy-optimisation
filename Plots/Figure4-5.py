#region Importation of modules
import os
os.sys.path.append(r'../')
from Functions.f_graphicTools import plot_capacity, plot_energy

#endregion

#First : execute runPACA.py Ref

outputPath='../Data/output/'
scenarioName='Ref_PACA'
outputFolder=outputPath+scenarioName

plot_capacity(outputFolder,LoadFac=False)
plot_energy(outputFolder)