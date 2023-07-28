
import os
os.sys.path.append(r'../')
from Functions.f_extract_data import extract_costs
from Functions.f_graphicTools import plot_costs2050
from Models.scenario_creation import scenarioDict

#First : execute runPACA.py BM_80 Ref BM_100

outputPath='../Data/output/'

ScenarioName='BM_80'
outputFolder=outputPath+ScenarioName+'_PACA'
df=extract_costs(scenarioDict[ScenarioName],outputFolder)
df['AEL BM=80€']=df.pop('AEL')
df['SMR BM=80€']=df.pop('SMR')


for l in df.keys() : df[l]=df[l].reset_index().loc[df[l].reset_index().YEAR==2050].set_index('YEAR')

ScenarioName='Ref'
outputFolder=outputPath+ScenarioName+'_PACA'
df2=extract_costs(scenarioDict[ScenarioName],outputFolder)
df2['AEL BM=90€']=df2.pop('AEL')
df2['SMR BM=90€']=df2.pop('SMR')


for l in df2.keys() : df2[l]=df2[l].reset_index().loc[df2[l].reset_index().YEAR==2050].set_index('YEAR')

df.update(df2)

ScenarioName='BM_100'
outputFolder=outputPath+ScenarioName+'_PACA'
df3=extract_costs(scenarioDict[ScenarioName],outputFolder)
df3['AEL BM=100€']=df3.pop('AEL')
df3['SMR BM=100€']=df3.pop('SMR')


for l in df3.keys() : df3[l]=df3[l].reset_index().loc[df3[l].reset_index().YEAR==2050].set_index('YEAR')

df.update(df3)

plot_costs2050(df,outputPath,comparaison=True)