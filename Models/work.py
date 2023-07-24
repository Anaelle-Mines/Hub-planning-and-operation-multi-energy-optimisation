import os
import numpy as np
import pandas as pd
import csv
os.sys.path.append(r'../')
from Functions.f_multiResourceModels import *
from Functions.f_optimization import *
from scenario_creation import scenarioDict
from scenario_creation_REsensibility import scenarioDict_RE

scenarioDict.update(scenarioDict_RE)

print(scenarioDict.keys())