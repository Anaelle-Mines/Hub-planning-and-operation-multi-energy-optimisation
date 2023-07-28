
import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


outputPath='../Data/output/'

YEAR=[2020,2030,2040,2050]
x=np.arange(len(YEAR))
meth=interp1d(YEAR,[3.5]*4,fill_value=(3.5,3.5),bounds_error=False)
st=interp1d(YEAR,[0,51,143.5,185],fill_value=(0,185),bounds_error=False)
ref=interp1d(YEAR,[102.5,78,20.5,20.5],fill_value=(102.5,20.5),bounds_error=False)
cl=interp1d(YEAR,[11.3]*4,fill_value=(3.5,3.5),bounds_error=False)

methanol=[meth(y+5) for y in YEAR]
steel=[st(y+5) for y in YEAR]
refinery=[ref(y+5) for y in YEAR]
chlore=[cl(y+5) for y in YEAR]

conso=[x+y+z for x,y,z in zip(methanol,steel,refinery)]
prod=chlore

df={'Methanol':methanol,'Refinery': refinery,'Steel' : steel}

net=[x-y for x,y in zip(conso,prod)]

fig, ax = plt.subplots(figsize=(4.6,3))

width = 0.4
col = plt.cm.tab20c


def kt_to_TWh(x):
    return x*33.33/1000

def TWh_to_kt(x):
    return x*1000/33.33

# Create light blue Bars

a=[[0,0,0,0]]
for n,i in enumerate(df.keys()) :
    a.append([x+y for x,y in zip(df[i],a[n])])
    print(a)
    plt.bar(x , df[i], width,bottom=a[n], color=col(4*n+1), label=i, zorder=2)

secax = ax.secondary_yaxis('right', functions=(kt_to_TWh, TWh_to_kt))
secax.set_ylabel('(TWh/yr)')

#get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()
#specify order of items in legend
order = [2,1,0]

plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order]) 
plt.ylabel("Hydrogen demand (kt/yr)")
plt.xticks(x, ['2020-2030', '2030-2040', '2040-2050', '2050-2060'])

plt.grid(axis='y', alpha=0.5, zorder=1)

plt.savefig(outputPath + '/Figure3.png')
plt.show()
