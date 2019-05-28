'''
Created on May 23, 2019

@author: robovie


read the csv log files from the data-driven memory training runs and create graphs for each column

'''


import os
from collections import OrderedDict
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy


#plt.style.use('seaborn-white')


#expLogDir = "C:/Users/robovie/eclipse-log/2019-05-17_11-30-24_actionPrediction12_dbl - save" # 2 training databases, presented in 20190524 meeting
#expLogDir = "C:/Users/robovie/eclipse-log/2019-05-21_14-54-06_actionPrediction13_dbl - save" # 10 training databases, presented in 20190524 meeting

#expLogDir = "C:/Users/robovie/eclipse-log/2019-05-24_19-00-07_actionPrediction13_dbl" # 10 databases, GT database entries given

expLogDir = "C:/Users/robovie/eclipse-log/2019-05-27_12-29-06_actionPrediction13_dbl" # 10 databases, sharpening used for addressing



#
# read the data 
#

runDirNames = os.listdir(expLogDir)
runDirNames.sort()

# this will contain the data from all the csv log files
runIdToData = {}

for rdn in runDirNames:
    
    runIdToData[rdn] = pd.read_csv("{}/{}/session_log_{}.csv".format(expLogDir, rdn, rdn))



#
# graph the data
#

"""
Train Cost Ave *
Train Cost SD 
Train DB Substring Correct All *
Train DB Substring Correct Ave *
Train DB Substring Correct SD

Test Cost Ave
Test Cost SD
Test DB Substring Correct All
Test DB Substring Correct Ave
Test DB Substring Correct SD
"""


#fig0, axes0 = plt.subplots()

fig, axes = plt.subplots(3, 2, sharex='col', sharey='row')


cmap = plt.get_cmap("tab10")
colors = cmap.colors
runIdToColor = {}

i = 0
for runId in runIdToData:
    runIdToColor[runId] = list(cmap.colors)[i]
    i += 1


for runId in runDirNames:
    
    # training
    
    # graph Cost Ave
    runIdToData[runId].plot(x="Epoch", y="Train Cost Ave ({})".format(runId), ax=axes[0,0],
                            color=runIdToColor[runId],
                            legend=None)
    
    
    # graph Substring Correct All
    runIdToData[runId].plot(x="Epoch", y="Train DB Substring Correct Ave ({})".format(runId), ax=axes[1,0],
                            color=runIdToColor[runId],
                            legend=None)
    
    
    # graph Substring Correct Ave
    runIdToData[runId].plot(x="Epoch", y="Train DB Substring Correct All ({})".format(runId), ax=axes[2,0],
                            color=runIdToColor[runId],
                            legend=None)
    
    #ax[2,0]
    
    
    
    
    # testing
    
    # graph Cost Ave
    runIdToData[runId].plot(x="Epoch", y="Test Cost Ave({})".format(runId), ax=axes[0,1],
                            color=runIdToColor[runId],
                            legend=None)
    
    
    
    # graph Substring Correct All
    runIdToData[runId].plot(x="Epoch", y="Test DB Substring Correct Ave ({})".format(runId), ax=axes[1,1],
                            color=runIdToColor[runId],
                            legend=None)
    
    
    # graph Substring Correct Ave
    runIdToData[runId].plot(x="Epoch", y="Test DB Substring Correct All ({})".format(runId), ax=axes[2,1],
                            color=runIdToColor[runId],
                            legend=None)
    

plt.legend(runDirNames,
           loc="lower center",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           title="Run Parameters - rs (random seed), ct (camera temp.), at (attribute temp.)",
           ncol=8,
           bbox_to_anchor=(0, -.5))

# Adjust the scaling factor to fit your legend text completely outside the plot
# (smaller value results in more space being made for the legend)
plt.subplots_adjust(bottom=.2)


cols = ["Training", "Testing"]
rows = ["Ave. Cost", "DB Substring Correct (Ave.)", "DB Substring Correct (All)"]


for ax, col in zip(axes[0], cols):
    ax.set_title(col)

for ax, row in zip(axes[:,0], rows):
    ax.set_ylabel(row, rotation=90, size='medium')



plt.subplots_adjust(wspace=.1, hspace=.05)
#fig.tight_layout()


for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
        axes[i,j].xaxis.set_tick_params(which='both', direction="in", length=5)
        axes[i,j].yaxis.set_tick_params(which='both', direction="in", length=5)
        


plt.show()



















