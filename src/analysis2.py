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

#expLogDir = "C:/Users/robovie/eclipse-log/2019-05-24_19-00-07_actionPrediction13_dbl - GT DB entries" # 10 databases, GT database entries given
#expLogDir = "C:/Users/robovie/eclipse-log/2019-05-31_11-34-31_actionPrediction13_dbl" # 10 databases, GT database entries given, and the new metrics, only shopkeeper responses to question about price included in data

#expLogDir = "C:/Users/robovie/eclipse-log/2019-06-03_18-17-08_actionPrediction13_dbl" # 10 databases, GT database entries given, DB entries padded with 0 vecs
#expLogDir = "C:/Users/robovie/eclipse-log/2019-06-05_13-47-27_actionPrediction13_dbl" # 10 databases, GT database entries given, DB entries padded with 0 vecs

#expLogDir = "C:/Users/robovie/eclipse-log/2019-06-06_16-33-38_actionPrediction13_dbl" # 10 databases, GT database entries given, DB entries padded with 0 vecs, reduced batch size and unrandomized training instance order
#expLogDir = "C:/Users/robovie/eclipse-log/2019-06-06_16-55-22_actionPrediction13_dbl" # 10 databases, GT database entries given, DB entries padded with 0 vecs, reduced batch size and randomized training instance order

#expLogDir = "C:/Users/robovie/eclipse-log/2019-06-07_15-05-37_actionPrediction13_dbl" # 10 databases all data, GT database entries given, DB entries padded with 0 vecs, reduced batch size and unrandomized training instance order



#expLogDir = "C:/Users/robovie/eclipse-log/2019-06-10_14-30-33_actionPrediction13_dbl" # 10 databases, relu and softmax addressing, DB entries padded with 0 vecs, reduced batch size and unrandomized training instance order, softmaxed over weighted DB entry sums




#expLogDir = "C:/Users/robovie/eclipse-log/2019-05-31_15-35-15_actionPrediction13_dbl" # 2 databases, GT database entries given, and the new metrics, only shopkeeper responses to question about price included in data



#expLogDir = "C:/Users/robovie/eclipse-log/2019-05-31_17-54-56_actionPrediction13_dbl" # 2 databases, temp grid search, and the new metrics, only shopkeeper responses to question about price included in data
#expLogDir = "C:/Users/robovie/eclipse-log/2019-06-03_13-03-18_actionPrediction13_dbl" # 10 databases, temp grid search, and the new metrics, only shopkeeper responses to question about price included in data

#expLogDir = "C:/Users/robovie/eclipse-log/2019-06-07_11-47-14_actionPrediction13_dbl" # 10 databases, temp grid search, and the new metrics, only shopkeeper responses to question about price included in data, 50 batch unrandomized


#expLogDir = "C:/Users/robovie/eclipse-log/2019-05-27_12-29-06_actionPrediction13_dbl" # 10 databases, sharpening used for addressing
#expLogDir = "C:/Users/robovie/eclipse-log/2019-05-28_14-34-02_actionPrediction13_dbl" # 10 databases, sharpening used for addressing, and the new metrics




# 10 databases
# only S_ANSWERS_QUESTION_ABOUT_FEATURE price included
# tanh and softmax addressing
# adam .0001 learning rate
# DB entries padded with 0 vecs
# 50 reduced batch size and unrandomized training instance order
# only utt input to addressing layer and decoder initialization
# result: attr addresses learned but not camera addresses (because loc was not input to addressing layer)
#expLogDir = "C:/Users/robovie/eclipse-log/2019-06-06_17-30-06_actionPrediction13_dbl" 

# 10 databases
# tanh and softmax addressing
# adam .001 learning rate
# DB entries padded with 0 vecs
# 50 reduced batch size and unrandomized training instance order
# combined loc and utt input to addressing layer, loc layer, and decoder initialization ***
# result: camera addresses are learned but not
# Q: why is the attr loc not learned? event though it was learned in the previous experiment? The only thing that changed besides the inputs is the learning rate...
expLogDir = "C:/Users/robovie/eclipse-log/2019-06-12_18-48-35_actionPrediction13_dbl" 

# 10 databases
# tanh and softmax addressing
# adam .0001 *** learning rate
# DB entries padded with 0 vecs
# 50 reduced batch size and unrandomized training instance order
# combined loc and utt input to addressing layer, loc layer, and decoder initialization
# result:
#expLogDir = "C:/Users/robovie/eclipse-log/" 


def plot_2_conditions_3_metrics(runIdToData, runDirNames, metric1Name, metric2Name, metric3Name):
    
    fig, axes = plt.subplots(3, 2, sharex='col', sharey='row')
    
    
    cmap = plt.get_cmap("tab20")
    colors = list(cmap.colors)
    runIdToColor = {}
    
    
    i = 0
    for runId in runIdToData:
        runIdToColor[runId] = colors[i % len(colors)]
        i += 1
    
    
    for runId in runDirNames:
        
        # training
        # graph Cost Ave
        runIdToData[runId].plot(x="Epoch", y="Train {} ({})".format(metric1Name, runId), ax=axes[0,0],
                                color=runIdToColor[runId],
                                legend=None)
        
        
        # graph Substring Correct All
        runIdToData[runId].plot(x="Epoch", y="Train {} ({})".format(metric2Name, runId), ax=axes[1,0],
                                color=runIdToColor[runId],
                                legend=None)
        
        
        # graph Substring Correct Ave
        runIdToData[runId].plot(x="Epoch", y="Train {} ({})".format(metric3Name, runId), ax=axes[2,0],
                                color=runIdToColor[runId],
                                legend=None)
        
            
        # testing
        
        # graph Cost Ave
        if metric1Name == "Cost Ave":
            yColName = "Test {}({})".format(metric1Name, runId) # there's a typo in these column names (missing space)
        else:
            yColName = "Test {} ({})".format(metric1Name, runId)
        
        
        runIdToData[runId].plot(x="Epoch", y=yColName, ax=axes[0,1],
                                color=runIdToColor[runId],
                                legend=None)
        
        
        
        # graph Substring Correct All
        runIdToData[runId].plot(x="Epoch", y="Test {} ({})".format(metric2Name, runId), ax=axes[1,1],
                                color=runIdToColor[runId],
                                legend=None)
        
        
        # graph Substring Correct Ave
        runIdToData[runId].plot(x="Epoch", y="Test {} ({})".format(metric3Name, runId), ax=axes[2,1],
                                color=runIdToColor[runId],
                                legend=None)
    
    
    plt.legend(runDirNames,
               loc="upper center",   # Position of legend
               borderaxespad=0.1,    # Small spacing around legend box
               title="Run Parameters - rs (random seed), ct (camera temp.), at (attribute temp.)",
               
               # for 120 run gridsearch
               #ncol=12,
               #bbox_to_anchor=(-0.05, -0.2)
               
               # for 8 runs
               ncol=8,
               bbox_to_anchor=(0, -.5)
               )
    
    
    # Adjust the scaling factor to fit your legend text completely outside the plot
    # (smaller value results in more space being made for the legend)
    
    # for 120 run gridsearch
    #plt.subplots_adjust(bottom=.3)
    
    # for 8 runs
    plt.subplots_adjust(bottom=.2)
    
    
    cols = ["Training", "Testing"]
    rows = [metric1Name, metric2Name, metric3Name]
    
    
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

plot_2_conditions_3_metrics(runIdToData, runDirNames, "Cost Ave", "DB Substring Correct Ave", "DB Substring Correct All")

plot_2_conditions_3_metrics(runIdToData, runDirNames, "Cam. Address Correct", "Attr. Address Correct", "Both Addresses Correct")




