'''
Created on May 23, 2019

@author: robovie


read the csv log files from the data-driven memory training runs and create graphs for each column
copied from analysis2
for visualizing output of actionPrediction17
'''


import os
from collections import OrderedDict
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy

import tools


#expLogDir = tools.logDir+"/2019-12-12_20-45-35_actionPrediction17_dbl_nodbinput"

#expLogDir = tools.logDir+"/2019-12-13_15-16-01_actionPrediction17_dbl_nodbinput_classbalance"

#expLogDir = tools.logDir+"/2019-12-13_16-05-14_actionPrediction17_dbl_withdbinput_classbalance"

expLogDir = tools.logDir+"/2019-12-16_16-51-41_actionPrediction17_dbl"
expLogDir = tools.logDir+"/2019-12-17_20-07-16_actionPrediction17_dbl"

#expLogDir = tools.logDir+"/2019-12-18_17-57-27_actionPrediction17_dbl" # with attr index weights

#expLogDir = tools.logDir+"/2019-12-18_19-25-34_actionPrediction17_dbl" # SGD with momentum


expLogDir = tools.logDir+"/2019-12-19_18-36-33_actionPrediction17_dbl" # with no attr label instances with weight 0, the rest with weight 1

expLogDir = tools.logDir+"/2019-12-19_19-56-59_actionPrediction17_dbl" # with no attr label instances with weight 0, the rest with weight 10

expLogDir = tools.logDir+"/2019-12-20_13-06-08_actionPrediction17_dbl" # with no attr label instances with weight 0, the rest with weight 2.5*balance_weight

#expLogDir = tools.logDir+"/2019-12-20_16-43-05_actionPrediction17_dbl" # with no attr label instances with weight 0, the rest with weight 2.5*balance_weight, two input encoders

expLogDir = tools.logDir+"/2019-12-23_16-52-37_actionPrediction17_dbl"  # with no attr label instances with weight 0, the rest with weight 2.5*balance_weight, 0 and 1 DB contents


expLogDir = tools.logDir+"/2019-12-23_17-40-41_actionPrediction17_dbl"  # with no attr label instances with weight 0, the rest with weight 2.5*balance_weight, 0 and 1 DB contents, reinitialize non DB-addr and opt at 200 epochs



expLogDir = tools.logDir+"/2020-01-09_22-31-00_actionPrediction17_dbl"

expLogDir = tools.logDir+"/2020-01-20_15-57-56_actionPrediction18_dbl/proposed"




def plot_2_conditions_3_metrics(runIdToData, runIds, metric1Name, metric2Name, metric3Name):
    
    fig, axes = plt.subplots(3, 2, sharex='col', sharey='row')
    
    
    cmap = plt.get_cmap("tab20")
    colors = list(cmap.colors)
    runIdToColor = {}
    
    ymax = 1.05
    
    i = 0
    for runId in runIdToData:
        runIdToColor[runId] = colors[i % len(colors)]
        i += 1
        
        
        # training
        # graph Cost Ave
        if metric1Name == "Cost Ave":
            metric1Ymax = 100
        else:
            metric1Ymax = ymax
        
        runIdToData[runId].plot(x="Epoch", y="Train {} ({})".format(metric1Name, runId), ax=axes[0,0],
                                color=runIdToColor[runId],
                                legend=None,
                                label=runId) 
                                #ylim=[0, metric1Ymax])
        
        # graph Substring Correct All
        runIdToData[runId].plot(x="Epoch", y="Train {} ({})".format(metric2Name, runId), ax=axes[1,0],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, ymax])
        
        
        # graph Substring Correct Ave
        runIdToData[runId].plot(x="Epoch", y="Train {} ({})".format(metric3Name, runId), ax=axes[2,0],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, ymax])
        
            
        # testing
        
        # graph Cost Ave
        if metric1Name == "Cost Ave":
            yColName = "Test {}({})".format(metric1Name, runId) # there's a typo in these column names (missing space)
            metric1Ymax = 100
        else:
            yColName = "Test {} ({})".format(metric1Name, runId)
            metric1Ymax = ymax
        
        
        runIdToData[runId].plot(x="Epoch", y=yColName, ax=axes[0,1],
                                color=runIdToColor[runId],
                                legend=None) 
                                #ylim=[0, metric1Ymax])
        
        
        # graph Substring Correct All
        runIdToData[runId].plot(x="Epoch", y="Test {} ({})".format(metric2Name, runId), ax=axes[1,1],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, ymax])
        
        
        # graph Substring Correct Ave
        runIdToData[runId].plot(x="Epoch", y="Test {} ({})".format(metric3Name, runId), ax=axes[2,1],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, ymax])
    
    
    
    plt.legend(runIds,
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
    
    
        #
    # plot the prob for teacher forcing
    #
    for runId in runIds:
        """
        try:
            axes2_00 = axes[0,0].twinx()  # instantiate a second axes that shares the same x-axis
            axes2_10 = axes[1,0].twinx()
            axes2_20 = axes[2,0].twinx()
            
            axes2_01 = axes[0,1].twinx()
            axes2_11 = axes[1,1].twinx()
            axes2_21 = axes[2,1].twinx()
            
            
            axes2_00.set_ylim(0, 1)
            axes2_10.set_ylim(0, 1)
            axes2_20.set_ylim(0, 1)
            
            axes2_01.set_ylim(0, 1)
            axes2_11.set_ylim(0, 1)
            axes2_21.set_ylim(0, 1)
            
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_00,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_10,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_20,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_01,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_11,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_21,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            
            axes2_11.set_ylabel("Teacher Forcing Decay Schedule", rotation=90, size='medium')
            
        except:
            pass
        """
    
    
        
    # Adjust the scaling factor to fit your legend text completely outside the plot
    # (smaller value results in more space being made for the legend)
    
    # for 120 run gridsearch
    #plt.subplots_adjust(bottom=.3)
    
    # for 8 runs
    plt.subplots_adjust(bottom=.2)
    
    
    cols = ["Training", "Testing"]
    hits = [metric1Name, metric2Name, metric3Name]
    
    
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)
    
    for ax, row in zip(axes[:,0], hits):
        ax.set_ylabel(row, rotation=90, size='medium')
    
    
    
    plt.subplots_adjust(wspace=.1, hspace=.05)
    #fig.tight_layout()
    
    
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i,j].xaxis.set_tick_params(which='both', direction="in", length=5)
            axes[i,j].yaxis.set_tick_params(which='both', direction="in", length=5)
            
    
    
    plt.show()



def plot_2_conditions_4_metrics(runIdToData, runIds, metric1Name, metric2Name, metric3Name, metric4Name):
    
    fig, axes = plt.subplots(4, 2, sharex='col', sharey='row')
    
    
    cmap = plt.get_cmap("tab20")
    colors = list(cmap.colors)
    runIdToColor = {}
    
    ymax = 1.05
    
    i = 0
    for runId in runIdToData:
        runIdToColor[runId] = colors[i % len(colors)]
        i += 1
        
        
        # training
        # graph Cost Ave
        if metric1Name == "Cost Ave":
            metric1Ymax = 100
        else:
            metric1Ymax = ymax
        
        runIdToData[runId].plot(x="Epoch", y="Train {} ({})".format(metric1Name, runId), ax=axes[0,0],
                                color=runIdToColor[runId],
                                legend=None,
                                label=runId) 
                                #ylim=[0, metric1Ymax])
        
        # graph Substring Correct All
        runIdToData[runId].plot(x="Epoch", y="Train {} ({})".format(metric2Name, runId), ax=axes[1,0],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, ymax])
        
        
        # graph Substring Correct Ave
        runIdToData[runId].plot(x="Epoch", y="Train {} ({})".format(metric3Name, runId), ax=axes[2,0],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, ymax])
        
        
        runIdToData[runId].plot(x="Epoch", y="Train {} ({})".format(metric4Name, runId), ax=axes[3,0],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, ymax])
        
            
        # testing
        
        # graph Cost Ave
        if metric1Name == "Cost Ave":
            yColName = "Test {}({})".format(metric1Name, runId) # there's a typo in these column names (missing space)
            metric1Ymax = 100
        else:
            yColName = "Test {} ({})".format(metric1Name, runId)
            metric1Ymax = ymax
        
        
        runIdToData[runId].plot(x="Epoch", y=yColName, ax=axes[0,1],
                                color=runIdToColor[runId],
                                legend=None) 
                                #ylim=[0, metric1Ymax])
        
        
        # graph Substring Correct All
        runIdToData[runId].plot(x="Epoch", y="Test {} ({})".format(metric2Name, runId), ax=axes[1,1],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, ymax])
        
        
        # graph Substring Correct Ave
        runIdToData[runId].plot(x="Epoch", y="Test {} ({})".format(metric3Name, runId), ax=axes[2,1],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, ymax])
        
        
        runIdToData[runId].plot(x="Epoch", y="Test {} ({})".format(metric4Name, runId), ax=axes[3,1],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, ymax])
    
    
    
    plt.legend(runIds,
               loc="upper center",   # Position of legend
               borderaxespad=0.1,    # Small spacing around legend box
               title="Run Parameters - rs (random seed), ct (camera temp.), at (attribute temp.)",
               
               # for 120 run gridsearch
               #ncol=12,
               #bbox_to_anchor=(-0.05, -0.2)
               
               # for 8 runs
               ncol=4,
               bbox_to_anchor=(0, -.5)
               )
    
    
    #
    # plot the prob for teacher forcing
    #
    for runId in runIds:
        """
        try:
            axes2_00 = axes[0,0].twinx()  # instantiate a second axes that shares the same x-axis
            axes2_10 = axes[1,0].twinx()
            axes2_20 = axes[2,0].twinx()
            
            axes2_01 = axes[0,1].twinx()
            axes2_11 = axes[1,1].twinx()
            axes2_21 = axes[2,1].twinx()
            
            
            axes2_00.set_ylim(0, 1)
            axes2_10.set_ylim(0, 1)
            axes2_20.set_ylim(0, 1)
            
            axes2_01.set_ylim(0, 1)
            axes2_11.set_ylim(0, 1)
            axes2_21.set_ylim(0, 1)
            
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_00,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_10,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_20,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_01,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_11,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_21,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            
            axes2_11.set_ylabel("Teacher Forcing Decay Schedule", rotation=90, size='medium')
            
        except:
            pass
        """
    
    
        
    # Adjust the scaling factor to fit your legend text completely outside the plot
    # (smaller value results in more space being made for the legend)
    
    # for 120 run gridsearch
    #plt.subplots_adjust(bottom=.3)
    
    # for 8 runs
    plt.subplots_adjust(bottom=.2)
    
    
    cols = ["Training", "Testing"]
    hits = [metric1Name, metric2Name, metric3Name, metric4Name]
    
    
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)
    
    for ax, row in zip(axes[:,0], hits):
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

runDirContents = os.listdir(expLogDir)
runIds = []

for rdc in runDirContents:
    if "." not in rdc:
        runIds.append(rdc)

runIds.sort()


temp = []
"""
for rdn in runIds:
    
    #if "ct3_" in rdn and rdn.endswith("at2"):
    if rdn.endswith("tf1.0"):
        temp.append(rdn)

runIds = temp
"""

# this will contain the data from all the csv log files
runIdToData = {}

for rdn in runIds:
    
    #runIdToData[rdn] = pd.read_csv("{}/{}/session_log_{}.csv".format(expLogDir, rdn, rdn))
    runIdToData[rdn] = pd.read_csv("{}/fold_log_{}.csv".format(expLogDir, rdn))




#
# graph the data
#
#plot_2_conditions_3_metrics(runIdToData, runIds, "Cost Ave", "DB Substring Correct Ave", "DB Substring Correct All")

#plot_2_conditions_3_metrics(runIdToData, runIds, "Cam. Address Correct", "Attr. Address Correct", "Both Addresses Correct")

plot_2_conditions_4_metrics(runIdToData, runIds, "Cost Ave", "Action ID Correct", "Camera Index Correct", "Attribute Index Exact Match")

#plot_2_conditions_3_metrics(runIdToData, runIds, "Cost Ave", "Action ID Correct", "Attribute Index Correct")

