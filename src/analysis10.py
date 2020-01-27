'''
Created on May 23, 2019

@author: robovie

for analysis of actionPrediction 18
count how often the silent speech cluster is correctly predicted
'''


import os
from collections import OrderedDict
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
from matplotlib.pyplot import subplots

import tools


#expLogName = "2020-01-20_15-57-56_actionPrediction18_dbl" # with speech cluster loss weights, ignore junk, training error fixed

expLogName = "2020-01-21_18-57-27_actionPrediction18_dbl" # no speech cluster loss weights, ignore junk, training error fixed

expLogDir = tools.logDir+"/"+expLogName



sessionDir = sessionDir = tools.create_session_dir("analysis10")



datasets = ["TRAIN", "VAL", "TEST"]


def compute_silence_accuracy(condition):
    print("computing silence accuracy for {}...".format(condition))
    
    conditionLogDir = expLogDir+"/"+condition
    runDirContents = os.listdir(conditionLogDir)
    runDirContents.sort()
    
    runIds = []
    for rdc in runDirContents:
        if "." not in rdc:
            runIds.append(rdc)
    
    silenceClusterId = None
    
    results = []
    
    for rId in runIds:
        
        foldDir = conditionLogDir+"/"+rId
        foldDirContents = os.listdir(foldDir)
        foldDirContents.sort()
        outputCsvs = [fn for fn in foldDirContents if "all_outputs.csv" in fn]
        
        for fn in outputCsvs:
            epoch = int(fn.split("_")[0])
            df = pd.read_csv(foldDir+"/"+fn, dtype=object, index_col=None)
            
            
            for ds in datasets:
                datasetDf = df[df.SET == ds]
                
                # make sure the silence speech cluster id is what we expect it to be
                temp = datasetDf[datasetDf.SHOPKEEPER_SPEECH.isna()]["TARG_SHOPKEEPER_SPEECH_CLUSTER_ID"].tolist()[0]
                
                if silenceClusterId == None:
                    silenceClusterId = temp
                else:
                    if temp != silenceClusterId:
                        print("WARNING: Differing silence cluster IDs:", silenceClusterId, temp)
                
                # find all instances where TARG_SHOPKEEPER_SPEECH_CLUSTER_ID is the silence cluster
                datasetDf = datasetDf[datasetDf.TARG_SHOPKEEPER_SPEECH_CLUSTER_ID == silenceClusterId]
                total = datasetDf.shape[0]
                
                # in what percent does TARG_SHOPKEEPER_SPEECH_CLUSTER_ID == PRED_OUTPUT_SHOPKEEPER_SPEECH_CLUSTER_ID
                datasetDf = datasetDf[datasetDf.TARG_SHOPKEEPER_SPEECH_CLUSTER_ID == datasetDf.PRED_OUTPUT_SHOPKEEPER_SPEECH_CLUSTER_ID]
                numCorrect = datasetDf.shape[0]
                
                percCorrect = numCorrect / total
                
                result = {}
                result["Epoch"] = epoch
                result["Fold"] = rId
                result["Set"] = ds
                result["Total GT Silence"] = total
                result["Num. Correct Silence"] = numCorrect
                result["Perc. Correct Silence"] = percCorrect
                
                results.append(result)
                
                print(foldDir, epoch, ds, total, numCorrect, percCorrect)
    
    # save
    results.sort(key=lambda x: x["Epoch"])
    
    fieldnames = ["Epoch", "Fold", "Set", "Total GT Silence", "Num. Correct Silence", "Perc. Correct Silence"]
    
    with open(sessionDir+"/{}_silence_accuracies.csv".format(condition), "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    

compute_silence_accuracy("proposed")

