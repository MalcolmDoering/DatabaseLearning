'''
Created on May 23, 2019

@author: robovie


aggregate the crossvalidation results from actionPrediction18


'''


import os
from collections import OrderedDict
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import traceback
from matplotlib.pyplot import subplots

import tools


sessionDir = sessionDir = tools.create_session_dir("analysis8")

#expLogName = "2020-01-16_19-56-41_actionPrediction18_dbl" # with speech cluster loss weights
#expLogName = "2020-01-17_17-43-48_actionPrediction18_dbl" # without speech cluster loss weights

expLogName = "2020-01-21_16-55-26_actionPrediction18_dbl" # with speech cluster loss weights, ignore junk, training error fixed
expLogName = "2020-01-21_18-57-27_actionPrediction18_dbl" # no speech cluster loss weights, ignore junk, training error fixed


expLogName = "2020-01-22_20-34-26_actionPrediction18_dbl" # silence speech clsuter error fixed, 11 folds, with speech cluster loss weights

expLogDir = tools.logDir+"/"+expLogName


#
# read the loss and accuracy log csvs for each fold
#
conditionToMetrics = {}

conditionToMetrics["proposed"] = ["Loss Ave", 
                                  "Shopkeeper Speech Cluster Loss Ave", 
                                  "Camera Index Loss Ave", 
                                  "Attribute Index Loss Ave", 
                                  "Location Loss Ave",
                                  "Spatial State Loss Ave", 
                                  "State Target Loss Ave", 
                                  "Speech Cluster Correct",
                                  "Camera Index Correct", 
                                  "Attribute Index Exact Match", 
                                  "Attribute Index Jaccard Index", 
                                  "Location Correct", 
                                  "Spatial State Correct", 
                                  "State Target Correct"]

conditionToMetrics["baseline1"] = ["Loss Ave", 
                                   "Shopkeeper Speech Cluster Loss Ave", 
                                   "Location Loss Ave",
                                   "Spatial State Loss Ave",
                                   "State Target Loss Ave",
                                   "Speech Cluster Correct",
                                   "Location Correct",
                                   "Spatial State Correct",
                                   "State Target Correct"]


datasets = ["Training", "Validation", "Testing"]


def aggregate_metrics(condition):
    print("started processing {}...".format(condition))
    
    metrics = conditionToMetrics[condition]
    conditionLogDir = expLogDir+"/"+condition
    
    newCsvFieldnames = []
    newCsvFieldnames.append("Epoch")
    
    for ds in datasets:
        for m in metrics:
            newCsvFieldnames.append("{} {} Average".format(ds, m.replace(" Ave", "")))
            newCsvFieldnames.append("{} {} SD".format(ds, m.replace(" Ave", "")))
    
    
    runDirContents = os.listdir(conditionLogDir)
    runDirContents.sort()
    
    runIds = []
    for rdc in runDirContents:
        if "." not in rdc:
            runIds.append(rdc)
    
    
    # this will contain the data from all the csv log files
    print("loading data...")
    
    runIdToData = {}
    
    for rId in runIds:
        
        df = pd.read_csv("{}/fold_log_{}.csv".format(conditionLogDir, rId))
        
        epochs = df["Epoch"].tolist()
        
        runIdToData[rId] = df
        
        """
        with open("{}/fold_log_{}.csv".format(conditionLogDir, rId)) as csvfile:
            reader = csv.DictReader(csvfile)
            
            data = []
            
            for row in reader:
                data.append(row)
                
                numEpochs = max(numEpochs, int(row["Epoch"]))
            
            runIdToData[rId] = data
        """
    
    #
    # compute averages and SDs
    #
    print("computing averages and SDs...")
    newCsvRows = []
    
    for e in epochs:
        row = {}
        row["Epoch"] = e
        
        for ds in datasets:
            for m in metrics:
                
                # compute average of scores from each fold
                foldScores = []
                
                for rId in runIds:
                    
                    df = runIdToData[rId]
                    colName = "{} {} ({})".format(ds, m, rId)
                    
                    epochRow = df.loc[df["Epoch"] == e]
                    
                    score = pd.array(epochRow[colName])[0]
                    
                    foldScores.append(score)
                    
                # compute average and SD
                mAve = np.mean(foldScores)
                mStd = np.std(foldScores)
                
                #mAve = e
                #mStd = e
                
                row["{} {} Average".format(ds, m.replace(" Ave", ""))] = mAve
                row["{} {} SD".format(ds, m.replace(" Ave", ""))] = mStd
        
        newCsvRows.append(row)
    
    
    #
    # save to csv file
    #
    print("saving...")
    with open(sessionDir+"/{}_aggregated_log.csv".format(condition), "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, newCsvFieldnames)
        writer.writeheader()
        writer.writerows(newCsvRows)
    
    print("finished processing {}.".format(condition))
    
    
    #
    # convert into dataframe and add the condition to the column names
    #
    df = pd.DataFrame(newCsvRows)
    
    newColNames = {}
    
    for cn in df.columns[1:]:
        newColNames[cn] = "{} {}".format(condition.capitalize(), cn)
    
    df = df.rename(columns=newColNames)
    
    return df


def create_evaluation_csv(conditions, epoch, numInteractionsPerFold):
    print("creating csv for evaluation...")
    
    conditionToData = {}
    newFieldnames = []
    
    for condition in conditions:
        conditionLogDir = expLogDir+"/"+condition
        runDirContents = os.listdir(conditionLogDir)
        runDirContents.sort()
        
        runIds = []
        for rdc in runDirContents:
            if "." not in rdc:
                runIds.append(rdc)
        
        foldToData = {}
        
        for rId in runIds:
            
            data = []
            
            # load the data
            with open("{}/{}/{}_all_outputs.csv".format(conditionLogDir, rId, epoch)) as csvfile:
                reader = csv.DictReader(csvfile)
                
                # fieldnames
                fieldnames = reader.fieldnames
                for fn in fieldnames:
                    
                    if ((not fn.startswith("TARG_")) and (not fn.startswith("PRED_"))):
                        if (fn not in newFieldnames):
                            newFieldnames.append(fn)
                    else:
                        newFieldName = "{}_{}".format(condition.upper(), fn)
                        if (newFieldName not in newFieldnames):
                            newFieldnames.append(newFieldName)
                
                # data
                for row in reader:
                    data.append(row)
            
            # get the first numInteractionsPerFold of the testing data
            data = [d for d in data if d["SET"] == "TEST"]
            
            # this assumes the data is already ordered
            firstTrialId = int(data[0]["TRIAL"])
            trials = list(range(firstTrialId, firstTrialId+numInteractionsPerFold))
            
            data = [d for d in data if int(d["TRIAL"]) in trials]
            
            foldToData[rId] = data
        
        conditionToData[condition] = foldToData
    
    
    #
    # save the data in a csv
    #
    
    # fieldnames
    for condition in conditions[::-1]:
        newFieldnames = ["{}_FOLD_ID".format(condition.upper())] + newFieldnames
    
    newFieldnames = ["DISPLAY_ID"] + newFieldnames
    
    for condition in conditions:
        newFieldnames.append("{}_CORRECT_RATER1".format(condition.upper()))
        newFieldnames.append("{}_CORRECT_RATER2".format(condition.upper()))
    
    
    # combine data from multiple conditions
    idToRow = {}
    
    for condition in conditionToData:
        for rId in conditionToData[condition]:
            for row in conditionToData[condition][rId]:
                
                uniqueId = int(row["ID"])
                
                if uniqueId not in idToRow:
                    
                    newRow = {}
                    newRow["DISPLAY_ID"] = None # to be set at the end
                    
                    # so that evaluation can later be cross referenced with the original data
                    for fn in row:
                        if (not fn.startswith("TARG_") and not fn.startswith("PRED_")):
                            newRow[fn] = row[fn]
                    
                    idToRow[uniqueId] = newRow
                
                # add the data for this condition
                for fn in row:
                    if (fn.startswith("TARG_") or fn.startswith("PRED_")):
                        idToRow[uniqueId]["{}_{}".format(condition.upper(), fn)] = row[fn]
                
                # add fields for the ratings
                idToRow[uniqueId]["{}_FOLD_ID".format(condition.upper())] = rId
                idToRow[uniqueId]["{}_CORRECT_RATER1".format(condition.upper())] = "INCOMPLETE"
                idToRow[uniqueId]["{}_CORRECT_RATER2".format(condition.upper())] = "INCOMPLETE"
    
    
    # save
    with open(sessionDir+"/predictions-2.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, newFieldnames)
        writer.writeheader()
        
        displayId = 1
        
        for key in sorted(idToRow.keys()):
            idToRow[key]["DISPLAY_ID"] = displayId
            displayId += 1
            
            writer.writerow(idToRow[key])


#
# load and process the data
#
print("preparing data for human evaluation...")

create_evaluation_csv(conditions=["proposed", "baseline1"], epoch=300, numInteractionsPerFold=4)




print("aggregating the data for visualization...")

conditionToDfs = {}
conditionToDfs["proposed"] = aggregate_metrics("proposed")
conditionToDfs["baseline1"] = aggregate_metrics("baseline1")


#
# create graphs
#
print("creating graphs...")

df = pd.concat(conditionToDfs.values(), axis=1)
df = df.loc[:,~df.columns.duplicated()]

for c in df.columns:
    print(c)

print(df)

allMetrics = []
for condition in conditionToMetrics:
    for metric in conditionToMetrics[condition]:
        if metric not in allMetrics:
            allMetrics.append(metric)

conditionToColors = {"proposed": ["#641E16", "#E6B0AA", "#C0392B"],
                     "baseline1": ["#145A32", "#A9DFBF", "#27AE60"]
                     }

for m in allMetrics:
    
    colNames = []
    labels = []
    colors = []
    
    dataToPlot = []
    
    for condition in conditionToDfs:
        if m not in conditionToMetrics[condition]:
            continue
           
        colNames += ["{} {} {} Average".format(condition.capitalize(), ds, m.replace(" Ave", "")) for ds in datasets]
        labels += ["{} {}".format(condition.capitalize(), ds) for ds in datasets]
        colors += conditionToColors[condition]
    
    if len(colNames) > 0:
        fig, ax = subplots()
        df.plot(x="Epoch", y=colNames, kind="line", title="{} Average".format(m.replace(" Ave", "")), ax=ax, color=colors)
        ax.legend(labels);
        
        fig.savefig((sessionDir+"/{} Average.png".format(m.replace(" Ave", ""))), format="png")
        #plt.show()




