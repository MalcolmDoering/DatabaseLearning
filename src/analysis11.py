'''
Created on May 23, 2019

@author: robovie

calculate the results of the human evaluation
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


RATER = 2

evalFilename = "predictions-2.csv" 
evalPath = tools.dataDir+"/"+evalFilename


sessionDir = sessionDir = tools.create_session_dir("analysis11")

conditions = ["proposed", "baseline1"]



#
# read in the data
#
df = pd.read_csv(evalPath, dtype=object, index_col=None)


#
# calculate
#
conditionToResult = {}

for condition in conditions:
    
    col = "{}_CORRECT_RATER{}".format(condition.upper(), RATER) 
    ratingCounts = df[col].value_counts()
    
    try:
        numGood = ratingCounts["GOOD"]
    except:
        numGood = 0
    
    try:
        numBad = ratingCounts["BAD"]
    except:
        numBad = 0
    
    try:
        numUnsure = ratingCounts["UNSURE"]
    except:
        numUnsure = 0
    
    try:
        numIncomplete = ratingCounts["INCOMPLETE"]
    except:
        numIncomplete = 0
    
    
    N = numGood + numBad
    percGood = numGood / N
    percBad = numBad / N
    
    conditionToResult[condition] = {}
    conditionToResult[condition]["Condition"] = condition
    conditionToResult[condition]["Num. Good"] = numGood
    conditionToResult[condition]["Num. Bad"] = numBad
    conditionToResult[condition]["Num. Unsure"] = numUnsure
    conditionToResult[condition]["Num. Incomplete"] = numIncomplete
    conditionToResult[condition]["N"] = N
    conditionToResult[condition]["Perc. Good"] = percGood
    conditionToResult[condition]["Perc. Bad"] = percBad


#
# save
#
fieldnames = ["Condition", "Num. Good", "Num. Bad", "Num. Unsure", "Num. Incomplete", "N", "Perc. Good", "Perc. Bad"]


with open(sessionDir+"/evaluation_reslults.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames)
    writer.writeheader()
    
    for condition, result in conditionToResult.items():
        writer.writerow(result)








