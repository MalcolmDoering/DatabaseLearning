'''
Created on Sep 30, 2019

@author: robovie
'''


import csv
import os


import tools



originalHitCsvDir = "E:/Dropbox/ATR/2018 database learning/crowdsourcing/2019-09-26_20-37-52_generateHITcsv/"

resultCsvDir = "E:/Dropbox/ATR/2018 database learning/crowdsourcing/2019-10-01_AMTresults/"


sessionDir = tools.create_session_dir("datacollectionanalysis1")


#
# read the csv files
#

# the originals
originalFilenames = os.listdir(originalHitCsvDir)
originalFilenames = [originalHitCsvDir+fn for fn in originalFilenames if "HIT" in fn]

originalHits = []

for fn in originalFilenames:
    with open(fn) as csvfile:
        reader = csv.DictReader(csvfile)
        originalFieldnames = reader.fieldnames
        
        for row in reader:
            originalHits.append(row)


# the AMT results
resultFilenames = os.listdir(resultCsvDir)
resultFilenames = [resultCsvDir+fn for fn in resultFilenames]

resultHits = []

for fn in resultFilenames:
    with open(fn) as csvfile:
        reader = csv.DictReader(csvfile)
        resultFieldnames = reader.fieldnames
        
        for row in reader:
            resultHits.append(row)


#
# count how many exist of each category
# per worker and total
#
originalCounts = {"all":{}}
resultCountsAndPercs = {"all":{}}

# the originals
for hit in originalHits:
    
    w = hit["WORKER_ID"]
    shkpAction = hit["OUTPUT_SHOPKEEPER_ACTION"]
    cam = hit["CURRENT_CAMERA_OF_CONVERSATION"]
    top= hit["SHOPKEEPER_TOPIC"]
    dbId = hit["DB_ID"]
    
    
    if w not in originalCounts:
        originalCounts[w] = {}
    if shkpAction not in originalCounts[w]:
        originalCounts[w][shkpAction] = {}
    if cam not in originalCounts[w][shkpAction]:
        originalCounts[w][shkpAction][cam] = {}
    if top not in originalCounts[w][shkpAction][cam]:
        originalCounts[w][shkpAction][cam][top] = {}
    if dbId not in originalCounts[w][shkpAction][cam][top]:
        originalCounts[w][shkpAction][cam][top][dbId] = 0

    originalCounts[w][shkpAction][cam][top][dbId] += 1
    
    
    w = "all"
    
    if shkpAction not in originalCounts[w]:
        originalCounts[w][shkpAction] = {}
    if cam not in originalCounts[w][shkpAction]:
        originalCounts[w][shkpAction][cam] = {}
    if top not in originalCounts[w][shkpAction][cam]:
        originalCounts[w][shkpAction][cam][top] = {}
    if dbId not in originalCounts[w][shkpAction][cam][top]:
        originalCounts[w][shkpAction][cam][top][dbId] = 0
    
    originalCounts[w][shkpAction][cam][top][dbId] += 1


# the AMT results
for hit in resultHits:
    
    w = hit["Input.WORKER_ID"]
    shkpAction = hit["Input.OUTPUT_SHOPKEEPER_ACTION"]
    cam = hit["Input.CURRENT_CAMERA_OF_CONVERSATION"]
    top= hit["Input.SHOPKEEPER_TOPIC"]
    dbId = hit["Input.DB_ID"]
    
    
    if w not in resultCountsAndPercs:
        resultCountsAndPercs[w] = {}
    if shkpAction not in resultCountsAndPercs[w]:
        resultCountsAndPercs[w][shkpAction] = {}
    if cam not in resultCountsAndPercs[w][shkpAction]:
        resultCountsAndPercs[w][shkpAction][cam] = {}
    if top not in resultCountsAndPercs[w][shkpAction][cam]:
        resultCountsAndPercs[w][shkpAction][cam][top] = {}
    if dbId not in resultCountsAndPercs[w][shkpAction][cam][top]:
        resultCountsAndPercs[w][shkpAction][cam][top][dbId] = [0, -1]

    resultCountsAndPercs[w][shkpAction][cam][top][dbId][0] += 1
    resultCountsAndPercs[w][shkpAction][cam][top][dbId][1] = resultCountsAndPercs[w][shkpAction][cam][top][dbId][0] / float(originalCounts[w][shkpAction][cam][top][dbId])
    
    w = "all"
    
    if shkpAction not in resultCountsAndPercs[w]:
        resultCountsAndPercs[w][shkpAction] = {}
    if cam not in resultCountsAndPercs[w][shkpAction]:
        resultCountsAndPercs[w][shkpAction][cam] = {}
    if top not in resultCountsAndPercs[w][shkpAction][cam]:
        resultCountsAndPercs[w][shkpAction][cam][top] = {}
    if dbId not in resultCountsAndPercs[w][shkpAction][cam][top]:
        resultCountsAndPercs[w][shkpAction][cam][top][dbId] = [0, -1]
    
    resultCountsAndPercs[w][shkpAction][cam][top][dbId][0] += 1
    resultCountsAndPercs[w][shkpAction][cam][top][dbId][1] = resultCountsAndPercs[w][shkpAction][cam][top][dbId][0] / float(originalCounts[w][shkpAction][cam][top][dbId])



# add the actions that have not been completed at all yet (because they won't appear in the results otherwise)
for w in originalCounts:
    for shkpAction in originalCounts[w]:
        for cam in originalCounts[w][shkpAction]:
            for top in originalCounts[w][shkpAction][cam]:
                for dbId in originalCounts[w][shkpAction][cam][top]:
                    
                    if w not in resultCountsAndPercs:
                        resultCountsAndPercs[w] = {}
                    if shkpAction not in resultCountsAndPercs[w]:
                        resultCountsAndPercs[w][shkpAction] = {}
                    if cam not in resultCountsAndPercs[w][shkpAction]:
                        resultCountsAndPercs[w][shkpAction][cam] = {}
                    if top not in resultCountsAndPercs[w][shkpAction][cam]:
                        resultCountsAndPercs[w][shkpAction][cam][top] = {}
                    if dbId not in resultCountsAndPercs[w][shkpAction][cam][top]:
                        resultCountsAndPercs[w][shkpAction][cam][top][dbId] = [0, 0.0]


# compute MAX 10 hit counts and percent completed
max10TargetCompletedAll = {}

w = "all"

for shkpAction in originalCounts[w]:
    for cam in originalCounts[w][shkpAction]:
        for top in originalCounts[w][shkpAction][cam]:
            for dbId in originalCounts[w][shkpAction][cam][top]:
                
                if shkpAction not in max10TargetCompletedAll:
                    max10TargetCompletedAll[shkpAction] = {}
                if cam not in max10TargetCompletedAll[shkpAction]:
                    max10TargetCompletedAll[shkpAction][cam] = {}
                if top not in max10TargetCompletedAll[shkpAction][cam]:
                    max10TargetCompletedAll[shkpAction][cam][top] = {}
                if dbId not in max10TargetCompletedAll[shkpAction][cam][top]:
                    max10TargetCompletedAll[shkpAction][cam][top][dbId] = [-1, -1]
                
                
                max10TargetCount = min(10, originalCounts[w][shkpAction][cam][top][dbId])
                
                max10TargetCompletedAll[shkpAction][cam][top][dbId] = [max10TargetCount, # target
                                                                       resultCountsAndPercs[w][shkpAction][cam][top][dbId][0] / float(max10TargetCount) # percent completed
                                                                       ]
                
                





# compute totals
totalHitCount = 0
totalCompleted = 0
max10TotalTargetCount = 0

w = "all"

for shkpAction in resultCountsAndPercs[w]:
    for cam in resultCountsAndPercs[w][shkpAction]:
        for top in resultCountsAndPercs[w][shkpAction][cam]:
            for dbId in resultCountsAndPercs[w][shkpAction][cam][top]:
                
                totalHitCount += originalCounts[w][shkpAction][cam][top][dbId]
                totalCompleted += resultCountsAndPercs[w][shkpAction][cam][top][dbId][0]
                max10TotalTargetCount += max10TargetCompletedAll[shkpAction][cam][top][dbId][0]


#
# save results to csv
#
fieldnames = ["WORKER_ID", 
              "OUTPUT_SHOPKEEPER_ACTION", 
              "CURRENT_CAMERA_OF_CONVERSATION", 
              "SHOPKEEPER_TOPIC", 
              "DB_ID", 
              "HIT_COUNT", 
              "COMPLETED_COUNT", 
              "COMPLETED_PERCENT", 
              "MAX_10_TARGET_COUNT",
              "MAX_10_COMPLETED_PERCENT"]


with open(sessionDir + "/data_collection_progress.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames)
    writer.writeheader()
    
    writer.writerow({"WORKER_ID": "total", 
                     "HIT_COUNT": totalHitCount,
                     "COMPLETED_COUNT": totalCompleted,
                     "COMPLETED_PERCENT": totalCompleted / float(totalHitCount),
                     "MAX_10_TARGET_COUNT": max10TotalTargetCount,
                     "MAX_10_COMPLETED_PERCENT": totalCompleted / float(max10TotalTargetCount)
                     })
    
    for w in resultCountsAndPercs:
        for shkpAction in resultCountsAndPercs[w]:
            for cam in resultCountsAndPercs[w][shkpAction]:
                for top in resultCountsAndPercs[w][shkpAction][cam]:
                    for dbId in resultCountsAndPercs[w][shkpAction][cam][top]:
                        
                        row = {"WORKER_ID": w, 
                               "OUTPUT_SHOPKEEPER_ACTION": shkpAction, 
                               "CURRENT_CAMERA_OF_CONVERSATION": cam, 
                               "SHOPKEEPER_TOPIC": top, 
                               "DB_ID": dbId, 
                               "HIT_COUNT": originalCounts[w][shkpAction][cam][top][dbId], 
                               "COMPLETED_COUNT": resultCountsAndPercs[w][shkpAction][cam][top][dbId][0], 
                               "COMPLETED_PERCENT": resultCountsAndPercs[w][shkpAction][cam][top][dbId][1]
                               }
                        
                        if w == "all":
                            row["MAX_10_TARGET_COUNT"] = max10TargetCompletedAll[shkpAction][cam][top][dbId][0]
                            row["MAX_10_COMPLETED_PERCENT"] = max10TargetCompletedAll[shkpAction][cam][top][dbId][1]
                        
                        
                        writer.writerow(row)



