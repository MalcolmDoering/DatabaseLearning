'''
Created on Oct 1, 2019

@author: robovie


Take the completed hits out of the old HIT csv files.

Reorder the HIT csv files so that a min of 10 of each action category will first be collected.

'''

import csv
import os
import random
import copy
import string

import tools


originalHitCsvDir = "E:/Dropbox/ATR/2018 database learning/crowdsourcing/2019-09-26_20-37-52_generateHITcsv/"

resultCsvDir = "E:/Dropbox/ATR/2018 database learning/crowdsourcing/2019-10-01_AMTresults/"


sessionDir = tools.create_session_dir("generateHITcsv2")



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
# count the original HITs
#
originalCounts = {"all":{}}

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


#
# count the completed HITs
#
completedCounts = {"all":{}}

for hit in resultHits:
    
    w = hit["Input.WORKER_ID"]
    shkpAction = hit["Input.OUTPUT_SHOPKEEPER_ACTION"]
    cam = hit["Input.CURRENT_CAMERA_OF_CONVERSATION"]
    top= hit["Input.SHOPKEEPER_TOPIC"]
    dbId = hit["Input.DB_ID"]
    
    
    if w not in completedCounts:
        completedCounts[w] = {}
    if shkpAction not in completedCounts[w]:
        completedCounts[w][shkpAction] = {}
    if cam not in completedCounts[w][shkpAction]:
        completedCounts[w][shkpAction][cam] = {}
    if top not in completedCounts[w][shkpAction][cam]:
        completedCounts[w][shkpAction][cam][top] = {}
    if dbId not in completedCounts[w][shkpAction][cam][top]:
        completedCounts[w][shkpAction][cam][top][dbId] = 0

    completedCounts[w][shkpAction][cam][top][dbId] += 1
    
    w = "all"
    
    if shkpAction not in completedCounts[w]:
        completedCounts[w][shkpAction] = {}
    if cam not in completedCounts[w][shkpAction]:
        completedCounts[w][shkpAction][cam] = {}
    if top not in completedCounts[w][shkpAction][cam]:
        completedCounts[w][shkpAction][cam][top] = {}
    if dbId not in completedCounts[w][shkpAction][cam][top]:
        completedCounts[w][shkpAction][cam][top][dbId] = 0
    
    completedCounts[w][shkpAction][cam][top][dbId] += 1


# mark which categories have 0 hits completed
for w in originalCounts:
    for shkpAction in originalCounts[w]:
        for cam in originalCounts[w][shkpAction]:
            for top in originalCounts[w][shkpAction][cam]:
                for dbId in originalCounts[w][shkpAction][cam][top]:
                    
                    if w not in completedCounts:
                        completedCounts[w] = []
                    if shkpAction not in completedCounts[w]:
                        completedCounts[w][shkpAction] = {}
                    if cam not in completedCounts[w][shkpAction]:
                        completedCounts[w][shkpAction][cam] = {}
                    if top not in completedCounts[w][shkpAction][cam]:
                        completedCounts[w][shkpAction][cam][top] = {}
                    if dbId not in completedCounts[w][shkpAction][cam][top]:
                        completedCounts[w][shkpAction][cam][top][dbId] = 0


#
# compute how many of the original are remaining
#
originalRemainingAll = {}

w = "all"

for shkpAction in originalCounts[w]:
    for cam in originalCounts[w][shkpAction]:
        for top in originalCounts[w][shkpAction][cam]:
            for dbId in originalCounts[w][shkpAction][cam][top]:
                
                if shkpAction not in originalRemainingAll:
                    originalRemainingAll[shkpAction] = {}
                if cam not in originalRemainingAll[shkpAction]:
                    originalRemainingAll[shkpAction][cam] = {}
                if top not in originalRemainingAll[shkpAction][cam]:
                    originalRemainingAll[shkpAction][cam][top] = {}
                if dbId not in originalRemainingAll[shkpAction][cam][top]:
                    originalRemainingAll[shkpAction][cam][top][dbId] = -1
                
                originalRemainingAll[shkpAction][cam][top][dbId] = originalCounts[w][shkpAction][cam][top][dbId] - completedCounts[w][shkpAction][cam][top][dbId]


#
# compute how many are remaining if we only collect a max of 10 of each category
#
max10RemainingAll = {}
max10RemainingTotal = 0

w = "all"

for shkpAction in originalCounts[w]:
    for cam in originalCounts[w][shkpAction]:
        for top in originalCounts[w][shkpAction][cam]:
            for dbId in originalCounts[w][shkpAction][cam][top]:
                
                if shkpAction not in max10RemainingAll:
                    max10RemainingAll[shkpAction] = {}
                if cam not in max10RemainingAll[shkpAction]:
                    max10RemainingAll[shkpAction][cam] = {}
                if top not in max10RemainingAll[shkpAction][cam]:
                    max10RemainingAll[shkpAction][cam][top] = {}
                if dbId not in max10RemainingAll[shkpAction][cam][top]:
                    max10RemainingAll[shkpAction][cam][top][dbId] = -1
                
                
                max10TargetCount = min(10, originalCounts[w][shkpAction][cam][top][dbId])
                
                max10RemainingAll[shkpAction][cam][top][dbId] = max(0, max10TargetCount-completedCounts[w][shkpAction][cam][top][dbId])
                max10RemainingTotal += max10RemainingAll[shkpAction][cam][top][dbId]

#
# generate new hit csv files that ensure we first collect at least 10 instances of each category before collecting further
#

# remove completed HITs from the original HITs
completedHitIds = [hit["Input.ID"] for hit in resultHits]
completedHitIds.sort()

remainingHits = []

for hit in originalHits:
    if hit["ID"] not in completedHitIds:
        remainingHits.append(hit)


print(len(originalHits), "total original HITs")
print(len(resultHits), "HITs completed")
print(len(remainingHits), "original HITs remaining")
print(max10RemainingTotal, "max 10 HITs remaining")

# select hits from the max 10 remaining list

hitsToWriteMax10 = {"all": []}

w = "all"

random.shuffle(remainingHits)

for shkpAction in max10RemainingAll:
    for cam in max10RemainingAll[shkpAction]:
        for top in max10RemainingAll[shkpAction][cam]:
            for dbId in max10RemainingAll[shkpAction][cam][top]:
                
                while max10RemainingAll[shkpAction][cam][top][dbId] > 0:
                    
                    # find a hit from the remaining list and add it
                    for i in range(len(remainingHits)):
                        
                        if (remainingHits[i]["OUTPUT_SHOPKEEPER_ACTION"] == shkpAction
                            and remainingHits[i]["CURRENT_CAMERA_OF_CONVERSATION"] == cam
                            and remainingHits[i]["SHOPKEEPER_TOPIC"] == top
                            and remainingHits[i]["DB_ID"] == dbId):
                            
                            hitsToWriteMax10[w].append(remainingHits[i])
                            
                            max10RemainingAll[shkpAction][cam][top][dbId] -= 1
                            originalRemainingAll[shkpAction][cam][top][dbId] -= 1
                            
                            del remainingHits[i]
                            break


# split up hits per worker
for hit in hitsToWriteMax10["all"]:
    w = hit["WORKER_ID"]
    
    if w not in hitsToWriteMax10:
        hitsToWriteMax10[w] = []
    
    hitsToWriteMax10[w].append(hit)


# save to csv file
csvFields = ["ID", "WORKER_ID", "OUTPUT_SHOPKEEPER_ACTION", "CURRENT_CAMERA_OF_CONVERSATION", "SHOPKEEPER_TOPIC", "DB_ID", "DB_CONTENTS", "CONTEXT", "INTENT", "DB_IMAGE"]


for w in hitsToWriteMax10:
    if w != "all":
        
        random.shuffle(hitsToWriteMax10[w])
        
        #  divide into groups of size no greater than 500 HITs each
        hitGroups = []
        
        for i in range(0, len(hitsToWriteMax10[w]), 500):
            
            hg = hitsToWriteMax10[w][i : min(i+500, len(hitsToWriteMax10[w]))]
            hitGroups.append(hg)
        
        
        for i in range(len(hitGroups)):
            
            with open(sessionDir + "/max10_worker_{}_group_{}_{}_HITs.csv".format(w, i, len(hitGroups[i])), "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, csvFields)
                writer.writeheader()
                
                for j in range(len(hitGroups[i])):
                    writer.writerow(hitGroups[i][j])



# save the remaining hits
print(len(remainingHits), "HITs remaining after max 10")

# split up hits per worker
hitsToWrite = {}

for hit in remainingHits:
    w = hit["WORKER_ID"]
    
    if w not in hitsToWrite:
        hitsToWrite[w] = []
    
    hitsToWrite[w].append(hit)


for w in hitsToWrite:
    
    random.shuffle(hitsToWrite[w])
    
    #  divide into groups of size no greater than 500 HITs each
    hitGroups = []
    
    for i in range(0, len(hitsToWrite[w]), 500):
        
        hg = hitsToWrite[w][i : min(i+500, len(hitsToWrite[w]))]
        hitGroups.append(hg)
    
    
    for i in range(len(hitGroups)):
        
        with open(sessionDir + "/worker_{}_group_{}_{}_HITs.csv".format(w, i, len(hitGroups[i])), "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, csvFields)
            writer.writeheader()
            
            for j in range(len(hitGroups[i])):
                writer.writerow(hitGroups[i][j])


