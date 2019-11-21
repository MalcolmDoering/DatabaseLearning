'''
Created on Nov 11, 2019

@author: robovie


Analyze the crowdsourced shopkeeper utterances

'''

import csv
import matplotlib.pyplot as plt
import numpy as np
import os


import tools



sessionDir = tools.create_session_dir("analysis6_database_learning")



#
# load the databases
#
numTrainDbs = 10
databaseDir = tools.dataDir+"2019-09-18_13-15-13_advancedSimulator9"


def read_database_file(filename):
    database = {}
    
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            database[row["camera_ID"]] = row
    
    return database, fieldnames



filenames = os.listdir(databaseDir)
filenames.sort()

databaseFilenamesAll = [databaseDir+"/"+fn for fn in filenames if "handmade" in fn]
databaseFilenames = databaseFilenamesAll[:numTrainDbs+1]


databases = []
databaseIds = []
dbFieldnames = None # these should be the same for all DBs

for dbFn in databaseFilenames:
    
    db, dbFieldnames = read_database_file(dbFn)
    databaseIds.append(dbFn.split("_")[-1].split(".")[0])
    databases.append(db)

numDatabases = len(databases)



#
# load the data
#
dataFilename = tools.dataDir + "2019-11-11_13-54-06_crowdsourcing_results_all_mod.csv"

hits = []

with open(dataFilename) as csvfile:
    
    reader = csv.DictReader(csvfile)
    hitFieldnames = reader.fieldnames
    
    for row in reader:
        
        # add missing fields... 
        cam = row["Input.CURRENT_CAMERA_OF_CONVERSATION"]
        top = row["Input.SHOPKEEPER_TOPIC"]
        
        
        # add the database ID and contents
        if row["Input.DB_CONTENTS"] == "":
            
            #https://camerashopdatabases.s3-ap-northeast-1.amazonaws.com/DATABASE_04-CAMERA_3.png
            dbId = row["Input.DB_IMAGE"][-15:-13]
            
            row["Input.DB_ID"] = dbId
            
            if row["Input.OUTPUT_SHOPKEEPER_ACTION"] != "S_INTRODUCES_CAMERA":
                row["Input.DB_CONTENTS"] = databases[int(dbId)][cam][top]
        
        
        dbId = int(row["Input.DB_ID"])
        
        
        # fix the problem of displaying the wrong DB contents for introducing cameras
        if row["Input.OUTPUT_SHOPKEEPER_ACTION"] == "S_INTRODUCES_CAMERA":
            row["Input.DB_CONTENTS"] = databases[int(dbId)][cam]["camera_name"]
        
        
        hits.append(row)


# re-write the data with the missing fields filled in
# with open(dataFilename[:-4]+"_mod.csv", "w", newline="") as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=hitFieldnames)
#     writer.writeheader()
#     
#     for hit in hits:
#         writer.writerow(hit)


#
# count the utterances
# unique utterances
# number of occurrences of uniqe utterances
# number of occurrences of uniqe sets of fields (action, camera, topic, etc.)
#
uniqueUtteranceCounts = {}
uniqueFieldValSetCounts = {}

importantFields = ["Answer.utterance", "Input.OUTPUT_SHOPKEEPER_ACTION", "Input.CURRENT_CAMERA_OF_CONVERSATION", "Input.SHOPKEEPER_TOPIC", "Input.DB_ID", "Input.DB_CONTENTS"]
fieldValKeyToFieldValDict = {}

def hit_dict_to_key(hitDict):
    
    fieldVals = []
    
    for f in importantFields:
        if f in hitDict or hitDict[f] == "":
            fieldVals.append(hitDict[f])
        else:
            fieldVals.append("NULL")
    
    return " ".join(fieldVals)


for hit in hits:
    
    utt = hit["Answer.utterance"]
    
    if utt not in uniqueUtteranceCounts:
        uniqueUtteranceCounts[utt] = 0.0
    uniqueUtteranceCounts[utt] += 1
    
    
    fieldValKey = hit_dict_to_key(hit)
    
    if fieldValKey not in uniqueFieldValSetCounts:
        uniqueFieldValSetCounts[fieldValKey] = 0.0
    uniqueFieldValSetCounts[fieldValKey] += 1
    
    fieldValKeyToFieldValDict[fieldValKey] = dict([(f, hit[f]) for f in importantFields])



uniqueUttsSortedByCount = sorted(list(uniqueUtteranceCounts.items()), key=lambda x: x[1], reverse=True)
uniqueFieldValSetsSortedByCounts = sorted(list(uniqueFieldValSetCounts.items()), key=lambda x: x[1], reverse=True)


print(len(hits), "utterances collected total")
print(len(uniqueUtteranceCounts), "unique utterances")
print(len(uniqueFieldValSetsSortedByCounts), "unique field value sets")


#
# how many of the utterances contain the exact DB contents?
#
numUttsThatContainExactDBContents = 0.0
numUniqueUttsThatContainExactDBContents = 0.0

numUttsThatCouldContainExactDBContents = 0
numUniqueUttsThatCouldContainExactDBContents = 0


for fvKey, count in uniqueFieldValSetsSortedByCounts:
    fieldVals = fieldValKeyToFieldValDict[fvKey]
    
    if fieldVals["Input.DB_CONTENTS"] != "":
        numUttsThatCouldContainExactDBContents += count
        numUniqueUttsThatCouldContainExactDBContents += 1
        
        if fieldVals["Input.DB_CONTENTS"] in fieldVals["Answer.utterance"]:
            numUttsThatContainExactDBContents += count
            numUniqueUttsThatContainExactDBContents += 1


print("{:} utterances of {:} possible utterances ({:}) contain the exact DB contents".format(
    numUttsThatContainExactDBContents, 
    numUttsThatCouldContainExactDBContents, 
    round(numUttsThatContainExactDBContents/numUttsThatCouldContainExactDBContents, 2)))

print("{:} unique utts. of {:} possible unique utts. ({:}) contain the exact DB contents".format(
    numUniqueUttsThatContainExactDBContents, 
    numUniqueUttsThatCouldContainExactDBContents, 
    round(numUniqueUttsThatContainExactDBContents/numUniqueUttsThatCouldContainExactDBContents, 2)))


#
# for each combination of camera, feature, and DB, how many times does the most frequent unique utterance occur?
#



#
# how many times does info from multiple DB entries appear in the utterance?
#




#
# save the counts
#
with open(sessionDir + "/unique_utterance_counts.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    
    writer.writerow(["Utterance", "Count", "Percent of All Utterances"])
    
    for utt, count in uniqueUttsSortedByCount:
        writer.writerow([utt, count, count / len(hits)])


with open(sessionDir + "/unique_field_value_set_counts.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["Count"] + importantFields)
    
    writer.writeheader()
    
    for fvKey, count in uniqueFieldValSetsSortedByCounts:
        row = fieldValKeyToFieldValDict[fvKey]
        row["Count"] = count
        
        writer.writerow(row)


#
# create visualizations
#
#fig, axs = plt.subplots(2, 1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)


x = range(len(uniqueUttsSortedByCount))
y = [u[1] for u in uniqueUttsSortedByCount]

ax1.set_xlabel("Unique utterance ID")
ax1.plot(x, y, '.', color='black');


x = range(len(uniqueFieldValSetsSortedByCounts))
y = [u[1] for u in uniqueFieldValSetsSortedByCounts]

ax2.set_xlabel("Unique utt-act-cam-feat-db combination ID")
ax2.plot(x, y, '.', color='black');


# Turn off axis lines and ticks of the big subplot
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

# shared labels
ax.set_ylabel("Number of Occurrences")

plt.show()













