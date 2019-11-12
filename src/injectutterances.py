'''
Created on Oct 3, 2019

@author: robovie

inject the crowdsourced shopkeeper utterances into the simulated interactions

'''

import csv
import os
import random
import copy
import string
import sys
print('current trace function', sys.gettrace())

import tools



interactionDir = tools.dataDir+"2019-11-12_17-40-29_advancedSimulator9"
shkpUttFilename = tools.dataDir + "2019-11-11_13-54-06_crowdsourcing_results_all_mod.csv"
databaseDir = tools.dataDir+"2019-09-18_13-15-13_advancedSimulator9"

sessionDir = tools.create_session_dir("injectutterances")


numInteractionsPerDb = 1000
numTrainDbs = 10




def read_database_file(filename):
    database = {}
    
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            database[row["camera_ID"]] = row
    
    return database, fieldnames


def read_simulated_interactions(filename, keepActions=None):
    interactions = []
    gtDbCamera = []
    gtDbAttribute = []
    
    
    shkpUttToDbEntryRange = {}
    
    """
    fieldnames = ["TRIAL",
                  "TURN_COUNT",
                  
                  "CURRENT_CAMERA_OF_CONVERSATION",
                  "PREVIOUS_CAMERAS_OF_CONVERSATION",
                  "PREVIOUS_FEATURES_OF_CONVERSATION",
                  
                  "CUSTOMER_ACTION",
                  "CUSTOMER_LOCATION",
                  "CUSTOMER_TOPIC",
                  "CUSTOMER_SPEECH",
                  
                  "OUTPUT_SHOPKEEPER_ACTION",
                  "OUTPUT_SHOPKEEPER_LOCATION",
                  "SHOPKEEPER_TOPIC",
                  "SHOPKEEPER_SPEECH"]
    """
    
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            
            #if int(row["TRIAL"]) >= numInteractionsPerDb:
            #    # only read in this many interactions
            #    break
            
            if (keepActions == None) or (row["OUTPUT_SHOPKEEPER_ACTION"] in keepActions): # and row["SHOPKEEPER_TOPIC"] == "price"):
                
                row["CUSTOMER_SPEECH"] = row["CUSTOMER_SPEECH"].lower().translate(str.maketrans('', '', string.punctuation))
                row["SHOPKEEPER_SPEECH"] = row["SHOPKEEPER_SPEECH"].lower()
                
                interactions.append(row)
            
            
            if row["SHOPKEEPER_SPEECH_DB_ENTRY_RANGE"] != "":
                shkpUttToDbEntryRange[row["SHOPKEEPER_SPEECH"]] = [int(i) for i in row["SHOPKEEPER_SPEECH_DB_ENTRY_RANGE"].split("~")]
    
    return interactions, shkpUttToDbEntryRange, gtDbCamera, gtDbAttribute




#
# load the databases
#
print("loading the databases...")

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
# read in the simulated interactions
#
print("loading the interactions...")

filenames = os.listdir(interactionDir)
filenames.sort()

interactionFilenamesAll = [interactionDir+"/"+fn for fn in filenames if "simulated" in fn]

interactionFilenames = interactionFilenamesAll


interactions = []
datasetSizes = []
databaseIds = []

for i in range(len(interactionFilenames)):
    iFn = interactionFilenames[i]
    
    dbId = iFn.split("_")[-1].split(".")[0][2:]
    databaseIds.append(dbId)
    
    inters, sutder, gtDbCamera, gtDbAttribute = read_simulated_interactions(iFn)
    
    interactions.append(inters)
    datasetSizes.append(len(inters))


#
# read in the crowdsourced utterances
#
print("loading the crowdsourced utterances...")

resultHits = []

with open(shkpUttFilename) as csvfile:
    reader = csv.DictReader(csvfile)
    resultFieldnames = reader.fieldnames
    
    for row in reader:
        resultHits.append(row)


# sort the utterances into a dict so it's easy to find them
resultUtterances = {}

# the AMT results
for hit in resultHits:
    
    w = hit["Input.WORKER_ID"]
    shkpAction = hit["Input.OUTPUT_SHOPKEEPER_ACTION"]
    cam = hit["Input.CURRENT_CAMERA_OF_CONVERSATION"]
    top= hit["Input.SHOPKEEPER_TOPIC"]
    dbId = hit["Input.DB_ID"]
    
    
    if shkpAction not in resultUtterances:
        resultUtterances[shkpAction] = {}
    if cam not in resultUtterances[shkpAction]:
        resultUtterances[shkpAction][cam] = {}
    if top not in resultUtterances[shkpAction][cam]:
        resultUtterances[shkpAction][cam][top] = {}
    if dbId not in resultUtterances[shkpAction][cam][top]:
        resultUtterances[shkpAction][cam][top][dbId] = []
    
    
    shkpUtt = hit["Answer.utterance"]
    resultUtterances[shkpAction][cam][top][dbId].append(shkpUtt)


# for sampling without replacement
resultUtterancesWithoutReplacement = copy.deepcopy(resultUtterances)


#
# replace the originally simulated shopkeeper utterances with the crowdsourced utterances
# 
fieldnames = ["TRIAL",
              "DATABASE_ID", # new field
              "DATABASE_CONTENTS", # new field
              "TURN_COUNT",
              "CURRENT_CAMERA_OF_CONVERSATION",
              "PREVIOUS_CAMERAS_OF_CONVERSATION",
              "PREVIOUS_FEATURES_OF_CONVERSATION",
              "CUSTOMER_ACTION",
              "CUSTOMER_LOCATION",
              "CUSTOMER_TOPIC",
              "CUSTOMER_SPEECH",
              "SPATIAL_STATE",
              "STATE_TARGET",
              "OUTPUT_SHOPKEEPER_ACTION",
              "OUTPUT_SHOPKEEPER_LOCATION",
              "SHOPKEEPER_TOPIC",
              "SHOPKEEPER_SPEECH",
              "OUTPUT_SPATIAL_STATE",
              "OUTPUT_STATE_TARGET",
              "SHOPKEEPER_SPEECH_DB_ENTRY_RANGE"]


modifiedInteractions = [[] for i in range(len(interactions))]


for i in range(len(interactions)):
    dbId = databaseIds[i]
    inters = interactions[i]
    
    for j in range(len(inters)):
        modTurn = copy.deepcopy(inters[j])
        modTurn["DATABASE_ID"] = dbId
        
        shkpAction = modTurn["OUTPUT_SHOPKEEPER_ACTION"]
        cam = modTurn["CURRENT_CAMERA_OF_CONVERSATION"]
        
        if shkpAction == "S_NOT_SURE":
            top = modTurn["CUSTOMER_TOPIC"]
        else:
            top = modTurn["SHOPKEEPER_TOPIC"]
        
        print(shkpAction)
                
        
        if (shkpAction == "S_INTRODUCES_CAMERA" or shkpAction == "S_INTRODUCES_FEATURE" or shkpAction == "S_ANSWERS_QUESTION_ABOUT_FEATURE" or shkpAction == "S_NOT_SURE"):
            print("hello!")
            
            # sample without replacement
            uttList = resultUtterancesWithoutReplacement[shkpAction][cam][top][dbId] # copy by reference
            listLen = len(uttList)
            randIndex = random.randrange(listLen)
            newShkpUtt = uttList.pop(randIndex)
            
            modTurn["SHOPKEEPER_SPEECH"] = newShkpUtt
            modTurn["SHOPKEEPER_SPEECH"] = newShkpUtt
            
            if len(uttList) == 0:
                print("all gone")
                resultUtterancesWithoutReplacement[shkpAction][cam][top][dbId] = copy.deepcopy(resultUtterances[shkpAction][cam][top][dbId])
        
        
        if (shkpAction == "S_INTRODUCES_CAMERA" or shkpAction == "S_INTRODUCES_FEATURE" or shkpAction == "S_ANSWERS_QUESTION_ABOUT_FEATURE"):
            
            if shkpAction == "S_INTRODUCES_CAMERA":
                top = "camera_name"
            
            
            modTurn["DATABASE_CONTENTS"] = databases[i][cam][top]
            
            # add the substring range if there is an exact match to the DB contents in the shkp utterance
            subStringStartIndex = modTurn["SHOPKEEPER_SPEECH"].find(modTurn["DATABASE_CONTENTS"])
            
            if subStringStartIndex != -1:
                modTurn["SHOPKEEPER_SPEECH_DB_ENTRY_RANGE"] = "{}~{}".format(subStringStartIndex, subStringStartIndex+len(modTurn["DATABASE_CONTENTS"]))
            else:
                modTurn["SHOPKEEPER_SPEECH_DB_ENTRY_RANGE"] = "NA"
        
        
       
       
        
        modifiedInteractions[i].append(modTurn)


#
# save the modified interactions
#

today = tools.time_now()[:10]


for i in range(len(modifiedInteractions)):
    
    inters = modifiedInteractions[i]
    dbId = databaseIds[i]
    
    with open(sessionDir+"/{}_simulated_data_csshkputts_{}_database_0-{}.csv".format(today, numInteractionsPerDb, dbId), "w", newline="") as csvfile:
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for row in inters:
            writer.writerow(row)








