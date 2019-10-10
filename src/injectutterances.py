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

import tools



interactionDir = tools.dataDir+"2019-09-18_13-15-13_advancedSimulator9"
resultCsvDir = "E:/Dropbox/ATR/2018 database learning/crowdsourcing/2019-10-01_AMTresults/"


sessionDir = tools.create_session_dir("injectutterances")


numInteractionsPerDb = 200


#
# read in the simulated interactions
#

def read_simulated_interactions(filename, keepActions=None):
    interactions = []
    gtDbCamera = []
    gtDbAttribute = []
    
    
    shkpUttToDbEntryRange = {}
    
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
    
    
    
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            
            if int(row["TRIAL"]) >= numInteractionsPerDb:
                # only read in this many interactions
                break
            
            if (keepActions == None) or (row["OUTPUT_SHOPKEEPER_ACTION"] in keepActions): # and row["SHOPKEEPER_TOPIC"] == "price"):
                
                row["CUSTOMER_SPEECH"] = row["CUSTOMER_SPEECH"].lower().translate(str.maketrans('', '', string.punctuation))
                row["SHOPKEEPER_SPEECH"] = row["SHOPKEEPER_SPEECH"].lower()
                
                interactions.append(row)
                 
            
            if row["SHOPKEEPER_SPEECH_DB_ENTRY_RANGE"] != "":
                shkpUttToDbEntryRange[row["SHOPKEEPER_SPEECH"]] = [int(i) for i in row["SHOPKEEPER_SPEECH_DB_ENTRY_RANGE"].split("~")]
    
    return interactions, shkpUttToDbEntryRange, gtDbCamera, gtDbAttribute


print ("loading the interactions...")

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
resultFilenames = os.listdir(resultCsvDir)
resultFilenames = [resultCsvDir+fn for fn in resultFilenames]

resultHits = []

for fn in resultFilenames:
    with open(fn) as csvfile:
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

def sampleShkpUtt(shkpAction, cam, top):
    
    if (len(resultUtterancesWithoutReplacement[shkpAction][cam][top]) == 0): # the sample pool is empty
        
        if (len(resultUtterances[shkpAction][cam][top]) == 0): # there are now crowdsourced utterances for this action
            
            return None
        
        else: # replenish the sampling pool
            resultUtterancesWithoutReplacement[shkpAction][cam][top] = copy.deepcopy(resultUtterances[shkpAction][cam][top])
            
    
    


fieldnames = ["TRIAL",
              "DATABASE_ID", # new field
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
              "SHOPKEEPER_SPEECH_DB_ENTRY_RANGE"
              ]


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
            top= modTurn["CUSTOMER_TOPIC"]
        else:
            top= modTurn["SHOPKEEPER_TOPIC"]
        
        
        if (shkpAction == "S_INTRODUCES_CAMERA"
            or shkpAction == "S_INTRODUCES_FEATURE"
            or shkpAction == "S_ANSWERS_QUESTION_ABOUT_FEATURE"
            or shkpAction == "S_NOT_SURE"):
            
            # sample with replacement, and refresh the utterances when they run out
            pass
        
        
        modifiedInteractions[i].append(modTurn)


#
# save the modified interactions
#
today = tools.time_now()


for i in range(len(modifiedInteractions)):
    
    inters = modifiedInteractions[i]
    dbId = databaseIds[i]
    
    with open(sessionDir+"/{}_simulated_data_{}_database_{}.csv".format(today, numInteractionsPerDb, dbId), "w", newline="") as csvfile:
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for row in inters:
            writer.writerow(row)








