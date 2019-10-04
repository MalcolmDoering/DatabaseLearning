'''
Created on Sep 4, 2019

@author: robovie
'''


import csv
import random
import copy
import os
import string

import tools



sessionDir = tools.create_session_dir("generateHITcsv")


numDatabases = 11

cameras = ["CAMERA_1", "CAMERA_2", "CAMERA_3"]

attributes = ["camera_type", "color", "weight", "preset_modes", "effects", "price", "resolution", "optical_zoom", "settings", "autofocus_points", "sensor_size", "ISO", "long_exposure"]

csvFields = ["ID", "WORKER_ID", "OUTPUT_SHOPKEEPER_ACTION", "CURRENT_CAMERA_OF_CONVERSATION", "SHOPKEEPER_TOPIC", "DB_ID", "DB_CONTENTS", "CONTEXT", "INTENT", "DB_IMAGE"]



dbUrlTemplate = "https://camerashopdatabases.s3-ap-northeast-1.amazonaws.com/DATABASE_{}-{}.png"



cAsksQuestionContexts = {"camera_type": "The customer asks, \"What <b>type</b> of camera is this?\"",
                         "color": "The customer asks, \"What <b>colors</b> does this camera come in?\"",
                         "weight": "The customer asks, \"How much does this camera <b>weigh</b>?\"",
                         "preset_modes": "The customer asks, \"What kind of <b>preset modes</b> does this camera have?\"",
                         "effects": "The customer asks, \"What kind of <b>effects</b> can this camera do?\"",
                         "price": "The customer asks, \"How much does this camera <b>cost</b>?\"",
                         "resolution": "The customer asks, \"What is the <b>resolution</b> of this camera?\"",
                         "optical_zoom": "The customer asks, \"What is the <b>optical zoom</b> on this camera?\"",
                         "settings": "The customer asks, \"Does this camera have manual or automatic <b>settings</b>?\"",
                         "autofocus_points": "The customer asks, \"How many <b>autofocus points</b> does this camera have?\"",
                         "sensor_size": "The customer asks, \"What is the <b>sensor size</b> on this camera?\"",
                         "ISO": "The customer asks, \"How is the <b>ISO</b> on this camera?\"",
                         "long_exposure": "The customer asks, \"Can this camera take <b>long exposures</b>?\""}


sIntroducesFeatIntents = {"camera_type": "You want to tell the customer what <b>type</b> of camera this is.",
                          "color": "You want to tell the customer what <b>colors</b> this camera comes in.",
                          "weight": "You want to tell the customer how much this camera <b>weighs</b>.",
                          "preset_modes": "You want to tell the customer about this camera's <b>preset modes</b>.",
                          "effects": "You want to tell the customer about this camera's <b>effects</b>.",
                          "price": "You want to tell the customer how much this camera <b>costs</b>.",
                          "resolution": "You want to tell the customer this camera's <b>resolution</b>.",
                          "optical_zoom": "You want to tell the customer about this camera's <b>optical zoom</b>.",
                          "settings": "You want to tell the customer about this camera's <b>settings</b>.",
                          "autofocus_points": "You want to tell the customer about this camera's <b>autofocus points</b>.",
                          "sensor_size": "You want to tell the customer about this camera's <b>sensor size</b>.",
                          "ISO": "You want to tell the customer about this camera's <b>ISO</b> feature.",
                          "long_exposure": "You want to tell the customer about this camera's <b>long exposure</b> capability."}     


def read_database_file(filename):
    database = {}
    
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            
            database[row["camera_ID"]] = {}
            
            for key in row:
                database[row["camera_ID"]][key] = row[key]
    
    return database, fieldnames


def read_simulated_interactions(filename, dbFieldnames, databaseID, keepActions=None):
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
            
            row["DATABASE_ID"] = databaseID
            
            if (keepActions == None) or (row["OUTPUT_SHOPKEEPER_ACTION"] in keepActions): # and row["SHOPKEEPER_TOPIC"] == "price"):
                
                row["CUSTOMER_SPEECH"] = row["CUSTOMER_SPEECH"].lower().translate(str.maketrans('', '', string.punctuation))
                row["SHOPKEEPER_SPEECH"] = row["SHOPKEEPER_SPEECH"].lower()
                
                interactions.append(row)
                
                try:
                    dbRow = cameras.index(row["CURRENT_CAMERA_OF_CONVERSATION"])
                except:
                    dbRow = -1
                
                try:
                    dbCol = dbFieldnames.index(row["SHOPKEEPER_TOPIC"])
                except:
                    dbCol = -1
                
                gtDbCamera.append(dbRow)
                gtDbAttribute.append(dbCol)        
            
            if row["SHOPKEEPER_SPEECH_DB_ENTRY_RANGE"] != "":
                shkpUttToDbEntryRange[row["SHOPKEEPER_SPEECH"]] = [int(i) for i in row["SHOPKEEPER_SPEECH_DB_ENTRY_RANGE"].split("~")]
    
    return interactions, shkpUttToDbEntryRange, gtDbCamera, gtDbAttribute



#
# load the database and interaction data
#
dataDirectory = tools.dataDir + "2019-09-18_13-15-13_advancedSimulator9"


filenames = os.listdir(dataDirectory)
filenames.sort()

databaseFilenames = [dataDirectory+"/"+fn for fn in filenames if "database" in fn and "simulated" not in fn]
interactionFilenames = [dataDirectory+"/"+fn for fn in filenames if "simulated" in fn]


databases = {}
databaseIds = []
dbFieldnames = None # these should be the same for all DBs

for dbFn in databaseFilenames:
    db, dbFieldnames = read_database_file(dbFn)
    
    dbId = dbFn.split("_")[-1].split(".")[0][2:]
    databaseIds.append(dbId)
    
    databases[dbId] = db



interactions = []
datasetSizes = []
gtDatabaseCameras = []
gtDatabaseAttributes = []

allCustUtts = []
allShkpUtts = []

shkpUttToDbEntryRange = {}

for i in range(len(interactionFilenames)):
    iFn = interactionFilenames[i]
    
    inters, sutder, gtDbCamera, gtDbAttribute = read_simulated_interactions(iFn, dbFieldnames, databaseIds[i])#, keepActions=["S_ANSWERS_QUESTION_ABOUT_FEATURE"]) # S_INTRODUCES_CAMERA S_INTRODUCES_FEATURE
    
    #if i < numTrainDbs:
    #    # reduce the amount of training data because we have increased the number of training databases (assumes 1000 interactions per DB)
    #    inters = inters[: int(2 * len(inters) / numTrainDbs)] # two is the minimum number of training databases 
    #    gtDbCamera = gtDbCamera[: int(2 * len(gtDbCamera) / numTrainDbs)]
    #    gtDbAttribute = gtDbAttribute[: int(2 * len(gtDbAttribute) / numTrainDbs)]
    
    
    interactions.append(inters)
    datasetSizes.append(len(inters))
    
    gtDatabaseCameras.append(gtDbCamera)
    gtDatabaseAttributes.append(gtDbAttribute)
    
    allCustUtts += [row["CUSTOMER_SPEECH"] for row in inters]
    allShkpUtts += [row["SHOPKEEPER_SPEECH"] for row in inters]
    
    
    # combine the three dictionaries into one
    shkpUttToDbEntryRange = {**shkpUttToDbEntryRange, **sutder}



#
# figure out how many HITs of each type must be generated
#
counts = {"S_ANSWERS_QUESTION_ABOUT_FEATURE": {},
          "S_INTRODUCES_FEATURE": {},
          "S_INTRODUCES_CAMERA": {},
          "S_NOT_SURE": {}
          }

for inters in interactions:
    
    for i in range(len(inters)):
        
        # only collect data for 200 interactions with each database
        if inters[i]["TRIAL"] == "200":
            break
        
        
        shkpAction = inters[i]["OUTPUT_SHOPKEEPER_ACTION"]
        
        if shkpAction in counts or shkpAction == "S_NOT_SURE":
            
            if shkpAction == "S_NOT_SURE":
                top = inters[i]["CUSTOMER_TOPIC"]
            else:
                top = inters[i]["SHOPKEEPER_TOPIC"]
            
            
            
            cam = inters[i]["CURRENT_CAMERA_OF_CONVERSATION"]
            dbId = inters[i]["DATABASE_ID"]
            
            
            if cam not in counts[shkpAction]:
                counts[shkpAction][cam] = {}
            if top not in counts[shkpAction][cam]:
                counts[shkpAction][cam][top] = {}
            if dbId not in counts[shkpAction][cam][top]:
                counts[shkpAction][cam][top][dbId] = 0
            
            counts[shkpAction][cam][top][dbId] += 1



workerToCount = {}
numWorkers = 5

for shkpAction in counts:
    for cam in counts[shkpAction]:
        for top in counts[shkpAction][cam]:
            for dbId in counts[shkpAction][cam][top]:
                
                count = counts[shkpAction][cam][top][dbId]
                countPerWorker = [0, 0, 0, 0, 0]
                
                for c in range(count):
                    countPerWorker[c % 5] += 1
                
                
                for w in range(numWorkers):
                    if w not in workerToCount:
                        workerToCount[w] = {}
                    if shkpAction not in workerToCount[w]:
                        workerToCount[w][shkpAction] = {}
                    if cam not in workerToCount[w][shkpAction]:
                        workerToCount[w][shkpAction][cam] = {}
                    if top not in workerToCount[w][shkpAction][cam]:
                        workerToCount[w][shkpAction][cam][top] = {}
                    if dbId not in workerToCount[w][shkpAction][cam][top]:
                        workerToCount[w][shkpAction][cam][top][dbId] = 0
                    
                    workerToCount[w][shkpAction][cam][top][dbId] = countPerWorker[w]
                
                
                
                



header = ["OUTPUT_SHOPKEEPER_ACTION", "CURRENT_CAMERA_OF_CONVERSATION", "SHOPKEEPER_TOPIC", "DATABASE_ID", 
          "COUNT", "P1", "P2", "P3", "P4", "P5"]

with open(sessionDir + "/{}_action_counts.csv".format(tools.time_now()), "w", newline="") as csvfile:
    
    writer = csv.writer(csvfile, delimiter=",", quotechar='"')
    writer.writerow(header)
    
    for shkpAction in counts:
        for cam in counts[shkpAction]:
            for top in counts[shkpAction][cam]:
                for dbId in counts[shkpAction][cam][top]:
                    
                    count = counts[shkpAction][cam][top][dbId]
                    
                    countPerWorker = [workerToCount[w][shkpAction][cam][top][dbId] for w in range(numWorkers)]
                    
                    writer.writerow([shkpAction, cam, top, dbId, counts[shkpAction][cam][top][dbId]] + countPerWorker)



#
# generate HITs
#
hits = []


for w in workerToCount:
    for shkpAction in workerToCount[w]:
        for cam in workerToCount[w][shkpAction]:
            for top in workerToCount[w][shkpAction][cam]:
                for dbId in workerToCount[w][shkpAction][cam][top]:
                    
                    dbUrl = dbUrlTemplate.format(dbId, cam)
                    
                    row = {"ID": -1,
                           "WORKER_ID": w,
                           "OUTPUT_SHOPKEEPER_ACTION": shkpAction,
                           "CURRENT_CAMERA_OF_CONVERSATION": cam,
                           "SHOPKEEPER_TOPIC": top,
                           "DB_ID": dbId,
                           "DB_IMAGE": dbUrl}
                    
                    
                    if shkpAction == "S_INTRODUCES_CAMERA":
                        row["DB_CONTENTS"] = databases[dbId][cam]["camera_type"]
                        row["CONTEXT"] = "The customer tells you that they are looking for a camera."
                        row["INTENT"] = "You want to introduce the below camera to the customer."
                    
                    
                    elif shkpAction == "S_ANSWERS_QUESTION_ABOUT_FEATURE" or shkpAction == "S_NOT_SURE":
                        row["DB_CONTENTS"] = databases[dbId][cam][top]
                        row["CONTEXT"] = cAsksQuestionContexts[top]
                        row["INTENT"] = "You want to give the customer their requested information."
                    
                    
                    elif shkpAction == "S_INTRODUCES_FEATURE":
                        row["DB_CONTENTS"] = databases[dbId][cam][top]
                        row["CONTEXT"] = "The customer seems interested in the camera."
                        row["INTENT"] = sIntroducesFeatIntents[top]
                    
                    
                    for c in range(workerToCount[w][shkpAction][cam][top][dbId]):
                        hits.append(copy.deepcopy(row))


#
# write HIT csvs for each worker
#
timeNow = tools.time_now()
hitCount = 0

for w in range(numWorkers):
    
    workerHits = [hit for hit in hits if hit["WORKER_ID"] == w]
    
    # randomize order of HITs
    random.shuffle(workerHits)
    
    
    #  divide into groups of size no greater than 500 HITs each
    hitGroups = []
    
    for i in range(0, len(workerHits), 500):
        
        hg = workerHits[i : min(i+500, len(workerHits))]
        hitGroups.append(hg)
    
    
    for i in range(len(hitGroups)):
        
        with open(sessionDir + "/worker_{}_group_{}_{}_HITs.csv".format(w, i, len(hitGroups[i])), "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, csvFields)
            writer.writeheader()
            
            for j in range(len(hitGroups[i])):
                
                hitGroups[i][j]["ID"] = hitCount
                hitCount += 1
                
                writer.writerow(hitGroups[i][j])






"""
for d in range(numDatabases):
    for c in cameras:
        
        dbUrl = dbUrlTemplate.format(d, c)
        
        # S_INTRODUCES_CAMERA
        row = {"ID": -1,
               "OUTPUT_SHOPKEEPER_ACTION": "S_INTRODUCES_CAMERA",
               "CURRENT_CAMERA_OF_CONVERSATION": c,
               "SHOPKEEPER_TOPIC": "", 
               "CONTEXT": "The customer tells you that they are looking for a camera.", 
               "INTENT": "You want to introduce the below camera to the customer.",
               "DB_IMAGE": dbUrl}
        
        hits.append(copy.deepcopy(row))
        
        
        # S_ANSWERS_QUESTION_ABOUT_FEATURE
        for a in attributes:
            row = {"ID": -1,
                   "OUTPUT_SHOPKEEPER_ACTION": "S_ANSWERS_QUESTION_ABOUT_FEATURE",
                   "CURRENT_CAMERA_OF_CONVERSATION": c,
                   "SHOPKEEPER_TOPIC": a, 
                   "CONTEXT": cAsksQuestionContexts[a], 
                   "INTENT": "You want to give the customer their requested information.",
                   "DB_IMAGE": dbUrl}
            hits.append(copy.deepcopy(row))
        
        
        # S_INTRODUCES_FEATURE
        for a in attributes:
            row = {"ID": -1,
                   "OUTPUT_SHOPKEEPER_ACTION": "S_INTRODUCES_FEATURE",
                   "CURRENT_CAMERA_OF_CONVERSATION": c,
                   "SHOPKEEPER_TOPIC": a, 
                   "CONTEXT": "The customer seems interested in the camera.", 
                   "INTENT": sIntroducesFeatIntents[a],
                   "DB_IMAGE": dbUrl}
            hits.append(copy.deepcopy(row))
            

# randomize order of HITs
#random.shuffle(hits)

with open(sessionDir + "/{}_shopkeeper_utterance_HITs.csv".format(tools.time_now()), "w", newline="") as csvfile:
    
    writer = csv.DictWriter(csvfile, csvFields)
    writer.writeheader()
    
    for i in range(len(hits)):
        hits[i]["ID"] = i
        writer.writerow(hits[i])
"""



