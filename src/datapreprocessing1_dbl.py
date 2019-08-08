'''
Created on Jul 24, 2019

@author: robovie
'''


import csv
import numpy as np
import string
import os


import tools
from utterancevectorizer import UtteranceVectorizer


#sessionDir = tools.create_session_dir("datapreprocessing1_dbl")
sessionDir = tools.dataDir+"2019-08-08_18-00-06_advancedSimulator8_input_sequence_vectors"

dataDirectory = tools.dataDir+"2019-08-08_18-00-06_advancedSimulator8"
numTrainDbs = 10

dtype = np.int8
inputSeqCutoffLen = 20

cameras = ["CAMERA_1", "CAMERA_2", "CAMERA_3"]
spatialFormations = ["NONE", "WAITING", "FACE_TO_FACE", "PRESENT_X"]
stateTargets = ["NONE", "CAMERA_1", "CAMERA_2", "CAMERA_3"]
locations = ["DOOR", "MIDDLE", "SERVICE_COUNTER", "CAMERA_1", "CAMERA_2", "CAMERA_3"]



def read_simulated_interactions(filename, dbFieldnames, keepActions=None):
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


def read_database_file(filename):
    database = []
    
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            
            for key in row:
                row[key] = row[key].lower()
            
            database.append(row)
    
    return database, fieldnames




#
# load the interaction data
#
print ("loading the data...")

filenames = os.listdir(dataDirectory)
filenames.sort()

databaseFilenamesAll = [dataDirectory+"/"+fn for fn in filenames if "simulated" not in fn]
interactionFilenamesAll = [dataDirectory+"/"+fn for fn in filenames if "simulated" in fn]

databaseFilenames = databaseFilenamesAll[:numTrainDbs+1]
interactionFilenames = interactionFilenamesAll[:numTrainDbs+1]


databases = []
databaseIds = []
dbFieldnames = None # these should be the same for all DBs

for dbFn in databaseFilenames:
    db, dbFieldnames = read_database_file(dbFn)
    
    databaseIds.append(dbFn.split("_")[-1].split(".")[0])
    
    databases.append(db)

numDatabases = len(databases)

interactions = []
datasetSizes = []
gtDatabaseCameras = []
gtDatabaseAttributes = []

allCustUtts = []
allShkpUtts = []

shkpUttToDbEntryRange = {}

for i in range(len(interactionFilenames)):
    iFn = interactionFilenames[i]
    
    inters, sutder, gtDbCamera, gtDbAttribute = read_simulated_interactions(iFn, dbFieldnames)#, keepActions=["S_ANSWERS_QUESTION_ABOUT_FEATURE"]) # S_INTRODUCES_CAMERA S_INTRODUCES_FEATURE
    
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
# vectorize the customer utterances
#
print("vectorizing customer utterances...")

custUttVectorizer = UtteranceVectorizer(allCustUtts,
                                        minCount=2, 
                                        keywordWeight=1.0, 
                                        keywordSet=[], 
                                        unigramsAndKeywordsOnly=True, 
                                        tfidf=False,
                                        useStopwords=False,
                                        lsa=False)

custUttToVec = {}

for cUtt in allCustUtts:
    if cUtt not in custUttToVec:
        custUttToVec[cUtt] = custUttVectorizer.get_utterance_vector(cUtt, unigramOnly=True)
        #custUttToVec[cUtt] = custUttVectorizer.get_lsa_vector(cUtt)

#
# vectorize the shopkeeper utterances
#
print("vectorizing shopkeeper utterances...")

#uniqueShkpUtts = list(set(allShkpUtts))

shkpUttVectorizer = UtteranceVectorizer(allShkpUtts,
                                        minCount=0,
                                        keywordWeight=1.0,
                                        keywordSet=[], 
                                        unigramsAndKeywordsOnly=True, 
                                        tfidf=False,
                                        useStopwords=False,
                                        lsa=False)

shkpUttToVec = {}

for sUtt in allShkpUtts:
    if sUtt not in shkpUttToVec:
        shkpUttToVec[sUtt] = shkpUttVectorizer.get_utterance_vector(sUtt, unigramOnly=True)
        #shkpUttToVec[sUtt] = shkpUttVectorizer.get_lsa_vector(sUtt)



#
# vectorize spatial formations and locations
#
print("creating spatial formation and location vectors...")

spatFormToVec = {}

for i in range(len(spatialFormations)):
    vec = np.zeros(len(spatialFormations), dtype=dtype)
    vec[i] = 1
    spatFormToVec[spatialFormations[i]] = vec


stateTargToVec= {}

for i in range(len(stateTargets)):
    vec = np.zeros(len(stateTargets), dtype=dtype)
    vec[i] = 1
    stateTargToVec[stateTargets[i]] = vec


locToVec = {}

for i in range(len(locations)):
    vec = np.zeros(len(locations), dtype=dtype)
    vec[i] = 1
    locToVec[locations[i]] = vec


#
# vectorize each turn
#
print("vectorizing each turn...")

for i in range(len(interactions)):
    for j in range(len(interactions[i])):
        
        if int(interactions[i][j]["TURN_COUNT"]) != 1:
            prevShkpUtt = interactions[i][j-1]["SHOPKEEPER_SPEECH"]
            prevShkpLoc = interactions[i][j-1]["OUTPUT_SHOPKEEPER_LOCATION"]
        else:
            prevShkpUtt = ""
            prevShkpLoc = "SERVICE_COUNTER"
        
        spatForm = interactions[i][j]["SPATIAL_STATE"]
        stateTarg = interactions[i][j]["STATE_TARGET"]
        
        custUtt = interactions[i][j]["CUSTOMER_SPEECH"]
        custLoc = interactions[i][j]["CUSTOMER_LOCATION"]
        
        
        prevShkpUttVec = shkpUttToVec[prevShkpUtt]
        prevShkpLocVec = locToVec[prevShkpLoc]
        prevSpatFormVec = spatFormToVec[spatForm]
        prevStateTargVec = stateTargToVec[stateTarg]
        custUttVec = custUttToVec[custUtt]
        custLocVec = locToVec[custLoc]
        
        jointStateVector = np.concatenate((prevShkpUttVec,
                                           prevShkpLocVec,
                                           prevSpatFormVec,
                                           prevStateTargVec,
                                           custUttVec,
                                           custLocVec))
        
        interactions[i][j]["JOINT_STATE_VECTOR"] = jointStateVector 
    

print("shkp utt vec len", len(prevShkpUttVec))
print("shkp loc vec len", len(prevShkpLocVec))
print("spat state vec len", len(prevSpatFormVec))
print("state targ vec len", len(prevStateTargVec))
print("cust utt vec len", len(custUttVec))
print("cust loc vec len", len(custLocVec))


#
# create input sequences
#
print("creating input sequences...")

# find the start and end indices of each interaction
startEndIndices = []
allInteractionLens = []


for i in range(len(interactions)):
    sei = []
    
    for j in range(len(interactions[i])):
        
        if int(interactions[i][j]["TURN_COUNT"]) == 1:
            startIndex = j
        
        elif (j == (len(interactions[i])-1)) or (interactions[i][j]["TRIAL"] != interactions[i][j+1]["TRIAL"]):
            endIndex = j + 1
            sei.append((startIndex, endIndex))
            allInteractionLens.append(endIndex-startIndex)
    
    startEndIndices.append(sei)


maxInterLen = max(allInteractionLens)
inputSeqCutoffLen = min(maxInterLen, inputSeqCutoffLen)
jsvDim = len(interactions[0][0]["JOINT_STATE_VECTOR"])

print("ave interaction len", np.mean(allInteractionLens), np.std(allInteractionLens))


for i in range(len(interactions)):
    
    inputSequences = []
    count = 0
    
    for sei in startEndIndices[i]:
        inSeq = []
        
        for j in range(sei[0], sei[1]):
            inSeq.append(interactions[i][j]["JOINT_STATE_VECTOR"])
            
            
            # add the pre padding
            padding = [np.zeros(jsvDim, dtype=dtype)] * (maxInterLen - len(inSeq))
            inSeqTemp = padding + inSeq
            
            inSeqTemp = inSeqTemp[-inputSeqCutoffLen:]
            
            #if len(inSeqTemp) != maxInterLen:
            #    print("hello", len(inSeqTemp), len(inSeq), len(padding))
            #    pass
            
            inSeqTemp = np.stack(inSeqTemp, axis=0)
            
            # append to the input sequence for this turn and continue
            inputSequences.append(inSeqTemp)
            
            count += 1
            #print("{} / {}".format(count, len(interactions[i])))
    
    
    fn = interactionFilenamesAll[i].split("/")[-1][:-4] + "_input_sequence_vectors_sl{}_dim{}".format(inputSeqCutoffLen, jsvDim)
    
    inputSequences = np.stack(inputSequences, axis=0)
    inputSequences = inputSequences.astype(np.float32)
    np.save(sessionDir+"/"+fn, inputSequences)
    
        
            
    