'''
Created on Jul 24, 2019

@author: robovie


copied from datapreprocessing1_dbl
but use shopkeeper clusters and string search camera indices for training targets
'''


import csv
import numpy as np
import string
import os
import ast
import nltk

import tools
from utterancevectorizer import UtteranceVectorizer


dataDirectory = tools.dataDir+"2020-01-08_advancedSimulator9"

#sessionDir = tools.create_session_dir("datapreprocessing2_dbl")
sessionDir = dataDirectory+"_input_sequence_vectors"
tools.create_directory(sessionDir)


shopkeeperSpeechClusterFilename = tools.modelDir + "20200109_withsymbols shopkeeper-tristm-3wgtkw-9wgtsym-3wgtnum-mc2-sw2-eucldist- speech_clusters.csv"
#shopkeeperSpeechClusterFilename = tools.modelDir + "20200116_nosymbols shopkeeper-tristm-3wgtkw-9wgtnum-mc2-sw3-eucldist- speech_clusters.csv"


numTrainDbs = 10
numInteractionsPerDb = 200
dtype = np.int8
inputSeqCutoffLen = 10
useSymbols = True


cameras = ["CAMERA_1", "CAMERA_2", "CAMERA_3"]
attributes = ["camera_ID", "camera_name", "camera_type", "color", "weight", "preset_modes", "effects", "price", "resolution", "optical_zoom", "settings", "autofocus_points", "sensor_size", "ISO", "long_exposure"]

spatialFormations = ["NONE", "WAITING", "FACE_TO_FACE", "PRESENT_X"]
stateTargets = ["NONE", "CAMERA_1", "CAMERA_2", "CAMERA_3"]
locations = ["DOOR", "MIDDLE", "SERVICE_COUNTER", "CAMERA_1", "CAMERA_2", "CAMERA_3"]



def read_simulated_interactions(filename, dbFieldnames, numInteractionsPerDb, keepActions=None):
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
                break # only load the first numInteractionsPerDb interactions
            
            if (keepActions == None) or (row["OUTPUT_SHOPKEEPER_ACTION"] in keepActions): # and row["SHOPKEEPER_TOPIC"] == "price"):
                
                row["CUSTOMER_SPEECH"] = row["CUSTOMER_SPEECH"].lower().translate(str.maketrans('', '', tools.punctuation))
                row["SHOPKEEPER_SPEECH"] = row["SHOPKEEPER_SPEECH"].lower().translate(str.maketrans('', '', tools.punctuation))
                
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
            
            if row["SHOPKEEPER_SPEECH_DB_ENTRY_RANGE"] != "" and row["SHOPKEEPER_SPEECH_DB_ENTRY_RANGE"] != "NA":
                shkpUttToDbEntryRange[row["SHOPKEEPER_SPEECH"]] = [int(i) for i in row["SHOPKEEPER_SPEECH_DB_ENTRY_RANGE"].split("~")]
            
            elif row["SHOPKEEPER_SPEECH_DB_ENTRY_RANGE"] == "NA":
                shkpUttToDbEntryRange[row["SHOPKEEPER_SPEECH"]] = "NA"
            
            
            row["SYMBOL_MATCH_SUBSTRINGS"] = ast.literal_eval(row["SYMBOL_MATCH_SUBSTRINGS"])
            row["SYMBOL_CANDIDATE_DATABASE_INDICES"] = ast.literal_eval(row["SYMBOL_CANDIDATE_DATABASE_INDICES"])
            row["SYMBOL_CANDIDATE_DATABASE_CONTENTS"] = ast.literal_eval(row["SYMBOL_CANDIDATE_DATABASE_CONTENTS"])
            
            
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

databaseFilenamesAll = [dataDirectory+"/"+fn for fn in filenames if "handmade_database" in fn]
interactionFilenamesAll = [dataDirectory+"/"+fn for fn in filenames if "withsymbols" in fn]

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
    
    inters, sutder, gtDbCamera, gtDbAttribute = read_simulated_interactions(iFn, dbFieldnames, numInteractionsPerDb)#, keepActions=["S_ANSWERS_QUESTION_ABOUT_FEATURE"]) # S_INTRODUCES_CAMERA S_INTRODUCES_FEATURE
    
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
# find the vocabulary in the shopkeeper utterances and the database contents
#
indexToWord = {}
wordToIndex = {}

# shopkeeper utterances
maxShkpSpeechLen = 0

for i in range(len(interactions)):
    for j in range(len(interactions[i])):
        
        tokens = nltk.word_tokenize(interactions[i][j]["SHOPKEEPER_SPEECH"])

        maxShkpSpeechLen = max(maxShkpSpeechLen, len(tokens))
        
        for w in tokens:
            if w not in wordToIndex:
                wordToIndex[w] = len(wordToIndex)
                indexToWord[wordToIndex[w]] = w

# database contents
maxCamLen = 0
maxAttrLen = 0
maxValLen = 0

for db in databases:
    for c in db:
        for attr, val in c.items():
            tokens = nltk.word_tokenize(val.lower().translate(str.maketrans('', '', tools.punctuation)))
            
            maxValLen = max(maxValLen, len(tokens))
            
            for w in tokens:
                if w not in wordToIndex:
                    wordToIndex[w] = len(wordToIndex)
                    indexToWord[wordToIndex[w]] = w
                
for attr in dbFieldnames:
    tokens = nltk.word_tokenize(attr.lower().translate(str.maketrans('', '', tools.punctuation)))
    
    maxAttrLen = max(maxAttrLen, len(tokens))
    
    for w in tokens:
        if w not in wordToIndex:
            wordToIndex[w] = len(wordToIndex)
            indexToWord[wordToIndex[w]] = w

for cam in cameras:
    tokens = nltk.word_tokenize(cam.lower().translate(str.maketrans('', '', tools.punctuation)))
    
    maxCamLen = max(maxCamLen, len(tokens))
    
    for w in tokens:
        if w not in wordToIndex:
            wordToIndex[w] = len(wordToIndex)
            indexToWord[wordToIndex[w]] = w



goToken = "<go>"
eofToken = "<eof>"

wordToIndex[goToken] = len(wordToIndex)
indexToWord[wordToIndex[goToken]] = goToken

wordToIndex[eofToken] = len(wordToIndex)
indexToWord[wordToIndex[eofToken]] = eofToken

# add 1 for EOF token
maxShkpSpeechLen += 1 
maxCamLen += 1
maxAttrLen += 1
maxValLen += 1

vocabSize = len(indexToWord)


# save the vocab
with open(sessionDir+"/db_and_output_vocab.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, ["INDEX", "TOKEN"])
    writer.writeheader()
    
    for i, w in indexToWord.items():
        row = {}
        row["INDEX"] = i
        row["TOKEN"] = w 
        writer.writerow(row)


#
# save the DB sequence vectorizations
#

# format 1
for i in range(len(databases)):
    db = []
    
    for c in range(len(databases[i])):
        valSeqs = []
        
        for attr in dbFieldnames:
            
            cam = databases[i][c]["camera_ID"]
            val = databases[i][c][attr]
            
            valSeq = []
            tokens = nltk.word_tokenize(val.lower().translate(str.maketrans('', '', tools.punctuation)))
            
            for w in tokens:
                valSeq.append(wordToIndex[w])
            valSeq.append(wordToIndex[eofToken])
            
            valSeq += [-1] * (maxValLen - len(valSeq))
            
            valSeqs.append(valSeq)
        
        db.append(valSeqs)
    
    db = np.asarray(db)
    fn = interactionFilenamesAll[i].split("/")[-1][:-4] + "_database_vector_sequences".format(i)
    np.save(sessionDir+"/"+fn, db)


# format 2 - as in the COREQA paper (He at al., 2017)
print("DB camera length:", maxCamLen)
print("DB attribute length:", maxAttrLen)
print("DB value length:", maxValLen)


for i in range(len(databases)):
    dbCams = []
    dbAttrs = []
    dbVals = []
    
    for c in range(len(databases[i])):
        for attr in dbFieldnames:
            
            cam = databases[i][c]["camera_ID"]
            val = databases[i][c][attr]
            
            camTokens = nltk.word_tokenize(cam.lower().translate(str.maketrans('', '', tools.punctuation)))
            attrTokens = nltk.word_tokenize(attr.lower().translate(str.maketrans('', '', tools.punctuation)))
            valTokens = nltk.word_tokenize(val.lower().translate(str.maketrans('', '', tools.punctuation)))
            
            
            camSeq = [wordToIndex[w] for w in camTokens]
            camSeq.append(wordToIndex[eofToken])
            camSeq += [-1] * (maxCamLen - len(camSeq))
            
            attrSeq = [wordToIndex[w] for w in attrTokens]
            attrSeq.append(wordToIndex[eofToken])
            attrSeq += [-1] * (maxAttrLen - len(attrSeq))
            
            valSeq = [wordToIndex[w] for w in valTokens]
            valSeq.append(wordToIndex[eofToken])
            valSeq += [-1] * (maxValLen - len(valSeq))
            
            camSeq = np.asarray(camSeq)
            attrSeq = np.asarray(attrSeq)
            valSeq = np.asarray(valSeq)
            
            
            dbCams.append(camSeq)
            dbAttrs.append(attrSeq)
            dbVals.append(valSeq)
            
            
    
    dbCams = np.asarray(dbCams)
    fn = interactionFilenamesAll[i].split("/")[-1][:-4] + "_database_fact_cams".format(i)
    np.save(sessionDir+"/"+fn, dbCams)
    
    dbAttrs = np.asarray(dbAttrs)
    fn = interactionFilenamesAll[i].split("/")[-1][:-4] + "_database_fact_attrs".format(i)
    np.save(sessionDir+"/"+fn, dbAttrs)
    
    dbVals = np.asarray(dbVals)
    fn = interactionFilenamesAll[i].split("/")[-1][:-4] + "_database_fact_vals".format(i)
    np.save(sessionDir+"/"+fn, dbVals)



#
# load the shopkeeper speech clusters
#
shkpUttToSpeechClustId, shkpSpeechClustIdToRepUtt, speechClustIdToShkpUtts, junkSpeechClusterIds = tools.load_shopkeeper_speech_clusters(shopkeeperSpeechClusterFilename, cleanPunct=not useSymbols)
print(len(speechClustIdToShkpUtts), "shopkeeper speech clusters")


#
# find set of output shopkeeper action clusters
# an action consists of the shopkeeper speech cluster + OUTPUT_SHOPKEEPER_LOCATION, OUTPUT_SPATIAL_STATE, and OUTPUT_STATE_TARGET
#
#tupleToSpatialClustId = {}
#spatialClustIdToTuple = {}
"""
tupleToShkpActionId = {}
shkpActionIdToTuple = {}

uniqueShkpUttsWithSymbols = []

for i in range(len(interactions)):
    for j in range(len(interactions[i])):
        
        shkpSpeech = interactions[i][j]["SHOPKEEPER_SPEECH_WITH_SYMBOLS"].lower()
        shkpSpeechClustId = shkpUttToSpeechClustId[shkpSpeech]
        outShkpLoc = interactions[i][j]["OUTPUT_SHOPKEEPER_LOCATION"]
        outShkpSpatSt = interactions[i][j]["OUTPUT_SPATIAL_STATE"]
        outShkpStTarg = interactions[i][j]["OUTPUT_STATE_TARGET"]
        
        if interactions[i][j]["SHOPKEEPER_SPEECH_WITH_SYMBOLS"] not in uniqueShkpUttsWithSymbols:
            uniqueShkpUttsWithSymbols.append(interactions[i][j]["SHOPKEEPER_SPEECH_WITH_SYMBOLS"])
        
        shkpActionTuple = (shkpSpeechClustId, outShkpLoc, outShkpSpatSt, outShkpStTarg)
        
        if shkpActionTuple not in tupleToShkpActionId:
            tupleToShkpActionId[shkpActionTuple] = len(tupleToShkpActionId)
        
        shkpActionIdToTuple[tupleToShkpActionId[shkpActionTuple]] = shkpActionTuple
        
        interactions[i][j]["OUTPUT_SHOPKEEPER_ACTION_CLUSTER_ID"] = tupleToShkpActionId[shkpActionTuple]
        interactions[i][j]["OUTPUT_SHOPKEEPER_ACTION_CLUSTER_TUPLE"] = shkpActionTuple

print(len(shkpActionIdToTuple), "shopkeeper action clusters")
print(len(uniqueShkpUttsWithSymbols), "unique shopkeeper utterances (with symbols)")

# save the actions 
with open(sessionDir+"/shopkeeper_action_clusters.csv", "w", newline="") as csvfile:
        
    fieldnames = ["ACTION_CLUSTER_ID", "SPEECH_CLUSTER_ID", "OUTPUT_SHOPKEEPER_LOCATION", "OUTPUT_SPATIAL_STATE", "OUTPUT_STATE_TARGET"]
    
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for actionId in shkpActionIdToTuple:
        row = {"ACTION_CLUSTER_ID":actionId, 
               "SPEECH_CLUSTER_ID":shkpActionIdToTuple[actionId][0], 
               "OUTPUT_SHOPKEEPER_LOCATION":shkpActionIdToTuple[actionId][1], 
               "OUTPUT_SPATIAL_STATE":shkpActionIdToTuple[actionId][2], 
               "OUTPUT_STATE_TARGET":shkpActionIdToTuple[actionId][3]
               }
        
        writer.writerow(row)
"""


#
# add shopkeeper speech cluster info to the interaction data
#
for i in range(len(interactions)):
    for j in range(len(interactions[i])):
        
        if useSymbols:
            shkpSpeech = interactions[i][j]["SHOPKEEPER_SPEECH_WITH_SYMBOLS"].lower()
        else:
            shkpSpeech = interactions[i][j]["SHOPKEEPER_SPEECH"]
        
        shkpSpeechClustId = shkpUttToSpeechClustId[shkpSpeech]        
        interactions[i][j]["OUTPUT_SHOPKEEPER_SPEECH_CLUSTER_ID"] = shkpSpeechClustId
        

#
# find the DB camera indices and attribute indices to be used for training targets
#
# camera options are CAMERA_1, CAMERA_2, CAMERA_3
# attribute options are camera_ID, camera_name, camera_type, color, weight, preset_modes, effects, price, resolution, optical_zoom, settings, autofocus_points, sensor_size, ISO, long_exposure
#
# attr vectors will be 0,1 binary vector that can have more than one 1 label (multilabel classification)
# cam vectors will be the same, but can only have one possible label
#
for i in range(len(interactions)):
    for j in range(len(interactions[i])):
        dbIndices = interactions[i][j]["SYMBOL_CANDIDATE_DATABASE_INDICES"]
        
        cameraIndices = []
        attributeIndices = []
        
        for k in range(len(dbIndices)):
            #cameraIndices.append(dbIndices[k][0])
            attributeIndices.append(dbIndices[k][1])
        
        # if there is only one index tuple, then use the camera index from that
        # if there are multiple index tuples, then use the camera index that occurs most frequently
        if len(dbIndices) > 0:
            allCamIndices = [x[0] for x in dbIndices]
            camIndex = max(set(allCamIndices), key=allCamIndices.count)
            cameraIndices.append(camIndex)
        
        interactions[i][j]["OUTPUT_CAMERA_INDEX"] = cameraIndices
        interactions[i][j]["OUTPUT_ATTRIBUTE_INDEX"] = attributeIndices


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

uniqueShkpUtts = list(set(allShkpUtts))

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
    

print(len(prevShkpUttVec), "shopkeeper utterance vector length")
print(len(prevShkpLocVec), "shopkeeper location vector length")
print(len(prevSpatFormVec), "spatial state vector length")
print(len(prevStateTargVec), "state target vector length")
print(len(custUttVec), "customer utterance vector length")
print(len(custLocVec), "customer location vector length")


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

print(np.mean(allInteractionLens), np.std(allInteractionLens), "average interaction length")

numDataOPointsSaved = 0

for i in range(len(interactions)):
    
    print("processing", i+1, "of", len(interactions), "...")
    
    inputSequences = []
    #outputActionIds = []
    
    outputSpeechClusters = []
    outputSpeechSequences = []
    outputSpeechSequenceLens = []
    outputLocations = []
    outputSpatialStates = []
    outputStateTargets = []
    
    outputCameraIndices = []
    outputAttributeIndices = []
    #outputDbIndexMasks = []
    databaseIds = []
    
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
            
            # append the outputs
            #outputActionIds.append(interactions[i][j]["OUTPUT_SHOPKEEPER_ACTION_CLUSTER_ID"])
            
            # create the camera and attribute output index target vectors
            cameraIndexOutputs = np.zeros(len(cameras))
            for index in interactions[i][j]["OUTPUT_CAMERA_INDEX"]:
                cameraIndexOutputs[index] = 1
            
            attributeIndexOutputs = np.zeros(len(attributes))
            for index in interactions[i][j]["OUTPUT_ATTRIBUTE_INDEX"]:
                attributeIndexOutputs[index] = 1
            
            
            
            outputSpeechClusters.append(interactions[i][j]["OUTPUT_SHOPKEEPER_SPEECH_CLUSTER_ID"])
            outputLocations.append(interactions[i][j]["OUTPUT_SHOPKEEPER_LOCATION"])
            outputSpatialStates.append(interactions[i][j]["OUTPUT_SPATIAL_STATE"])
            outputStateTargets.append(interactions[i][j]["OUTPUT_STATE_TARGET"])
            
            outputCameraIndices.append(cameraIndexOutputs)
            outputAttributeIndices.append(attributeIndexOutputs)
            databaseIds.append(int(interactions[i][j]["DATABASE_ID"]))
            
            
            #
            outputSpeechSeq = []
            tokens = nltk.word_tokenize(interactions[i][j]["SHOPKEEPER_SPEECH"])
            
            for w in tokens:
                outputSpeechSeq.append(wordToIndex[w])
            outputSpeechSeq.append(wordToIndex[eofToken])
            
            outputSpeechSeq += [-1] * (maxShkpSpeechLen - len(outputSpeechSeq))
            outputSpeechSequences.append(outputSpeechSeq)
            
            outputSpeechSequenceLens.append(len(tokens)+1)
            #
            
            
            count += 1
            #print("{} / {}".format(count, len(interactions[i])))
    
    
    fn = interactionFilenamesAll[i].split("/")[-1][:-4] + "_input_sequence_vectors_sl{}_dim{}".format(inputSeqCutoffLen, jsvDim)
    inputSequences = np.stack(inputSequences, axis=0)
    inputSequences = inputSequences.astype(np.float32)
    np.save(sessionDir+"/"+fn, inputSequences)
    
    #fn = interactionFilenamesAll[i].split("/")[-1][:-4] + "_output_action_ids"
    #outputActionIds = np.asarray(outputActionIds)
    #np.save(sessionDir+"/"+fn, outputActionIds)
    
    
    if useSymbols:
        fn = interactionFilenamesAll[i].split("/")[-1][:-4] + "_output_speech_cluster_ids_withsymbols"
    else:
        fn = interactionFilenamesAll[i].split("/")[-1][:-4] + "_output_speech_cluster_ids_nosymbols"
    outputSpeechClusters = np.asarray(outputSpeechClusters)
    np.save(sessionDir+"/"+fn, outputSpeechClusters)
    
    
    fn = interactionFilenamesAll[i].split("/")[-1][:-4] + "_output_speech_vector_sequences"
    outputSpeechSequences = np.asarray(outputSpeechSequences)
    np.save(sessionDir+"/"+fn, outputSpeechSequences)
    
    fn = interactionFilenamesAll[i].split("/")[-1][:-4] + "_output_speech_vector_sequence_lens"
    outputSpeechSequenceLens = np.asarray(outputSpeechSequenceLens)
    np.save(sessionDir+"/"+fn, outputSpeechSequenceLens)
    
    fn = interactionFilenamesAll[i].split("/")[-1][:-4] + "_output_locations"
    outputLocations = np.asarray(outputLocations)
    np.save(sessionDir+"/"+fn, outputLocations)
    
    fn = interactionFilenamesAll[i].split("/")[-1][:-4] + "_output_spatial_states"
    outputSpatialStates = np.asarray(outputSpatialStates)
    np.save(sessionDir+"/"+fn, outputSpatialStates)
    
    fn = interactionFilenamesAll[i].split("/")[-1][:-4] + "_output_state_targets"
    outputStateTargets = np.asarray(outputStateTargets)
    np.save(sessionDir+"/"+fn, outputStateTargets)
    
    fn = interactionFilenamesAll[i].split("/")[-1][:-4] + "_output_camera_indices"
    outputCameraIndices = np.asarray(outputCameraIndices)
    np.save(sessionDir+"/"+fn, outputCameraIndices)
    
    fn = interactionFilenamesAll[i].split("/")[-1][:-4] + "_output_attribute_indices"
    outputCameraIndices = np.asarray(outputAttributeIndices)
    np.save(sessionDir+"/"+fn, outputAttributeIndices)
    
    fn = interactionFilenamesAll[i].split("/")[-1][:-4] + "_database_indices"
    databaseIds = np.asarray(databaseIds)
    np.save(sessionDir+"/"+fn, databaseIds)
    
    
    print("completed", i+1, "of", len(interactions))
    numDataOPointsSaved += len(outputSpeechClusters)
    
    
print(sum(len(x) for x in interactions), "total data points")
print(numDataOPointsSaved, "data points saved")
    