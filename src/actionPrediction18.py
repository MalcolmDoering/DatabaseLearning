'''
Created on April 9, 2019

@author: MalcolmD


modified from actionPrediction17

make it so that proposed and baselines can be run in the same scripts (to ensure that the same data is used for training/testing for each, etc.)

'''


import numpy as np
from six.moves import range
import editdistance
import csv
import os
import time
import random
import sys
import string
import ast
from sklearn.metrics import accuracy_score, jaccard_score
from multiprocessing import Process
import traceback
from operator import itemgetter
import copy
from tabulate import tabulate

import tools




#################################################################################################################
# running params
#################################################################################################################

DEBUG = True
RUN_PARALLEL = False
sessionDir = tools.create_session_dir("actionPrediction18_dbl")


# what to run. only one of these should be true at a time
prop_run = True
bl1_run = False


# params that should be the same for all conditions (predictors)
numDatabases = 11
numTrainDbs = 8
numValDbs = 2
numTestDbs = 1

numInteractionsPerDb = 200
batchSize = 64
randomizeTrainingBatches = False
numEpochs = 1000
evalEvery = 1

dataDirectory = tools.dataDir+"2020-01-08_advancedSimulator9" # handmade databases, customer-driven interactions, deterministic introductions, crowdsourced shopkeeper utts
inputSequenceVectorDirectory = dataDirectory + "_input_sequence_vectors"

inputSeqLen = 10 
inputDim = 2226


# params for proposed
prop_embeddingSize = 100
prop_shopkeeperActionClusterFilename = inputSequenceVectorDirectory+"/shopkeeper_action_clusters.csv"
prop_shopkeeperSpeechClusterFilename = tools.modelDir + "20200109_withsymbols shopkeeper-tristm-3wgtkw-9wgtsym-3wgtnum-mc2-sw2-eucldist- speech_clusters.csv"


# params for baseline 1
bl1_embeddingSize = 100
bl1_shopkeeperActionClusterFilename = inputSequenceVectorDirectory+"/shopkeeper_action_clusters.csv"
bl1_shopkeeperSpeechClusterFilename = tools.modelDir + "20200109_withsymbols shopkeeper-tristm-3wgtkw-9wgtsym-3wgtnum-mc2-sw2-eucldist- speech_clusters.csv"


#################################################################################################################
# global stuff
#################################################################################################################
eosChar = "#"
goChar = "~"

cameras = ["CAMERA_1", "CAMERA_2", "CAMERA_3"]
attributes = ["camera_ID", "camera_name", "camera_type", "color", "weight", "preset_modes", "effects", "price", "resolution", "optical_zoom", "settings", "autofocus_points", "sensor_size", "ISO", "long_exposure"]

locations = ["CAMERA_1", "CAMERA_2", "CAMERA_3", "SERVICE_COUNTER"] # for the shopkeeper
spatialStates = ["WAITING", "FACE_TO_FACE", "PRESENT_X"]
stateTargets = ["CAMERA_1", "CAMERA_2", "CAMERA_3", "NONE"]


#################################################################################################################
# function definitions
#################################################################################################################

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


def get_db_content_lens(database, dbFieldnames):
    
    numCams = len(cameras)
    numFields = len(dbFieldnames) 
    
    contentLens = np.zeros((numCams, numFields))
    
    for i in range(numCams):
        for j in range(numFields):
            contentLens[i,j] = 1 if (len(database[i][dbFieldnames[j]]) > 0) else 0
    
    return contentLens


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
    
    
    
    with open(filename, encoding="cp932") as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            
            if int(row["TRIAL"]) >= numInteractionsPerDb:
                break # only load the first numInteractionsPerDb interactions
            
            if (keepActions == None) or (row["OUTPUT_SHOPKEEPER_ACTION"] in keepActions): # and row["SHOPKEEPER_TOPIC"] == "price"):
                
                row["CUSTOMER_SPEECH"] = row["CUSTOMER_SPEECH"].lower().translate(str.maketrans('', '', string.punctuation))
                row["SHOPKEEPER_SPEECH"] = row["SHOPKEEPER_SPEECH"].lower().translate(str.maketrans('', '', string.punctuation))
                
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


#################################################################################################################
# setup terminal output / file
#################################################################################################################
sessionTerminalOutputLogFile = sessionDir + "/terminal_output_log_main.txt"


if DEBUG:
    sessionTerminalOutputStream = sys.stdout
else:
    sessionTerminalOutputStream = open(sessionTerminalOutputLogFile, "a")


#################################################################################################################
# load the data
#################################################################################################################
print("loading data...", flush=True, file=sessionTerminalOutputStream)

filenames = os.listdir(dataDirectory)
filenames.sort()

databaseFilenamesAll = [dataDirectory+"/"+fn for fn in filenames if "handmade_database" in fn]
interactionFilenamesAll = [dataDirectory+"/"+fn for fn in filenames if "withsymbols" in fn]

databaseFilenames = databaseFilenamesAll[:numDatabases]
interactionFilenames = interactionFilenamesAll[:numDatabases]


# load databases
databases = []
databaseConentLengths = []
databaseIds = []
dbFieldnames = None # these should be the same for all DBs

for dbFn in databaseFilenames:
    db, dbFieldnames = read_database_file(dbFn)
    
    databaseIds.append(dbFn.split("_")[-1].split(".")[0])
    databaseConentLengths.append(get_db_content_lens(db, dbFieldnames))
    databases.append(db)


# load other data
interactions = []
datasetSizes = []
databaseConentLengthsForInput = []

gtDatabaseCameras = []
gtDatabaseAttributes = []

shkpUttToDbEntryRange = {}

for i in range(len(interactionFilenames)):
    iFn = interactionFilenames[i]
    
    inters, sutder, gtDbCamera, gtDbAttribute = read_simulated_interactions(iFn, dbFieldnames, numInteractionsPerDb)
    
    interactions += inters
    datasetSizes.append(len(inters))
    databaseConentLengthsForInput += [databaseConentLengths[i]] * datasetSizes[i]
    
    gtDatabaseCameras += gtDbCamera
    gtDatabaseAttributes += gtDbAttribute
    
    # combine the three dictionaries into one
    shkpUttToDbEntryRange = {**shkpUttToDbEntryRange, **sutder}

databaseConentLengthsForInput = np.asarray(databaseConentLengthsForInput)

totalSamples = sum(datasetSizes)

#
# load the input sequence vectors and outputs
#
inputSequenceVectors = []
outputShopkeeperLocations = []
outputSpatialStates = []
outputStateTargets = []

for i in range(numDatabases):
    iFn = interactionFilenames[i]
    
    # input sequence vectors
    fn = iFn.split("/")[-1][:-4] + "_input_sequence_vectors_sl{}_dim{}.npy".format(inputSeqLen, inputDim)
    temp = np.load(inputSequenceVectorDirectory+"/"+fn)
    inputSequenceVectors.append(temp)
    del temp
    
    # locations
    fn = iFn.split("/")[-1][:-4] + "_output_locations.npy"
    temp = np.load(inputSequenceVectorDirectory+"/"+fn)
    temp = [locations.index(x) for x in temp]
    outputShopkeeperLocations.append(temp)
            
    # spatial states
    fn = iFn.split("/")[-1][:-4] + "_output_spatial_states.npy"
    temp = np.load(inputSequenceVectorDirectory+"/"+fn)
    temp = [spatialStates.index(x) for x in temp]
    outputSpatialStates.append(temp)
    
    # state targets
    fn = iFn.split("/")[-1][:-4] + "_output_state_targets.npy"
    temp = np.load(inputSequenceVectorDirectory+"/"+fn)
    temp = [stateTargets.index(x) for x in temp]
    outputStateTargets.append(temp)
    
inputSequenceVectors = np.concatenate(inputSequenceVectors)
outputShopkeeperLocations = np.concatenate(outputShopkeeperLocations)
outputSpatialStates = np.concatenate(outputSpatialStates)
outputStateTargets = np.concatenate(outputStateTargets)


#
# for PROPOSED
#
if prop_run:
    
    # load the shopkeeper speech clusters
    prop_shkpUttToSpeechClustId, prop_shkpSpeechClustIdToRepUtt, prop_speechClustIdToShkpUtts, prop_junkSpeechClusterIds = tools.load_shopkeeper_speech_clusters(prop_shopkeeperSpeechClusterFilename)
    prop_numSpeechClusters = len(prop_speechClustIdToShkpUtts)
    print(prop_numSpeechClusters, "shopkeeper speech clusters for PROPOSED", flush=True, file=sessionTerminalOutputStream)
    
    
    # load the output targets
    prop_outputSpeechClusterIds = []
    prop_outputCameraIndices = []
    prop_outputAttributeIndices = []
    
    for i in range(numDatabases):
        iFn = interactionFilenames[i]
        
        # speech clusters
        fn = iFn.split("/")[-1][:-4] + "_output_speech_cluster_ids.npy"
        prop_outputSpeechClusterIds.append(np.load(inputSequenceVectorDirectory+"/"+fn))
        
        # output DB camera index
        fn = iFn.split("/")[-1][:-4] + "_output_camera_indices.npy"
        prop_outputCameraIndices.append(np.load(inputSequenceVectorDirectory+"/"+fn))
        
        # output DB attribute index
        fn = iFn.split("/")[-1][:-4] + "_output_attribute_indices.npy"
        prop_outputAttributeIndices.append(np.load(inputSequenceVectorDirectory+"/"+fn))
    
    prop_outputSpeechClusterIds = np.concatenate(prop_outputSpeechClusterIds)
    prop_outputCameraIndices = np.concatenate(prop_outputCameraIndices)
    prop_outputAttributeIndices = np.concatenate(prop_outputAttributeIndices)
    

#
# for BASELINE 1
#
if bl1_run:
    
    # load the shopkeeper speech clusters
    bl1_shkpUttToSpeechClustId, bl1_shkpSpeechClustIdToRepUtt, bl1_speechClustIdToShkpUtts, bl1_junkSpeechClusterIds = tools.load_shopkeeper_speech_clusters(bl1_shopkeeperSpeechClusterFilename)
    bl1_numSpeechClusters = len(bl1_speechClustIdToShkpUtts)
    print(bl1_numSpeechClusters, "shopkeeper speech clusters for BASELINE 1", flush=True, file=sessionTerminalOutputStream)
    
    
    # load the output targets
    bl1_outputSpeechClusterIds = []
        
    for i in range(numDatabases):
        iFn = interactionFilenames[i]
        
        # speech clusters
        fn = iFn.split("/")[-1][:-4] + "_output_speech_cluster_ids.npy"
        bl1_outputSpeechClusterIds.append(np.load(inputSequenceVectorDirectory+"/"+fn))
    
    bl1_outputSpeechClusterIds = np.concatenate(bl1_outputSpeechClusterIds)



#################################################################################################################
# split the data into train, val, and test sets
#################################################################################################################
print("splitting data...", flush=True, file=sessionTerminalOutputStream)

# get the start and end indices for the data with each database for after put all the data into a single list
dbStartEndIndices = []

for i in range(len(datasetSizes)):
    numSamplesForDb = datasetSizes[i]
    
    if len(dbStartEndIndices) == 0: 
        start = 0
    else:
        start = dbStartEndIndices[-1][1]
    
    end = start + numSamplesForDb
    
    dbStartEndIndices.append((start, end))


trainSplits = [] # each training set should consist of all the data from 8 databases
valSplits = [] # each validation set should consist of all the data from 2 databases
testSplits = [] # each testing set should consist of all the data from 1 database


for i in range(numDatabases):
    trainDbs = [(i+j) % numDatabases for j in range(numTrainDbs)]
    valDbs = [(i+j+numTrainDbs) % numDatabases for j in range(numValDbs)]
    testDbs = [(i+j+numTrainDbs+numValDbs) % numDatabases for j in range(numTestDbs)]
    
    temp = []
    for db in trainDbs:
        temp += range(dbStartEndIndices[db][0], dbStartEndIndices[db][1])
    trainSplits.append(temp)
    
    temp = []
    for db in valDbs:
        temp += range(dbStartEndIndices[db][0], dbStartEndIndices[db][1])
    valSplits.append(temp)
    
    temp = []
    for db in testDbs:
        temp += range(dbStartEndIndices[db][0], dbStartEndIndices[db][1])
    testSplits.append(temp)



#################################################################################################################
# parralell process each fold
#################################################################################################################

def run_fold(foldId, randomSeed, gpu):
    
    #################################################################################################################
    # setup logging directories
    #################################################################################################################
    foldIdentifier = "rs{}_fold{}".format(randomSeed, foldId)
    
    foldDir = tools.create_directory(sessionDir + "/" + foldIdentifier)
    
    foldLogFile = sessionDir + "/fold_log_{}.csv".format(foldIdentifier)
    foldTerminalOutputLogFile = foldDir + "/terminal_output_log_{}.txt".format(foldIdentifier)
    
    
    if DEBUG:
        foldTerminalOutputStream = sys.stdout
    else:
        foldTerminalOutputStream = open(foldTerminalOutputLogFile, "a")
    
    
    #
    # training / testing aggregate scores log file
    # and training / testing outputs log file
    #
    if prop_run:
        with open(foldLogFile, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Epoch",
                             
                             "Training Cost Ave ({})".format(foldIdentifier), 
                             "Training Cost SD ({})".format(foldIdentifier),
                             "Training Shopkeeper Action Loss Ave ({})".format(foldIdentifier),
                             "Training Shopkeeper Action Loss SD ({})".format(foldIdentifier),
                             "Training Camera Index Loss Ave ({})".format(foldIdentifier),
                             "Training Camera Index Loss SD ({})".format(foldIdentifier),
                             "Training Attribute Index Loss Ave ({})".format(foldIdentifier),
                             "Training Attribute Index SD Ave ({})".format(foldIdentifier),
                             "Training Location Loss Ave ({})".format(foldIdentifier),
                             "Training Location Loss SD ({})".format(foldIdentifier),
                             "Training Spatial State Loss Ave ({})".format(foldIdentifier),
                             "Training Spatial State SD Ave ({})".format(foldIdentifier),
                             "Training State Target Loss Ave ({})".format(foldIdentifier),
                             "Training State Target SD Ave ({})".format(foldIdentifier),
                             "Training Action ID Correct ({})".format(foldIdentifier), 
                             "Training Camera Index Correct ({})".format(foldIdentifier),
                             "Training Attribute Index Exact Match ({})".format(foldIdentifier),
                             "Training Attribute Index Jaccard Index ({})".format(foldIdentifier),
                             "Training Location Correct ({})".format(foldIdentifier),
                             "Training Spatial State Correct ({})".format(foldIdentifier),
                             "Train State Target Correct ({})".format(foldIdentifier),
                             
                             "Validation Cost Ave ({})".format(foldIdentifier),
                             "Validation Cost SD ({})".format(foldIdentifier),
                             "Validation Shopkeeper Action Loss Ave ({})".format(foldIdentifier),
                             "Validation Shopkeeper Action Loss SD ({})".format(foldIdentifier),
                             "Validation Camera Index Loss Ave ({})".format(foldIdentifier),
                             "Validation Camera Index Loss SD ({})".format(foldIdentifier),
                             "Validation Attribute Index Loss Ave ({})".format(foldIdentifier),
                             "Validation Attribute Index SD Ave ({})".format(foldIdentifier),
                             "Validation Location Loss Ave ({})".format(foldIdentifier),
                             "Validation Location Loss SD ({})".format(foldIdentifier),
                             "Validation Spatial State Loss Ave ({})".format(foldIdentifier),
                             "Validation Spatial State SD Ave ({})".format(foldIdentifier),
                             "Validation State Target Loss Ave ({})".format(foldIdentifier),
                             "Validation State Target SD Ave ({})".format(foldIdentifier),
                             "Validation Action ID Correct ({})".format(foldIdentifier), 
                             "Validation Camera Index Correct ({})".format(foldIdentifier), 
                             "Validation Attribute Index Exact Match ({})".format(foldIdentifier),
                             "Validation Attribute Index Jaccard Index ({})".format(foldIdentifier),
                             "Validation Location Correct ({})".format(foldIdentifier),
                             "Validation Spatial State Correct ({})".format(foldIdentifier),
                             "Validation State Target Correct ({})".format(foldIdentifier),
                             
                             "Testing Cost Ave ({})".format(foldIdentifier),
                             "Testing Cost SD ({})".format(foldIdentifier),
                             "Testing Shopkeeper Action Loss Ave ({})".format(foldIdentifier),
                             "Testing Shopkeeper Action Loss SD ({})".format(foldIdentifier),
                             "Testing Camera Index Loss Ave ({})".format(foldIdentifier),
                             "Testing Camera Index Loss SD ({})".format(foldIdentifier),
                             "Testing Attribute Index Loss Ave ({})".format(foldIdentifier),
                             "Testing Attribute Index SD Ave ({})".format(foldIdentifier),
                             "Testing Location Loss Ave ({})".format(foldIdentifier),
                             "Testing Location Loss SD ({})".format(foldIdentifier),
                             "Testing Spatial State Loss Ave ({})".format(foldIdentifier),
                             "Testing Spatial State SD Ave ({})".format(foldIdentifier),
                             "Testing State Target Loss Ave ({})".format(foldIdentifier),
                             "Testing State Target SD Ave ({})".format(foldIdentifier),
                             "Testing Action ID Correct ({})".format(foldIdentifier), 
                             "Testing Camera Index Correct ({})".format(foldIdentifier), 
                             "Testing Attribute Index Exact Match ({})".format(foldIdentifier),
                             "Testing Attribute Index Jaccard Index ({})".format(foldIdentifier),
                             "Testing Location Correct ({})".format(foldIdentifier),
                             "Testing Spatial State Correct ({})".format(foldIdentifier),
                             "Testing State Target Correct ({})".format(foldIdentifier)
                             ])
    
    
        interactionsFieldnames = ["SET",
                                  "ID",
                                  "TRIAL",
                                  "TURN_COUNT",
                                  
                                  "DATABASE_ID",
                                  
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
                                  "DATABASE_CONTENTS",
                                  "OUTPUT_SPATIAL_STATE",
                                  "OUTPUT_STATE_TARGET",
                                  "SHOPKEEPER_SPEECH_DB_ENTRY_RANGE",
                                  
                                  'SYMBOL_MATCH_SUBSTRINGS', 
                                  'SHOPKEEPER_SPEECH_WITH_SYMBOLS',
                                  "SHOPKEEPER_SPEECH_STRING_SEARCH_TOKENS",
                                  'SYMBOL_CANDIDATE_DATABASE_INDICES', 
                                  'SYMBOL_CANDIDATE_DATABASE_CONTENTS',
                                  
                                  "TARG_SHOPKEEPER_SPEECH_CLUSTER_ID",
                                  "TARG_SHOPKEEPER_SPEECH_CLUSTER_ID_IS_JUNK",
                                  "TARG_OUTPUT_CAMERA_INDEX",
                                  "TARG_ATTRIBUTE_INDEX",
                                  "TARG_SHOPKEEPER_SPEECH_TEMPLATE", 
                                  "TARG_OUTPUT_SHOPKEEPER_LOCATION", 
                                  "TARG_OUTPUT_SPATIAL_STATE", 
                                  "TARG_OUTPUT_STATE_TARGET",
                                  
                                  "PRED_OUTPUT_SHOPKEEPER_SPEECH_CLUSTER_ID",
                                  "PRED_OUTPUT_CAMERA_INDEX",
                                  "PRED_OUTPUT_CAMERA_INDEX_NO_NONE",
                                  "PRED_ATTRIBUTE_INDEX",
                                  "PRED_SHOPKEEPER_SPEECH",
                                  "PRED_SHOPKEEPER_SPEECH_TEMPLATE",
                                  "PRED_DB_INDICES",
                                  "PRED_DB_CONTENTS",
                                  "PRED_OUTPUT_SHOPKEEPER_LOCATION",
                                  "PRED_OUTPUT_SPATIAL_STATE",
                                  "PRED_OUTPUT_STATE_TARGET",
                                  "PRED_WEIGHTED_DB_CONTENT_SUM"]
    
        for c in cameras:
            interactionsFieldnames.append("{}_WEIGHT".format(c))
        for a in attributes:
            interactionsFieldnames.append("{}_WEIGHT".format(a))
    
    
    
    
    #################################################################################################################
    # do data preprocessing
    #################################################################################################################
    trainIndices = trainSplits[foldId]
    valIndices = valSplits[foldId]
    testIndices = testSplits[foldId]
    
    
    #################################################################################################################
    # compute loss weights for speech cluster and attribute targets
    #################################################################################################################
    
    if prop_run:
        #
        # output masks for loss (not used)
        #
        prop_outputMasks = np.ones(totalSamples)
        
        
        #
        # for speech clusters
        #
        
        # count number of occurrences of each speech cluster in the training dataset
        prop_speechClustCounts = {}
        
        for i in trainIndices:
            speechClustId = prop_outputSpeechClusterIds[i]
            
            if speechClustId not in prop_speechClustCounts:
                prop_speechClustCounts[speechClustId] = 0
            prop_speechClustCounts[speechClustId] += 1
        
        numSamples = len(trainIndices)
        
        
        # compute weights
        prop_speechClustWeights = [None] * prop_numSpeechClusters
        
        for clustId in prop_speechClustIdToShkpUtts:
            # as in scikit learn - The “balanced” heuristic is inspired by Logistic Regression in Rare Events Data, King, Zen, 2001.
            prop_speechClustWeights[clustId] = numSamples / (prop_numSpeechClusters * prop_speechClustCounts[clustId])
        
        # don't train on junk speech clusters
        for clustId in prop_junkSpeechClusterIds:
            prop_speechClustWeights[clustId] = 0
        
        
        if None in prop_speechClustWeights:
            print("WARNING: missing training weight for PROPOSED shopkeeper speech cluster!", flush=True, file=foldTerminalOutputStream)
    
        
        #
        # for attributes
        #
        
        # count the number of occurrences of each attribute index target in the training dataset
        prop_outputAttributeIndexCounts = {}
        numSamples = 0
        
        for a in attributes:
            prop_outputAttributeIndexCounts[a] = 0
        
        for i in trainIndices:
        
            if sum(prop_outputAttributeIndices[i]) < 1:
                continue
            
            numSamples += 1
            
            for k in range(len(attributes)):
                prop_outputAttributeIndexCounts[attributes[k]] += prop_outputAttributeIndices[i][k] # value will be either 0 or 1
        
        # compute weights
        # treat each attribute index sigmoid as a binary classifier, so need a weight for each class
        prop_attributeIndexWeights0 = [None] * len(attributes)
        prop_attributeIndexWeights1 = [None] * len(attributes)
        
        for a in prop_outputAttributeIndexCounts:
            if prop_outputAttributeIndexCounts[a] == 0:
                prop_outputAttributeIndexCounts[a] = 1 # just to make sure none are 0
            
            prop_attributeIndexWeights0[attributes.index(a)] = numSamples / (2 * (numSamples - prop_outputAttributeIndexCounts[a]))
            prop_attributeIndexWeights1[attributes.index(a)] = numSamples / (2 * prop_outputAttributeIndexCounts[a])
    
    
    if bl1_run:
        #
        # for speech clusters
        #
        
        # count number of occurrences of each speech cluster in the training dataset
        bl1_speechClustCounts = {}
        
        for i in trainIndices:
            speechClustId = bl1_outputSpeechClusterIds[i]
            
            if speechClustId not in bl1_speechClustCounts:
                bl1_speechClustCounts[speechClustId] = 0
            bl1_speechClustCounts[speechClustId] += 1
        
        numSamples = len(trainIndices)
        
        
        # compute weights
        bl1_speechClustWeights = [None] * bl1_numSpeechClusters
        
        for clustId in bl1_speechClustIdToShkpUtts:
            # as in scikit learn - The “balanced” heuristic is inspired by Logistic Regression in Rare Events Data, King, Zen, 2001.
            bl1_speechClustWeights[clustId] = numSamples / (bl1_numSpeechClusters * prop_speechClustCounts[clustId])
        
        if None in bl1_speechClustWeights:
            print("WARNING: missing training weight for BASELINE 1 shopkeeper speech cluster!", flush=True, file=foldTerminalOutputStream)
    
    
    
    #################################################################################################################
    # prepare the learning model
    #################################################################################################################
    print("setting up the model...", flush=True, file=foldTerminalOutputStream)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import learning4
    
    if prop_run:
        learner = learning4.CustomNeuralNetwork(inputDim=inputDim, 
                                                inputSeqLen=inputSeqLen, 
                                                numOutputClasses=prop_numSpeechClusters,
                                                numUniqueCams=len(cameras),
                                                numAttributes=len(dbFieldnames),
                                                numLocations = len(locations), # cam 1, 2, 3, service counter
                                                numSpatialStates = len(spatialStates), # f2f, preesnt x, waiting
                                                numStateTargets = len(stateTargets), # cam 1, 2, 3, NONE
                                                batchSize=batchSize,
                                                embeddingSize_prop=prop_embeddingSize,
                                                seed=randomSeed,
                                                speechClusterWeights=prop_speechClustWeights,
                                                attributeIndexWeights=[prop_attributeIndexWeights0, prop_attributeIndexWeights1]
                                                )
    
    if bl1_run:
        # TODO
        pass
    
    
    #################################################################################################################
    # train and test...
    #################################################################################################################
    print("training and testing...", flush=True, file=foldTerminalOutputStream)
    
    for e in range(numEpochs+1):
        
        if prop_run:
            
            if e != 0:
                # train
                trainCost, trainShopkeeperActionLoss, trainCameraIndexLoss, trainAttributeIndexLoss, trainLocationLoss, trainSpatialStateLoss, trainStateTargetLoss = learner.train(
                    inputSequenceVectors[trainIndices],
                    prop_outputSpeechClusterIds[trainIndices],
                    prop_outputCameraIndices[trainIndices],
                    prop_outputAttributeIndices[trainIndices],
                    outputShopkeeperLocations[trainIndices],
                    outputSpatialStates[trainIndices],
                    outputStateTargets[trainIndices],                                           
                    prop_outputMasks[trainIndices],
                    databaseConentLengthsForInput[trainIndices])
            
            else:
                # get loss for initialized network before any training is done
                trainCost, trainShopkeeperActionLoss, trainCameraIndexLoss, trainAttributeIndexLoss, trainLocationLoss, trainSpatialStateLoss, trainStateTargetLoss = learner.get_loss(
                    inputSequenceVectors[trainIndices],
                    prop_outputSpeechClusterIds[trainIndices],
                    prop_outputCameraIndices[trainIndices],
                    prop_outputAttributeIndices[trainIndices],
                    outputShopkeeperLocations[trainIndices],
                    outputSpatialStates[trainIndices],
                    outputStateTargets[trainIndices],                                           
                    prop_outputMasks[trainIndices],
                    databaseConentLengthsForInput[trainIndices])
            
            
            # test
            if (e-1) % evalEvery == 0:
                
                # validation loss
                valCost, valShopkeeperActionLoss, valCameraIndexLoss, valAttributeIndexLoss, valLocationLoss, valSpatialStateLoss, valStateTargetLoss = learner.train(
                    inputSequenceVectors[valIndices],
                    prop_outputSpeechClusterIds[valIndices],
                    prop_outputCameraIndices[valIndices],
                    prop_outputAttributeIndices[valIndices],
                    outputShopkeeperLocations[valIndices],
                    outputSpatialStates[valIndices],
                    outputStateTargets[valIndices],                                           
                    prop_outputMasks[valIndices],
                    databaseConentLengthsForInput[valIndices])
                
                # test loss
                testCost, testShopkeeperActionLoss, testCameraIndexLoss, testAttributeIndexLoss, testLocationLoss, testSpatialStateLoss, testStateTargetLoss = learner.train(
                    inputSequenceVectors[testIndices],
                    prop_outputSpeechClusterIds[testIndices],
                    prop_outputCameraIndices[testIndices],
                    prop_outputAttributeIndices[testIndices],
                    outputShopkeeperLocations[testIndices],
                    outputSpatialStates[testIndices],
                    outputStateTargets[testIndices],                                           
                    prop_outputMasks[testIndices],
                    databaseConentLengthsForInput[testIndices])
                
                # compute loss averages and s.d. for aggregate log
                # train
                trainCostAve = np.mean(trainCost)
                trainShopkeeperActionLossAve = np.mean(trainShopkeeperActionLoss)
                trainCameraIndexLossAve = np.mean(trainCameraIndexLoss)
                trainAttributeIndexLossAve = np.mean(trainAttributeIndexLoss)
                trainLocationLossAve = np.mean(trainLocationLoss)
                trainSpatialStateLossAve = np.mean(trainSpatialStateLoss)
                trainStateTargetLossAve = np.mean(trainStateTargetLoss)
                
                trainCostStd = np.std(trainCost)
                trainShopkeeperActionLossStd = np.std(trainShopkeeperActionLoss)
                trainCameraIndexLossStd = np.std(trainCameraIndexLoss)
                trainAttributeIndexLossStd = np.std(trainAttributeIndexLoss)
                trainLocationLossStd = np.std(trainLocationLoss)
                trainSpatialStateLossStd = np.std(trainSpatialStateLoss)
                trainStateTargetLossStd = np.std(trainStateTargetLoss)
                
                # validation
                valCostAve = np.mean(valCost)
                valShopkeeperActionLossAve = np.mean(valShopkeeperActionLoss)
                valCameraIndexLossAve = np.mean(valCameraIndexLoss)
                valAttributeIndexLossAve = np.mean(valAttributeIndexLoss)
                valLocationLossAve = np.mean(valLocationLoss)
                valSpatialStateLossAve = np.mean(valSpatialStateLoss)
                valStateTargetLossAve = np.mean(valStateTargetLoss)
                
                valCostStd = np.std(valCost)
                valShopkeeperActionLossStd = np.std(valShopkeeperActionLoss)
                valCameraIndexLossStd = np.std(valCameraIndexLoss)
                valAttributeIndexLossStd = np.std(valAttributeIndexLoss)
                valLocationLossStd = np.std(valLocationLoss)
                valSpatialStateLossStd = np.std(valSpatialStateLoss)
                valStateTargetLossStd = np.std(valStateTargetLoss)
                
                # test
                testCostAve = np.mean(testCost)
                testShopkeeperActionLossAve = np.mean(testShopkeeperActionLoss)
                testCameraIndexLossAve = np.mean(testCameraIndexLoss)
                testAttributeIndexLossAve = np.mean(testAttributeIndexLoss)
                testLocationLossAve = np.mean(testLocationLoss)
                testSpatialStateLossAve = np.mean(testSpatialStateLoss)
                testStateTargetLossAve = np.mean(testStateTargetLoss)
                
                testCostStd = np.std(testCost)
                testShopkeeperActionLossStd = np.std(testShopkeeperActionLoss)
                testCameraIndexLossStd = np.std(testCameraIndexLoss)
                testAttributeIndexLossStd = np.std(testAttributeIndexLoss)
                testLocationLossStd = np.std(testLocationLoss)
                testSpatialStateLossStd = np.std(testSpatialStateLoss)
                testStateTargetLossStd = np.std(testStateTargetLoss)
                
                
                # predict
                predShkpSpeechClustID, predCameraIndicesRaw, predAttrIndicesRaw, predLocations, predSpatialStates, predStateTargets, predCamIndexWeights, predAttributeIndexWeights, predWeightedDbContentsSum = learner.predict(
                    inputSequenceVectors,
                    prop_outputSpeechClusterIds,
                    prop_outputCameraIndices,
                    prop_outputAttributeIndices,
                    outputShopkeeperLocations, 
                    outputSpatialStates, 
                    outputStateTargets,
                    prop_outputMasks,
                    databaseConentLengthsForInput)
                
                
                def evaluate_predictions_prop(evalSetName, evalIndices, csvLogRows):
                    
                    # for computing accuracies
                    speechClusts_gt = []
                    speechClusts_pred = []
                    
                    cams_gt = []
                    cams_pred = []
                    
                    exactAttrs_gt = []
                    exactAttrs_pred = []
                    
                    setAttrs_gt = []
                    setAttrs_pred = []
                    
                    camAttrs_gt = []
                    camAttrs_pred = []
                    
                    locs_gt = []
                    locs_pred = []
                    
                    spatSts_gt = []
                    spatSts_pred = []
                    
                    stTargs_gt = []
                    stTargs_pred = []
                    
                    
                    for i in evalIndices:
                        
                        # check if the index is one of the ones that was cut off because of the batch size
                        if i >= len(predShkpSpeechClustID):
                            continue
                        
                        csvLogRows[i]["SET"] = evalSetName
                        csvLogRows[i]["ID"] = i
                        
                        #
                        # target info
                        #
                        csvLogRows[i]["TARG_SHOPKEEPER_SPEECH_CLUSTER_ID"] = prop_outputSpeechClusterIds[i]
                        csvLogRows[i]["TARG_SHOPKEEPER_SPEECH_CLUSTER_ID_IS_JUNK"] = 1 if prop_outputSpeechClusterIds[i] in prop_junkSpeechClusterIds else 0
                        csvLogRows[i]["TARG_OUTPUT_CAMERA_INDEX"] = prop_outputCameraIndices[i]
                        
                        targAttrIndexList = np.where(prop_outputAttributeIndices[i] == 1)[0].tolist()
                        csvLogRows[i]["TARG_ATTRIBUTE_INDEX"] = targAttrIndexList
                        
                        
                        if prop_outputSpeechClusterIds[i] not in prop_junkSpeechClusterIds:
                            csvLogRows[i]["TARG_SHOPKEEPER_SPEECH_TEMPLATE"] = prop_shkpSpeechClustIdToRepUtt[prop_outputSpeechClusterIds[i]]
                        else:
                            try:
                                csvLogRows[i]["TARG_SHOPKEEPER_SPEECH_TEMPLATE"] = prop_shkpSpeechClustIdToRepUtt[prop_outputSpeechClusterIds[i]]
                            except:
                                csvLogRows[i]["TARG_SHOPKEEPER_SPEECH_TEMPLATE"] = "THIS_JUNK_CLUST_HAS_NONE"
                        
                        
                        csvLogRows[i]["TARG_OUTPUT_SHOPKEEPER_LOCATION"] = locations[outputShopkeeperLocations[i]]
                        csvLogRows[i]["TARG_OUTPUT_SPATIAL_STATE"] = spatialStates[outputSpatialStates[i]]
                        csvLogRows[i]["TARG_OUTPUT_STATE_TARGET"] = stateTargets[outputStateTargets[i]]
                        
                        
                        #
                        # prediction info
                        #
                        
                        # DB indices
                        
                        # cam
                        camPred = predCameraIndicesRaw[i]
                        camPredArgmax = np.argmax(camPred)
                                    
                        if camPred[camPredArgmax] > 0.5:
                            camPredArgmaxOver05 = camPredArgmax
                        else:
                            camPredArgmaxOver05 = ""
                        
                        # attr
                        attrPred = predAttrIndicesRaw[i]
                        predAttrIndexList= np.where(attrPred > 0.5)[0].tolist() # can take more than one attribute
                        
                        # combined
                        outDbIndices = []
                        outDbContents = []
                        
                        
                        try:
                            shkpSpeechTemplate = prop_shkpSpeechClustIdToRepUtt[predShkpSpeechClustID[i]]
                        except Exception as err:
                            traceback.print_tb(err.__traceback__)
                            print(err, flush=True, file=foldTerminalOutputStream)
                            
                            shkpSpeechTemplate = "ERROR: No representative utterance found for cluster {}.".format(predShkpSpeechClustID[i])
                        
                        outShkpSpeechTemplate = shkpSpeechTemplate
                        
                        
                        # find any DB contents symbols in the speech template
                        dbId = interactions[i]["DATABASE_ID"]
                        cameraInfo = databases[int(dbId)][camPredArgmax]
                        
                        for j in range(len(dbFieldnames)):
                            attr = dbFieldnames[j]
                            symbol = "<{}>".format(attr.lower())
                            
                            if symbol in outShkpSpeechTemplate:
                                outShkpSpeech = outShkpSpeechTemplate.replace(symbol, cameraInfo[attr])
                                
                                outDbIndices.append((camPredArgmax, j))
                                outDbContents.append(cameraInfo[attr])
                            else:
                                outShkpSpeech = outShkpSpeechTemplate
                        
                        
                        csvLogRows[i]["PRED_OUTPUT_SHOPKEEPER_SPEECH_CLUSTER_ID"] = predShkpSpeechClustID[i]
                        csvLogRows[i]["PRED_OUTPUT_CAMERA_INDEX"] = camPredArgmaxOver05
                        csvLogRows[i]["PRED_OUTPUT_CAMERA_INDEX_NO_NONE"] = camPredArgmax
                        csvLogRows[i]["PRED_ATTRIBUTE_INDEX"] = predAttrIndexList
                        csvLogRows[i]["PRED_SHOPKEEPER_SPEECH"] = outShkpSpeech
                        csvLogRows[i]["PRED_SHOPKEEPER_SPEECH_TEMPLATE"] = outShkpSpeechTemplate
                        csvLogRows[i]["PRED_DB_INDICES"] = outDbIndices
                        csvLogRows[i]["PRED_DB_CONTENTS"] = outDbContents
                        csvLogRows[i]["PRED_OUTPUT_SHOPKEEPER_LOCATION"] = locations[predLocations[i]]
                        csvLogRows[i]["PRED_OUTPUT_SPATIAL_STATE"] = spatialStates[predSpatialStates[i]]
                        csvLogRows[i]["PRED_OUTPUT_STATE_TARGET"] = stateTargets[predStateTargets[i]]
                        
                        
                        csvLogRows[i]["PRED_WEIGHTED_DB_CONTENT_SUM"] = predWeightedDbContentsSum[i]
                        
                        for c in range(len(cameras)):
                            csvLogRows[i]["{}_WEIGHT".format(cameras[c])] = predCamIndexWeights[i,c]
                            
                        for a in range(len(attributes)):
                            csvLogRows[i]["{}_WEIGHT".format(attributes[a])] = predAttributeIndexWeights[i,a]
                        
                        
                        #
                        # for computing accuracies
                        #
                        if prop_outputSpeechClusterIds[i] not in prop_junkSpeechClusterIds:
                            speechClusts_gt.append(prop_outputSpeechClusterIds[i])
                            speechClusts_pred.append(predShkpSpeechClustID[i])
                        
                        
                        if np.sum(prop_outputCameraIndices[i]) > 0:
                            cams_gt.append(np.argmax(prop_outputCameraIndices[i]))
                            cams_pred.append(camPredArgmax)
                        
                        
                        if len(targAttrIndexList) > 0:
                            
                            targAttrIndexList.sort()
                            predAttrIndexList.sort()
                            
                            targAttrIndexStr = "-".join([str(a) for a in targAttrIndexList])
                            predAttrIndexStr = "-".join([str(a) for a in predAttrIndexList])
                            
                            targAttrIndexVec = np.zeros(len(dbFieldnames))
                            predAttrIndexVec = np.zeros(len(dbFieldnames))
                            
                            targAttrIndexVec[targAttrIndexList] = 1
                            predAttrIndexVec[predAttrIndexList] = 1
                            
                            exactAttrs_gt.append(targAttrIndexStr)
                            exactAttrs_pred.append(predAttrIndexStr)
                            
                            setAttrs_gt.append(targAttrIndexVec)
                            setAttrs_pred.append(predAttrIndexVec)
                            
                        
                        locs_gt.append(csvLogRows[i]["TARG_OUTPUT_SHOPKEEPER_LOCATION"])
                        locs_pred.append(csvLogRows[i]["PRED_OUTPUT_SHOPKEEPER_LOCATION"])
                        
                        spatSts_gt.append(csvLogRows[i]["TARG_OUTPUT_SPATIAL_STATE"])
                        spatSts_pred.append(csvLogRows[i]["PRED_OUTPUT_SPATIAL_STATE"])
                        
                        stTargs_gt.append(csvLogRows[i]["TARG_OUTPUT_STATE_TARGET"])
                        stTargs_pred.append(csvLogRows[i]["PRED_OUTPUT_STATE_TARGET"])
                    
                    
                    #
                    # compute accuracies
                    #
                    speechClustCorrAcc = accuracy_score(speechClusts_gt, speechClusts_pred)
                    camCorrAcc = accuracy_score(cams_gt, cams_pred)
                    attrExactMatch = accuracy_score(exactAttrs_gt, exactAttrs_pred)
                    attrJaccardIndex = jaccard_score(np.asarray(setAttrs_gt), np.asarray(setAttrs_pred), average="samples")
                    locCorrAcc = accuracy_score(locs_gt, locs_pred)
                    spatStCorrAcc = accuracy_score(spatSts_gt, spatSts_pred)
                    stTargCorrAcc = accuracy_score(stTargs_gt, stTargs_pred)
                    
                    
                    
                    
                    return csvLogRows, speechClustCorrAcc, camCorrAcc, attrExactMatch, attrJaccardIndex, locCorrAcc, spatStCorrAcc, stTargCorrAcc
                
                
                csvLogRows = copy.deepcopy(interactions)
                
                csvLogRows, trainSpeechClustCorrAcc, trainCamCorrAcc, trainAttrExactMatch, trainAttrJaccardIndex, trainLocCorrAcc, trainSpatStCorrAcc, trainStTargCorrAcc = evaluate_predictions_prop("TRAIN", trainIndices, csvLogRows)
                
                csvLogRows, valSpeechClustCorrAcc, valCamCorrAcc, valAttrExactMatch, valAttrJaccardIndex, valLocCorrAcc, valSpatStCorrAcc, valStTargCorrAcc = evaluate_predictions_prop("VAL", valIndices, csvLogRows)
                
                csvLogRows, testSpeechClustCorrAcc, testCamCorrAcc, testAttrExactMatch, testAttrJaccardIndex, testLocCorrAcc, testSpatStCorrAcc, testStTargCorrAcc = evaluate_predictions_prop("TEST", testIndices, csvLogRows)
                
                
                
                
                
                #
                # save the evaluation results
                #
                with open(foldDir+"/{:}_all_outputs.csv".format(e), "w", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, interactionsFieldnames)
                    writer.writeheader()
                    writer.writerows(csvLogRows)
                
                
                # append to session log   
                with open(foldLogFile, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([e,                 #"Epoch",
                                     
                                     # training
                                     trainCostAve,      #"Train Cost Ave ({})".format(seed), 
                                     trainCostStd,      #"Train Cost SD ({})".format(seed),
                                     trainShopkeeperActionLossAve,
                                     trainShopkeeperActionLossStd,
                                     trainCameraIndexLossAve,
                                     trainCameraIndexLossStd,
                                     trainAttributeIndexLossAve,
                                     trainAttributeIndexLossStd,
                                     trainLocationLossAve,
                                     trainLocationLossStd,
                                     trainSpatialStateLossAve,
                                     trainSpatialStateLossStd,
                                     trainStateTargetLossAve,
                                     trainStateTargetLossStd,
                                     
                                     trainSpeechClustCorrAcc,
                                     trainCamCorrAcc,
                                     trainAttrExactMatch,
                                     trainAttrJaccardIndex,
                                     trainLocCorrAcc,
                                     trainSpatStCorrAcc,
                                     trainStTargCorrAcc,
                                     
                                     # validation
                                     valCostAve,
                                     valCostStd,
                                     valShopkeeperActionLossAve,
                                     valShopkeeperActionLossStd,
                                     valCameraIndexLossAve,
                                     valCameraIndexLossStd,
                                     valAttributeIndexLossAve,
                                     valAttributeIndexLossStd,
                                     valLocationLossAve,
                                     valLocationLossStd,
                                     valSpatialStateLossAve,
                                     valSpatialStateLossStd,
                                     valStateTargetLossAve,
                                     valStateTargetLossStd,
                                     
                                     valSpeechClustCorrAcc,
                                     valCamCorrAcc,
                                     valAttrExactMatch,
                                     valAttrJaccardIndex,
                                     valLocCorrAcc,
                                     valSpatStCorrAcc,
                                     valStTargCorrAcc,
                                     
                                     # testing
                                     testCostAve,
                                     testCostStd,
                                     testShopkeeperActionLossAve,
                                     testShopkeeperActionLossStd,
                                     testCameraIndexLossAve,
                                     testCameraIndexLossStd,
                                     testAttributeIndexLossAve,
                                     testAttributeIndexLossStd,
                                     testLocationLossAve,
                                     testLocationLossStd,
                                     testSpatialStateLossAve,
                                     testSpatialStateLossStd,
                                     testStateTargetLossAve,
                                     testStateTargetLossStd,
                                     
                                     testSpeechClustCorrAcc,
                                     testCamCorrAcc,
                                     testAttrExactMatch,
                                     testAttrJaccardIndex,
                                     testLocCorrAcc,
                                     testSpatStCorrAcc,
                                     testStTargCorrAcc
                                     ])    
                    
        
        print("Epoch", e, flush=True, file=foldTerminalOutputStream)
        
        # training
        print("=====LOSSES AND ACCURACIES=====", flush=True, file=foldTerminalOutputStream)
        tableData = []
        
        tableData.append(["CostAve", trainCostAve, valCostAve, testCostAve])
        tableData.append(["CostStd", trainCostStd, valCostStd, testCostStd])
        tableData.append(["ShopkeeperActionLossAve", trainShopkeeperActionLossAve, valShopkeeperActionLossAve, testShopkeeperActionLossAve])
        tableData.append(["ShopkeeperActionLossStd", trainShopkeeperActionLossStd, valShopkeeperActionLossStd, testShopkeeperActionLossStd])
        tableData.append(["CameraIndexLossAve", trainCameraIndexLossAve, valCameraIndexLossAve, testCameraIndexLossAve])
        tableData.append(["CameraIndexLossStd", trainCameraIndexLossStd, valCameraIndexLossStd, testCameraIndexLossStd])
        tableData.append(["AttributeIndexLossAve", trainAttributeIndexLossAve, valAttributeIndexLossAve, testAttributeIndexLossAve])
        tableData.append(["AttributeIndexLossStd", trainAttributeIndexLossStd, valAttributeIndexLossStd, testAttributeIndexLossStd])
        tableData.append(["LocationLossAve", trainLocationLossAve, valLocationLossAve, testLocationLossAve])
        tableData.append(["LocationLossStd", trainLocationLossStd, valLocationLossStd, testLocationLossStd])
        tableData.append(["SpatialStateLossAve", trainSpatialStateLossAve, valSpatialStateLossAve, testSpatialStateLossAve])
        tableData.append(["SpatialStateLossStd", trainSpatialStateLossStd, valSpatialStateLossStd, testSpatialStateLossStd])
        tableData.append(["StateTargetLossAve", trainStateTargetLossAve, valStateTargetLossAve, testStateTargetLossAve])
        tableData.append(["StateTargetLossStd", trainStateTargetLossStd, valStateTargetLossStd, testStateTargetLossStd])
        
        tableData.append(["SpeechClustCorrAcc", trainSpeechClustCorrAcc, valSpeechClustCorrAcc, testSpeechClustCorrAcc])
        tableData.append(["CamCorrAcc", trainCamCorrAcc, valCamCorrAcc, testCamCorrAcc])
        tableData.append(["AttrExactMatch", trainAttrExactMatch, valAttrExactMatch, testAttrExactMatch])
        tableData.append(["AttrJaccardIndex", trainAttrJaccardIndex, valAttrJaccardIndex, testAttrJaccardIndex])
        tableData.append(["LocCorrAcc", trainLocCorrAcc, valLocCorrAcc, testLocCorrAcc])
        tableData.append(["SpatStCorrAcc", trainSpatStCorrAcc, valSpatStCorrAcc, testSpatStCorrAcc])
        tableData.append(["StTargCorrAcc", trainStTargCorrAcc, valStTargCorrAcc, testStTargCorrAcc])
        
        print(tabulate(tableData, headers=["METRIC", "TRAINING", "VALIDATION", "TESTING"], floatfmt=".3f", tablefmt="grid"), flush=True, file=foldTerminalOutputStream)
                
        print("", flush=True, file=foldTerminalOutputStream)


#
# run here...
#
if not RUN_PARALLEL:
    run_fold(randomSeed=0, foldId=0, gpu=0)

else:
    for gpu in range(8):
        process = Process(target=run_fold, args=[0, gpu, gpu])
        process.start()



