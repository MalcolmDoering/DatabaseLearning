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
import math

import tools



mainDir = tools.create_session_dir("actionPrediction18_dbl")


def main(mainDir, condition, gpuCount):

    #################################################################################################################
    # running params
    #################################################################################################################
    
    DEBUG = False
    RUN_PARALLEL = True
    
    SPEECH_CLUSTER_LOSS_WEIGHTS = True
    
    NUM_GPUS = 8
    
    sessionDir = mainDir + "/" + condition
    tools.create_directory(sessionDir)
    
    # what to run. only one of these should be true at a time
    prop_run = False
    bl1_run = False
    copy_run = False
    coreqa_run = False

    if condition == "proposed":
        prop_run = True
    
    elif condition == "baseline1":
        bl1_run = True
    
    elif condition == "copynet":
        copy_run = True
    
    elif condition == "coreqa":
        coreqa_run = True
    
    
    
    # params that should be the same for all conditions (predictors)
    numDatabases = 11
    numTrainDbs = 9
    numValDbs = 1
    numTestDbs = 1
    
    numInteractionsPerDb = 200
    batchSize = 32
    randomizeTrainingBatches = False
    numEpochs = 500
    evalEvery = 1
    
    dataDirectory = tools.dataDir+"2020-01-08_advancedSimulator9" # handmade databases, customer-driven interactions, deterministic introductions, crowdsourced shopkeeper utts
    inputSequenceVectorDirectory = dataDirectory + "_input_sequence_vectors"
    
    inputSeqLen = 10 
    inputDim = 2226
    
    maxCamLen = 2
    maxAttrLen = 2
    maxValLen = 15
    
    
    # params for proposed
    prop_embeddingSize = 100
    prop_shopkeeperSpeechClusterFilename = tools.modelDir + "20200109_withsymbols shopkeeper-tristm-3wgtkw-9wgtsym-3wgtnum-mc2-sw2-eucldist- speech_clusters.csv"
    
    
    # params for baseline 1
    bl1_embeddingSize = 100
    bl1_shopkeeperSpeechClusterFilename = tools.modelDir + "20200116_nosymbols shopkeeper-tristm-3wgtkw-9wgtnum-mc2-sw3-eucldist- speech_clusters.csv"
    
    
    # params for copynet-based
    copy_embeddingSize = 200
    
    
    # params for coreqa
    coreqa_embeddingSize = 100
    
    
    
    #################################################################################################################
    # global stuff
    #################################################################################################################
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
        prop_silenceSpeechClusterId = prop_shkpUttToSpeechClustId[""]
        print(prop_numSpeechClusters, "shopkeeper speech clusters for PROPOSED", flush=True, file=sessionTerminalOutputStream)
        
        
        # load the output targets
        prop_outputSpeechClusterIds = []
        prop_outputCameraIndices = []
        prop_outputAttributeIndices = []
        
        for i in range(numDatabases):
            iFn = interactionFilenames[i]
            
            # speech clusters
            fn = iFn.split("/")[-1][:-4] + "_output_speech_cluster_ids_withsymbols.npy"
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
        bl1_silenceSpeechClusterId = bl1_shkpUttToSpeechClustId[""]
        print(bl1_numSpeechClusters, "shopkeeper speech clusters for BASELINE 1", flush=True, file=sessionTerminalOutputStream)
        
        
        # load the output targets
        bl1_outputSpeechClusterIds = []
            
        for i in range(numDatabases):
            iFn = interactionFilenames[i]
            
            # speech clusters
            fn = iFn.split("/")[-1][:-4] + "_output_speech_cluster_ids_nosymbols.npy"
            bl1_outputSpeechClusterIds.append(np.load(inputSequenceVectorDirectory+"/"+fn))
        
        bl1_outputSpeechClusterIds = np.concatenate(bl1_outputSpeechClusterIds)
    
    
    #
    # for COPYNET
    #
    if copy_run:
        
        copy_wordToIndex = {}
        copy_indexToWord = {}
        
        with open(inputSequenceVectorDirectory+"/db_and_output_vocab.csv") as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                index = int(row["INDEX"])
                w = row["TOKEN"]
                
                copy_wordToIndex[w] = index
                copy_indexToWord[index] = w
        
        
        copy_outputSpeechSequenceVecs = []
        copy_outputSpeechSequenceLens = []
        copy_databaseSequenceVecs = []
        
        for i in range(numDatabases):
            iFn = interactionFilenames[i]
            
            # load the shopkeeper speech token vector sequences
            fn = iFn.split("/")[-1][:-4] + "_output_speech_vector_sequences.npy"
            copy_outputSpeechSequenceVecs.append(np.load(inputSequenceVectorDirectory+"/"+fn))
            
            fn = iFn.split("/")[-1][:-4] + "_output_speech_vector_sequence_lens.npy"
            copy_outputSpeechSequenceLens.append(np.load(inputSequenceVectorDirectory+"/"+fn))
            
            
            # load the database token vector sequences and indices
            fn = iFn.split("/")[-1][:-4] + "_database_vector_sequences.npy"
            copy_databaseSequenceVecs.append(np.load(inputSequenceVectorDirectory+"/"+fn))
            
        
        temp = []
        for i in range(numDatabases):
            temp.append([copy_databaseSequenceVecs[i]] * len(copy_outputSpeechSequenceVecs[i]))
        copy_databaseSequenceVecs = temp 
        del temp
        
        
        copy_outputSpeechSequenceVecs = np.concatenate(copy_outputSpeechSequenceVecs)
        copy_outputSpeechSequenceLens = np.concatenate(copy_outputSpeechSequenceLens)
        copy_databaseSequenceVecs = np.concatenate(copy_databaseSequenceVecs)
        
        
        copy_vocabSize = len(copy_indexToWord)
        copy_outputSpeechSeqLen = copy_outputSpeechSequenceVecs[0].shape[0]
        copy_dbSeqLen = copy_databaseSequenceVecs[0].shape[2]
        
    
    #
    # for COREQA
    #
    if coreqa_run:
        
        coreqa_wordToIndex = {}
        coreqa_indexToWord = {}
        
        with open(inputSequenceVectorDirectory+"/db_and_output_vocab.csv") as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                index = int(row["INDEX"])
                w = row["TOKEN"]
                
                coreqa_wordToIndex[w] = index
                coreqa_indexToWord[index] = w
        
        
        coreqa_outputSpeechSequenceVecs = []
        coreqa_outputSpeechSequenceLens = []     
        coreqa_databaseCamVecs = []
        coreqa_databaseAttrVecs = []
        coreqa_databaseValVecs = []
        
        for i in range(numDatabases):
            iFn = interactionFilenames[i]
            
            # load the shopkeeper speech token vector sequences
            fn = iFn.split("/")[-1][:-4] + "_output_speech_vector_sequences.npy"
            coreqa_outputSpeechSequenceVecs.append(np.load(inputSequenceVectorDirectory+"/"+fn))
            
            fn = iFn.split("/")[-1][:-4] + "_output_speech_vector_sequence_lens.npy"
            coreqa_outputSpeechSequenceLens.append(np.load(inputSequenceVectorDirectory+"/"+fn))
            
            
            # load the database token vector sequences and indices
            fn = iFn.split("/")[-1][:-4] + "_database_fact_cams.npy"
            coreqa_databaseCamVecs.append(np.load(inputSequenceVectorDirectory+"/"+fn))
            
            fn = iFn.split("/")[-1][:-4] + "_database_fact_attrs.npy"
            coreqa_databaseAttrVecs.append(np.load(inputSequenceVectorDirectory+"/"+fn))
            
            fn = iFn.split("/")[-1][:-4] + "_database_fact_vals.npy"
            coreqa_databaseValVecs.append(np.load(inputSequenceVectorDirectory+"/"+fn))
            
        
        # duplicate DBs so there is one for each input
        temp = []
        for i in range(numDatabases):
            temp.append([coreqa_databaseCamVecs[i]] * len(coreqa_outputSpeechSequenceVecs[i]))
        coreqa_databaseCamVecs = temp 
        del temp
        
        temp = []
        for i in range(numDatabases):
            temp.append([coreqa_databaseAttrVecs[i]] * len(coreqa_outputSpeechSequenceVecs[i]))
        coreqa_databaseAttrVecs = temp 
        del temp
        
        temp = []
        for i in range(numDatabases):
            temp.append([coreqa_databaseValVecs[i]] * len(coreqa_outputSpeechSequenceVecs[i]))
        coreqa_databaseValVecs = temp 
        del temp
        
        coreqa_outputSpeechSequenceVecs = np.concatenate(coreqa_outputSpeechSequenceVecs)
        coreqa_outputSpeechSequenceLens = np.concatenate(coreqa_outputSpeechSequenceLens)
        
        coreqa_databaseCamVecs = np.concatenate(coreqa_databaseCamVecs)
        coreqa_databaseAttrVecs = np.concatenate(coreqa_databaseAttrVecs)
        coreqa_databaseValVecs = np.concatenate(coreqa_databaseValVecs)
        
        coreqa_vocabSize = len(coreqa_indexToWord)
        coreqa_outputSpeechSeqLen = coreqa_outputSpeechSequenceVecs[0].shape[0]
        
        
    
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
    
    
    # for debugging...
    #trainSplits[0] = trainSplits[0][:batchSize]
    #valSplits = valSplits[0][:batchSize]
    #testSplits = testSplits[0][:batchSize]
    
    
    #################################################################################################################
    # parralell process each fold
    #################################################################################################################
    
    def run_fold(randomSeed, foldId, gpu):
        
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
                                 
                                 "Training Loss Ave ({})".format(foldIdentifier), 
                                 "Training Loss SD ({})".format(foldIdentifier),
                                 "Training Shopkeeper Speech Cluster Loss Ave ({})".format(foldIdentifier),
                                 "Training Shopkeeper Speech Cluster Loss SD ({})".format(foldIdentifier),
                                 "Training Camera Index Loss Ave ({})".format(foldIdentifier),
                                 "Training Camera Index Loss SD ({})".format(foldIdentifier),
                                 "Training Attribute Index Loss Ave ({})".format(foldIdentifier),
                                 "Training Attribute Index Loss SD ({})".format(foldIdentifier),
                                 "Training Location Loss Ave ({})".format(foldIdentifier),
                                 "Training Location Loss SD ({})".format(foldIdentifier),
                                 "Training Spatial State Loss Ave ({})".format(foldIdentifier),
                                 "Training Spatial State Loss SD ({})".format(foldIdentifier),
                                 "Training State Target Loss Ave ({})".format(foldIdentifier),
                                 "Training State Target Loss SD ({})".format(foldIdentifier),
                                 "Training Speech Cluster Correct ({})".format(foldIdentifier), 
                                 "Training Camera Index Correct ({})".format(foldIdentifier),
                                 "Training Attribute Index Exact Match ({})".format(foldIdentifier),
                                 "Training Attribute Index Jaccard Index ({})".format(foldIdentifier),
                                 "Training Location Correct ({})".format(foldIdentifier),
                                 "Training Spatial State Correct ({})".format(foldIdentifier),
                                 "Training State Target Correct ({})".format(foldIdentifier),
                                 "Training Silence Correct ({})".format(foldIdentifier),
                                 
                                 "Validation Loss Ave ({})".format(foldIdentifier), 
                                 "Validation Loss SD ({})".format(foldIdentifier),
                                 "Validation Shopkeeper Speech Cluster Loss Ave ({})".format(foldIdentifier),
                                 "Validation Shopkeeper Speech Cluster Loss SD ({})".format(foldIdentifier),
                                 "Validation Camera Index Loss Ave ({})".format(foldIdentifier),
                                 "Validation Camera Index Loss SD ({})".format(foldIdentifier),
                                 "Validation Attribute Index Loss Ave ({})".format(foldIdentifier),
                                 "Validation Attribute Index Loss SD ({})".format(foldIdentifier),
                                 "Validation Location Loss Ave ({})".format(foldIdentifier),
                                 "Validation Location Loss SD ({})".format(foldIdentifier),
                                 "Validation Spatial State Loss Ave ({})".format(foldIdentifier),
                                 "Validation Spatial State Loss SD ({})".format(foldIdentifier),
                                 "Validation State Target Loss Ave ({})".format(foldIdentifier),
                                 "Validation State Target Loss SD ({})".format(foldIdentifier),
                                 "Validation Speech Cluster Correct ({})".format(foldIdentifier), 
                                 "Validation Camera Index Correct ({})".format(foldIdentifier),
                                 "Validation Attribute Index Exact Match ({})".format(foldIdentifier),
                                 "Validation Attribute Index Jaccard Index ({})".format(foldIdentifier),
                                 "Validation Location Correct ({})".format(foldIdentifier),
                                 "Validation Spatial State Correct ({})".format(foldIdentifier),
                                 "Validation State Target Correct ({})".format(foldIdentifier),
                                 "Validation Silence Correct ({})".format(foldIdentifier),
                                 
                                 "Testing Loss Ave ({})".format(foldIdentifier), 
                                 "Testing Loss SD ({})".format(foldIdentifier),
                                 "Testing Shopkeeper Speech Cluster Loss Ave ({})".format(foldIdentifier),
                                 "Testing Shopkeeper Speech Cluster Loss SD ({})".format(foldIdentifier),
                                 "Testing Camera Index Loss Ave ({})".format(foldIdentifier),
                                 "Testing Camera Index Loss SD ({})".format(foldIdentifier),
                                 "Testing Attribute Index Loss Ave ({})".format(foldIdentifier),
                                 "Testing Attribute Index Loss SD ({})".format(foldIdentifier),
                                 "Testing Location Loss Ave ({})".format(foldIdentifier),
                                 "Testing Location Loss SD ({})".format(foldIdentifier),
                                 "Testing Spatial State Loss Ave ({})".format(foldIdentifier),
                                 "Testing Spatial State Loss SD ({})".format(foldIdentifier),
                                 "Testing State Target Loss Ave ({})".format(foldIdentifier),
                                 "Testing State Target Loss SD ({})".format(foldIdentifier),
                                 "Testing Speech Cluster Correct ({})".format(foldIdentifier), 
                                 "Testing Camera Index Correct ({})".format(foldIdentifier),
                                 "Testing Attribute Index Exact Match ({})".format(foldIdentifier),
                                 "Testing Attribute Index Jaccard Index ({})".format(foldIdentifier),
                                 "Testing Location Correct ({})".format(foldIdentifier),
                                 "Testing Spatial State Correct ({})".format(foldIdentifier),
                                 "Testing State Target Correct ({})".format(foldIdentifier),
                                 "Testing Silence Correct ({})".format(foldIdentifier),
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
                interactionsFieldnames.append("PRED_{}_WEIGHT".format(c))
            for a in attributes:
                interactionsFieldnames.append("PRED_{}_WEIGHT".format(a))
        
        
        
        if bl1_run:
            with open(foldLogFile, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Epoch",
                                 
                                 "Training Loss Ave ({})".format(foldIdentifier), 
                                 "Training Loss SD ({})".format(foldIdentifier),
                                 "Training Shopkeeper Speech Cluster Loss Ave ({})".format(foldIdentifier),
                                 "Training Shopkeeper Speech Cluster Loss SD ({})".format(foldIdentifier),
                                 "Training Location Loss Ave ({})".format(foldIdentifier),
                                 "Training Location Loss SD ({})".format(foldIdentifier),
                                 "Training Spatial State Loss Ave ({})".format(foldIdentifier),
                                 "Training Spatial State Loss SD ({})".format(foldIdentifier),
                                 "Training State Target Loss Ave ({})".format(foldIdentifier),
                                 "Training State Target Loss SD ({})".format(foldIdentifier),
                                 "Training Speech Cluster Correct ({})".format(foldIdentifier), 
                                 "Training Location Correct ({})".format(foldIdentifier),
                                 "Training Spatial State Correct ({})".format(foldIdentifier),
                                 "Training State Target Correct ({})".format(foldIdentifier),
                                 "Training Silence Correct ({})".format(foldIdentifier),
                                 
                                 "Validation Loss Ave ({})".format(foldIdentifier), 
                                 "Validation Loss SD ({})".format(foldIdentifier),
                                 "Validation Shopkeeper Speech Cluster Loss Ave ({})".format(foldIdentifier),
                                 "Validation Shopkeeper Speech Cluster Loss SD ({})".format(foldIdentifier),
                                 "Validation Location Loss Ave ({})".format(foldIdentifier),
                                 "Validation Location Loss SD ({})".format(foldIdentifier),
                                 "Validation Spatial State Loss Ave ({})".format(foldIdentifier),
                                 "Validation Spatial State Loss SD ({})".format(foldIdentifier),
                                 "Validation State Target Loss Ave ({})".format(foldIdentifier),
                                 "Validation State Target Loss SD ({})".format(foldIdentifier),
                                 "Validation Speech Cluster Correct ({})".format(foldIdentifier), 
                                 "Validation Location Correct ({})".format(foldIdentifier),
                                 "Validation Spatial State Correct ({})".format(foldIdentifier),
                                 "Validation State Target Correct ({})".format(foldIdentifier),
                                 "Validation Silence Correct ({})".format(foldIdentifier),
                                 
                                 "Testing Loss Ave ({})".format(foldIdentifier), 
                                 "Testing Loss SD ({})".format(foldIdentifier),
                                 "Testing Shopkeeper Speech Cluster Loss Ave ({})".format(foldIdentifier),
                                 "Testing Shopkeeper Speech Cluster Loss SD ({})".format(foldIdentifier),
                                 "Testing Location Loss Ave ({})".format(foldIdentifier),
                                 "Testing Location Loss SD ({})".format(foldIdentifier),
                                 "Testing Spatial State Loss Ave ({})".format(foldIdentifier),
                                 "Testing Spatial State Loss SD ({})".format(foldIdentifier),
                                 "Testing State Target Loss Ave ({})".format(foldIdentifier),
                                 "Testing State Target Loss SD ({})".format(foldIdentifier),
                                 "Testing Speech Cluster Correct ({})".format(foldIdentifier), 
                                 "Testing Location Correct ({})".format(foldIdentifier),
                                 "Testing Spatial State Correct ({})".format(foldIdentifier),
                                 "Testing State Target Correct ({})".format(foldIdentifier),
                                 "Testing Silence Correct ({})".format(foldIdentifier),
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
                                      "TARG_OUTPUT_SHOPKEEPER_LOCATION", 
                                      "TARG_OUTPUT_SPATIAL_STATE", 
                                      "TARG_OUTPUT_STATE_TARGET",
                                      
                                      "PRED_OUTPUT_SHOPKEEPER_SPEECH_CLUSTER_ID",
                                      "PRED_SHOPKEEPER_SPEECH",
                                      "PRED_OUTPUT_SHOPKEEPER_LOCATION",
                                      "PRED_OUTPUT_SPATIAL_STATE",
                                      "PRED_OUTPUT_STATE_TARGET"]
            
        if copy_run:
            with open(foldLogFile, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Epoch",
                                 
                                 "Training Loss Ave ({})".format(foldIdentifier), 
                                 "Training Loss SD ({})".format(foldIdentifier),
                                 "Training Shopkeeper Speech Sequence Loss Ave ({})".format(foldIdentifier),
                                 "Training Shopkeeper Speech Sequence Loss SD ({})".format(foldIdentifier),
                                 "Training Location Loss Ave ({})".format(foldIdentifier),
                                 "Training Location Loss SD ({})".format(foldIdentifier),
                                 "Training Spatial State Loss Ave ({})".format(foldIdentifier),
                                 "Training Spatial State Loss SD ({})".format(foldIdentifier),
                                 "Training State Target Loss Ave ({})".format(foldIdentifier),
                                 "Training State Target Loss SD ({})".format(foldIdentifier),
                                 
                                 "Training Speech Cluster Correct ({})".format(foldIdentifier), 
                                 "Training Location Correct ({})".format(foldIdentifier),
                                 "Training Spatial State Correct ({})".format(foldIdentifier),
                                 "Training State Target Correct ({})".format(foldIdentifier),
                                 "Training Camera Index Correct ({})".format(foldIdentifier),
                                 "Training Attribute Index Correct ({})".format(foldIdentifier),
                                 
                                 
                                 "Validation Loss Ave ({})".format(foldIdentifier), 
                                 "Validation Loss SD ({})".format(foldIdentifier),
                                 "Validation Shopkeeper Speech Sequence Loss Ave ({})".format(foldIdentifier),
                                 "Validation Shopkeeper Speech Sequence Loss SD ({})".format(foldIdentifier),
                                 "Validation Location Loss Ave ({})".format(foldIdentifier),
                                 "Validation Location Loss SD ({})".format(foldIdentifier),
                                 "Validation Spatial State Loss Ave ({})".format(foldIdentifier),
                                 "Validation Spatial State Loss SD ({})".format(foldIdentifier),
                                 "Validation State Target Loss Ave ({})".format(foldIdentifier),
                                 "Validation State Target Loss SD ({})".format(foldIdentifier),
                                 
                                 "Validation Speech Cluster Correct ({})".format(foldIdentifier), 
                                 "Validation Location Correct ({})".format(foldIdentifier),
                                 "Validation Spatial State Correct ({})".format(foldIdentifier),
                                 "Validation State Target Correct ({})".format(foldIdentifier),
                                 "Validation Camera Index Correct ({})".format(foldIdentifier),
                                 "Validation Attribute Index Correct ({})".format(foldIdentifier),
                                 
                                 
                                 "Testing Loss Ave ({})".format(foldIdentifier), 
                                 "Testing Loss SD ({})".format(foldIdentifier),
                                 "Testing Shopkeeper Speech Sequence Loss Ave ({})".format(foldIdentifier),
                                 "Testing Shopkeeper Speech Sequence Loss SD ({})".format(foldIdentifier),
                                 "Testing Location Loss Ave ({})".format(foldIdentifier),
                                 "Testing Location Loss SD ({})".format(foldIdentifier),
                                 "Testing Spatial State Loss Ave ({})".format(foldIdentifier),
                                 "Testing Spatial State Loss SD ({})".format(foldIdentifier),
                                 "Testing State Target Loss Ave ({})".format(foldIdentifier),
                                 "Testing State Target Loss SD ({})".format(foldIdentifier),
                                 
                                 "Testing Speech Cluster Correct ({})".format(foldIdentifier), 
                                 "Testing Location Correct ({})".format(foldIdentifier),
                                 "Testing Spatial State Correct ({})".format(foldIdentifier),
                                 "Testing State Target Correct ({})".format(foldIdentifier),
                                 "Testing Camera Index Correct ({})".format(foldIdentifier),
                                 "Testing Attribute Index Correct ({})".format(foldIdentifier),
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
                                      
                                      "TARG_SHOPKEEPER_SPEECH_SEQUENCE",
                                      "TARG_SHOPKEEPER_SPEECH",
                                      "TARG_OUTPUT_SHOPKEEPER_LOCATION", 
                                      "TARG_OUTPUT_SPATIAL_STATE", 
                                      "TARG_OUTPUT_STATE_TARGET",
                                      
                                      "PRED_SHOPKEEPER_SPEECH_SEQUENCE",
                                      "PRED_SHOPKEEPER_SPEECH",
                                      "PRED_OUTPUT_SHOPKEEPER_LOCATION",
                                      "PRED_OUTPUT_SPATIAL_STATE",
                                      "PRED_OUTPUT_STATE_TARGET",
                                      "PRED_CAMERA_INDEX_MAX",
                                      "PRED_ATTRIBUTE_INDEX_MAX"
                                      ]
            
            for c in cameras:
                interactionsFieldnames.append("PRED_{}_WEIGHT".format(c))
            for a in attributes:
                interactionsFieldnames.append("PRED_{}_WEIGHT".format(a))
        
        
        if coreqa_run:
            with open(foldLogFile, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Epoch",
                                 
                                 "Training Loss Ave ({})".format(foldIdentifier), 
                                 "Training Loss SD ({})".format(foldIdentifier),
                                 "Training Shopkeeper Speech Sequence Loss Ave ({})".format(foldIdentifier),
                                 "Training Shopkeeper Speech Sequence Loss SD ({})".format(foldIdentifier),
                                 "Training Location Loss Ave ({})".format(foldIdentifier),
                                 "Training Location Loss SD ({})".format(foldIdentifier),
                                 "Training Spatial State Loss Ave ({})".format(foldIdentifier),
                                 "Training Spatial State Loss SD ({})".format(foldIdentifier),
                                 "Training State Target Loss Ave ({})".format(foldIdentifier),
                                 "Training State Target Loss SD ({})".format(foldIdentifier),
                                 
                                 "Training Speech Cluster Correct ({})".format(foldIdentifier), 
                                 "Training Location Correct ({})".format(foldIdentifier),
                                 "Training Spatial State Correct ({})".format(foldIdentifier),
                                 "Training State Target Correct ({})".format(foldIdentifier),
                                 "Training Camera Index Correct ({})".format(foldIdentifier),
                                 "Training Attribute Index Correct ({})".format(foldIdentifier),
                                 
                                 
                                 "Validation Loss Ave ({})".format(foldIdentifier), 
                                 "Validation Loss SD ({})".format(foldIdentifier),
                                 "Validation Shopkeeper Speech Sequence Loss Ave ({})".format(foldIdentifier),
                                 "Validation Shopkeeper Speech Sequence Loss SD ({})".format(foldIdentifier),
                                 "Validation Location Loss Ave ({})".format(foldIdentifier),
                                 "Validation Location Loss SD ({})".format(foldIdentifier),
                                 "Validation Spatial State Loss Ave ({})".format(foldIdentifier),
                                 "Validation Spatial State Loss SD ({})".format(foldIdentifier),
                                 "Validation State Target Loss Ave ({})".format(foldIdentifier),
                                 "Validation State Target Loss SD ({})".format(foldIdentifier),
                                 
                                 "Validation Speech Cluster Correct ({})".format(foldIdentifier), 
                                 "Validation Location Correct ({})".format(foldIdentifier),
                                 "Validation Spatial State Correct ({})".format(foldIdentifier),
                                 "Validation State Target Correct ({})".format(foldIdentifier),
                                 "Validation Camera Index Correct ({})".format(foldIdentifier),
                                 "Validation Attribute Index Correct ({})".format(foldIdentifier),
                                 
                                 
                                 "Testing Loss Ave ({})".format(foldIdentifier), 
                                 "Testing Loss SD ({})".format(foldIdentifier),
                                 "Testing Shopkeeper Speech Sequence Loss Ave ({})".format(foldIdentifier),
                                 "Testing Shopkeeper Speech Sequence Loss SD ({})".format(foldIdentifier),
                                 "Testing Location Loss Ave ({})".format(foldIdentifier),
                                 "Testing Location Loss SD ({})".format(foldIdentifier),
                                 "Testing Spatial State Loss Ave ({})".format(foldIdentifier),
                                 "Testing Spatial State Loss SD ({})".format(foldIdentifier),
                                 "Testing State Target Loss Ave ({})".format(foldIdentifier),
                                 "Testing State Target Loss SD ({})".format(foldIdentifier),
                                 
                                 "Testing Speech Cluster Correct ({})".format(foldIdentifier), 
                                 "Testing Location Correct ({})".format(foldIdentifier),
                                 "Testing Spatial State Correct ({})".format(foldIdentifier),
                                 "Testing State Target Correct ({})".format(foldIdentifier),
                                 "Testing Camera Index Correct ({})".format(foldIdentifier),
                                 "Testing Attribute Index Correct ({})".format(foldIdentifier),
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
                                      
                                      "TARG_SHOPKEEPER_SPEECH_SEQUENCE",
                                      "TARG_SHOPKEEPER_SPEECH",
                                      "TARG_OUTPUT_SHOPKEEPER_LOCATION", 
                                      "TARG_OUTPUT_SPATIAL_STATE", 
                                      "TARG_OUTPUT_STATE_TARGET",
                                      
                                      "PRED_SHOPKEEPER_SPEECH_SEQUENCE",
                                      "PRED_SHOPKEEPER_SPEECH",
                                      "PRED_OUTPUT_SHOPKEEPER_LOCATION",
                                      "PRED_OUTPUT_SPATIAL_STATE",
                                      "PRED_OUTPUT_STATE_TARGET",
                                      
                                      "PRED_DB_FACT_MAX",
                                      "PRED_DB_CAMERA_MAX",
                                      "PRED_DB_ATTRIBUTE_MAX",
                                      "PRED_DB_VALUE_MAX"
                                      ]

            for i in range(len(cameras)*len(attributes)):  
                interactionsFieldnames.append("PRED_DB_FACT_{}_WEIGHT".format(i))
        
        
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
            if SPEECH_CLUSTER_LOSS_WEIGHTS:
                # count number of occurrences of each speech cluster in the training dataset
                prop_speechClustCounts = {}
                
                for i in trainIndices:
                    speechClustId = prop_outputSpeechClusterIds[i]
                    
                    if speechClustId not in prop_speechClustCounts:
                        prop_speechClustCounts[speechClustId] = 0
                    prop_speechClustCounts[speechClustId] += 1
                
                # remove junk cluster counts
                for speechClustId in prop_junkSpeechClusterIds:
                    del prop_speechClustCounts[speechClustId]
                
                numSamples = sum(prop_speechClustCounts.values())
                
                
                # compute weights
                prop_speechClustWeights = [None] * prop_numSpeechClusters
                
                for clustId in prop_speechClustIdToShkpUtts:

                    if clustId in prop_speechClustCounts:
                        # as in scikit learn - The “balanced” heuristic is inspired by Logistic Regression in Rare Events Data, King, Zen, 2001.
                        prop_speechClustWeights[clustId] = numSamples / ((prop_numSpeechClusters-len(prop_junkSpeechClusterIds)) * prop_speechClustCounts[clustId])
                    
                    else:       
                        # sometimes a cluster won't appear in the training set, so give it a weight of 1
                        prop_speechClustWeights[clustId] = 1.0
                
                
                # don't train on junk speech clusters
                for clustId in prop_junkSpeechClusterIds:
                    prop_speechClustWeights[clustId] = 0
                
                
                if None in prop_speechClustWeights:
                    print("WARNING: missing training weight for PROPOSED shopkeeper speech cluster!", flush=True, file=foldTerminalOutputStream)
            
            else:
                prop_speechClustWeights = []
                for clustId in range(prop_numSpeechClusters):
                    prop_speechClustWeights.append(1.0)
            
            
            for clustId in prop_junkSpeechClusterIds:
                prop_speechClustWeights[clustId] = 0.0
            
            
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
            # output masks for loss (not used)
            #
            bl1_outputMasks = np.ones(totalSamples)
            
            #
            # for speech clusters
            #
            if SPEECH_CLUSTER_LOSS_WEIGHTS:
                # count number of occurrences of each speech cluster in the training dataset
                bl1_speechClustCounts = {}
                
                for i in trainIndices:
                    speechClustId = bl1_outputSpeechClusterIds[i]
                    
                    if speechClustId not in bl1_speechClustCounts:
                        bl1_speechClustCounts[speechClustId] = 0
                    bl1_speechClustCounts[speechClustId] += 1
                
                # remove junk cluster counts
                for speechClustId in bl1_junkSpeechClusterIds:
                    del bl1_speechClustCounts[speechClustId]
                
                numSamples = sum(bl1_speechClustCounts.values())
                
                
                # compute weights
                bl1_speechClustWeights = [None] * bl1_numSpeechClusters
                
                for clustId in bl1_speechClustIdToShkpUtts:
                    # as in scikit learn - The “balanced” heuristic is inspired by Logistic Regression in Rare Events Data, King, Zen, 2001.
                    if clustId in bl1_speechClustCounts:
                        bl1_speechClustWeights[clustId] = numSamples / ((bl1_numSpeechClusters-len(bl1_junkSpeechClusterIds)) * bl1_speechClustCounts[clustId])
                    else:
                        # sometimes a cluster won't appear in the training set, so give it a weight of 1
                        bl1_speechClustWeights[clustId] = 1.0
                        
                if None in bl1_speechClustWeights:
                    print("WARNING: missing training weight for BASELINE 1 shopkeeper speech cluster!", flush=True, file=foldTerminalOutputStream)
            
            else:
                bl1_speechClustWeights = []
                for clustId in range(bl1_numSpeechClusters):
                    bl1_speechClustWeights.append(1.0)
            
            
            for clustId in bl1_junkSpeechClusterIds:
                bl1_speechClustWeights[clustId] = 0.0
        
        
        
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
                                                    embeddingSize=prop_embeddingSize,
                                                    seed=randomSeed,
                                                    speechClusterWeights=prop_speechClustWeights,
                                                    attributeIndexWeights=[prop_attributeIndexWeights0, prop_attributeIndexWeights1]
                                                    )
        
        if bl1_run:
            learner = learning4.Baseline1(inputDim=inputDim, 
                                                    inputSeqLen=inputSeqLen, 
                                                    numOutputClasses=bl1_numSpeechClusters,
                                                    numLocations = len(locations), # cam 1, 2, 3, service counter
                                                    numSpatialStates = len(spatialStates), # f2f, preesnt x, waiting
                                                    numStateTargets = len(stateTargets), # cam 1, 2, 3, NONE
                                                    batchSize=batchSize,
                                                    embeddingSize=bl1_embeddingSize,
                                                    seed=randomSeed,
                                                    speechClusterWeights=bl1_speechClustWeights
                                                    )
        
        if copy_run:
            learner = learning4.CopynetBased(inputDim=inputDim, 
                                             inputSeqLen=inputSeqLen, 
                                             vocabSize=copy_vocabSize,
                                             outputSeqLen=copy_outputSpeechSeqLen,
                                             dbSeqLen=copy_dbSeqLen,
                                             numCameras=len(cameras),
                                             numAttributes=len(dbFieldnames),
                                             numLocations = len(locations), # cam 1, 2, 3, service counter
                                             numSpatialStates = len(spatialStates), # f2f, preesnt x, waiting
                                             numStateTargets = len(stateTargets), # cam 1, 2, 3, NONE
                                             batchSize=batchSize,
                                             embeddingSize=copy_embeddingSize,
                                             seed=randomSeed,
                                             wordToIndex=copy_wordToIndex)
        
        if coreqa_run:
            learner = learning4.CoreqaBased(inputDim=inputDim, 
                                             inputSeqLen=inputSeqLen, 
                                             vocabSize=coreqa_vocabSize,
                                             outputSeqLen=coreqa_outputSpeechSeqLen,
                                             dbCamLen=maxCamLen,
                                             dbAttrLen=maxAttrLen,
                                             dbValLen=maxValLen,
                                             numDbFacts = len(cameras) * len(dbFieldnames),
                                             numLocations = len(locations), # cam 1, 2, 3, service counter
                                             numSpatialStates = len(spatialStates), # f2f, preesnt x, waiting
                                             numStateTargets = len(stateTargets), # cam 1, 2, 3, NONE
                                             batchSize=batchSize,
                                             embeddingSize=coreqa_embeddingSize,
                                             seed=randomSeed,
                                             wordToIndex=coreqa_wordToIndex)
        
        
        
        #################################################################################################################
        # train and test...
        #################################################################################################################
        print("training and testing...", flush=True, file=foldTerminalOutputStream)
        
        for e in range(numEpochs+1):
            startTime = time.time()
            
            if prop_run:
                
                #################################################################################################################
                # BEGIN PROPOSED RUN!
                #################################################################################################################
                
                if e != 0:
                    # train
                    learner.train(
                        inputSequenceVectors[trainIndices],
                        prop_outputSpeechClusterIds[trainIndices],
                        prop_outputCameraIndices[trainIndices],
                        prop_outputAttributeIndices[trainIndices],
                        outputShopkeeperLocations[trainIndices],
                        outputSpatialStates[trainIndices],
                        outputStateTargets[trainIndices],                                           
                        prop_outputMasks[trainIndices],
                        databaseConentLengthsForInput[trainIndices])
                
                
                # evaluate
                if e % evalEvery == 0 or e == numEpochs:
                    
                    # training loss
                    trainCost, trainShopkeeperSpeechClusterLoss, trainCameraIndexLoss, trainAttributeIndexLoss, trainLocationLoss, trainSpatialStateLoss, trainStateTargetLoss = learner.get_loss(
                        inputSequenceVectors[trainIndices],
                        prop_outputSpeechClusterIds[trainIndices],
                        prop_outputCameraIndices[trainIndices],
                        prop_outputAttributeIndices[trainIndices],
                        outputShopkeeperLocations[trainIndices],
                        outputSpatialStates[trainIndices],
                        outputStateTargets[trainIndices],                                           
                        prop_outputMasks[trainIndices],
                        databaseConentLengthsForInput[trainIndices])
                    
                    # validation loss
                    valCost, valShopkeeperSpeechClusterLoss, valCameraIndexLoss, valAttributeIndexLoss, valLocationLoss, valSpatialStateLoss, valStateTargetLoss = learner.get_loss(
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
                    testCost, testShopkeeperSpeechClusterLoss, testCameraIndexLoss, testAttributeIndexLoss, testLocationLoss, testSpatialStateLoss, testStateTargetLoss = learner.get_loss(
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
                    trainShopkeeperSpeechClusterLossAve = np.mean(trainShopkeeperSpeechClusterLoss)
                    trainCameraIndexLossAve = np.mean(trainCameraIndexLoss)
                    trainAttributeIndexLossAve = np.mean(trainAttributeIndexLoss)
                    trainLocationLossAve = np.mean(trainLocationLoss)
                    trainSpatialStateLossAve = np.mean(trainSpatialStateLoss)
                    trainStateTargetLossAve = np.mean(trainStateTargetLoss)
                    
                    trainCostStd = np.std(trainCost)
                    trainShopkeeperSpeechClusterLossStd = np.std(trainShopkeeperSpeechClusterLoss)
                    trainCameraIndexLossStd = np.std(trainCameraIndexLoss)
                    trainAttributeIndexLossStd = np.std(trainAttributeIndexLoss)
                    trainLocationLossStd = np.std(trainLocationLoss)
                    trainSpatialStateLossStd = np.std(trainSpatialStateLoss)
                    trainStateTargetLossStd = np.std(trainStateTargetLoss)
                    
                    # validation
                    valCostAve = np.mean(valCost)
                    valShopkeeperSpeechClusterLossAve = np.mean(valShopkeeperSpeechClusterLoss)
                    valCameraIndexLossAve = np.mean(valCameraIndexLoss)
                    valAttributeIndexLossAve = np.mean(valAttributeIndexLoss)
                    valLocationLossAve = np.mean(valLocationLoss)
                    valSpatialStateLossAve = np.mean(valSpatialStateLoss)
                    valStateTargetLossAve = np.mean(valStateTargetLoss)
                    
                    valCostStd = np.std(valCost)
                    valShopkeeperSpeechClusterLossStd = np.std(valShopkeeperSpeechClusterLoss)
                    valCameraIndexLossStd = np.std(valCameraIndexLoss)
                    valAttributeIndexLossStd = np.std(valAttributeIndexLoss)
                    valLocationLossStd = np.std(valLocationLoss)
                    valSpatialStateLossStd = np.std(valSpatialStateLoss)
                    valStateTargetLossStd = np.std(valStateTargetLoss)
                    
                    # test
                    testCostAve = np.mean(testCost)
                    testShopkeeperSpeechClusterLossAve = np.mean(testShopkeeperSpeechClusterLoss)
                    testCameraIndexLossAve = np.mean(testCameraIndexLoss)
                    testAttributeIndexLossAve = np.mean(testAttributeIndexLoss)
                    testLocationLossAve = np.mean(testLocationLoss)
                    testSpatialStateLossAve = np.mean(testSpatialStateLoss)
                    testStateTargetLossAve = np.mean(testStateTargetLoss)
                    
                    testCostStd = np.std(testCost)
                    testShopkeeperSpeechClusterLossStd = np.std(testShopkeeperSpeechClusterLoss)
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
                        
                        silence_gt = []
                        silence_pred = []
                        
                        
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
                                outShkpSpeechTemplate = prop_shkpSpeechClustIdToRepUtt[predShkpSpeechClustID[i]]
                            except Exception as err:
                                traceback.print_tb(err.__traceback__)
                                print(err, flush=True, file=foldTerminalOutputStream)
                                
                                outShkpSpeechTemplate = "ERROR: No representative utterance found for cluster {}.".format(predShkpSpeechClustID[i])
                            
                            
                            # find any DB contents symbols in the speech template
                            outShkpSpeech = outShkpSpeechTemplate
                            
                            dbId = interactions[i]["DATABASE_ID"]
                            cameraInfo = databases[int(dbId)][camPredArgmax]
                            
                            for j in range(len(dbFieldnames)):
                                attr = dbFieldnames[j]
                                symbol = "<{}>".format(attr.lower())
                                
                                while symbol in outShkpSpeech:
                                    outShkpSpeech = outShkpSpeech.replace(symbol, cameraInfo[attr])
                                    
                                    outDbIndices.append((camPredArgmax, j))
                                    outDbContents.append(cameraInfo[attr])
                                
                            
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
                                csvLogRows[i]["PRED_{}_WEIGHT".format(cameras[c])] = predCamIndexWeights[i,c]
                                
                            for a in range(len(attributes)):
                                csvLogRows[i]["PRED_{}_WEIGHT".format(attributes[a])] = predAttributeIndexWeights[i,a]
                            
                            
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
                            
                            
                            if ((csvLogRows[i]["TARG_SHOPKEEPER_SPEECH_CLUSTER_ID"] == prop_silenceSpeechClusterId) or (predShkpSpeechClustID[i] == prop_silenceSpeechClusterId)):
                                silence_gt.append(csvLogRows[i]["TARG_SHOPKEEPER_SPEECH_CLUSTER_ID"])
                                silence_pred.append(predShkpSpeechClustID[i])
                                
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
                        silenceCorrAcc = accuracy_score(silence_gt, silence_pred)
                        
                        return csvLogRows, speechClustCorrAcc, camCorrAcc, attrExactMatch, attrJaccardIndex, locCorrAcc, spatStCorrAcc, stTargCorrAcc, silenceCorrAcc
                    
                    
                    csvLogRows = copy.deepcopy(interactions)
                    
                    csvLogRows, trainSpeechClustCorrAcc, trainCamCorrAcc, trainAttrExactMatch, trainAttrJaccardIndex, trainLocCorrAcc, trainSpatStCorrAcc, trainStTargCorrAcc, trainSilenceCorrAcc = evaluate_predictions_prop("TRAIN", trainIndices, csvLogRows)
                    
                    csvLogRows, valSpeechClustCorrAcc, valCamCorrAcc, valAttrExactMatch, valAttrJaccardIndex, valLocCorrAcc, valSpatStCorrAcc, valStTargCorrAcc, valSilenceCorrAcc = evaluate_predictions_prop("VAL", valIndices, csvLogRows)
                    
                    csvLogRows, testSpeechClustCorrAcc, testCamCorrAcc, testAttrExactMatch, testAttrJaccardIndex, testLocCorrAcc, testSpatStCorrAcc, testStTargCorrAcc, testSilenceCorrAcc = evaluate_predictions_prop("TEST", testIndices, csvLogRows)
                    
                    
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
                        writer.writerow([e,
                                         
                                         # training
                                         trainCostAve,
                                         trainCostStd,
                                         trainShopkeeperSpeechClusterLossAve,
                                         trainShopkeeperSpeechClusterLossStd,
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
                                         trainSilenceCorrAcc,
                                         
                                         # validation
                                         valCostAve,
                                         valCostStd,
                                         valShopkeeperSpeechClusterLossAve,
                                         valShopkeeperSpeechClusterLossStd,
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
                                         valSilenceCorrAcc,
                                         
                                         # testing
                                         testCostAve,
                                         testCostStd,
                                         testShopkeeperSpeechClusterLossAve,
                                         testShopkeeperSpeechClusterLossStd,
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
                                         testStTargCorrAcc,
                                         testSilenceCorrAcc
                                         ])    
                
                    
                    # training
                    print("===== {} EPOCH {} LOSSES AND ACCURACIES=====".format(condition.upper(), e), flush=True, file=foldTerminalOutputStream)
                    tableData = []
                    
                    tableData.append(["CostAve", trainCostAve, valCostAve, testCostAve])
                    tableData.append(["CostStd", trainCostStd, valCostStd, testCostStd])
                    tableData.append(["ShopkeeperSpeechClusterLossAve", trainShopkeeperSpeechClusterLossAve, valShopkeeperSpeechClusterLossAve, testShopkeeperSpeechClusterLossAve])
                    tableData.append(["ShopkeeperSpeechClusterLossStd", trainShopkeeperSpeechClusterLossStd, valShopkeeperSpeechClusterLossStd, testShopkeeperSpeechClusterLossStd])
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
                    tableData.append(["SilenceCorrAcc", trainSilenceCorrAcc, valSilenceCorrAcc, testSilenceCorrAcc])
                    
                    
                    print(tabulate(tableData, headers=["METRIC", "TRAINING", "VALIDATION", "TESTING"], floatfmt=".3f", tablefmt="grid"), flush=True, file=foldTerminalOutputStream)
                            
                    print("", flush=True, file=foldTerminalOutputStream)
            
                #################################################################################################################
                # END PROPOSED RUN!
                #################################################################################################################
            
            
            if bl1_run:
            
                #################################################################################################################
                # BEGIN BASELINE 1 RUN!
                #################################################################################################################
                
                if e != 0:
                    # train
                    learner.train(
                        inputSequenceVectors[trainIndices],
                        bl1_outputSpeechClusterIds[trainIndices],
                        outputShopkeeperLocations[trainIndices],
                        outputSpatialStates[trainIndices],
                        outputStateTargets[trainIndices],                                           
                        bl1_outputMasks[trainIndices])
                
                
                # evaluate
                if e % evalEvery == 0 or e == numEpochs:
                    
                    # training loss
                    trainCost, trainShopkeeperSpeechClusterLoss, trainLocationLoss, trainSpatialStateLoss, trainStateTargetLoss = learner.get_loss(
                        inputSequenceVectors[trainIndices],
                        bl1_outputSpeechClusterIds[trainIndices],
                        outputShopkeeperLocations[trainIndices],
                        outputSpatialStates[trainIndices],
                        outputStateTargets[trainIndices],                                           
                        bl1_outputMasks[trainIndices])
                    
                    # validation loss
                    valCost, valShopkeeperSpeechClusterLoss, valLocationLoss, valSpatialStateLoss, valStateTargetLoss = learner.get_loss(
                        inputSequenceVectors[valIndices],
                        bl1_outputSpeechClusterIds[valIndices],
                        outputShopkeeperLocations[valIndices],
                        outputSpatialStates[valIndices],
                        outputStateTargets[valIndices],                                           
                        bl1_outputMasks[valIndices])
                    
                    # test loss
                    testCost, testShopkeeperSpeechClusterLoss, testLocationLoss, testSpatialStateLoss, testStateTargetLoss = learner.get_loss(
                        inputSequenceVectors[testIndices],
                        bl1_outputSpeechClusterIds[testIndices],
                        outputShopkeeperLocations[testIndices],
                        outputSpatialStates[testIndices],
                        outputStateTargets[testIndices],                                           
                        bl1_outputMasks[testIndices])
                    
                    # compute loss averages and s.d. for aggregate log
                    # train
                    trainCostAve = np.mean(trainCost)
                    trainShopkeeperSpeechClusterLossAve = np.mean(trainShopkeeperSpeechClusterLoss)
                    trainLocationLossAve = np.mean(trainLocationLoss)
                    trainSpatialStateLossAve = np.mean(trainSpatialStateLoss)
                    trainStateTargetLossAve = np.mean(trainStateTargetLoss)
                    
                    trainCostStd = np.std(trainCost)
                    trainShopkeeperSpeechClusterLossStd = np.std(trainShopkeeperSpeechClusterLoss)
                    trainLocationLossStd = np.std(trainLocationLoss)
                    trainSpatialStateLossStd = np.std(trainSpatialStateLoss)
                    trainStateTargetLossStd = np.std(trainStateTargetLoss)
                    
                    # validation
                    valCostAve = np.mean(valCost)
                    valShopkeeperSpeechClusterLossAve = np.mean(valShopkeeperSpeechClusterLoss)
                    valLocationLossAve = np.mean(valLocationLoss)
                    valSpatialStateLossAve = np.mean(valSpatialStateLoss)
                    valStateTargetLossAve = np.mean(valStateTargetLoss)
                    
                    valCostStd = np.std(valCost)
                    valShopkeeperSpeechClusterLossStd = np.std(valShopkeeperSpeechClusterLoss)
                    valLocationLossStd = np.std(valLocationLoss)
                    valSpatialStateLossStd = np.std(valSpatialStateLoss)
                    valStateTargetLossStd = np.std(valStateTargetLoss)
                    
                    # test
                    testCostAve = np.mean(testCost)
                    testShopkeeperSpeechClusterLossAve = np.mean(testShopkeeperSpeechClusterLoss)
                    testLocationLossAve = np.mean(testLocationLoss)
                    testSpatialStateLossAve = np.mean(testSpatialStateLoss)
                    testStateTargetLossAve = np.mean(testStateTargetLoss)
                    
                    testCostStd = np.std(testCost)
                    testShopkeeperSpeechClusterLossStd = np.std(testShopkeeperSpeechClusterLoss)
                    testLocationLossStd = np.std(testLocationLoss)
                    testSpatialStateLossStd = np.std(testSpatialStateLoss)
                    testStateTargetLossStd = np.std(testStateTargetLoss)
                    
                    
                    # predict
                    predShkpSpeechClustID, predLocations, predSpatialStates, predStateTargets = learner.predict(
                        inputSequenceVectors,
                        bl1_outputSpeechClusterIds,
                        outputShopkeeperLocations, 
                        outputSpatialStates, 
                        outputStateTargets,
                        bl1_outputMasks)
                    
                    
                    def evaluate_predictions_bl1(evalSetName, evalIndices, csvLogRows):
                        
                        # for computing accuracies
                        speechClusts_gt = []
                        speechClusts_pred = []
                        
                        locs_gt = []
                        locs_pred = []
                        
                        spatSts_gt = []
                        spatSts_pred = []
                        
                        stTargs_gt = []
                        stTargs_pred = []
                        
                        silence_gt = []
                        silence_pred = []
                        
                        for i in evalIndices:
                            
                            # check if the index is one of the ones that was cut off because of the batch size
                            if i >= len(predShkpSpeechClustID):
                                continue
                            
                            csvLogRows[i]["SET"] = evalSetName
                            csvLogRows[i]["ID"] = i
                            
                            #
                            # target info
                            #
                            csvLogRows[i]["TARG_SHOPKEEPER_SPEECH_CLUSTER_ID"] = bl1_outputSpeechClusterIds[i]
                            csvLogRows[i]["TARG_SHOPKEEPER_SPEECH_CLUSTER_ID_IS_JUNK"] = 1 if bl1_outputSpeechClusterIds[i] in bl1_junkSpeechClusterIds else 0
                            csvLogRows[i]["TARG_OUTPUT_SHOPKEEPER_LOCATION"] = locations[outputShopkeeperLocations[i]]
                            csvLogRows[i]["TARG_OUTPUT_SPATIAL_STATE"] = spatialStates[outputSpatialStates[i]]
                            csvLogRows[i]["TARG_OUTPUT_STATE_TARGET"] = stateTargets[outputStateTargets[i]]
                            
                            
                            #
                            # prediction info
                            #
                            outShkpSpeech = bl1_shkpSpeechClustIdToRepUtt[predShkpSpeechClustID[i]]
                            
                            csvLogRows[i]["PRED_OUTPUT_SHOPKEEPER_SPEECH_CLUSTER_ID"] = predShkpSpeechClustID[i]
                            csvLogRows[i]["PRED_SHOPKEEPER_SPEECH"] = outShkpSpeech
                            csvLogRows[i]["PRED_OUTPUT_SHOPKEEPER_LOCATION"] = locations[predLocations[i]]
                            csvLogRows[i]["PRED_OUTPUT_SPATIAL_STATE"] = spatialStates[predSpatialStates[i]]
                            csvLogRows[i]["PRED_OUTPUT_STATE_TARGET"] = stateTargets[predStateTargets[i]]
                            
                            
                            #
                            # for computing accuracies
                            #
                            if bl1_outputSpeechClusterIds[i] not in bl1_junkSpeechClusterIds:
                                speechClusts_gt.append(bl1_outputSpeechClusterIds[i])
                                speechClusts_pred.append(predShkpSpeechClustID[i])
                            
                            locs_gt.append(csvLogRows[i]["TARG_OUTPUT_SHOPKEEPER_LOCATION"])
                            locs_pred.append(csvLogRows[i]["PRED_OUTPUT_SHOPKEEPER_LOCATION"])
                            
                            spatSts_gt.append(csvLogRows[i]["TARG_OUTPUT_SPATIAL_STATE"])
                            spatSts_pred.append(csvLogRows[i]["PRED_OUTPUT_SPATIAL_STATE"])
                            
                            stTargs_gt.append(csvLogRows[i]["TARG_OUTPUT_STATE_TARGET"])
                            stTargs_pred.append(csvLogRows[i]["PRED_OUTPUT_STATE_TARGET"])
                            
                            if ((csvLogRows[i]["TARG_SHOPKEEPER_SPEECH_CLUSTER_ID"] == bl1_silenceSpeechClusterId) or (predShkpSpeechClustID[i] == bl1_silenceSpeechClusterId)):
                                silence_gt.append(csvLogRows[i]["TARG_SHOPKEEPER_SPEECH_CLUSTER_ID"])
                                silence_pred.append(predShkpSpeechClustID[i])
                        
                        
                        #
                        # compute accuracies
                        #
                        speechClustCorrAcc = accuracy_score(speechClusts_gt, speechClusts_pred)
                        locCorrAcc = accuracy_score(locs_gt, locs_pred)
                        spatStCorrAcc = accuracy_score(spatSts_gt, spatSts_pred)
                        stTargCorrAcc = accuracy_score(stTargs_gt, stTargs_pred)
                        silenceCorrAcc = accuracy_score(silence_gt, silence_pred)
                        
                        
                        return csvLogRows, speechClustCorrAcc, locCorrAcc, spatStCorrAcc, stTargCorrAcc, silenceCorrAcc
                    
                    
                    csvLogRows = copy.deepcopy(interactions)
                    
                    csvLogRows, trainSpeechClustCorrAcc, trainLocCorrAcc, trainSpatStCorrAcc, trainStTargCorrAcc, trainSilenceCorrAcc = evaluate_predictions_bl1("TRAIN", trainIndices, csvLogRows)
                    
                    csvLogRows, valSpeechClustCorrAcc, valLocCorrAcc, valSpatStCorrAcc, valStTargCorrAcc, valSilenceCorrAcc = evaluate_predictions_bl1("VAL", valIndices, csvLogRows)
                    
                    csvLogRows, testSpeechClustCorrAcc, testLocCorrAcc, testSpatStCorrAcc, testStTargCorrAcc, testSilenceCorrAcc = evaluate_predictions_bl1("TEST", testIndices, csvLogRows)
                    
                    
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
                        writer.writerow([e,
                                         
                                         # training
                                         trainCostAve,
                                         trainCostStd,
                                         trainShopkeeperSpeechClusterLossAve,
                                         trainShopkeeperSpeechClusterLossStd,
                                         trainLocationLossAve,
                                         trainLocationLossStd,
                                         trainSpatialStateLossAve,
                                         trainSpatialStateLossStd,
                                         trainStateTargetLossAve,
                                         trainStateTargetLossStd,
                                         
                                         trainSpeechClustCorrAcc,
                                         trainLocCorrAcc,
                                         trainSpatStCorrAcc,
                                         trainStTargCorrAcc,
                                         trainSilenceCorrAcc,
                                         
                                         # validation
                                         valCostAve,
                                         valCostStd,
                                         valShopkeeperSpeechClusterLossAve,
                                         valShopkeeperSpeechClusterLossStd,
                                         valLocationLossAve,
                                         valLocationLossStd,
                                         valSpatialStateLossAve,
                                         valSpatialStateLossStd,
                                         valStateTargetLossAve,
                                         valStateTargetLossStd,
                                         
                                         valSpeechClustCorrAcc,
                                         valLocCorrAcc,
                                         valSpatStCorrAcc,
                                         valStTargCorrAcc,
                                         valSilenceCorrAcc,
                                         
                                         # testing
                                         testCostAve,
                                         testCostStd,
                                         testShopkeeperSpeechClusterLossAve,
                                         testShopkeeperSpeechClusterLossStd,
                                         testLocationLossAve,
                                         testLocationLossStd,
                                         testSpatialStateLossAve,
                                         testSpatialStateLossStd,
                                         testStateTargetLossAve,
                                         testStateTargetLossStd,
                                         
                                         testSpeechClustCorrAcc,
                                         testLocCorrAcc,
                                         testSpatStCorrAcc,
                                         testStTargCorrAcc,
                                         testSilenceCorrAcc
                                         ])    
                
                
                    # training
                    print("===== {} EPOCH {} LOSSES AND ACCURACIES=====".format(condition.upper(), e), flush=True, file=foldTerminalOutputStream)
                    tableData = []
                    
                    tableData.append(["CostAve", trainCostAve, valCostAve, testCostAve])
                    tableData.append(["CostStd", trainCostStd, valCostStd, testCostStd])
                    tableData.append(["ShopkeeperSpeechClusterLossAve", trainShopkeeperSpeechClusterLossAve, valShopkeeperSpeechClusterLossAve, testShopkeeperSpeechClusterLossAve])
                    tableData.append(["ShopkeeperSpeechClusterLossStd", trainShopkeeperSpeechClusterLossStd, valShopkeeperSpeechClusterLossStd, testShopkeeperSpeechClusterLossStd])
                    tableData.append(["LocationLossAve", trainLocationLossAve, valLocationLossAve, testLocationLossAve])
                    tableData.append(["LocationLossStd", trainLocationLossStd, valLocationLossStd, testLocationLossStd])
                    tableData.append(["SpatialStateLossAve", trainSpatialStateLossAve, valSpatialStateLossAve, testSpatialStateLossAve])
                    tableData.append(["SpatialStateLossStd", trainSpatialStateLossStd, valSpatialStateLossStd, testSpatialStateLossStd])
                    tableData.append(["StateTargetLossAve", trainStateTargetLossAve, valStateTargetLossAve, testStateTargetLossAve])
                    tableData.append(["StateTargetLossStd", trainStateTargetLossStd, valStateTargetLossStd, testStateTargetLossStd])
                    
                    tableData.append(["SpeechClustCorrAcc", trainSpeechClustCorrAcc, valSpeechClustCorrAcc, testSpeechClustCorrAcc])
                    tableData.append(["LocCorrAcc", trainLocCorrAcc, valLocCorrAcc, testLocCorrAcc])
                    tableData.append(["SpatStCorrAcc", trainSpatStCorrAcc, valSpatStCorrAcc, testSpatStCorrAcc])
                    tableData.append(["StTargCorrAcc", trainStTargCorrAcc, valStTargCorrAcc, testStTargCorrAcc])
                    tableData.append(["SilenceCorrAcc", trainSilenceCorrAcc, valSilenceCorrAcc, testSilenceCorrAcc])
                    
                    print(tabulate(tableData, headers=["METRIC", "TRAINING", "VALIDATION", "TESTING"], floatfmt=".3f", tablefmt="grid"), flush=True, file=foldTerminalOutputStream)
                            
                    print("", flush=True, file=foldTerminalOutputStream)
            
            #################################################################################################################
            # END BASELINE 1 RUN!
            #################################################################################################################
            
            
            if copy_run:
            
                #################################################################################################################
                # BEGIN COPYNET BASED RUN!
                #################################################################################################################
                if e != 0:
                    # train
                    learner.train(
                        inputSequenceVectors[trainIndices],
                        copy_outputSpeechSequenceVecs[trainIndices],
                        copy_outputSpeechSequenceLens[trainIndices],
                        outputShopkeeperLocations[trainIndices],
                        outputSpatialStates[trainIndices],
                        outputStateTargets[trainIndices],
                        copy_databaseSequenceVecs[trainIndices])
                
                # evaluate
                if e % evalEvery == 0 or e == numEpochs:
                    
                    # training loss
                    trainCost, trainShopkeeperSpeechSequenceLoss, trainLocationLoss, trainSpatialStateLoss, trainStateTargetLoss = learner.get_loss(
                        inputSequenceVectors[trainIndices],
                        copy_outputSpeechSequenceVecs[trainIndices],
                        copy_outputSpeechSequenceLens[trainIndices],
                        outputShopkeeperLocations[trainIndices],
                        outputSpatialStates[trainIndices],
                        outputStateTargets[trainIndices],
                        copy_databaseSequenceVecs[trainIndices])
                    
                    # validation loss
                    valCost, valShopkeeperSpeechSequenceLoss, valLocationLoss, valSpatialStateLoss, valStateTargetLoss = learner.get_loss(
                        inputSequenceVectors[valIndices],
                        copy_outputSpeechSequenceVecs[valIndices],
                        copy_outputSpeechSequenceLens[valIndices],
                        outputShopkeeperLocations[valIndices],
                        outputSpatialStates[valIndices],
                        outputStateTargets[valIndices],
                        copy_databaseSequenceVecs[valIndices])
                    
                    # test loss
                    testCost, testShopkeeperSpeechSequenceLoss, testLocationLoss, testSpatialStateLoss, testStateTargetLoss = learner.get_loss(
                        inputSequenceVectors[testIndices],
                        copy_outputSpeechSequenceVecs[testIndices],
                        copy_outputSpeechSequenceLens[testIndices],
                        outputShopkeeperLocations[testIndices],
                        outputSpatialStates[testIndices],
                        outputStateTargets[testIndices],
                        copy_databaseSequenceVecs[testIndices])
                    
                    # compute loss averages and s.d. for aggregate log
                    # train
                    trainCostAve = np.mean(trainCost)
                    trainShopkeeperSpeechSequenceLossAve = np.mean(trainShopkeeperSpeechSequenceLoss)
                    trainLocationLossAve = np.mean(trainLocationLoss)
                    trainSpatialStateLossAve = np.mean(trainSpatialStateLoss)
                    trainStateTargetLossAve = np.mean(trainStateTargetLoss)
                    
                    trainCostStd = np.std(trainCost)
                    trainShopkeeperSpeechSequenceLossStd = np.std(trainShopkeeperSpeechSequenceLoss)
                    trainLocationLossStd = np.std(trainLocationLoss)
                    trainSpatialStateLossStd = np.std(trainSpatialStateLoss)
                    trainStateTargetLossStd = np.std(trainStateTargetLoss)
                    
                    # validation
                    valCostAve = np.mean(valCost)
                    valShopkeeperSpeechSequenceLossAve = np.mean(valShopkeeperSpeechSequenceLoss)
                    valLocationLossAve = np.mean(valLocationLoss)
                    valSpatialStateLossAve = np.mean(valSpatialStateLoss)
                    valStateTargetLossAve = np.mean(valStateTargetLoss)
                    
                    valCostStd = np.std(valCost)
                    valShopkeeperSpeechSequenceLossStd = np.std(valShopkeeperSpeechSequenceLoss)
                    valLocationLossStd = np.std(valLocationLoss)
                    valSpatialStateLossStd = np.std(valSpatialStateLoss)
                    valStateTargetLossStd = np.std(valStateTargetLoss)
                    
                    # test
                    testCostAve = np.mean(testCost)
                    testShopkeeperSpeechSequenceLossAve = np.mean(testShopkeeperSpeechSequenceLoss)
                    testLocationLossAve = np.mean(testLocationLoss)
                    testSpatialStateLossAve = np.mean(testSpatialStateLoss)
                    testStateTargetLossAve = np.mean(testStateTargetLoss)
                    
                    testCostStd = np.std(testCost)
                    testShopkeeperSpeechSequenceLossStd = np.std(testShopkeeperSpeechSequenceLoss)
                    testLocationLossStd = np.std(testLocationLoss)
                    testSpatialStateLossStd = np.std(testSpatialStateLoss)
                    testStateTargetLossStd = np.std(testStateTargetLoss)
                
                
                    # predict
                    predShkpSpeechSequences, predLocations, predSpatialStates, predStateTargets, predCamIndexWeights, predAttributeIndexWeights = learner.predict(
                        inputSequenceVectors,
                        copy_outputSpeechSequenceVecs,
                        copy_outputSpeechSequenceLens,
                        outputShopkeeperLocations,
                        outputSpatialStates,
                        outputStateTargets,
                        copy_databaseSequenceVecs)
                    
                    
                    def evaluate_predictions_copy(evalSetName, evalIndices, csvLogRows):
                        
                        # for computing accuracies
                        speechSequenceAccuracies = []
                        
                        locs_gt = []
                        locs_pred = []
                        
                        spatSts_gt = []
                        spatSts_pred = []
                        
                        stTargs_gt = []
                        stTargs_pred = []
                        
                        camIndex_gt = []
                        camIndex_pred = []
                        
                        attrIndex_gt = []
                        attrIndex_pred = []
                        
                        
                        for i in evalIndices:
                            
                            # check if the index is one of the ones that was cut off because of the batch size
                            if i >= len(predShkpSpeechSequences):
                                continue
                            
                            csvLogRows[i]["SET"] = evalSetName
                            csvLogRows[i]["ID"] = i
                            
                            #
                            # target info
                            #
                            targShkpSpeechSeq = [copy_indexToWord[w] for w in copy_outputSpeechSequenceVecs[i] if w != -1]
                            targShkpSpeech = " ".join(targShkpSpeechSeq[:targShkpSpeechSeq.index("<eof>")])
                                                        
                            csvLogRows[i]["TARG_SHOPKEEPER_SPEECH_SEQUENCE"] = targShkpSpeechSeq
                            csvLogRows[i]["TARG_SHOPKEEPER_SPEECH"] = targShkpSpeech
                            csvLogRows[i]["TARG_OUTPUT_SHOPKEEPER_LOCATION"] = locations[outputShopkeeperLocations[i]]
                            csvLogRows[i]["TARG_OUTPUT_SPATIAL_STATE"] = spatialStates[outputSpatialStates[i]]
                            csvLogRows[i]["TARG_OUTPUT_STATE_TARGET"] = stateTargets[outputStateTargets[i]]
                            
                            
                            #
                            # prediction info
                            #
                            predShkpSpeechSeq = [copy_indexToWord[w] for w in predShkpSpeechSequences[i]]
                            
                            try:
                                predShkpSpeechLen = predShkpSpeechSeq.index("<eof>")
                                outShkpSpeech = " ".join(predShkpSpeechSeq[:predShkpSpeechLen])
                                
                            except:
                                predShkpSpeechLen = 0
                                outShkpSpeech = " ".join(predShkpSpeechSeq)
                            
                            
                            csvLogRows[i]["PRED_SHOPKEEPER_SPEECH_SEQUENCE"] = predShkpSpeechSeq
                            csvLogRows[i]["PRED_SHOPKEEPER_SPEECH"] = outShkpSpeech
                            csvLogRows[i]["PRED_OUTPUT_SHOPKEEPER_LOCATION"] = locations[predLocations[i]]
                            csvLogRows[i]["PRED_OUTPUT_SPATIAL_STATE"] = spatialStates[predSpatialStates[i]]
                            csvLogRows[i]["PRED_OUTPUT_STATE_TARGET"] = stateTargets[predStateTargets[i]]
                            
                            maxPredCamIndex = np.argmax(predCamIndexWeights[i])
                            maxPredAttrIndex = np.argmax(predAttributeIndexWeights[i])
                            
                            csvLogRows[i]["PRED_CAMERA_INDEX_MAX"] = cameras[maxPredCamIndex]
                            csvLogRows[i]["PRED_ATTRIBUTE_INDEX_MAX"] = dbFieldnames[maxPredAttrIndex]
                            
                            for c in range(len(cameras)):
                                csvLogRows[i]["PRED_{}_WEIGHT".format(cameras[c])] = predCamIndexWeights[i,c]
                            
                            for a in range(len(attributes)):
                                csvLogRows[i]["PRED_{}_WEIGHT".format(attributes[a])] = predAttributeIndexWeights[i,a]
                            
                            
                            #
                            # for computing accuracies
                            #
                            numTokensTotal = max(copy_outputSpeechSequenceLens[i], predShkpSpeechLen)
                            numTokensCorrect = 0
                            
                            for w in range(min(copy_outputSpeechSequenceLens[i], predShkpSpeechLen)):
                                if targShkpSpeechSeq[w] == predShkpSpeechSeq[w]:
                                    numTokensCorrect += 1
                            
                            percTokensCorrect = numTokensCorrect / numTokensTotal
                            speechSequenceAccuracies.append(percTokensCorrect)
                            
                            
                            locs_gt.append(csvLogRows[i]["TARG_OUTPUT_SHOPKEEPER_LOCATION"])
                            locs_pred.append(csvLogRows[i]["PRED_OUTPUT_SHOPKEEPER_LOCATION"])
                            
                            spatSts_gt.append(csvLogRows[i]["TARG_OUTPUT_SPATIAL_STATE"])
                            spatSts_pred.append(csvLogRows[i]["PRED_OUTPUT_SPATIAL_STATE"])
                            
                            stTargs_gt.append(csvLogRows[i]["TARG_OUTPUT_STATE_TARGET"])
                            stTargs_pred.append(csvLogRows[i]["PRED_OUTPUT_STATE_TARGET"])
                            
                            
                            if csvLogRows[i]["CURRENT_CAMERA_OF_CONVERSATION"] != "NONE":
                                camIndex_gt.append(cameras.index(csvLogRows[i]["CURRENT_CAMERA_OF_CONVERSATION"]))
                                camIndex_pred.append(maxPredCamIndex)
                            
                            # this only looks at cases where the customer asks about a feature, not when shkp introduces a feature/camera
                            if csvLogRows[i]["CUSTOMER_TOPIC"] != "NONE":
                                attrIndex_gt.append(dbFieldnames.index(csvLogRows[i]["CUSTOMER_TOPIC"]))
                                attrIndex_pred.append(maxPredAttrIndex)
                        
                        #
                        # compute accuracies
                        #
                        speechSeqCorrAcc = np.mean(speechSequenceAccuracies)
                        locCorrAcc = accuracy_score(locs_gt, locs_pred)
                        spatStCorrAcc = accuracy_score(spatSts_gt, spatSts_pred)
                        stTargCorrAcc = accuracy_score(stTargs_gt, stTargs_pred)
                        camCorrAcc = accuracy_score(camIndex_gt, camIndex_pred)
                        attrCorrAcc = accuracy_score(attrIndex_gt, attrIndex_pred)
                        
                        
                        return csvLogRows, speechSeqCorrAcc, locCorrAcc, spatStCorrAcc, stTargCorrAcc, camCorrAcc, attrCorrAcc
                    
                    
                    csvLogRows = copy.deepcopy(interactions)
                    
                    csvLogRows, trainSpeechSeqCorrAcc, trainLocCorrAcc, trainSpatStCorrAcc, trainStTargCorrAcc, trainCamCorrAcc, trainAttrCorrAcc = evaluate_predictions_copy("TRAIN", trainIndices, csvLogRows)
                    
                    csvLogRows, valSpeechSeqCorrAcc, valLocCorrAcc, valSpatStCorrAcc, valStTargCorrAcc, valCamCorrAcc, valAttrCorrAcc = evaluate_predictions_copy("VAL", valIndices, csvLogRows)
                    
                    csvLogRows, testSpeechSeqCorrAcc, testLocCorrAcc, testSpatStCorrAcc, testStTargCorrAcc, testCamCorrAcc, testAttrCorrAcc = evaluate_predictions_copy("TEST", testIndices, csvLogRows)
                    
                    
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
                        writer.writerow([e,
                                         
                                         # training
                                         trainCostAve,
                                         trainCostStd,
                                         trainShopkeeperSpeechSequenceLossAve,
                                         trainShopkeeperSpeechSequenceLossStd,
                                         trainLocationLossAve,
                                         trainLocationLossStd,
                                         trainSpatialStateLossAve,
                                         trainSpatialStateLossStd,
                                         trainStateTargetLossAve,
                                         trainStateTargetLossStd,
                                         
                                         trainSpeechSeqCorrAcc,
                                         trainLocCorrAcc,
                                         trainSpatStCorrAcc,
                                         trainStTargCorrAcc,
                                         trainCamCorrAcc,
                                         trainAttrCorrAcc,
                                         
                                         # validation
                                         valCostAve,
                                         valCostStd,
                                         valShopkeeperSpeechSequenceLossAve,
                                         valShopkeeperSpeechSequenceLossStd,
                                         valLocationLossAve,
                                         valLocationLossStd,
                                         valSpatialStateLossAve,
                                         valSpatialStateLossStd,
                                         valStateTargetLossAve,
                                         valStateTargetLossStd,
                                         
                                         valSpeechSeqCorrAcc,
                                         valLocCorrAcc,
                                         valSpatStCorrAcc,
                                         valStTargCorrAcc,
                                         valCamCorrAcc,
                                         valAttrCorrAcc,
                                         
                                         # testing
                                         testCostAve,
                                         testCostStd,
                                         testShopkeeperSpeechSequenceLossAve,
                                         testShopkeeperSpeechSequenceLossStd,
                                         testLocationLossAve,
                                         testLocationLossStd,
                                         testSpatialStateLossAve,
                                         testSpatialStateLossStd,
                                         testStateTargetLossAve,
                                         testStateTargetLossStd,
                                         
                                         testSpeechSeqCorrAcc,
                                         testLocCorrAcc,
                                         testSpatStCorrAcc,
                                         testStTargCorrAcc,
                                         testCamCorrAcc,
                                         testAttrCorrAcc
                                         ])    
                    
                    # training
                    print("===== {} EPOCH {} LOSSES AND ACCURACIES=====".format(condition.upper(), e), flush=True, file=foldTerminalOutputStream)
                    tableData = []
                    
                    tableData.append(["CostAve", trainCostAve, valCostAve, testCostAve])
                    tableData.append(["CostStd", trainCostStd, valCostStd, testCostStd])
                    tableData.append(["ShopkeeperSpeechSequenceLossAve", trainShopkeeperSpeechSequenceLossAve, valShopkeeperSpeechSequenceLossAve, testShopkeeperSpeechSequenceLossAve])
                    tableData.append(["ShopkeeperSpeechSequenceLossStd", trainShopkeeperSpeechSequenceLossStd, valShopkeeperSpeechSequenceLossStd, testShopkeeperSpeechSequenceLossStd])
                    tableData.append(["LocationLossAve", trainLocationLossAve, valLocationLossAve, testLocationLossAve])
                    tableData.append(["LocationLossStd", trainLocationLossStd, valLocationLossStd, testLocationLossStd])
                    tableData.append(["SpatialStateLossAve", trainSpatialStateLossAve, valSpatialStateLossAve, testSpatialStateLossAve])
                    tableData.append(["SpatialStateLossStd", trainSpatialStateLossStd, valSpatialStateLossStd, testSpatialStateLossStd])
                    tableData.append(["StateTargetLossAve", trainStateTargetLossAve, valStateTargetLossAve, testStateTargetLossAve])
                    tableData.append(["StateTargetLossStd", trainStateTargetLossStd, valStateTargetLossStd, testStateTargetLossStd])
                    
                    tableData.append(["SpeechSeqCorrAcc", trainSpeechSeqCorrAcc, valSpeechSeqCorrAcc, testSpeechSeqCorrAcc])
                    tableData.append(["LocCorrAcc", trainLocCorrAcc, valLocCorrAcc, testLocCorrAcc])
                    tableData.append(["SpatStCorrAcc", trainSpatStCorrAcc, valSpatStCorrAcc, testSpatStCorrAcc])
                    tableData.append(["StTargCorrAcc", trainStTargCorrAcc, valStTargCorrAcc, testStTargCorrAcc])
                    tableData.append(["AttrCorrAcc", trainAttrCorrAcc, valAttrCorrAcc, testAttrCorrAcc])
                    tableData.append(["CamCorrAcc", trainCamCorrAcc, valCamCorrAcc, testCamCorrAcc])
                    
                    
                    print(tabulate(tableData, headers=["METRIC", "TRAINING", "VALIDATION", "TESTING"], floatfmt=".3f", tablefmt="grid"), flush=True, file=foldTerminalOutputStream)
                            
                    print("", flush=True, file=foldTerminalOutputStream)
                    
                
                #################################################################################################################
                # END COPYNET BASED RUN!
                #################################################################################################################
            
            
            
            if coreqa_run:
            
                #################################################################################################################
                # BEGIN COREQA RUN!
                #################################################################################################################
                if e != 0:
                    # train
                    learner.train(
                        inputSequenceVectors[trainIndices],
                        coreqa_outputSpeechSequenceVecs[trainIndices],
                        coreqa_outputSpeechSequenceLens[trainIndices],
                        outputShopkeeperLocations[trainIndices],
                        outputSpatialStates[trainIndices],
                        outputStateTargets[trainIndices],
                        coreqa_databaseCamVecs[trainIndices],
                        coreqa_databaseAttrVecs[trainIndices],
                        coreqa_databaseValVecs[trainIndices])
                
                # evaluate
                if e % evalEvery == 0 or e == numEpochs:
                    
                    # training loss
                    trainCost, trainShopkeeperSpeechSequenceLoss, trainLocationLoss, trainSpatialStateLoss, trainStateTargetLoss = learner.get_loss(
                        inputSequenceVectors[trainIndices],
                        coreqa_outputSpeechSequenceVecs[trainIndices],
                        coreqa_outputSpeechSequenceLens[trainIndices],
                        outputShopkeeperLocations[trainIndices],
                        outputSpatialStates[trainIndices],
                        outputStateTargets[trainIndices],
                        coreqa_databaseCamVecs[trainIndices],
                        coreqa_databaseAttrVecs[trainIndices],
                        coreqa_databaseValVecs[trainIndices])
                    
                    # validation loss
                    valCost, valShopkeeperSpeechSequenceLoss, valLocationLoss, valSpatialStateLoss, valStateTargetLoss = learner.get_loss(
                        inputSequenceVectors[valIndices],
                        coreqa_outputSpeechSequenceVecs[valIndices],
                        coreqa_outputSpeechSequenceLens[valIndices],
                        outputShopkeeperLocations[valIndices],
                        outputSpatialStates[valIndices],
                        outputStateTargets[valIndices],
                        coreqa_databaseCamVecs[valIndices],
                        coreqa_databaseAttrVecs[valIndices],
                        coreqa_databaseValVecs[valIndices])
                    
                    # test loss
                    testCost, testShopkeeperSpeechSequenceLoss, testLocationLoss, testSpatialStateLoss, testStateTargetLoss = learner.get_loss(
                        inputSequenceVectors[testIndices],
                        coreqa_outputSpeechSequenceVecs[testIndices],
                        coreqa_outputSpeechSequenceLens[testIndices],
                        outputShopkeeperLocations[testIndices],
                        outputSpatialStates[testIndices],
                        outputStateTargets[testIndices],
                        coreqa_databaseCamVecs[testIndices],
                        coreqa_databaseAttrVecs[testIndices],
                        coreqa_databaseValVecs[testIndices])
                    
                    # compute loss averages and s.d. for aggregate log
                    # train
                    trainCostAve = np.mean(trainCost)
                    trainShopkeeperSpeechSequenceLossAve = np.mean(trainShopkeeperSpeechSequenceLoss)
                    trainLocationLossAve = np.mean(trainLocationLoss)
                    trainSpatialStateLossAve = np.mean(trainSpatialStateLoss)
                    trainStateTargetLossAve = np.mean(trainStateTargetLoss)
                    
                    trainCostStd = np.std(trainCost)
                    trainShopkeeperSpeechSequenceLossStd = np.std(trainShopkeeperSpeechSequenceLoss)
                    trainLocationLossStd = np.std(trainLocationLoss)
                    trainSpatialStateLossStd = np.std(trainSpatialStateLoss)
                    trainStateTargetLossStd = np.std(trainStateTargetLoss)
                    
                    # validation
                    valCostAve = np.mean(valCost)
                    valShopkeeperSpeechSequenceLossAve = np.mean(valShopkeeperSpeechSequenceLoss)
                    valLocationLossAve = np.mean(valLocationLoss)
                    valSpatialStateLossAve = np.mean(valSpatialStateLoss)
                    valStateTargetLossAve = np.mean(valStateTargetLoss)
                    
                    valCostStd = np.std(valCost)
                    valShopkeeperSpeechSequenceLossStd = np.std(valShopkeeperSpeechSequenceLoss)
                    valLocationLossStd = np.std(valLocationLoss)
                    valSpatialStateLossStd = np.std(valSpatialStateLoss)
                    valStateTargetLossStd = np.std(valStateTargetLoss)
                    
                    # test
                    testCostAve = np.mean(testCost)
                    testShopkeeperSpeechSequenceLossAve = np.mean(testShopkeeperSpeechSequenceLoss)
                    testLocationLossAve = np.mean(testLocationLoss)
                    testSpatialStateLossAve = np.mean(testSpatialStateLoss)
                    testStateTargetLossAve = np.mean(testStateTargetLoss)
                    
                    testCostStd = np.std(testCost)
                    testShopkeeperSpeechSequenceLossStd = np.std(testShopkeeperSpeechSequenceLoss)
                    testLocationLossStd = np.std(testLocationLoss)
                    testSpatialStateLossStd = np.std(testSpatialStateLoss)
                    testStateTargetLossStd = np.std(testStateTargetLoss)
                
                
                    # predict
                    predShkpSpeechSequences, predLocations, predSpatialStates, predStateTargets, predDbFactMatchScores = learner.predict(
                        inputSequenceVectors,
                        coreqa_outputSpeechSequenceVecs,
                        coreqa_outputSpeechSequenceLens,
                        outputShopkeeperLocations,
                        outputSpatialStates,
                        outputStateTargets,
                        coreqa_databaseCamVecs,
                        coreqa_databaseAttrVecs,
                        coreqa_databaseValVecs)
                    
                    
                    def evaluate_predictions_coreqa(evalSetName, evalIndices, csvLogRows):
                        
                        # for computing accuracies
                        speechSequenceAccuracies = []
                        
                        locs_gt = []
                        locs_pred = []
                        
                        spatSts_gt = []
                        spatSts_pred = []
                        
                        stTargs_gt = []
                        stTargs_pred = []
                        
                        camIndex_gt = []
                        camIndex_pred = []
                        
                        attrIndex_gt = []
                        attrIndex_pred = []
                        
                        
                        for i in evalIndices:
                            
                            # check if the index is one of the ones that was cut off because of the batch size
                            if i >= len(predShkpSpeechSequences):
                                continue
                            
                            csvLogRows[i]["SET"] = evalSetName
                            csvLogRows[i]["ID"] = i
                            
                            #
                            # target info
                            #
                            targShkpSpeechSeq = [coreqa_indexToWord[w] for w in coreqa_outputSpeechSequenceVecs[i] if w != -1]
                            targShkpSpeech = " ".join(targShkpSpeechSeq[:targShkpSpeechSeq.index("<eof>")])
                                                        
                            csvLogRows[i]["TARG_SHOPKEEPER_SPEECH_SEQUENCE"] = targShkpSpeechSeq
                            csvLogRows[i]["TARG_SHOPKEEPER_SPEECH"] = targShkpSpeech
                            csvLogRows[i]["TARG_OUTPUT_SHOPKEEPER_LOCATION"] = locations[outputShopkeeperLocations[i]]
                            csvLogRows[i]["TARG_OUTPUT_SPATIAL_STATE"] = spatialStates[outputSpatialStates[i]]
                            csvLogRows[i]["TARG_OUTPUT_STATE_TARGET"] = stateTargets[outputStateTargets[i]]
                            
                            
                            #
                            # prediction info
                            #
                            predShkpSpeechSeq = [coreqa_indexToWord[w] for w in predShkpSpeechSequences[i]]
                            
                            try:
                                predShkpSpeechLen = predShkpSpeechSeq.index("<eof>")
                                outShkpSpeech = " ".join(predShkpSpeechSeq[:predShkpSpeechLen])
                                
                            except:
                                predShkpSpeechLen = 0
                                outShkpSpeech = " ".join(predShkpSpeechSeq)
                            
                            
                            csvLogRows[i]["PRED_SHOPKEEPER_SPEECH_SEQUENCE"] = predShkpSpeechSeq
                            csvLogRows[i]["PRED_SHOPKEEPER_SPEECH"] = outShkpSpeech
                            csvLogRows[i]["PRED_OUTPUT_SHOPKEEPER_LOCATION"] = locations[predLocations[i]]
                            csvLogRows[i]["PRED_OUTPUT_SPATIAL_STATE"] = spatialStates[predSpatialStates[i]]
                            csvLogRows[i]["PRED_OUTPUT_STATE_TARGET"] = stateTargets[predStateTargets[i]]
                            
                            
                            maxPredDbFact = np.argmax(predDbFactMatchScores[i])
                            maxPredCamIndex = math.floor(maxPredDbFact / len(dbFieldnames))
                            maxPredAttrIndex = maxPredDbFact % len(dbFieldnames)
                            
                            csvLogRows[i]["PRED_DB_FACT_MAX"] = maxPredDbFact
                            csvLogRows[i]["PRED_DB_CAMERA_MAX"] = maxPredCamIndex
                            csvLogRows[i]["PRED_DB_ATTRIBUTE_MAX"] = maxPredAttrIndex
                            csvLogRows[i]["PRED_DB_VALUE_MAX"] = databases[int(csvLogRows[i]["DATABASE_ID"])][maxPredCamIndex][dbFieldnames[maxPredAttrIndex]]
                            
                            for j in range(len(cameras)*len(attributes)):
                                csvLogRows[i]["PRED_DB_FACT_{}_WEIGHT".format(j)] = predDbFactMatchScores[i][j][0]
                            
                            
                            #
                            # for computing accuracies
                            #
                            numTokensTotal = max(coreqa_outputSpeechSequenceLens[i], predShkpSpeechLen)
                            numTokensCorrect = 0
                            
                            for w in range(min(coreqa_outputSpeechSequenceLens[i], predShkpSpeechLen)):
                                if targShkpSpeechSeq[w] == predShkpSpeechSeq[w]:
                                    numTokensCorrect += 1
                            
                            percTokensCorrect = numTokensCorrect / numTokensTotal
                            speechSequenceAccuracies.append(percTokensCorrect)
                            
                            
                            locs_gt.append(csvLogRows[i]["TARG_OUTPUT_SHOPKEEPER_LOCATION"])
                            locs_pred.append(csvLogRows[i]["PRED_OUTPUT_SHOPKEEPER_LOCATION"])
                            
                            spatSts_gt.append(csvLogRows[i]["TARG_OUTPUT_SPATIAL_STATE"])
                            spatSts_pred.append(csvLogRows[i]["PRED_OUTPUT_SPATIAL_STATE"])
                            
                            stTargs_gt.append(csvLogRows[i]["TARG_OUTPUT_STATE_TARGET"])
                            stTargs_pred.append(csvLogRows[i]["PRED_OUTPUT_STATE_TARGET"])
                            
                            
                            if csvLogRows[i]["CURRENT_CAMERA_OF_CONVERSATION"] != "NONE":
                                camIndex_gt.append(cameras.index(csvLogRows[i]["CURRENT_CAMERA_OF_CONVERSATION"]))
                                camIndex_pred.append(maxPredCamIndex)
                            
                            # this only looks at cases where the customer asks about a feature, not when shkp introduces a feature/camera
                            if csvLogRows[i]["CUSTOMER_TOPIC"] != "NONE":
                                attrIndex_gt.append(dbFieldnames.index(csvLogRows[i]["CUSTOMER_TOPIC"]))
                                attrIndex_pred.append(maxPredAttrIndex)
                            
                            
                        #
                        # compute accuracies
                        #
                        speechSeqCorrAcc = np.mean(speechSequenceAccuracies)
                        locCorrAcc = accuracy_score(locs_gt, locs_pred)
                        spatStCorrAcc = accuracy_score(spatSts_gt, spatSts_pred)
                        stTargCorrAcc = accuracy_score(stTargs_gt, stTargs_pred)
                        camCorrAcc = accuracy_score(camIndex_gt, camIndex_pred)
                        attrCorrAcc = accuracy_score(attrIndex_gt, attrIndex_pred)
                        
                        
                        return csvLogRows, speechSeqCorrAcc, locCorrAcc, spatStCorrAcc, stTargCorrAcc, camCorrAcc, attrCorrAcc
                    
                    
                    csvLogRows = copy.deepcopy(interactions)
                    
                    csvLogRows, trainSpeechSeqCorrAcc, trainLocCorrAcc, trainSpatStCorrAcc, trainStTargCorrAcc, trainCamCorrAcc, trainAttrCorrAcc = evaluate_predictions_coreqa("TRAIN", trainIndices, csvLogRows)
                    
                    csvLogRows, valSpeechSeqCorrAcc, valLocCorrAcc, valSpatStCorrAcc, valStTargCorrAcc, valCamCorrAcc, valAttrCorrAcc = evaluate_predictions_coreqa("VAL", valIndices, csvLogRows)
                    
                    csvLogRows, testSpeechSeqCorrAcc, testLocCorrAcc, testSpatStCorrAcc, testStTargCorrAcc, testCamCorrAcc, testAttrCorrAcc = evaluate_predictions_coreqa("TEST", testIndices, csvLogRows)
                    
                    
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
                        writer.writerow([e,
                                         
                                         # training
                                         trainCostAve,
                                         trainCostStd,
                                         trainShopkeeperSpeechSequenceLossAve,
                                         trainShopkeeperSpeechSequenceLossStd,
                                         trainLocationLossAve,
                                         trainLocationLossStd,
                                         trainSpatialStateLossAve,
                                         trainSpatialStateLossStd,
                                         trainStateTargetLossAve,
                                         trainStateTargetLossStd,
                                         
                                         trainSpeechSeqCorrAcc,
                                         trainLocCorrAcc,
                                         trainSpatStCorrAcc,
                                         trainStTargCorrAcc,
                                         trainCamCorrAcc,
                                         trainAttrCorrAcc,
                                         
                                         # validation
                                         valCostAve,
                                         valCostStd,
                                         valShopkeeperSpeechSequenceLossAve,
                                         valShopkeeperSpeechSequenceLossStd,
                                         valLocationLossAve,
                                         valLocationLossStd,
                                         valSpatialStateLossAve,
                                         valSpatialStateLossStd,
                                         valStateTargetLossAve,
                                         valStateTargetLossStd,
                                         
                                         valSpeechSeqCorrAcc,
                                         valLocCorrAcc,
                                         valSpatStCorrAcc,
                                         valStTargCorrAcc,
                                         valCamCorrAcc,
                                         valAttrCorrAcc,
                                         
                                         # testing
                                         testCostAve,
                                         testCostStd,
                                         testShopkeeperSpeechSequenceLossAve,
                                         testShopkeeperSpeechSequenceLossStd,
                                         testLocationLossAve,
                                         testLocationLossStd,
                                         testSpatialStateLossAve,
                                         testSpatialStateLossStd,
                                         testStateTargetLossAve,
                                         testStateTargetLossStd,
                                         
                                         testSpeechSeqCorrAcc,
                                         testLocCorrAcc,
                                         testSpatStCorrAcc,
                                         testStTargCorrAcc,
                                         testCamCorrAcc,
                                         testAttrCorrAcc
                                         ])    
                    
                    # training
                    print("===== {} EPOCH {} LOSSES AND ACCURACIES=====".format(condition.upper(), e), flush=True, file=foldTerminalOutputStream)
                    tableData = []
                    
                    tableData.append(["CostAve", trainCostAve, valCostAve, testCostAve])
                    tableData.append(["CostStd", trainCostStd, valCostStd, testCostStd])
                    tableData.append(["ShopkeeperSpeechSequenceLossAve", trainShopkeeperSpeechSequenceLossAve, valShopkeeperSpeechSequenceLossAve, testShopkeeperSpeechSequenceLossAve])
                    tableData.append(["ShopkeeperSpeechSequenceLossStd", trainShopkeeperSpeechSequenceLossStd, valShopkeeperSpeechSequenceLossStd, testShopkeeperSpeechSequenceLossStd])
                    tableData.append(["LocationLossAve", trainLocationLossAve, valLocationLossAve, testLocationLossAve])
                    tableData.append(["LocationLossStd", trainLocationLossStd, valLocationLossStd, testLocationLossStd])
                    tableData.append(["SpatialStateLossAve", trainSpatialStateLossAve, valSpatialStateLossAve, testSpatialStateLossAve])
                    tableData.append(["SpatialStateLossStd", trainSpatialStateLossStd, valSpatialStateLossStd, testSpatialStateLossStd])
                    tableData.append(["StateTargetLossAve", trainStateTargetLossAve, valStateTargetLossAve, testStateTargetLossAve])
                    tableData.append(["StateTargetLossStd", trainStateTargetLossStd, valStateTargetLossStd, testStateTargetLossStd])
                    
                    tableData.append(["SpeechSeqCorrAcc", trainSpeechSeqCorrAcc, valSpeechSeqCorrAcc, testSpeechSeqCorrAcc])
                    tableData.append(["LocCorrAcc", trainLocCorrAcc, valLocCorrAcc, testLocCorrAcc])
                    tableData.append(["SpatStCorrAcc", trainSpatStCorrAcc, valSpatStCorrAcc, testSpatStCorrAcc])
                    tableData.append(["StTargCorrAcc", trainStTargCorrAcc, valStTargCorrAcc, testStTargCorrAcc])
                    tableData.append(["AttrCorrAcc", trainAttrCorrAcc, valAttrCorrAcc, testAttrCorrAcc])
                    tableData.append(["CamCorrAcc", trainCamCorrAcc, valCamCorrAcc, testCamCorrAcc])
                    
                    
                    print(tabulate(tableData, headers=["METRIC", "TRAINING", "VALIDATION", "TESTING"], floatfmt=".3f", tablefmt="grid"), flush=True, file=foldTerminalOutputStream)
                            
                    print("", flush=True, file=foldTerminalOutputStream)
                    
                
                #################################################################################################################
                # END COREQA RUN!
                #################################################################################################################
            
            
            
            print("Epoch {} time: {}s".format(e, round(time.time() - startTime, 2)), flush=True, file=sessionTerminalOutputStream)
            
    #
    # start parallel processing
    #
    if not RUN_PARALLEL:
        run_fold(randomSeed=0, foldId=0, gpu=0)
    
    else:
        processes = []
        
        for fold in range(numDatabases): #numDatabases
            process = Process(target=run_fold, args=[0, fold, gpuCount%NUM_GPUS]) # randomSeed, foldId, gpu
            process.start()
            processes.append(process)
            gpuCount += 1
        
        """
        for gpu in range(8):
            process = Process(target=run_fold, args=[0, gpu, gpu]) # randomSeed, foldId, gpu
            process.start()
            processes.append(process)
        
        for p in processes:
            p.join()
        """
    
    return gpuCount



#
# start here...
#

gpuCount = 0
#gpuCount = main(mainDir, "proposed", gpuCount)
#gpuCount = main(mainDir, "baseline1", gpuCount)
#gpuCount = main(mainDir, "copynet", gpuCount)
gpuCount = main(mainDir, "coreqa", gpuCount)






