'''
Created on April 9, 2019

@author: MalcolmD


modified from actionPrediction15

train the network to output shopkeeper action cluster IDs and DB camera indices
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

import tools



DEBUG = False


eosChar = "#"
goChar = "~"


cameras = ["CAMERA_1", "CAMERA_2", "CAMERA_3"]
attributes = ["camera_ID", "camera_name", "camera_type", "color", "weight", "preset_modes", "effects", "price", "resolution", "optical_zoom", "settings", "autofocus_points", "sensor_size", "ISO", "long_exposure"]

locations = ["CAMERA_1", "CAMERA_2", "CAMERA_3", "SERVICE_COUNTER"] # for the shopkeeper
spatialStates = ["WAITING", "FACE_TO_FACE", "PRESENT_X"]
stateTargets = ["CAMERA_1", "CAMERA_2", "CAMERA_3", "NONE"]


numTrainDbs = 10
batchSize = 64
embeddingSize = 100
numEpochs = 1000
evalEvery = 1
randomizeTrainingBatches = False

numInteractionsPerDb = 200

inputSeqLen = 10 #20
inputDim = 2217


previousSessionDir = None

dataDirectory = tools.dataDir+"2019-12-05_14-58-11_advancedSimulator9" # handmade databases, customer-driven interactions, deterministic introductions, crowdsourced shopkeeper utts
inputSequenceVectorDirectory = dataDirectory + "_input_sequence_vectors"
shopkeeperActionClusterFilename = inputSequenceVectorDirectory+"/shopkeeper_action_clusters.csv"
shopkeeperSpeechClusterFilename = tools.modelDir + "20191212 - modified_speech_clusters.csv"


if previousSessionDir != None:
    sessionDir = previousSessionDir
else:
    sessionDir = tools.create_session_dir("actionPrediction17_dbl")
    


def normalized_edit_distance(s1, s2):
    return editdistance.eval(s1, s2) / float(max(len(s1), len(s2)))



def compute_db_substring_match(groundTruthUtterances, predictedUtterances, shkpUttToDbEntryRange):
    
    subStringCharMatchAccuracies = []
    
    for i in range(len(groundTruthUtterances)):
        
        gtUtt = groundTruthUtterances[i]
        predUtt = predictedUtterances[i]
        
        if gtUtt in shkpUttToDbEntryRange and shkpUttToDbEntryRange[gtUtt] != "NA":
            
            # TODO: take this out later
            # this only looks at things from the price column
            #if "$" in gtUtt[shkpUttToDbEntryRange[gtUtt][0]:shkpUttToDbEntryRange[gtUtt][1]]:
                
            subStringCharMatchCount = 0.0
            subStringCharTotalCount = 0.0
            
            for j in range(shkpUttToDbEntryRange[gtUtt][0], shkpUttToDbEntryRange[gtUtt][1]):
                
                if gtUtt[j] == predUtt[j]:
                    subStringCharMatchCount += 1
                
                subStringCharTotalCount += 1
            
            
            subStringCharMatchAccuracies.append(subStringCharMatchCount / subStringCharTotalCount)
    
    return subStringCharMatchAccuracies



def compute_db_address_match(gtCamIndex, gtAttrIndex, predCamIndex, predAttIndex):
        
    camCorrect = []
    attrCorrect = []
    bothCorrect = []
    
    for i in range(len(gtCamIndex)):
        
        # make sure this is an instance that contains a DB string
        if gtCamIndex[i] != -1 and gtAttrIndex[i] != -1:
            
            camMatch = 0.0
            attrMatch = 0.0
            bothMatch = 0.0
            
            if gtCamIndex[i] == predCamIndex[i]:
                camMatch = 1.0
            
            if gtAttrIndex[i] == predAttIndex[i]:
                attrMatch = 1.0
                
            if camMatch and attrMatch:
                bothMatch = 1.0
                
            camCorrect.append(camMatch)
            attrCorrect.append(attrMatch)
            bothCorrect.append(bothMatch)
        
    return camCorrect, attrCorrect, bothCorrect


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


def get_input_output_strings_and_location_vectors(interactions, locationToIndex, spatialStateToIndex, stateTargetToIndex):
    
    inputStrings = []
    outputStrings = []
    
    inputCustomerLocations = []
    outputShopkeeperLocations = []
    
    outputSpatialStates = []
    outputStateTargets = []
    
    for turn in interactions:
        
        iString = turn["CUSTOMER_SPEECH"]
        oString = turn["SHOPKEEPER_SPEECH"]
        
        inputStrings.append(iString)
        outputStrings.append(oString)
        
        inputCustomerLocations.append(turn["CUSTOMER_LOCATION"])
        outputShopkeeperLocations.append(turn["OUTPUT_SHOPKEEPER_LOCATION"])
        
        outputSpatialStates.append(turn["OUTPUT_SPATIAL_STATE"])
        outputStateTargets.append(turn["OUTPUT_STATE_TARGET"])
        
        
    inputCustomerLocationVectors = one_hot_vectorize(inputCustomerLocations, locationToIndex)
    outputShopkeeperLocationVectors = one_hot_vectorize(outputShopkeeperLocations, locationToIndex)
    
    outputSpatialStateVectors = one_hot_vectorize(outputSpatialStates, spatialStateToIndex)
    outputStateTargetVectors = one_hot_vectorize(outputStateTargets, stateTargetToIndex)
    
    
    return inputStrings, outputStrings, inputCustomerLocationVectors, outputShopkeeperLocationVectors, outputSpatialStateVectors, outputStateTargetVectors


def one_hot_vectorize(labels, labelToIndex):
    
    oneHotEncodings = []
    
    for label in labels:
        vec = np.zeros(len(labelToIndex))
        vec[labelToIndex[label]] = 1
        oneHotEncodings.append(vec)
    
    return oneHotEncodings


def get_database_value_strings(database, fieldnames):
    
    valueStrings = []
    
    for row in database:
        
        rowStrings = []
        
        for col in fieldnames:
            rowStrings.append(row[col])
        
        valueStrings.append(rowStrings)
    
    return valueStrings


def get_db_content_lens(database, dbFieldnames):
    
    numCams = len(cameras)
    numFields = len(dbFieldnames) 
    
    contentLens = np.zeros((numCams, numFields))
    
    for i in range(numCams):
        for j in range(numFields):
            contentLens[i,j] = 1 if (len(database[i][dbFieldnames[j]]) > 0) else 0
    
    return contentLens


def run(gpu, seed, camTemp, attTemp, teacherForcingProb, sessionDir):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import learning3
    
    
    #
    # setup log files
    #
    sessionIdentifier = "rs{}_ct{}_at{}".format(seed, camTemp, attTemp)
    
    sessionDir = tools.create_directory(sessionDir + "/" + sessionIdentifier)
    
    sessionLogFile = sessionDir + "/session_log_{}.csv".format(sessionIdentifier)
    sessionTerminalOutputLogFile = sessionDir + "/terminal_output_log_{}.txt".format(sessionIdentifier)
    
    
    if DEBUG:
        sessionTerminalOutputStream = sys.stdout
    else:
        sessionTerminalOutputStream = open(sessionTerminalOutputLogFile, "a")
    
    
    
    #
    # load the shopkeeper speech and action clusters
    #
    shkpUttToSpeechClustId, shkpSpeechClustIdToRepUtt, speechClustIdToShkpUtts, junkSpeechClusterIds = tools.load_shopkeeper_speech_clusters(shopkeeperSpeechClusterFilename)
    numSpeechClusters = len(speechClustIdToShkpUtts)
    print(numSpeechClusters, "shopkeeper speech clusters")
    
    #shkpActionIdToTuple, tupleToShkpActionId = tools.load_shopkeeper_action_clusters(shopkeeperActionClusterFilename)
    #numActionClusters = len(shkpActionIdToTuple)
    #print(numActionClusters, "shopkeeper action clusters")
    
    
    
    #
    # load the simulated interactions and databases
    #
    print("loading data...", flush=True, file=sessionTerminalOutputStream)
    
    filenames = os.listdir(dataDirectory)
    filenames.sort()
    
    databaseFilenamesAll = [dataDirectory+"/"+fn for fn in filenames if "handmade_database" in fn]
    interactionFilenamesAll = [dataDirectory+"/"+fn for fn in filenames if "withsymbols" in fn]
    
    databaseFilenames = databaseFilenamesAll[:numTrainDbs+1]
    interactionFilenames = interactionFilenamesAll[:numTrainDbs+1]
    
    
    databases = []
    databaseConentLengths = []
    databaseIds = []
    dbFieldnames = None # these should be the same for all DBs
    
    for dbFn in databaseFilenames:
        db, dbFieldnames = read_database_file(dbFn)
        
        databaseIds.append(dbFn.split("_")[-1].split(".")[0])
        databaseConentLengths.append(get_db_content_lens(db, dbFieldnames))
        databases.append(db)
    
    numDatabases = len(databases)
    
    interactions = []
    datasetSizes = []
    gtDatabaseCameras = []
    gtDatabaseAttributes = []
    
    shkpUttToDbEntryRange = {}
    
    for i in range(len(interactionFilenames)):
        iFn = interactionFilenames[i]
        
        inters, sutder, gtDbCamera, gtDbAttribute = read_simulated_interactions(iFn, dbFieldnames, numInteractionsPerDb)
        
        interactions.append(inters)
        datasetSizes.append(len(inters))
        
        gtDatabaseCameras.append(gtDbCamera)
        gtDatabaseAttributes.append(gtDbAttribute)
        
        # combine the three dictionaries into one
        shkpUttToDbEntryRange = {**shkpUttToDbEntryRange, **sutder}
    
    
    #
    # load the input sequence vectors and outputs
    #
    inputSequenceVectors = []
    #outputActionIds = []
    outputSpeechClusterIds = []
    outputShopkeeperLocations = []
    outputSpatialStates = []
    outputStateTargets = []
    outputCameraIndices = []
    outputAttributeIndices = []
    
    for i in range(len(interactions)):
        
        # input sequence vectors
        iFn = interactionFilenames[i]
        isvFn = iFn.split("/")[-1][:-4] + "_input_sequence_vectors_sl{}_dim{}.npy".format(inputSeqLen, inputDim)
        
        inVecSeqs = np.load(inputSequenceVectorDirectory+"/"+isvFn)
        
        inputSequenceVectors.append(list(inVecSeqs[:len(interactions[i]), :, :]))
        del inVecSeqs
        
        # output shopkeeper action ID
        #fn = iFn.split("/")[-1][:-4] + "_output_action_ids.npy"
        #outAct = np.load(inputSequenceVectorDirectory+"/"+fn)
        #outputActionIds.append(outAct)
        
        # speech clusters
        fn = iFn.split("/")[-1][:-4] + "_output_speech_cluster_ids.npy"
        outputSpeechClusterIds.append(np.load(inputSequenceVectorDirectory+"/"+fn))
        
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
        
        # output DB camera index
        fn = iFn.split("/")[-1][:-4] + "_output_camera_indices.npy"
        outputCameraIndices.append(np.load(inputSequenceVectorDirectory+"/"+fn))
        
        # output DB attribute index
        fn = iFn.split("/")[-1][:-4] + "_output_attribute_indices.npy"
        outputAttributeIndices.append(np.load(inputSequenceVectorDirectory+"/"+fn))
        
        
        """
        # get the speech cluster id, location, and spatial info from the action id
        outSpeech = []
        outLoc = []
        outSpatSt= []
        outStTarg = []
        
        for actClustId in outAct:
            outSpeech.append(shkpActionIdToTuple[actClustId][0])
            outLoc.append(locations.index(shkpActionIdToTuple[actClustId][1]))
            outSpatSt.append(spatialStates.index(shkpActionIdToTuple[actClustId][2]))
            outStTarg.append(stateTargets.index(shkpActionIdToTuple[actClustId][3]))
        
        outputSpeechClusterIds.append(outSpeech)
        outputShopkeeperLocations.append(outLoc)
        outputSpatialStates.append(outSpatSt)
        outputStateTargets.append(outStTarg)
        """
    
    
    #
    # count number of occurrences of each speech cluster in the training dataset
    # this is for weighting the losses
    #
    speechClustCounts = {}
    
    for i in range(numTrainDbs):
        for j in range(len(outputSpeechClusterIds[i])):
            speechClustId = outputSpeechClusterIds[i][j]
            
            if speechClustId not in speechClustCounts:
                speechClustCounts[speechClustId] = 0
            speechClustCounts[speechClustId] += 1
    
    speechClustWeights = [None] * numSpeechClusters
    
    numSamples = sum(len(x) for x in outputSpeechClusterIds[:numTrainDbs])
    
    for clustId in speechClustIdToShkpUtts:
        # as in scikit learn - The “balanced” heuristic is inspired by Logistic Regression in Rare Events Data, King, Zen, 2001.
        speechClustWeights[clustId] = numSamples / (numSpeechClusters * speechClustCounts[clustId])
    
    if None in speechClustWeights:
        print("WARNING: missing training weight for shopkeeper speech cluster!")
    
    
    #
    # count the number of occurrences of each attribute index target in the training dataset
    # this is for weighting the losses
    #
    outputAttributeIndexCounts = {}
    numSamples = 0
    
    for a in attributes:
        outputAttributeIndexCounts[a] = 0
    
    for i in range(numTrainDbs):
        for j in range(len(outputAttributeIndices[i])):
            
            if sum(outputAttributeIndices[i][j]) < 1:
                continue
            
            numSamples += 1
            
            for k in range(len(attributes)):
                outputAttributeIndexCounts[attributes[k]] += outputAttributeIndices[i][j][k] # value will be either 0 or 1
    
    # treat each attribute index sigmoid as a binary classifier, so need a weight for each class
    attributeIndexWeights0 = [None] * len(attributes)
    attributeIndexWeights1 = [None] * len(attributes)
    
    
    for a in outputAttributeIndexCounts:
        if outputAttributeIndexCounts[a] == 0:
            outputAttributeIndexCounts[a] = 1 # just to make sure none are 0
    
        attributeIndexWeights0[attributes.index(a)] = numSamples / (2 * (numSamples - outputAttributeIndexCounts[a]))
        attributeIndexWeights1[attributes.index(a)] = numSamples / (2 * outputAttributeIndexCounts[a])
    
    
        
    def realize_actions(splitInteractions, spliActionIdPreds, splitCamIndexPreds, splitLocationsPreds, splitSpatialStatePreds, splitStateTargetsPreds):
        """transform a predicted action and camera index into an utterance"""
        
        outputSpeech = []
        outputSpeechTemplates = []
        outputDbIndices = []
        outputDbContents = []
        outputShopkeeperLocations = []
        outputSpatialStates = []
        outputStateTargets = []
        
        
        for i in range(len(splitInteractions)):
            outDbIndx = []
            outDbCont = []
            
            #actionTup =  shkpActionIdToTuple[spliActionIdPreds[i]]
            camIndex = splitCamIndexPreds[i]
            
            try:
                shkpSpeechTemplate = shkpSpeechClustIdToRepUtt[spliActionIdPreds[i]]
            except Exception as e:
                traceback.print_tb(e.__traceback__)
                print(e)
                shkpSpeechTemplate = "ERROR: No representative utterance found for cluster {}.".format(spliActionIdPreds[i])
            
            outShkpSpeech = shkpSpeechTemplate
            
            # use the shopkeeper locations, spatial states and targets from the action clusters
            #outShkpLoc = actionTup[1]
            #outShkpSpatSt = actionTup[2]
            #outShkpStTarg = actionTup[3]
            
            # use the outputs of the neural network for these things
            outShkpLoc = splitLocationsPreds[i]
            outShkpSpatSt = splitSpatialStatePreds[i]
            outShkpStTarg = splitStateTargetsPreds[i]
            
            if camIndex != 3: # 3 is NONE
                
                # find any DB contents symbols int he speech template
                dbId = splitInteractions[i]["DATABASE_ID"]
                cameraInfo = databases[int(dbId)][camIndex]
                
                for j in range(len(dbFieldnames)):
                    attr = dbFieldnames[j]
                    symbol = "<{}>".format(attr.lower())
                    
                    if symbol in outShkpSpeech:
                        outShkpSpeech = outShkpSpeech.replace(symbol, cameraInfo[attr])
                        
                        outDbIndx.append((camIndex, j))
                        outDbCont.append(cameraInfo[attr])
            
            
            outputSpeech.append(outShkpSpeech)
            outputSpeechTemplates.append(shkpSpeechTemplate)
            outputDbIndices.append(outDbIndx)
            outputDbContents.append(outDbCont)
            outputShopkeeperLocations.append(outShkpLoc)
            outputSpatialStates.append(outShkpSpatSt)
            outputStateTargets.append(outShkpStTarg)
        
        return outputSpeech, outputSpeechTemplates, outputDbIndices, outputDbContents, outputShopkeeperLocations, outputSpatialStates, outputStateTargets
    
    
    #
    # split into training and testing sets
    #
    def prepare_split(startDbIndex, stopDbIndex, splitName):
        """input which DB to start with and which DB to end with + 1"""
        
        uniqueId = 0
        
        for i in range(startDbIndex, stopDbIndex):    
            for j in range(len(interactions[i])):
                interactions[i][j]["SET"] = splitName
                interactions[i][j]["ID"] = uniqueId
                uniqueId += 1
        
        splitInteractions = []
        splitInputSequenceVectors = []
        #splitOutputActionIds = []
        splitOutputCameraIndices = []
        splitOutputAttributeIndices = []
        splitGtDatabasebCameras = []
        splitGtDatabaseAttributes = []
        splitOutputMasks = []
        
        splitOutputSpeechClusterIds = []
        splitOutputShopkeeperLocations = []
        splitOutputSpatialStates = []
        splitOutputStateTargets = []
        
        splitDbContentLengths = []
        
        
        for i in range(startDbIndex, stopDbIndex):
            splitInteractions += interactions[i]
            splitInputSequenceVectors += inputSequenceVectors[i]
            #splitOutputActionIds += outputActionIds[i].tolist()
            splitOutputCameraIndices += outputCameraIndices[i].tolist()
            splitOutputAttributeIndices += outputAttributeIndices[i].tolist()
            splitGtDatabasebCameras += gtDatabaseCameras[i]
            splitGtDatabaseAttributes += gtDatabaseAttributes[i]
            
            splitOutputSpeechClusterIds += outputSpeechClusterIds[i].tolist()
            splitOutputShopkeeperLocations += outputShopkeeperLocations[i]
            splitOutputSpatialStates += outputSpatialStates[i]
            splitOutputStateTargets += outputStateTargets[i]
            
            splitDbContentLengths += [databaseConentLengths[i]] * len(interactions[i])
            
            
            # to ignore outputs with junk shopkpeeper speech clusters during training
            splitOutputMasks += [0 if (x in junkSpeechClusterIds) else 1 for x in outputSpeechClusterIds[i].tolist()]
        
        return splitInteractions, splitInputSequenceVectors, splitOutputCameraIndices, splitOutputAttributeIndices, splitOutputSpeechClusterIds, splitOutputShopkeeperLocations, splitOutputSpatialStates, splitOutputStateTargets, splitGtDatabasebCameras, splitGtDatabaseAttributes, splitOutputMasks, splitDbContentLengths
    
    
    # training
    trainInteractions, trainInputSequenceVectors, trainOutputCameraIndices, trainOutputAttributeIndices, trainOutputSpeechClusterIds, trainOutputShopkeeperLocations, trainOutputSpatialStates, trainOutputStateTargets, trainGtDatabasebCameras, trainGtDatabaseAttributes, trainOutputMasks, trainDbContentLengths = prepare_split(0, numTrainDbs, "Train")
    
    # testing
    testInteractions, testInputSequenceVectors, testOutputCameraIndices, testOutputAttributeIndices, testOutputSpeechClusterIds, testOutputShopkeeperLocations, testOutputSpatialStates, testOutputStateTargets, testGtDatabasebCameras, testGtDatabaseAttributes, testOutputMasks, testDbContentLengths = prepare_split(numTrainDbs, numDatabases, "Test")
    
    numTrainExamples = len(trainOutputSpeechClusterIds)
    numTestExamples = len(testOutputSpeechClusterIds)
    
    print(numTrainExamples, "training examples", flush=True, file=sessionTerminalOutputStream)
    print(numTestExamples, "testing examples", flush=True, file=sessionTerminalOutputStream)
    print(inputDim, "input utterance vector size", flush=True, file=sessionTerminalOutputStream)
    print(inputSeqLen, "input sequence length", flush=True, file=sessionTerminalOutputStream)
    
    
    #
    # setup the model
    #
    print("setting up the model...", flush=True, file=sessionTerminalOutputStream)
    
    
    learner = learning3.CustomNeuralNetwork(inputDim=inputDim, 
                                            inputSeqLen=inputSeqLen, 
                                            numOutputClasses=numSpeechClusters,
                                            numUniqueCams=len(cameras),
                                            numAttributes=len(dbFieldnames),
                                            numLocations = len(locations), # cam 1, 2, 3, service counter
                                            numSpatialStates = len(spatialStates), # f2f, preesnt x, waiting
                                            numStateTargets = len(stateTargets), # cam 1, 2, 3, NONE
                                            batchSize=batchSize, 
                                            embeddingSize=embeddingSize,
                                            camTemp=camTemp,
                                            seed=seed,
                                            speechClusterWeights=speechClustWeights,
                                            attributeIndexWeights=[attributeIndexWeights0, attributeIndexWeights1]
                                            )
    
    
    #
    # load previous session if we're not starting a new one
    #
    if previousSessionDir != None:
        print("loading previous session...", flush=True, file=sessionTerminalOutputStream)
        
        # find where the run left off
        filenames = os.listdir(sessionDir)
        
        checkpointDirs = []
        
        for fn in filenames:
            try:
                checkpointDirs.append(int(fn))
            except:
                pass
        
        lastCheckpointDir = max(checkpointDirs)
        
        learner.load("{}/{}/saved_session".format(sessionDir, lastCheckpointDir))
        runFromEpoch = int(lastCheckpointDir) + 1
    
    else: 
        runFromEpoch = 0
    
    
    #
    # train and test
    # 
    print("training and testing...", flush=True, file=sessionTerminalOutputStream)
    
    # for faster testing...
    """
    trainInputUttVectors = trainInputUttVectors[:1500]
    trainOutputIndexLists = trainOutputIndexLists[:1500]
    trainDbVectors = trainDbVectors[:1500]
    
    testInputUttVectors = testInputUttVectors[:1500]
    testOutputIndexLists = testOutputIndexLists[:1500]
    testDbVectors = testDbVectors[:1500]
    """
    
    
    trainBatchEndIndices = list(range(batchSize, numTrainExamples, batchSize))
    testBatchEndIndices = list(range(batchSize, numTestExamples, batchSize))
    
    
    camerasOfInterest = ["CAMERA_1", "CAMERA_2", "CAMERA_3"]
    featuresOfInterest = ["price", "camera_type", "color", "weight", "preset_modes", "effects", "resolution", "optical_zoom", "settings", "autofocus_points", "sensor_size", "ISO", "long_exposure"]
    
    #
    trainInterestingInstances = []
    
    for f in featuresOfInterest:
        for c in camerasOfInterest:
            for i in range(numTrainDbs):
                for j in range(datasetSizes[i]):
                    if interactions[i][j]["CUSTOMER_TOPIC"] == f and interactions[i][j]["CURRENT_CAMERA_OF_CONVERSATION"] == c:
                        index = j + sum(len(l) for l in interactions[:i])
                        
                        # the training data that doesn't fit into one of the batches will not be included, so skip interesting instance that fall in this range
                        if index < trainBatchEndIndices[-1]: 
                            trainInterestingInstances.append((index, "DB{} {} {} {}".format(databaseIds[i], c, f, databases[i][cameras.index(c)][f])))
                        break
        
    #
    testInterestingInstances = []
    
    for f in featuresOfInterest:
        for c in camerasOfInterest:
            for i in range(numTrainDbs, numDatabases):
                for j in range(datasetSizes[i]):
                    if interactions[i][j]["CUSTOMER_TOPIC"] == f and interactions[i][j]["CURRENT_CAMERA_OF_CONVERSATION"] == c:
                        index = j
                        
                        # same as above
                        if index < testBatchEndIndices[-1]: 
                            testInterestingInstances.append((index, "DB{} {} {} {}".format(databaseIds[i], c, f, databases[i][cameras.index(c)][f])))
                        break
    
    
    # write header in csv log file
    with open(sessionLogFile, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch",
                         "Teacher Forcing Probability",
                         "Train Cost Ave ({})".format(sessionIdentifier), 
                         "Train Cost SD ({})".format(sessionIdentifier), 
                         "Train Action ID Correct ({})".format(sessionIdentifier), 
                         "Train Camera Index Correct ({})".format(sessionIdentifier),
                         "Train Attribute Index Correct ({})".format(sessionIdentifier), 
                         "Train Both Correct ({})".format(sessionIdentifier),
                         "Train Location Correct ({})".format(sessionIdentifier),
                         "Train Spatial State Correct ({})".format(sessionIdentifier),
                         "Train State Target Correct ({})".format(sessionIdentifier),
                         
                         "Test Cost Ave({})".format(sessionIdentifier),
                         "Test Cost SD ({})".format(sessionIdentifier), 
                         "Test Action ID Correct ({})".format(sessionIdentifier), 
                         "Test Camera Index Correct ({})".format(sessionIdentifier), 
                         "Test Attribute Index Exact Match ({})".format(sessionIdentifier),
                         "Test Attribute Index Jaccard Index ({})".format(sessionIdentifier),
                         "Test Both Correct ({})".format(sessionIdentifier),
                         "Test Location Correct ({})".format(sessionIdentifier),
                         "Test Spatial State Correct ({})".format(sessionIdentifier),
                         "Test State Target Correct ({})".format(sessionIdentifier),
                         
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
                      "PRED_OUTPUT_STATE_TARGET"]
    
    
    
    def evaluate_split(splitName,
                       splitInputSequenceVectors,
                       splitOutputSpeechClusterIds,
                       splitOutputCameraIndices,
                       splitOutputAttributeIndices,
                       splitOutputShopkeeperLocations, 
                       splitOutputSpatialStates, 
                       splitOutputStateTargets,
                       splitOutputMasks,
                       splitDatabaseContentLengths,
                       splitBatchEndIndices, 
                       splitInterestingInstances,
                       splitInteractions,
                       splitGtDatabasebCameras,
                       splitGtDatabaseAttributes,
                       splitCostAve,
                       splitCostStd
                       ):
        
        splitSpeechClusterIdPreds = []
        splitCameraIndexPreds = []
        splitCameraIndexPredsNoNone = []
        splitAttrIndexPreds = []
        
        splitUttPreds = []
        splitSpeechTemplates = []
        splitDbIndexPreds = []
        splitDbContentPreds = []
        splitLocPreds = []
        splitSpatStatePreds = []
        splitStateTargPreds = []
        
        splitSpeechClustIdTargets = []
        splitCameraIndexTargets = []
        splitAttributeIndexTargets = []
        
        splitLocTargets = []
        splitSpatStTargets = []
        splitStTargTargets = []
        
        splitOutMask = []
        
        
        for i in splitBatchEndIndices:
            predShkpSpeechClustID, predCameraIndicesRaw, predAttrIndicesRaw, predLocations, predSpatialStates, predStateTargets = learner.predict(splitInputSequenceVectors[i-batchSize:i],
                                                                                                                                                          splitOutputSpeechClusterIds[i-batchSize:i],
                                                                                                                                                          splitOutputCameraIndices[i-batchSize:i],
                                                                                                                                                          splitOutputAttributeIndices[i-batchSize:i],
                                                                                                                                                          splitOutputShopkeeperLocations[i-batchSize:i], 
                                                                                                                                                          splitOutputSpatialStates[i-batchSize:i], 
                                                                                                                                                          splitOutputStateTargets[i-batchSize:i],
                                                                                                                                                          splitOutputMasks[i-batchSize:i],
                                                                                                                                                          splitDatabaseContentLengths[i-batchSize:i])
            
            #predCameraIndicesArgmax = np.argmax(predCameraIndicesSoftmax, axis=1) # includes NONE
            #predCameraIndicesArgmaxNoNone = np.argmax(predCameraIndicesSoftmax[:,:3], axis=1) # does not include NONE
            
            #predAttrIndicesArgmax = np.argmax(predAttrIndicesSoftmax, axis=1)
            
            predCameraIndices = []
            predCameraIndicesNoNone = []
            predAttributeIndices = []
            
            
            for j in range(predCameraIndicesRaw.shape[0]):
                
                camPred = predCameraIndicesRaw[j,:]
                camPredArgmax = np.argmax(camPred)
                
                
                if camPred[camPredArgmax] > 0.5:
                    predCameraIndices.append(camPredArgmax)
                else:
                    predCameraIndices.append("")
                
                predCameraIndicesNoNone.append(camPredArgmax)
                
                
                attrPred = predAttrIndicesRaw[j,:]
                attrPred = np.where(attrPred > 0.5)[0].tolist() # can take more than one attribute
                
                predAttributeIndices.append(attrPred) 
            
            
            
            batchSplitUttPreds, batchSplitSpeechTemplates, batchSplitDbIndexPreds, batchSplitDbContentPreds, batchSplitLocPreds, batchSplitSpatStatePreds, batchSplitStateTargPreds = realize_actions(splitInteractions[i-batchSize:i], 
                                                                                                                                                                                                      predShkpSpeechClustID, 
                                                                                                                                                                                                      predCameraIndicesNoNone,
                                                                                                                                                                                                      predLocations, 
                                                                                                                                                                                                      predSpatialStates, 
                                                                                                                                                                                                      predStateTargets)
            
            
            splitSpeechClusterIdPreds.append(predShkpSpeechClustID)
            splitCameraIndexPreds.append(predCameraIndices)
            splitCameraIndexPredsNoNone.append(predCameraIndicesNoNone)
            splitAttrIndexPreds += predAttributeIndices
            splitUttPreds.append(batchSplitUttPreds)
            splitSpeechTemplates.append(batchSplitSpeechTemplates)
            splitDbIndexPreds += batchSplitDbIndexPreds
            splitDbContentPreds += batchSplitDbContentPreds
            splitLocPreds.append(batchSplitLocPreds)
            splitSpatStatePreds.append(batchSplitSpatStatePreds)
            splitStateTargPreds.append(batchSplitStateTargPreds)
            splitSpeechClustIdTargets.append(splitOutputSpeechClusterIds[i-batchSize:i])
            splitCameraIndexTargets.append(splitOutputCameraIndices[i-batchSize:i])
            splitAttributeIndexTargets.append(splitOutputAttributeIndices[i-batchSize:i])
            splitLocTargets += splitOutputShopkeeperLocations[i-batchSize:i]
            splitSpatStTargets += splitOutputSpatialStates[i-batchSize:i]
            splitStTargTargets += splitOutputStateTargets[i-batchSize:i]
            splitOutMask += splitOutputMasks[i-batchSize:i]
            
        
        splitSpeechClusterIdPreds = np.concatenate(splitSpeechClusterIdPreds)
        splitCameraIndexPreds = np.concatenate(splitCameraIndexPreds)
        splitCameraIndexPredsNoNone = np.concatenate(splitCameraIndexPredsNoNone)
        #splitAttrIndexPreds = np.concatenate(splitAttrIndexPreds)
        splitUttPreds = np.concatenate(splitUttPreds)
        splitSpeechTemplates = np.concatenate(splitSpeechTemplates)
        #splitDbIndexPreds = np.concatenate(splitDbIndexPreds)
        #splitDbContentPreds = np.concatenate(splitDbContentPreds)
        splitLocPreds = np.concatenate(splitLocPreds)
        splitSpatStatePreds = np.concatenate(splitSpatStatePreds)
        splitStateTargPreds = np.concatenate(splitStateTargPreds)
        splitSpeechClustIdTargets = np.concatenate(splitSpeechClustIdTargets)
        splitCameraIndexTargets = np.concatenate(splitCameraIndexTargets)
        splitAttributeIndexTargets = np.concatenate(splitAttributeIndexTargets)
        
        
        splitActionIdCorrAcc = accuracy_score(splitSpeechClustIdTargets, splitSpeechClusterIdPreds, splitOutMask)
        splitLocCorrAcc = accuracy_score(splitLocTargets, splitLocPreds, splitOutMask)
        splitSpatStCorrAcc = accuracy_score(splitSpatStTargets, splitSpatStatePreds, splitOutMask)
        splitStTargCorrAcc = accuracy_score(splitStTargTargets, splitStateTargPreds, splitOutMask)
        
        
        #
        # compute prediction accuracies
        #
        attrTargStrings = []
        attrPredStrings = []
        attrTargVecs = []
        attrPredVecs = []
        attrJunkClustMask = []
        
        for i in range(len(splitAttrIndexPreds)):
            
            if sum(splitAttributeIndexTargets[i]) < 1:
                continue # ignore samples where there is no attribute label
            
            attrTargs = np.where(splitAttributeIndexTargets[i] == 1)[0].tolist()
            attrTargStrings.append("-".join([str(a) for a in attrTargs]))
            attrTargVecs.append(splitAttributeIndexTargets[i])
            
            attrPredStrings.append("-".join([str(a) for a in splitAttrIndexPreds[i]]))
            
            attrPredVec = np.zeros(len(dbFieldnames))
            attrPredVec[splitAttrIndexPreds[i]] = 1
            
            attrPredVecs.append(attrPredVec)
            
            attrJunkClustMask.append(splitOutMask[i])
        
        splitAttrExactMatch = accuracy_score(attrTargStrings, attrPredStrings)
        splitAttrJaccardIndex = jaccard_score(np.asarray(attrTargVecs), np.asarray(attrPredVecs), average="samples")
        
        camTargStrings = []
        camPredStrings = []
        camJunkClustMask = []
        
        for i in range(len(splitCameraIndexTargets)):
            
            if sum(splitCameraIndexTargets[i]) < 1:
                continue # ignore samples where there is no camera label
            
            camTarg = np.where(splitCameraIndexTargets[i] == 1)[0]
            
            if len(camTarg) > 0:
                camTargStrings.append(str(camTarg[0]))
            else:
                camTargStrings.append("")
            
            camPredStrings.append(splitCameraIndexPreds[i])
            
            camJunkClustMask.append(splitOutMask[i])
        
        
        splitCamCorrAcc = accuracy_score(camTargStrings, camPredStrings, camJunkClustMask)
        
        
        #bothTargs = ["{}={}".format(splitSpeechClustIdTargets[i], camTargStrings[i]) for i in range(len(splitSpeechClustIdTargets))]
        #bothPreds = ["{}={}".format(splitSpeechClusterIdPreds[i], splitCameraIndexPreds[i]) for i in range(len(splitSpeechClusterIdPreds))]
        
        splitBothCorrAcc = -1.0 #accuracy_score(bothTargs, bothPreds)
        
        print("{} DB addressing correctness: speech cluster {:.3}, cam. {:.3}, attr. {:.3}, {:.3}, both {:.3}, shkp loc {:.3}, spat st. {:.3} st targ. {:.3}".format(splitName, 
                                                                                                                               splitActionIdCorrAcc, 
                                                                                                                               splitCamCorrAcc, 
                                                                                                                               splitAttrExactMatch,
                                                                                                                               splitAttrJaccardIndex,
                                                                                                                               splitBothCorrAcc,
                                                                                                                               splitLocCorrAcc,
                                                                                                                               splitSpatStCorrAcc,                                                                                                                     
                                                                                                                               splitStTargCorrAcc
                                                                                                                               ), flush=True, file=sessionTerminalOutputStream)
        
        
        #
        # save all predictions to file
        #
        with open(sessionDir+"/{:}_all_outputs.csv".format(e), "a", newline="") as csvfile:
            
            writer = csv.DictWriter(csvfile, interactionsFieldnames)
            writer.writeheader()
            
            for i in range(splitBatchEndIndices[-1]):
                row = splitInteractions[i]
                
                # add the important prediction information to the row
                row["TARG_SHOPKEEPER_SPEECH_CLUSTER_ID"] = splitSpeechClustIdTargets[i]
                row["TARG_SHOPKEEPER_SPEECH_CLUSTER_ID_IS_JUNK"] = 1 if splitSpeechClustIdTargets[i] in junkSpeechClusterIds else 0
                row["TARG_OUTPUT_CAMERA_INDEX"] = splitCameraIndexTargets[i]
                row["TARG_ATTRIBUTE_INDEX"] = np.where(splitAttributeIndexTargets[i] == 1)[0].tolist()
                
                
                if splitSpeechClustIdTargets[i] not in junkSpeechClusterIds:
                    row["TARG_SHOPKEEPER_SPEECH_TEMPLATE"] = shkpSpeechClustIdToRepUtt[splitSpeechClustIdTargets[i]]
                else:
                    try:
                        row["TARG_SHOPKEEPER_SPEECH_TEMPLATE"] = shkpSpeechClustIdToRepUtt[splitSpeechClustIdTargets[i]]
                    except:
                        row["TARG_SHOPKEEPER_SPEECH_TEMPLATE"] = "THIS_JUNK_CLUST_HAS_NONE"
                
                
                row["TARG_OUTPUT_SHOPKEEPER_LOCATION"] = locations[splitOutputShopkeeperLocations[i]]
                row["TARG_OUTPUT_SPATIAL_STATE"] = spatialStates[splitOutputSpatialStates[i]]
                row["TARG_OUTPUT_STATE_TARGET"] = stateTargets[splitOutputStateTargets[i]]
                
                row["PRED_OUTPUT_SHOPKEEPER_SPEECH_CLUSTER_ID"] = splitSpeechClusterIdPreds[i]
                row["PRED_OUTPUT_CAMERA_INDEX"] = splitCameraIndexPreds[i]
                row["PRED_OUTPUT_CAMERA_INDEX_NO_NONE"] = splitCameraIndexPredsNoNone[i]
                row["PRED_ATTRIBUTE_INDEX"] = splitAttrIndexPreds[i]
                row["PRED_SHOPKEEPER_SPEECH"] = splitUttPreds[i]
                row["PRED_SHOPKEEPER_SPEECH_TEMPLATE"] = splitSpeechTemplates[i]
                row["PRED_DB_INDICES"] = splitDbIndexPreds[i]
                row["PRED_DB_CONTENTS"] = splitDbContentPreds[i]
                row["PRED_OUTPUT_SHOPKEEPER_LOCATION"] = locations[splitLocPreds[i]]
                row["PRED_OUTPUT_SPATIAL_STATE"] = spatialStates[splitSpatStatePreds[i]]
                row["PRED_OUTPUT_STATE_TARGET"] = stateTargets[splitStateTargPreds[i]]
                
                writer.writerow(row)
        
        
        return splitActionIdCorrAcc, splitCamCorrAcc, splitAttrExactMatch, splitAttrJaccardIndex, splitBothCorrAcc, splitLocCorrAcc, splitSpatStCorrAcc, splitStTargCorrAcc
    
    
    
    
    for e in range(runFromEpoch, numEpochs):
        
        startTime = time.time()
        
        #teacherForcingProb = 0.6 #1.0 - 1.0 / (1.0 + np.exp( - (e-200.0)/10.0))
        
        
        if e == 200:
            print("reinitializing the speech and spatial output layer weights", flush=True, file=sessionTerminalOutputStream)
            learner.reinitialize_speech_and_spatial_output_vars()
            
            #print("setting to use the sharpened softmax addressing", flush=True, file=sessionTerminalOutputStream)
            #sharpeningCoefficient = 1.0
            
            #print("reinitializing the decoding weights", flush=True, file=sessionTerminalOutputStream)
            #learner.reinitialize_decoding_weights()
            
            print("resetting the optimizer", flush=True, file=sessionTerminalOutputStream)
            learner.reset_optimizer()
            
            #print("using train_op_2 (for only the decoding parts of the network and not the addressing parts)", flush=True, file=sessionTerminalOutputStream)
            #learner.set_train_op(2)
        
        
        
        
        #
        # train
        #
        trainCosts = []
        
        # shuffle the training data
        
        temp = list(zip(trainInteractions, trainInputSequenceVectors, trainOutputCameraIndices, trainOutputAttributeIndices,
                        trainOutputSpeechClusterIds, trainOutputShopkeeperLocations, trainOutputSpatialStates, trainOutputStateTargets,
                        trainGtDatabasebCameras, trainGtDatabaseAttributes, trainOutputMasks,
                        trainDbContentLengths))
        
        if randomizeTrainingBatches:
            random.shuffle(temp)
        trainInteractions_shuf, trainInputSequenceVectors_shuf, trainOutputCameraIndices_shuf, trainOutputAttributeIndices_shuf, trainOutputSpeechClusterIds_shuf, trainOutputShopkeeperLocations_shuf, trainOutputSpatialStates_shuf, trainOutputStateTargets_shuf, trainGtDatabasebCameras_shuf, trainGtDatabaseAttributes_shuf, trainOutputMasks_shuf, trainDbContentLengths_shuf = zip(*temp)
        
        
        for i in trainBatchEndIndices:
            
            batchTrainCost = learner.train(trainInputSequenceVectors_shuf[i-batchSize:i],
                                           trainOutputSpeechClusterIds_shuf[i-batchSize:i], # trainOutputActionIds_shuf[i-batchSize:i],
                                           trainOutputCameraIndices_shuf[i-batchSize:i],
                                           trainOutputAttributeIndices_shuf[i-batchSize:i],
                                           trainOutputShopkeeperLocations_shuf[i-batchSize:i],
                                           trainOutputSpatialStates_shuf[i-batchSize:i],
                                           trainOutputStateTargets_shuf[i-batchSize:i],                                           
                                           trainOutputMasks_shuf[i-batchSize:i],
                                           trainDbContentLengths_shuf[i-batchSize:i])
            
            trainCosts.append(batchTrainCost)
            #print("\t", batchTrainCost, flush=True, file=sessionTerminalOutputStream)
            #break
            
        trainCostAve = np.mean(trainCosts)
        trainCostStd = np.std(trainCosts)
        print(e, "train cost", trainCostAve, trainCostStd, flush=True, file=sessionTerminalOutputStream)
        
        
        if e % evalEvery == 0 or e == (numEpochs-1):
            
            #saveModelDir = tools.create_directory(sessionDir+"/{}".format(e))
            #learner.save(saveModelDir+"/saved_session".format(e))
            
            #
            # compute accuracy, etc.
            #
            
            # TRAIN
            trainActionIdCorrAcc, trainCamCorrAcc, trainAttrExactMatch, trainAttrJaccardIndex, trainBothCorrAcc, trainLocCorrAcc, trainSpatStCorrAcc, trainStTargCorrAcc = evaluate_split("TRAIN",
                                                                                     trainInputSequenceVectors,
                                                                                     trainOutputSpeechClusterIds, #trainOutputActionIds,
                                                                                     trainOutputCameraIndices,
                                                                                     trainOutputAttributeIndices,
                                                                                     trainOutputShopkeeperLocations, 
                                                                                     trainOutputSpatialStates, 
                                                                                     trainOutputStateTargets,
                                                                                     trainOutputMasks,
                                                                                     trainDbContentLengths,
                                                                                     trainBatchEndIndices, 
                                                                                     trainInterestingInstances,
                                                                                     trainInteractions,
                                                                                     trainGtDatabasebCameras,
                                                                                     trainGtDatabaseAttributes,
                                                                                     trainCostAve,
                                                                                     trainCostStd
                                                                                     )

            
            
            # TEST            
            testCosts = []
            
            for i in testBatchEndIndices:
                
                batchTestCost = learner.get_loss(testInputSequenceVectors[i-batchSize:i],
                                                 testOutputSpeechClusterIds[i-batchSize:i],
                                                 testOutputCameraIndices[i-batchSize:i],
                                                 testOutputAttributeIndices[i-batchSize:i],
                                                 testOutputShopkeeperLocations[i-batchSize:i],
                                                 testOutputSpatialStates[i-batchSize:i],
                                                 testOutputStateTargets[i-batchSize:i],                                                 
                                                 testOutputMasks[i-batchSize:i],
                                                 testDbContentLengths[i-batchSize:i],
                                                 )
                
                testCosts.append(batchTestCost)
            
            
            testCostAve = np.mean(testCosts)
            testCostStd = np.std(testCosts)
            
            
            testActionIdCorrAcc, testCamCorrAcc, testAttrExactMatch, testAttrJaccardIndex, testBothCorrAcc, testLocCorrAcc, testSpatStCorrAcc, testStTargCorrAcc = evaluate_split("TEST", 
                                                                                  testInputSequenceVectors,
                                                                                  testOutputSpeechClusterIds,
                                                                                  testOutputCameraIndices,
                                                                                  testOutputAttributeIndices,
                                                                                  testOutputShopkeeperLocations, 
                                                                                  testOutputSpatialStates, 
                                                                                  testOutputStateTargets,
                                                                                  testOutputMasks,
                                                                                  testDbContentLengths,
                                                                                  testBatchEndIndices,
                                                                                  testInterestingInstances,
                                                                                  testInteractions,
                                                                                  testGtDatabasebCameras,
                                                                                  testGtDatabaseAttributes,
                                                                                  testCostAve,
                                                                                  testCostStd,
                                                                                  )


            
            
            
            # append to session log   
            with open(sessionLogFile, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([e,                 #"Epoch", 
                                 teacherForcingProb,
                                 trainCostAve,      #"Train Cost Ave ({})".format(seed), 
                                 trainCostStd,      #"Train Cost SD ({})".format(seed), 
                                 trainActionIdCorrAcc,
                                 trainCamCorrAcc,
                                 trainAttrExactMatch,
                                 trainAttrJaccardIndex,
                                 trainBothCorrAcc,
                                 trainLocCorrAcc,
                                 trainSpatStCorrAcc,
                                 trainStTargCorrAcc,
                                 
                                 testCostAve,       #"Test Cost Ave({})".format(seed),
                                 testCostStd,       #"Test Cost SD ({})".format(seed), 
                                 testActionIdCorrAcc,
                                 testCamCorrAcc,
                                 testAttrExactMatch,
                                 testAttrJaccardIndex,
                                 testBothCorrAcc,
                                 testLocCorrAcc,
                                 testSpatStCorrAcc,
                                 testStTargCorrAcc
                                 ])    
        
        
        print("epoch time", round(time.time() - startTime, 2), flush=True, file=sessionTerminalOutputStream)



if __name__ == "__main__":
    
    
    camTemp = 0
    attTemp = 0
    
    
    #run(0, 0, camTemp, attTemp, 0.0, sessionDir)
    
    
    for gpu in range(8):
        
        seed = gpu
        
        process = Process(target=run, args=[gpu, seed, camTemp, attTemp, 0.0, sessionDir])
        process.start()
    
    
    
    #gpu = 0
    """
    processes = []
    
    #for camTemp in [2, 2.5, 3]:    
    #    for attTemp in [2, 3, 4, 5, 6]:
     
    for tfp in [0.7, 0.8, 0.9, 0.3]:
        for gpu in range(8):
            process = Process(target=run, args=[gpu, gpu, camTemp, attTemp, tfp, sessionDir])
            process.start()
            processes.append(process)
        
        for process in processes:
            process.join()
    """
    
    



