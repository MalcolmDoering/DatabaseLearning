'''
Created on April 9, 2019

@author: MalcolmD


modified from actionPrediction12


allow more than two training databases

'''


import numpy as np
from six.moves import range
from datetime import datetime
from sklearn import metrics
import editdistance
import csv
import os
import time
import random
from sklearn import preprocessing
from multiprocessing import Process
import sys
import string
from collections import OrderedDict


import tools
from utterancevectorizer import UtteranceVectorizer



DEBUG = False


eosChar = "#"
goChar = "~"


cameras = ["CAMERA_1", "CAMERA_2", "CAMERA_3"]


numTrainDbs = 10
batchSize = 128
embeddingSize = 100
numEpochs = 10000
evalEvery = 10
randomizeTrainingBatches = False

sessionDir = tools.create_session_dir("actionPrediction14_dbl")



def normalized_edit_distance(s1, s2):
    return editdistance.eval(s1, s2) / float(max(len(s1), len(s2)))



def compute_db_substring_match(groundTruthUtterances, predictedUtterances, shkpUttToDbEntryRange):
    
    subStringCharMatchAccuracies = []
    
    for i in range(len(groundTruthUtterances)):
        
        gtUtt = groundTruthUtterances[i]
        predUtt = predictedUtterances[i]
        
        if gtUtt in shkpUttToDbEntryRange:
            
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



def vectorize_sentences(sentences, charToIndex, maxSentLen, padChar=None, useEosChar=False, padPre=False):
    
    maxSentLen += 1 # for the EOS char
    
    sentVecs = []
    sentCharIndexLists = []
    sentLens = [] # including EoS char, to be used for masking losses
    
    for i in range(len(sentences)):
        
        # vectorize the main portion of the sentence
        sentCharVecs = []
        sentCharIndexList = []
        sentLen = len(sentences[i])
        
        for j in range(len(sentences[i])):
            
            charVec = np.zeros(len(charToIndex))
            charVec[charToIndex[sentences[i][j]]] = 1.0
            charIndex = charToIndex[sentences[i][j]]
            
            sentCharVecs.append(charVec)
            sentCharIndexList.append(charIndex)
        
        
        # add EoS char
        if useEosChar:
            
            charVec = np.zeros(len(charToIndex))
            charVec[charToIndex[eosChar]] = 1.0
            charIndex = charToIndex[eosChar]
            
            sentCharVecs.append(charVec)
            sentCharIndexList.append(charIndex)
            
            sentLen += 1
        
        
        # add padding
        padLen = maxSentLen - len(sentCharVecs)
        
        if padChar == None:
            padCharVec = np.zeros(len(charToIndex))
            padCharIndex = -1
        
        else:
            padCharVec = np.zeros(len(charToIndex))
            padCharVec[charToIndex[padChar]] = 1.0
            padCharIndex = charToIndex[padChar]
        
        padCharVecs = [padCharVec] * padLen
        padCharIndexList = [padCharIndex] * padLen
        
        
        if padPre:
            sentCharVecs = padCharVecs + sentCharVecs
            sentCharIndexList = padCharIndexList + sentCharIndexList
        
        else:
            sentCharVecs = sentCharVecs + padCharVecs
            sentCharIndexList = sentCharIndexList + padCharIndexList
        
        
        sentVec = np.vstack(sentCharVecs)
        
        
        sentVecs.append(sentVec)
        sentCharIndexLists.append(sentCharIndexList)
        sentLens.append(sentLen)
    
    
    return sentVecs, sentCharIndexLists, sentLens



def vectorize_databases(dbStrings, charToIndex, maxDbValLen):
    
    dbVectors = []
    dbIndexLists = []
    
    for row in dbStrings:
        valVecs, valCharIndexLists, _ = vectorize_sentences(row, charToIndex, maxDbValLen, padChar=None, useEosChar=False, padPre=False)
        
        dbVectors.append(valVecs)
        dbIndexLists.append(valCharIndexLists)
    
    return dbVectors, dbIndexLists



def unvectorize_sentences(sentCharIndexLists, indexToChar):
    
    sentences = []
    
    #for i in range(sentCharIndexLists.shape[0]):
    for i in range(len(sentCharIndexLists)):
            
        sent = ""
        
        for j in range(sentCharIndexLists[i].shape[0]):
            
            sent += indexToChar[sentCharIndexLists[i][j]]
        
        sentences.append(sent)
        
    return sentences



def color_results(outputStringsPred, outputIndexGt, copyScores, genScores, charToIndex):
    
    coloredOutputStrings = []
    
    copyScoresPred = []
    genScoresPred = []
    
    copyScoresGt = []
    genScoresGt = []
    
    
    for i in range(len(outputStringsPred)):
        colOutStr = ""
        
        cScoresPred = []
        gScoresPred = []
        
        cScoresGt = []
        gScoresGt = []
        
        
        for j in range(len(outputStringsPred[i])):
            cs = copyScores[i][j, charToIndex[outputStringsPred[i][j]]]
            gs = genScores[i][j, charToIndex[outputStringsPred[i][j]]]
            
            csGt = copyScores[i][j, outputIndexGt[i][j]]
            gsGt = genScores[i][j, outputIndexGt[i][j]]
            
            
            # color the char if it was coppied from the input
            if cs > gs:
                colOutStr += "\x1b[36m" + outputStringsPred[i][j] + "\x1b[0m" # blue-green
                
            elif cs == gs:
                colOutStr += "\x1b[33m" + outputStringsPred[i][j] + "\x1b[0m" # yellow
            
            else:
                colOutStr += outputStringsPred[i][j]
            
            
            cScoresPred.append(cs)
            gScoresPred.append(gs)
            
            cScoresGt.append(csGt)
            gScoresGt.append(gsGt)
            
        
        coloredOutputStrings.append(colOutStr)
        
        
        copyScoresPred.append(cScoresPred)
        genScoresPred.append(gScoresPred)
        
        copyScoresGt.append(cScoresGt)
        genScoresGt.append(gScoresGt)
        
        
    return coloredOutputStrings, copyScoresPred, genScoresPred, copyScoresGt, genScoresGt



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


def get_input_output_strings_and_location_vectors(interactions, locationToIndex):
    
    inputStrings = []
    outputStrings = []
    
    inputCustomerLocations = []
    outputShopkeeperLocations = []
    
    for turn in interactions:
        
        iString = turn["CUSTOMER_SPEECH"]
        oString = turn["SHOPKEEPER_SPEECH"]
        
        inputStrings.append(iString)
        outputStrings.append(oString)
        
        inputCustomerLocations.append(turn["CUSTOMER_LOCATION"])
        outputShopkeeperLocations.append(turn["OUTPUT_SHOPKEEPER_LOCATION"])
    
        
    inputCustomerLocationVectors = one_hot_vectorize(inputCustomerLocations, locationToIndex)
    outputShopkeeperLocationVectors = one_hot_vectorize(outputShopkeeperLocations, locationToIndex)
    
    return inputStrings, outputStrings, inputCustomerLocationVectors, outputShopkeeperLocationVectors


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



def run(gpu, seed, camTemp, attTemp, teacherForcingProb, sessionDir):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import learning2
    
    
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
    # load the simulated interactions and databases
    #
    print("loading data...", flush=True, file=sessionTerminalOutputStream)
    
    
    #dataDirectory = tools.dataDir+"/2019-05-21_14-11-57_advancedSimulator8" # only one possible sentence per customer and shopkeeper action
    #dataDirectory = tools.dataDir+"/handmade_0" # only one possible sentence per customer and shopkeeper action
    
    #dataDirectory = tools.dataDir+"/2019-07-03_15-16-05_advancedSimulator8" # many possible sentences for customer actions (from h-h dataset)
    #dataDirectory = tools.dataDir+"/2019-07-22_handmade_0" # many possible sentences for customer actions (from h-h dataset)
    
    dataDirectory = tools.dataDir+"/2019-07-24_14-58-47_advancedSimulator8" # many possible sentences for customer actions (from h-h dataset), all attributes change
    
    
    
    filenames = os.listdir(dataDirectory)
    filenames.sort()
    
    databaseFilenamesAll = [dataDirectory+"/"+fn for fn in filenames if "simulated" not in fn]
    interactionFilenamesAll = [dataDirectory+"/"+fn for fn in filenames if "simulated" in fn]
    
    
    
    
    """
    databaseFilenames = [tools.modelDir+"/database_0a.csv",
                         tools.modelDir+"/database_0b.csv",
                         tools.modelDir+"/database_0c.csv"]
    
    interactionFilenames = [tools.dataDir+"/20190508_simulated_data_1000_database_0a.csv",
                            tools.dataDir+"/20190508_simulated_data_1000_database_0b.csv",
                            tools.dataDir+"/20190508_simulated_data_1000_database_0c.csv"]
    """
    
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
    
    shkpUttToDbEntryRange = {}
    
    for i in range(len(interactionFilenames)):
        iFn = interactionFilenames[i]
        
        inters, sutder, gtDbCamera, gtDbAttribute = read_simulated_interactions(iFn, dbFieldnames)#, keepActions=["S_ANSWERS_QUESTION_ABOUT_FEATURE"]) # S_INTRODUCES_CAMERA S_INTRODUCES_FEATURE
        
        if i < numTrainDbs:
            # reduce the amount of training data because we have increased the number of training databases (assumes 1000 interactions per DB)
            inters = inters[: int(2 * len(inters) / numTrainDbs)] # two is the minimum number of training databases 
            gtDbCamera = gtDbCamera[: int(2 * len(gtDbCamera) / numTrainDbs)]
            gtDbAttribute = gtDbAttribute[: int(2 * len(gtDbAttribute) / numTrainDbs)]
        
        
        interactions.append(inters)
        datasetSizes.append(len(inters))
        
        gtDatabaseCameras.append(gtDbCamera)
        gtDatabaseAttributes.append(gtDbAttribute)
        
        
        # combine the three dictionaries into one
        shkpUttToDbEntryRange = {**shkpUttToDbEntryRange, **sutder}
    
    
    
    #
    # get the strings to be encoded
    #
    inputStrings = []
    outputStrings = []
    inputCustomerLocationVectors = []
    outputShopkeeperLocationVectors = []
    
    locationToIndex = {}
    locationToIndex["DOOR"] = 0
    locationToIndex["MIDDLE"] = len(locationToIndex)
    locationToIndex["CAMERA_1"] = len(locationToIndex)
    locationToIndex["CAMERA_2"] = len(locationToIndex)
    locationToIndex["CAMERA_3"] = len(locationToIndex)
    locationToIndex["SERVICE_COUNTER"] = len(locationToIndex)
    
    indexToLocation = dict(map(reversed, locationToIndex.items()))
    
    
    for intSet in interactions:
        inStrs, outStrs, inCustLocVecs, outCustLocVecs = get_input_output_strings_and_location_vectors(intSet, locationToIndex)
        
        inputStrings.append(inStrs)
        outputStrings.append(outStrs)
        inputCustomerLocationVectors.append(inCustLocVecs)
        outputShopkeeperLocationVectors.append(outCustLocVecs)
    
    
    dbStrings = []
    
    for db in databases:
        dbStrs = get_database_value_strings(db, dbFieldnames)
        dbStrings.append(dbStrs)
    
    
    
    
    
    
    #
    # create a character encoder
    #
    print("creating character encoder...", flush=True, file=sessionTerminalOutputStream)
    
    allStrings = []
    
    for inStrs in inputStrings:
        allStrings += inStrs
    
    for outStrs in outputStrings:
        allStrings += outStrs
    
    for dbStrs in dbStrings:
        for row in dbStrs:
            allStrings += row
    
    
    charToIndex = {}
    indexToChar = {}
    
    for s in allStrings:
        for c in s:
            if c not in charToIndex:
                charToIndex[c] = len(charToIndex)
                indexToChar[charToIndex[c]] = c
    
    # add eos and go chars
    charToIndex[eosChar] = len(charToIndex)
    indexToChar[charToIndex[eosChar]] = eosChar
    charToIndex[goChar] = len(charToIndex)
    indexToChar[charToIndex[goChar]] = goChar
    
    
    
    #
    # vectorize the simulated interactions and databases
    #
    print("vectorizing the string data...", flush=True, file=sessionTerminalOutputStream)
    
    # find max input length
    allInputLens = []
    
    for inStrs in inputStrings:
        allInputLens += [len(i) for i in inStrs]
    
    maxInputLen = max(allInputLens)
    
    
    # find max output length
    allOutputLens = []
    
    for outStrs in outputStrings:
        allOutputLens += [len(i) for i in outStrs]
    
    maxOutputLen = max(allOutputLens)
    
    
    # find max DB value length
    allDbValueLens = []
    
    for dbStrs in dbStrings:
        for row in dbStrs:
            allDbValueLens += [len(v) for v in row]
    
    maxDbValueLen = max(allDbValueLens)
    
    
    # vectorize the inputs
    allInputStrings = sum(inputStrings, [])
    
    inputUttVectorizer = UtteranceVectorizer(allInputStrings,
                                        minCount=2, 
                                        keywordWeight=1.0, 
                                        keywordSet=[], 
                                        unigramsAndKeywordsOnly=False, 
                                        tfidf=False,
                                        useStopwords=False,
                                        lsa=False)
    
    inputUttVectors = []
    
    for inStrs in inputStrings:
        inUttVecs = inputUttVectorizer.get_utterance_vectors(inStrs, asArray=False)
        inputUttVectors.append(inUttVecs)
    
    inputUttVecDim = inputUttVectors[0][0].size
    
    """
    inputIndexLists = []
    
    for inStrs in inputStrings:
        _, inIndLst, _ = vectorize_sentences(inStrs, charToIndex, maxInputLen, padChar=None, useEosChar=False, padPre=True)
        inputIndexLists.append(inIndLst)
    """
    
    # vectorize the outputs
    outputIndexLists = []
    outputStringLens = []
    
    for outStrs in outputStrings:
        _, outIndLst, outSentLens = vectorize_sentences(outStrs, charToIndex, maxOutputLen, padChar=None, useEosChar=True, padPre=False)
        outputIndexLists.append(outIndLst)
        outputStringLens.append(outSentLens)
    
    
    # vectorize the DB values
    dbVectors = []
    
    for dbStrs in dbStrings:
        dbVecs, _ = vectorize_databases(dbStrs, charToIndex, maxDbValueLen)
        dbVectors.append(dbVecs)
    
    
    #
    # split into training and testing sets
    #
    
    def prepare_split(startDbIndex, stopDbIndex, splitName):
        """input which DB to start with and which DB to end with + 1"""
        
        splitInputUttVectors = []
        splitOutputIndexLists = []
        splitOutputStrings = []
        splitOutputStringLens = []
        
        splitGtDatabasebCameras = []
        splitGtDatabaseAttributes = []
        
        splitInteractions = []
        
        uniqueId = 0
        
        
        for i in range(startDbIndex, stopDbIndex):
            splitInputUttVectors += inputUttVectors[i]
            splitOutputIndexLists += outputIndexLists[i]
            splitOutputStrings += outputStrings[i]
            splitOutputStringLens += outputStringLens[i]
            
            splitGtDatabasebCameras += gtDatabaseCameras[i]
            splitGtDatabaseAttributes += gtDatabaseAttributes[i]
            
            for j in range(len(interactions[i])):
                interactions[i][j]["SET"] = splitName 
                interactions[i][j]["ID"] = uniqueId
                uniqueId += 1
            
            splitInteractions += interactions[i]
        
        
        splitInputCustomerLocations = np.concatenate(inputCustomerLocationVectors[startDbIndex:stopDbIndex])
        splitOutputShopkeeperLocations = np.concatenate(outputShopkeeperLocationVectors[startDbIndex:stopDbIndex])
        
        splitDbVectors = []
        
        for i in range(startDbIndex, stopDbIndex):
            for j in range(datasetSizes[i]):
                splitDbVectors.append(dbVectors[i])
        
        
        
        
        
        
        return splitInteractions, splitInputUttVectors, splitOutputIndexLists, splitOutputStrings, splitOutputStringLens, splitInputCustomerLocations, splitOutputShopkeeperLocations, splitDbVectors, splitGtDatabasebCameras, splitGtDatabaseAttributes
        
    
    
    # training
    trainInteractions, trainInputUttVectors, trainOutputIndexLists, trainOutputStrings, trainOutputStringLens, trainInputCustomerLocations, trainOutputShopkeeperLocations, trainDbVectors, trainGtDatabasebCameras, trainGtDatabaseAttributes = prepare_split(0, numTrainDbs, "Train")
    
    # testing
    testInteractions, testInputUttVectors, testOutputIndexLists, testOutputStrings, testOutputStringLens, testInputCustomerLocations, testOutputShopkeeperLocations, testDbVectors, testGtDatabasebCameras, testGtDatabaseAttributes = prepare_split(numTrainDbs, numDatabases, "Test")
    
    
    
    print(len(trainInputUttVectors), "training examples", flush=True, file=sessionTerminalOutputStream)
    print(len(testInputUttVectors), "testing examples", flush=True, file=sessionTerminalOutputStream)
    print(inputUttVecDim, "input utterance vector size", flush=True, file=sessionTerminalOutputStream)
    print(maxInputLen, "input sequence length", flush=True, file=sessionTerminalOutputStream)
    print(maxOutputLen, "output sequence length", flush=True, file=sessionTerminalOutputStream)
    print(maxDbValueLen, "DB value sequence length", flush=True, file=sessionTerminalOutputStream)
    print(len(charToIndex), "unique characters", flush=True, file=sessionTerminalOutputStream)
    
    
    #
    # setup the model
    #
    print("setting up the model...", flush=True, file=sessionTerminalOutputStream)
    
    
    inputLen = maxInputLen + 1
    outputLen = maxOutputLen + 1
    dbValLen = maxDbValueLen + 1
    
    locationVecLen = len(locationToIndex)
    
    
    """
    # for computing accuracy
    trainGroundTruthFlat = []
    testGroundTruthFlat = []
    
    for i in range(numExamples):    
        groundTruthFlat = groundTruthOutputs[i]
        
        if i < 4:
            trainGroundTruthFlat += groundTruthFlat
        else:
            testGroundTruthFlat += groundTruthFlat
    """
    
    learner = learning2.CustomNeuralNetwork(inputUttVecDim=inputUttVecDim, 
                                  dbSeqLen=dbValLen, 
                                  outputSeqLen=outputLen,
                                  locationVecLen=locationVecLen,
                                  numUniqueCams=len(cameras),
                                  numUniqueAtts=len(dbFieldnames),
                                  batchSize=batchSize, 
                                  vocabSize=len(charToIndex),
                                  embeddingSize=embeddingSize,
                                  charToIndex=charToIndex,
                                  camTemp=camTemp,
                                  attTemp=attTemp,
                                  seed=seed)
    
    
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
    
    
    trainBatchEndIndices = list(range(batchSize, len(trainInputUttVectors), batchSize))
    testBatchEndIndices = list(range(batchSize, len(testInputUttVectors), batchSize))
    
    
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
    
    
    #
    # count the occurrences of different chars in the DB substrings in the training data
    # This code only gives useful data if the training data is not shuffled (because we do not know which data is excluded  for being outside batches) 
    # TODO: this will have to be updated if we include things beyond the price responses in the data
    #
    """
    dbSubstrCharCounts = {}
    
    dbSubstrCharCounts["CAMERA_1"] = {}
    dbSubstrCharCounts["CAMERA_2"] = {}
    dbSubstrCharCounts["CAMERA_3"] = {}
    dbSubstrCharCounts["Combined"] = {}
    
    
    # iterate over the batches
    for i in trainBatchEndIndices:
        
        # iterate over the output strings
        for j in range(i-batchSize, i):
            
            trainOutStr = trainOutputStrings[j]
            
            
            # iterate over the substring from the DB
            countFromSubstrBegining = 0
            
            for k in range(shkpUttToDbEntryRange[trainOutStr][0], shkpUttToDbEntryRange[trainOutStr][1]):
                
                char = trainOutStr[k]
                
                
                # compute for only the current camera
                #
                cam = cameras[trainGtDatabasebCameras[j]]
                
                if countFromSubstrBegining not in dbSubstrCharCounts[cam]:
                    dbSubstrCharCounts[cam][countFromSubstrBegining] = {}
                
                if char not in dbSubstrCharCounts[cam][countFromSubstrBegining]:
                    dbSubstrCharCounts[cam][countFromSubstrBegining][char] = 0
                
                dbSubstrCharCounts[cam][countFromSubstrBegining][char] += 1
                
                
                # compute for all cameras combined
                #
                if countFromSubstrBegining not in dbSubstrCharCounts["Combined"]:
                    dbSubstrCharCounts["Combined"][countFromSubstrBegining] = {}
                
                if char not in dbSubstrCharCounts["Combined"][countFromSubstrBegining]:
                    dbSubstrCharCounts["Combined"][countFromSubstrBegining][char] = 0
                
                dbSubstrCharCounts["Combined"][countFromSubstrBegining][char] += 1
                
                
                countFromSubstrBegining += 1
    
    
    # write to a file
    with open(sessionDir + "/DB_substring_char_counts_{}.csv".format(sessionIdentifier), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        chars = ["$", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        majorCols = ["CAMERA_1", "CAMERA_2", "CAMERA_3", "Combined"]
        
        minorCols = []
        
        for majCol in majorCols:
            for i in range(5):
                minorCols.append("{} {}".format(majCol, i))
        
        
        rows = OrderedDict()
        
        for char in chars:
            rows[char] = []
            
            for majCol in majorCols:
                for i in range(5):
                    
                    try:
                        rows[char].append(dbSubstrCharCounts[majCol][i][char])
                    except:
                        rows[char].append(0)
            
        
        writer.writerow(["Char"] + minorCols)
        
        for char, row in rows.items():
            writer.writerow([char] + row)
    """
    
    
    
    # write header in csv log file
    with open(sessionLogFile, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch",
                         "Teacher Forcing Probability",
                         "Train Cost Ave ({})".format(sessionIdentifier), 
                         "Train Cost SD ({})".format(sessionIdentifier), 
                         "Train DB Substring Correct All ({})".format(sessionIdentifier), 
                         "Train DB Substring Correct Ave ({})".format(sessionIdentifier), 
                         "Train DB Substring Correct SD ({})".format(sessionIdentifier),
                         
                         "Train Cam. Address Correct ({})".format(sessionIdentifier),
                         "Train Attr. Address Correct ({})".format(sessionIdentifier),
                         "Train Both Addresses Correct ({})".format(sessionIdentifier),
                         
                         "Test Cost Ave({})".format(sessionIdentifier),
                         "Test Cost SD ({})".format(sessionIdentifier), 
                         "Test DB Substring Correct All ({})".format(sessionIdentifier), 
                         "Test DB Substring Correct Ave ({})".format(sessionIdentifier), 
                         "Test DB Substring Correct SD ({})".format(sessionIdentifier), 
                         
                         "Test Cam. Address Correct ({})".format(sessionIdentifier),
                         "Test Attr. Address Correct ({})".format(sessionIdentifier),
                         "Test Both Addresses Correct ({})".format(sessionIdentifier)
                         ])
    
    
    interactionsFieldnames = ["SET",
                      "ID",
                      "TRIAL",
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
                      "SHOPKEEPER_SPEECH_DB_ENTRY_RANGE",
                      
                      "PRED_OUTPUT_SHOPKEEPER_LOCATION",
                      "PRED_SHOPKEEPER_SPEECH",
                      "PRED_CAM_MATCH",
                      "PRED_ATT_MATCH"]
        
    for c in range(len(cameras)):
        interactionsFieldnames.append("PRED_CAM_MATCH_SCORE_{}".format(cameras[c]))
        
    for a in range(len(dbFieldnames)):
        interactionsFieldnames.append("PRED_ATT_MATCH_SCORE_{}".format(dbFieldnames[a]))
    
    
    def evaluate_split(splitName, splitBatchEndIndices, indexToChar, splitInterestingInstances,
                       splitInteractions,
                       splitInputUttVectors,
                       splitInputCustomerLocations, 
                       splitDbVectors, 
                       splitOutputIndexLists,
                       splitOutputStringLens,
                       splitOutputShopkeeperLocations,
                       splitGtDatabasebCameras,
                       splitGtDatabaseAttributes,
                       splitOutputStrings,
                       
                       splitCostAve=None,
                       splitCostStd=None):
        
        
        
        
        if splitCostAve == None:
            splitCosts = []
            
            for i in splitBatchEndIndices:
                
                batchSplitCost = learner.train_loss(splitInputUttVectors[i-batchSize:i], 
                                             splitInputCustomerLocations[i-batchSize:i], 
                                             splitDbVectors[i-batchSize:i], 
                                             splitOutputIndexLists[i-batchSize:i],
                                             splitOutputStringLens[i-batchSize:i],
                                             splitOutputShopkeeperLocations[i-batchSize:i],
                                             splitGtDatabasebCameras[i-batchSize:i],
                                             splitGtDatabaseAttributes[i-batchSize:i],
                                             0.0)
                
                splitCosts.append(batchSplitCost)
            
            splitCostAve = np.mean(splitCosts)
            splitCostStd = np.mean(splitCosts)
        
        
        splitUttPreds = []
        splitLocPreds = []
        splitCopyScores = []
        splitGenScores = []
        splitCamMatchArgMax = []
        splitAttMatchArgMax = []
        splitCamMatch = []
        splitAttMatch  = []
        splitDbReadWeights = []
        splitCopyWeights = [] 
        splitGenWeights = []    
        
        splitGtSents = []
        splitGtCamIndx = []
        splitGtAttIndx = []
        
        
        for i in splitBatchEndIndices:
            batchPredResults = learner.predict(splitInputUttVectors[i-batchSize:i], 
                                               splitInputCustomerLocations[i-batchSize:i],
                                               splitDbVectors[i-batchSize:i], 
                                               splitOutputIndexLists[i-batchSize:i],
                                               splitOutputStringLens[i-batchSize:i],
                                               splitOutputShopkeeperLocations[i-batchSize:i],
                                               splitGtDatabasebCameras[i-batchSize:i],
                                               splitGtDatabaseAttributes[i-batchSize:i])
            
            batchSplitUttPreds, batchSplitLocPreds, batchSplitCopyScores, batchSplitGenScores, batchSplitCamMatchArgMax, batchSplitAttMatchArgMax, batchSplitCamMatch, batchSplitAttMatch, batchSplitDbReadWeights, batchSplitCopyWeights, batchSplitGenWeights = batchPredResults 
            
            """
            batchDbMatchVal = learner.get_db_match_val(splitInputUttVectors[i-batchSize:i], 
                                               splitInputCustomerLocations[i-batchSize:i],
                                               splitDbVectors[i-batchSize:i], 
                                               splitOutputIndexLists[i-batchSize:i],
                                               splitOutputStringLens[i-batchSize:i],
                                               splitOutputShopkeeperLocations[i-batchSize:i],
                                               splitGtDatabasebCameras[i-batchSize:i],
                                               splitGtDatabaseAttributes[i-batchSize:i])
            """
            
            
            splitUttPreds.append(batchSplitUttPreds)
            splitLocPreds.append(batchSplitLocPreds)
            splitCopyScores.append(batchSplitCopyScores)
            splitGenScores.append(batchSplitGenScores)
            splitCamMatchArgMax.append(batchSplitCamMatchArgMax)
            splitAttMatchArgMax.append(batchSplitAttMatchArgMax)
            splitCamMatch.append(batchSplitCamMatch)
            splitAttMatch.append(batchSplitAttMatch)
            splitDbReadWeights.append(batchSplitDbReadWeights)
            splitCopyWeights.append(batchSplitCopyWeights)
            splitGenWeights.append(batchSplitGenWeights)
            
            splitGtSents += splitOutputStrings[i-batchSize:i]
            splitGtCamIndx += splitGtDatabasebCameras[i-batchSize:i]
            splitGtAttIndx += splitGtDatabaseAttributes[i-batchSize:i]
            
        splitUttPreds = np.concatenate(splitUttPreds)
        splitLocPreds = np.concatenate(splitLocPreds)
        splitCopyScores = np.concatenate(splitCopyScores)
        splitGenScores = np.concatenate(splitGenScores)
        splitCamMatchArgMax = np.concatenate(splitCamMatchArgMax)
        splitAttMatchArgMax = np.concatenate(splitAttMatchArgMax)
        splitCamMatch = np.concatenate(splitCamMatch)
        splitAttMatch  = np.concatenate(splitAttMatch)
        splitDbReadWeights = np.concatenate(splitDbReadWeights)
        splitCopyWeights = np.concatenate(splitCopyWeights)
        splitGenWeights = np.concatenate(splitGenWeights)
        
        splitPredSents = unvectorize_sentences(splitUttPreds, indexToChar)
        
        
        # compute how much of the substring that should be copied from the DB is correct
        splitSubstrCharMatchAccs = compute_db_substring_match(splitGtSents, splitPredSents, shkpUttToDbEntryRange)
        
        # how many 100% correct substrings
        splitSubstrAllCorr = splitSubstrCharMatchAccs.count(1.0) / len(splitSubstrCharMatchAccs)
        
        # average substring correctness
        splitSubstrAveCorr = np.mean(splitSubstrCharMatchAccs)
        splitSubstrSdCorr = np.std(splitSubstrCharMatchAccs)
        
        print("{} substr all correct {}, ave correctness {} ({})".format(splitName, splitSubstrAllCorr, splitSubstrAveCorr, splitSubstrSdCorr), flush=True, file=sessionTerminalOutputStream)
        
        
        
        splitCamCorr, splitAttrCorr, splitBothCorr = compute_db_address_match(splitGtCamIndx, splitGtAttIndx, splitCamMatchArgMax, splitAttMatchArgMax)
        
        splitCamCorrAve = np.mean(splitCamCorr)
        splitAttrCorrAve = np.mean(splitAttrCorr)
        splitBothCorrAve = np.mean(splitBothCorr)
        
        print("{} DB addressing correctness: cam. {}, attr. {}, both {}".format(splitName, splitCamCorrAve, splitAttrCorrAve, splitBothCorrAve), flush=True, file=sessionTerminalOutputStream)
        
            
        
        splitPredSentsColored, splitCopyScoresPred, splitGenScoresPred, splitCopyScoresTrue, splitGenScoresTrue = color_results(splitPredSents, 
                                                                                                                                splitOutputIndexLists,
                                                                                                                                splitCopyScores, 
                                                                                                                                splitGenScores, 
                                                                                                                                charToIndex)
        
        
        #
        # output to terminal
        #
        print(splitName, e, round(splitCostAve, 3), round(splitCostStd, 3), flush=True, file=sessionTerminalOutputStream)
            
        for i in range(len(splitInterestingInstances)):
            index = splitInterestingInstances[i][0]
            info = splitInterestingInstances[i][1]
            
            print("TRUE:", indexToLocation[np.argmax(splitOutputShopkeeperLocations[index])], splitOutputStrings[index], flush=True, file=sessionTerminalOutputStream)
            print("PRED:", indexToLocation[splitLocPreds[index]], splitPredSents[index], flush=True, file=sessionTerminalOutputStream)
            
            print(np.round(splitCamMatch[index], 4), flush=True, file=sessionTerminalOutputStream)
            print(np.round(splitAttMatch[index], 4), flush=True, file=sessionTerminalOutputStream)
            print("best match:", cameras[splitCamMatchArgMax[index]], dbFieldnames[splitAttMatchArgMax[index]], flush=True, file=sessionTerminalOutputStream)
            print("true match:", info, flush=True, file=sessionTerminalOutputStream)
            
            print("", flush=True, file=sessionTerminalOutputStream) 
            
        print("", flush=True, file=sessionTerminalOutputStream)
        
        
        
        #
        # save all predictions to file
        #
        with open(sessionDir+"/{:}_all_outputs.csv".format(e), "a", newline="") as csvfile:
            
            writer = csv.DictWriter(csvfile, interactionsFieldnames)
            writer.writeheader()
            
            for i in range(splitBatchEndIndices[-1]):
                row = splitInteractions[i]
                
                # add the important prediction information to the row
                row["PRED_OUTPUT_SHOPKEEPER_LOCATION"] = indexToLocation[splitLocPreds[i]]
                row["PRED_SHOPKEEPER_SPEECH"] = splitPredSents[i]
                
                
                row["PRED_CAM_MATCH"] = cameras[splitCamMatchArgMax[i]]
                row["PRED_ATT_MATCH"] = dbFieldnames[splitAttMatchArgMax[i]]
                
                
                for c in range(len(cameras)):
                    row["PRED_CAM_MATCH_SCORE_{}".format(cameras[c])] = splitCamMatch[i][c]
                
                for a in range(len(dbFieldnames)):
                    row["PRED_ATT_MATCH_SCORE_{}".format(dbFieldnames[a])] = splitAttMatch[i][a]
                
                
                writer.writerow(row)
        
        
        #
        # save interesting instances to file
        #
        with open(sessionDir+"/{:}_outputs.csv".format(e), "a", newline="") as csvfile:
            
            writer = csv.writer(csvfile)
            
            
            writer.writerow([splitName, round(splitCostAve, 3), round(splitCostStd, 3)])
            
            for i in range(len(splitInterestingInstances)):
                index = splitInterestingInstances[i][0]
                info = splitInterestingInstances[i][1]
                
                
                writer.writerow(["TRUE:"] + 
                                [indexToLocation[np.argmax(splitOutputShopkeeperLocations[index])]] +
                                [c for c in splitOutputStrings[index]])
                
                writer.writerow(["PRED:"] + 
                                [indexToLocation[splitLocPreds[index]]] +
                                [c for c in splitPredSents[index]])
                
                writer.writerow(["PRED COPY WEIGHT:"] + [""] + [c for c in splitCopyWeights[i]])
                writer.writerow(["PRED GEN WEIGHT:"] + [""] + [c for c in splitGenWeights[i]])
                writer.writerow(["PRED DB TO CAND WEIGHT:"] + [""] + [c for c in splitDbReadWeights[i]])
                
                writer.writerow(["PRED COPY:"] + [""] + [c for c in splitCopyScoresPred[index]])
                writer.writerow(["PRED GEN:"] + [""] + [c for c in splitGenScoresPred[index]])
                
                writer.writerow(["TRUE COPY:"] + [""] + [c for c in splitCopyScoresTrue[index]])
                writer.writerow(["TRUE GEN:"] + [""] + [c for c in splitGenScoresTrue[index]])
                
                writer.writerow(["CAM MATCH:"] + [np.round(p, 3) for p in splitCamMatch[index]])
                writer.writerow(["ATT MATCH:"] + [np.round(p, 3) for p in splitAttMatch[index]])
                
                writer.writerow(["TOP MATCH:", cameras[splitCamMatchArgMax[index]]+" "+dbFieldnames[splitAttMatchArgMax[index]]])
                writer.writerow(["TRUE MATCH:", info])
                
                writer.writerow([])
        
        
        
        return splitSubstrAllCorr, splitSubstrAveCorr, splitSubstrSdCorr, splitCamCorrAve, splitAttrCorrAve, splitBothCorrAve
    
    
    
    
    
    for e in range(numEpochs):
        
        startTime = time.time()
        
        #teacherForcingProb = 0.6 #1.0 - 1.0 / (1.0 + np.exp( - (e-200.0)/10.0))
        
        #
        # train
        #
        trainCosts = []
        
        # shuffle the training data
        
        temp = list(zip(trainInputUttVectors, trainOutputStringLens, trainDbVectors, trainOutputIndexLists, trainInputCustomerLocations, trainOutputShopkeeperLocations))
        if randomizeTrainingBatches:
            random.shuffle(temp)
        trainInputIndexLists_shuf, trainOutputStringLens_shuf, trainDbVectors_shuf, trainOutputIndexLists_shuf, trainInputCustomerLocations_shuf, trainOutputShopkeeperLocations_shuf = zip(*temp)
        
        
        for i in trainBatchEndIndices:
            
            batchTrainCost = learner.train(trainInputIndexLists_shuf[i-batchSize:i],
                                           trainInputCustomerLocations_shuf[i-batchSize:i],
                                           trainDbVectors_shuf[i-batchSize:i], 
                                           trainOutputIndexLists_shuf[i-batchSize:i],
                                           trainOutputStringLens_shuf[i-batchSize:i],
                                           trainOutputShopkeeperLocations_shuf[i-batchSize:i],
                                           trainGtDatabasebCameras[i-batchSize:i],
                                           trainGtDatabaseAttributes[i-batchSize:i],
                                           teacherForcingProb)
            
            trainCosts.append(batchTrainCost)
            #print("\t", batchTrainCost, flush=True, file=sessionTerminalOutputStream)
            #break
        trainCostAve = np.mean(trainCosts)
        trainCostStd = np.std(trainCosts)
        print(e, "train cost", trainCostAve, trainCostStd, flush=True, file=sessionTerminalOutputStream)
        
        
        if e % evalEvery == 0 or e == (numEpochs-1):
            
            saveModelDir = tools.create_directory(sessionDir+"/{}".format(e))
            learner.save(saveModelDir+"/saved_session".format(e))
            
            #
            # compute accuracy, etc.
            #
            
            # TRAIN
            trainSubstrAllCorr, trainSubstrAveCorr, trainSubstrSdCorr, trainCamCorrAve, trainAttrCorrAve, trainBothCorrAve = evaluate_split("TRAIN", trainBatchEndIndices, indexToChar, trainInterestingInstances,
                                                                                                                                            trainInteractions,
                                                                                                                                            trainInputUttVectors, 
                                                                                                                                            trainInputCustomerLocations, 
                                                                                                                                            trainDbVectors, 
                                                                                                                                            trainOutputIndexLists,
                                                                                                                                            trainOutputStringLens,
                                                                                                                                            trainOutputShopkeeperLocations,
                                                                                                                                            trainGtDatabasebCameras,
                                                                                                                                            trainGtDatabaseAttributes,
                                                                                                                                            trainOutputStrings,
                                                                                                                                            trainCostAve,
                                                                                                                                            trainCostStd)
            
            
            
            # TEST            
            testCosts = []
            
            for i in testBatchEndIndices:
                
                batchTestCost = learner.train_loss(testInputUttVectors[i-batchSize:i], 
                                             testInputCustomerLocations[i-batchSize:i], 
                                             testDbVectors[i-batchSize:i], 
                                             testOutputIndexLists[i-batchSize:i],
                                             testOutputStringLens[i-batchSize:i],
                                             testOutputShopkeeperLocations[i-batchSize:i],
                                             testGtDatabasebCameras[i-batchSize:i],
                                             testGtDatabaseAttributes[i-batchSize:i],
                                             0.0)
                
                testCosts.append(batchTestCost)
            
            
            testCostAve = np.mean(testCosts)
            testCostStd = np.std(testCosts)
            
            
            testSubstrAllCorr, testSubstrAveCorr, testSubstrSdCorr, testCamCorrAve, testAttrCorrAve, testBothCorrAve = evaluate_split("TEST", testBatchEndIndices, indexToChar, testInterestingInstances,
                                                                                                                                      testInteractions,
                                                                                                                                      testInputUttVectors, 
                                                                                                                                      testInputCustomerLocations, 
                                                                                                                                      testDbVectors, 
                                                                                                                                      testOutputIndexLists,
                                                                                                                                      testOutputStringLens,
                                                                                                                                      testOutputShopkeeperLocations,
                                                                                                                                      testGtDatabasebCameras,
                                                                                                                                      testGtDatabaseAttributes,
                                                                                                                                      testOutputStrings,
                                                                                                                                      testCostAve,
                                                                                                                                      testCostStd)
            
            
            
            
            
            # append to session log   
            with open(sessionLogFile, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([e,                 #"Epoch", 
                                 teacherForcingProb,
                                 trainCostAve,      #"Train Cost Ave ({})".format(seed), 
                                 trainCostStd,      #"Train Cost SD ({})".format(seed), 
                                 trainSubstrAllCorr,    #"Train DB Substring Correct All ({})".format(seed), 
                                 trainSubstrAveCorr,    #"Train DB Substring Correct Ave ({})".format(seed), 
                                 trainSubstrSdCorr, #"Train DB Substring Correct SD ({})".format(seed), 
                                 trainCamCorrAve,
                                 trainAttrCorrAve,
                                 trainBothCorrAve,
                                 
                                 testCostAve,       #"Test Cost Ave({})".format(seed),
                                 testCostStd,       #"Test Cost SD ({})".format(seed), 
                                 testSubstrAllCorr, #"Test DB Substring Correct All ({})".format(seed), 
                                 testSubstrAveCorr, #"Test DB Substring Correct Ave ({})".format(seed), 
                                 testSubstrSdCorr,  #"Test DB Substring Correct SD ({})".format(seed),
                                 testCamCorrAve,
                                 testAttrCorrAve,
                                 testBothCorrAve
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
    
    



