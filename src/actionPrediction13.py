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

import tools


DEBUG = False


eosChar = "#"
goChar = "~"


cameras = ["CAMERA_1", "CAMERA_2", "CAMERA_3"]


sessionDir = tools.create_session_dir("actionPrediction12_dbl")
    


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
            if "$" in gtUtt[shkpUttToDbEntryRange[gtUtt][0]:shkpUttToDbEntryRange[gtUtt][1]]:
                
                subStringCharMatchCount = 0.0
                subStringCharTotalCount = 0.0
                
                for j in range(shkpUttToDbEntryRange[gtUtt][0], shkpUttToDbEntryRange[gtUtt][1]):
                    
                    if gtUtt[j] == predUtt[j]:
                        subStringCharMatchCount += 1
                    
                    subStringCharTotalCount += 1
                
                
                subStringCharMatchAccuracies.append(subStringCharMatchCount / subStringCharTotalCount)
        
    return subStringCharMatchAccuracies


def vectorize_sentences(sentences, charToIndex, maxSentLen):
    
    maxSentLen += 1 # for the EOS char
    
    sentVecs = []
    sentCharIndexLists = []
    
    for i in range(len(sentences)):
        
        sentVec = np.zeros(shape=(maxSentLen, len(charToIndex)))
        sentCharIndexList = []
        
        for j in range(maxSentLen):
            
            if j < len(sentences[i]):
                sentVec[j, charToIndex[sentences[i][j]]] = 1.0
                sentCharIndexList.append(charToIndex[sentences[i][j]])
            else:
                sentVec[j, charToIndex[" "]] = 1.0 # pad the end of sentences with spaces
                sentCharIndexList.append(charToIndex[" "])
        
        
        sentVec[-1, charToIndex[eosChar]] = 1
        sentCharIndexList.append(charToIndex[eosChar])
        
        sentVecs.append(sentVec)
        sentCharIndexLists.append(sentCharIndexList)
    
    
    return sentVecs, sentCharIndexLists



def vectorize_databases(dbStrings, charToIndex, maxDbValLen):
    
    dbVectors = []
    dbIndexLists = []
    
    for row in dbStrings:
        valVecs, valCharIndexLists = vectorize_sentences(row, charToIndex, maxDbValLen)
        
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
            
            database.append(row)
    
    return database, fieldnames


def read_simulated_interactions(filename, keepActions=None):
    interactions = []
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
            if keepActions == None:
                interactions.append(row)
            elif row["OUTPUT_SHOPKEEPER_ACTION"] in keepActions:
                interactions.append(row)
            
            if row["SHOPKEEPER_SPEECH_DB_ENTRY_RANGE"] != "":
                shkpUttToDbEntryRange[row["SHOPKEEPER_SPEECH"]] = [int(i) for i in row["SHOPKEEPER_SPEECH_DB_ENTRY_RANGE"].split("~")]
                
    return interactions, shkpUttToDbEntryRange


def get_input_output_strings_and_location_vectors(interactions):
    
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
    
    
    locationLabelEncoder = preprocessing.LabelEncoder()
    temp = locationLabelEncoder.fit_transform(inputCustomerLocations + outputShopkeeperLocations)
    oneHotEncoder = preprocessing.OneHotEncoder(sparse=False)
    oneHotEncoder.fit(temp.reshape(-1, 1))
    
    inputCustomerLocationVectors = oneHotEncoder.transform(locationLabelEncoder.transform(inputCustomerLocations).reshape(-1, 1))
    outputShopkeeperLocationVectors = oneHotEncoder.transform(locationLabelEncoder.transform(outputShopkeeperLocations).reshape(-1, 1))
    
    return inputStrings, outputStrings, inputCustomerLocationVectors, outputShopkeeperLocationVectors, locationLabelEncoder


def get_database_value_strings(database, fieldnames):
    
    valueStrings = []
    
    for row in database:
        
        rowStrings = []
        
        for col in fieldnames:
            rowStrings.append(row[col])
        
        valueStrings.append(rowStrings)
    
    return valueStrings



def run(gpu, seed, camTemp, attTemp, sessionDir):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import learning
    
    
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
    
    database0Filename = tools.modelDir+"/database_0a.csv"
    database1Filename = tools.modelDir+"/database_0b.csv"
    database2Filename = tools.modelDir+"/database_0c.csv"
    
    interactions0Filename = tools.dataDir+"/20190508_simulated_data_1000_database_0a.csv"
    interactions1Filename = tools.dataDir+"/20190508_simulated_data_1000_database_0b.csv"
    interactions2Filename = tools.dataDir+"/20190508_simulated_data_1000_database_0c.csv"
    
    
    database0, dbFieldnames = read_database_file(database0Filename)
    database1, _ = read_database_file(database1Filename)
    database2, _ = read_database_file(database2Filename)
    
    interactions0, shkpUttToDbEntryRange0 = read_simulated_interactions(interactions0Filename, keepActions=["S_ANSWERS_QUESTION_ABOUT_FEATURE"]) # S_INTRODUCES_CAMERA S_INTRODUCES_FEATURE
    interactions1, shkpUttToDbEntryRange1 = read_simulated_interactions(interactions1Filename, keepActions=["S_ANSWERS_QUESTION_ABOUT_FEATURE"])
    interactions2, shkpUttToDbEntryRange2 = read_simulated_interactions(interactions2Filename, keepActions=["S_ANSWERS_QUESTION_ABOUT_FEATURE"])
    
    
    # combine the three dictionaries into one
    shkpUttToDbEntryRange = {**shkpUttToDbEntryRange0, **shkpUttToDbEntryRange1, **shkpUttToDbEntryRange2}
    
    
    dataset0Size = len(interactions0)
    dataset1Size = len(interactions1)
    dataset2Size = len(interactions2)
    
    
    #
    # get the strings to be encoded
    #
    inputStrings0, outputStrings0, inputCustomerLocationVectors0, outputShopkeeperLocationVectors0, locationLabelEncoder = get_input_output_strings_and_location_vectors(interactions0)
    inputStrings1, outputStrings1, inputCustomerLocationVectors1, outputShopkeeperLocationVectors1, _ = get_input_output_strings_and_location_vectors(interactions1)
    inputStrings2, outputStrings2, inputCustomerLocationVectors2, outputShopkeeperLocationVectors2, _  = get_input_output_strings_and_location_vectors(interactions2)
    
    dbStrings0 = get_database_value_strings(database0, dbFieldnames)
    dbStrings1 = get_database_value_strings(database1, dbFieldnames)
    dbStrings2 = get_database_value_strings(database2, dbFieldnames)
    
    
    #
    # create a character encoder
    #
    print("creating character encoder...", flush=True, file=sessionTerminalOutputStream)
    
    allStrings = inputStrings0 + outputStrings0
    allStrings += inputStrings1 + outputStrings1
    allStrings += inputStrings2 + outputStrings2
    
    for row in dbStrings0+dbStrings1+dbStrings2:
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
    allInputLens = [len(i) for i in inputStrings0+inputStrings1+inputStrings2]
    maxInputLen = max(allInputLens)
    
    # find max output length
    allOutputLens = [len(i) for i in outputStrings0+outputStrings1+outputStrings2]
    maxOutputLen = max(allOutputLens)
    
    # find max DB value length
    allDbValueLens = []
    
    for row in dbStrings0+dbStrings1+dbStrings2:
        allDbValueLens += [len(v) for v in row]
    
    maxDbValueLen = max(allDbValueLens)
    
    
    # vectorize the inputs
    _, inputIndexLists0 = vectorize_sentences(inputStrings0, charToIndex, maxInputLen)
    _, inputIndexLists1 = vectorize_sentences(inputStrings1, charToIndex, maxInputLen)
    _, inputIndexLists2 = vectorize_sentences(inputStrings2, charToIndex, maxInputLen)
    
    # vectorize the outputs
    _, outputIndexLists0 = vectorize_sentences(outputStrings0, charToIndex, maxOutputLen)
    _, outputIndexLists1 = vectorize_sentences(outputStrings1, charToIndex, maxOutputLen)
    _, outputIndexLists2 = vectorize_sentences(outputStrings2, charToIndex, maxOutputLen)
    
    # vectorize the DB values
    dbVectors0, _ = vectorize_databases(dbStrings0, charToIndex, maxDbValueLen)
    dbVectors1, _ = vectorize_databases(dbStrings1, charToIndex, maxDbValueLen)
    dbVectors2, _ = vectorize_databases(dbStrings2, charToIndex, maxDbValueLen)        
    
    
    #
    # split into training and testing sets
    #
    
    # training
    trainInputIndexLists = inputIndexLists0 + inputIndexLists1
    trainOutputIndexLists = outputIndexLists0 + outputIndexLists1
    trainOutputStrings = outputStrings0 + outputStrings1
    
    trainInputCustomerLocations = np.concatenate([inputCustomerLocationVectors0, inputCustomerLocationVectors1])
    trainOutputShopkeeperLocations = np.concatenate([outputShopkeeperLocationVectors0, outputShopkeeperLocationVectors1])
    
    
    trainDbVectors = []
    
    for i in range(len(inputIndexLists0)):
        trainDbVectors.append(dbVectors0)

    for i in range(len(inputIndexLists1)):
        trainDbVectors.append(dbVectors1)
    
    
    # testing
    testInputIndexLists = inputIndexLists2
    testOutputIndexLists = outputIndexLists2
    testOutputStrings = outputStrings2
    
    testInputCustomerLocations = inputCustomerLocationVectors2
    testOutputShopkeeperLocations = outputShopkeeperLocationVectors2
    
    
    testDbVectors = []
    
    for i in range(len(inputIndexLists2)):
        testDbVectors.append(dbVectors2)
    
    
    print(len(trainInputIndexLists), "training examples", flush=True, file=sessionTerminalOutputStream)
    print(len(testInputIndexLists), "testing examples", flush=True, file=sessionTerminalOutputStream)
    print(maxInputLen, "input sequence length", flush=True, file=sessionTerminalOutputStream)
    print(maxOutputLen, "output sequence length", flush=True, file=sessionTerminalOutputStream)
    print(maxDbValueLen, "DB value sequence length", flush=True, file=sessionTerminalOutputStream)
    print(len(charToIndex), "unique characters", flush=True, file=sessionTerminalOutputStream)
    
    
    #
    # setup the model
    #
    print("setting up the model...", flush=True, file=sessionTerminalOutputStream)
    
    batchSize = 256
    embeddingSize = 30
    
    
    inputLen = maxInputLen + 2
    outputLen = maxOutputLen + 2
    dbValLen = maxDbValueLen + 1
    
    locationVecLen = locationLabelEncoder.classes_.size
    
    
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
    
    learner = learning.CustomNeuralNetwork(inputSeqLen=inputLen, 
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
    trainInputIndexLists = trainInputIndexLists[:1500]
    trainOutputIndexLists = trainOutputIndexLists[:1500]
    trainDbVectors = trainDbVectors[:1500]
    
    testInputIndexLists = testInputIndexLists[:1500]
    testOutputIndexLists = testOutputIndexLists[:1500]
    testDbVectors = testDbVectors[:1500]
    """
    
    numEpochs = 20000
    trainBatchEndIndices = list(range(batchSize, len(trainInputIndexLists), batchSize))
    testBatchEndIndices = list(range(batchSize, len(testInputIndexLists), batchSize))
    
    
    camerasOfInterest = ["CAMERA_1", "CAMERA_2"]
    featuresOfInterest = ["price"] #, "camera_type"]
    
    #
    interestingTrainInstances = []
    
    for f in featuresOfInterest:
        
        # from DB 0
        for c in camerasOfInterest:
            for i in range(len(interactions0)):
                if interactions0[i]["CUSTOMER_TOPIC"] == f and interactions0[i]["CURRENT_CAMERA_OF_CONVERSATION"] == c:
                    interestingTrainInstances.append((i, "{} {} {}".format("DB0", c, f)))
                    break
        
        # from DB 1
        for c in camerasOfInterest:
            for i in range(len(interactions1)):
                if interactions1[i]["CUSTOMER_TOPIC"] == f and interactions1[i]["CURRENT_CAMERA_OF_CONVERSATION"] == c:
                    interestingTrainInstances.append((i+dataset0Size, "{} {} {}".format("DB1", c, f)))
                    break
    
    #
    interestingTestInstances = []
    
    for f in featuresOfInterest:
        
        # from DB 2
        for c in camerasOfInterest:
            for i in range(len(interactions2)):
                if interactions2[i]["CUSTOMER_TOPIC"] == f and interactions2[i]["CURRENT_CAMERA_OF_CONVERSATION"] == c:
                    interestingTestInstances.append((i, "{} {} {}".format("DB2", c, f)))
                    break
    
    
    """
    interestingTrainInstances = [(898, "DB0, Cam1, Price"),
                                 (812, "DB0, Cam2, Price"),
                                 (5+len(inputStrings0), "DB1, Cam1, Price"),
                                 (237+len(inputStrings0), "DB1, Cam2, Price"),
                                 
                                 (1004, "DB0, Cam1, AF"),
                                 (1024, "DB0, Cam2, AF"),
                                 (4+len(inputStrings0), "DB1, Cam1, AF"),
                                 (13+len(inputStrings0), "DB1, Cam2, AF")
                                 ]
    
    interestingTestInstances = [(203, "DB2, Cam1, Price"),
                                (163, "DB2, Cam2, Price"),
                                
                                (171, "DB2, Cam1, AF"),
                                (234, "DB2, Cam2, AF")
                                ]
    """
    
    
    # write header in csv log file
    with open(sessionLogFile, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", 
                         "Train Cost Ave ({})".format(sessionIdentifier), 
                         "Train Cost SD ({})".format(sessionIdentifier), 
                         "Train DB Substring Correct All ({})".format(sessionIdentifier), 
                         "Train DB Substring Correct Ave ({})".format(sessionIdentifier), 
                         "Train DB Substring Correct SD ({})".format(sessionIdentifier), 
                         
                         "Test Cost Ave({})".format(sessionIdentifier),
                         "Test Cost SD ({})".format(sessionIdentifier), 
                         "Test DB Substring Correct All ({})".format(sessionIdentifier), 
                         "Test DB Substring Correct Ave ({})".format(sessionIdentifier), 
                         "Test DB Substring Correct SD ({})".format(sessionIdentifier), 
                         ])
    
    
    for e in range(numEpochs):
        
        startTime = time.time()
        
        #
        # train
        #
        trainCosts = []
        
        # shuffle the training data
        
        temp = list(zip(trainInputIndexLists, trainDbVectors, trainOutputIndexLists, trainInputCustomerLocations, trainOutputShopkeeperLocations))
        random.shuffle(temp)
        trainInputIndexLists_shuf, trainDbVectors_shuf, trainOutputIndexLists_shuf, trainInputCustomerLocations_shuf, trainOutputShopkeeperLocations_shuf = zip(*temp)
        
        
        for i in trainBatchEndIndices:
            
            batchTrainCost = learner.train(trainInputIndexLists_shuf[i-batchSize:i],
                                           trainInputCustomerLocations_shuf[i-batchSize:i],
                                           trainDbVectors_shuf[i-batchSize:i], 
                                           trainOutputIndexLists_shuf[i-batchSize:i],
                                           trainOutputShopkeeperLocations_shuf[i-batchSize:i])
            
            trainCosts.append(batchTrainCost)
            #print("\t", batchTrainCost, flush=True, file=sessionTerminalOutputStream)
            #break
        trainCostAve = np.mean(trainCosts)
        trainCostStd = np.std(trainCosts)
        print(e, "train cost", trainCostAve, trainCostStd, flush=True, file=sessionTerminalOutputStream)
        
        
        if e % 50 == 0 or e == (numEpochs-1):
            
            saveModelDir = tools.create_directory(sessionDir+"/{}".format(e))
            learner.save(saveModelDir+"/saved_session".format(e))
            
            #
            # compute accuracy, etc.
            #
            
            # TRAIN
            
            trainUttPreds = []
            trainLocPreds = []
            trainCopyScores = []
            trainGenScores = []
            trainCamMatchArgMax = []
            trainAttMatchArgMax = []
            trainCamMatch = []
            trainAttMatch  = []
            trainGtSents = []
            
            for i in trainBatchEndIndices:
                
                batchTrainUttPreds, batchTrainLocPreds, batchTrainCopyScores, batchTrainGenScores, batchTrainCamMatchArgMax, batchTrainAttMatchArgMax, batchTrainCamMatch, batchTrainAttMatch = learner.predict(trainInputIndexLists[i-batchSize:i], 
                                                                                                                                                                                         trainInputCustomerLocations[i-batchSize:i],
                                                                                                                                                                                         trainDbVectors[i-batchSize:i], 
                                                                                                                                                                                         trainOutputIndexLists[i-batchSize:i],
                                                                                                                                                                                         trainOutputShopkeeperLocations[i-batchSize:i])
                
                trainUttPreds.append(batchTrainUttPreds)
                trainLocPreds.append(batchTrainLocPreds)
                trainCopyScores.append(batchTrainCopyScores)
                trainGenScores.append(batchTrainGenScores)
                trainCamMatchArgMax.append(batchTrainCamMatchArgMax)
                trainAttMatchArgMax.append(batchTrainAttMatchArgMax)
                trainCamMatch.append(batchTrainCamMatch)
                trainAttMatch.append(batchTrainAttMatch)
                trainGtSents += trainOutputStrings[i-batchSize:i]
                
            trainUttPreds = np.concatenate(trainUttPreds)
            trainLocPreds = np.concatenate(trainLocPreds)
            trainCopyScores = np.concatenate(trainCopyScores)
            trainGenScores = np.concatenate(trainGenScores)
            trainCamMatchArgMax = np.concatenate(trainCamMatchArgMax)
            trainAttMatchArgMax = np.concatenate(trainAttMatchArgMax)
            trainCamMatch = np.concatenate(trainCamMatch)
            trainAttMatch  = np.concatenate(trainAttMatch)
            
            trainPredSents = unvectorize_sentences(trainUttPreds, indexToChar)
            
            
            # compute how much of the substring that should be copied from the DB is correct
            trainSubstrCharMatchAccs = compute_db_substring_match(trainGtSents, trainPredSents, shkpUttToDbEntryRange)
            
            # how many 100% correct substrings
            trainSubstrAllCorr = trainSubstrCharMatchAccs.count(1.0) / len(trainSubstrCharMatchAccs)
            
            # average substring correctness
            trainSubstrAveCorr = np.mean(trainSubstrCharMatchAccs)
            trainSubstrSdCorr = np.std(trainSubstrCharMatchAccs)
            
            print("train substr all correct {}, ave correctness {} ({})".format(trainSubstrAllCorr, trainSubstrAveCorr, trainSubstrSdCorr), flush=True, file=sessionTerminalOutputStream)
            
            
            #trainAcc = 0.0
            #for i in range(len(trainUttPreds)):
            #    trainAcc += normalized_edit_distance(trainOutputIndexLists[i], trainUttPreds[i])
            #trainAcc /= len(trainOutputIndexLists)
            
            #trainPredsFlat = np.array(trainUttPreds).flatten()
            #trainAcc = metrics.accuracy_score(trainPredsFlat, trainGroundTruthFlat)
            
            trainUttPreds_ = []
            trainLocPreds_ = []
            trainCopyScores_ = []
            trainGenScores_ = []
            trainCamMatchArgMax_ = []
            trainAttMatchArgMax_ = []
            trainCamMatch_ = []
            trainAttMatch_  = []
            
            trainOutputIndexLists_ = []
            trainPredSents_ = []
            
            for tup in interestingTrainInstances:
                i = tup[0]
                info = tup[1]
                
                trainUttPreds_.append(trainUttPreds[i])
                trainLocPreds_.append(trainLocPreds[i])
                trainCopyScores_.append(trainCopyScores[i])
                trainGenScores_.append(trainGenScores[i])
                trainCamMatchArgMax_.append(trainCamMatchArgMax[i])
                trainAttMatchArgMax_.append(trainAttMatchArgMax[i])
                trainCamMatch_.append(trainCamMatch[i])
                trainAttMatch_.append(trainAttMatch[i])
                
                trainOutputIndexLists_.append(trainOutputIndexLists[i])
                trainPredSents_.append(trainPredSents[i])
                
            
            trainPredSentsColored, trainCopyScoresPred, trainGenScoresPred, trainCopyScoresTrue, trainGenScoresTrue = color_results(trainPredSents_, 
                                                                                                                                    trainOutputIndexLists_,
                                                                                                                                    trainCopyScores_, 
                                                                                                                                    trainGenScores_, 
                                                                                                                                    charToIndex)
            
            
            # TEST
            testUttPreds = []
            testLocPreds = []
            testCopyScores = []
            testGenScores = []
            testCamMatchArgMax = []
            testAttMatchArgMax = []
            testCamMatch = []
            testAttMatch  = []
            testGtSents = []
            
            testCosts = []
            
            for i in testBatchEndIndices:
                
                #print(i-batchSize, i, flush=True, file=sessionTerminalOutputStream)
                
                batchTestCost = learner.loss(testInputIndexLists[i-batchSize:i], 
                                             testInputCustomerLocations[i-batchSize:i], 
                                             testDbVectors[i-batchSize:i], 
                                             testOutputIndexLists[i-batchSize:i],
                                             testOutputShopkeeperLocations[i-batchSize:i])
                
                testCosts.append(batchTestCost)
                #print("\t", batchTestCost, flush=True, file=sessionTerminalOutputStream)
                
                
                batchTestUttPreds, batchTestLocPreds, batchTestCopyScores, batchTestGenScores, batchTestCamMatchArgMax, batchTestAttMatchArgMax, batchTestCamMatch, batchTestAttMatch = learner.predict(testInputIndexLists[i-batchSize:i], 
                                                                                                                                                                                  testInputCustomerLocations[i-batchSize:i],
                                                                                                                                                                                  testDbVectors[i-batchSize:i], 
                                                                                                                                                                                  testOutputIndexLists[i-batchSize:i],
                                                                                                                                                                                  testOutputShopkeeperLocations[i-batchSize:i])
                
                testUttPreds.append(batchTestUttPreds)
                testLocPreds.append(batchTestLocPreds)
                testCopyScores.append(batchTestCopyScores)
                testGenScores.append(batchTestGenScores)
                testCamMatchArgMax.append(batchTestCamMatchArgMax)
                testAttMatchArgMax.append(batchTestAttMatchArgMax)
                testCamMatch.append(batchTestCamMatch)
                testAttMatch.append(batchTestAttMatch)
                testGtSents += testOutputStrings[i-batchSize:i]
                
                
            testUttPreds = np.concatenate(testUttPreds)
            testLocPreds = np.concatenate(testLocPreds)
            testCopyScores = np.concatenate(testCopyScores)
            testGenScores = np.concatenate(testGenScores)
            testCamMatchArgMax = np.concatenate(testCamMatchArgMax)
            testAttMatchArgMax = np.concatenate(testAttMatchArgMax)
            testCamMatch = np.concatenate(testCamMatch)
            testAttMatch  = np.concatenate(testAttMatch)
            
            
            testCostAve = np.mean(testCosts)
            testCostStd = np.std(testCosts)
            print("test cost", testCostAve, testCostStd, flush=True, file=sessionTerminalOutputStream)
            
            
            testPredSents = unvectorize_sentences(testUttPreds, indexToChar)
            
            
            # compute how much of the substring that should be copied from the DB is correct
            testSubstrCharMatchAccs = compute_db_substring_match(testGtSents, testPredSents, shkpUttToDbEntryRange)
            
            # how many 100% correct substrings
            testSubstrAllCorr = testSubstrCharMatchAccs.count(1.0) / len(testSubstrCharMatchAccs)
            
            # average substring correctness
            testSubstrAveCorr = np.mean(testSubstrCharMatchAccs)
            testSubstrSdCorr = np.std(testSubstrCharMatchAccs)
            
            print("test substr all correct {}, ave correctness {} ({})".format(testSubstrAllCorr, testSubstrAveCorr, testSubstrSdCorr), flush=True, file=sessionTerminalOutputStream)
            
            
            #testAcc = 0.0
            #for i in range(len(testUttPreds)):
            #    testAcc += normalized_edit_distance(testOutputIndexLists[i], testUttPreds[i])
            #testAcc /= len(testOutputIndexLists)
            
            #testPredsFlat = np.array(testUttPreds).flatten()
            #testAcc = metrics.accuracy_score(testPredsFlat, testGroundTruthFlat)
            
            testUttPreds_ = []
            testLocPreds_ = []
            testCopyScores_ = []
            testGenScores_ = []
            testCamMatchArgMax_ = []
            testAttMatchArgMax_ = []
            testCamMatch_ = []
            testAttMatch_  = []
            
            testOutputIndexLists_ = []
            testPredSents_ = []
            
            for tup in interestingTestInstances:
                i = tup[0]
                info = tup[1]
                
                testUttPreds_.append(testUttPreds[i])
                testLocPreds_.append(testLocPreds[i])
                testCopyScores_.append(testCopyScores[i])
                testGenScores_.append(testGenScores[i])
                testCamMatchArgMax_.append(testCamMatchArgMax[i])
                testAttMatchArgMax_.append(testAttMatchArgMax[i])
                testCamMatch_.append(testCamMatch[i])
                testAttMatch_.append(testAttMatch[i])
                
                testOutputIndexLists_.append(testOutputIndexLists[i])
                testPredSents_.append(testPredSents[i])
                
            
            testPredSentsColored, testCopyScoresPred, testGenScoresPred, testCopyScoresTrue, testGenScoresTrue = color_results(testPredSents_, 
                                                                                                                               testOutputIndexLists_,
                                                                                                                               testCopyScores_, 
                                                                                                                               testGenScores_, 
                                                                                                                               charToIndex)
            
            
            print("****************************************************************", flush=True, file=sessionTerminalOutputStream)
            print("TRAIN", e, round(trainCostAve, 3), round(trainCostStd, 3), flush=True, file=sessionTerminalOutputStream)
            
            for i in range(len(interestingTrainInstances)):
                index = interestingTrainInstances[i][0]
                info = interestingTrainInstances[i][1]
                
                print("TRUE:", locationLabelEncoder.classes_[np.argmax(trainOutputShopkeeperLocations[index])], trainOutputStrings[index], flush=True, file=sessionTerminalOutputStream)
                print("PRED:", locationLabelEncoder.classes_[trainLocPreds_[i]], trainPredSents_[i], flush=True, file=sessionTerminalOutputStream)
                
                print(np.round(trainCamMatch_[i], 3), flush=True, file=sessionTerminalOutputStream)
                print(np.round(trainAttMatch_[i], 3), flush=True, file=sessionTerminalOutputStream)
                print("best match:", cameras[trainCamMatchArgMax[i]], dbFieldnames[trainAttMatchArgMax[i]], flush=True, file=sessionTerminalOutputStream)
                print("true match:", info, flush=True, file=sessionTerminalOutputStream)
                
                #print("\x1b[31m"+ indexToCam[trainCamMatchArgMax[i]] +" "+ indexToAtt[trainAttMatchArgMax[i]] +" "+ databases[i][indexToCam[trainCamMatchArgMax[i]]][indexToAtt[trainAttMatchArgMax[i]]] +"\x1b[0m", trainCamMatch[i], trainAttMatch[i])
                print("", flush=True, file=sessionTerminalOutputStream) 
                
            print("", flush=True, file=sessionTerminalOutputStream)
            
            
            print("TEST", e, round(testCostAve, 3), round(testCostStd, 3), flush=True, file=sessionTerminalOutputStream)
            
            for i in range(len(interestingTestInstances)):
                index = interestingTestInstances[i][0]
                info = interestingTestInstances[i][1]
                
                print("TRUE:", locationLabelEncoder.classes_[np.argmax(testOutputShopkeeperLocations[index])], testOutputStrings[index], flush=True, file=sessionTerminalOutputStream)
                print("PRED:", locationLabelEncoder.classes_[testLocPreds_[i]], testPredSents_[i], flush=True, file=sessionTerminalOutputStream)
                
                print(np.round(testCamMatch_[i], 3), flush=True, file=sessionTerminalOutputStream)
                print(np.round(testAttMatch_[i], 3), flush=True, file=sessionTerminalOutputStream)
                print("best match:", cameras[testCamMatchArgMax[i]], dbFieldnames[testAttMatchArgMax[i]], flush=True, file=sessionTerminalOutputStream)
                print("true match:", info, flush=True, file=sessionTerminalOutputStream)
                
                #print("\x1b[31m"+ indexToCam[testCamMatchArgMax[i]] +" "+ indexToAtt[testAttMatchArgMax[i]] +" "+ databases[i+4][indexToCam[testCamMatchArgMax[i]]][indexToAtt[testAttMatchArgMax[i]]] +"\x1b[0m", testCamMatch[i], testAttMatch[i])
                print("", flush=True, file=sessionTerminalOutputStream)
            
            print("****************************************************************", flush=True, file=sessionTerminalOutputStream)
            
            
            
            #
            # save to file
            #
            with open(sessionDir+"/{:}_outputs.csv".format(e), "w", newline="") as csvfile:
                
                writer = csv.writer(csvfile)
                
                
                writer.writerow(["TRAIN", round(trainCostAve, 3), round(trainCostStd, 3)])
                
                for i in range(len(interestingTrainInstances)):
                    index = interestingTrainInstances[i][0]
                    info = interestingTrainInstances[i][1]
                    
                    
                    writer.writerow(["TRUE:"] + 
                                    [locationLabelEncoder.classes_[np.argmax(trainOutputShopkeeperLocations[index])]] +
                                    [c for c in trainOutputStrings[index]])
                    
                    writer.writerow(["PRED:"] + 
                                    [locationLabelEncoder.classes_[trainLocPreds_[i]]] +
                                    [c for c in trainPredSents_[i]])
                    
                    writer.writerow(["PRED COPY:"] + [""] + [c for c in trainCopyScoresPred[i]])
                    writer.writerow(["PRED GEN:"] + [""] + [c for c in trainGenScoresPred[i]])
                    
                    writer.writerow(["TRUE COPY:"] + [""] + [c for c in trainCopyScoresTrue[i]])
                    writer.writerow(["TRUE GEN:"] + [""] + [c for c in trainGenScoresTrue[i]])
                    
                    writer.writerow(["CAM MATCH:"] + [np.round(p, 3) for p in trainCamMatch_[i]])
                    writer.writerow(["ATT MATCH:"] + [np.round(p, 3) for p in trainAttMatch_[i]])
                    
                    writer.writerow(["TOP MATCH:", cameras[trainCamMatchArgMax[i]]+" "+dbFieldnames[trainAttMatchArgMax[i]]])
                    writer.writerow(["TRUE MATCH:", info])
                    
                    writer.writerow([])
                
                
                writer.writerow(["TEST", round(testCostAve, 3), round(testCostStd, 3)])
                
                for i in range(len(interestingTestInstances)):
                    index = interestingTestInstances[i][0]
                    info = interestingTestInstances[i][1]
                    
                    
                    writer.writerow(["TRUE:"] + 
                                    [locationLabelEncoder.classes_[np.argmax(testOutputShopkeeperLocations[index])]] +
                                    [c for c in testOutputStrings[index]])
                    
                    writer.writerow(["PRED:"] + 
                                    [locationLabelEncoder.classes_[testLocPreds_[i]]] +
                                    [c for c in testPredSents_[i]])
                    
                    
                    writer.writerow(["PRED COPY:"] + [""] +[c for c in testCopyScoresPred[i]])
                    writer.writerow(["PRED GEN:"] + [""] +[c for c in testGenScoresPred[i]])
                    
                    writer.writerow(["TRUE COPY:"] + [""] +[c for c in testCopyScoresTrue[i]])
                    writer.writerow(["TRUE GEN:"] + [""] +[c for c in testGenScoresTrue[i]])
                    
                    writer.writerow(["CAM MATCH:"] + [np.round(p, 3) for p in testCamMatch_[i]])
                    writer.writerow(["ATT MATCH:"] + [np.round(p, 3) for p in testAttMatch_[i]])
                    
                    writer.writerow(["TOP MATCH:", cameras[testCamMatchArgMax[i]]+" "+dbFieldnames[testAttMatchArgMax[i]]])
                    writer.writerow(["TRUE MATCH:", info])
                    
                    writer.writerow([])
        
        
            # append to session log   
            with open(sessionLogFile, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([e,                 #"Epoch", 
                                 trainCostAve,      #"Train Cost Ave ({})".format(seed), 
                                 trainCostStd,      #"Train Cost SD ({})".format(seed), 
                                 trainSubstrAllCorr,    #"Train DB Substring Correct All ({})".format(seed), 
                                 trainSubstrAveCorr,    #"Train DB Substring Correct Ave ({})".format(seed), 
                                 trainSubstrSdCorr, #"Train DB Substring Correct SD ({})".format(seed), 
                                 
                                 testCostAve,       #"Test Cost Ave({})".format(seed),
                                 testCostStd,       #"Test Cost SD ({})".format(seed), 
                                 testSubstrAllCorr, #"Test DB Substring Correct All ({})".format(seed), 
                                 testSubstrAveCorr, #"Test DB Substring Correct Ave ({})".format(seed), 
                                 testSubstrSdCorr,  #"Test DB Substring Correct SD ({})".format(seed), 
                                 ])    
        
        
        print("epoch time", round(time.time() - startTime, 2), flush=True, file=sessionTerminalOutputStream)
        



if __name__ == "__main__":
    
    
    camTemp = 3
    attTemp = 6
    
    
    
    for gpu in range(8):
        
        seed = gpu
        
            
        process = Process(target=run, args=[gpu, seed, camTemp, attTemp, sessionDir])
        process.start()
        
    




