'''
Created on Oct 17, 2017

@author: MalcolmD
'''

import time
import os
import csv
from scipy.stats import entropy


# for Malcolm desktop (robovie)
projectDir = "C:/Users/robovie/eclipse-workspace/DatabaseLearning/"
logDir = "E:/eclipse-log"

# for malcolm @ gpgpu1
#projectDir = "/home/malcolm//eclipse-workspace/DatabaseLearning/"
#logDir = "/home/malcolm/eclipse-log"


dataDir = projectDir + "data/"
modelDir = projectDir + "models/"

punctuation = r"""!"#%&()*+,:;<=>?@[\]^_`{|}~""" # leave in $ . / - '



def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def time_now():
    return time.strftime("%Y-%m-%d_%H-%M-%S")



def create_directory(dirName):
    '''
    Creates the directory if it does not already exist.
    '''
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    
    return dirName



def append_to_log(text, directory):
    print("    " + text)
    
    f = open(directory+"/log.txt", "a")
    f.write(text+"\n")
    f.close()



def create_session_dir(dirname):
    now = time_now()
    sessionDirname = logDir + "/" + now + "_" + dirname
    create_directory(sessionDirname)
    
    return sessionDirname



def jensen_shannon_divergence(probdistrP, probdistrQ):
    
    if len(probdistrP) != len(probdistrQ):
        return None
    
    probdistrM = (probdistrP + probdistrQ) / 2.0
    
    jsd = (entropy(probdistrP, probdistrM) + entropy(probdistrQ, probdistrM)) / 2.0
    
    return jsd



def color_non_matching(inputSent, decodedSent):
    
    # find letters that don't match
    nonMatchingIndices = []
    for i in range(min(len(inputSent),len(decodedSent))):
        
        if inputSent[i] != decodedSent[i]:
            nonMatchingIndices.append(i)
    
    nonMatchingIndices.reverse()
    
    if len(decodedSent) > len(inputSent):
        decodedSent = decodedSent[:len(inputSent)] + "\x1b[31m" + decodedSent[len(inputSent):] + "\x1b[0m"
    
    for index in nonMatchingIndices:
        
        temp = decodedSent[:index] + "\x1b[31;1m" + decodedSent[index] + "\x1b[0m" + decodedSent[index+1:] 
        
        decodedSent = temp
    
    return decodedSent



def load_passive_proactive_data(filename):
    
    data = []
    
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            data.append(row)
    
    data.sort(key=lambda x: x["TIMESTAMP"])
    
    
    interactionMap = {}
    
    for datum in data:
        
        trialId = int(datum["TRIAL"])
        
        if trialId not in interactionMap:
            interactionMap[trialId] = []
        
        interactionMap[trialId].append(datum)
    
    return interactionMap



def read_crossvalidation_splits(numFolds):
    
    dataSplits = {}
    
    splitDir = dataDir + "folds/{:}".format(numFolds)
    
    for fold in range(numFolds):
        
        dataSplits[fold] = {}
        
        trainFile = open(splitDir + "/{:}_train.txt".format(fold))
        testFile = open(splitDir + "/{:}_test.txt".format(fold))
        
        dataSplits[fold]["trainExpIds"] = [int(expId) for expId in trainFile.readline().strip().split(",")]
        dataSplits[fold]["testExpIds"] = [int(expId) for expId in testFile.readline().strip().split(",")]
        
        trainFile.close()
        testFile.close()
        
    return dataSplits


def load_shopkeeper_speech_clusters(shopkeeperSpeechClusterFilename):
    shkpUttToSpeechClustId = {}
    speechClustIdToShkpUtts = {}
    shkpSpeechClustIdToRepUtt = {}
    
    goodSpeechClusterIds = []
    junkSpeechClusterIds = []
    
    with open(shopkeeperSpeechClusterFilename, encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            
            
            for row in reader:
                utt = row["Utterance"].lower()
                speechClustId = int(row["Cluster.ID"])
            
                if utt not in shkpUttToSpeechClustId:
                    shkpUttToSpeechClustId[utt] = speechClustId
                
                elif utt in shkpUttToSpeechClustId:
                    if speechClustId != shkpUttToSpeechClustId[utt]:
                        print("WARNING: Shopkeeper utterance \"{}\" is in multiple speech clusters!".format(utt))
                
                
                if speechClustId not in speechClustIdToShkpUtts:
                    speechClustIdToShkpUtts[speechClustId] = []
                speechClustIdToShkpUtts[speechClustId].append(utt)
                
                
                if row["IS_NEW_REPRESENTATIVE"] == "1":
                    shkpSpeechClustIdToRepUtt[speechClustId] = utt
                
                if row["Is.Junk"] == "0" and speechClustId not in goodSpeechClusterIds:
                    goodSpeechClusterIds.append(speechClustId)
                
                if row["Is.Junk"] == "1" and speechClustId not in junkSpeechClusterIds:
                    junkSpeechClusterIds.append(speechClustId)
                
                
                
    # if any of the utterances in the cluster are marked as not junk, then treat this as a normal cluster
    # (junk utterances that were sorted into good clusters still have there original junk marking)
    junkSpeechClusterIds = [clustId for clustId in junkSpeechClusterIds if clustId not in goodSpeechClusterIds]
    
    
    # add a cluster for no speech
    shkpUttToSpeechClustId[""] = len(shkpSpeechClustIdToRepUtt)
    shkpSpeechClustIdToRepUtt[shkpUttToSpeechClustId[""]] = ""
    
    return shkpUttToSpeechClustId, shkpSpeechClustIdToRepUtt, speechClustIdToShkpUtts, junkSpeechClusterIds


def load_shopkeeper_action_clusters(shopkeeperActionClusterFilename):
    shkpActionIdToTuple = {}
    tupleToShkpActionId = {}
    
    with open(shopkeeperActionClusterFilename, encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            actionClusterId = int(row["ACTION_CLUSTER_ID"])
            
            shkpActionIdToTuple[actionClusterId] = (int(row["SPEECH_CLUSTER_ID"]),
                                                    row["OUTPUT_SHOPKEEPER_LOCATION"],
                                                    row["OUTPUT_SPATIAL_STATE"],
                                                    row["OUTPUT_STATE_TARGET"])
            
            tupleToShkpActionId[shkpActionIdToTuple[actionClusterId]] = actionClusterId
            
    return shkpActionIdToTuple, tupleToShkpActionId
                

if __name__ == '__main__':
    pass




