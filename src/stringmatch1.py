'''
Created on Nov 14, 2019

@author: Malcolm
'''


import csv
import os
import string
import numpy as np
import fuzzysearch
import copy
from sklearn.metrics import accuracy_score
import nltk
from nltk.stem import WordNetLemmatizer

import tools


sessionDir = tools.create_session_dir("stringmatch1")

dataDirectory = tools.dataDir+"2019-11-12_17-40-29_advancedSimulator9"
numTrainDbs = 10
numInteractionsPerDb = 200

cameras = ["CAMERA_1", "CAMERA_2", "CAMERA_3"]



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



def tokenize_utterance(utt, lemmatize=False):
    tokenized = nltk.word_tokenize(utt)
    
    if lemmatize:
        lemmatized = []
        
        for t in tokenized:
            lemmatized.append(lemmatize_word(t))
        
        if len(tokenized) != len(lemmatized):
            print("WARNING: Some tokens were lost while lemmatizing..." )
        
        return lemmatized
    
    else:
        return tokenized


wnl = WordNetLemmatizer()

def lemmatize_word(w):
    return wnl.lemmatize(w).lower()


#
# read in the utterance data
#
print ("loading the data...")

filenames = os.listdir(dataDirectory)
filenames.sort()

databaseFilenamesAll = [dataDirectory+"/"+fn for fn in filenames if "handmade_database" in fn]
interactionFilenamesAll = [dataDirectory+"/"+fn for fn in filenames if "simulated_data_csshkputts" in fn]

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
    
    inters, sutder, gtDbCamera, gtDbAttribute = read_simulated_interactions(iFn, dbFieldnames, numInteractionsPerDb)
    
    interactions.append(inters)
    datasetSizes.append(len(inters))
    
    gtDatabaseCameras.append(gtDbCamera)
    gtDatabaseAttributes.append(gtDbAttribute)
    
    allCustUtts += [row["CUSTOMER_SPEECH"] for row in inters]
    allShkpUtts += [row["SHOPKEEPER_SPEECH"] for row in inters]



#
# find each unique combination of shopkeeper utterance and DB ID - to be used for matching
#
print("finding unique combinations of shopkeeper utterance and DB IDs...")

uniqueUttDbCombos = []
uttToTokenized = {}

for i in range(len(interactions)):
    for j in range(len(interactions[i])):
        
        shkpUtt = interactions[i][j]["SHOPKEEPER_SPEECH"]
        dbId = interactions[i][j]["DATABASE_ID"]
        
        gtDbCamera = gtDatabaseCameras[i][j]
        gtDbAttribute = gtDatabaseAttributes[i][j]
        gtDbContents = interactions[i][j]["DATABASE_CONTENTS"]
        
        gtShkpAction = interactions[i][j]["OUTPUT_SHOPKEEPER_ACTION"]
        
        if gtShkpAction == "S_INTRODUCES_CAMERA":
            gtDbCamera = cameras.index(interactions[i][j]["OUTPUT_STATE_TARGET"])
            gtDbAttribute = dbFieldnames.index("camera_name")
        
        combo = (shkpUtt, dbId, gtDbCamera, gtDbAttribute, gtDbContents, gtShkpAction)
        
        if combo not in uniqueUttDbCombos and shkpUtt != "":
            uniqueUttDbCombos.append(combo)
        
        if shkpUtt not in uttToTokenized:
            uttToTokenized[shkpUtt] = tokenize_utterance(shkpUtt, lemmatize=True)
        
        


#
# do the string matching...
# for each unique shkp utt, DB pair find the closest matching substring in the utt to each DB entry
#
print("string matching...")

numDbRows = len(cameras)
numDbCols = len(dbFieldnames)

comboToStringMatches = {}

judgmentCounts = {}

for combo in uniqueUttDbCombos:
    
    # store the match score and match substring for each DB entry
    
    shkpUtt = combo[0]
    dbId = int(combo[1])
    gtDbCamIndex = combo[2]
    gtDbAttrIndex = combo[3]
    gtDbContents = combo[4].lower().translate(str.maketrans('', '', string.punctuation))
    gtShkpAction = combo[5]
    
    if gtShkpAction not in judgmentCounts:
        judgmentCounts[gtShkpAction] = {}
    
    
    # to store the matches
    #substringMatches = np.ones(shape=(numDbRows, numDbCols), dtype=object)
    #substringMatches[:] = ""
    #substringMatchScores = np.ones(shape=(numDbRows, numDbCols), dtype=np.float64) * np.inf
    
    potentialMatches = []
    
    for candCamIndex in range(len(cameras)):
        candCam = cameras[candCamIndex]   
        for candAttrIndex in range(1, len(dbFieldnames)): # skip the camera_ID field
            
            candAttr = dbFieldnames[candAttrIndex]
            candDbContents = databases[dbId][candCamIndex][candAttr].lower().translate(str.maketrans('', '', string.punctuation))
                
            # if the candidate DB contents is empty, we don't have to find a match
            if candDbContents == "":
                continue
            
            
            # the substring search function requires a max distance parameter
            # TODO: what's the best way to set this?
            maxDist = int(np.ceil(len(candDbContents)/5))
            
            # get the candidate shkp utt substring matches
            candMatches = fuzzysearch.find_near_matches(candDbContents, shkpUtt, max_l_dist=maxDist)
            
            
            # what is the ground truth for whether the DB contents should be found in this shkp utt?
            gtIsMatch = candDbContents == gtDbContents
            
            
            # print the match results...
            """
            if gtIsMatch or len(candMatches) > 0:
                print("SHKP UTT:", shkpUtt.encode("utf-8"))
                if gtDbCamIndex != -1 and gtDbAttrIndex != -1:
                    print("GT DB:    ({}, {}, {}) {}".format(cameras[gtDbCamIndex], dbFieldnames[gtDbAttrIndex], gtShkpAction, gtDbContents))
                else:
                    print("GT DB:    ({}, {}, {}) {}".format(gtDbCamIndex, gtDbAttrIndex, gtShkpAction, gtDbContents))
                print("CANDIDATE: ({}, {}) {}".format(candCam, candAttr, candDbContents))
                
                if gtIsMatch:
                    if len(candMatches) == 0:
                        print("SHOULD HAVE FOUND A MATCH!")
                    else:
                        for match in candMatches:
                            print("GOOD MATCH:", match[2], shkpUtt[match[0]:match[1]].encode("utf-8"))
                elif len(candMatches) > 0:
                    for match in candMatches:
                        print("WRONG MATCH:", match[2], shkpUtt[match[0]:match[1]].encode("utf-8"))
                
                print()
            """
            
            
            # some naive judgment of whether the matches are correct or not...
            if gtIsMatch:
                if len(candMatches) == 0:
                    judgment = "SHOULD_HAVE_FOUND_MATCH"
                else:
                    judgment = "POTENTIAL_MATCH"
            elif len(candMatches) > 0:
                judgment = "WRONG_MATCH"
            else:
                judgment = "SUCCESSFULLY_FOUND_NO_MATCHES"
            
            
            if judgment not in judgmentCounts[gtShkpAction]:
                judgmentCounts[gtShkpAction][judgment] = 0
            judgmentCounts[gtShkpAction][judgment] += 1
            
            
            # store the best match
            for match in candMatches:
                m = {"SHOPKEEPER_SPEECH": shkpUtt,
                     "OUTPUT_SHOPKEEPER_ACTION": gtShkpAction,
                     "DATABASE_ID": dbId,
                     "GT_CAMERA_INDEX": gtDbCamIndex,
                     "GT_ATTRIBUTE_INDEX": gtDbAttrIndex,
                     "GT_DATABASE_CONTENTS": gtDbContents,
                     
                     "CANDIDATE_DATABASE_CONTENTS": candDbContents,
                     "CANDIDATE_CAMERA_INDEX": candCamIndex,
                     "CANDIDATE_ATTRIBUTE_INDEX": candAttrIndex,
                     
                     "POTENTIAL_MATCH_SUBSTRING": shkpUtt[match[0]:match[1]],
                     "POTENTIAL_MATCH_DISTANCE": match[2],
                     "POTENTIAL_MATCH_START_INDEX": match[0],
                     "POTENTIAL_MATCH_END_INDEX": match[1],
                     
                     "NAIVE_JUDGEMENT": judgment
                     }
                
                potentialMatches.append(m)
            
            
            #if len(candMatches) > 0:
            #    bestMatch = min(candMatches, key=lambda x: x[2])
            #    substringMatches[candCamIndex, candAttrIndex] = shkpUtt[bestMatch[0]:bestMatch[1]]
            #    substringMatchScores[candCamIndex, candAttrIndex] = bestMatch[2]

    
    comboToStringMatches[combo] = potentialMatches #(substringMatches, substringMatchScores)

allMatches = []
for combo in comboToStringMatches:
    allMatches += comboToStringMatches[combo]


#
# How many unique utterance-DB combinations have a correct match among the candidates?
#
gtAmongPotentialMatches = []

for combo in uniqueUttDbCombos:
    shkpUtt = combo[0]
    dbId = int(combo[1])
    gtDbCamIndex = combo[2]
    gtDbAttrIndex = combo[3]
    gtShkpAction = combo[5]
    
    if gtShkpAction == "S_INTRODUCES_FEATURE" or gtShkpAction == "S_INTRODUCES_CAMERA" or gtShkpAction == "S_ANSWERS_QUESTION_ABOUT_FEATURE":
        potentialAttrMatches = [m["CANDIDATE_ATTRIBUTE_INDEX"] for m in comboToStringMatches[combo]]
        
        if gtDbAttrIndex in potentialAttrMatches:
            gtAmongPotentialMatches.append(1)
        else:
            gtAmongPotentialMatches.append(0)
    
    #else:
    #    if len(comboToStringMatches[combo]) == 0:
    #        gtAmongPotentialMatches.append(1)
    #    else:
    #        gtAmongPotentialMatches.append(0)


print(len(gtAmongPotentialMatches), "unique utterance-DB combinations (INTRO_FEAT, INTRO_CAM, ANS_Q_FEAT")
print("{} ({}) of these have the correct match among the potential matches".format(sum(gtAmongPotentialMatches), float(sum(gtAmongPotentialMatches))/len(gtAmongPotentialMatches)))


#print(len(uniqueUttDbCombos), "unique utterance-DB combinations")
#print("{} ({}) of these have the correct match among the potential matches".format(sum(gtAmongPotentialMatches), float(sum(gtAmongPotentialMatches))/len(uniqueUttDbCombos)))


#
# choose from among the potential matches
#
comboToBestMatches = {}

for combo in comboToStringMatches:
    potentialMatches = copy.deepcopy(comboToStringMatches[combo])
    bestMatches = []
    
    while len(potentialMatches) > 0:
        # take the match with the shortest distance
        nearestDist = min([m["POTENTIAL_MATCH_DISTANCE"] for m in potentialMatches])
        nearestMatches = [m for m in potentialMatches if m["POTENTIAL_MATCH_DISTANCE"] == nearestDist]
        
        # if there are ties, take the longer match
        tieBreakerLen = max([len(m["POTENTIAL_MATCH_SUBSTRING"]) for m in nearestMatches])
        tieBreakerMatches = [m for m in nearestMatches if len(m["POTENTIAL_MATCH_SUBSTRING"]) == tieBreakerLen]
        
        #tieBreakerMatches = nearestMatches
        
        if len(tieBreakerMatches) > 0:
            # if there are still ties, arbitrarily choose one of the matches
            bestMatches.append(tieBreakerMatches[0])
        else:
            break
        
        # if there are other remaining matches, remove all those that overlap with the current best matches and continue searching
        toRemove = []
        
        for match in bestMatches:
            startA = match["POTENTIAL_MATCH_START_INDEX"]
            endA = match["POTENTIAL_MATCH_START_INDEX"]
            
            for remainingMatch in potentialMatches:
                startB = remainingMatch["POTENTIAL_MATCH_START_INDEX"]
                endB = remainingMatch["POTENTIAL_MATCH_START_INDEX"]
                
                if startA <= endB and startB <= endA:
                    toRemove.append(remainingMatch)
        
        for tr in toRemove:
            potentialMatches.remove(tr)
    
    comboToBestMatches[combo] = bestMatches

#
# compute the accuracy of the string search results
#
targetOutputs = []
resultOutputs = []

for combo in uniqueUttDbCombos:
    shkpUtt = combo[0]
    dbId = int(combo[1])
    gtDbCamIndex = combo[2]
    gtDbAttrIndex = combo[3]
    gtShkpAction = combo[5]
    
    # target output...
    tarOut = -1
    
    if gtShkpAction == "S_INTRODUCES_FEATURE" or gtShkpAction == "S_INTRODUCES_CAMERA" or gtShkpAction == "S_ANSWERS_QUESTION_ABOUT_FEATURE":
        tarOut = gtDbAttrIndex
    
    targetOutputs.append(tarOut)
    
    # result...
    out = -1
    
    for match in comboToBestMatches[combo]:
        if tarOut == match["CANDIDATE_ATTRIBUTE_INDEX"]:
            out = match["CANDIDATE_ATTRIBUTE_INDEX"]
    
    resultOutputs.append(out)

stringSearchAcc = accuracy_score(targetOutputs, resultOutputs)

print(stringSearchAcc, "string search accuracy")


#
#
#
for shkpAction in judgmentCounts:    
    for judgment in judgmentCounts[shkpAction]:
        
        # compute avera distance for this category
        if judgment == "POTENTIAL_MATCH" or judgment == "WRONG_MATCH":            
            dists = [m["POTENTIAL_MATCH_DISTANCE"] for m in allMatches if m["OUTPUT_SHOPKEEPER_ACTION"] == shkpAction and m["NAIVE_JUDGEMENT"] == judgment]
            aveDist = np.mean(dists)
            stdDist = np.std(dists)
        else:
            aveDist = ""
            stdDist = ""
        
        print(shkpAction, judgment, judgmentCounts[shkpAction][judgment], aveDist, stdDist)



fieldnames = ["SHOPKEEPER_SPEECH",
              "OUTPUT_SHOPKEEPER_ACTION",
              "DATABASE_ID",
              "GT_CAMERA_INDEX",
              "GT_ATTRIBUTE_INDEX",
              "GT_DATABASE_CONTENTS",
              
              "CANDIDATE_DATABASE_CONTENTS",
              "CANDIDATE_CAMERA_INDEX",
              "CANDIDATE_ATTRIBUTE_INDEX",
              
              "POTENTIAL_MATCH_SUBSTRING",
              "POTENTIAL_MATCH_DISTANCE",
              "POTENTIAL_MATCH_START_INDEX",
              "POTENTIAL_MATCH_END_INDEX",
              
              "NAIVE_JUDGEMENT"]

with open(sessionDir + "/potential_substring_matches.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for combo in comboToBestMatches:
        for match in comboToStringMatches[combo]:
            writer.writerow(match)


#
# find the best attribute match for each combo
#
"""
print("finding the best matching attribute for each combo...")

matchResults = []

numCorrectAttr = 0


for combo in uniqueUttDbCombos:
    shkpUtt = combo[0]
    dbId = combo[1]
    gtDbCamera = combo[2]
    gtDbAttribute = combo[3]
    gtDbContents = combo[4]
    
    substringMatches = comboToStringMatches[combo][0]
    substringMatchScores = comboToStringMatches[combo][1]
    
    # collapse the rows since we don't care about cameras, we just want to find the best matching attribute
    substringMatchesCollapsed = np.ones(shape=len(dbFieldnames), dtype=object)
    substringMatchesCollapsed[:] = ""
    substringMatchScoresCollapsed = np.ones(shape=len(dbFieldnames)) * np.inf
    
    for a in range(len(dbFieldnames)):
        bestCamIndex = np.argmin(substringMatchScores[:,a])
        
        substringMatchesCollapsed[a] = substringMatches[bestCamIndex, a]
        substringMatchScoresCollapsed[a] = substringMatchScores[bestCamIndex, a]
    
    # TODO: find out if there are ties...
    bestAttrIndex = np.argmin(substringMatchScoresCollapsed)
    
    bestAttr = dbFieldnames[bestAttrIndex]
    bestSubstring = substringMatchesCollapsed[bestAttrIndex]
    
    print("SHKP UTT:", shkpUtt.encode("utf-8"))
    print("GT DB:    ({}, {}) {}".format(cameras[gtDbCamera], dbFieldnames[gtDbAttribute], gtDbContents))
    print("MATCH:    ({}) {}".format(bestAttr, bestSubstring))
    print()
    
    if bestAttr == dbFieldnames[gtDbAttribute]:
        numCorrectAttr += 1
    

print("{} of {} ({}) correct attribute matches".format(numCorrectAttr, len(uniqueUttDbCombos), numCorrectAttr/float(len(uniqueUttDbCombos))))
"""
    
    


