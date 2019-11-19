'''
Created on Nov 14, 2019

@author: Malcolm

copied from stringmatch1 but match lemma seqs, not chars
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
from tabulate import tabulate

import tools


sessionDir = tools.create_session_dir("stringmatch1")

dataDirectory = tools.dataDir+"2019-11-12_17-40-29_advancedSimulator9"
numTrainDbs = 10
numInteractionsPerDb = 200

cameras = ["CAMERA_1", "CAMERA_2", "CAMERA_3"]


punctuation = r"""!"#%&'()*+,-./:;<=>?@[\]^_`{|}~""" # took $ out of punctuation


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
                
                row["CUSTOMER_SPEECH"] = row["CUSTOMER_SPEECH"].lower().translate(str.maketrans('', '', punctuation))
                row["SHOPKEEPER_SPEECH"] = row["SHOPKEEPER_SPEECH"].lower().translate(str.maketrans('', '', punctuation))
                
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
    utt = utt.lower().translate(str.maketrans('', '', punctuation))
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


def contains_digit(inputString):
    return any(char.isdigit() for char in inputString)



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
# tokenize any utterances or database contents that we will be searching
#
print("tokenizing utterances and DB contents...")

uttToTokenized = {}

for i in range(len(interactions)):
    for j in range(len(interactions[i])):
        shkpUtt = interactions[i][j]["SHOPKEEPER_SPEECH"]
        if shkpUtt not in uttToTokenized:
            uttToTokenized[shkpUtt] = tokenize_utterance(shkpUtt, lemmatize=True)

for i in range(len(databases)):
    for r in range(len(databases[i])):
        for key in databases[i][r]:
            dbContents = databases[i][r][key]
            
            if dbContents not in uttToTokenized:
                uttToTokenized[dbContents] = tokenize_utterance(dbContents, lemmatize=True)



#
# find each unique combination of shopkeeper utterance and DB ID - to be used for matching
#
print("finding unique combinations of shopkeeper utterance and DB IDs...")

uniqueUttDbCombos = []

for i in range(len(interactions)):
    for j in range(len(interactions[i])):
        
        shkpUtt = interactions[i][j]["SHOPKEEPER_SPEECH"]
        dbId = interactions[i][j]["DATABASE_ID"]
        
        gtDbCamera = gtDatabaseCameras[i][j]
        gtDbAttribute = gtDatabaseAttributes[i][j]
        gtDbContents = interactions[i][j]["DATABASE_CONTENTS"].lower().translate(str.maketrans('', '', punctuation))
        
        gtShkpAction = interactions[i][j]["OUTPUT_SHOPKEEPER_ACTION"]
        
        if gtShkpAction == "S_INTRODUCES_CAMERA":
            gtDbCamera = cameras.index(interactions[i][j]["OUTPUT_STATE_TARGET"])
            gtDbAttribute = dbFieldnames.index("camera_name")
        
        combo = (shkpUtt, dbId, gtDbCamera, gtDbAttribute, gtDbContents, gtShkpAction)
        
        if combo not in uniqueUttDbCombos and shkpUtt != "":
            uniqueUttDbCombos.append(combo)



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
    gtDbContents = combo[4]
    gtShkpAction = combo[5]
    
    if gtShkpAction not in judgmentCounts:
        judgmentCounts[gtShkpAction] = {}
    
    
    potentialMatches = []
    
    for candCamIndex in range(len(cameras)):
        candCam = cameras[candCamIndex]   
        for candAttrIndex in range(1, len(dbFieldnames)): # skip the camera_ID field
            
            candAttr = dbFieldnames[candAttrIndex]
            #candDbContents = databases[dbId][candCamIndex][candAttr].lower().translate(str.maketrans('', '', punctuation))
            candDbContents = databases[dbId][candCamIndex][candAttr]
            candDbContentsTokens = uttToTokenized[candDbContents]
                
            # if the candidate DB contents is empty, we don't have to find a match
            #if candDbContents == "":
            if candDbContentsTokens == []:
                continue
            
            
            # the substring search function requires a max distance parameter
            # TODO: what's the best way to set this?
            maxDist = int(np.floor(len(candDbContentsTokens)/2))
            
            
            # get the candidate shkp utt substring matches
            shkpUttTokens = uttToTokenized[shkpUtt]
            
            #print([w.encode("utf-8") for w in shkpUttTokens])
            #print(candDbContents)
            #print(maxDist)
            
            #if candDbContents == ['11', 'scene', 'mode', 'eg', 'night', 'scene', 'soft', 'skin', 'and', 'beach'] \
            #and shkpUttTokens == ['the', 'preset', 'mode', 'ha', '11', 'scene']:
            #    print("hello")
            
            
            if len(candDbContentsTokens) > len(shkpUttTokens):
                candMatches = []
            
            else:
                candMatches = fuzzysearch.find_near_matches(candDbContentsTokens, shkpUttTokens, max_l_dist=maxDist)
            
            #print(candMatches)
            #print()
            
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
                    judgment = "GOOD_MATCH"
            elif len(candMatches) > 0:
                judgment = "BAD_MATCH"
            else:
                judgment = "SUCCESSFUL_NO_MATCH"
            
            
            if judgment not in judgmentCounts[gtShkpAction]:
                judgmentCounts[gtShkpAction][judgment] = 0
            judgmentCounts[gtShkpAction][judgment] += 1
            
            
            # store the best match
            for match in candMatches:
                #potentialMatchSubString = shkpUtt[match[0]:match[1]]
                potentialMatchSubString = " ".join(uttToTokenized[shkpUtt][match[0]:match[1]])
                
                
                m = {"SHOPKEEPER_SPEECH": shkpUtt,
                     "OUTPUT_SHOPKEEPER_ACTION": gtShkpAction,
                     "DATABASE_ID": dbId,
                     "GT_CAMERA_INDEX": gtDbCamIndex,
                     "GT_ATTRIBUTE_INDEX": gtDbAttrIndex,
                     "GT_DATABASE_CONTENTS": gtDbContents,
                     
                     "CANDIDATE_DATABASE_CONTENTS": candDbContents,
                     "CANDIDATE_CAMERA_INDEX": candCamIndex,
                     "CANDIDATE_ATTRIBUTE_INDEX": candAttrIndex,
                     
                     "POTENTIAL_MATCH_SUBSTRING": potentialMatchSubString,
                     "POTENTIAL_MATCH_DISTANCE": match[2],
                     "POTENTIAL_MATCH_START_INDEX": match[0],
                     "POTENTIAL_MATCH_END_INDEX": match[1],
                     
                     "NAIVE_JUDGEMENT": judgment
                     }
                
                potentialMatches.append(m)
    
    comboToStringMatches[combo] = potentialMatches #(substringMatches, substringMatchScores)

allMatches = []
for combo in comboToStringMatches:
    allMatches += comboToStringMatches[combo]

print()



#
# print the naive judgements about the initial matching results
#
possibleJudgments = ["SHOULD_HAVE_FOUND_MATCH", "GOOD_MATCH", "BAD_MATCH", "SUCCESSFUL_NO_MATCH"]

tableData = []

for shkpAction in judgmentCounts:
    tableRow = []
    tableRow.append(shkpAction)
    
    for j in possibleJudgments:
        
        if j not in judgmentCounts[shkpAction]:
            judgmentCounts[shkpAction][j] = 0
            
        tableRow.append(judgmentCounts[shkpAction][j])
    tableData.append(tableRow)

print(tabulate(tableData, headers=["Action type"]+possibleJudgments))
print()

# print the mean distances for good and bad matches
tableData = []

for shkpAction in judgmentCounts:
    
    if judgmentCounts[shkpAction]["GOOD_MATCH"] > 0:
        goodMatchAve = np.mean([m["POTENTIAL_MATCH_DISTANCE"] for m in allMatches if m["OUTPUT_SHOPKEEPER_ACTION"] == shkpAction and m["NAIVE_JUDGEMENT"] == "GOOD_MATCH"])
        goodMatchSd = np.std([m["POTENTIAL_MATCH_DISTANCE"] for m in allMatches if m["OUTPUT_SHOPKEEPER_ACTION"] == shkpAction and m["NAIVE_JUDGEMENT"] == "GOOD_MATCH"])
    else:
        goodMatchAve = -1
        goodMatchSd = -1
            
    if judgmentCounts[shkpAction]["BAD_MATCH"] > 0:
        badMatchAve = np.mean([m["POTENTIAL_MATCH_DISTANCE"] for m in allMatches if m["OUTPUT_SHOPKEEPER_ACTION"] == shkpAction and m["NAIVE_JUDGEMENT"] == "BAD_MATCH"])
        badMatchSd = np.std([m["POTENTIAL_MATCH_DISTANCE"] for m in allMatches if m["OUTPUT_SHOPKEEPER_ACTION"] == shkpAction and m["NAIVE_JUDGEMENT"] == "BAD_MATCH"])
    else:
        badMatchAve = -1
        badMatchSd = -1
        
    tableData.append([shkpAction, goodMatchAve, goodMatchSd, badMatchAve, badMatchSd])
    
print(tabulate(tableData, headers=["Action type", 
                                   "Good match AVE. dist.",
                                   "S.D.",
                                   "Bad match AVE. dist.",
                                   "S.D."],
                                   floatfmt=".3f"))
print()



#
# How many unique utterance-DB combinations have a correct match among the candidates?
#

def compute_whether_potential_matches_contain_true_match(CtoM):
    gtAmongPotentialMatches = {}
    
    for combo in uniqueUttDbCombos:
        shkpUtt = combo[0]
        dbId = int(combo[1])
        gtDbCamIndex = combo[2]
        gtDbAttrIndex = combo[3]
        gtShkpAction = combo[5]
        
        
        if gtShkpAction not in gtAmongPotentialMatches:
            gtAmongPotentialMatches[gtShkpAction] = []
        
        # these action types should have a match among the potential matches
        if gtShkpAction == "S_INTRODUCES_FEATURE" or gtShkpAction == "S_INTRODUCES_CAMERA" or gtShkpAction == "S_ANSWERS_QUESTION_ABOUT_FEATURE":
            potentialCamAttrMatches = [(m["CANDIDATE_CAMERA_INDEX"], m["CANDIDATE_ATTRIBUTE_INDEX"]) for m in CtoM[combo]]
            
            if (gtDbCamIndex, gtDbAttrIndex) in potentialCamAttrMatches:
                gtAmongPotentialMatches[gtShkpAction].append(1)
            else:
                gtAmongPotentialMatches[gtShkpAction].append(0)
        
        # the other action types should not have a match (i.e. they should not have any potential matches)
        else:
            if len(CtoM[combo]) == 0:
                gtAmongPotentialMatches[gtShkpAction].append(1)
            else:
                gtAmongPotentialMatches[gtShkpAction].append(0)
    
    
    gtAmongPotentialMatchesAllActions = []
    for key, val in gtAmongPotentialMatches.items():
        gtAmongPotentialMatchesAllActions += val
    gtAmongPotentialMatches["ALL"] = gtAmongPotentialMatchesAllActions
    
    
    tableData = []
    for key, val in gtAmongPotentialMatches.items():    
        tableData.append([key, len(val), sum(val), float(sum(val))/len(val)])
    
    print(tabulate(tableData, headers=["Action type", 
                                       "Num unique utterance-DB\ncombinations", 
                                       "Num with corrent match\namong the potential matches", 
                                       "Percent"],
                                       floatfmt=".3f"))
    

print("Results of initial string search:")
compute_whether_potential_matches_contain_true_match(comboToStringMatches)
print()

#
# choose best matches from among the potential matches (reduce number of false positives
#
comboToBestMatches = {}

for combo in comboToStringMatches:
    potentialMatches = copy.deepcopy(comboToStringMatches[combo])
    bestMatches = []
    
    # filter out potential matches for DB contents that contain a number not in the potential match
    potentialMatchesFiltered = []
    
    for match in potentialMatches:
        keep = True
        
        s = match["POTENTIAL_MATCH_START_INDEX"]
        e = match["POTENTIAL_MATCH_END_INDEX"]
        shkpUttTokenSeq = uttToTokenized[match["SHOPKEEPER_SPEECH"]]
        potMatTokenSeq = shkpUttTokenSeq[s:e]
        
        dbContentsTokenSeq = uttToTokenized[match["CANDIDATE_DATABASE_CONTENTS"]]
        
        for tok in dbContentsTokenSeq:
            if contains_digit(tok):
                if tok not in potMatTokenSeq:
                    keep = False
                    break
        
        for tok in potMatTokenSeq:
            if contains_digit(tok):
                if tok not in dbContentsTokenSeq:
                    keep = False
                    break
        
        if keep:
            potentialMatchesFiltered.append(match)
    
    potentialMatches = potentialMatchesFiltered
    
    
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
print("Results after finding best match(es) (only non overlapping matches remain):")
compute_whether_potential_matches_contain_true_match(comboToBestMatches)
print()

targetOutputs = {"ALL":[]}
resultOutputs = {"ALL":[]}


for combo in uniqueUttDbCombos:
    shkpUtt = combo[0]
    dbId = int(combo[1])
    gtDbCamIndex = combo[2]
    gtDbAttrIndex = combo[3]
    gtDbContents = combo[4]
    gtShkpAction = combo[5]
    
    if gtShkpAction == "S_INTRODUCES_FEATURE" or gtShkpAction == "S_INTRODUCES_CAMERA" or gtShkpAction == "S_ANSWERS_QUESTION_ABOUT_FEATURE":
        # if it's one of these actions, the target result is to find a match to the GT database contents
        tarOut = gtDbContents
        
    else:
        # the target is to find no match
        tarOut = "NO_MATCH"
    
    
    
    if gtShkpAction == "S_INTRODUCES_FEATURE" or gtShkpAction == "S_INTRODUCES_CAMERA" or gtShkpAction == "S_ANSWERS_QUESTION_ABOUT_FEATURE":
        for match in comboToBestMatches[combo]:
            # if the correct match is among the best matches, count the combo as having a correct match
            #if tarOut == str(match["CANDIDATE_CAMERA_INDEX"]) + str(match["CANDIDATE_ATTRIBUTE_INDEX"]):
            #    out = str(match["CANDIDATE_CAMERA_INDEX"]) + str(match["CANDIDATE_ATTRIBUTE_INDEX"])
            if tarOut == match["CANDIDATE_DATABASE_CONTENTS"]:   
                out = match["CANDIDATE_DATABASE_CONTENTS"]
    
    else:
        if len(comboToBestMatches[combo]) > 0:
            out = "SHOULD_NOT_HAVE_FOUND_MATCH"
        else:
            out = "NO_MATCH"
    
    
    targetOutputs["ALL"].append(tarOut)
    resultOutputs["ALL"].append(out)
    
    if gtShkpAction not in targetOutputs:
        targetOutputs[gtShkpAction] = []
        resultOutputs[gtShkpAction] = []
    
    targetOutputs[gtShkpAction].append(tarOut)
    resultOutputs[gtShkpAction].append(out)


tableData = []

for shkpAction in targetOutputs:
    tableRow = []
    tableRow.append(shkpAction)
    tableRow.append(accuracy_score(targetOutputs[shkpAction], resultOutputs[shkpAction]))
    
    resOutArr = np.asarray(resultOutputs[shkpAction])
    tarOutArr = np.asarray(targetOutputs[shkpAction])
    
    temp = np.where((resOutArr == "NO_MATCH") & (tarOutArr != "NO_MATCH"))[0]
    numMissingMatch = len(temp)
    
    temp = np.where(resOutArr == "SHOULD_NOT_HAVE_FOUND_MATCH")[0]
    numShouldNotHaveMatch = len(temp)
    
    temp = np.where((tarOutArr != "NO_MATCH") & (tarOutArr != resOutArr))[0]
    numIncorrectMatch = len(temp)
    
    tableRow.append(numMissingMatch)
    tableRow.append(numMissingMatch/len(targetOutputs[shkpAction]))
    
    tableRow.append(numShouldNotHaveMatch)
    tableRow.append(numShouldNotHaveMatch/len(targetOutputs[shkpAction]))
    
    tableRow.append(numIncorrectMatch)
    tableRow.append(numIncorrectMatch/len(targetOutputs[shkpAction]))
    
    tableData.append(tableRow)


print(tabulate(tableData, headers=["Action type",
                                   "ACCURACY",
                                   "Num MISSING MATCH",
                                   "Percent",
                                   "Num SHOULD NOT HAVE MATCH",
                                   "Percent",
                                   "Num INCORRECT MATCH",
                                   "Percent"],
                                   floatfmt=".3f"))
print()



#
# print the final results...
#
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
        for match in comboToBestMatches[combo]:
            writer.writerow(match)

