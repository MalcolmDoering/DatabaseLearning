'''
Created on Feb 13, 2017

@author: MalcolmD
'''



import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances
from nltk.stem import WordNetLemmatizer
import nltk
import csv
import pickle as pkl
import os
import multiprocessing
import time


import tools



numCpus = multiprocessing.cpu_count()


"""
def normalized_levenshtein_stem_distance(utt1, utt2):
    
    wnl = WordNetLemmatizer()
    
    uttLemmas1 = [wnl.lemmatize(t).lower() for t in nltk.word_tokenize(utt1)]
    uttLemmas2 = [wnl.lemmatize(t).lower() for t in nltk.word_tokenize(utt2)]
    
    dist = editdistance.eval(uttLemmas1, uttLemmas2) / float(max(len(uttLemmas1), len(uttLemmas2)))
    
    return dist
""" 
    
class UtteranceVectorizer(object):
    
    def __init__(self, allUtterances, minCount=None, keywordWeight=None, keywordSet=None, unigramsAndKeywordsOnly=False, tfidf=False, useStopwords=False, lsa=False):
        
        self.keywordWeight = keywordWeight
        self.keywordSet = keywordSet
        self.minCount = minCount
        self.unigramsAndKeywordsOnly = unigramsAndKeywordsOnly
        self.tfidf = tfidf
        self.backchannelPlaceholder = "<backchannel>"
        self.useNumbers = False
        
        self.lsa = lsa
        
        if self.lsa:
            self.tfidf = False
            self.keywordWeight = 1
            self.unigramsAndKeywordsOnly = False
        
        if useStopwords:
            
            #stopwords1 = ["okay","a", "see", "alright", "and", "so", "oh", "sound", "sounds", "of",
            #              "interesting", "interest" , "does", "it", "those", "was", "for", "'s", 
            #              "the", "are", "is" "that", "one", "cool", "what", "OK", "trip", "yeah", 
            #              "um","oh okay", "okay okay", "okay i see", "i see", "OK", "okay cool", 
            #              "yeah", "okay", "ok i see", "oh i see", "ah ok"]
            
            self.stopwords = ["okay","a", "see", "alright", "and", "so", "oh", "sound", "sounds", "of",
                               "interesting", "interest" , "does", "it", "those", "was", "for", "'s", 
                               "the", "are", "is" "that", "one", "cool", "what", "OK", "trip", "yeah", 
                               "um","oh okay", "okay okay", "okay i see", "i see", "OK", "okay cool", 
                               "yeah", "okay", "ok i see", "oh i see", "ah ok"]
            
            # if the input utterance matches one of these exactly, the backchannel index is 1
            self.backchannelList = ["okay", "ok", "oh yeah it does", "yeah", "alright", "oh yeah ok", "yeah ok", "okay okay", "yeah yeah yeah", "interesting", "interesting ok",
                                    "ok ok", "okay Jose", "okay interesting", "OIC", "cic", "ICICI", "IC", "ic"]
        
        else:
            self.stopwords = []
            self.backchannelList = []
            
            
        # default values
        if self.minCount == None:
            self.minCount = 2
        if self.keywordWeight == None:
            self.keywordWeight = 3.0
        
        
        
        self.wnl = WordNetLemmatizer()
        
        # make sure that the DB contents symbols are left in and not split up
        self.tokenizer = nltk.tokenize.MWETokenizer()
        
        self.tokenizer.add_mwe(('<', 'camera_ID', '>'))
        self.tokenizer.add_mwe(('<', 'camera_id', '>'))
        self.tokenizer.add_mwe(('<', 'camera_name', '>'))
        self.tokenizer.add_mwe(('<', 'camera_type', '>'))
        self.tokenizer.add_mwe(('<', 'color', '>'))
        self.tokenizer.add_mwe(('<', 'weight', '>'))
        self.tokenizer.add_mwe(('<', 'preset_modes', '>'))
        self.tokenizer.add_mwe(('<', 'effects', '>'))
        self.tokenizer.add_mwe(('<', 'price', '>'))
        self.tokenizer.add_mwe(('<', 'resolution', '>'))
        self.tokenizer.add_mwe(('<', 'optical_zoom', '>'))
        self.tokenizer.add_mwe(('<', 'settings', '>'))
        self.tokenizer.add_mwe(('<', 'autofocus_points', '>'))
        self.tokenizer.add_mwe(('<', 'sensor_size', '>'))
        self.tokenizer.add_mwe(('<', 'ISO', '>'))
        self.tokenizer.add_mwe(('<', 'iso', '>'))
        self.tokenizer.add_mwe(('<', 'long_exposure', '>'))
        
        self.dbSymbols = []
        self.dbSymbols.append("<_camera_ID_>")
        self.dbSymbols.append("<_camera_id_>")
        self.dbSymbols.append("<_camera_name_>")
        self.dbSymbols.append("<_camera_type_>")
        self.dbSymbols.append("<_color_>")
        self.dbSymbols.append("<_weight_>")
        self.dbSymbols.append("<_preset_modes_>")
        self.dbSymbols.append("<_effects_>")
        self.dbSymbols.append("<_price_>")
        self.dbSymbols.append("<_resolution_>")
        self.dbSymbols.append("<_optical_zoom_>")
        self.dbSymbols.append("<_settings_>")
        self.dbSymbols.append("<_autofocus_points_>")
        self.dbSymbols.append("<_sensor_size_>")
        self.dbSymbols.append("<_ISO_>")
        self.dbSymbols.append("<_iso_>")
        self.dbSymbols.append("<_long_exposure_>")
        
        
        stopwordLemmas = []
        for w in self.stopwords:
            w = ":".join(self.lemmatize_utterance(w))
            stopwordLemmas.append(w)
        self.stopwords = stopwordLemmas
        
        
        #
        # find which words appear and their counts
        #
        self.unigrams = {}
        self.bigrams = {}
        self.trigrams = {}
        
        for utt in allUtterances:
            
            # lemmatize the words
            uttLemmas = self.lemmatize_utterance(utt)
            
            # find unigrams
            for w in uttLemmas:
                if w not in self.stopwords:
                    if w not in self.unigrams:
                        self.unigrams[w] = 0
                    self.unigrams[w] += 1
            
            # find bigrams
            for i in range(len(uttLemmas)-1):
                if uttLemmas[i] not in self.stopwords and uttLemmas[i+1] not in self.stopwords:
                    bigram = uttLemmas[i] + ":" + uttLemmas[i+1]
                    if bigram not in self.stopwords and bigram not in self.bigrams:
                        self.bigrams[bigram] = 0
                    self.bigrams[bigram] += 1
            
            # find trigrams
            for i in range(len(uttLemmas)-2):
                bigram1 = uttLemmas[i] + ":" + uttLemmas[i+1]
                bigram2 = uttLemmas[i+1] + ":" + uttLemmas[i+2]
                
                if uttLemmas[i] not in self.stopwords and uttLemmas[i+1] not in self.stopwords and uttLemmas[i+2] not in self.stopwords and bigram1 not in self.stopwords and bigram2 not in self.stopwords:
                    trigram = uttLemmas[i] + ":" + uttLemmas[i+1] + ":" + uttLemmas[i+2]
                    
                    if trigram not in self.stopwords and trigram not in self.trigrams:
                        self.trigrams[trigram] = 0
                    self.trigrams[trigram] += 1
        
        #
        # find the keywords and numbers
        #        
        # process keywords
        self.processedKeywordSet = []
        
        for kw in self.keywordSet:
            #f.write(kw+"\n")
            
            kws = kw.split()
            
            for w in kws:
                w = self.lemmatize_word(w)
                
                if w not in self.processedKeywordSet and w not in self.stopwords and w in self.unigrams:
                    if self.unigrams[w] >= minCount:
                        self.processedKeywordSet.append(w)
        #f.close()
        
        # find all the numbers
        self.numbers = {}
        
        for utt in allUtterances:
            
            for word in utt.split():
                if word.startswith("$"):
                    word = word[1:]
                
                if tools.is_number(word):
                    if word not in self.numbers:
                        self.numbers[word] = 0
                    self.numbers[word] += 1
        
        
        #
        # create the vectorization dict by removing all words that are in stopwords and occur infrequently
        #
        self.unigramToIndex = {}
        self.bigramToIndex = {}
        self.trigramToIndex = {}
        self.keywordToIndex = {}
        self.numberToIndex = {}
        
        self.indexToWord = {}
        
        if self.backchannelList != []:
            self.unigramToIndex[self.backchannelPlaceholder] = 0
            self.indexToWord[self.unigramToIndex[self.backchannelPlaceholder]] = self.backchannelPlaceholder
        
        for unigram, count in list(self.unigrams.items()):
            if count >= minCount:
                self.unigramToIndex[unigram] = len(self.unigramToIndex)
                self.indexToWord[self.unigramToIndex[unigram]] = unigram
        
        for bigram, count in list(self.bigrams.items()):
            if count >= minCount:
                self.bigramToIndex[bigram] = len(self.unigramToIndex) + len(self.bigramToIndex)
                self.indexToWord[self.bigramToIndex[bigram]] = bigram
        
        for trigram, count in list(self.trigrams.items()):
            if count >= minCount:
                self.trigramToIndex[trigram] = len(self.unigramToIndex) + len(self.bigramToIndex) + len(self.trigramToIndex)
                self.indexToWord[self.trigramToIndex[trigram]] = trigram
                
        """
        for kw in self.processedKeywordSet:
            self.keywordToIndex[kw] = len(self.unigramToIndex) + len(self.bigramToIndex) + len(self.trigramToIndex) + len(self.keywordToIndex)
            self.indexToWord[self.keywordToIndex[kw]] = kw
        self.keywordsStartIndex = min(self.keywordToIndex.values())
        
        for num in self.numbers:
            #print num
            self.numberToIndex[num] = len(self.unigramToIndex) + len(self.bigramToIndex) + len(self.trigramToIndex) + len(self.keywordToIndex) + len(self.numberToIndex)
            self.indexToWord[self.numberToIndex[num]] = num
        """
        
        self.numIndices = len(self.unigramToIndex) + len(self.bigramToIndex) + len(self.trigramToIndex) + len(self.keywordToIndex)
        
        
        if self.useNumbers:
            self.numIndices += len(self.numberToIndex)
        
        
        self.numIndices += len(self.dbSymbols)
        
        
        #
        # do LSA on the utterance and keyword vectors separately
        #
        if self.lsa:
            allUttVecs = self.get_utterance_vectors(allUtterances, returnAll=True, asArray=False)
            
            ngramUttVecs = []
            #keywordUttVecs = []
            
            while len(allUttVecs) > 0:
                uttVec = allUttVecs.pop()
                #ngramUttVecs.append(uttVec[:self.keywordsStartIndex])
                ngramUttVecs.append(uttVec)
                #keywordUttVecs.append(uttVec[self.keywordsStartIndex:])
                
            #for i in range(len(allUttVecs)):
            #    ngramUttVecs.append(allUttVecs[i][:self.keywordsStartIndex])
            #    keywordUttVecs.append(allUttVecs[i][self.keywordsStartIndex:])
                
            
            self.nrgamTfidfVectorizer = TfidfTransformer()
            ngramAllTfidfUttVecs = self.nrgamTfidfVectorizer.fit_transform(ngramUttVecs)
            
            #self.keywordTfidfVectorizer = TfidfTransformer()
            #keywordAllTfidfUttVecs = self.keywordTfidfVectorizer.fit_transform(keywordUttVecs)
            
            self.ngramLsaModel = TruncatedSVD(n_components=min(1000, len(ngramUttVecs[0])-1))
            self.ngramLsaModel.fit(ngramAllTfidfUttVecs)
            
            #self.keywordLsaModel = TruncatedSVD(n_components=min(200, len(keywordUttVecs[0])-1))
            #self.keywordLsaModel.fit(keywordAllTfidfUttVecs)
    
    
        
    def lemmatize_utterance(self, utt):
        uttLemmas = []
        tokenized = self.tokenizer.tokenize(nltk.word_tokenize(utt))
        
        for t in tokenized:
            uttLemmas.append(self.lemmatize_word(t))
        
        return uttLemmas
    
    
    
    def lemmatize_word(self, w):
        return self.wnl.lemmatize(w).lower()
    
    
    
    def convert_array_to_list_of_tuples(self, array):
        return [(i, array[i]) for i in range(array.shape[0])]
    
    
    
    def get_lsa_vector(self, utt):
        
        uttVec = self.get_utterance_vector(utt)
        
        #ngramTfidfVec = self.nrgamTfidfVectorizer.transform([uttVec[:self.keywordsStartIndex]])
        #keywordTfidfVec = self.keywordTfidfVectorizer.transform([uttVec[self.keywordsStartIndex:]])
        
        ngramTfidfVec = self.nrgamTfidfVectorizer.transform([uttVec])
        
        
        ngramLsaVec = self.ngramLsaModel.transform(ngramTfidfVec)[0]
        #keywordLsaVec = self.keywordLsaModel.transform(keywordTfidfVec)[0]
        
        #return np.concatenate([ngramLsaVec, keywordLsaVec], axis=0)
        return ngramLsaVec
    
    
    
    def get_lsa_vectors(self, uttList):
        
        uttVecs = self.get_utterance_vectors(uttList)
        uttVecs = np.array(uttVecs)
        
        ngramTfidfUttVecs = self.nrgamTfidfVectorizer.fit_transform(uttVecs[:,:self.keywordsStartIndex])
        keywordTfidfUttVecs = self.keywordTfidfVectorizer.fit_transform(uttVecs[:,self.keywordsStartIndex:])
        
        ngramLsaUttVecs = self.ngramLsaModel.transform(ngramTfidfUttVecs)
        keywordLsaUttVecs = self.keywordLsaModel.transform(keywordTfidfUttVecs)
        
        lsaVecs = np.concatenate([ngramLsaUttVecs, keywordLsaUttVecs], axis=1)
        
        return lsaVecs
    
    
    
    def get_utterance_vector(self, utt, returnAll=False, unigramOnly=False):
        
        uttVec = np.zeros(self.numIndices, dtype=np.float32)
        
        if self.backchannelList != [] and utt.lower() in self.backchannelList:
            uttVec[self.unigramToIndex[self.backchannelPlaceholder]] = 1.0
                
        else:
            uttLemmas = self.lemmatize_utterance(utt)
            length = len(uttLemmas)
            
            for i in range(length):
                
                w = uttLemmas[i]
                
                if w in self.unigramToIndex:
                    uttVec[self.unigramToIndex[w]] = 1.0
                
                if w in self.keywordToIndex:
                    uttVec[self.keywordToIndex[w]] = 1.0 * self.keywordWeight
                
                if self.useNumbers:
                    wStripped = w.strip("$")
                    if wStripped in self.numberToIndex:
                        uttVec[self.numberToIndex[wStripped]] = 1.0 * self.keywordWeight

                
                if i < length-1:
                    bigram = uttLemmas[i] + ":" + uttLemmas[i+1]
                    
                    if bigram in self.bigramToIndex:
                        uttVec[self.bigramToIndex[bigram]] = 1.0
                
                if i < length-2:
                    trigram = uttLemmas[i] + ":" + uttLemmas[i+1] + ":" + uttLemmas[i+2]
                    
                    if trigram in self.trigramToIndex:
                        uttVec[self.trigramToIndex[trigram]] = 1.0
        
        
        if self.tfidf and not returnAll:
            uttVec = uttVec * self.docInvFreqs
        
        if self.unigramsAndKeywordsOnly and not returnAll and not unigramOnly:
            uttVec = np.concatenate((uttVec[:len(self.unigrams)], uttVec[self.keywordsStartIndex:]))
        
        if unigramOnly:
            uttVec = uttVec[:len(self.unigrams)]
        
        return uttVec
    
    
    
    def get_utterance_vectors(self, uttList, returnAll=False, unigramOnly=False, asArray=True):
        
        uttVecs = []
        
        for utt in uttList:
            uttVecs.append(self.get_utterance_vector(utt, returnAll, unigramOnly))
        
        if asArray:
            uttVecs = np.asarray(uttVecs)
        
        return uttVecs
    
    
    
    def get_dimensionality(self, unigramOnly=False):
        if self.unigramsAndKeywordsOnly and not unigramOnly:
            return len(self.unigrams) + (self.numIndices - self.keywordsStartIndex)
        elif unigramOnly:
            return len(self.unigrams)
        else:
            return self.numIndices




def vectorize1():
    sessionDir = tools.create_session_dir("utteranceVectorizer - proactive passive camera")
    
    
    #
    # read in the condition info
    #
    print("reading in the condition info...")
    
    """
    conditions = {}
    
    with open(tools.dataDir+"experiments table.csv", "rb") as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            
            if row["num"] in conditions:
                print "Warning: Duplicate experiment num!", row["num"]
            
            conditions[row["num"]] = row["conditions"]
    """
    
    #
    # read in the customer utterance data
    # read in the cluster sequence data that was used for training
    #
    print("reading in the cluster sequence data...")
    
    clusterSequenceData = []
    #with open(tools.dataDir+"ClusterSequence.csv", "rb") as csvfile: # 2013 dataset
    with open(tools.dataDir+"combined_from_curiosity/ClusterSequence_detailed_fix.csv", "rb") as csvfile: # passive/proactive dataset
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            #row["CONDITION"] =keywordAllTfidfUttVecs conditions[row["TRIAL"]]
            clusterSequenceData.append(row)
        
    
    #
    # find the customer speech that triggered each robot action
    #
    print("finding the customer speech that triggered each robot action...")
    
    utteranceIds = []
    trialIds = []
    #conditions = []
    timestamps = []
    utterances = []
    
    keywords = []
    
    for i in range(len(clusterSequenceData)):
        
        #if clusterSequenceData[i]["CUSTOMER_SPEECH"] != "NONE":
        if clusterSequenceData[i]["SHOPKEEPER_SPEECH"] != "NONE":
            
            utteranceIds.append(clusterSequenceData[i]["INDEX"])
            trialIds.append(clusterSequenceData[i]["TRIAL"])
            #conditions.append(clusterSequenceData[i]["CONDITION"])
            timestamps.append(clusterSequenceData[i]["TIMESTAMP"])
            #utterances.append(clusterSequenceData[i]["CUSTOMER_SPEECH"])
            utterances.append(clusterSequenceData[i]["SHOPKEEPER_SPEECH"])
            
            #keywords += clusterSequenceData[i]["CUSTOMER_KEYWORDS"].split()
            keywords += clusterSequenceData[i]["SHOPKEEPER_KEYWORDS"].split()
    
    
    #
    # vectorize the customer utterance
    #
    print("vectorizing customer utterances...")
    
    keywordsSet = sorted(list(set(keywords)))
    keywordsSet.remove("NO_KEYWORD")
    
    uttVectorizer = UtteranceVectorizer(utterances,
                                        minCount=2, 
                                        keywordWeight=1.0, 
                                        keywordSet=keywordsSet, 
                                        unigramsAndKeywordsOnly=False, 
                                        tfidf=False, 
                                        useStopwords=True,
                                        lsa=True)
    
    vectors = uttVectorizer.get_utterance_vectors(utterances)
    lsaVectors = uttVectorizer.get_lsa_vectors(utterances)
    
    print("lsa vectors shape", lsaVectors.shape)
    
    vectorsNoNan = []
    utteranceIdsNoNan = []
    trialIdsNoNan = []
    #conditionsNoNan = []
    timestampsNoNan = []
    utterancesNoNan = []
    
    for i in range(len(utterances)):
        
        if not vectors[i,:].any():
            print("removing:", utterances[i])
        
        else:
            vectorsNoNan.append(vectors[i,:])
            utteranceIdsNoNan.append(utteranceIds[i])
            trialIdsNoNan.append(trialIds[i])
            #conditionsNoNan.append(conditions[i])
            timestampsNoNan.append(timestamps[i])
            utterancesNoNan.append(utterances[i])
    
    
    #
    # compute distances
    #
    print("computing distances...")
    vectorsNoNan = np.asarray(vectorsNoNan)
    distMatrix = squareform(pdist(vectorsNoNan, "cosine"))
    
    
    #
    # save vectorizations, etc. to file
    #
    print("saving...")
    
    print("num no nan utts:", len(utterancesNoNan))
    print("dimensionality:", vectorsNoNan.shape[1])
    
    condition = "passive proactive camera shopkeeper - lsa test - tri stm - 1 wgt kw - mc2 - stopwords 1"
    
    np.savetxt(sessionDir+"/utterance cos dists - {:}.txt".format(condition), distMatrix, fmt="%.4f")
    
    np.savetxt(sessionDir+"/utterance vectors - {:}.txt".format(condition), vectorsNoNan, fmt="%d")
    
    
    with open(sessionDir+"/utterance data - {:}.csv".format(condition), "wb") as csvfile:
        
        fieldnames = ["Utterance ID", "Timestamp", "Trial ID", "Condition", "Utterance"]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for i in range(len(utterancesNoNan)):
            
            data = {"Utterance ID":utteranceIdsNoNan[i],
                    "Timestamp":timestampsNoNan[i],
                    "Trial ID":trialIdsNoNan[i],
                    #"Condition":conditionsNoNan[i],
                    "Utterance":utterancesNoNan[i]}
            
            writer.writerow(data)



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
    
    
    with open(filename, encoding="cp932") as csvfile:
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



if __name__ == '__main__':
    
    sessionDir = tools.create_session_dir("utteranceVectorizer_databaseLearning")
    
    dataDirectory = tools.dataDir+"2019-12-05_14-58-11_advancedSimulator9"
    numTrainDbs = 10
    numInteractionsPerDb = 200
    
    participant = "shopkeeper" 
    
    
    #
    # read in the utterance data
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
        allShkpUtts += [row["SHOPKEEPER_SPEECH_WITH_SYMBOLS"] for row in inters]
    
    
    
    if participant == "shopkeeper":
        utterances = allShkpUtts
    elif participant == "customer":
        utterances = allCustUtts
    else:
        print("WARNING:", participant, "is not a valid participant!")
        utterances = []
    
    
    uniqueUtterances = list(set(utterances))
    keywords = []
    
    print("loaded", len(utterances), participant, "utterances.")
    print(len(uniqueUtterances), "unique utterances")
    
    
    #
    # load the keywords
    #
    keywordsDir = dataDirectory+"/keywords/"
    
    # read the csv
    uniqueShkpUttsWithSymbols = []
    
    with open(keywordsDir+"unique_shopkeeper_speech_with_symbols.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            uniqueShkpUttsWithSymbols.append(row["SHOPKEEPER_SPEECH_WITH_SYMBOLS"])
    
    # read the json keyword files
    keywordsForUniqueShkpUttsWithSymbols = [None] * len(uniqueShkpUttsWithSymbols)
    
    keywordsFilenames = filenames = [fn for fn in os.listdir(keywordsDir) if "keywords" in fn]
    
    for fn in keywordsFilenames:
        index = int(fn.split("_")[0])
        kwList = []
        
        with open(keywordsDir+fn) as f:
            contents = f.read()
            contents = eval(contents)
            
            if contents != "NA":
                for kw in contents["keywords"]:
                    kwList.append(kw["text"])
        
        keywordsForUniqueShkpUttsWithSymbols[index] = kwList
    
    
    keywordCounts = {}
    shkpUttToKeywords = {}
    
    for i in range(len(keywordsForUniqueShkpUttsWithSymbols)):
        kwList = keywordsForUniqueShkpUttsWithSymbols[i]
        
        shkpUtt = uniqueShkpUttsWithSymbols[i].lower()
        
        shkpUttToKeywords[shkpUtt] = []
        
        for kw in kwList:
            for w in kw.split():
                w = w.lower()
                
                if w not in keywordCounts:
                    keywordCounts[w] = 0
                keywordCounts[w] += 1
                
                shkpUttToKeywords[shkpUtt].append(w)
    
    importantKeywords = []
    for w, count in keywordCounts.items():
        if count >= 2:
            importantKeywords.append(w) 
    
    
    #
    # create the keyword vectors
    #
    shkpUttToKwVec = {}
    
    for utt, kws in shkpUttToKeywords.items():
        vec = np.zeros(len(importantKeywords))
        
        for w in kws:
            try:
                i = importantKeywords.index(w)
                vec[i] = 1.0
            except:
                pass
        
        shkpUttToKwVec[utt] = vec
    
    
    #
    # sample utterances to reduce the number to N
    #
    """
    N = 40000
    
    #uniqueUtts = uniqueUtterances.keys()
    #sampleProbs = [count/float(sum(uniqueUtterances.values())) for count in uniqueUtterances.values()]
    
    numToSample = N - len(uniqueUtterances)
    sampledUtterances = []
    
    # make sure each utterance appears at least once
    sampledUtterances += list(uniqueUtterances)
    
    # make sure that less common utterances are well represented (because these include mem-dep utterances)
    #for utt, count in uniqueUtterances.items():
    #    if count <= 5:
    #        sampledUtterances += [utt] * 5
    
    # sample
    sampledUtterances += np.random.choice(utterances, size=numToSample).tolist()
    
    utterances = sampledUtterances
    """
    
    
    #
    # vectorize the utterances
    #
    print("vectorizing utterances...")
    
    keywords = sorted(list(set(keywords)))
    
    uttVectorizer = UtteranceVectorizer(utterances,
                                        minCount=2, 
                                        keywordWeight=1.0, 
                                        keywordSet=keywords, 
                                        unigramsAndKeywordsOnly=False, 
                                        tfidf=False,
                                        useStopwords=False,
                                        lsa=False)
    
    
    vectors = uttVectorizer.get_utterance_vectors(utterances)
    uniqueVectors = uttVectorizer.get_utterance_vectors(uniqueUtterances)
    
    vectorsNoNan = []
    utterancesNoNan = []
    
    uniqueVectorsNoNan = []
    #uniqueUtterancesNoNan = []
    
    for i in range(len(utterances)):
        if not vectors[i,:].any():
            #print("removing:", utterances[i])
            pass
        else:
            vectorsNoNan.append(vectors[i,:])
            utterancesNoNan.append(utterances[i])
    
    """
    for i in range(len(uniqueUtterances)):
        if not uniqueVectors[i,:].any():
            #print("removing:", uniqueUtterances[i])
            pass
        else:
            uniqueVectorsNoNan.append(uniqueVectors[i,:])
            uniqueUtterancesNoNan.append(uniqueUtterances[i])
    """
    
    #
    # add on the keyword vectors
    #
    for i in range(len(utterancesNoNan)):
        vectorsNoNan[i] = np.concatenate((vectorsNoNan[i], shkpUttToKwVec[utterancesNoNan[i].lower()]))
    
    
    #
    # compute distances
    #
    print("computing distances...")
    
    # compute distances between each unique pair of utterances    
    #uniqueVectorsNoNan = np.asarray(uniqueVectorsNoNan)
    #uniqueDistMatrix = pairwise_distances(uniqueVectorsNoNan, metric="cosine", n_jobs=50)
    
        
    print("creating full distance matrix...")
    
    distMatrix = pairwise_distances(vectorsNoNan, metric="cosine", n_jobs=50)
    #distMatrix = np.zeros((len(utterancesNoNan), len(utterancesNoNan)))
    
    """
    # the reason the code is written this way is to make it run faster...
    uniqueUttNoNanToIndex = {} # shows where to find the utterances in the uniqueUtterancesNoNan list
    uttNoNanToIndices = {} # shows where to find the utterance in the utterancesNoNan list
    
    for utt in uniqueUtterancesNoNan:
        uniqueUttNoNanToIndex[utt] = uniqueUtterancesNoNan.index(utt)
        uttNoNanToIndices[utt] = []
    
    for i in range(len(utterancesNoNan)):
        uttNoNanToIndices[utterancesNoNan[i]].append(i)
    
    
    
    count = 0
    numToCompute = math.pow(len(uniqueUtterancesNoNan), 2) / 2
    
    
    for k in range(len(uniqueUtterancesNoNan)-1):
        for l in range(k+1, len(uniqueUtterancesNoNan)):
            
            utt1 = uniqueUtterancesNoNan[k]
            utt2 = uniqueUtterancesNoNan[l]
            
            distMatrix[uttNoNanToIndices[utt1]+uttNoNanToIndices[utt2], uttNoNanToIndices[utt2]+uttNoNanToIndices[utt1]] = uniqueDistMatrix[uniqueUttNoNanToIndex[utt1], uniqueUttNoNanToIndex[utt2]]            
            
            count += 1
            print("completed {} of {} ({:.2})".format(count, numToCompute, count/numToCompute))
    """ 
    
    
    #
    # save vectorizations, etc. to file
    #
    print("saving...")
    
    vectorsNoNan = np.asarray(vectorsNoNan)
    
    print("num no nan utts:", len(utterancesNoNan))
    print("dimensionality:", vectorsNoNan.shape[1])
    
    
    date = time.strftime("%Y%m%d")    
    condition = "{}_simulated_data_csshkputts_withsymbols_200 {} - tri stm - 1 wgt kw - mc2 - stopwords 1".format(date, participant)
    
    
    np.savetxt(sessionDir+"/{} - utterance cos dists.txt".format(condition), distMatrix, fmt="%.4f")
    
    np.savetxt(sessionDir+"/{} - utterance vectors.txt".format(condition), vectorsNoNan, fmt="%d")
    
    
    print("pickling the utterance vectorizer...")
    pkl.dump(uttVectorizer, open(sessionDir+"/{}_utterance_vectorizer.pkl".format(condition) ,"wb"))
    
    
    with open(sessionDir+"/{} - utterance data.csv".format(condition), "w", newline="") as csvfile:
        
        fieldnames = ["Utterance ID", "Timestamp", "Trial ID", "Condition", "Utterance"]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(len(utterancesNoNan)):
            
            data = {"Utterance ID": i,
                    "Utterance":utterancesNoNan[i]}
            
            writer.writerow(data)
    
    
    




        
    




