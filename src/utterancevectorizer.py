'''
Created on Feb 13, 2017

@author: MalcolmD
'''



import numpy as np
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from scipy.spatial.distance import cosine, cdist, pdist, squareform
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from nltk.stem import WordNetLemmatizer
import nltk
import copy
import csv
#import editdistance
from sklearn.manifold import TSNE
#from gensim import corpora, matutils, models
from scipy import sparse
import pickle as pkl

import tools




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
                
                if word.isdigit():
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
        tokenized = nltk.word_tokenize(utt)
        
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
    
    
    
    def get_utterance_vectors(self, uttList, returnAll=False, asArray=True):
        
        uttVecs = []
        
        for utt in uttList:
            uttVecs.append(self.get_utterance_vector(utt, returnAll))
        
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
    
    



if __name__ == '__main__':
    
    sessionDir = tools.create_session_dir("utteranceVectorizer - 50000_simulated_interactions_2018-4-26")
    
    
    #
    # read in the utterance data
    #
    print("loading interaction data...")
    
    participant = "shopkeeper" 
    
    utterances = []
    uniqueUtteranceToCount = {}
    keywords = []
    
    with open(tools.dataDir+"50000_simulated_interactions_2018-4-26.csv", "rb") as csvfile: 
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            
            if participant == "shopkeeper":
                if row["SHOPKEEPER_SPEECH"] != "" and row["SHOPKEEPER_SPEECH"] != "NONE":
                    
                    utt = row["SHOPKEEPER_SPEECH"]
                    
                    utterances.append(utt)
                    if row["SHOPKEEPER_KEYWORDS"] != "" and row["SHOPKEEPER_KEYWORDS"] != "NONE" and row["SHOPKEEPER_KEYWORDS"] != "NO_KEYWORD":
                        keywords += row["SHOPKEEPER_KEYWORDS"].split(";")
                    
                    if utt not in uniqueUtteranceToCount:
                        uniqueUtteranceToCount[utt] = 0
                    uniqueUtteranceToCount[utt] += 1
                    
            elif participant == "customer":
                if row["CUSTOMER_SPEECH"] != "" and row["CUSTOMER_SPEECH"] != "NONE":   
                    
                    utt = row["CUSTOMER_SPEECH"]
                    
                    utterances.append(row["CUSTOMER_SPEECH"])
                    if row["CUSTOMER_KEYWORDS"] != "" and row["CUSTOMER_KEYWORDS"] != "NONE" and row["CUSTOMER_KEYWORDS"] != "NO_KEYWORD":
                        keywords += row["CUSTOMER_KEYWORDS"].split(";")
                    
                    if utt not in uniqueUtteranceToCount:
                        uniqueUtteranceToCount[utt] = 0
                    uniqueUtteranceToCount[utt] += 1
                    
    
    print("loaded", len(utterances), "utterances.")
    print(len(uniqueUtteranceToCount), "unique utterances")
    
    
    
    #
    # sample utterances to reduce the number to N
    #
    N = 25000
    
    #uniqueUtts = uniqueUtteranceToCount.keys()
    #sampleProbs = [count/float(sum(uniqueUtteranceToCount.values())) for count in uniqueUtteranceToCount.values()]
    
    numToSample = N - len(uniqueUtteranceToCount)
    sampledUtterances = []
    
    # make sure each utterance appears at least once
    sampledUtterances += list(uniqueUtteranceToCount.keys())
    
    # make sure that less common utterances are well represented (because these include mem-dep utterances)
    #for utt, count in uniqueUtteranceToCount.items():
    #    if count <= 5:
    #        sampledUtterances += [utt] * 5
            
            
    
    # sample
    sampledUtterances += np.random.choice(utterances, size=numToSample).tolist()
    
    utterances = sampledUtterances
    
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
    
    
    print("pickling the utterance vectorizer...")
    pkl.dump(uttVectorizer, open(sessionDir+"/{}_utterance_vectorizer.pkl".format(participant) ,"w"))
    
    
    vectors = uttVectorizer.get_utterance_vectors(utterances)
    #vectors = uttVectorizer.get_lsa_vectors(utterances)
    
    vectorsNoNan = []
    utterancesNoNan = []
    
    for i in range(len(utterances)):
        
        if not vectors[i,:].any():
            print("removing:", utterances[i])
        
        else:
            vectorsNoNan.append(vectors[i,:])
            utterancesNoNan.append(utterances[i])
    
    
    #
    # compute distances
    #
    print("computing distances...")
    
    # compute distances between each unique pair of utterances
    uttToUttToDist = {}
    
    for utt1 in list(uniqueUtteranceToCount.keys()):
        uttToUttToDist[utt1] = dict.fromkeys(list(uniqueUtteranceToCount.keys()))
        
        uttVec1 = uttVectorizer.get_utterance_vector(utt1)
        
        for utt2 in list(uniqueUtteranceToCount.keys()):
            
            uttVec2 = uttVectorizer.get_utterance_vector(utt2)
            
            uttToUttToDist[utt1][utt2] = cdist(uttVec1.reshape(1, uttVec1.shape[0]), uttVec2.reshape(1, uttVec2.shape[0]), "cosine")
    
    
    distMatrix = np.ones((len(utterancesNoNan), len(utterancesNoNan)))
    
    for i in range(len(utterancesNoNan)):
        for j in range(len(utterancesNoNan)):
            distMatrix[i,j] = uttToUttToDist[utterancesNoNan[i]][utterancesNoNan[j]]
    
    
    vectorsNoNan = np.asarray(vectorsNoNan)
    #distMatrix = squareform(pdist(vectorsNoNan, "cosine"))
    
    
    #
    # save vectorizations, etc. to file
    #
    print("saving...")
    
    print("num no nan utts:", len(utterancesNoNan))
    print("dimensionality:", vectorsNoNan.shape[1])
    
    condition = "50000_simulated_interactions_2018-4-26 {} - tri stm - 1 wgt kw - mc2 - stopwords 1".format(participant)
    
    np.savetxt(sessionDir+"/utterance cos dists - {:}.txt".format(condition), distMatrix, fmt="%.4f")
    
    #np.savetxt(sessionDir+"/utterance vectors - {:}.txt".format(condition), vectorsNoNan, fmt="%d")
    
    
    with open(sessionDir+"/utterance data - {:}.csv".format(condition), "wb") as csvfile:
        
        fieldnames = ["Utterance ID", "Timestamp", "Trial ID", "Condition", "Utterance"]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for i in range(len(utterancesNoNan)):
            
            data = {"Utterance":utterancesNoNan[i]}
            
            writer.writerow(data)
    
    
    




        
    




