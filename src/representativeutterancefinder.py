'''
Created on Feb 16, 2017

@author: MalcolmD
'''

import numpy as np
from scipy.spatial.distance import cosine, jaccard
import csv
from jellyfish import damerau_levenshtein_distance, levenshtein_distance, jaro_winkler
import editdistance
import nltk
from nltk.stem import WordNetLemmatizer

import tools
from utteranceVectorizer import UtteranceVectorizer



class RepresentativeUtteranceFinder(object):
    
    def __init__(self, clustToTopic, clustToUttIds, clustToUtts, clustToRepUtt, utteranceVectorizer, distanceMetric, tfidf):
        
        
        
        self.distanceMetric = distanceMetric
        self.tfidf = tfidf
        
        self.wnl = WordNetLemmatizer()
        
        clustToCentroid = {}
        
        for clustId, utts in clustToUtts.items():
            
            uttVecs = []
            
            for utt in utts:
                uttVec = utteranceVectorizer.get_utterance_vector(utt)
                uttVecs.append(uttVec)
            
            clustToCentroid[clustId] = np.average(np.asarray(uttVecs, dtype=np.float64), axis=0)
        
        
        #
        # find the utterance closest to the centroid
        #
        self.clustToClosestToCentroidUtt = {}
        self.clustToClosestToCentroidUtt[0] = "BAD CLUSTER"
        
        count = 0
        
        for clustId, utts in clustToUtts.items():
            
            closestUtt = None
            closestDist = None
            
            for utt in utts:
                uttVec = utteranceVectorizer.get_utterance_vector(utt)
                dist = self.distanceMetric(uttVec,clustToCentroid[clustId])
                
                if closestUtt == None or dist < closestDist:
                    closestUtt = utt
                    closestDist = dist
            
            self.clustToClosestToCentroidUtt[clustId] = closestUtt
            
            count += 1
            print "centroid", count, "of", len(clustToUtts)
        
        
        #
        # find the utterance most similar to all other utterances in the cluster
        #
        self.clustToMinAveDistUtt = {}
        self.clustToMinAveDistUtt[0] = "BAD CLUSTER"
        
        count = 0
        
        for clustId, utts in clustToUtts.items():
            
            mostSimUtt = None
            minAveDistDist = None
            
            if clustId == 0:
                continue
            
            for utt in utts:
                
                uttLemmas = [self.wnl.lemmatize(t).lower() for t in nltk.word_tokenize(utt)]
                
                uttVec = utteranceVectorizer.get_utterance_vector(utt)
                distSum = 0.0
                
                for utt2 in utts:
                    
                    uttLemmas2 = [self.wnl.lemmatize(t).lower() for t in nltk.word_tokenize(utt2)]
                    
                    uttVec2 = utteranceVectorizer.get_utterance_vector(utt2)
                    
                    #dist = self.distanceMetric(uttVec,uttVec2)
                    
                    #dist = levenshtein_distance(utt.decode("utf-8"),utt2.decode("utf-8")) / float(max(len(utt), len(utt2)))
                    
                    dist = editdistance.eval(uttLemmas, uttLemmas2) / float(max(len(uttLemmas), len(uttLemmas2)))
                    
                    
                    distSum += dist
                
                aveDist = distSum / float(len(utts))
                
                if mostSimUtt == None or aveDist < minAveDistDist:
                    mostSimUtt = utt
                    minAveDistDist = aveDist
            
            self.clustToMinAveDistUtt[clustId] = mostSimUtt
            
            count += 1
            print "medoid", count, "of", len(clustToUtts)
        
    
    def load_speech_cluster_data(self, filename):
        
        self.speechClusterData = []
        
        with open(filename, encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    self.speechClusterData.append(row)
        
        
        
        
    
    def find_new_representative_utterances(self):
        pass
    
    
    
    def save_speech_clusters(self, filename):
        header = ["topic", "clusterIds", "isOldMedoid", "isNewMedoid", "isClosestToCentroid", "utteranceIds", "utts"]
        rows = []
        
        clustIds = list(set(self.clustToUttIds.keys()))
        clustIds.sort()
        
        # create file contents
        for clustId in clustIds:
            
            for i in range(len(self.clustToUttIds[clustId])):
                row = {}
                
                uttId = self.clustToUttIds[clustId][i]
                utt = self.clustToUtts[clustId][i]
                
                row["clusterIds"] = clustId
                row["utteranceIds"] = uttId
                row["utts"] = '"'+utt+'"'
                
                
                if i == 0 and clustId >= 0: 
                    try:
                        row["topic"] = self.clustToTopic[clustId]
                    except:
                        print "Warning: No topic for cluster {:}!".format(clustId)
                else:
                    row["topic"] = ""
                
                
                try:
                    if clustId >= 0 and self.clustToRepUtt[clustId] == utt:
                        row["isOldMedoid"] = 1
                    else:
                        row["isOldMedoid"] = 0
                    
                    
                    if clustId >= 0 and self.clustToClosestToCentroidUtt[clustId] == utt:
                        row["isClosestToCentroid"] = 1
                    else:
                        row["isClosestToCentroid"] = 0
                    
                    
                    if clustId >= 0 and self.clustToMinAveDistUtt[clustId] == utt:
                        row["isNewMedoid"] = 1
                    else:
                        row["isNewMedoid"] = 0
                
                except Exception, e:
                    print "Warning: clust", clustId
                    print str(e)
                    
                
                rows.append(row)
        
        # write to file
        with open(filename, "wb") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
        
            writer.writeheader()
            
            for row in rows:
                writer.writerow(row)
        
    
    
    def save_representative_utterances(self, filename):
        
        header = ["clustId","clustSize","oldMedoid","newMedoid","closestToCentroid"]
        rows = []
        
        for clustId in self.clustToUtts:
            
            row = {"clustId":clustId,
                   "clustSize":len(self.clustToUtts[clustId]),
                   "oldMedoid":self.clustToRepUtt[clustId],
                   "newMedoid":self.clustToMinAveDistUtt[clustId],
                   "closestToCentroid":self.clustToClosestToCentroidUtt[clustId]
                   }
            rows.append(row)
        
        # write to file
        with open(filename, "wb") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
        
            writer.writeheader()
            
            for row in rows:
                writer.writerow(row)
            
            


if __name__ == '__main__':
    
    sessionDir = tools.create_session_dir("representativeUtteranceFinder")
    
    
    #
    # load the raw sequences
    #
    dialogMap, cUniqueUtterances, aUniqueUtterances, cAllUtterances, aAllUtterances = tools.load_raw_sequences(tools.dataDir + "new raw sequences - replace 40.csv")
    
    """
    #
    # load the agent speech clusters
    #
    aClustToTopic, aClustToUttIds, aClustToUtts, aClustToRepUtt = tools.load_r_clusters(tools.modelDir + "action clusterings/new travel agent clusters 4 - with rs - renumbered.csv")
    aUttToClust = tools.get_utt_to_clust_map(aClustToUtts)
    
    
    #
    # train the vectorizer
    #
    agentUtteranceVectorizer = UtteranceVectorizer("agent", aAllUtterances, minCount=0, keywordWeight=2.0)
    
    
    #
    # find the new rep utts
    #
    aRepUttFinder = RepresentativeUtteranceFinder(aClustToTopic, aClustToUttIds, aClustToUtts, aClustToRepUtt, 
                                                 agentUtteranceVectorizer, distanceMetric=cosine, tfidf=False)
    
    aRepUttFinder.save_speech_clusters(sessionDir+"/agent speech clusters.csv")
    
    aRepUttFinder.save_representative_utterances(sessionDir+"/agent representative utterances - old lev cent.csv")
    
    tools.save_representative_utterances(aRepUttFinder.clustToClosestToCentroidUtt, sessionDir+"/agent representative utterances - closest to centroid tf cosine kw2 mc0.csv")
    tools.save_representative_utterances(aRepUttFinder.clustToMinAveDistUtt, sessionDir+"/agent representative utterances - levenshtein normalized medoid.csv")
    
    
        
    """
    #
    # load the customer speech clusters
    #
    cClustToTopic, cClustToUttIds, cClustToUtts, cClustToRepUtt = tools.load_r_clusters(tools.modelDir + "action clusterings/new customer clusters 4 - with rs 1 - lev rep utt - merged - renumbered.csv")
    cUttToClust = tools.get_utt_to_clust_map(cClustToUtts)
    
    
    #
    # train the vectorizer
    #
    customerUtteranceVectorizer = UtteranceVectorizer("customer", cAllUtterances, minCount=0, keywordWeight=3.0)
    
    
    #
    # find the new rep utts
    #
    cRepUttFinder = RepresentativeUtteranceFinder(cClustToTopic, cClustToUttIds, cClustToUtts, cClustToRepUtt, 
                                                 customerUtteranceVectorizer, distanceMetric=cosine, tfidf=False)
    
    cRepUttFinder.save_speech_clusters(sessionDir+"/customer speech clusters.csv")
    
    cRepUttFinder.save_representative_utterances(sessionDir+"/customer representative utterances - old lev cent.csv")
    
    tools.save_representative_utterances(cRepUttFinder.clustToClosestToCentroidUtt, sessionDir+"/customer representative utterances - closest to centroid tf cosine kw3 mc0.csv")
    tools.save_representative_utterances(cRepUttFinder.clustToMinAveDistUtt, sessionDir+"/customer representative utterances - levenshtein normalized medoid.csv")
    
    
    
    
