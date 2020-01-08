'''
Created on Dec 16, 2016

@author: MalcolmD
'''


import numpy as np
import csv
from sklearn.metrics.pairwise import pairwise_distances
import copy
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import MWETokenizer
import nltk
import editdistance

import tools


# for proposed
#shopkeeperSpeechClusterFilename = tools.modelDir + "20191212_simulated_data_csshkputts_withsymbols_200 shopkeeper - tri stm - 1 wgt kw - mc2 - stopwords 1- speech_clusters.csv"
#shopkeeperSpeechVectorFilename = tools.dataDir+"/utterance vectors/" + "20191212_simulated_data_csshkputts_withsymbols_200 shopkeeper - tri stm - 1 wgt kw - mc2 - stopwords 1 - utterance vectors.txt"

# for baseline 1
shopkeeperSpeechClusterFilename = tools.modelDir + "20191219_simulated_data_csshkputts_nosymbols_200 shopkeeper - tri stm - 1 wgt kw - mc2 - stopwords 1- speech_clusters.csv"
shopkeeperSpeechVectorFilename = tools.dataDir+"/utterance vectors/" + "20191219_simulated_data_csshkputts_nosymbols_200 shopkeeper - tri stm - 1 wgt kw - mc2 - stopwords 1 - utterance vectors.txt"


class ShopkeeperSpeechClusters(object):
    
    def __init__(self, sessionDir, shopkeeperSpeechClusterFilename, shopkeeperSpeechVectorFilename):
        
        self.sessionDir = sessionDir
        
        # load the speech clusters
        print("loading speech clusters...")
        self.load_speech_clusters(shopkeeperSpeechClusterFilename)
        
        # load the utterance vectorizer
        print("loading speech vectors...")
        self.load_speech_vectors(shopkeeperSpeechVectorFilename)
        
        
        print("finding representative utterances...")
        self.find_new_representative_utterances()
        print("finished finding representative utterances.")
        
        
        print("computing nearest neighbors of junk utterances...")
        self.compute_nearest_neighbors_of_junk_utterances()
        print("finished computing nearest neighbors of junk utterances.")
        
        
        self.save_speech_clusters(sessionDir+"/modified_speech_clusters.csv")
        
        
        
    
    def load_speech_clusters(self, filename):
        
        self.allData = []
        
        self.shkpUttToSpeechClustId = {}
        self.shkpSpeechClustIdToRepUtt = {}
        self.junkSpeechClusterIds = []
        
        with open(filename, encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                self.header = reader.fieldnames
                
                for row in reader:
                    row["Utterance.ID"] = int(row["Utterance.ID"])
                    row["Cluster.ID"] = int(row["Cluster.ID"])
                    row["Is.Representative"] = int(row["Is.Representative"])
                    row["Is.Junk"] = int(row["Is.Junk"])
                    
                    utt = row["Utterance"].lower()
                    
                    if utt not in self.shkpUttToSpeechClustId:
                        self.shkpUttToSpeechClustId[utt] = row["Cluster.ID"]
                    
                    elif utt in self.shkpUttToSpeechClustId:
                        if row["Cluster.ID"] != self.shkpUttToSpeechClustId[utt]:
                            print("WARNING: Shopkeeper utterance \"{}\" is in multiple speech clusters!".format(utt))
                    
                    if row["Is.Representative"] == 1:
                        self.shkpSpeechClustIdToRepUtt[int(row["Cluster.ID"])] = utt
                    
                    if row["Is.Junk"] == 1 and row["Cluster.ID"] not in self.junkSpeechClusterIds:
                        self.junkSpeechClusterIds.append(int(row["Cluster.ID"]))
                    
                    self.allData.append(row)
    
    
    def load_speech_vectors(self, filename):
        # the Utterance.ID field in the speech cluster data should point 
        # to where in this file the utterance vector can be found
        self.utteranceVectors = np.loadtxt(filename)
    
    
    def save_speech_clusters(self, filename):
        
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, self.header)
            writer.writeheader()
            
            for row in self.allData:
                writer.writerow(row)
    
    
    def compute_centroids(self):
        """compute centroids of the good speech clusters"""
        
        clustIdToSpeechVecs = {}
        
        for row in self.allData:
            clustId = row["Cluster.ID"]
            speechVec = self.utteranceVectors[row["Utterance.ID"]]
            
            if clustId not in clustIdToSpeechVecs:
                clustIdToSpeechVecs[clustId] = []
                
            clustIdToSpeechVecs[clustId].append(speechVec)
            
        
        self.clustIdToCentroid = {}
        
        for clustId in clustIdToSpeechVecs:
            if clustId not in self.junkSpeechClusterIds:
                self.clustIdToCentroid[clustId] = np.mean(clustIdToSpeechVecs[clustId], axis=0)
    
    
    def compute_nearest_neighbors_of_junk_utterances(self):
        #
        # compute centroids
        #
        self.compute_centroids()
        
        #
        # find nearest neighbors
        #
        classes = np.asarray(list(self.clustIdToCentroid.keys()))
        centroids = list(self.clustIdToCentroid.values())
        
        junkRows = [row for row in self.allData if row["Cluster.ID"] in self.junkSpeechClusterIds]
        junkSpeechVecs = [self.utteranceVectors[row["Utterance.ID"]] for row in junkRows]
        
        distances = pairwise_distances(junkSpeechVecs, centroids, metric="cosine")
        
        nearestIndices = distances.argmin(axis=1)
        nearestCentroids = classes[nearestIndices]
        nearestCentroidDistances = []
        for i in range(len(junkRows)):
            nearestCentroidDistances.append(distances[i, nearestIndices[i]])
        
        #
        # print the results (we need to find a good threshold distance 
        # for deciding whether to add the junk utt to a good cluster)
        #
        
        # add the nearest centroid data to the csv rows
        junkUttNearestCentroidRows = copy.deepcopy(junkRows)
        
        for i in range(len(junkUttNearestCentroidRows)):
            junkUttNearestCentroidRows[i]["NEAREST_GOOD_SPEECH_CLUSTER"] = nearestCentroids[i]
            junkUttNearestCentroidRows[i]["NEAREST_GOOD_SPEECH_CLUSTER_CENTROID_DISTANCE"] = nearestCentroidDistances[i]
            junkUttNearestCentroidRows[i]["NEAREST_GOOD_SPEECH_CLUSTER_REPRESENTATIVE_UTTERANCE"] = self.shkpSpeechClustIdToRepUtt[nearestCentroids[i]]
        
        # sort for convenience
        #junkUttNearestCentroidRows.sort(key=lambda x: x["NEAREST_GOOD_SPEECH_CLUSTER_CENTROID_DISTANCE"])
        
        # save
        with open(sessionDir+"/junk_utterance_nearest_centroids.csv", "w", newline="") as csvfile:
            
            fieldnames = self.header + ["NEAREST_GOOD_SPEECH_CLUSTER", "NEAREST_GOOD_SPEECH_CLUSTER_CENTROID_DISTANCE", "NEAREST_GOOD_SPEECH_CLUSTER_REPRESENTATIVE_UTTERANCE"]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in junkUttNearestCentroidRows:
                writer.writerow(row)
        
        
        #
        # add utterances with nearest centroids under a threshold distance away to their nearest cluster
        #
        thresh = 0.70
        
        for i in range(len(junkRows)):
            junkRow = junkRows[i]
            junkUttNearestCentroidRow = junkUttNearestCentroidRows[i]
            
            if junkUttNearestCentroidRow["NEAREST_GOOD_SPEECH_CLUSTER_CENTROID_DISTANCE"] <= thresh:
                
                # find the row in the original cluster csv rows and change the cluster ID
                index = self.allData.index(junkRow)
                self.allData[index]["Cluster.ID"] = junkUttNearestCentroidRow["NEAREST_GOOD_SPEECH_CLUSTER"]
                self.allData[index]["Is.Representative"] = 0
                self.allData[index]["IS_NEW_REPRESENTATIVE"] = 0
                
    
    
    def find_new_representative_utterances(self):
        """find the medoid with levenshtein distanace"""
        
        #
        # tokenize each utterance        
        # make sure that the DB contents symbols are left in and not split up
        #
        tokenizer = nltk.tokenize.MWETokenizer()
        
        tokenizer.add_mwe(('<', 'camera_ID', '>'))
        tokenizer.add_mwe(('<', 'camera_id', '>'))
        tokenizer.add_mwe(('<', 'camera_name', '>'))
        tokenizer.add_mwe(('<', 'camera_type', '>'))
        tokenizer.add_mwe(('<', 'color', '>'))
        tokenizer.add_mwe(('<', 'weight', '>'))
        tokenizer.add_mwe(('<', 'preset_modes', '>'))
        tokenizer.add_mwe(('<', 'effects', '>'))
        tokenizer.add_mwe(('<', 'price', '>'))
        tokenizer.add_mwe(('<', 'resolution', '>'))
        tokenizer.add_mwe(('<', 'optical_zoom', '>'))
        tokenizer.add_mwe(('<', 'settings', '>'))
        tokenizer.add_mwe(('<', 'autofocus_points', '>'))
        tokenizer.add_mwe(('<', 'sensor_size', '>'))
        tokenizer.add_mwe(('<', 'ISO', '>'))
        tokenizer.add_mwe(('<', 'iso', '>'))
        tokenizer.add_mwe(('<', 'long_exposure', '>'))
        
        tokenizedRowData = copy.deepcopy(self.allData)
        
        for row in tokenizedRowData:
            tokenized = tokenizer.tokenize(nltk.word_tokenize(row["Utterance"].lower()))
            row["TOKENIZED_UTTERANCE"] = tokenized
        
        
        #
        # find the new representative utterances
        #
        clustIdToRowData = {}
        clustIdToNewRepresentativeUtterance = {}
        
        for row in tokenizedRowData:
            if row["Cluster.ID"] not in clustIdToRowData:
                clustIdToRowData[row["Cluster.ID"]] = []
            
            clustIdToRowData[row["Cluster.ID"]].append(row)
            
        
        # compute levenshtein distances between each pair of utterance in each cluster
        for clustId in clustIdToRowData:
            if clustId not in self.junkSpeechClusterIds:
                
                mostSimIndex = None
                mostSimAveDist = None
                
                for i in range(len(clustIdToRowData[clustId])):
                    tokensA = clustIdToRowData[clustId][i]["TOKENIZED_UTTERANCE"]
                    
                    distances = []
                    
                    for j in range(len(clustIdToRowData[clustId])):
                        tokensB = clustIdToRowData[clustId][j]["TOKENIZED_UTTERANCE"]
                        
                        # normalized (by token sequence len) levenshtein distance
                        distance = editdistance.eval(tokensA, tokensB) / float(max(len(tokensA), len(tokensB)))
                        distances.append(distance)
                    
                    aveDist = np.mean(distances)
                    
                    if mostSimIndex == None or aveDist < mostSimAveDist:
                        mostSimIndex = i
                        mostSimAveDist = aveDist
                    
                    
                    clustIdToNewRepresentativeUtterance[clustId] = clustIdToRowData[clustId][mostSimIndex]["Utterance"]
        
        
        # add the new representative utterance to the csv row data
        for i in range(len(self.allData)):
            if self.allData[i]["Cluster.ID"] not in self.junkSpeechClusterIds and self.allData[i]["Utterance"].lower() == clustIdToNewRepresentativeUtterance[self.allData[i]["Cluster.ID"]].lower():
                self.allData[i]["IS_NEW_REPRESENTATIVE"] = 1
            else:
                self.allData[i]["IS_NEW_REPRESENTATIVE"] = 0
        
        self.header.insert(self.header.index("Is.Representative"), "IS_NEW_REPRESENTATIVE")
        
    
    
    







if __name__ == '__main__':
    
    sessionDir = tools.create_session_dir("speechClusterPostProcessing")
    
    shkpSpeechClusts = ShopkeeperSpeechClusters(sessionDir, shopkeeperSpeechClusterFilename, shopkeeperSpeechVectorFilename)
    
    print("finished")