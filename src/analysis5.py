'''
Created on Aug 1, 2019

@author: robovie

count which features the shopkeeper introduces

'''


import csv
import numpy as np


import tools



sessionDir = tools.create_session_dir("analysis5")


#
# load the human-human data
#
humanHumanDataFilename = tools.dataDir+"ClusterTrainer_ClusterSequence_detailed_annotated.csv"


rawData = []

with open(humanHumanDataFilename) as csvfile:
    reader = csv.DictReader(csvfile)
      
    for row in reader:
        rawData.append(row)



introducedFeatures = []
introduceFeatureActionCount = 0.0

for i in range(len(rawData)):
    if rawData[i]["SHOPKEEPER_NEW_FEATURE"] == "1" and rawData[i]["CAMERA_FEATURE"] != "":
        introducedFeatures += rawData[i]["CAMERA_FEATURE"].split(",")
        introduceFeatureActionCount += 1.0

features = sorted(list(set(introducedFeatures)))



featureCounts = {}

for f in features:
    featureCounts[f] = introducedFeatures.count(f)
    print(f, "\t", round(featureCounts[f]/introduceFeatureActionCount, 4))







"""
with open(sessionDir+"/transition_counts.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["PRE", "POST", "COUNT", "CONDITIONAL_PROBABILITY"])
    
    for transition, count in sortedTransitions:
        pre, post = transition.split("->")
        
        print(transition, count, transitionProbs[pre][post])
        writer.writerow([pre, post, count, transitionProbs[pre][post]])
"""

