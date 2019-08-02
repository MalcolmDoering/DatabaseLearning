'''
Created on Aug 1, 2019

@author: robovie

compute camera transitions from the human-human dataset

'''


import csv
import numpy as np


import tools



sessionDir = tools.create_session_dir("analysis4")


#
# load the human-human data
#
humanHumanDataFilename = tools.dataDir+"ClusterSequence_detailed_fix_memoryannotations.csv"


hhInteractions = []
rawData = []

with open(humanHumanDataFilename) as csvfile:
    reader = csv.DictReader(csvfile)
      
    for row in reader:
        rawData.append(row)



interaction = []
for i in range(len(rawData)):
    
    if (i > 0) and (rawData[i]["TRIAL"] != rawData[i-1]["TRIAL"]):
        hhInteractions.append(interaction)
        interaction = []
        
    interaction.append(rawData[i])


#
# compute location sequences
#
spatialFormationSequences = []

for interaction in hhInteractions:
    
    sfs = []
    
    for turn in interaction:
        
        sf = "{}-{}".format(turn["SPATIAL_STATE"], turn["STATE_TARGET"])
        
        if ((len(sfs) == 0) or (sfs[-1] != sf)) and (turn["SPATIAL_STATE"] == "PRESENT_X"):
            sfs.append(sf)
    
    spatialFormationSequences.append(sfs)

sfsLens = [len(sfs) for sfs in spatialFormationSequences]

print("spatial formation sequences lenght ave (sd)", np.mean(sfsLens), np.std(sfsLens))


#
# compute transitions
#
transitionCounts = {}
transitionProbs = {}

for sfs in spatialFormationSequences:
    
    for i in range(len(sfs)):
        
        preSf = sfs[:i]
        preSf = sorted(list(set(preSf)))
        
        pre = ",".join(preSf)
        post = sfs[i]
        
        
        
        if pre not in transitionCounts:
            transitionCounts[pre] = {}
        if post not in transitionCounts[pre]:
            transitionCounts[pre][post] = 0.0
        
        transitionCounts[pre][post] += 1.0


for pre in transitionCounts:
    transitionProbs[pre] = {}
    
    for post in transitionCounts[pre]:
        transitionProbs[pre][post] = transitionCounts[pre][post] / sum(transitionCounts[pre][p] for p in transitionCounts[pre].keys())
    

transitionCountsFlat = {}
for pre in transitionCounts:
    for post in transitionCounts[pre]:
        transition = "{}->{}".format(pre, post)
        
        transitionCountsFlat[transition] = transitionCounts[pre][post]


sortedTransitions = sorted(transitionCountsFlat.items(), key=lambda x: x[i], reverse=True)



with open(sessionDir+"/transition_counts.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["PRE", "POST", "COUNT", "CONDITIONAL_PROBABILITY"])
    
    for transition, count in sortedTransitions:
        pre, post = transition.split("->")
        
        print(transition, count, transitionProbs[pre][post])
        writer.writerow([pre, post, count, transitionProbs[pre][post]])


