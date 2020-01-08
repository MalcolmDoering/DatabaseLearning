'''
Created on Oct 3, 2019

@author: robovie

In the crowdsourced shopkeeper utts:
- replace 'dollars' with '$' before the number

'''

import csv
import os
import random
import copy
import string
import sys


import tools



shkpUttFilename = tools.dataDir + "2019-11-11_13-54-06_crowdsourcing_results_all_mod_preprocc.txt"
newShkpUttFilename = shkpUttFilename.split("/")[-1].split(".")[0] + ".csv"


sessionDir = tools.create_session_dir("preprocessshkputts")


#
# read in the crowdsourced utterances
#
print("loading the crowdsourced utterances...")

resultHits = []

with open(shkpUttFilename) as csvfile:
    reader = csv.DictReader(csvfile, delimiter="\t")
    resultFieldnames = reader.fieldnames
    
    for row in reader:
        resultHits.append(row)


#
# do the pre processing
#
for i in range(len(resultHits)):
    
    # replace 'dollars' with '$' before the number
    splitUtt = resultHits[i]["Answer.utterance"].split(" ")
    
    
    # "dollars"
    searchStartIndex = 0
    while True:
    
        try:
            dollarIndex = splitUtt.index("dollars", searchStartIndex)
            
            if tools.is_number(splitUtt[dollarIndex-1].replace(",", "")):
                splitUtt[dollarIndex-1] = "$" + splitUtt[dollarIndex-1]
                splitUtt.pop(dollarIndex)
                    
            elif splitUtt[dollarIndex-1][0] == "$":
                # sometimes they write a dollar sign and the word 'dollars'...
                splitUtt.pop(dollarIndex)
        
            searchStartIndex = dollarIndex + 1
            
        except Exception as e:
            break
    
    
    # "dollars."
    searchStartIndex = 0
    while True:
    
        try:
            dollarIndex = splitUtt.index("dollars.", searchStartIndex)
            
            if tools.is_number(splitUtt[dollarIndex-1].replace(",", "")):
                splitUtt[dollarIndex-1] = "$" + splitUtt[dollarIndex-1] + "."
                splitUtt.pop(dollarIndex)
                    
            elif splitUtt[dollarIndex-1][0] == "$":
                # sometimes they write a dollar sign and the word 'dollars'...
                splitUtt[dollarIndex-1] = splitUtt[dollarIndex-1] + "."
                splitUtt.pop(dollarIndex)
        
            searchStartIndex = dollarIndex + 1
            
        except Exception as e:
            break
    
    
    
    
        
    resultHits[i]["Answer.utterance"] = " ".join(splitUtt)
    

#
# save results
#
with open(sessionDir+"/"+newShkpUttFilename, "w", newline="") as csvfile:
        
        writer = csv.DictWriter(csvfile, fieldnames=resultFieldnames)
        
        writer.writeheader()
        
        for row in resultHits:
            writer.writerow(row)



