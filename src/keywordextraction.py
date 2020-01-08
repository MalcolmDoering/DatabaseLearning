'''
Created on Nov 15, 2016

@author: MalcolmD
'''


#from __future__ import print_function
import json
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions

import ast
import csv
import os
import traceback

import tools


sessionDir = sessionDir = tools.create_session_dir("keywordextraction")

cameras = ["CAMERA_1", "CAMERA_2", "CAMERA_3"]

oldDataDirectory = tools.dataDir+"2019-12-05_14-58-11_advancedSimulator9"
dataDirectory = tools.dataDir+"2020-01-08_advancedSimulator9"

useSymbols = False


# NLU example
""""
# from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# Authentication via IAM
# authenticator = IAMAuthenticator('your_api_key')
# service = NaturalLanguageUnderstandingV1(
#     version='2018-03-16',
#     authenticator=authenticator)
# service.set_service_url('https://gateway.watsonplatform.net/natural-language-understanding/api')

# Authentication via external config like VCAP_SERVICES
service = NaturalLanguageUnderstandingV1(
    version='2018-03-16')
service.set_service_url('https://gateway.watsonplatform.net/natural-language-understanding/api')

response = service.analyze(
    text='Bruce Banner is the Hulk and Bruce Wayne is BATMAN! '
    'Superman fears not Banner, but Wayne.',
    features=Features(entities=EntitiesOptions(),
                      keywords=KeywordsOptions())).get_result()

print(json.dumps(response, indent=2))
"""


#
# load the old keywords
#
if useSymbols:
    keywordsDir = oldDataDirectory+"/shopkeeper_keywords_with_symbols/"
    uniqueUttFilename = "unique_shopkeeper_speech_with_symbols.csv"
else:
    keywordsDir = oldDataDirectory+"/shopkeeper_keywords_without_symbols/"
    uniqueUttFilename = "unique_shopkeeper_speech.csv"


# read the csv
uniqueShkpUtts_old = []
jsonKeywords_old = []

with open(keywordsDir+uniqueUttFilename) as csvfile:
    reader = csv.DictReader(csvfile)
    
    for row in reader:
        if useSymbols:
            uniqueShkpUtts_old.append(row["SHOPKEEPER_SPEECH_WITH_SYMBOLS"]) 
        else:
            uniqueShkpUtts_old.append(row["SHOPKEEPER_SPEECH"])
        
        
        with open(keywordsDir+str(row["UNIQUE_SPEECH_ID"])+"_keywords.txt") as f:
            contents = f.read()
            jsonKeywords_old.append(json.loads(contents))
        
        


#
# load the new unique shopkeeper utterances
#
filenames = os.listdir(dataDirectory)
filenames.sort()
interactionFilenames = [dataDirectory+"/"+fn for fn in filenames if "withsymbols" in fn]


uniqueShkpUtts = []

for fn in interactionFilenames:
    with open(fn) as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            if useSymbols:
                if row["SHOPKEEPER_SPEECH_WITH_SYMBOLS"] not in uniqueShkpUtts:
                    uniqueShkpUtts.append(row["SHOPKEEPER_SPEECH_WITH_SYMBOLS"])
            
            else:
                if row["SHOPKEEPER_SPEECH"] not in uniqueShkpUtts:
                    uniqueShkpUtts.append(row["SHOPKEEPER_SPEECH"])
            

#
# do the keyword extraction
#

# for doering@i.kyoto-u.ac.jp
iam_apikey = "cfrkW4LG3jGgs2Rl7vM4I9sbaWVBej9pNtWD8LwS1BVG"
url = "https://api.jp-tok.natural-language-understanding.watson.cloud.ibm.com/instances/88666344-cdec-4bff-85a1-2ff4c99e045d/v1/analyze?version=2019-07-12"

# for 
#iam_apikey = "wCnKsSecZzo2fGx1Lo_Dw2Q8-NzSm-AgsWS-jEQO_78r"
#url = "https://gateway.watsonplatform.net/natural-language-understanding/api"

service = NaturalLanguageUnderstandingV1(version='2018-03-16',
                                         url=url,
                                         iam_apikey=iam_apikey)

features = Features(entities=EntitiesOptions(), keywords=KeywordsOptions())

responses = []

for i in range(len(uniqueShkpUtts)):
    
    # first check if we already have this utt's keywords in the old set
    try:
        uniqueSpeechId_old = uniqueShkpUtts_old.index(uniqueShkpUtts[i])
        
        # found
        result = jsonKeywords_old[uniqueSpeechId_old]
        
    except:
        # not found
        try:
            print("calling watson api for unique utt id")
            response = service.analyze(text=uniqueShkpUtts[i], features=features)
            result = response.get_result()
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            print(e)
            result = "NA"
    
    
    responses.append(result)
    
    
    # save the keyword info in a separate file
    f = open(sessionDir+"/{}_keywords.txt".format(i), "w")
    f.write(json.dumps(responses[i], indent=2))
    f.close()
    
    print("completed", i+1, "of", len(uniqueShkpUtts))

#
# save the results
#
if useSymbols:
    fieldnames = ["UNIQUE_SPEECH_ID", "SHOPKEEPER_SPEECH_WITH_SYMBOLS"]
else:
    fieldnames = ["UNIQUE_SPEECH_ID", "SHOPKEEPER_SPEECH"]


if useSymbols:
    newUniqueUttFilename = "unique_shopkeeper_speech_with_symbols.csv"
else:
    newUniqueUttFilename = "unique_shopkeeper_speech.csv"

    
with open(sessionDir+"/"+newUniqueUttFilename, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for i in range(len(uniqueShkpUtts)):
        
        row = {"UNIQUE_SPEECH_ID": i}
        
        if useSymbols:
            row["SHOPKEEPER_SPEECH_WITH_SYMBOLS"] = uniqueShkpUtts[i]
        else:
            row["SHOPKEEPER_SPEECH"] = uniqueShkpUtts[i]
        
        writer.writerow(row)
        
        
        
        
        
        
        