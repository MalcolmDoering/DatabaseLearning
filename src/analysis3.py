'''
Created on Jul 22, 2019

@author: robovie

annotate all_outputs file with metrics

'''


import csv
import numpy as np

import tools


eosChar = "#"

allOutputsFileName = "E:/eclipse-log/2019-07-18_13-34-09_actionPrediction14_dbl - GT addresses all data/rs0_ct0_at0/500_all_outputs.csv"
allOutputsFileName = "E:/eclipse-log/2019-07-21_14-44-40_actionPrediction14_dbl - SM addresses all data/rs0_ct0_at0/500_all_outputs.csv"

sessionDir = tools.create_session_dir("analysis3")


cameras = ["CAMERA_1", "CAMERA_2", "CAMERA_3"]
dbFieldnames = ["camera_ID", "camera_name", "camera_type", "color", "weight", "preset_modes", "effects", "price", "resolution", "optical_zoom", "settings", "autofocus_points", "sensor_size", "ISO", "long_exposure"]


def compute_db_substring_match(gtUtt, predUtt, dbSubstrRange):    
        
    subStringCharMatchCount = 0.0
    subStringCharTotalCount = 0.0
    
    for j in range(dbSubstrRange[0], dbSubstrRange[1]):
        
        try:
            if gtUtt[j] == predUtt[j]:
                subStringCharMatchCount += 1
        except IndexError:
            pass
        
        subStringCharTotalCount += 1
    
    subStringCharMatchAccuracy = subStringCharMatchCount / subStringCharTotalCount

    return subStringCharMatchAccuracy



#
# read the file
#
allOutputs = []

with open(allOutputsFileName) as csvfile:
    reader = csv.DictReader(csvfile)
    fieldnames = reader.fieldnames
    
    for row in reader:
        allOutputs.append(row)



#
# process the outputs
#
for i in range(len(allOutputs)):
    
    # remove everything after the EoS char from the predicted output shopkeeper utterances
    allOutputs[i]["PRED_SHOPKEEPER_SPEECH"] = allOutputs[i]["PRED_SHOPKEEPER_SPEECH"].split(eosChar, 1)[0]
    
    
    # start and end indices of DB substring
    if "~" in allOutputs[i]["SHOPKEEPER_SPEECH_DB_ENTRY_RANGE"]:
        shouldHaveDbSubstring = True
        dbSubstrRange = allOutputs[i]["SHOPKEEPER_SPEECH_DB_ENTRY_RANGE"].split("~")
        dbSubstrRange = [int(index) for index in dbSubstrRange]
    else:
        shouldHaveDbSubstring = False
    
    
    
    camAddressScores = []
    for cam in cameras:
        camAddressScores.append(allOutputs[i]["PRED_CAM_MATCH_SCORE_{}".format(cam)])
    
    attrAddressScores = []
    for attr in dbFieldnames:
        attrAddressScores.append(allOutputs[i]["PRED_ATT_MATCH_SCORE_{}".format(attr)])
    
    topCam = cameras[np.argmax(camAddressScores)]
    topAttr = dbFieldnames[np.argmax(attrAddressScores)]
    
    
    
    # metrics
    
    # shopkeeper utterance correct
    if allOutputs[i]["PRED_SHOPKEEPER_SPEECH"] == allOutputs[i]["SHOPKEEPER_SPEECH"]:
        allOutputs[i]["PRED_SHOPKEEPER_SPEECH_CORRECT"] = 1    
    else:
        allOutputs[i]["PRED_SHOPKEEPER_SPEECH_CORRECT"] = 0    
    
    # shopkeeper location correct
    if allOutputs[i]["PRED_OUTPUT_SHOPKEEPER_LOCATION"] == allOutputs[i]["OUTPUT_SHOPKEEPER_LOCATION"]:
        allOutputs[i]["OUTPUT_SHOPKEEPER_LOCATION_CORRECT"] = 1    
    else:
        allOutputs[i]["OUTPUT_SHOPKEEPER_LOCATION_CORRECT"] = 0
        
    # both shopkeeper utterance and shopkeeper location correct
    if allOutputs[i]["PRED_SHOPKEEPER_SPEECH_CORRECT"] and allOutputs[i]["OUTPUT_SHOPKEEPER_LOCATION_CORRECT"]:
        allOutputs[i]["PRED_SHOPKEEPER_SPEECH_AND_OUTPUT_LOCATION_CORRECT"] = 1
    else:
        allOutputs[i]["PRED_SHOPKEEPER_SPEECH_AND_OUTPUT_LOCATION_CORRECT"] = 0
    
    # db substring correct
    if shouldHaveDbSubstring:
        allOutputs[i]["PERCENT_DB_SUBSTRING_CORRECT"] = compute_db_substring_match(allOutputs[i]["SHOPKEEPER_SPEECH"], allOutputs[i]["PRED_SHOPKEEPER_SPEECH"], dbSubstrRange)
    else:
        allOutputs[i]["PERCENT_DB_SUBSTRING_CORRECT"] = ""
    
    # db camera address correct
    if allOutputs[i]["CURRENT_CAMERA_OF_CONVERSATION"] != "NONE":
        if topCam == allOutputs[i]["CURRENT_CAMERA_OF_CONVERSATION"]:
            allOutputs[i]["CAMERA_ADDRESS_CORRECT"] = 1
        else:
            allOutputs[i]["CAMERA_ADDRESS_CORRECT"] = 0
    else:
        allOutputs[i]["CAMERA_ADDRESS_CORRECT"] = ""
    
    # db attribute address correct
    if allOutputs[i]["SHOPKEEPER_TOPIC"] != "NONE" and allOutputs[i]["SHOPKEEPER_TOPIC"] != "":
        if topAttr == allOutputs[i]["SHOPKEEPER_TOPIC"]:
            allOutputs[i]["ATTRIBUTE_ADDRESS_CORRECT"] = 1
        else:
            allOutputs[i]["ATTRIBUTE_ADDRESS_CORRECT"] = 0
    else:
        allOutputs[i]["ATTRIBUTE_ADDRESS_CORRECT"] = ""
    
    
    # both db addresses correct
    if allOutputs[i]["CAMERA_ADDRESS_CORRECT"] != "" and allOutputs[i]["ATTRIBUTE_ADDRESS_CORRECT"] != "":
        if allOutputs[i]["CAMERA_ADDRESS_CORRECT"] and allOutputs[i]["ATTRIBUTE_ADDRESS_CORRECT"]:
            allOutputs[i]["BOTH_ADDRESSES_CORRECT"] = 1
        else:
            allOutputs[i]["BOTH_ADDRESSES_CORRECT"] = 0
    else:
        allOutputs[i]["BOTH_ADDRESSES_CORRECT"] = ""
    
    
    
    
    


# add new fieldnames
fieldnames.append("PRED_SHOPKEEPER_SPEECH_CORRECT")
fieldnames.append("OUTPUT_SHOPKEEPER_LOCATION_CORRECT")
fieldnames.append("PRED_SHOPKEEPER_SPEECH_AND_OUTPUT_LOCATION_CORRECT")
fieldnames.append("PERCENT_DB_SUBSTRING_CORRECT")
fieldnames.append("CAMERA_ADDRESS_CORRECT")
fieldnames.append("ATTRIBUTE_ADDRESS_CORRECT")
fieldnames.append("BOTH_ADDRESSES_CORRECT")
#fieldnames.append("")



#
# write the file
#
with open(sessionDir+"/all_outputs_annotated.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames)
    writer.writeheader()
    writer.writerows(allOutputs)


