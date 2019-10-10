'''
Created on Sep 26, 2019

@author: robovie
'''


import csv
import random
import copy
import os

import tools



databaseDir = tools.dataDir + "2019-09-18_13-15-13_advancedSimulator9"

hitCsvFile = "E:/Dropbox/ATR/2018 database learning/crowdsourcing/2019-09-13_14-24-58_shopkeeper_utterance_HITs - batch 00-02 results.csv"
modifiedHitCsvFile = hitCsvFile[:-4] + "_mod.csv"


def read_database_file(filename):
    database = {}
    
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            
            database[row["camera_ID"]] = {}
            
            for key in row:
                database[row["camera_ID"]][key] = row[key]
    
    return database, fieldnames






#
# read in the camera databases
#
filenames = os.listdir(databaseDir)
filenames.sort()


databaseFilenames = [databaseDir+"/"+fn for fn in filenames if "database" in fn and "simulated" not in fn]

databases = {}

for dbFn in databaseFilenames:
    db, dbFieldnames = read_database_file(dbFn)
    
    databaseId = dbFn.split("_")[-1].split(".")[0]
    
    databases[databaseId] = db
    


#
# read in the HIT csv
#
hits = []

with open(hitCsvFile) as csvfile:
    reader = csv.DictReader(csvfile)
    fieldnames = reader.fieldnames
    
    for row in reader:
        hits.append(row)



#
# add the fields
#
fieldnames.append("DATABASE_ID")
fieldnames.append("DATABASE_CONTENTS")


for i in range(len(hits)):
    
    # add the database ID
    dbImage = hits[i]["Input.DB_IMAGE"]

    #https://camerashopdatabases.s3-ap-northeast-1.amazonaws.com/DATABASE_04-CAMERA_3.png
    databaseID = "0-" + dbImage[-15:-13]
    
    hits[i]["DATABASE_ID"] = databaseID
    
    
    # add the GT database contents
    camera = hits[i]["Input.CURRENT_CAMERA_OF_CONVERSATION"]
    feature = hits[i]["Input.SHOPKEEPER_TOPIC"]
    
    if feature == "":
        feature = "camera_name"
    
    hits[i]["DATABASE_CONTENTS"] = databases[databaseID][camera][feature]
        

#
# write the modified csv
#
with open(modifiedHitCsvFile, "w", newline="") as csvfile:
    
    writer = csv.DictWriter(csvfile, fieldnames)
    writer.writeheader()
    
    for i in range(len(hits)):
        writer.writerow(hits[i])




