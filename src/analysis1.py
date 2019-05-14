'''
Created on Apr 26, 2019

@author: robovie
'''



from collections import OrderedDict
import csv


import tools



sessionDir = tools.create_session_dir("analysis1_dbl")


def process_log_file(date, seed, sessionDir):
    
    f = open("C:/Users/robovie/Dropbox/ATR/2018 database learning/analyses/{}_logs_3_8/{}_DBLearner_actionPrediction11_output_{}.txt".format(date, date, seed))
    lines = f.readlines()
    f.close()
    
    
    
    epochToRow = OrderedDict()
    
    for line in lines:
        
        if "train cost" in line:
            splitLine = line.split()
            
            epoch = int(splitLine[0])
            trainCost = splitLine[3]
            
            epochToRow[epoch] = [trainCost]
        
        if "test cost" in line:
            splitLine = line.split()
            
            testCost = splitLine[2]
            
            epochToRow[epoch].append(testCost)
    
    
    with open(sessionDir+"/log_{}.csv".format(seed), "w", newline="") as csvfile:
                
                writer = csv.writer(csvfile)
                
                writer.writerow(["Epoch", "Train Cost ({})".format(seed), "Test Cost ({})".format(seed)])
                
                for key, val in epochToRow.items():
                    writer.writerow([key] + val)
    




if __name__ == '__main__':
    
    
    date = "20190426"
    
    for seed in range(8, 16):
        process_log_file(date, seed, sessionDir)
            
    
    
            
    
    
    