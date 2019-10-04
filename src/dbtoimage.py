'''
Created on Sep 2, 2019

@author: robovie


save camera information images for AMT

'''

import csv
import os
#import pandas as pd
#from pandas.plotting import table 
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
from plotly.offline import iplot
from IPython.display import Image


import tools


#plotly.io.orca.config.executable = "C:/Users/robovie/Anaconda3/Lib/site-packages/orca"


dataDirectory = tools.dataDir+"2019-09-13_11-52-56_advancedSimulator8" # many possible sentences for customer actions (from h-h dataset), all attributes change
numTrainDbs = 10


sessionDir = tools.create_session_dir("dbtoimage")


def read_database_file(filename):
    database = []
    
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            
            #for key in row:
            #    row[key] = row[key].lower()
            #    
            #    row[key] = nltk.word_tokenize(row[key])
            
            database.append(row)
    
    return database, fieldnames



#
# load the databases
#
filenames = os.listdir(dataDirectory)
filenames.sort()

databaseFilenamesAll = [dataDirectory+"/"+fn for fn in filenames if "simulated" not in fn]

databaseFilenames = databaseFilenamesAll[:numTrainDbs+1]


databases = []
databaseIds = []
dbFieldnames = None # these should be the same for all DBs

for dbFn in databaseFilenames:
    
    db, dbFieldnames = read_database_file(dbFn)
    #db = pd.read_csv(dbFn)
    
    databaseIds.append(dbFn.split("_")[-1].split(".")[0])
    
    databases.append(db)

numDatabases = len(databases)


#
# create and save the images
#
col1 = [f.upper().replace("_", " ") for f in dbFieldnames[1:]]
headerColor='royalblue'
rowOddColor = "white"
rowEvenColor = "#D5E1FE"

cameras = ["CAMERA_1", "CAMERA_2", "CAMERA_3"]

for i in range(numDatabases):
    dbId = databaseIds[i]
    db = databases[i]
    
    for j in range(len(cameras)): 
        #camData = db[db.camera_ID==cameras[j]]
        
        col2 = list(db[j].values())
        cam = col2[0]
        col2 = col2[1:]
        
        values = [col1, col2]
        
        table=[go.Table(columnorder = [1,2],
                        columnwidth = [25,70],
                        #columnwidth = [40,60], # for sample
                        
                        
                        header=dict(values=['<b>KEY</b>','<b>VALUE</b>'],
                                    line_color='black',
                                    fill_color=headerColor,
                                    align=['center','center'],
                                    font=dict(color='white', size=16)),
                        
                        cells=dict(values=values,
                                   line_color='black',
                                   fill_color = [[rowOddColor, rowEvenColor,
                                                  rowOddColor, rowEvenColor,
                                                  rowOddColor, rowEvenColor,
                                                  rowOddColor, rowEvenColor,
                                                  rowOddColor, rowEvenColor,
                                                  rowOddColor, rowEvenColor,
                                                  rowOddColor, rowEvenColor]*2],
                                   align=['left', 'left'],
                                   font_size=16,
                                   font_color="black",
                                   height=30))]
        
        
        
        fig = go.FigureWidget(data=table,
                              layout={'margin': {'l': 5, 'r': 5, 't': 50, 'b': 5}
                                      }
                              )
        
        fig.update_layout(title=go.layout.Title(text="<b>DATABASE_{} : {}</b>".format(dbId[-2:], cam),
                                                #text="<b>DATABASE_example : CAMERA_example</b>",
                                                xref="paper",
                                                xanchor="center",
                                                x=0.5)
                                                )
        
        
        """
        fig.update_layout(autosize=True,
                          #width=500,
                          #height=500,
                          margin=go.layout.Margin(l=5,
                                                  r=5,
                                                  b=5,
                                                  t=5,
                                                  pad=5),
                          paper_bgcolor="LightSteelBlue")
        """
        
        #fig.show()
        
        
        
        
        fig.write_image("{}/DATABASE_{}-{}.png".format(sessionDir, dbId[-2:], cam),
                        #"{}/DATABASE_example-CAMERA_example.png".format(sessionDir),
                        width=900, # 500 for sample
                        height=515,
                        scale=2
                        )
        
        
        #iplot(fig, image='svg', filename="{}/{}.svg".format(sessionDir, dbId), image_width=1280, image_height=1280)
        
        #img_bytes = fig.to_image(format="png", width=600, height=350, scale=2)
        #Image(img_bytes)
        
        
        
        """
        ax = plt.subplot(111, frame_on=False) # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        
        table(ax, camData)  # where df is your data frame
        plt.show()
        plt.savefig("{}/{}.png".format(sessionDir, dbId))
        """
        #break
    #break
    
    




