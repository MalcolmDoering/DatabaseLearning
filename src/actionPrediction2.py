#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 14:46:19 2018

@author: malcolm

implement a simplified version of the architecture with a memory network


replace the output candidate selection with a character sequence model


"""



import utterancevectorizer



#
# simulate some interactions
#
databases = []
inputs = []
outputs = []


database = ["D | Sony | LOCATION | DISPLAY_1",
            "D | Sony | PRICE | $550",
            "D | Nikon | LOCATION | DISPLAY_2",
            "D | Nikon | PRICE | $68"]

# example 1
databases.append(database)
inputs.append("C | DISPLAY_1 | How much is this one?")
outputs.append("S | DISPLAY_1 | This one is $550.")


# example 2
databases.append(database)
inputs.append("C | DISPLAY_2 | How much is this one?")
outputs.append("S | DISPLAY_2 | This one is $68.")



database = ["D | Sony | LOCATION | DISPLAY_1",
            "D | Sony | PRICE | $500",
            "D | Nikon | LOCATION | DISPLAY_2",
            "D | Nikon | PRICE | $60"]

# example 3
databases.append(database)
inputs.append("C | DISPLAY_1 | How much is this one?")
outputs.append("S | DISPLAY_1 | This one is $500.")


# example 4
databases.append(database)
inputs.append("C | DISPLAY_2 | How much is this one?")
outputs.append("S | DISPLAY_2 | This one is $60.")




#
# vectorize the database contents and inputs as BoWs
# for now, just create a single vectorizer
#

strings = []

for db in databases:
    for entry in db:
        strings.append(entry)

for i in inputs:
    strings. append(i)


stringVectorizer = utterancevectorizer.UtteranceVectorizer(strings,
                                                           minCount=2,
                                                           keywordWeight=1.0,
                                                           keywordSet=["D"],
                                                           unigramsAndKeywordsOnly=True,
                                                           tfidf=False,
                                                           useStopwords=False,
                                                           lsa=False)

encodedDatabases = []

for db in databases:
    encodedDb = []
    
    for entry in db:
        encodedEntry = stringVectorizer.get_utterance_vector(entry, unigramOnly=True)
        encodedDb.append(encodedEntry)
    
    encodedDatabases.append(encodedDb)


encodedInputs = []

for i in inputs:
    encodedI = stringVectorizer.get_utterance_vector(i, unigramOnly=True)
    encodedInputs.append(encodedI)



#
# assign each unique output an ID
#
outputToId = {}
idToOutput = {}

outputIds = []


for o in outputs:
    if o not in outputToId:
        outputToId[o] = len(outputToId)
        idToOutput[outputToId[o]] = o
    
    outputIds.append(outputToId[o])



#
# setup the network
#

model = MemN2NDialog(batch_size=4, 
                     self.vocab_size, 
                     candidates_size=4, 
                     self.sentence_size, 
                     self.embedding_size, 
                     self.candidates_vec,
                     hops=1)




#
# train the network
#





#
# test the network
#












