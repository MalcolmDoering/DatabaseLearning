'''
Created on Oct 30, 2018

@author: malcolm



Do training and testing on the simulated data with a dynamic memory network.

'''

import numpy as np
import argparse
import time



def vectorize_candidates(candidates, charToIndex, maxSentLen):
    shape=(len(candidates), maxSentLen)
    C=[]
    
    for i, candidate in enumerate(candidates):
        lc=max(0, maxSentLen-len(candidate))
        C.append([charToIndex[w] if w in charToIndex else 0 for w in candidate] + [0] * lc)
        
    return C




def vectorize_data(data, word_idx, sentence_size, batch_size, candidates_size, max_memory_size):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    S = []
    Q = []
    A = []
    
    #data.sort(key=lambda x:len(x[0]),reverse=True)
    
    for i, (story, query, answer) in enumerate(data):
    
        if i%batch_size==0:
            memory_size=max(1,min(max_memory_size,len(story)))
        
        ss = []
        
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] if w in word_idx else 0 for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] if w in word_idx else 0 for w in query] + [0] * lq


        S.append(np.array(ss))
        Q.append(np.array(q))
        A.append(np.array(answer))
        
        
    return S, Q, A





#
# simulate some interactions
#
databases = []
inputs = []
outputs = []


database = ["D | Sony | LOCATION | DISPLAY_1.",
            "D | Sony | PRICE | $550.",
            "D | Nikon | LOCATION | DISPLAY_2.",
            "D | Nikon | PRICE | $68."]

# example 1
databases.append(database)
inputs.append("C | DISPLAY_1 | How much is this one?")
outputs.append("S | DISPLAY_1 | This one is $550.")


# example 2
databases.append(database)
inputs.append("C | DISPLAY_2 | How much is this one?")
outputs.append("S | DISPLAY_2 | This one is $68.")



database = ["D | Sony | LOCATION | DISPLAY_1.",
            "D | Sony | PRICE | $500.",
            "D | Nikon | LOCATION | DISPLAY_2.",
            "D | Nikon | PRICE | $60."]

# example 3
databases.append(database)
inputs.append("C | DISPLAY_1 | How much is this one?")
outputs.append("S | DISPLAY_1 | This one is $500.")


# example 4
databases.append(database)
inputs.append("C | DISPLAY_2 | How much is this one?")
outputs.append("S | DISPLAY_2 | This one is $60.")


numExamples = len(outputs)


inputBufLens = [len(db) for db in databases]


dbSentLens = []
for db in databases:
    for entry in db:
        dbSentLens.append(len(entry))

inputSentLens = []
for i in inputs:
    inputSentLens.append(len(i))

outputSentLens = []
for i in outputs:
    outputSentLens.append(len(i))


uniqueChars = []

for db in databases:
    for entry in db:
        for c in entry:
            if c not in uniqueChars:
                uniqueChars.append(c)


for i in inputs:
    for c in i:
        if c not in uniqueChars:
            uniqueChars.append(c)

for o in outputs:
    for c in o:
        if c not in uniqueChars:
            uniqueChars.append(c)


maxBufLen = max(inputBufLens)
maxSentLen = max(dbSentLens + inputSentLens + outputSentLens)


#
#
#
outputIds = [0, 1, 2, 3]


#
# char to index encoder
# make sure all chars and nums are included so that the network can generalize
#
alphanum = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

for i in alphanum:
    if i not in uniqueChars:
        uniqueChars.append(i)

charToIndex = {}
indexToChar = {}

for i in range(len(uniqueChars)):
    charToIndex[uniqueChars[i]] = i
    indexToChar[i] = uniqueChars[i]

numUniqueChars = len(uniqueChars)




outputCandidateVectors = vectorize_candidates(outputs, charToIndex, maxSentLen)

data = []

for i in range(numExamples):
    data.append((databases[i], inputs[i], outputIds[i]))

s, q, a = vectorize_data(data, charToIndex, maxSentLen, 4, 4, maxBufLen)


inputMasks = []
inputMasks.append(np.array([index for index, w in enumerate(databases) if w == '.'], dtype=np.int32)) 







