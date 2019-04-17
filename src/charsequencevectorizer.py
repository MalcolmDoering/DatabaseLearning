'''
Created on Feb 20, 2019

@author: malcolm
'''


import numpy as np


startChar = "*"
endChar = "#"
padChar = " "

charToIndex = {}
indexToChar = {}


def add_char_to_vocab(c, charToIndex, indexToChar):
    """returns False if the char is already in the vocab"""
    
    if c not in charToIndex:
        charToIndex[c] = len(charToIndex)
        indexToChar[charToIndex[c]] = c
        
        return True
    
    else:
        return False



def build_vocab(strings):
    
    add_to_vocab(strings)
    
    # pad char
    add_char_to_vocab(padChar, charToIndex, indexToChar)
    
    # start char
    if not add_char_to_vocab(startChar, charToIndex, indexToChar):
        print "Warning: start char is already in the vocab!"
    
    # end char
    if not add_char_to_vocab(endChar, charToIndex, indexToChar):
        print "Warning: end char is already in the vocab!"
    
    return charToIndex, indexToChar



def add_to_vocab(strings):
    for string in strings:
        for c in string:
            add_char_to_vocab(c, charToIndex, indexToChar)



def clear_vocab():
    charToIndex = {}
    indexToChar = {}



def find_char_sequence_max_len(strings):
    maxSeqLen = max([len(string) for string in strings])
    return maxSeqLen


def find_db_char_sequence_max_len(listOfListsOfStrings):
    """Find the max len of all strings in a DB. A 'DB' is a list of strings."""
    
    maxDbSeqLen = max([find_char_sequence_max_len(db) for db in listOfListsOfStrings])
    return maxDbSeqLen


def vectorize_char_sequences(strings, maxCharSeqLen=None):
    
    if maxCharSeqLen == None:
        maxCharSeqLen = find_char_sequence_max_len(strings)
    
    vecLen = maxCharSeqLen + 2 # need space for the start and stop chars
    
    vecs = []
    
    for string in strings:
        vec = [charToIndex[padChar]] * vecLen
        
        for i in range(len(string)):
            
            vec[i+1] = charToIndex[string[i]]
        
        vec[0] = charToIndex[startChar]
        vec[-1] = charToIndex[endChar]
        
        vecs.append(vec)
    
    return vecs, vecLen


def vectorize_db(databases, maxCharSeqLen=None):
    
    if maxCharSeqLen == None:
        maxCharSeqLen = find_db_char_sequence_max_len(databases)
    
    dbVecs = []
    
    for db in databases:
        
        vecs, vecLen = vectorize_char_sequences(db, maxCharSeqLen)
        dbVecs.append(vecs)
    
    return dbVecs, vecLen





