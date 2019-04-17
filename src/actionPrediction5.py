'''
Created on Oct 30, 2018

@author: malcolm



Do training and testing on the simulated data with a dynamic memory network.

'''

import tensorflow as tf

import numpy as np
import argparse
import time
import editdistance


import tools
from copynet.copynetoriginal import copynet



#sessionDir = tools.create_session_dir("actionPrediction4_dbl")



def normalized_edit_distance(s1, s2):
    
    return editdistance.eval(s1, s2) / float(max(len(s1), len(s2)))



def vectorize_sentences(sentences, charToIndex, maxSentLen):
    
    maxSentLen += 1 # for the EOS char
    
    sentVecs = []
    sentCharIndexLists = []
    
    for i in range(len(sentences)):
        
        sentVec = np.zeros(shape=(maxSentLen, len(charToIndex)))
        sentCharIndexList = []
        
        for j in range(maxSentLen):
            
            if j < len(sentences[i]):
                sentVec[j, charToIndex[sentences[i][j]]] = 1.0
                sentCharIndexList.append(charToIndex[sentences[i][j]])
            else:
                sentVec[j, charToIndex[" "]] = 1.0 # pad the end of sentences with spaces
                sentCharIndexList.append(charToIndex[" "])
        
        
        sentVec[-1, charToIndex["<EOS>"]] = 1
        sentCharIndexList.append(charToIndex["<EOS>"])
        
        sentVecs.append(sentVec)
        sentCharIndexLists.append(sentCharIndexList)
    
    
    return sentVecs, sentCharIndexLists



def unvectorize_sentences(sentCharIndexLists, indexToChar):
    
    sentences = []
    
    for i in range(sentCharIndexLists.shape[0]):
        
        sent = ""
        
        for j in range(sentCharIndexLists.shape[1]):
            
            sent += indexToChar[sentCharIndexLists[i,j]]
        
        sentences.append(sent)
        
    return sentences






#
# simulate some interactions
#
databases = []
questions = []
outputs = []


database = ["D | Sony | LOCATION | DISPLAY_1.",
            "D | Sony | PRICE | $800.",
            "D | Nikon | LOCATION | DISPLAY_2.",
            "D | Nikon | PRICE | $90."]

# example 1
databases.append(database)
questions.append("C | DISPLAY_1 | How much is this one?")
outputs.append("S | DISPLAY_1 | This one is $800.")


# example 2
databases.append(database)
questions.append("C | DISPLAY_2 | How much is this one?")
outputs.append("S | DISPLAY_2 | This one is $90.")



database = ["D | Sony | LOCATION | DISPLAY_1.",
            "D | Sony | PRICE | $500.",
            "D | Nikon | LOCATION | DISPLAY_2.",
            "D | Nikon | PRICE | $60."]

# example 3
databases.append(database)
questions.append("C | DISPLAY_1 | How much is this one?")
outputs.append("S | DISPLAY_1 | This one is $500.")


# example 4
databases.append(database)
questions.append("C | DISPLAY_2 | How much is this one?")
outputs.append("S | DISPLAY_2 | This one is $60.")


#
#
#
database = ["D | Sony | LOCATION | DISPLAY_1.",
            "D | Sony | PRICE | $600.",
            "D | Nikon | LOCATION | DISPLAY_2.",
            "D | Nikon | PRICE | $50."]

# example 5
databases.append(database)
questions.append("C | DISPLAY_1 | How much is this one?")
outputs.append("S | DISPLAY_1 | This one is $600.")


# example 6
databases.append(database)
questions.append("C | DISPLAY_2 | How much is this one?")
outputs.append("S | DISPLAY_2 | This one is $50.")


database = ["D | Sony | LOCATION | DISPLAY_1.",
            "D | Sony | PRICE | $700.",
            "D | Nikon | LOCATION | DISPLAY_2.",
            "D | Nikon | PRICE | $80."]

# example 7
databases.append(database)
questions.append("C | DISPLAY_1 | How much is this one?")
outputs.append("S | DISPLAY_1 | This one is $700.")


# example 8
databases.append(database)
questions.append("C | DISPLAY_2 | How much is this one?")
outputs.append("S | DISPLAY_2 | This one is $80.")


numExamples = len(outputs)



uniqueChars = []

for db in databases:
    for entry in db:
        for c in entry:
            if c not in uniqueChars:
                uniqueChars.append(c)

for q in questions:
    for c in q:
        if c not in uniqueChars:
            uniqueChars.append(c)

for o in outputs:
    for c in o:
        if c not in uniqueChars:
            uniqueChars.append(c)



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


charToIndex["<GO>"] = len(uniqueChars)
indexToChar[charToIndex["<GO>"]] = "<GO>"

charToIndex["<EOS>"] = len(uniqueChars)
indexToChar[charToIndex["<EOS>"]] = "<EOS>"

numUniqueChars = len(uniqueChars)



#
# vectorize the customer inputs as lists of one-hot char vectors
#
maxCustInputLen = max([len(q) for q in questions])
custInputVectors, custInputOneHotIndexLists = vectorize_sentences(questions, charToIndex, maxCustInputLen)


#
# vectorize the database entries
#
databaseSizes = [len(db) for db in databases]
maxDatabaseSize = max(databaseSizes)

databaseEntryLens = []
for db in databases:
    for entry in db:
        databaseEntryLens.append(len(entry))
maxDatabaseEntryLen = max(databaseEntryLens)

databaseVectors = []
databaseOneHotIndexLists = []

for db in databases:
    dbv, dbohil = vectorize_sentences(db, charToIndex, maxDatabaseEntryLen)
    
    databaseVectors.append(dbv)
    databaseOneHotIndexLists.append(dbohil)


#
# vectorize the outputs as lists of one-hot char vectors
#
maxShkpOutputLen = max([len(o) for o in outputs])
shkpOutputVectors, shkpOutputOneHotIndexLists = vectorize_sentences(outputs, charToIndex, maxShkpOutputLen)


#
# setup the model
#
embedding_size = 20
batch_size = 4

#
# inputs and outputs
#
cust_inputs = tf.placeholder(tf.int32, [batch_size, maxCustInputLen], name='cust_inputs')
database_entries = tf.placeholder(tf.int32, [batch_size, maxDatabaseSize, maxDatabaseEntryLen], name='database_entries')
shkp_output_targets = tf.placeholder(tf.int32, [batch_size, maxShkpOutputLen], name='shkp_output_targets') 


#
# customer input encoder
#
cust_input_char_embeddings = tf.contrib.layers.embed_sequence(cust_inputs, 
                                                              vocab_size=numUniqueChars, 
                                                              embed_dim=embedding_size)

cell = tf.contrib.rnn.GRUCell(embedding_size, name="cust_input_enc_cell")

cust_input_encoder_outputs, cust_input_encoder_state = tf.nn.dynamic_rnn(cell, cust_input_char_embeddings, dtype=tf.float32)


#
# database entry encoder
#

# reshape the databases
flatted_database_entries = tf.reshape(database_entries, (batch_size*maxDatabaseSize, maxDatabaseEntryLen))

flatted_database_entry_char_embeddings = tf.contrib.layers.embed_sequence(flatted_database_entries, 
                                                                  vocab_size=numUniqueChars, 
                                                                  embed_dim=embedding_size)

cell = tf.contrib.rnn.GRUCell(embedding_size, name="db_enc_cell")

flatted_database_entry_encoder_outputs, flatted_database_entry_encoder_state = tf.nn.dynamic_rnn(cell, flatted_database_entry_char_embeddings, dtype=tf.float32)

database_entry_encoder_outputs = tf.reshape(flatted_database_entry_encoder_outputs, (batch_size, maxDatabaseSize, maxDatabaseEntryLen, embedding_size))
database_entry_encoder_state = tf.reshape(flatted_database_entry_encoder_state, (batch_size, maxDatabaseSize, embedding_size))


#
# database and customer input match calculation (as in a Memory Network)
#
num_hops = 1

H = tf.Variable(tf.random_normal_initializer([embedding_size, embedding_size], stddev=0.1), name="H")

u_0 = cust_input_encoder_state
u = [u_0]


for _ in range(num_hops):
    
    # dot product of customer input embedding and database entries
    
    # hack to get around no reduce_dot
    u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
    dotted = tf.reduce_sum(database_entry_encoder_state * u_temp, 2)
    
    
    # calculate match probabilities between customer input and each database entry
    probs = tf.nn.softmax(dotted)
    
    
    probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1]) # ?
    c_temp = tf.transpose(database_entry_encoder_state, [0, 2, 1]) # ?
    
    # get the weighted database representation
    o_k = tf.reduce_sum(c_temp * probs_temp, 2)
    
    # combine the output from the previous hop with the current weighted database representation
    # it uses the customer input representation for hop 0
    u_k = tf.matmul(u[-1], H) + o_k
    
    #if nonlin:
    #    u_k = nonlin(u_k)
    
    u.append(u_k)
    
    #
    # how to do copynetoriginal copy from only the most important database entries???
    #
    
    
    
    
#
# generate the output sentence
#
target_vocab_size = len(target_vocab_to_int)
dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

#cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(rnn_size) for _ in range(num_layers)]) # , state_is_tuple=False
cell = tf.nn.rnn_cell.GRUCell(embedding_size)
copynet_cell = copynet.CopyNetWrapper(cell=cell,
                                      encoder_states=encoder_outputs, # GRU state is the same as its output
                                      encoder_input_ids=input_data,
                                      vocab_size=target_vocab_size)


with tf.variable_scope("decode"):
    output_layer = tf.layers.Dense(target_vocab_size)
    
    decoder_initial_state = copynet_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state) # cell_state=?
    
    # for only input layer
    helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, target_sequence_length)
    
    decoder = tf.contrib.seq2seq.BasicDecoder(copynet_cell,
                                              helper,
                                              decoder_initial_state,
                                              output_layer)

    # unrolling the decoder layer
    train_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                           impute_finished=True, 
                                                           maximum_iterations=max_target_sequence_length)
    

with tf.variable_scope("decode", reuse=True):
    
    decoder_initial_state = copynet_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state) # cell_state=?
    
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, 
                                                      tf.fill([batch_size], target_vocab_to_int['<GO>']), 
                                                      target_vocab_to_int['<EOS>'])
    
    decoder = tf.contrib.seq2seq.BasicDecoder(copynet_cell, 
                                              helper, 
                                              decoder_initial_state, 
                                              output_layer)
    
    infer_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                           impute_finished=True, 
                                                           maximum_iterations=max_target_sequence_length)







