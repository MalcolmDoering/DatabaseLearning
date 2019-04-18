'''
Created on Feb 20, 2019

@author: malcolm

a two network system where 
the first network predicts the action cluster ID
the second network modifies the typical utterance based on DB contents

'''


import tensorflow as tf
from sklearn.metrics import accuracy_score

import charsequencevectorizer


tf.set_random_seed(0)


#
# simulate some interactions
#
databases = []
inputs = []
outputs = []


# DB 1
database = ["D | Sony | LOCATION | DISPLAY_1.",
            "D | Sony | PRICE | $89.",
            "D | Nikon | LOCATION | DISPLAY_2.",
            "D | Nikon | PRICE | $90."]

# example 1
databases.append(database)
inputs.append("C | DISPLAY_1 | How much is this one?")
outputs.append("S | DISPLAY_1 | This one is $89.")


# example 2
databases.append(database)
inputs.append("C | DISPLAY_2 | How much is this one?")
outputs.append("S | DISPLAY_2 | This one is $90.")


# DB 2
database = ["D | Sony | LOCATION | DISPLAY_1.",
            "D | Sony | PRICE | $56.",
            "D | Nikon | LOCATION | DISPLAY_2.",
            "D | Nikon | PRICE | $67."]

# example 3
databases.append(database)
inputs.append("C | DISPLAY_1 | How much is this one?")
outputs.append("S | DISPLAY_1 | This one is $56.")


# example 4
databases.append(database)
inputs.append("C | DISPLAY_2 | How much is this one?")
outputs.append("S | DISPLAY_2 | This one is $67.")


# DB 3
database = ["D | Sony | LOCATION | DISPLAY_1.",
            "D | Sony | PRICE | $78.",
            "D | Nikon | LOCATION | DISPLAY_2.",
            "D | Nikon | PRICE | $45."]

# example 5
databases.append(database)
inputs.append("C | DISPLAY_1 | How much is this one?")
outputs.append("S | DISPLAY_1 | This one is $78.")


# example 6
databases.append(database)
inputs.append("C | DISPLAY_2 | How much is this one?")
outputs.append("S | DISPLAY_2 | This one is $45.")


# DB 4
database = ["D | Sony | LOCATION | DISPLAY_1.",
            "D | Sony | PRICE | $12.",
            "D | Nikon | LOCATION | DISPLAY_2.",
            "D | Nikon | PRICE | $23."]

# example 5
databases.append(database)
inputs.append("C | DISPLAY_1 | How much is this one?")
outputs.append("S | DISPLAY_1 | This one is $12.")


# example 6
databases.append(database)
inputs.append("C | DISPLAY_2 | How much is this one?")
outputs.append("S | DISPLAY_2 | This one is $23.")



allStrings = inputs + outputs

for db in databases:
    allStrings += db


#
# vectorize
# actually, just turn the strings into lists of char indices
#
charsequencevectorizer.build_vocab(allStrings)

dbVecs, dbVecLen = charsequencevectorizer.vectorize_db(databases)
inputVecs, inputVecLens = charsequencevectorizer.vectorize_char_sequences(inputs)
outputVecs, outputVecLens = charsequencevectorizer.vectorize_char_sequences(outputs)


# assign each output string an ID
outputToIndex = {}
indexToOutput = {}

outputClasses = []

for o in outputs:
    if o not in outputToIndex:
        outputToIndex[o] = len(outputToIndex)
        indexToOutput[outputToIndex[o]] = o
    
    outputClasses.append(outputToIndex[o])


#
# setup the first network, which is used for selecting a robot action cluster given a customer input
#
class Network1(object):
    """For now, this is a simple many to one sequence model."""
    
    def __init__(self, inputSeqLen, numOutputClasses, batchSize, vocabSize, embeddingSize):
        
        self.inputSeqLen = inputSeqLen
        self.numOutputClasses = numOutputClasses
        self.batchSize = batchSize
        self.vocabSize = vocabSize
        self.embeddingSize = embeddingSize
    
    
        #
        # build the model
        #
        self.ground_truth_outputs = tf.placeholder(tf.int32, [self.batchSize, ], name='ground_truth_outputs') 
        
        
        with tf.variable_scope("input_encoder"):
    
            self._inputs = tf.placeholder(tf.int32, [self.batchSize, self.inputSeqLen], name='customer_inputs')
            self._input_one_hot = tf.one_hot(self._inputs, self.vocabSize)
            
            # put each input char sequence through a GRU
            num_units = [self.embeddingSize]
            cells = [tf.nn.rnn_cell.GRUCell(num_units=n, kernel_initializer=tf.initializers.glorot_normal()) for n in num_units]
            stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)
            
            # use the last output of the GRU as the input encoding
            _, input_encoding = tf.nn.dynamic_rnn(stacked_rnn_cell, self._input_one_hot, dtype=tf.float32)
            self._input_encoding = input_encoding
        
        
        
        with tf.variable_scope("output_layers"):
            
            #out_1 = tf.layers.dense(self._input_encoding, self.embeddingSize, activation=tf.nn.relu)
            self.out_final = tf.layers.dense(self._input_encoding, self.numOutputClasses, activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling())
        
        
        with tf.variable_scope("loss"):
            
            self.groundTruthOutputsOneHot = tf.one_hot(self.ground_truth_outputs[:], self.numOutputClasses)
            self.loss = tf.losses.softmax_cross_entropy(self.groundTruthOutputsOneHot, self.out_final, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
        
        
        
        #
        # training
        #
        opt = tf.train.AdamOptimizer(learning_rate=1e-3)
        
        gradients = opt.compute_gradients(self.loss)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        
        
        self.train_op = opt.apply_gradients(capped_gradients, name="train_op")
        
        
        #
        # prediction
        #
        self.pred_op = tf.argmax(self.out_final, 1, name="predict_op")
        
        
        self._init_op = tf.initialize_all_variables()
        
        self.initialize()
    
    
    
    def initialize(self):
        self._sess = tf.Session()
        self._sess.run(self._init_op)
        
    
    def train(self, inputs, groundTruthOutputs):
        feedDict = {self._inputs: inputs, self.ground_truth_outputs: groundTruthOutputs}
        
        _, trainingLoss= self._sess.run([self.train_op, self.loss], feed_dict=feedDict)
        
        return trainingLoss
    
    
    def predict(self, inputs, groundTruthOutputs):
        feedDict = {self._inputs: inputs, self.ground_truth_outputs: groundTruthOutputs}
        
        preds, loss = self._sess.run([self.pred_op, self.loss], feed_dict=feedDict)
        
        return preds, loss
        


network1 = Network1(inputSeqLen=inputVecLens,
                    numOutputClasses=len(outputToIndex), 
                    batchSize=4, 
                    vocabSize=len(charsequencevectorizer.charToIndex),
                    embeddingSize=40)


#
# train network 1
#
numEpochs = 100

for e in range(numEpochs):
    
    trainLoss = network1.train(inputVecs[:4], outputClasses[:4])
    
    trainUttPreds, _ = network1.predict(inputVecs[:4], outputClasses[:4])
    testUttPreds, testLoss = network1.predict(inputVecs[4:], outputClasses[4:])
    
    trainAcc = accuracy_score(outputClasses[:4], trainUttPreds)
    testAcc = accuracy_score(outputClasses[4:], testUttPreds)
    
    print e, trainLoss, trainAcc, testLoss, testAcc
    
    
















