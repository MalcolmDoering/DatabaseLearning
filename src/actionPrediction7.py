'''
Created on Nov 15, 2018

@author: MalcolmD

copied from actionPrediction6, but separate the DB inputs before running through encoder
'''


import tensorflow as tf

import numpy as np
from six.moves import range
from datetime import datetime
from sklearn import metrics
import editdistance


import tools
from copynet import copynet


print "tensorflow version", tf.__version__


#sessionDir = tools.create_session_dir("actionPrediction4_dbl")

eosChar = "#"
goChar = "~"



class CustomNeuralNetwork(object):
    
    def __init__(self, inputSeqLen, dbSeqLen, outputSeqLen, batchSize, vocabSize, dbSize, embeddingSize, charToIndex):
        
        self.inputSeqLen = inputSeqLen
        self.dbSeqLen = dbSeqLen
        self.outputSeqLen = outputSeqLen
        self.batchSize = batchSize
        self.vocabSize = vocabSize
        self.dbSize = dbSize
        self.embeddingSize = embeddingSize
        self.charToIndex = charToIndex
        
        #
        # build the model
        #
        with tf.variable_scope("input_encoder"):
            # input encoder
            #with tf.name_scope("input encoder"):
            self._inputs = tf.placeholder(tf.int32, [self.batchSize, self.inputSeqLen, ], name='customer_inputs')
            self._input_one_hot = tf.one_hot(self._inputs, self.vocabSize)
            
            """
            # condense the char embedding with three layers
            reshaped = tf.reshape(self._input_one_hot, (self.batchSize*self.inputSeqLen, self.vocabSize))
            
            enc1 = tf.layers.dense(reshaped, self.embeddingSize, activation=tf.nn.relu)
            end2 = tf.layers.dense(enc1, self.embeddingSize, activation=tf.nn.relu)
            enc3 = tf.layers.dense(end2, self.embeddingSize, activation=tf.nn.relu)
            
            # run the input sequences through a GRU to get a single input encoding
            reshaped = tf.reshape(enc3, (self.batchSize, self.inputSeqLen, self.embeddingSize))
            """
            
            num_units = [self.vocabSize, self.embeddingSize, self.embeddingSize]
            cells = [tf.nn.rnn_cell.GRUCell(num_units=n) for n in num_units]
            stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)
            
            _, input_encoding = tf.nn.dynamic_rnn(stacked_rnn_cell, self._input_one_hot, dtype=tf.float32)
            self._input_encoding = input_encoding[-1]
        
        
        with tf.variable_scope("DB_encoder"):
            # DB encoder
            #with tf.name_scope("DB encoder"):
            self._db_entries = tf.placeholder(tf.int32, [self.batchSize, self.dbSize, self.dbSeqLen], name='DB_entries')
            self._db_entries_flattened = tf.reshape(self._db_entries, (self.batchSize, self.dbSize*self.dbSeqLen))
            
            self._db_entries_one_hot = tf.one_hot(self._db_entries, self.vocabSize)
            
            """
            # condense the char embedding with three layers
            reshaped = tf.reshape(self._db_entries_one_hot, (self.batchSize*self.dbSize*self.dbSeqLen, self.vocabSize))
            
            enc1 = tf.layers.dense(reshaped, self.embeddingSize, activation=tf.nn.relu)
            end2 = tf.layers.dense(enc1, self.embeddingSize, activation=tf.nn.relu)
            enc3 = tf.layers.dense(end2, self.embeddingSize, activation=tf.nn.relu)
            
            # run the DB entry sequences through a GRU to get a single encoding for each entry
            reshaped = tf.reshape(enc3, (self.batchSize*self.dbSize, self.dbSeqLen, self.embeddingSize))
            cell = tf.nn.rnn_cell.GRUCell(self.embeddingSize)
            self._db_entry_encodings, _ = tf.nn.dynamic_rnn(cell, reshaped, dtype=tf.float32)
            """
            
            reshaped = tf.reshape(self._db_entries_one_hot, (self.batchSize*self.dbSize, self.dbSeqLen, self.vocabSize))
            
            num_units = [self.vocabSize, self.embeddingSize, self.embeddingSize]
            cells = [tf.nn.rnn_cell.GRUCell(num_units=n) for n in num_units]
            stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)
            
            self._db_entry_encodings, _ = tf.nn.dynamic_rnn(stacked_rnn_cell, reshaped, dtype=tf.float32)
            
            self._db_entry_encodings_flattened = tf.reshape(self._db_entry_encodings, (self.batchSize, self.dbSize*self.dbSeqLen, self.embeddingSize))
        
        
        #
        # setup the decoder
        #
        self._ground_truth_outputs = tf.placeholder(tf.int32, [self.batchSize, self.outputSeqLen], name='true_robot_outputs') 
        #self._ground_truth_outputs_one_hot = tf.one_hot(self._ground_truth_outputs, self.vocabSize)
        
        self.copynet_cell = copynet.CopyNetWrapper(cell=tf.nn.rnn_cell.GRUCell(self.embeddingSize),
                                              encoder_states=self._db_entry_encodings_flattened,
                                              encoder_input_ids=self._db_entries_flattened,
                                              vocab_size=self.vocabSize)
            
            
        # this should actually take the output of an attention layer over the encoder outputs
        self.decoder_initial_state = self.copynet_cell.zero_state(self.batchSize, tf.float32).clone(cell_state=self._input_encoding) # what to use for the cell state?
        
        
        # append start char on to beginning of outputs so they can be used for teacher forcing - i.e. as inputs to the coipynet decoder
        after_slice = tf.strided_slice(self._ground_truth_outputs, [0, 0], [self.batchSize, -1], [1, 1]) # slice of the last char of each output sequence (is this necessary?)
        decoder_inputs = tf.concat( [tf.fill([self.batchSize, 1], charToIndex[goChar]), after_slice], 1) # concatenate on a go char onto the start of each output sequence
        
        self.decoder_inputs_one_hot = tf.one_hot(decoder_inputs, self.vocabSize)
        
        # rollout the decoder two times - once for use with teacher forcing (training) and once without (testing)
        # for training
        self._loss, self._train_predicted_output_sequences, self._train_copy_scores, self._train_gen_scores = self.build_decoder(teacherForcing=False, scopeName="decoder_train")
        
        # for testing
        self._test_loss, self._test_predicted_output_sequences, self._test_copy_scores, self._test_gen_scores = self.build_decoder(teacherForcing=False, scopeName="decoder_test")
        
        
        #
        # setup the training function
        #
        opt = tf.train.AdamOptimizer(learning_rate=1e-3)
        
        gradients = opt.compute_gradients(self._loss)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        
        self._train_op = opt.apply_gradients(capped_gradients, name="train_op")
        
        
        #
        # setup the prediction function
        #
        self._pred_op = tf.argmax(self._test_predicted_output_sequences, 2, name="predict_op")
        #self._pred_prob_op = tf.nn.softmax(predicted_output_sequences, axis=2, name="predict_prob_op")
        #self._pred_log_prob_op = tf.log(predict_proba_op, name="predict_log_proba_op")
        
        
        self._init_op = tf.initialize_all_variables()
        
        self.initialize()
    
    
    def build_decoder(self, teacherForcing=False, scopeName="decoder"):
        
        
        with tf.variable_scope(scopeName):
            
            loss = 0
            predicted_output_sequences = []
            copy_scores = []
            gen_scores = []
            
            
            state = self.decoder_initial_state
            output = self.decoder_inputs_one_hot[:, 0, :] # each output sequence must have a 'start' char appended to the beginning
    
        
            
            for i in range(self.outputSeqLen):
                
                #
                # TODO: don't use teacher forcing for prediction
                #
                
                if teacherForcing:
                    # if using teacher forcing
                    output, state, copy_score, gen_score = self.copynet_cell(self.decoder_inputs_one_hot[:, i, :], state)
                else:
                    # if not using teacher forcing
                    output, state, copy_score, gen_score = self.copynet_cell(output, state)
                
                
                predicted_output_sequences.append(tf.reshape(output, shape=(tf.shape(output)[0], 1, self.vocabSize)))
                copy_scores.append(tf.reshape(copy_score, shape=(tf.shape(copy_score)[0], 1, self.vocabSize)))
                gen_scores.append(tf.reshape(gen_score, shape=(tf.shape(gen_score)[0], 1, self.vocabSize)))
                
                # get the ground truth output
                ground_truth_output = tf.one_hot(self._ground_truth_outputs[:, i], self.vocabSize) # these are one-hot char encodings at timestep i
                
                # compute the loss                
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_output, logits=output)
                current_loss = tf.reduce_sum(cross_entropy)
                loss = loss + current_loss
            
            
            predicted_output_sequences = tf.concat(predicted_output_sequences, 1)
            copy_scores = tf.concat(copy_scores, 1)
            gen_scores = tf.concat(gen_scores, 1)
        
        return loss, predicted_output_sequences, copy_scores, gen_scores
    
    
    def initialize(self):
        self._sess = tf.Session()
        self._sess.run(self._init_op)
        
    
    def train(self, inputs, databases, groundTruthOutputs):
        feedDict = {self._inputs: inputs, self._db_entries: databases, self._ground_truth_outputs: groundTruthOutputs}
        
        trainingLoss, _ = self._sess.run([self._loss, self._train_op], feed_dict=feedDict)
        
        return trainingLoss
    
    
    def predict(self, inputs, databases, groundTruthOutputs):
        feedDict = {self._inputs: inputs, self._db_entries: databases, self._ground_truth_outputs: groundTruthOutputs}
        
        preds, copyScores, genScores = self._sess.run([self._pred_op, self._test_copy_scores, self._test_gen_scores], feed_dict=feedDict)
        
        return preds, copyScores, genScores        



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
        
        
        sentVec[-1, charToIndex[eosChar]] = 1
        sentCharIndexList.append(charToIndex[eosChar])
        
        sentVecs.append(sentVec)
        sentCharIndexLists.append(sentCharIndexList)
    
    
    return sentVecs, sentCharIndexLists



def vectorize_databases(databases, charToIndex, maxSentLen):
    dbVecs = []
    dbCharIndexLists = []
    
    for db in databases:
        dbv, dbcil = vectorize_sentences(db, charToIndex, maxSentLen)
        dbVecs.append(dbv)
        dbCharIndexLists.append(dbcil)
    
    return dbVecs, dbCharIndexLists



def unvectorize_sentences(sentCharIndexLists, indexToChar):
    
    sentences = []
    
    for i in range(sentCharIndexLists.shape[0]):
        
        sent = ""
        
        for j in range(sentCharIndexLists.shape[1]):
            
            sent += indexToChar[sentCharIndexLists[i,j]]
        
        sentences.append(sent)
        
    return sentences



def color_results(outputStrings, copyScores, genScores, charToIndex):
    
    coloredOutputStrings = []
    
    for i in range(len(outputStrings)):
        colOutStr = ""
        
        for j in range(len(outputStrings[i])):
            cs = copyScores[i,j,charToIndex[outputStrings[i][j]]]
            gs = genScores[i,j,charToIndex[outputStrings[i][j]]]
            
            # color the char if it was coppied from the input
            if cs > gs:
                colOutStr += "\x1b[36m" + outputStrings[i][j] + "\x1b[0m" # blue-green
                
            elif cs == gs:
                colOutStr += "\x1b[33m" + outputStrings[i][j] + "\x1b[0m" # yellow
            
            else:
                colOutStr += outputStrings[i][j]
        
        coloredOutputStrings.append(colOutStr)
    
    return coloredOutputStrings



if __name__ == "__main__":
    
    #
    # simulate some interactions
    #
    databases = []
    inputs = []
    outputs = []
    
    
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
    
    
    #
    #
    #
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
    
    
    #
    # calculate the data sizes
    #
    numExamples = len(outputs)
    dbSizes = [len(db) for db in databases]
    
    inputSentLens = [len(i) for i in inputs]
    
    outputSentLens = [len(o) for o in outputs]
    
    dbSentLens = []
    for db in databases:
        for entry in db:
            dbSentLens.append(len(entry))
    
    
    maxDbSize = max(dbSizes)
    maxInputSentLen = max(inputSentLens)
    maxOutputSentLen = max(outputSentLens)
    maxDbSentLen = max(dbSentLens)
    
    
    
    #
    # create the char vocab
    #
    uniqueChars = [eosChar, goChar]
    
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
    
    
    #
    # encode the data
    #
    inputOneHotVectors, inputIndexLists = vectorize_sentences(inputs, charToIndex, max(inputSentLens))
    dbOneHotVectors, dbIndexLists = vectorize_databases(databases, charToIndex, max(dbSentLens))
    outputOneHotVectors, outputIndexLists = vectorize_sentences(outputs, charToIndex, max(outputSentLens))
    
    
    #
    # setup the model
    #
    batchSize = 4
    embeddingSize = 35
    
    inputs = inputIndexLists
    databases = dbIndexLists
    groundTruthOutputs = outputIndexLists
    
    # add one to account for the eos char
    maxInputSeqLen = maxInputSentLen + 2
    maxOutputSeqLen = maxOutputSentLen + 2
    maxDbSeqLen = maxDbSentLen + 2
    
    
    #
    # split training and testing data
    #
    trainInputs = inputs[:4]
    testInputs = inputs[4:]
    
    trainDbs = databases[:4]
    testDbs = databases[4:]
    
    trainGroundTruth = groundTruthOutputs[:4]
    testGroundTruth = groundTruthOutputs[4:]
    
    
    # for computing accuracy
    trainGroundTruthFlat = []
    testGroundTruthFlat = []
    
    for i in range(numExamples):    
        groundTruthFlat = groundTruthOutputs[i]
        
        if i < 4:
            trainGroundTruthFlat += groundTruthFlat
        else:
            testGroundTruthFlat += groundTruthFlat
    
    
    #
    # setup the network
    #
    learner = CustomNeuralNetwork(inputSeqLen=maxInputSeqLen, 
                                  dbSeqLen=maxDbSeqLen, 
                                  outputSeqLen=maxOutputSeqLen, 
                                  batchSize=batchSize, 
                                  vocabSize=numUniqueChars, 
                                  dbSize=maxDbSize, 
                                  embeddingSize=embeddingSize,
                                  charToIndex=charToIndex)
    
    
    numEpochs = 10000
    
    for e in range(numEpochs):
        
        trainCost = learner.train(trainInputs, trainDbs, trainGroundTruth)
        
        
        if e % 100 == 0:
            
            #
            # compute accuracy, etc.
            #
            
            # TRAIN
            trainUttPreds, trainCopyScores, trainGenScores = learner.predict(trainInputs, trainDbs, trainGroundTruth)
            
            
            trainAcc = 0.0
            for i in range(len(trainUttPreds)):
                trainAcc = normalized_edit_distance(trainGroundTruth[i], trainUttPreds[i])
            trainAcc /= len(trainGroundTruth)
            
            #trainPredsFlat = np.array(trainUttPreds).flatten()
            #trainAcc = metrics.accuracy_score(trainPredsFlat, trainGroundTruthFlat)
            trainPredSents = unvectorize_sentences(trainUttPreds, indexToChar)
            trainPredSents = color_results(trainPredSents, trainCopyScores, trainGenScores, charToIndex)
            
            
            # TEST
            testUttPreds, testCopyScores, testGenScores = learner.predict(testInputs, testDbs, testGroundTruth)
            
            testAcc = 0.0
            for i in range(len(testUttPreds)):
                testAcc = normalized_edit_distance(testGroundTruth[i], testUttPreds[i])
            testAcc /= len(testGroundTruth)
            
            #testPredsFlat = np.array(testUttPreds).flatten()
            #testAcc = metrics.accuracy_score(testPredsFlat, testGroundTruthFlat)
            testPredSents = unvectorize_sentences(testUttPreds, indexToChar)
            testPredSents = color_results(testPredSents, testCopyScores, testGenScores, charToIndex)
            
            
            print "****************************************************************"
            print "TRAIN", e, round(trainCost, 3), round(trainAcc, 2)
            
            for i in range(len(trainPredSents)):
                print "TRUE:", outputs[i]
                print "PRED:", trainPredSents[i]
                print
            
            
            print "TEST", e, round(testAcc, 2)
            
            for i in range(len(testPredSents)):
                print "TRUE:", outputs[i+4]
                print "PRED:", testPredSents[i]
                print
        
            print "****************************************************************"









