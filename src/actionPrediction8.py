'''
Created on Nov 15, 2018

@author: MalcolmD

copied from actionPrediction7, but only allow copynet to copy from the best matching DB entry.
I.e. combine copynet with memory network.
'''



import tensorflow as tf

import numpy as np
from six.moves import range
from datetime import datetime
from sklearn import metrics
import editdistance
import csv


import tools
from copynet import copynet


print "tensorflow version", tf.__version__


sessionDir = tools.create_session_dir("actionPrediction8_dbl")

eosChar = "#"
goChar = "~"

tf.reset_default_graph()
tf.set_random_seed(0)


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
            
            num_units = [self.embeddingSize, self.embeddingSize]
            #cells = [tf.nn.rnn_cell.LSTMCell(num_units=n, initializer=tf.initializers.glorot_normal()) for n in num_units]
            cells = [tf.nn.rnn_cell.GRUCell(num_units=n, kernel_initializer=tf.initializers.glorot_normal()) for n in num_units]
            
            stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)
            
            _, input_encoding = tf.nn.dynamic_rnn(stacked_rnn_cell, self._input_one_hot, dtype=tf.float32)
            
            # for single layer GRU
            self._input_encoding = input_encoding[-1] # TODO why is this here??? [-1] A: get output instead of candidate 
        
            # for two layer LSTM
            #self._input_encoding = tf.concat([input_encoding[0][-1], input_encoding[1][-1]], axis=1)
            
        
        
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
            
            
            """
            # one-direction GRU
            num_units = [self.embeddingSize]
            cells = [tf.nn.rnn_cell.GRUCell(num_units=n, kernel_initializer=tf.initializers.glorot_normal()) for n in num_units]
            stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)
            
            
            self._db_entry_encodings, database_entry_final_states = tf.nn.dynamic_rnn(stacked_rnn_cell, reshaped, dtype=tf.float32)
            self._db_entry_encodings_reformed = tf.reshape(self._db_entry_encodings, (self.batchSize, self.dbSize, self.dbSeqLen, self.embeddingSize))
            
            """
            # bi-directional GRU
            num_units = [self.embeddingSize, self.embeddingSize]
            
            #cells_fw = [tf.nn.rnn_cell.LSTMCell(num_units=n, initializer=tf.initializers.glorot_normal()) for n in num_units]
            #cells_bw = [tf.nn.rnn_cell.LSTMCell(num_units=n, initializer=tf.initializers.glorot_normal()) for n in num_units]
            
            cells_fw = [tf.nn.rnn_cell.GRUCell(num_units=n, kernel_initializer=tf.initializers.glorot_normal()) for n in num_units]
            cells_bw = [tf.nn.rnn_cell.GRUCell(num_units=n, kernel_initializer=tf.initializers.glorot_normal()) for n in num_units]
            
            
            self._db_entry_encodings, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw,
                                                                                            cells_bw=cells_bw,
                                                                                            inputs=reshaped,
                                                                                            dtype=tf.float32)
            
            self._db_entry_encodings_reformed = tf.reshape(self._db_entry_encodings, (self.batchSize, self.dbSize, self.dbSeqLen, self.embeddingSize*2))
            
            
        """
        with tf.variable_scope("input_DB_combined_encoder"):
            
            ###################################################################################################################
            # combine the input encodings with each timestep of each db entry encoding
            ###################################################################################################################
            temp = tf.expand_dims(self._input_encoding, axis=1)
            temp = tf.expand_dims(temp, axis=1)
            temp = tf.tile(temp, [1, self.dbSize, self.dbSeqLen, 1])
            
            input_db_comb_enc = tf.concat([self._db_entry_encodings_reformed, temp], axis=3)
            
            
            
            reshaped = tf.reshape(input_db_comb_enc, (self.batchSize*self.dbSize, self.dbSeqLen, self.embeddingSize*3))
            
            
            # one-directional
            #num_units = [self.embeddingSize, self.embeddingSize]
            #cells = [tf.nn.rnn_cell.GRUCell(num_units=n, kernel_initializer=tf.initializers.glorot_normal()) for n in num_units]
            #stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)
            
            
            #self._db_entry_encodings, database_entry_final_states = tf.nn.dynamic_rnn(stacked_rnn_cell, reshaped, dtype=tf.float32)
            #self._db_entry_encodings_reformed = tf.reshape(self._db_entry_encodings, (self.batchSize, self.dbSize, self.dbSeqLen, self.embeddingSize))
            
            
            
            # bi-directional
            num_units = [self.embeddingSize, self.embeddingSize]
            
            cells_fw = [tf.nn.rnn_cell.GRUCell(num_units=n, kernel_initializer=tf.initializers.glorot_normal()) for n in num_units]
            cells_bw = [tf.nn.rnn_cell.GRUCell(num_units=n, kernel_initializer=tf.initializers.glorot_normal()) for n in num_units]
            
            self._db_entry_encodings, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw,
                                                                                            cells_bw=cells_bw,
                                                                                            inputs=reshaped,
                                                                                            dtype=tf.float32)
            
            self._db_entry_encodings_reformed = tf.reshape(self._db_entry_encodings, (self.batchSize, self.dbSize, self.dbSeqLen, self.embeddingSize*2))
            
            
            ###################################################################################################################
            ###################################################################################################################
        """

            
        self._db_entry_encodings_flattened = tf.reshape(self._db_entry_encodings, (self.batchSize, self.dbSize*self.dbSeqLen, self.embeddingSize*2))
        
        
        with tf.variable_scope("attention_mechanism"):
            
            
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(self.embeddingSize, self._db_entry_encodings_flattened, normalize=True)
            
            self.cell = tf.contrib.seq2seq.AttentionWrapper(tf.nn.rnn_cell.GRUCell(self.embeddingSize, kernel_initializer=tf.initializers.glorot_normal()),
                                                       attention_mechanism,
                                                       output_attention=False)
        
        
        
        
        """
        #
        # set up the network for finding the most relevant DB entry
        #
        with tf.variable_scope("DB_entry_and_input_matcher"):
            
            database_entry_final_states = database_entry_final_states[-1]
            database_entry_final_states = tf.reshape(database_entry_final_states, (self.batchSize, self.dbSize, self.embeddingSize))
            
            input_encoding_temp = tf.expand_dims(self._input_encoding, -1)
            input_encoding_temp = tf.transpose(input_encoding_temp, [0, 2, 1])
            
            
            # compute DB entry & input match score
            dotted = database_entry_final_states * input_encoding_temp
            summed = tf.reduce_sum(dotted, 2) # the match scores for the input to each DB entry
            match_score = tf.nn.softmax(summed)
            
            
            # for now only allow copying from the first DB entry
            #db_entry_0_encoding = self._db_entry_encodings_reformed[:,1,:,:]
            #db_entry_0_input = self._db_entries[:,1,:]
            #db_entry_0_input_one_hot = self._db_entries_one_hot[:,1,:,:]
            
            
            # concatenate all the DB entries to eachother
            db_entry_concat_encoding = tf.concat([self._db_entry_encodings_reformed[:,0,:,:],
                                                  self._db_entry_encodings_reformed[:,1,:,:],
                                                  self._db_entry_encodings_reformed[:,2,:,:],
                                                  self._db_entry_encodings_reformed[:,3,:,:]], axis=1)
            db_entry_concat_input = tf.concat([self._db_entries[:,0,:],
                                               self._db_entries[:,1,:],
                                               self._db_entries[:,2,:],
                                               self._db_entries[:,3,:]], axis=1)
            db_entry_concat_input_one_hot = tf.concat([self._db_entries_one_hot[:,0,:,:],
                                                       self._db_entries_one_hot[:,1,:,:],
                                                       self._db_entries_one_hot[:,2,:,:],
                                                       self._db_entries_one_hot[:,3,:,:]], axis=1)
            
            
            
            # multiply DB entries by their match score
            self._weighted_summed_db_entry_encodings = tf.einsum("ij,ijkl->ikl", match_score, self._db_entry_encodings_reformed)
            self._weighted_summed_db_entries_one_hot = tf.einsum("ij,ijkl->ikl", match_score, self._db_entries_one_hot)
            
            
            
            
            # compute the weighted summed encoding sequence and "DB entry char sequence" to input to the copynet
            # TODO: is this multiplying the way I want it to???
            
            #match_score = tf.reshape(match_score, (self.batchSize, self.dbSize, 1, 1))
            #self._weighted_db_entry_encodings = tf.multiply(match_score, self._db_entry_encodings_reformed) 
            #self._weighted_db_entries_one_hot = tf.multiply(match_score, self._db_entries_one_hot)     
            
            
            
            #self._weighted_db_entry_encodings_flattened = tf.reshape(self._weighted_db_entry_encodings, (self.batchSize, self.dbSize*self.dbSeqLen, self.embeddingSize))
        """
        
        # concatenate all the DB entries to eachother
        db_entry_concat_encoding = tf.concat([self._db_entry_encodings_reformed[:,0,:,:],
                                              self._db_entry_encodings_reformed[:,1,:,:],
                                              self._db_entry_encodings_reformed[:,2,:,:],
                                              self._db_entry_encodings_reformed[:,3,:,:]], axis=1)
        db_entry_concat_input = tf.concat([self._db_entries[:,0,:],
                                           self._db_entries[:,1,:],
                                           self._db_entries[:,2,:],
                                           self._db_entries[:,3,:]], axis=1)
        
        
        #
        # setup the decoder
        #
        self._ground_truth_outputs = tf.placeholder(tf.int32, [self.batchSize, self.outputSeqLen], name='true_robot_outputs') 
        #self._ground_truth_outputs_one_hot = tf.one_hot(self._ground_truth_outputs, self.vocabSize)
        
        #self.copynet_cell = copynet.CopyNetWrapper(cell=tf.nn.rnn_cell.GRUCell(self.embeddingSize),
        #                                      encoder_states=self._db_entry_encodings_flattened,
        #                                      encoder_input_ids=self._db_entries_flattened,
        #                                      vocab_size=self.vocabSize)
        
        
        #self.copynet_cell = copynet.CopyNetWrapper2(cell=tf.nn.rnn_cell.GRUCell(self.embeddingSize),
        #                                      encoder_states=self._weighted_summed_db_entry_encodings,
        #                                      encoder_input_ids=self._weighted_summed_db_entries_one_hot,
        #                                      vocab_size=self.vocabSize)
        #db_entry_concat_encoding = tf.zeros(db_entry_concat_encoding.shape)
        
        #self.copynet_cell = copynet.CopyNetWrapper(cell=tf.nn.rnn_cell.LSTMCell(self.embeddingSize, initializer=tf.initializers.glorot_normal()),
        self.copynet_cell = copynet.CopyNetWrapper(cell=tf.nn.rnn_cell.GRUCell(self.embeddingSize, kernel_initializer=tf.initializers.glorot_normal()),
        #self.copynet_cell = copynet.CopyNetWrapper(cell=self.cell,
                                                   encoder_states=db_entry_concat_encoding,
                                                   encoder_input_ids=db_entry_concat_input,
                                                   vocab_size=self.vocabSize)
        
        
            
        # this should actually take the output of an attention layer over the encoder outputs
        #decoder_initial_state = self.cell.zero_state(self.batchSize, dtype=tf.float32)
        #decoder_initial_state = decoder_initial_state.clone(cell_state=self._input_encoding)
        
        self.decoder_initial_state = self.copynet_cell.zero_state(self.batchSize, tf.float32).clone(cell_state=self._input_encoding) # what to use for the cell state?
        #self.decoder_initial_state = decoder_initial_state
        
        
        # append start char on to beginning of outputs so they can be used for teacher forcing - i.e. as inputs to the coipynet decoder
        after_slice = tf.strided_slice(self._ground_truth_outputs, [0, 0], [self.batchSize, -1], [1, 1]) # slice of the last char of each output sequence (is this necessary?)
        decoder_inputs = tf.concat( [tf.fill([self.batchSize, 1], charToIndex[goChar]), after_slice], 1) # concatenate on a go char onto the start of each output sequence
        
        self.decoder_inputs_one_hot = tf.one_hot(decoder_inputs, self.vocabSize)
        
        # rollout the decoder two times - once for use with teacher forcing (training) and once without (testing)
        # for training
        self._loss, self._train_predicted_output_sequences, self._train_copy_scores, self._train_gen_scores = self.build_decoder(teacherForcing=True, scopeName="decoder_train")
        
        #self._loss = tf.check_numerics(self._loss, "_loss")
        
        # for testing
        self._test_loss, self._test_predicted_output_sequences, self._test_copy_scores, self._test_gen_scores = self.build_decoder(teacherForcing=True, scopeName="decoder_test")
        
        
        #
        # setup the training function
        #
        opt = tf.train.AdamOptimizer(learning_rate=1e-3)
        
        gradients = opt.compute_gradients(self._loss)
        
        #tf.check_numerics(gradients, "gradients")
        
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
                #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_output, logits=output)
                cross_entropy = tf.losses.softmax_cross_entropy(ground_truth_output, output, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE) # is this right for sequence eval?
                #current_loss = tf.reduce_sum(cross_entropy)
                loss = loss + cross_entropy
            
            
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



def color_results(outputStringsPred, outputIndexGt, copyScores, genScores, charToIndex):
    
    coloredOutputStrings = []
    
    copyScoresPred = []
    genScoresPred = []
    
    copyScoresGt = []
    genScoresGt = []
    
    
    for i in range(len(outputStringsPred)):
        colOutStr = ""
        
        cScoresPred = []
        gScoresPred = []
        
        cScoresGt = []
        gScoresGt = []
        
        
        for j in range(len(outputStringsPred[i])):
            cs = copyScores[i, j, charToIndex[outputStringsPred[i][j]]]
            gs = genScores[i, j, charToIndex[outputStringsPred[i][j]]]
            
            csGt = copyScores[i, j, outputIndexGt[i][j]]
            gsGt = genScores[i, j, outputIndexGt[i][j]]
            
            
            # color the char if it was coppied from the input
            if cs > gs:
                colOutStr += "\x1b[36m" + outputStringsPred[i][j] + "\x1b[0m" # blue-green
                
            elif cs == gs:
                colOutStr += "\x1b[33m" + outputStringsPred[i][j] + "\x1b[0m" # yellow
            
            else:
                colOutStr += outputStringsPred[i][j]
            
            
            cScoresPred.append(cs)
            gScoresPred.append(gs)
            
            cScoresGt.append(csGt)
            gScoresGt.append(gsGt)
            
        
        coloredOutputStrings.append(colOutStr)
        
        
        copyScoresPred.append(cScoresPred)
        genScoresPred.append(gScoresPred)
        
        copyScoresGt.append(cScoresGt)
        genScoresGt.append(gScoresGt)
        
        
    return coloredOutputStrings, copyScoresPred, genScoresPred, copyScoresGt, genScoresGt



if __name__ == "__main__":
    
    #
    # simulate some interactions
    #
    databases = []
    inputs = []
    outputs = []
    
    
    database = ["D | Canon | LOCATION | DISPLAY_1111111111.",
                "D | Canon | PRICE | $9999999999.",
                "D | Nikon | LOCATION | DISPLAY_2222222222.",
                "D | Nikon | PRICE | $8888888888."]
    
    # example 1
    databases.append(database)
    inputs.append("C | DISPLAY_1111111111 | How much is this one?")
    outputs.append("S | DISPLAY_1111111111 | This one is $9999999999.")
    
    
    # example 2
    databases.append(database)
    inputs.append("C | DISPLAY_2222222222 | How much is this one?")
    outputs.append("S | DISPLAY_2222222222 | This one is $8888888888.")
    
    
    
    database = ["D | Canon | LOCATION | DISPLAY_1111111111.",
                "D | Canon | PRICE | $7777777777.",
                "D | Nikon | LOCATION | DISPLAY_2222222222.",
                "D | Nikon | PRICE | $6666666666."]
    
    # example 3
    databases.append(database)
    inputs.append("C | DISPLAY_1111111111 | How much is this one?")
    outputs.append("S | DISPLAY_1111111111 | This one is $7777777777.")
    
    
    # example 4
    databases.append(database)
    inputs.append("C | DISPLAY_2222222222 | How much is this one?")
    outputs.append("S | DISPLAY_2222222222 | This one is $6666666666.")
    
    
    #
    #
    #
    database = ["D | Canon | LOCATION | DISPLAY_1111111111.",
                "D | Canon | PRICE | $8888888888.",
                "D | Nikon | LOCATION | DISPLAY_2222222222.",
                "D | Nikon | PRICE | $9999999999."]
    
    # example 5
    databases.append(database)
    inputs.append("C | DISPLAY_1111111111 | How much is this one?")
    outputs.append("S | DISPLAY_1111111111 | This one is $8888888888.")
    
    
    # example 6
    databases.append(database)
    inputs.append("C | DISPLAY_2222222222 | How much is this one?")
    outputs.append("S | DISPLAY_2222222222 | This one is $9999999999.")
    
    
    database = ["D | Canon | LOCATION | DISPLAY_1111111111.",
                "D | Canon | PRICE | $6666666666.",
                "D | Nikon | LOCATION | DISPLAY_2222222222.",
                "D | Nikon | PRICE | $7777777777."]
    
    # example 5
    databases.append(database)
    inputs.append("C | DISPLAY_1111111111 | How much is this one?")
    outputs.append("S | DISPLAY_1111111111 | This one is $6666666666.")
    
    
    # example 6
    databases.append(database)
    inputs.append("C | DISPLAY_2222222222 | How much is this one?")
    outputs.append("S | DISPLAY_2222222222 | This one is $7777777777.")
    
    
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
    embeddingSize = 20
    
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
    
    
    numEpochs = 20000
    
    for e in range(numEpochs):
        
        trainCost = learner.train(trainInputs, trainDbs, trainGroundTruth)
        #print trainCost
        
        if e % 100 == 0:
            
            #
            # compute accuracy, etc.
            #
            
            # TRAIN
            trainPreds, trainCopyScores, trainGenScores = learner.predict(trainInputs, trainDbs, trainGroundTruth)
            
            
            trainAcc = 0.0
            for i in range(len(trainPreds)):
                trainAcc += normalized_edit_distance(trainGroundTruth[i], trainPreds[i])
            trainAcc /= len(trainGroundTruth)
            
            #trainPredsFlat = np.array(trainPreds).flatten()
            #trainAcc = metrics.accuracy_score(trainPredsFlat, trainGroundTruthFlat)
            trainPredSents = unvectorize_sentences(trainPreds, indexToChar)
            trainPredSents2, trainCopyScores2, trainGenScores2, trainCopyScoresGt, trainGenScoresGt = color_results(trainPredSents, 
                                                                                                                    trainGroundTruth,
                                                                                                                    trainCopyScores, 
                                                                                                                    trainGenScores, 
                                                                                                                    charToIndex)
            
            
            # TEST
            testPreds, testCopyScores, testGenScores = learner.predict(testInputs, testDbs, testGroundTruth)
            
            testAcc = 0.0
            for i in range(len(testPreds)):
                testAcc += normalized_edit_distance(testGroundTruth[i], testPreds[i])
            testAcc /= len(testGroundTruth)
            
            #testPredsFlat = np.array(testPreds).flatten()
            #testAcc = metrics.accuracy_score(testPredsFlat, testGroundTruthFlat)
            testPredSents = unvectorize_sentences(testPreds, indexToChar)
            testPredSents2, testCopyScores2, testGenScores2, testCopyScoresGt, testGenScoresGt = color_results(testPredSents,
                                                                                                               testGroundTruth, 
                                                                                                               testCopyScores, 
                                                                                                               testGenScores, 
                                                                                                               charToIndex)
            
            
            print "****************************************************************"
            print "TRAIN", e, round(trainCost, 3), round(trainAcc, 3)
            
            for i in range(len(trainPredSents2)):
                print "TRUE:", outputs[i]
                print "PRED:", trainPredSents2[i]
                print
            
            
            print "TEST", e, round(testAcc, 3)
            
            for i in range(len(testPredSents2)):
                print "TRUE:", outputs[i+4]
                print "PRED:", testPredSents2[i]
                print
        
            print "****************************************************************"
            
            
            
            with open(sessionDir+"/{:}_outputs.csv".format(e), "wb") as csvfile:
                
                writer = csv.writer(csvfile)
                
                writer.writerow(["TRAIN", round(trainCost, 3), round(trainAcc, 3)])
                
                for i in range(len(trainPredSents)):
                    writer.writerow(["TRUE:"] + [c for c in outputs[i]])
                    writer.writerow(["PRED:"] + [c for c in trainPredSents[i]])
                    
                    writer.writerow(["PRED COPY:"] + [c for c in trainCopyScores2[i]])
                    writer.writerow(["PRED GEN:"] + [c for c in trainGenScores2[i]])
                    
                    writer.writerow(["TRUE COPY:"] + [c for c in trainCopyScoresGt[i]])
                    writer.writerow(["TRUE GEN:"] + [c for c in trainGenScoresGt[i]])
                    
                    writer.writerow([])
                
                
                writer.writerow(["TEST", round(testAcc, 3)])
                
                for i in range(len(testPredSents)):
                    writer.writerow(["TRUE:"] + [c for c in outputs[i+4]])
                    writer.writerow(["PRED:"] + [c for c in testPredSents[i]])
                    
                    writer.writerow(["PRED COPY:"] + [c for c in testCopyScores2[i]])
                    writer.writerow(["PRED GEN:"] + [c for c in testGenScores2[i]])
                    
                    writer.writerow(["TRUE COPY:"] + [c for c in testCopyScoresGt[i]])
                    writer.writerow(["TRUE GEN:"] + [c for c in testGenScoresGt[i]])    
                    
                    writer.writerow([])
            
            
            if trainAcc == 0.0:
                print "training error is 0.0"
                break








