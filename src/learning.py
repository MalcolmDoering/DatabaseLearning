'''
Created on 2019/05/17

@author: malcolm
'''


import tensorflow as tf

from copynet import copynet



print("tensorflow version", tf.__version__, flush=True)



#eosChar = "#"
goChar = "~"


class CustomNeuralNetwork(object):
    
    def __init__(self, inputUttVecDim, dbSeqLen, outputSeqLen, locationVecLen, 
                 batchSize, numUniqueCams, numUniqueAtts, vocabSize, 
                 embeddingSize, charToIndex, camTemp, attTemp,
                 seed):
        
        self.inputUttVecDim = inputUttVecDim
        self.dbSeqLen = dbSeqLen
        self.outputSeqLen = outputSeqLen
        self.locationVecLen = locationVecLen
        self.batchSize = batchSize
        self.numUniqueCams = numUniqueCams
        self.numUniqueAtts = numUniqueAtts
        self.vocabSize = vocabSize
        self.embeddingSize = embeddingSize
        self.charToIndex = charToIndex
        
        self.gumbel_softmax_temp_cams = camTemp
        self.gumbel_softmax_temp_atts = attTemp
        
        
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        
        
        #
        # build the model
        #
        
        self._gumbel_softmax_temp = tf.placeholder(tf.float32, shape=(), name='gumbel_softmax_temp')
        #self.gumbel_softmax_temp_cams = self._gumbel_softmax_temp
        #self.gumbel_softmax_temp_atts = self._gumbel_softmax_temp
        
        #with tf.name_scope("input encoder"):
        
        # for sequence
        #self._inputs = tf.placeholder(tf.int32, [self.batchSize, self.inputSeqLen, ], name='customer_inputs')
        #self._input_one_hot = tf.one_hot(self._inputs, self.vocabSize)
        
        # for BoW
        self._inputs = tf.placeholder(tf.float32, [self.batchSize, self.inputUttVecDim, ], name='customer_inputs')
        
        
        self._location_inputs = tf.placeholder(tf.float32, [self.batchSize, self.locationVecLen])
        
        
        self._gtDbCamIndices = tf.placeholder(tf.int32, [self.batchSize, ], name='gt_db_cameras')
        self._gtDbAttIndices = tf.placeholder(tf.int32, [self.batchSize, ], name='gt_db_attributes')
        
        self._gtDbCams = tf.one_hot(self._gtDbCamIndices, self.numUniqueCams)
        self._gtDbAtts = tf.one_hot(self._gtDbAttIndices, self.numUniqueAtts)
        
        
        
        with tf.variable_scope("decoding/input_encoder_1"):
            # input encoder for the initial state of the copynet
            
            """
            num_units = [self.embeddingSize, self.embeddingSize]
            #cells = [tf.nn.rnn_cell.LSTMCell(num_units=n, initializer=tf.initializers.glorot_normal()) for n in num_units]
            cells = [tf.nn.rnn_cell.GRUCell(num_units=n, kernel_initializer=tf.initializers.glorot_normal()) for n in num_units]
            
            stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)
            
            _, input_encoding = tf.nn.dynamic_rnn(stacked_rnn_cell, self._input_one_hot, dtype=tf.float32)
            
            # for single layer GRU
            self._input_utt_encoding = input_encoding[-1] # TODO why is this here??? [-1] A: get output instead of candidate 
        
            # for two layer LSTM
            #self._input_encoding = tf.concat([input_encoding[0][-1], input_encoding[1][-1]], axis=1)
            
            
            
            self._loc_utt_combined_input_encoding = tf.layers.dense(tf.concat([self._input_utt_encoding, self._location_inputs], axis=1),
                                                                    self.embeddingSize, 
                                                                    activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal())
            """
            
            self._loc_utt_combined_input_encoding = tf.concat([self._inputs, self._location_inputs], axis=1)
            
            self._loc_utt_combined_input_encoding = tf.layers.dense(self._loc_utt_combined_input_encoding,
                                                                    1900, 
                                                                    activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal())
            
            self._loc_utt_combined_input_encoding = tf.layers.dense(self._loc_utt_combined_input_encoding,
                                                                    self.embeddingSize, 
                                                                    activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal())
            
        
        
        with tf.variable_scope("input_encoder_2"):
            # input encoder for finding the most relevant camera and attribute
            
            """
            num_units = [self.embeddingSize, self.embeddingSize]
            #cells = [tf.nn.rnn_cell.LSTMCell(num_units=n, initializer=tf.initializers.glorot_normal()) for n in num_units]
            cells = [tf.nn.rnn_cell.GRUCell(num_units=n, kernel_initializer=tf.initializers.glorot_normal()) for n in num_units]
            
            stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)
            
            _, input_encoding = tf.nn.dynamic_rnn(stacked_rnn_cell, self._input_one_hot, dtype=tf.float32)
            
            # for single layer GRU
            self._input_utt_encoding_2 = input_encoding[-1] # TODO why is this here??? [-1] A: get output instead of candidate 
        
            # for two layer LSTM
            #self._input_encoding_2 = tf.concat([input_encoding[0][-1], input_encoding[1][-1]], axis=1)
            
            self._loc_utt_combined_input_encoding_2 = tf.layers.dense(tf.concat([self._input_utt_encoding_2, self._location_inputs], axis=1),
                                                                    self.embeddingSize, 
                                                                    activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal())
            """
            
            self._loc_utt_combined_input_encoding_2 = tf.concat([self._inputs, self._location_inputs], axis=1)
            
            self._loc_utt_combined_input_encoding_2 = tf.layers.dense(self._loc_utt_combined_input_encoding_2,
                                                                    1900, 
                                                                    activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal())
            
            self._loc_utt_combined_input_encoding_2 = tf.layers.dense(self._loc_utt_combined_input_encoding_2,
                                                                    self.embeddingSize, 
                                                                    activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal())
        
        
        
        with tf.variable_scope("DB_matcher"):
            # find the best matching camera and attribute from the database
            
            # use only softmax for addressing
            #cam1 = tf.layers.dense(self._loc_utt_combined_input_encoding_2, self.numUniqueCams, activation=tf.nn.tanh, use_bias=True, kernel_initializer=tf.initializers.he_normal())
            #att1 = tf.layers.dense(self._loc_utt_combined_input_encoding_2, self.numUniqueAtts, activation=tf.nn.tanh, use_bias=True, kernel_initializer=tf.initializers.he_normal())
            
            #self.camMatch = tf.nn.softmax(cam1)
            #self.attMatch = tf.nn.softmax(att1)
            
            """
            # combined softmax and GS
            cam1 = tf.layers.dense(self._loc_utt_combined_input_encoding_2, self.numUniqueCams, activation=tf.nn.tanh, use_bias=True, kernel_initializer=tf.initializers.he_normal())
            att1 = tf.layers.dense(self._loc_utt_combined_input_encoding_2, self.numUniqueAtts, activation=tf.nn.tanh, use_bias=True, kernel_initializer=tf.initializers.he_normal())
            
            smCamMatch = tf.nn.softmax(cam1)
            smAttMatch = tf.nn.softmax(att1)
            
            #gsCamMatch = tf.contrib.distributions.RelaxedOneHotCategorical(self.gumbel_softmax_temp_cams, probs=smCamMatch).sample()
            #gsAttMatch = tf.contrib.distributions.RelaxedOneHotCategorical(self.gumbel_softmax_temp_atts, probs=smAttMatch).sample()
            
            #self.camMatch = self._gumbel_softmax_temp * gsCamMatch + (1.0-self._gumbel_softmax_temp) * smCamMatch
            #self.attMatch = self._gumbel_softmax_temp * gsAttMatch + (1.0-self._gumbel_softmax_temp) * smAttMatch
            
            sharpCamMatch = tf.pow(smCamMatch, 10)
            sharpAttMatch = tf.pow(smAttMatch, 10)
            
            sharpCamDen = tf.reduce_sum(sharpCamMatch, 1, keepdims=True)
            sharpAttDen = tf.reduce_sum(sharpAttMatch, 1, keepdims=True)
            
            sharpCamMatch = sharpCamMatch / sharpCamDen
            sharpAttMatch = sharpAttMatch / sharpAttDen
            
            self.camMatch = self._gumbel_softmax_temp * sharpCamMatch + (1.0-self._gumbel_softmax_temp) * smCamMatch
            self.attMatch = self._gumbel_softmax_temp * sharpAttMatch + (1.0-self._gumbel_softmax_temp) * smAttMatch
            """
            
            # gumbel softmax used till 20190525
            #cam1 = tf.layers.dense(self._input_encoding_2, self.numUniqueCams, activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal())
            #att1 = tf.layers.dense(self._input_encoding_2, self.numUniqueAtts, activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal())
            
            #cam3 = tf.nn.softmax(cam1)
            #att3 = tf.nn.softmax(att1)
            
            #self.camMatch = tf.contrib.distributions.RelaxedOneHotCategorical(self.gumbel_softmax_temp_cams, probs=cam3).sample()
            #self.attMatch = tf.contrib.distributions.RelaxedOneHotCategorical(self.gumbel_softmax_temp_atts, probs=att3).sample()
            
            
            # use sharpening instead of gumbel softmax
            #cam3 = tf.pow(cam1, 2)
            #att3 = tf.pow(att1, 2)
            
            #self.camMatch = tf.nn.softmax(cam3)
            #self.attMatch = tf.nn.softmax(att3)
            
            
            # provide the ground truth DB entries
            self.camMatch = self._gtDbCams
            self.attMatch = self._gtDbAtts
            
            
            self.camMatchIndex = tf.argmax(self.camMatch, axis=1)
            self.attMatchIndex = tf.argmax(self.attMatch, axis=1)
            
            
        with tf.variable_scope("decoding/DB_encoder"):
            # DB encoder
            self._db_entries = tf.placeholder(tf.float32, [self.batchSize, self.numUniqueCams, self.numUniqueAtts, self.dbSeqLen, self.vocabSize], name='DB_entries')
            
            
            # multiply by the match vectors and sum so that only the matching value remains
            self.db_match_val = tf.einsum("bcalv,ba->bclv", self._db_entries, self.attMatch)
            self.db_match_val = tf.einsum("bclv,bc->blv", self.db_match_val, self.camMatch)
            
            # normalize each char vec so that it sums to 1
            # problem: with 0 padding we get NaNs
            #self.db_match_val = self.db_match_val / tf.reduce_sum(self.db_match_val, axis=2, keepdims=True)
            
            
            self.db_match_val_charindices = tf.argmax(self.db_match_val, axis=2)
            
            
            # bi-directional GRU
            num_units = [self.embeddingSize, self.embeddingSize]
            
            cells_fw = [tf.nn.rnn_cell.GRUCell(num_units=n, kernel_initializer=tf.initializers.glorot_normal()) for n in num_units]
            cells_bw = [tf.nn.rnn_cell.GRUCell(num_units=n, kernel_initializer=tf.initializers.glorot_normal()) for n in num_units]
            
            
                       
            #
            # try encoding after summing
            #
            
            self.db_match_val_encoding, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw,
                                                                                            cells_bw=cells_bw,
                                                                                            inputs=self.db_match_val,
                                                                                            dtype=tf.float32)
            
            
            
            #
            # try encoding before summing
            #
            """
            self._db_entries_flat = tf.reshape(self._db_entries, (self.batchSize*self.numUniqueCams*self.numUniqueAtts, self.dbSeqLen, self.vocabSize))
            
            db_entry_encodings_flat, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw,
                                                                                            cells_bw=cells_bw,
                                                                                            inputs=self._db_entries_flat,
                                                                                            dtype=tf.float32)
            
            db_entry_encodings = tf.reshape(db_entry_encodings_flat, (self.batchSize, self.numUniqueCams, self.numUniqueAtts, self.dbSeqLen, self.embeddingSize*2))
            
            
            self.db_match_val_encoding = tf.einsum("bcalv,ba->bclv", db_entry_encodings, self.attMatch)
            self.db_match_val_encoding = tf.einsum("bclv,bc->blv", self.db_match_val_encoding, self.camMatch)
            """
            
        
        with tf.variable_scope("decoding/location_layer"):
            # get the shopkeeper output location
            
            locHid = tf.layers.dense(self._loc_utt_combined_input_encoding,
                                     self.embeddingSize, 
                                     activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal())
            
            self.locationOut = tf.layers.dense(locHid,
                                               self.locationVecLen, 
                                               activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal())
            
            
            
            
        
        #
        # setup the decoder
        #
        self._ground_truth_outputs = tf.placeholder(tf.int32, [self.batchSize, self.outputSeqLen], name='true_robot_outputs')
        #self._ground_truth_outputs_one_hot = tf.one_hot(self._ground_truth_outputs, self.vocabSize)
        
        
        self._ground_truth_output_lens = tf.placeholder(tf.int32, [self.batchSize,], name="true_robot_output_lens")
        self._output_mask = tf.sequence_mask(self._ground_truth_output_lens, self.outputSeqLen, dtype=tf.int32)
        
        
        self._ground_truth_location_outputs = tf.placeholder(tf.int32, [self.batchSize, self.locationVecLen])
        
        
        self.copynet_cell = copynet.CopyNetWrapper3(cell=tf.nn.rnn_cell.GRUCell(self.embeddingSize, kernel_initializer=tf.initializers.glorot_normal()),
                                                   encoder_states=self.db_match_val_encoding,
                                                   #encoder_input_ids=self._db_entries[:,0,0,:,:],
                                                   encoder_input_ids=self.db_match_val,
                                                   vocab_size=self.vocabSize)
        
        
        self.decoder_initial_state = self.copynet_cell.zero_state(self.batchSize, tf.float32).clone(cell_state=self._loc_utt_combined_input_encoding)
        
        
        # append start char on to beginning of outputs so they can be used for teacher forcing - i.e. as inputs to the coipynet decoder
        after_slice = tf.strided_slice(self._ground_truth_outputs, [0, 0], [self.batchSize, -1], [1, 1]) # slice of the last char of each output sequence (is this necessary?)
        decoder_inputs = tf.concat( [tf.fill([self.batchSize, 1], charToIndex[goChar]), after_slice], 1) # concatenate on a go char onto the start of each output sequence
        
        self.decoder_inputs_one_hot = tf.one_hot(decoder_inputs, self.vocabSize)
        
        # rollout the decoder two times - once for use with teacher forcing (training) and once without (testing)
        self._teacher_forcing_prob = tf.placeholder(tf.float32, shape=(), name='teacher_forcing_prob')
        
        # for training
        self._loss, self._train_predicted_output_sequences, self._train_copy_scores, self._train_gen_scores = self.build_decoder(teacherForcing=True, scopeName="decoding/decoder_train")
        
        #self._loss = tf.check_numerics(self._loss, "_loss")                                                                                                                                                                  
        
        # for testing
        self._test_loss, self._test_predicted_output_sequences, self._test_copy_scores, self._test_gen_scores = self.build_decoder(teacherForcing=False, scopeName="decoder_test")
        
        
        # add the loss from the location predictions
        locLoss = tf.losses.softmax_cross_entropy(self._ground_truth_location_outputs, self.locationOut, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE) # is this right for sequence eval?
        
        self._loss += locLoss
        self._test_loss += locLoss
        
        
        #
        # setup the training function
        #
        opt = tf.train.AdamOptimizer(learning_rate=1e-4)
        #opt = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
        
        self.reset_optimizer_op = tf.variables_initializer(opt.variables())
        
        
        #
        # for training the entire network
        #
        gradients = opt.compute_gradients(self._loss)
        #tf.check_numerics(gradients, "gradients")
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        
        self._train_op_1 = opt.apply_gradients(capped_gradients, name="train_op")
        
        
        #
        # for only training the decoding part (and not the addressing part)
        #
        decoding_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoding")
        decoding_gradients = opt.compute_gradients(self._loss, var_list=decoding_train_vars)
        decoding_capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in decoding_gradients if grad is not None]
        
        self._train_op_2 = opt.apply_gradients(decoding_capped_gradients)
        
        
        self._train_op = self._train_op_1
        
        
        #
        # setup the prediction function
        #
        self._pred_utt_op = tf.argmax(self._test_predicted_output_sequences, 2, name="predict_utterance_op")
        self._pred_shkp_loc_op = tf.argmax(self.locationOut, 1, name="predict_shopkeeper_location_op")
        #self._pred_prob_op = tf.nn.softmax(predicted_output_sequences, axis=2, name="predict_prob_op")
        #self._pred_log_prob_op = tf.log(predict_proba_op, name="predict_log_proba_op")
        
        
        self._init_op = tf.initialize_all_variables()
        self._init_decoding_op = tf.initialize_variables(decoding_train_vars)
        
        
        self.initialize()
        
        self.saver = tf.train.Saver()
        
    
    
    def build_decoder(self, teacherForcing=False, scopeName="decoder"):
        
        
        with tf.variable_scope(scopeName):
            
            loss = 0
            predicted_output_sequences = []
            copy_scores = []
            gen_scores = []
            
            state = self.decoder_initial_state
            output = self.decoder_inputs_one_hot[:, 0, :] # each output sequence must have a 'start' char appended to the beginning
            
            
            for i in range(self.outputSeqLen):
                
                if teacherForcing:
                    # if using teacher forcing
                    bernoulliSample = tf.to_float(tf.distributions.Bernoulli(probs=self._teacher_forcing_prob).sample())
                    output = tf.math.scalar_mul(bernoulliSample, self.decoder_inputs_one_hot[:, i, :]) + tf.math.scalar_mul((1.0-bernoulliSample), output)
                    output, state, copy_score, gen_score = self.copynet_cell(output, state)
                    
                    #output, state, copy_score, gen_score = self.copynet_cell(self.decoder_inputs_one_hot[:, i, :], state)
                    
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
                cross_entropy = tf.losses.softmax_cross_entropy(ground_truth_output, output, weights=self._output_mask[:,i], reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE) # is this right for sequence eval?
                #current_loss = tf.reduce_sum(cross_entropy)
                loss = loss + cross_entropy
            
            
            predicted_output_sequences = tf.concat(predicted_output_sequences, 1)
            copy_scores = tf.concat(copy_scores, 1)
            gen_scores = tf.concat(gen_scores, 1)
        
        return loss, predicted_output_sequences, copy_scores, gen_scores
    
    
    def initialize(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        self._sess = tf.Session(config=config)
        self._sess.run(self._init_op)
        
    
    def train(self, inputUtts, inputCustLocs, databases, groundTruthUttOutputs, groundTruthOutputStringLens, groundTruthOutputShkpLocs, gtDbCams, gtDbAtts, gumbelSoftmaxTemp, teacherForcingProb=1.0):
        feedDict = {self._inputs: inputUtts, 
                    self._location_inputs: inputCustLocs, 
                    self._db_entries: databases, 
                    self._ground_truth_outputs: groundTruthUttOutputs,
                    self._ground_truth_output_lens: groundTruthOutputStringLens,
                    self._ground_truth_location_outputs: groundTruthOutputShkpLocs,
                    self._gtDbCamIndices: gtDbCams, 
                    self._gtDbAttIndices: gtDbAtts,
                    self._teacher_forcing_prob: teacherForcingProb,
                    self._gumbel_softmax_temp: gumbelSoftmaxTemp}
        
        trainingLoss, _ = self._sess.run([self._loss, self._train_op], feed_dict=feedDict)
        
        return trainingLoss
    
    
    def train_loss(self, inputUtts, inputCustLocs, databases, groundTruthOutputs, groundTruthOutputStringLens, groundTruthOutputShkpLocs, gtDbCams, gtDbAtts, gumbelSoftmaxTemp, teacherForcingProb=1.0):
        feedDict = {self._inputs: inputUtts, 
                    self._location_inputs: inputCustLocs, 
                    self._db_entries: databases, 
                    self._ground_truth_outputs: groundTruthOutputs,
                    self._ground_truth_output_lens: groundTruthOutputStringLens,
                    self._ground_truth_location_outputs: groundTruthOutputShkpLocs,
                    self._gtDbCamIndices: gtDbCams, 
                    self._gtDbAttIndices: gtDbAtts,
                    self._teacher_forcing_prob: teacherForcingProb,
                    self._gumbel_softmax_temp: gumbelSoftmaxTemp}
        
        loss = self._sess.run(self._loss, feed_dict=feedDict)
        
        return loss
    
    
    def predict(self, inputUtts, inputCustLocs, databases, groundTruthOutputs, groundTruthOutputStringLens, groundTruthOutputShkpLocs, gtDbCams, gtDbAtts, gumbelSoftmaxTemp, teacherForcingProb=1.0):
        feedDict = {self._inputs: inputUtts, 
                    self._location_inputs: inputCustLocs, 
                    self._db_entries: databases, 
                    self._ground_truth_outputs: groundTruthOutputs,
                    self._ground_truth_output_lens: groundTruthOutputStringLens,
                    self._ground_truth_location_outputs: groundTruthOutputShkpLocs,
                    self._gtDbCamIndices: gtDbCams, 
                    self._gtDbAttIndices: gtDbAtts,
                    self._teacher_forcing_prob: teacherForcingProb,
                    self._gumbel_softmax_temp: gumbelSoftmaxTemp}
        
        predUtts, predShkpLocs, copyScores, genScores, camMatchArgMax, attMatchArgMax, camMatch, attMatch = self._sess.run([self._pred_utt_op,
                                                                                                                            self._pred_shkp_loc_op,
                                                                                                                            self._test_copy_scores, 
                                                                                                                            self._test_gen_scores,
                                                                                                                            self.camMatchIndex,
                                                                                                                            self.attMatchIndex,
                                                                                                                            self.camMatch,
                                                                                                                            self.attMatch], feed_dict=feedDict)
        
        return predUtts, predShkpLocs, copyScores, genScores, camMatchArgMax, attMatchArgMax, camMatch, attMatch
    
    
    def save(self, filename):
        self.saver.save(self._sess, filename)
    
    
    def reset_optimizer(self):
        
        self._sess.run(self.reset_optimizer_op)
        
    
    def reinitialize_decoding_weights(self):
        
        self._sess.run(self._init_decoding_op)
        
    
    def set_train_op(self, op):
        if op == 1:
            self._train_op = self._train_op_1
        elif op == 2:
            self._train_op = self._train_op_2
        else:
            self._train_op = None
            print("Warning: Invalid train op", op)
        
    
    