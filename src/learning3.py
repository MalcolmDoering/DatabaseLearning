'''
Created on 2019/05/17

@author: malcolm

don't use CopyNet

output shopkeeper action IDs and DB camera indices instead of generating char sequence
'''


import tensorflow as tf


print("tensorflow version", tf.__version__, flush=True)



eosChar = "#"
goChar = "~"


class CustomNeuralNetwork(object):
    
    def __init__(self, 
                 inputDim, 
                 inputSeqLen, 
                 numOutputClasses,
                 numUniqueCams,
                 numAttributes,
                 numLocations,
                 numSpatialStates,
                 numStateTargets,
                 batchSize, 
                 embeddingSize,
                 camTemp,
                 seed,
                 speechClusterWeights,
                 attributeIndexWeights):
        
        self.inputDim = inputDim
        self.inputSeqLen = inputSeqLen
        self.numOutputClasses = numOutputClasses
        self.numUniqueCams = numUniqueCams
        self.numAttributes = numAttributes
        self.numLocations = numLocations
        self.numSpatialStates = numSpatialStates
        self.numStateTargets = numStateTargets
        self.batchSize = batchSize
        self.embeddingSize = embeddingSize
        self.camTemp = camTemp
        self.seed = seed
        self.speechClusterWeights = speechClusterWeights
        self.attributeIndexWeights = attributeIndexWeights
        
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        
        
        #
        # build the model
        #
        
        # inputs
        self.input_sequences = tf.placeholder(tf.float32, [self.batchSize, self.inputSeqLen, self.inputDim], name="input_sequences")
        self.database_content_lengths = tf.placeholder(tf.float32, [self.batchSize, self.numUniqueCams, self.numAttributes], name="database_content_lengths")
        self.speechClusterWeightsTensor = tf.placeholder(tf.float32, [self.numOutputClasses], name="speech_cluster_weights")
        self.attributeIndexWeightsTensor = tf.placeholder(tf.float32, [2, self.numAttributes], name="attribute_index_weights")
        
        # targets
        self.shopkeeper_action_id_targets = tf.placeholder(tf.int32, [self.batchSize, ], name="shopkeeper_action_id_targets")
        self.cam_index_targets = tf.placeholder(tf.int32, [self.batchSize, self.numUniqueCams], name="cam_index_targets")
        self.attr_index_targets = tf.placeholder(tf.int32, [self.batchSize, self.numAttributes], name="attr_index_targets")
        self.shopkeeper_locations = tf.placeholder(tf.int32, [self.batchSize, ], name="shopkeeper_locations")
        self.spatial_states = tf.placeholder(tf.int32, [self.batchSize, ], name="spatial_states")
        self.state_targets = tf.placeholder(tf.int32, [self.batchSize, ], name="state_targets")
        
        self.shopkeeper_action_id_targets_onehot = tf.one_hot(self.shopkeeper_action_id_targets, self.numOutputClasses)
        #self.cam_index_targets_onehot = tf.one_hot(self.cam_index_targets, self.numUniqueCams+1) # add one for NONE
        self.shopkeeper_locations_one_hot = tf.one_hot(self.shopkeeper_locations, self.numLocations)
        self.spatial_states_one_hot = tf.one_hot(self.spatial_states, self.numSpatialStates)
        self.state_targets_one_hot = tf.one_hot(self.state_targets, self.numStateTargets)
        
        
        # mask for not training on bad speech clusters
        self.output_mask = tf.placeholder(tf.int32, [self.batchSize, ], name="output_mask")
        
        
        with tf.variable_scope("input_encoder"):
            
            # first condense the input vector from each turn (2 layers)
            inputs_reshaped = tf.reshape(self.input_sequences, [self.batchSize*self.inputSeqLen, self.inputDim])
            
            inputs_reshaped_condensed = tf.layers.dense(inputs_reshaped,
                                                        self.embeddingSize,
                                                        activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal())
            
            inputs_reshaped_condensed = tf.layers.dense(inputs_reshaped_condensed,
                                                        self.embeddingSize,
                                                        activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal())
                                                        
            inputs_condensed = tf.reshape(inputs_reshaped_condensed, [self.batchSize, self.inputSeqLen, self.embeddingSize])
            
            
            # then feed the sequence of condensed inputs into an two layer RNN
            num_units = [self.embeddingSize, self.embeddingSize]
            
            cells = [tf.nn.rnn_cell.GRUCell(num_units=n, kernel_initializer=tf.initializers.glorot_normal()) for n in num_units]
            
            stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)
            
            _, input_encoding = tf.nn.dynamic_rnn(stacked_rnn_cell, inputs_condensed, dtype=tf.float32)
            
            # for two layer GRU - get the final encoding from each layer
            self.input_encoding = input_encoding[0] + input_encoding[1] # TODO why is this here??? [-1] A: get output instead of candidate
        
        
        """
        with tf.variable_scope("input_encoder_2"):
            
            # first condense the input vector from each turn (2 layers)
            inputs_reshaped = tf.reshape(self.input_sequences, [self.batchSize*self.inputSeqLen, self.inputDim])
            
            inputs_reshaped_condensed = tf.layers.dense(inputs_reshaped,
                                                        self.embeddingSize,
                                                        activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal())
            
            inputs_reshaped_condensed = tf.layers.dense(inputs_reshaped_condensed,
                                                        self.embeddingSize,
                                                        activation=tf.nn.leaky_relu, kernel_initializer=tf.initializers.he_normal())
                                                        
            inputs_condensed = tf.reshape(inputs_reshaped_condensed, [self.batchSize, self.inputSeqLen, self.embeddingSize])
            
            
            # then feed the sequence of condensed inputs into an two layer RNN
            num_units = [self.embeddingSize, self.embeddingSize]
            
            cells = [tf.nn.rnn_cell.GRUCell(num_units=n, kernel_initializer=tf.initializers.glorot_normal()) for n in num_units]
            
            stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)
            
            _, input_encoding = tf.nn.dynamic_rnn(stacked_rnn_cell, inputs_condensed, dtype=tf.float32)
            
            # for two layer GRU - get the final encoding from each layer
            self.input_encoding_2 = input_encoding[0] + input_encoding[1] # TODO why is this here??? [-1] A: get output instead of candidate
        """
        
        
        with tf.variable_scope("db_addressing/camera_index_decoder"):
            self.cam_index_decoder = tf.layers.dense(self.input_encoding, self.numUniqueCams, activation=tf.nn.sigmoid, use_bias=True, kernel_initializer=tf.initializers.he_normal())
            #self.cam_index_softmax = tf.nn.softmax(self.cam_index_decoder)
            #self.cam_index_argmax = tf.argmax(self.cam_index_softmax, axis=1)
        
        
        with tf.variable_scope("db_addressing/attribute_index_decoder"):
            self.attr_index_decoder = tf.layers.dense(self.input_encoding, self.numAttributes, activation=tf.nn.sigmoid, use_bias=True, kernel_initializer=tf.initializers.he_normal())
            #self.attr_index_softmax = tf.nn.softmax(self.attr_index_decoder)
            #self.attr_index_argmax = tf.argmax(self.attr_index_softmax, axis=1)
        
        
        with tf.variable_scope("db_addressing/database_contents_selector"):
            # get the length of the most relevant DB contents
            # this is the weighted sum of DB content lens
            self.mostReleveantDbContentLens = self.database_content_lengths * tf.expand_dims(self.cam_index_decoder, 2)
            self.mostReleveantDbContentLens = self.database_content_lengths * tf.expand_dims(self.attr_index_decoder, 1)
            self.mostReleveantDbContentLens = tf.reduce_sum(self.mostReleveantDbContentLens, axis=2)
            self.mostReleveantDbContentLens = tf.reduce_sum(self.mostReleveantDbContentLens, axis=1)
            
            # TODO try feeding this into the action decoder...
        
        
        with tf.variable_scope("speech_and_spatial_output/location_decoder"):
            self.location_decoder = tf.layers.dense(self.input_encoding, self.numLocations, activation=tf.nn.leaky_relu, use_bias=True, kernel_initializer=tf.initializers.he_normal())
            self.location_softmax = tf.nn.softmax(self.location_decoder)
            self.location_argmax = tf.argmax(self.location_softmax, axis=1)
        
        
        with tf.variable_scope("speech_and_spatial_output/spatial_state_decoder"):
            self.spatial_state_decoder = tf.layers.dense(self.input_encoding, self.numSpatialStates, activation=tf.nn.leaky_relu, use_bias=True, kernel_initializer=tf.initializers.he_normal())
            self.spatial_state_softmax = tf.nn.softmax(self.spatial_state_decoder)
            self.spatial_state_argmax = tf.argmax(self.spatial_state_softmax, axis=1)
        
        
        with tf.variable_scope("speech_and_spatial_output/state_target_decoder"):
            self.state_target_decoder = tf.layers.dense(self.input_encoding, self.numStateTargets, activation=tf.nn.leaky_relu, use_bias=True, kernel_initializer=tf.initializers.he_normal())
            self.state_target_softmax = tf.nn.softmax(self.state_target_decoder)
            self.state_target_argmax = tf.argmax(self.state_target_softmax, axis=1)
        
        
        with tf.variable_scope("speech_and_spatial_output/shopkeeper_action_decoder"):
            action_decoder_input = tf.concat((self.input_encoding, tf.expand_dims(self.mostReleveantDbContentLens,1)), axis=1)
            
            # is one layer enough?
            self.shopkeeper_action_decoder = tf.layers.dense(action_decoder_input, self.numOutputClasses, activation=tf.nn.leaky_relu, use_bias=True, kernel_initializer=tf.initializers.he_normal())
            self.shopkeeper_action_softmax = tf.nn.softmax(self.shopkeeper_action_decoder)
            self.shopkeeper_action_argmax = tf.argmax(self.shopkeeper_action_softmax, axis=1)
        
        
        with tf.variable_scope("loss"):
            
            self.location_loss = tf.losses.softmax_cross_entropy(self.shopkeeper_locations_one_hot, 
                                                                 self.location_decoder,
                                                                 weights=self.output_mask,
                                                                 reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            
            self.spatial_state_loss = tf.losses.softmax_cross_entropy(self.spatial_states_one_hot, 
                                                                 self.spatial_state_decoder,
                                                                 weights=self.output_mask,
                                                                 reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            
            self.state_target_loss = tf.losses.softmax_cross_entropy(self.state_targets_one_hot, 
                                                                 self.state_target_decoder,
                                                                 weights=self.output_mask,
                                                                 reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            
            
            zero = tf.constant(0, dtype=tf.int32)
            
            camIndexLossWeights = tf.dtypes.cast(tf.reduce_any(tf.not_equal(self.cam_index_targets, zero), axis=1), tf.int32)
            camIndexLossWeights = tf.multiply(camIndexLossWeights, self.output_mask) # element-wise mult
            
            self.camera_index_loss = tf.losses.sigmoid_cross_entropy(self.cam_index_targets, 
                                                                     self.cam_index_decoder,
                                                                     weights=tf.expand_dims(camIndexLossWeights, axis=1),
                                                                     reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            
            
            
            temp = tf.gather(self.attributeIndexWeightsTensor, self.attr_index_targets)           
            tempSplit = tf.split(temp, self.numAttributes, axis=2)
            
            
            attributeIndexLossWeights1 = []
            for i in range(self.numAttributes):
                temp2 = tempSplit[i]
                tempSplit2 = tf.split(temp2, self.numAttributes, axis=1)
                
                attributeIndexLossWeights1.append(tempSplit2[i])
            
            attributeIndexLossWeights1 = tf.concat(attributeIndexLossWeights1, axis=1)
            attributeIndexLossWeights1 = tf.squeeze(attributeIndexLossWeights1)
            
            
            attributeIndexLossWeights2 = tf.dtypes.cast(tf.reduce_any(tf.not_equal(self.attr_index_targets, zero), axis=1), tf.int32)
            attributeIndexLossWeights2 = tf.multiply(attributeIndexLossWeights2, self.output_mask) # element-wise mult
            
            
            attributeIndexLossWeights = tf.multiply(attributeIndexLossWeights1, tf.expand_dims(tf.dtypes.cast(attributeIndexLossWeights2, tf.float32), axis=1))
            
            self.attribute_index_loss = tf.losses.sigmoid_cross_entropy(self.attr_index_targets, 
                                                                     self.attr_index_decoder,
                                                                     weights=attributeIndexLossWeights,
                                                                     reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            
            
            
            temp = tf.squeeze(tf.gather_nd(tf.expand_dims(self.speechClusterWeightsTensor, 1), tf.expand_dims(self.shopkeeper_action_id_targets, 1)))
            speechClusterLossWeights = tf.dtypes.cast(self.output_mask, tf.float32) * temp
            
            self.shopkeeper_action_loss = tf.losses.softmax_cross_entropy(self.shopkeeper_action_id_targets_onehot, 
                                                                          self.shopkeeper_action_decoder,
                                                                          weights=speechClusterLossWeights,
                                                                          reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            
            self.loss = self.shopkeeper_action_loss
            self.loss += self.camera_index_loss
            self.loss += (self.attribute_index_loss * 2.5)
            self.loss += self.location_loss
            self.loss += self.spatial_state_loss
            self.loss += self.state_target_loss
        
        
        #
        # setup the training function
        #
        opt = tf.train.AdamOptimizer(learning_rate=3e-4)
        #opt = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.09, use_nesterov=True)
        
        self.reset_optimizer_op = tf.variables_initializer(opt.variables())
        
        
        #
        # for training the entire network
        #
        gradients = opt.compute_gradients(self.loss)
        #tf.check_numerics(gradients, "gradients")
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        
        self.train_op_1 = opt.apply_gradients(capped_gradients, name="train_op")
        
        
        #
        # for training everything except the db_addressing layers
        #
        #decoding_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "other")
        #decoding_gradients = opt.compute_gradients(self._loss, var_list=decoding_train_vars)
        #decoding_capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in decoding_gradients if grad is not None]
        #
        #self._train_op_2 = opt.apply_gradients(decoding_capped_gradients)
        
        
        self.train_op = self.train_op_1
        
        
        #
        # setup initializer for reinitializing part of the network
        #
        speech_and_spatial_output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "speech_and_spatial_output")
        self.reinit_speech_and_spatial_output_vars_op = tf.initialize_variables(speech_and_spatial_output_vars)
        
        
        #
        # setup the prediction function
        #
        self.init_op = tf.initialize_all_variables()
        
        self.initialize()
        
        self.saver = tf.train.Saver()
    
    
    
    def initialize(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        self.sess = tf.Session(config=config)
        self.sess.run(self.init_op)
    
    
    def reinitialize_speech_and_spatial_output_vars(self):
        
        self.sess.run(self.reinit_speech_and_spatial_output_vars_op)
    
    
    def train(self, 
              inputSequenceVectors,
              outputActionIds,
              outputCameraIndices,
              outputAttributeIndices,
              outputLocations,
              outputSpatialStates,
              outputStateTargets,
              outputMasks,
              dbContentLens):
        
        feedDict = {self.input_sequences: inputSequenceVectors, 
                    self.shopkeeper_action_id_targets: outputActionIds, 
                    self.cam_index_targets: outputCameraIndices,
                    self.attr_index_targets: outputAttributeIndices,
                    self.shopkeeper_locations: outputLocations,
                    self.spatial_states: outputSpatialStates,
                    self.state_targets: outputStateTargets,
                    self.output_mask: outputMasks,
                    self.database_content_lengths: dbContentLens,
                    self.speechClusterWeightsTensor: self.speechClusterWeights,
                    self.attributeIndexWeightsTensor: self.attributeIndexWeights}
        
        trainingLoss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feedDict)
        
        return trainingLoss
    
    
    def get_loss(self, 
              inputSequenceVectors,
              outputActionIds,
              outputCameraIndices,
              outputAttributeIndices,
              outputLocations,
              outputSpatialStates,
              outputStateTargets,
              outputMasks,
              dbContentLens):
        
        feedDict = {self.input_sequences: inputSequenceVectors, 
                    self.shopkeeper_action_id_targets: outputActionIds, 
                    self.cam_index_targets: outputCameraIndices,
                    self.attr_index_targets: outputAttributeIndices,
                    self.shopkeeper_locations: outputLocations,
                    self.spatial_states: outputSpatialStates,
                    self.state_targets: outputStateTargets,
                    self.output_mask: outputMasks,
                    self.database_content_lengths: dbContentLens,
                    self.speechClusterWeightsTensor: self.speechClusterWeights,
                    self.attributeIndexWeightsTensor: self.attributeIndexWeights}
        
        loss = self.sess.run(self.loss, feed_dict=feedDict)
        
        return loss
    
    
    def predict(self, 
              inputSequenceVectors,
              outputActionIds,
              outputCameraIndices,
              outputAttributeIndices,
              outputLocations,
              outputSpatialStates,
              outputStateTargets,
              outputMasks,
              dbContentLens):
        
        feedDict = {self.input_sequences: inputSequenceVectors, 
                    self.shopkeeper_action_id_targets: outputActionIds, 
                    self.cam_index_targets: outputCameraIndices,
                    self.attr_index_targets: outputAttributeIndices,
                    self.shopkeeper_locations: outputLocations,
                    self.spatial_states: outputSpatialStates,
                    self.state_targets: outputStateTargets,
                    self.output_mask: outputMasks,
                    self.database_content_lengths: dbContentLens,
                    self.speechClusterWeightsTensor: self.speechClusterWeights,
                    self.attributeIndexWeightsTensor: self.attributeIndexWeights}
        
        predShkpActionID, predCameraIndices, predAttrIndices, predLocations, predSpatialStates, predStateTargets = self.sess.run([self.shopkeeper_action_argmax, 
                                                                                                                 self.cam_index_decoder,
                                                                                                                 self.attr_index_decoder,
                                                                                                                 self.location_argmax,
                                                                                                                 self.spatial_state_argmax,
                                                                                                                 self.state_target_argmax], feedDict)
        
        return predShkpActionID, predCameraIndices, predAttrIndices, predLocations, predSpatialStates, predStateTargets
    
    
    def save(self, filename):
        self.saver.save(self.sess, filename)
    
    
    def load(self, filename):
        self.saver.restore(self.sess, filename)
    
    
    def reset_optimizer(self):    
        self.sess.run(self.reset_optimizer_op)
        
    
        