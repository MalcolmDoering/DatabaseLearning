'''
Created on 2019/05/17

@author: malcolm


changes to learning3 to workd with actionPrediction18
'''


import tensorflow as tf
import numpy as np

from copynet import copynet


print("tensorflow version", tf.__version__, flush=True)



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
                 seed,
                 speechClusterWeights,
                 attributeIndexWeights):
        
        self.inputDim = inputDim
        self.inputSeqLen = inputSeqLen
        self.numOutputClasses = numOutputClasses
        self.numCameras = numUniqueCams
        self.numAttributes = numAttributes
        self.numLocations = numLocations
        self.numSpatialStates = numSpatialStates
        self.numStateTargets = numStateTargets
        self.batchSize = batchSize
        self.embeddingSize = embeddingSize
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
        self.database_content_lengths = tf.placeholder(tf.float32, [self.batchSize, self.numCameras, self.numAttributes], name="database_content_lengths")
        self.speechClusterWeightsTensor = tf.placeholder(tf.float32, [self.numOutputClasses], name="speech_cluster_weights")
        self.attributeIndexWeightsTensor = tf.placeholder(tf.float32, [2, self.numAttributes], name="attribute_index_weights")
        
        # targets
        self.shopkeeper_action_id_targets = tf.placeholder(tf.int32, [self.batchSize, ], name="shopkeeper_action_id_targets")
        self.cam_index_targets = tf.placeholder(tf.int32, [self.batchSize, self.numCameras], name="cam_index_targets")
        self.attr_index_targets = tf.placeholder(tf.int32, [self.batchSize, self.numAttributes], name="attr_index_targets")
        self.shopkeeper_locations = tf.placeholder(tf.int32, [self.batchSize, ], name="shopkeeper_locations")
        self.spatial_states = tf.placeholder(tf.int32, [self.batchSize, ], name="spatial_states")
        self.state_targets = tf.placeholder(tf.int32, [self.batchSize, ], name="state_targets")
        
        self.shopkeeper_action_id_targets_onehot = tf.one_hot(self.shopkeeper_action_id_targets, self.numOutputClasses)
        #self.cam_index_targets_onehot = tf.one_hot(self.cam_index_targets, self.numCameras+1) # add one for NONE
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
            self.cam_index_decoder = tf.layers.dense(self.input_encoding, self.numCameras, activation=tf.nn.sigmoid, use_bias=True, kernel_initializer=tf.initializers.he_normal())
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
            
            self.shopkeeper_speech_loss = tf.losses.softmax_cross_entropy(self.shopkeeper_action_id_targets_onehot, 
                                                                          self.shopkeeper_action_decoder,
                                                                          weights=speechClusterLossWeights,
                                                                          reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            
            self.attributeIndexLossWeight = 2.5
            
            self.loss = self.shopkeeper_speech_loss
            self.loss += self.camera_index_loss
            self.loss += (self.attribute_index_loss * self.attributeIndexLossWeight)
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
    
    
    def get_batches(self, numSamples):
        # note - this will chop off data doesn't factor into a batch of batchSize
        
        batchStartEndIndices = []
        
        for endIndex in range(self.batchSize, numSamples, self.batchSize):
            batchStartEndIndices.append((endIndex-self.batchSize, endIndex))
        
        return batchStartEndIndices
    
    
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
        
        batchStartEndIndices = self.get_batches(len(outputActionIds))
        
        all_loss = [] 
        all_shopkeeper_action_loss = [] 
        all_camera_index_loss = []
        all_attribute_index_loss = []
        all_location_loss = []
        all_spatial_state_loss = [] 
        all_state_target_loss = []
        
        for sei in batchStartEndIndices:
            
            ###
            
            feedDict = {self.input_sequences: inputSequenceVectors[sei[0]:sei[1]], 
                        self.shopkeeper_action_id_targets: outputActionIds[sei[0]:sei[1]], 
                        self.cam_index_targets: outputCameraIndices[sei[0]:sei[1]],
                        self.attr_index_targets: outputAttributeIndices[sei[0]:sei[1]],
                        self.shopkeeper_locations: outputLocations[sei[0]:sei[1]],
                        self.spatial_states: outputSpatialStates[sei[0]:sei[1]],
                        self.state_targets: outputStateTargets[sei[0]:sei[1]],
                        self.output_mask: outputMasks[sei[0]:sei[1]],
                        self.database_content_lengths: dbContentLens[sei[0]:sei[1]],
                        self.speechClusterWeightsTensor: self.speechClusterWeights,
                        self.attributeIndexWeightsTensor: self.attributeIndexWeights}
            
            loss, shopkeeper_action_loss, camera_index_loss, attribute_index_loss, location_loss, spatial_state_loss, state_target_loss, _ = self.sess.run([self.loss, self.shopkeeper_speech_loss, self.camera_index_loss, self.attribute_index_loss, self.location_loss, self.spatial_state_loss, self.state_target_loss, self.train_op], 
                                                                                                                                                    feed_dict=feedDict)
            attribute_index_loss *= self.attributeIndexLossWeight
            
            ###
            
            all_loss.append(loss) 
            all_shopkeeper_action_loss.append(shopkeeper_action_loss) 
            all_camera_index_loss.append(camera_index_loss)
            all_attribute_index_loss.append(attribute_index_loss)
            all_location_loss.append(location_loss)
            all_spatial_state_loss.append(spatial_state_loss) 
            all_state_target_loss.append(state_target_loss)
            
        return all_loss, all_shopkeeper_action_loss, all_camera_index_loss, all_attribute_index_loss, all_location_loss, all_spatial_state_loss, all_state_target_loss
    
    
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
        
        batchStartEndIndices = self.get_batches(len(outputActionIds))
        
        all_loss = [] 
        all_shopkeeper_action_loss = [] 
        all_camera_index_loss = []
        all_attribute_index_loss = []
        all_location_loss = []
        all_spatial_state_loss = [] 
        all_state_target_loss = []
        
        for sei in batchStartEndIndices:
            
            ###
            
            feedDict = {self.input_sequences: inputSequenceVectors[sei[0]:sei[1]], 
                        self.shopkeeper_action_id_targets: outputActionIds[sei[0]:sei[1]], 
                        self.cam_index_targets: outputCameraIndices[sei[0]:sei[1]],
                        self.attr_index_targets: outputAttributeIndices[sei[0]:sei[1]],
                        self.shopkeeper_locations: outputLocations[sei[0]:sei[1]],
                        self.spatial_states: outputSpatialStates[sei[0]:sei[1]],
                        self.state_targets: outputStateTargets[sei[0]:sei[1]],
                        self.output_mask: outputMasks[sei[0]:sei[1]],
                        self.database_content_lengths: dbContentLens[sei[0]:sei[1]],
                        self.speechClusterWeightsTensor: self.speechClusterWeights,
                        self.attributeIndexWeightsTensor: self.attributeIndexWeights}
            
            loss, shopkeeper_action_loss, camera_index_loss, attribute_index_loss, location_loss, spatial_state_loss, state_target_loss = self.sess.run([self.loss, self.shopkeeper_speech_loss, self.camera_index_loss, self.attribute_index_loss, self.location_loss, self.spatial_state_loss, self.state_target_loss], 
                                                                                                                                                    feed_dict=feedDict)
            
            attribute_index_loss *= self.attributeIndexLossWeight
            
            ###
            
            all_loss.append(loss) 
            all_shopkeeper_action_loss.append(shopkeeper_action_loss) 
            all_camera_index_loss.append(camera_index_loss)
            all_attribute_index_loss.append(attribute_index_loss)
            all_location_loss.append(location_loss)
            all_spatial_state_loss.append(spatial_state_loss) 
            all_state_target_loss.append(state_target_loss)
            
        return all_loss, all_shopkeeper_action_loss, all_camera_index_loss, all_attribute_index_loss, all_location_loss, all_spatial_state_loss, all_state_target_loss
    
    
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
        
        batchStartEndIndices = self.get_batches(len(outputActionIds))
        
        all_predShkpActionID = [] 
        all_predCameraIndices = [] 
        all_predAttrIndices = [] 
        all_predLocations = [] 
        all_predSpatialStates = [] 
        all_predStateTargets = [] 
        all_camIndexWeights = [] 
        all_attributeIndexWeights = [] 
        all_weightedDbContentsSum = []
        
        for sei in batchStartEndIndices:
            
            ###
            
            feedDict = {self.input_sequences: inputSequenceVectors[sei[0]:sei[1]], 
                        self.shopkeeper_action_id_targets: outputActionIds[sei[0]:sei[1]], 
                        self.cam_index_targets: outputCameraIndices[sei[0]:sei[1]],
                        self.attr_index_targets: outputAttributeIndices[sei[0]:sei[1]],
                        self.shopkeeper_locations: outputLocations[sei[0]:sei[1]],
                        self.spatial_states: outputSpatialStates[sei[0]:sei[1]],
                        self.state_targets: outputStateTargets[sei[0]:sei[1]],
                        self.output_mask: outputMasks[sei[0]:sei[1]],
                        self.database_content_lengths: dbContentLens[sei[0]:sei[1]],
                        self.speechClusterWeightsTensor: self.speechClusterWeights,
                        self.attributeIndexWeightsTensor: self.attributeIndexWeights}
            
            predShkpActionID, predCameraIndices, predAttrIndices, predLocations, predSpatialStates, predStateTargets, camIndexWeights, attributeIndexWeights, weightedDbContentsSum = self.sess.run([self.shopkeeper_action_argmax, 
                                                                                                                 self.cam_index_decoder,
                                                                                                                 self.attr_index_decoder,
                                                                                                                 self.location_argmax,
                                                                                                                 self.spatial_state_argmax,
                                                                                                                 self.state_target_argmax,
                                                                                                                 self.cam_index_decoder,
                                                                                                                 self.attr_index_decoder,
                                                                                                                 self.mostReleveantDbContentLens], feedDict)
            
            ###
            
            all_predShkpActionID.append(predShkpActionID)
            all_predCameraIndices.append(predCameraIndices)
            all_predAttrIndices.append(predAttrIndices)
            all_predLocations.append(predLocations)
            all_predSpatialStates.append(predSpatialStates) 
            all_predStateTargets.append(predStateTargets)
            all_camIndexWeights.append(camIndexWeights)
            all_attributeIndexWeights.append(attributeIndexWeights) 
            all_weightedDbContentsSum.append(weightedDbContentsSum)
            
        
        all_predShkpActionID = np.concatenate(all_predShkpActionID)
        all_predCameraIndices = np.concatenate(all_predCameraIndices)
        all_predAttrIndices = np.concatenate(all_predAttrIndices)
        all_predLocations = np.concatenate(all_predLocations)
        all_predSpatialStates = np.concatenate(all_predSpatialStates) 
        all_predStateTargets = np.concatenate(all_predStateTargets)
        all_camIndexWeights = np.concatenate(all_camIndexWeights)
        all_attributeIndexWeights = np.concatenate(all_attributeIndexWeights) 
        all_weightedDbContentsSum = np.concatenate(all_weightedDbContentsSum)
            
            
        return all_predShkpActionID, all_predCameraIndices, all_predAttrIndices, all_predLocations, all_predSpatialStates, all_predStateTargets, all_camIndexWeights, all_attributeIndexWeights, all_weightedDbContentsSum
    
    
    def save(self, filename):
        self.saver.save(self.sess, filename)
    
    
    def load(self, filename):
        self.saver.restore(self.sess, filename)
    
    
    def reset_optimizer(self):    
        self.sess.run(self.reset_optimizer_op)



#
# This is Baseline 1 - same as proposed but without the DB addressing parts
#
class Baseline1(object):
    
    def __init__(self, 
                 inputDim, 
                 inputSeqLen, 
                 numOutputClasses,
                 numLocations,
                 numSpatialStates,
                 numStateTargets,
                 batchSize, 
                 embeddingSize,
                 seed,
                 speechClusterWeights):
        
        self.inputDim = inputDim
        self.inputSeqLen = inputSeqLen
        self.numOutputClasses = numOutputClasses
        self.numLocations = numLocations
        self.numSpatialStates = numSpatialStates
        self.numStateTargets = numStateTargets
        self.batchSize = batchSize
        self.embeddingSize = embeddingSize
        self.seed = seed
        self.speechClusterWeights = speechClusterWeights
        
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        
        
        #
        # build the model
        #
        
        # inputs
        self.input_sequences = tf.placeholder(tf.float32, [self.batchSize, self.inputSeqLen, self.inputDim], name="input_sequences")
        self.speechClusterWeightsTensor = tf.placeholder(tf.float32, [self.numOutputClasses], name="speech_cluster_weights")
        
        # targets
        self.shopkeeper_action_id_targets = tf.placeholder(tf.int32, [self.batchSize, ], name="shopkeeper_action_id_targets")
        self.shopkeeper_locations = tf.placeholder(tf.int32, [self.batchSize, ], name="shopkeeper_locations")
        self.spatial_states = tf.placeholder(tf.int32, [self.batchSize, ], name="spatial_states")
        self.state_targets = tf.placeholder(tf.int32, [self.batchSize, ], name="state_targets")
        
        self.shopkeeper_action_id_targets_onehot = tf.one_hot(self.shopkeeper_action_id_targets, self.numOutputClasses)
        #self.cam_index_targets_onehot = tf.one_hot(self.cam_index_targets, self.numCameras+1) # add one for NONE
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
            action_decoder_input = self.input_encoding
            
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
            
            
            temp = tf.squeeze(tf.gather_nd(tf.expand_dims(self.speechClusterWeightsTensor, 1), tf.expand_dims(self.shopkeeper_action_id_targets, 1)))
            speechClusterLossWeights = tf.dtypes.cast(self.output_mask, tf.float32) * temp
            
            self.shopkeeper_speech_loss = tf.losses.softmax_cross_entropy(self.shopkeeper_action_id_targets_onehot, 
                                                                          self.shopkeeper_action_decoder,
                                                                          weights=speechClusterLossWeights,
                                                                          reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            
            
            self.loss = self.shopkeeper_speech_loss
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
    
    
    def get_batches(self, numSamples):
        # note - this will chop off data doesn't factor into a batch of batchSize
        
        batchStartEndIndices = []
        
        for endIndex in range(self.batchSize, numSamples, self.batchSize):
            batchStartEndIndices.append((endIndex-self.batchSize, endIndex))
        
        return batchStartEndIndices
    
    
    def train(self, 
              inputSequenceVectors,
              outputActionIds,
              outputLocations,
              outputSpatialStates,
              outputStateTargets,
              outputMasks):
        
        batchStartEndIndices = self.get_batches(len(outputActionIds))
        
        for sei in batchStartEndIndices:
            
            ###
            
            feedDict = {self.input_sequences: inputSequenceVectors[sei[0]:sei[1]], 
                        self.shopkeeper_action_id_targets: outputActionIds[sei[0]:sei[1]],
                        self.shopkeeper_locations: outputLocations[sei[0]:sei[1]],
                        self.spatial_states: outputSpatialStates[sei[0]:sei[1]],
                        self.state_targets: outputStateTargets[sei[0]:sei[1]],
                        self.output_mask: outputMasks[sei[0]:sei[1]],
                        self.speechClusterWeightsTensor: self.speechClusterWeights}
            
            _ = self.sess.run([self.train_op], feed_dict=feedDict)
            
            ###
        
        return None
    
    
    def get_loss(self, 
              inputSequenceVectors,
              outputActionIds,
              outputLocations,
              outputSpatialStates,
              outputStateTargets,
              outputMasks):
        
        batchStartEndIndices = self.get_batches(len(outputActionIds))
        
        all_loss = [] 
        all_shopkeeper_action_loss = []
        all_location_loss = []
        all_spatial_state_loss = [] 
        all_state_target_loss = []
        
        for sei in batchStartEndIndices:
            
            ###
            
            feedDict = {self.input_sequences: inputSequenceVectors[sei[0]:sei[1]], 
                        self.shopkeeper_action_id_targets: outputActionIds[sei[0]:sei[1]],
                        self.shopkeeper_locations: outputLocations[sei[0]:sei[1]],
                        self.spatial_states: outputSpatialStates[sei[0]:sei[1]],
                        self.state_targets: outputStateTargets[sei[0]:sei[1]],
                        self.output_mask: outputMasks[sei[0]:sei[1]],
                        self.speechClusterWeightsTensor: self.speechClusterWeights}
            
            loss, shopkeeper_action_loss, location_loss, spatial_state_loss, state_target_loss = self.sess.run([self.loss, self.shopkeeper_speech_loss, self.location_loss, self.spatial_state_loss, self.state_target_loss], 
                                                                                                                                                    feed_dict=feedDict)
            
            ###
            
            all_loss.append(loss) 
            all_shopkeeper_action_loss.append(shopkeeper_action_loss)
            all_location_loss.append(location_loss)
            all_spatial_state_loss.append(spatial_state_loss) 
            all_state_target_loss.append(state_target_loss)
            
        return all_loss, all_shopkeeper_action_loss, all_location_loss, all_spatial_state_loss, all_state_target_loss
    
    
    def predict(self, 
              inputSequenceVectors,
              outputActionIds,
              outputLocations,
              outputSpatialStates,
              outputStateTargets,
              outputMasks):
        
        batchStartEndIndices = self.get_batches(len(outputActionIds))
        
        all_predShkpActionID = []
        all_predLocations = [] 
        all_predSpatialStates = [] 
        all_predStateTargets = []
        
        for sei in batchStartEndIndices:
            
            ###
            
            feedDict = {self.input_sequences: inputSequenceVectors[sei[0]:sei[1]], 
                        self.shopkeeper_action_id_targets: outputActionIds[sei[0]:sei[1]],
                        self.shopkeeper_locations: outputLocations[sei[0]:sei[1]],
                        self.spatial_states: outputSpatialStates[sei[0]:sei[1]],
                        self.state_targets: outputStateTargets[sei[0]:sei[1]],
                        self.output_mask: outputMasks[sei[0]:sei[1]],
                        self.speechClusterWeightsTensor: self.speechClusterWeights}
            
            predShkpActionID, predLocations, predSpatialStates, predStateTargets = self.sess.run([self.shopkeeper_action_argmax,
                                                                                                  self.location_argmax,
                                                                                                  self.spatial_state_argmax,
                                                                                                  self.state_target_argmax], feedDict)
            
            ###
            
            all_predShkpActionID.append(predShkpActionID)
            all_predLocations.append(predLocations)
            all_predSpatialStates.append(predSpatialStates) 
            all_predStateTargets.append(predStateTargets)
            
        
        all_predShkpActionID = np.concatenate(all_predShkpActionID)
        all_predLocations = np.concatenate(all_predLocations)
        all_predSpatialStates = np.concatenate(all_predSpatialStates) 
        all_predStateTargets = np.concatenate(all_predStateTargets)
            
            
        return all_predShkpActionID, all_predLocations, all_predSpatialStates, all_predStateTargets
    
    
    def save(self, filename):
        self.saver.save(self.sess, filename)
    
    
    def load(self, filename):
        self.saver.restore(self.sess, filename)
    
    
    def reset_optimizer(self):    
        self.sess.run(self.reset_optimizer_op)



class CopynetBased(object):
    
    def __init__(self,
                 inputDim, 
                 inputSeqLen,         
                 vocabSize,
                 outputSeqLen,
                 dbSeqLen,
                 numCameras,
                 numAttributes,
                 numLocations,
                 numSpatialStates,
                 numStateTargets,
                 batchSize, 
                 embeddingSize,
                 seed,
                 wordToIndex):
        
        self.inputDim = inputDim
        self.inputSeqLen = inputSeqLen
        self.vocabSize = vocabSize
        self.outputSeqLen = outputSeqLen
        self.dbSeqLen = dbSeqLen
        self.numCameras = numCameras
        self.numAttributes = numAttributes
        self.numLocations = numLocations
        self.numSpatialStates = numSpatialStates
        self.numStateTargets = numStateTargets
        self.batchSize = batchSize
        self.embeddingSize = embeddingSize
        self.seed = seed
        self.wordToIndex = wordToIndex
        
        self.goToken = "<go>"
        
        
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        
        
        #
        # build the model
        #
        
        # inputs
        self.input_sequences = tf.placeholder(tf.float32, [self.batchSize, self.inputSeqLen, self.inputDim], name="input_sequences")
        self.database_sequences = tf.placeholder(tf.int32, [self.batchSize, 
                                                              self.numCameras, 
                                                              self.numAttributes, 
                                                              self.dbSeqLen, 
                                                              ], name="database_sequences")
        
        self.database_sequences_one_hot = tf.one_hot(self.database_sequences, depth=self.vocabSize)
        
        
        # targets
        self.shopkeeper_speech_sequence_targets = tf.placeholder(tf.int32, [self.batchSize, self.outputSeqLen, ], name="shopkeeper_speech_token_targets")
        self.shopkeeper_speech_sequence_lengths = tf.placeholder(tf.int32, [self.batchSize, ], name="shopkeeper_speech_sequence_lengths")
        self.shopkeeper_locations = tf.placeholder(tf.int32, [self.batchSize, ], name="shopkeeper_locations")
        self.spatial_states = tf.placeholder(tf.int32, [self.batchSize, ], name="spatial_states")
        self.state_targets = tf.placeholder(tf.int32, [self.batchSize, ], name="state_targets")
        
        self.shopkeeper_locations_one_hot = tf.one_hot(self.shopkeeper_locations, self.numLocations)
        self.spatial_states_one_hot = tf.one_hot(self.spatial_states, self.numSpatialStates)
        self.state_targets_one_hot = tf.one_hot(self.state_targets, self.numStateTargets)
        
        
        #
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
        
        
        
        # encode the DB contents
        with tf.variable_scope("db_encoding"):
            
            # bi-directional GRU
            num_units = [self.embeddingSize, self.embeddingSize]
            
            cells_fw = [tf.nn.rnn_cell.GRUCell(num_units=n, kernel_initializer=tf.initializers.glorot_normal()) for n in num_units]
            cells_bw = [tf.nn.rnn_cell.GRUCell(num_units=n, kernel_initializer=tf.initializers.glorot_normal()) for n in num_units]
            
            
            database_sequences_flat = tf.reshape(self.database_sequences_one_hot, (self.batchSize*self.numCameras*self.numAttributes, self.dbSeqLen, self.vocabSize))
            
            database_encodings_flat, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw,
                                                                                           cells_bw=cells_bw,
                                                                                           inputs=database_sequences_flat,
                                                                                           dtype=tf.float32)
            
            self.database_encodings = tf.reshape(database_encodings_flat, (self.batchSize, self.numCameras, self.numAttributes, self.dbSeqLen, self.embeddingSize*2))
        
        
        
        with tf.variable_scope("db_addressing/camera_index_decoder"):
            self.cam_index_decoder = tf.layers.dense(self.input_encoding, self.numCameras, activation=tf.nn.sigmoid, use_bias=True, kernel_initializer=tf.initializers.he_normal())
            #self.cam_index_softmax = tf.nn.softmax(self.cam_index_decoder)
            #self.cam_index_argmax = tf.argmax(self.cam_index_softmax, axis=1)
        
        
        
        with tf.variable_scope("db_addressing/attribute_index_decoder"):
            self.attr_index_decoder = tf.layers.dense(self.input_encoding, self.numAttributes, activation=tf.nn.sigmoid, use_bias=True, kernel_initializer=tf.initializers.he_normal())
            #self.attr_index_softmax = tf.nn.softmax(self.attr_index_decoder)
            #self.attr_index_argmax = tf.argmax(self.attr_index_softmax, axis=1)
        
        
        
        with tf.variable_scope("db_addressing/database_contents_selector"):
            # get the length of the most relevant DB contents
            # this is the weighted sum of DB content lens
            
            #
            self.database_encodings_most_relevant = self.database_encodings * tf.reshape(self.cam_index_decoder, (self.batchSize, self.numCameras, 1, 1, 1))
            self.database_encodings_most_relevant = self.database_encodings_most_relevant * tf.reshape(self.attr_index_decoder, (self.batchSize, 1, self.numAttributes, 1, 1))
            
            self.database_encodings_most_relevant = tf.reduce_sum(self.database_encodings_most_relevant, axis=1)
            self.database_encodings_most_relevant = tf.reduce_sum(self.database_encodings_most_relevant, axis=1)
            
            #
            self.database_sequences_most_relevant = self.database_sequences_one_hot * tf.reshape(self.cam_index_decoder, (self.batchSize, self.numCameras, 1, 1, 1))
            self.database_sequences_most_relevant = self.database_sequences_most_relevant * tf.reshape(self.attr_index_decoder, (self.batchSize, 1, self.numAttributes, 1, 1))
            
            self.database_sequences_most_relevant = tf.reduce_sum(self.database_sequences_most_relevant, axis=1)
            self.database_sequences_most_relevant = tf.reduce_sum(self.database_sequences_most_relevant, axis=1)
        
        
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
        
        
        #
        # setup the copynet for decoding shopkeeper speech
        #
        self.shopkeeper_speech_sequence_mask = tf.sequence_mask(self.shopkeeper_speech_sequence_lengths, self.outputSeqLen, dtype=tf.int32)
        
        self.copynet_cell = copynet.CopyNetWrapper3(cell=tf.nn.rnn_cell.GRUCell(self.embeddingSize, kernel_initializer=tf.initializers.glorot_normal()),
                                                   db_val_seq_encodings=self.database_encodings_most_relevant,
                                                   db_val_seq_ids=self.database_sequences_most_relevant,
                                                   vocab_size=self.vocabSize)
        
        self.decoder_initial_state = self.copynet_cell.zero_state(self.batchSize, tf.float32).clone(cell_state=self.input_encoding)
        
        
        # append start char on to beginning of outputs so they can be used for teacher forcing - i.e. as inputs to the coipynet decoder
        after_slice = tf.strided_slice(self.shopkeeper_speech_sequence_targets, [0, 0], [self.batchSize, -1], [1, 1]) # slice of the last char of each output sequence (is this necessary?)
        decoder_inputs = tf.concat( [tf.fill([self.batchSize, 1], self.wordToIndex[self.goToken]), after_slice], 1) # concatenate on a go char onto the start of each output sequence
        self.decoder_inputs_one_hot = tf.one_hot(decoder_inputs, self.vocabSize)
        
        
        # rollout the decoder two times - once for use with teacher forcing (training) and once without (testing)
        #self.teacher_forcing_prob = tf.placeholder(tf.float32, shape=(), name='teacher_forcing_prob')
        self.teacher_forcing_prob = tf.zeros((), tf.float32, name='teacher_forcing_prob')
        
        
        self.shopkeeper_speech_train_loss, self.shopkeeper_speech_sequence_train_preds, self.copy_scores_train, self.gen_scores_train = self.build_decoder(teacherForcing=True, scopeName="speech_decoder_train")
        self.shopkeeper_speech_test_loss, self.shopkeeper_speech_sequence_test_preds, self.copy_scores_test, self.gen_scores_test = self.build_decoder(teacherForcing=False, scopeName="speech_decoder_test")
        
        
        #
        # add up losses
        #
        with tf.variable_scope("loss"):
            
            self.location_loss = tf.losses.softmax_cross_entropy(self.shopkeeper_locations_one_hot, 
                                                                 self.location_decoder,
                                                                 reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            
            self.spatial_state_loss = tf.losses.softmax_cross_entropy(self.spatial_states_one_hot, 
                                                                 self.spatial_state_decoder,
                                                                 reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            
            self.state_target_loss = tf.losses.softmax_cross_entropy(self.state_targets_one_hot, 
                                                                 self.state_target_decoder,
                                                                 reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
        
        
        self.loss = self.shopkeeper_speech_train_loss
        self.loss += self.location_loss
        self.loss += self.spatial_state_loss
        self.loss += self.state_target_loss
        
        
        #
        # setup the training function
        #
        opt = tf.train.AdamOptimizer(learning_rate=3e-4)
        
        self.reset_optimizer_op = tf.variables_initializer(opt.variables())
        
        
        #
        # for training the entire network
        #
        gradients = opt.compute_gradients(self.loss)
        #tf.check_numerics(gradients, "gradients")
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        
        self.train_op_1 = opt.apply_gradients(capped_gradients, name="train_op")
        
        
        #
        # for only training the decoding part (and not the addressing part)
        #
        #decoding_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoding")
        #decoding_gradients = opt.compute_gradients(self.loss, var_list=decoding_train_vars)
        #decoding_capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in decoding_gradients if grad is not None]
        
        #self.train_op_2 = opt.apply_gradients(decoding_capped_gradients)
        
        
        self.train_op = self.train_op_1
        
        
        #
        # setup the prediction function
        #
        self.init_op = tf.initialize_all_variables()
        
        self.initialize()
        
        self.saver = tf.train.Saver()
        
        self.pred_utt_op = tf.argmax(self.shopkeeper_speech_sequence_test_preds, 2, name="predict_utterance_op")
    
    
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
                    bernoulliSample = tf.to_float(tf.distributions.Bernoulli(probs=self.teacher_forcing_prob).sample())
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
                ground_truth_output = tf.one_hot(self.shopkeeper_speech_sequence_targets[:, i], self.vocabSize) # these are one-hot char encodings at timestep i
                
                # compute the loss
                #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_output, logits=output)
                cross_entropy = tf.losses.softmax_cross_entropy(ground_truth_output, output, weights=self.shopkeeper_speech_sequence_mask[:,i], reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE) # is this right for sequence eval?
                #current_loss = tf.reduce_sum(cross_entropy)
                loss = loss + cross_entropy
            
            
            predicted_output_sequences = tf.concat(predicted_output_sequences, 1)
            copy_scores = tf.concat(copy_scores, 1)
            gen_scores = tf.concat(gen_scores, 1)
        
        return loss, predicted_output_sequences, copy_scores, gen_scores
    
    
    
    def initialize(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        self.sess = tf.Session(config=config)
        self.sess.run(self.init_op)
    
    
    def get_batches(self, numSamples):
        # note - this will chop off data doesn't factor into a batch of batchSize
        
        batchStartEndIndices = []
        
        for endIndex in range(self.batchSize, numSamples, self.batchSize):
            batchStartEndIndices.append((endIndex-self.batchSize, endIndex))
        
        return batchStartEndIndices
    
    
    def train(self, 
              inputSequenceVectors,
              outputSpeechSequences,
              outputSpeechSequenceLens,
              outputLocations,
              outputSpatialStates,
              outputStateTargets,
              databaseSequences):
        
        batchStartEndIndices = self.get_batches(len(outputSpeechSequences))
        
        all_loss = []
        all_shopkeeper_speech_loss = []
        all_location_loss = []
        all_spatial_state_loss = []
        all_state_target_loss = []
        
        for sei in batchStartEndIndices:
            
            ###            
            feedDict = {self.input_sequences: inputSequenceVectors[sei[0]:sei[1]],
                        self.shopkeeper_speech_sequence_targets: outputSpeechSequences[sei[0]:sei[1]],
                        self.shopkeeper_speech_sequence_lengths: outputSpeechSequenceLens[sei[0]:sei[1]],
                        self.shopkeeper_locations: outputLocations[sei[0]:sei[1]],
                        self.spatial_states: outputSpatialStates[sei[0]:sei[1]],
                        self.state_targets: outputStateTargets[sei[0]:sei[1]],
                        self.database_sequences: databaseSequences[sei[0]:sei[1]]
                        }
            
            loss, shopkeeper_action_loss, location_loss, spatial_state_loss, state_target_loss, _ = self.sess.run([self.loss, 
                                                                                                                   self.shopkeeper_speech_train_loss, 
                                                                                                                   self.location_loss, 
                                                                                                                   self.spatial_state_loss, 
                                                                                                                   self.state_target_loss, 
                                                                                                                   self.train_op],
                                                                                                                   feed_dict=feedDict)
            ###
            
            all_loss.append(loss)
            all_shopkeeper_speech_loss.append(shopkeeper_action_loss) 
            all_location_loss.append(location_loss)
            all_spatial_state_loss.append(spatial_state_loss) 
            all_state_target_loss.append(state_target_loss)
            
        return all_loss, all_shopkeeper_speech_loss, all_location_loss, all_spatial_state_loss, all_state_target_loss
    
    
    def get_loss(self, 
              inputSequenceVectors,
              outputSpeechSequences,
              outputSpeechSequenceLens,
              outputLocations,
              outputSpatialStates,
              outputStateTargets,
              databaseSequences):
        
        batchStartEndIndices = self.get_batches(len(outputSpeechSequences))
        
        all_loss = []
        all_shopkeeper_speech_loss = []
        all_location_loss = []
        all_spatial_state_loss = []
        all_state_target_loss = []
        
        for sei in batchStartEndIndices:
            
            ###            
            feedDict = {self.input_sequences: inputSequenceVectors[sei[0]:sei[1]],
                        self.shopkeeper_speech_sequence_targets: outputSpeechSequences[sei[0]:sei[1]],
                        self.shopkeeper_speech_sequence_lengths: outputSpeechSequenceLens[sei[0]:sei[1]],
                        self.shopkeeper_locations: outputLocations[sei[0]:sei[1]],
                        self.spatial_states: outputSpatialStates[sei[0]:sei[1]],
                        self.state_targets: outputStateTargets[sei[0]:sei[1]],
                        self.database_sequences: databaseSequences[sei[0]:sei[1]]
                        }
            
            loss, shopkeeper_action_loss, location_loss, spatial_state_loss, state_target_loss, = self.sess.run([self.loss, 
                                                                                                                   self.shopkeeper_speech_test_loss, 
                                                                                                                   self.location_loss, 
                                                                                                                   self.spatial_state_loss, 
                                                                                                                   self.state_target_loss],
                                                                                                                   feed_dict=feedDict)
            ###
            
            all_loss.append(loss)
            all_shopkeeper_speech_loss.append(shopkeeper_action_loss) 
            all_location_loss.append(location_loss)
            all_spatial_state_loss.append(spatial_state_loss) 
            all_state_target_loss.append(state_target_loss)
            
        return all_loss, all_shopkeeper_speech_loss, all_location_loss, all_spatial_state_loss, all_state_target_loss
    
    
    def predict(self, 
              inputSequenceVectors,
              outputSpeechSequences,
              outputSpeechSequenceLens,
              outputLocations,
              outputSpatialStates,
              outputStateTargets,
              databaseSequences):
        
        batchStartEndIndices = self.get_batches(len(outputSpeechSequences))
        
        all_predShkpSpeechSeqs = [] 
        all_predLocations = []
        all_predSpatialStates = [] 
        all_predStateTargets = []
        all_camIndexWeights = [] 
        all_attributeIndexWeights = []
        
        for sei in batchStartEndIndices:
            
            ###
            feedDict = {self.input_sequences: inputSequenceVectors[sei[0]:sei[1]],
                        self.shopkeeper_speech_sequence_targets: outputSpeechSequences[sei[0]:sei[1]],
                        self.shopkeeper_speech_sequence_lengths: outputSpeechSequenceLens[sei[0]:sei[1]],
                        self.shopkeeper_locations: outputLocations[sei[0]:sei[1]],
                        self.spatial_states: outputSpatialStates[sei[0]:sei[1]],
                        self.state_targets: outputStateTargets[sei[0]:sei[1]],
                        self.database_sequences: databaseSequences[sei[0]:sei[1]]
                        }
            
            predShkpSpeechSeqs, predLocations, predSpatialStates, predStateTargets, camIndexWeights, attributeIndexWeights = self.sess.run([
                        self.pred_utt_op,
                        self.location_argmax,
                        self.spatial_state_argmax,
                        self.state_target_argmax,
                        self.cam_index_decoder,
                        self.attr_index_decoder],
                        feed_dict=feedDict)
                        
            ###
            
            all_predShkpSpeechSeqs.append(predShkpSpeechSeqs)
            all_predLocations.append(predLocations)
            all_predSpatialStates.append(predSpatialStates) 
            all_predStateTargets.append(predStateTargets)
            all_camIndexWeights.append(camIndexWeights)
            all_attributeIndexWeights.append(attributeIndexWeights)
            
        
        all_predShkpSpeechSeqs = np.concatenate(all_predShkpSpeechSeqs)
        all_predLocations = np.concatenate(all_predLocations)
        all_predSpatialStates = np.concatenate(all_predSpatialStates) 
        all_predStateTargets = np.concatenate(all_predStateTargets)
        all_camIndexWeights = np.concatenate(all_camIndexWeights)
        all_attributeIndexWeights = np.concatenate(all_attributeIndexWeights)
        
        
        return all_predShkpSpeechSeqs, all_predLocations, all_predSpatialStates, all_predStateTargets, all_camIndexWeights, all_attributeIndexWeights
    
    
    def save(self, filename):
        self.saver.save(self.sess, filename)
    
    
    def load(self, filename):
        self.saver.restore(self.sess, filename)
    
    
    def reset_optimizer(self):    
        self.sess.run(self.reset_optimizer_op)



class CoreqaBased(object):
    
    def __init__(self,
                 inputDim, 
                 inputSeqLen,         
                 vocabSize,
                 outputSeqLen,
                 dbCamLen,
                 dbAttrLen,
                 dbValLen,
                 numDbFacts,
                 numLocations,
                 numSpatialStates,
                 numStateTargets,
                 batchSize, 
                 embeddingSize,
                 seed,
                 wordToIndex):
        
        self.inputDim = inputDim
        self.inputSeqLen = inputSeqLen
        self.vocabSize = vocabSize
        self.outputSeqLen = outputSeqLen
        self.dbCamLen = dbCamLen
        self.dbAttrLen = dbAttrLen
        self.dbValLen = dbValLen
        self.numDbFacts = numDbFacts
        self.numLocations = numLocations
        self.numSpatialStates = numSpatialStates
        self.numStateTargets = numStateTargets
        self.batchSize = batchSize
        self.embeddingSize = embeddingSize
        self.seed = seed
        self.wordToIndex = wordToIndex
        
        self.goToken = "<go>"
        
        
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        
        
        #
        # build the model
        #
        
        # inputs
        self.input_sequences = tf.placeholder(tf.float32, [self.batchSize, self.inputSeqLen, self.inputDim], name="input_sequences")
        
        self.database_cams = tf.placeholder(tf.int32, [self.batchSize,
                                                       self.numDbFacts,
                                                       self.dbCamLen,
                                                       ], name="database_cameras")
        
        self.database_attrs = tf.placeholder(tf.int32, [self.batchSize,
                                                        self.numDbFacts,
                                                        self.dbAttrLen,
                                                       ], name="database_attributes")
        
        self.database_vals = tf.placeholder(tf.int32, [self.batchSize,
                                                       self.numDbFacts,
                                                       self.dbValLen,
                                                       ], name="database_values")
        
        self.database_cams_one_hot = tf.one_hot(self.database_cams, depth=self.vocabSize)
        self.database_attrs_one_hot = tf.one_hot(self.database_attrs, depth=self.vocabSize)
        self.database_vals_one_hot = tf.one_hot(self.database_vals, depth=self.vocabSize)
        
        
        
        # targets
        self.shopkeeper_speech_sequence_targets = tf.placeholder(tf.int32, [self.batchSize, self.outputSeqLen, ], name="shopkeeper_speech_token_targets")
        self.shopkeeper_speech_sequence_lengths = tf.placeholder(tf.int32, [self.batchSize, ], name="shopkeeper_speech_sequence_lengths")
        self.shopkeeper_locations = tf.placeholder(tf.int32, [self.batchSize, ], name="shopkeeper_locations")
        self.spatial_states = tf.placeholder(tf.int32, [self.batchSize, ], name="spatial_states")
        self.state_targets = tf.placeholder(tf.int32, [self.batchSize, ], name="state_targets")
        
        self.shopkeeper_locations_one_hot = tf.one_hot(self.shopkeeper_locations, self.numLocations)
        self.spatial_states_one_hot = tf.one_hot(self.spatial_states, self.numSpatialStates)
        self.state_targets_one_hot = tf.one_hot(self.state_targets, self.numStateTargets)
        
        
        #
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
        
        
        
        # encode the DB contents
        self.dbFactEncodingSize = 0
        
        with tf.variable_scope("db_encoding/cam"):
            
            # bi-directional GRU
            num_units = [self.embeddingSize]
            
            cells_fw = [tf.nn.rnn_cell.GRUCell(num_units=n, kernel_initializer=tf.initializers.glorot_normal()) for n in num_units]
            cells_bw = [tf.nn.rnn_cell.GRUCell(num_units=n, kernel_initializer=tf.initializers.glorot_normal()) for n in num_units]
            
            
            database_sequences_flat = tf.reshape(self.database_cams_one_hot, (self.batchSize*self.numDbFacts, self.dbCamLen, self.vocabSize))
            
            _, database_encodings_fw, database_encodings_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw,
                                                                                                             cells_bw=cells_bw,
                                                                                                             inputs=database_sequences_flat,
                                                                                                             dtype=tf.float32)
            
            database_encodings_fw = database_encodings_fw[0]
            database_encodings_bw = database_encodings_bw[0]
            
            self.db_cam_encodings = tf.concat([database_encodings_fw, database_encodings_bw], axis=1)
            self.db_cam_encodings = tf.reshape(self.db_cam_encodings, (self.batchSize, self.numDbFacts, self.embeddingSize*2))
            self.dbFactEncodingSize += self.embeddingSize * 2
            
        
        with tf.variable_scope("db_encoding/attr"):
            
            # bi-directional GRU
            num_units = [self.embeddingSize]
            
            cells_fw = [tf.nn.rnn_cell.GRUCell(num_units=n, kernel_initializer=tf.initializers.glorot_normal()) for n in num_units]
            cells_bw = [tf.nn.rnn_cell.GRUCell(num_units=n, kernel_initializer=tf.initializers.glorot_normal()) for n in num_units]
            
            
            database_sequences_flat = tf.reshape(self.database_attrs_one_hot, (self.batchSize*self.numDbFacts, self.dbAttrLen, self.vocabSize))
            
            _, database_encodings_fw, database_encodings_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw,
                                                                                                             cells_bw=cells_bw,
                                                                                                             inputs=database_sequences_flat,
                                                                                                             dtype=tf.float32)
            
            database_encodings_fw = database_encodings_fw[0]
            database_encodings_bw = database_encodings_bw[0]
            
            self.db_attr_encodings = tf.concat([database_encodings_fw, database_encodings_bw], axis=1)
            self.db_attr_encodings = tf.reshape(self.db_attr_encodings, (self.batchSize, self.numDbFacts, self.embeddingSize*2))
            self.dbFactEncodingSize += self.embeddingSize * 2
            
        
        with tf.variable_scope("db_encoding/vals"):
            
            # bi-directional GRU
            num_units = [self.embeddingSize, self.embeddingSize]
            
            cells_fw = [tf.nn.rnn_cell.GRUCell(num_units=n, kernel_initializer=tf.initializers.glorot_normal()) for n in num_units]
            cells_bw = [tf.nn.rnn_cell.GRUCell(num_units=n, kernel_initializer=tf.initializers.glorot_normal()) for n in num_units]
            
            
            database_sequences_flat = tf.reshape(self.database_vals_one_hot, (self.batchSize*self.numDbFacts, self.dbValLen, self.vocabSize))
            
            database_encodings_flat, database_encodings_fw, database_encodings_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw,
                                                                                                             cells_bw=cells_bw,
                                                                                                             inputs=database_sequences_flat,
                                                                                                             dtype=tf.float32)
            
            database_encodings_fw_0 = database_encodings_fw[0]
            database_encodings_fw_1 = database_encodings_fw[1]
            database_encodings_bw_0 = database_encodings_bw[0]
            database_encodings_bw_1 = database_encodings_bw[1]
            
            self.db_val_encodings = tf.concat([database_encodings_fw_0, database_encodings_fw_1, database_encodings_bw_0, database_encodings_bw_1], axis=1)
            self.db_val_encodings = tf.reshape(self.db_val_encodings, (self.batchSize, self.numDbFacts, tf.shape(self.db_val_encodings)[-1]))
            self.dbFactEncodingSize += self.embeddingSize * 4
            
            self.db_val_seq_encodings = tf.reshape(database_encodings_flat, (self.batchSize, self.numDbFacts, self.dbValLen, self.embeddingSize*2))
                        
        
        with tf.variable_scope("db_addressing/database_contents_selector"):
            # get the most relevant DB contents
            
            # prepare DB fact encodings
            self.db_fact_encodings = tf.concat([self.db_cam_encodings, self.db_attr_encodings, self.db_val_encodings], axis=2)
            #self.db_fact_match_scores = tf.zeros((self.batchSize, self.numDbFacts, 1), tf.float32, name='dummy_db_fact_match_scores')
            
            
            """
            db_fact_encodings_reshaped = tf.reshape(self.db_fact_encodings, (self.batchSize*self.numDbFacts, tf.shape(self.db_fact_encodings)[-1]))
            
            # prepare input encodings
            input_encodings_reshaped = tf.keras.backend.repeat(self.input_encoding, self.numDbFacts)
            input_encodings_reshaped = tf.reshape(input_encodings_reshaped, (self.batchSize*self.numDbFacts, tf.shape(input_encodings_reshaped)[-1]))    
            
            # combine encodings
            self.combined_encodings_for_db_addressing = tf.concat([db_fact_encodings_reshaped, input_encodings_reshaped], axis=-1)
            
            # get the scores using both input and DB facts
            self.db_fact_match_scores = tf.layers.dense(self.combined_encodings_for_db_addressing, self.embeddingSize, activation=tf.nn.tanh, use_bias=True, kernel_initializer=tf.initializers.he_normal())
            self.db_fact_match_scores = tf.layers.dense(self.db_fact_match_scores, 1, activation=tf.nn.tanh, use_bias=True, kernel_initializer=tf.initializers.he_normal())
            self.db_fact_match_scores = tf.reshape(self.db_fact_match_scores, (self.batchSize, self.numDbFacts, 1))
            
            # apply the scores...
            
            # for the additive hist_DB
            self.db_match_fact_encoding = tf.multiply(self.db_fact_encodings, self.db_fact_match_scores)
            self.db_match_fact_encoding = tf.reduce_sum(self.db_match_fact_encoding, axis=1)
            
            # for copying from
            temp = tf.expand_dims(self.db_fact_match_scores, axis=-1)
            self.db_match_val_seq_encoding = tf.multiply(self.db_val_seq_encodings, temp)
            self.db_match_val_seq_encoding = tf.reduce_sum(self.db_match_val_seq_encoding, axis=1)
            
            temp = tf.expand_dims(self.db_fact_match_scores, axis=-1)
            self.db_match_val_seq = tf.multiply(self.database_vals_one_hot, temp)
            self.db_match_val_seq = tf.reduce_sum(self.db_match_val_seq, axis=1)
            """
        
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
        
        
        #
        # setup the copynet for decoding shopkeeper speech
        #
        self.shopkeeper_speech_sequence_mask = tf.sequence_mask(self.shopkeeper_speech_sequence_lengths, self.outputSeqLen, dtype=tf.int32)
        
        self.copynet_cell = copynet.CopyNetWrapper4(cell=tf.nn.rnn_cell.GRUCell(self.embeddingSize, kernel_initializer=tf.initializers.glorot_normal()),
                                                   db_val_seq_encodings=self.db_val_seq_encodings,
                                                   db_val_seq_ids=self.database_vals_one_hot,
                                                   interaction_state_encoding=self.input_encoding,
                                                   db_fact_encodings=self.db_fact_encodings,
                                                   vocab_size=self.vocabSize,
                                                   db_fact_encoding_size=self.dbFactEncodingSize)
        
        self.decoder_initial_state = self.copynet_cell.zero_state(self.batchSize, tf.float32).clone(cell_state=self.input_encoding)
        
        
        # append start char on to beginning of outputs so they can be used for teacher forcing - i.e. as inputs to the coipynet decoder
        after_slice = tf.strided_slice(self.shopkeeper_speech_sequence_targets, [0, 0], [self.batchSize, -1], [1, 1]) # slice of the last char of each output sequence (is this necessary?)
        decoder_inputs = tf.concat( [tf.fill([self.batchSize, 1], self.wordToIndex[self.goToken]), after_slice], 1) # concatenate on a go char onto the start of each output sequence
        self.decoder_inputs_one_hot = tf.one_hot(decoder_inputs, self.vocabSize)
        
        
        # rollout the decoder two times - once for use with teacher forcing (training) and once without (testing)
        #self.teacher_forcing_prob = tf.placeholder(tf.float32, shape=(), name='teacher_forcing_prob')
        self.teacher_forcing_prob = tf.zeros((), tf.float32, name='teacher_forcing_prob')
        
        
        self.shopkeeper_speech_train_loss, self.shopkeeper_speech_sequence_train_preds, self.copy_scores_train, self.gen_scores_train, self.ave_db_fact_match_scores_train = self.build_decoder_with_db_addressing(teacherForcing=True, scopeName="speech_decoder_train")
        self.shopkeeper_speech_test_loss, self.shopkeeper_speech_sequence_test_preds, self.copy_scores_test, self.gen_scores_test, self.ave_db_fact_match_scores_test = self.build_decoder_with_db_addressing(teacherForcing=False, scopeName="speech_decoder_test")
        
        
        #
        # add up losses
        #
        with tf.variable_scope("loss"):
            
            self.location_loss = tf.losses.softmax_cross_entropy(self.shopkeeper_locations_one_hot, 
                                                                 self.location_decoder,
                                                                 reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            
            self.spatial_state_loss = tf.losses.softmax_cross_entropy(self.spatial_states_one_hot, 
                                                                 self.spatial_state_decoder,
                                                                 reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            
            self.state_target_loss = tf.losses.softmax_cross_entropy(self.state_targets_one_hot, 
                                                                 self.state_target_decoder,
                                                                 reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
        
        
        self.loss = self.shopkeeper_speech_train_loss
        self.loss += self.location_loss
        self.loss += self.spatial_state_loss
        self.loss += self.state_target_loss
        
        
        #
        # setup the training function
        #
        opt = tf.train.AdamOptimizer(learning_rate=3e-4)
        
        self.reset_optimizer_op = tf.variables_initializer(opt.variables())
        
        
        #
        # for training the entire network
        #
        gradients = opt.compute_gradients(self.loss)
        #tf.check_numerics(gradients, "gradients")
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        
        self.train_op_1 = opt.apply_gradients(capped_gradients, name="train_op")
        
        
        #
        # for only training the decoding part (and not the addressing part)
        #
        #decoding_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoding")
        #decoding_gradients = opt.compute_gradients(self.loss, var_list=decoding_train_vars)
        #decoding_capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in decoding_gradients if grad is not None]
        
        #self.train_op_2 = opt.apply_gradients(decoding_capped_gradients)
        
        
        self.train_op = self.train_op_1
        
        
        #
        # setup the prediction function
        #
        self.init_op = tf.initialize_all_variables()
        
        self.initialize()
        
        self.saver = tf.train.Saver()
        
        self.pred_utt_op = tf.argmax(self.shopkeeper_speech_sequence_test_preds, 2, name="predict_utterance_op")
    
    
    
    def build_decoder_with_db_addressing(self, teacherForcing=False, scopeName="decoder"):
        
        
        with tf.variable_scope(scopeName):
            
            loss = 0
            predicted_output_sequences = []
            copy_scores = []
            gen_scores = []
            ave_db_fact_match_scores = tf.zeros((self.batchSize, self.numDbFacts, 1), tf.float32)
            
            state = self.decoder_initial_state
            output = self.decoder_inputs_one_hot[:, 0, :] # each output sequence must have a 'start' char appended to the beginning
            
            
            for i in range(self.outputSeqLen):
                
                if teacherForcing:
                    # if using teacher forcing
                    bernoulliSample = tf.to_float(tf.distributions.Bernoulli(probs=self.teacher_forcing_prob).sample())
                    output = tf.math.scalar_mul(bernoulliSample, self.decoder_inputs_one_hot[:, i, :]) + tf.math.scalar_mul((1.0-bernoulliSample), output)
                    output, state, copy_score, gen_score, db_fact_match_scores = self.copynet_cell(output, state)
                    
                    #output, state, copy_score, gen_score = self.copynet_cell(self.decoder_inputs_one_hot[:, i, :], state)
                    
                else:
                    # if not using teacher forcing
                    output, state, copy_score, gen_score, db_fact_match_scores = self.copynet_cell(output, state)
                
                
                predicted_output_sequences.append(tf.reshape(output, shape=(tf.shape(output)[0], 1, self.vocabSize)))
                copy_scores.append(tf.reshape(copy_score, shape=(tf.shape(copy_score)[0], 1, self.vocabSize)))
                gen_scores.append(tf.reshape(gen_score, shape=(tf.shape(gen_score)[0], 1, self.vocabSize)))
                
                m = tf.reshape(self.shopkeeper_speech_sequence_mask[:,i], (self.batchSize, 1, 1))
                db_fact_match_scores = tf.multiply(db_fact_match_scores, tf.cast(m, tf.float32))
                ave_db_fact_match_scores = tf.add(ave_db_fact_match_scores, db_fact_match_scores)
                
                
                # get the ground truth output
                ground_truth_output = tf.one_hot(self.shopkeeper_speech_sequence_targets[:, i], self.vocabSize) # these are one-hot char encodings at timestep i
                
                # compute the loss
                #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_output, logits=output)
                cross_entropy = tf.losses.softmax_cross_entropy(ground_truth_output, output, weights=self.shopkeeper_speech_sequence_mask[:,i], reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE) # is this right for sequence eval?
                #current_loss = tf.reduce_sum(cross_entropy)
                loss = loss + cross_entropy
            
            
            predicted_output_sequences = tf.concat(predicted_output_sequences, 1)
            copy_scores = tf.concat(copy_scores, 1)
            gen_scores = tf.concat(gen_scores, 1)
            
            mSum = tf.reshape(tf.reduce_sum(self.shopkeeper_speech_sequence_mask, axis=1), (self.batchSize, 1, 1))
            ave_db_fact_match_scores = tf.divide(ave_db_fact_match_scores, tf.cast(mSum, tf.float32))
            
        
        return loss, predicted_output_sequences, copy_scores, gen_scores, ave_db_fact_match_scores
    
    
    
    
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
                    bernoulliSample = tf.to_float(tf.distributions.Bernoulli(probs=self.teacher_forcing_prob).sample())
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
                ground_truth_output = tf.one_hot(self.shopkeeper_speech_sequence_targets[:, i], self.vocabSize) # these are one-hot char encodings at timestep i
                
                # compute the loss
                #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_output, logits=output)
                cross_entropy = tf.losses.softmax_cross_entropy(ground_truth_output, output, weights=self.shopkeeper_speech_sequence_mask[:,i], reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE) # is this right for sequence eval?
                #current_loss = tf.reduce_sum(cross_entropy)
                loss = loss + cross_entropy
            
            
            predicted_output_sequences = tf.concat(predicted_output_sequences, 1)
            copy_scores = tf.concat(copy_scores, 1)
            gen_scores = tf.concat(gen_scores, 1)
        
        return loss, predicted_output_sequences, copy_scores, gen_scores
    
    
    
    def initialize(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        self.sess = tf.Session(config=config)
        self.sess.run(self.init_op)
    
    
    def get_batches(self, numSamples):
        # note - this will chop off data doesn't factor into a batch of batchSize
        
        batchStartEndIndices = []
        
        for endIndex in range(self.batchSize, numSamples, self.batchSize):
            batchStartEndIndices.append((endIndex-self.batchSize, endIndex))
        
        return batchStartEndIndices
    
    
    def train(self, 
              inputSequenceVectors,
              outputSpeechSequences,
              outputSpeechSequenceLens,
              outputLocations,
              outputSpatialStates,
              outputStateTargets,
              databaseCams,
              databaseAttrs,
              databaseVals):
        
        batchStartEndIndices = self.get_batches(len(outputSpeechSequences))
        
        all_loss = []
        all_shopkeeper_speech_loss = []
        all_location_loss = []
        all_spatial_state_loss = []
        all_state_target_loss = []
        
        for sei in batchStartEndIndices:
            
            ###            
            feedDict = {self.input_sequences: inputSequenceVectors[sei[0]:sei[1]],
                        self.shopkeeper_speech_sequence_targets: outputSpeechSequences[sei[0]:sei[1]],
                        self.shopkeeper_speech_sequence_lengths: outputSpeechSequenceLens[sei[0]:sei[1]],
                        self.shopkeeper_locations: outputLocations[sei[0]:sei[1]],
                        self.spatial_states: outputSpatialStates[sei[0]:sei[1]],
                        self.state_targets: outputStateTargets[sei[0]:sei[1]],
                        self.database_cams: databaseCams[sei[0]:sei[1]],
                        self.database_attrs: databaseAttrs[sei[0]:sei[1]],
                        self.database_vals: databaseVals[sei[0]:sei[1]]
                        }
            
            loss, shopkeeper_action_loss, location_loss, spatial_state_loss, state_target_loss, _ = self.sess.run([self.loss, 
                                                                                                                   self.shopkeeper_speech_train_loss, 
                                                                                                                   self.location_loss, 
                                                                                                                   self.spatial_state_loss, 
                                                                                                                   self.state_target_loss, 
                                                                                                                   self.train_op],
                                                                                                                   feed_dict=feedDict)
            ###
            
            all_loss.append(loss)
            all_shopkeeper_speech_loss.append(shopkeeper_action_loss) 
            all_location_loss.append(location_loss)
            all_spatial_state_loss.append(spatial_state_loss) 
            all_state_target_loss.append(state_target_loss)
            
        return all_loss, all_shopkeeper_speech_loss, all_location_loss, all_spatial_state_loss, all_state_target_loss
    
    
    def get_loss(self, 
              inputSequenceVectors,
              outputSpeechSequences,
              outputSpeechSequenceLens,
              outputLocations,
              outputSpatialStates,
              outputStateTargets,
              databaseCams,
              databaseAttrs,
              databaseVals):
        
        batchStartEndIndices = self.get_batches(len(outputSpeechSequences))
        
        all_loss = []
        all_shopkeeper_speech_loss = []
        all_location_loss = []
        all_spatial_state_loss = []
        all_state_target_loss = []
        
        for sei in batchStartEndIndices:
            
            ###            
            feedDict = {self.input_sequences: inputSequenceVectors[sei[0]:sei[1]],
                        self.shopkeeper_speech_sequence_targets: outputSpeechSequences[sei[0]:sei[1]],
                        self.shopkeeper_speech_sequence_lengths: outputSpeechSequenceLens[sei[0]:sei[1]],
                        self.shopkeeper_locations: outputLocations[sei[0]:sei[1]],
                        self.spatial_states: outputSpatialStates[sei[0]:sei[1]],
                        self.state_targets: outputStateTargets[sei[0]:sei[1]],
                        self.database_cams: databaseCams[sei[0]:sei[1]],
                        self.database_attrs: databaseAttrs[sei[0]:sei[1]],
                        self.database_vals: databaseVals[sei[0]:sei[1]]
                        }
            
            loss, shopkeeper_action_loss, location_loss, spatial_state_loss, state_target_loss, = self.sess.run([self.loss, 
                                                                                                                   self.shopkeeper_speech_test_loss, 
                                                                                                                   self.location_loss, 
                                                                                                                   self.spatial_state_loss, 
                                                                                                                   self.state_target_loss],
                                                                                                                   feed_dict=feedDict)
            ###
            
            all_loss.append(loss)
            all_shopkeeper_speech_loss.append(shopkeeper_action_loss) 
            all_location_loss.append(location_loss)
            all_spatial_state_loss.append(spatial_state_loss) 
            all_state_target_loss.append(state_target_loss)
            
        return all_loss, all_shopkeeper_speech_loss, all_location_loss, all_spatial_state_loss, all_state_target_loss
    
    
    def predict(self, 
              inputSequenceVectors,
              outputSpeechSequences,
              outputSpeechSequenceLens,
              outputLocations,
              outputSpatialStates,
              outputStateTargets,
              databaseCams,
              databaseAttrs,
              databaseVals):
        
        batchStartEndIndices = self.get_batches(len(outputSpeechSequences))
        
        all_predShkpSpeechSeqs = [] 
        all_predLocations = []
        all_predSpatialStates = [] 
        all_predStateTargets = []
        all_dbFactMatchScores = []
        
        for sei in batchStartEndIndices:
            
            ###            
            feedDict = {self.input_sequences: inputSequenceVectors[sei[0]:sei[1]],
                        self.shopkeeper_speech_sequence_targets: outputSpeechSequences[sei[0]:sei[1]],
                        self.shopkeeper_speech_sequence_lengths: outputSpeechSequenceLens[sei[0]:sei[1]],
                        self.shopkeeper_locations: outputLocations[sei[0]:sei[1]],
                        self.spatial_states: outputSpatialStates[sei[0]:sei[1]],
                        self.state_targets: outputStateTargets[sei[0]:sei[1]],
                        self.database_cams: databaseCams[sei[0]:sei[1]],
                        self.database_attrs: databaseAttrs[sei[0]:sei[1]],
                        self.database_vals: databaseVals[sei[0]:sei[1]]
                        }
            
            predShkpSpeechSeqs, predLocations, predSpatialStates, predStateTargets, dbFactMatchScores = self.sess.run([
                        self.pred_utt_op,
                        self.location_argmax,
                        self.spatial_state_argmax,
                        self.state_target_argmax,
                        self.ave_db_fact_match_scores_test],
                        feed_dict=feedDict)
                        
            ###
            
            all_predShkpSpeechSeqs.append(predShkpSpeechSeqs)
            all_predLocations.append(predLocations)
            all_predSpatialStates.append(predSpatialStates) 
            all_predStateTargets.append(predStateTargets)
            all_dbFactMatchScores.append(dbFactMatchScores)
            
        
        all_predShkpSpeechSeqs = np.concatenate(all_predShkpSpeechSeqs)
        all_predLocations = np.concatenate(all_predLocations)
        all_predSpatialStates = np.concatenate(all_predSpatialStates) 
        all_predStateTargets = np.concatenate(all_predStateTargets)
        all_dbFactMatchScores = np.concatenate(all_dbFactMatchScores)
        
        
        return all_predShkpSpeechSeqs, all_predLocations, all_predSpatialStates, all_predStateTargets, all_dbFactMatchScores
    
    
    def save(self, filename):
        self.saver.save(self.sess, filename)
    
    
    def load(self, filename):
        self.saver.restore(self.sess, filename)
    
    
    def reset_optimizer(self):    
        self.sess.run(self.reset_optimizer_op)





