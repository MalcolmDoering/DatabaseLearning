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
                 batchSize, 
                 embeddingSize,
                 camTemp,
                 seed):
        
        self.inputDim = inputDim
        self.inputSeqLen = inputSeqLen
        self.numOutputClasses = numOutputClasses
        self.numUniqueCams = numUniqueCams
        self.batchSize = batchSize
        self.embeddingSize = embeddingSize
        self.camTemp = camTemp
        self.seed = seed
        
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        
        
        #
        # build the model
        #
        
        # inputs
        self.input_sequences = tf.placeholder(tf.float32, [self.batchSize, self.inputSeqLen, self.inputDim], name="input_sequences")
        
        
        # targets
        self.shopkeeper_action_id_targets = tf.placeholder(tf.int32, [self.batchSize, ], name="shopkeeper_action_id_targets")
        self.cam_index_targets = tf.placeholder(tf.int32, [self.batchSize, ], name="cam_index_targets")
        
        self.shopkeeper_action_id_targets_onehot = tf.one_hot(self.shopkeeper_action_id_targets, self.numOutputClasses)
        self.cam_index_targets_onehot = tf.one_hot(self.cam_index_targets, self.numUniqueCams+1) # add one for NONE
        
        
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
            
        
        with tf.variable_scope("camera_index_decoder"):
            self.cam_index_decoder = tf.layers.dense(self.input_encoding, self.numUniqueCams+1, activation=tf.nn.leaky_relu, use_bias=True, kernel_initializer=tf.initializers.he_normal())
            self.cam_index_softmax = tf.nn.softmax(self.cam_index_decoder)
            self.cam_index_argmax = tf.argmax(self.cam_index_softmax, axis=1)
        
        
        with tf.variable_scope("shopkeeper_action_decoder"):
            # is one layer enough?
            self.shopkeeper_action_decoder = tf.layers.dense(self.input_encoding, self.numOutputClasses, activation=tf.nn.leaky_relu, use_bias=True, kernel_initializer=tf.initializers.he_normal())
            self.shopkeeper_action_softmax = tf.nn.softmax(self.shopkeeper_action_decoder)
            self.shopkeeper_action_argmax = tf.argmax(self.shopkeeper_action_softmax, axis=1)
        
        
        with tf.variable_scope("loss"):
            self.camera_index_loss = tf.losses.softmax_cross_entropy(self.cam_index_targets_onehot, 
                                                                     self.cam_index_decoder,
                                                                     weights=self.output_mask,
                                                                     reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            
            self.shopkeeper_action_loss = tf.losses.softmax_cross_entropy(self.shopkeeper_action_id_targets_onehot, 
                                                                          self.shopkeeper_action_decoder,
                                                                          weights=self.output_mask,
                                                                          reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            
            self.loss = self.camera_index_loss + self.shopkeeper_action_loss
        
        
        #
        # setup the training function
        #
        opt = tf.train.AdamOptimizer(learning_rate=1e-3)
        #opt = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
        
        self.reset_optimizer_op = tf.variables_initializer(opt.variables())
        
        
        #
        # for training the entire network
        #
        gradients = opt.compute_gradients(self.loss)
        #tf.check_numerics(gradients, "gradients")
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        
        self.train_op = opt.apply_gradients(capped_gradients, name="train_op")
        
        
        #
        # setup the prediction function
        #
        self.pred_shopkeeper_action = self.shopkeeper_action_argmax
        self.pred_camera_index = self.cam_index_argmax
        
        
        self.init_op = tf.initialize_all_variables()
        
        self.initialize()
        
        self.saver = tf.train.Saver()
    
    
    
    def initialize(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        self.sess = tf.Session(config=config)
        self.sess.run(self.init_op)
        
    
    def train(self, 
              inputSequenceVectors,
              outputActionIds,
              outputCameraIndices,
              outputMasks):
        
        feedDict = {self.input_sequences: inputSequenceVectors, 
                    self.shopkeeper_action_id_targets: outputActionIds, 
                    self.cam_index_targets: outputCameraIndices,
                    self.output_mask: outputMasks}
        
        trainingLoss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feedDict)
        
        return trainingLoss
    
    
    def get_loss(self, 
             inputSequenceVectors,
             outputActionIds,
             outputCameraIndices,
             outputMasks):
        
        feedDict = {self.input_sequences: inputSequenceVectors, 
                    self.shopkeeper_action_id_targets: outputActionIds, 
                    self.cam_index_targets: outputCameraIndices,
                    self.output_mask: outputMasks}
        
        loss = self.sess.run(self.loss, feed_dict=feedDict)
        
        return loss
    
    
    def predict(self, 
                inputSequenceVectors,
                outputActionIds,
                outputCameraIndices):
        
        
        outputMasks = [1] * len(outputActionIds)
        
        feedDict = {self.input_sequences: inputSequenceVectors, 
                    self.shopkeeper_action_id_targets: outputActionIds, 
                    self.cam_index_targets: outputCameraIndices,
                    self.output_mask: outputMasks}
        
        predShkpActionID, predCameraIndices = self.sess.run([self.shopkeeper_action_argmax, self.cam_index_argmax], feedDict)
        
        return predShkpActionID, predCameraIndices
    
    
    def save(self, filename):
        self.saver.save(self.sess, filename)
    
    
    def load(self, filename):
        self.saver.restore(self.sess, filename)
    
    
    def reset_optimizer(self):    
        self.sess.run(self.reset_optimizer_op)
        
    
        