

import collections

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.framework.python.framework import tensor_util
#import tensorflow_probability as tfp



class CopyNetWrapperState(
    collections.namedtuple("CopyNetWrapperState", ("cell_state", "last_ids", "prob_c"))):

    def clone(self, **kwargs):
        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, tf.Tensor) and isinstance(new, tf.Tensor):
                return tensor_util.with_same_shape(old, new)
            return new

        return nest.map_structure(
            with_same_shape,
            self,
            super(CopyNetWrapperState, self)._replace(**kwargs))



class CopyNetWrapper(tf.nn.rnn_cell.RNNCell):
    
    def __init__(self, 
                 cell, 
                 encoder_states, 
                 encoder_input_ids, 
                 vocab_size,
                 gen_vocab_size=None, 
                 encoder_state_size=None, 
                 initial_cell_state=None, 
                 name=None):
        """
        Args:
            cell:
            encoder_states:
            encoder_input_ids:
            tgt_vocab_size:
            gen_vocab_size:
            encoder_state_size:
            initial_cell_state:
        """
        
        super(CopyNetWrapper, self).__init__(name=name)
        
        self._cell = cell
        self._vocab_size = vocab_size
        self._gen_vocab_size = gen_vocab_size or vocab_size
        self._encoder_input_ids = encoder_input_ids
        self._encoder_states = encoder_states
        
        
        if encoder_state_size is None:
            encoder_state_size = self._encoder_states.shape[-1].value
        
            if encoder_state_size is None:
                raise ValueError("encoder_state_size must be set if we can't infer encoder_states last dimension size.")
        
        self._encoder_state_size = encoder_state_size
        
        self._initial_cell_state = initial_cell_state
        
        self._copy_weight = tf.get_variable('CopyWeight', [self._encoder_state_size , self._cell.output_size])
        
        self._projection = tf.layers.Dense(self._gen_vocab_size, use_bias=False, name="OutputProjection")
        
    
        
    def __call__(self, inputs, state, scope=None):
        
        if not isinstance(state, CopyNetWrapperState):
            raise TypeError("Expected state to be instance of CopyNetWrapperState. "
                      "Received type %s instead."  % type(state))
        
        last_ids = state.last_ids
        prob_c = state.prob_c
        cell_state = state.cell_state
        
        # find places in the input where the previous char is the same as the char previously output by copynet
        mask = tf.cast(tf.equal(tf.expand_dims(last_ids, 1),  self._encoder_input_ids), tf.float32) 
        mask_sum = tf.reduce_sum(mask, axis=1)
        mask = tf.where(tf.less(mask_sum, 1e-7), mask, mask / tf.expand_dims(mask_sum, 1))
        
        # compute selective read
        rou = mask * prob_c
        selective_read = tf.einsum("ijk,ij->ik", self._encoder_states, rou)
        
        # setup inputs
        #selective_read = tf.zeros(selective_read.shape) # for testing without copy score
        inputs = tf.concat([inputs, selective_read], 1)
        
        # compute outputs
        outputs, cell_state = self._cell(inputs, cell_state, scope)
        generate_score = self._projection(outputs)
        
        # compute copy score
        copy_score = tf.einsum("ijk,km->ijm", self._encoder_states, self._copy_weight)
        copy_score = tf.nn.tanh(copy_score)
        copy_score = tf.einsum("ijm,im->ij", copy_score, outputs)
        
        # ?
        encoder_input_mask = tf.one_hot(self._encoder_input_ids, self._vocab_size)
        expanded_copy_score = tf.einsum("ijn,ij->ij", encoder_input_mask, copy_score)
        
        
        prob_g = generate_score
        prob_c = expanded_copy_score
        #mixed_score = tf.concat([generate_score, expanded_copy_score], 1)
        #probs = tf.nn.softmax(mixed_score)
        #prob_g = probs[:, :self._gen_vocab_size]
        #prob_c = probs[:, self._gen_vocab_size:]
        
        
        # compute final output by summing generate and copy char probabilities
        prob_c_one_hot = tf.einsum("ijn,ij->in", encoder_input_mask, prob_c)
        prob_g_total = tf.pad(prob_g, [[0, 0], [0, self._vocab_size - self._gen_vocab_size]])
        outputs = prob_g_total + prob_c_one_hot
        
        last_ids = tf.argmax(outputs, axis=-1, output_type=tf.int32)
        #prob_c.set_shape([None, self._encoder_state_size])
        
        state = CopyNetWrapperState(cell_state=cell_state, last_ids=last_ids, prob_c=prob_c)
        
        return outputs, state, prob_c_one_hot, prob_g_total
    
    
    
    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.

            It can be represented by an Integer, a TensorShape or a tuple of Integers
            or TensorShapes.
        """
        return CopyNetWrapperState(cell_state=self._cell.state_size, last_ids=tf.TensorShape([]),
            prob_c = self._encoder_state_size)
    
    
    
    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        return self._vocab_size
    
    
    
    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            last_ids = tf.zeros([batch_size], tf.int32) - 1
            prob_c = tf.zeros([batch_size, tf.shape(self._encoder_states)[1]], tf.float32)
            return CopyNetWrapperState(cell_state=cell_state, last_ids=last_ids, prob_c=prob_c)



class CopyNetWrapper2(tf.nn.rnn_cell.RNNCell):
    """Allow encoder_input_ids to be sequences of one-hot encodings instead of sequences of char IDs (ints)."""
    
    def __init__(self, 
                 cell, 
                 encoder_states, 
                 encoder_input_ids, 
                 vocab_size,
                 gen_vocab_size=None, 
                 encoder_state_size=None, 
                 initial_cell_state=None, 
                 name=None):
        """
        Args:
            cell:
            encoder_states:
            encoder_input_ids:
            tgt_vocab_size:
            gen_vocab_size:
            encoder_state_size:
            initial_cell_state:
        """
        
        super(CopyNetWrapper2, self).__init__(name=name)
        
        self._cell = cell
        self._vocab_size = vocab_size
        self._gen_vocab_size = gen_vocab_size or vocab_size
        self._encoder_input_ids = encoder_input_ids
        self._encoder_states = encoder_states
        
        
        if encoder_state_size is None:
            encoder_state_size = self._encoder_states.shape[-1].value
        
            if encoder_state_size is None:
                raise ValueError("encoder_state_size must be set if we can't infer encoder_states last dimension size.")
        
        self._encoder_state_size = encoder_state_size
        
        self._initial_cell_state = initial_cell_state
        
        self._copy_weight = tf.get_variable('CopyWeight', [self._encoder_state_size , self._cell.output_size])
        
        self._projection = tf.layers.Dense(self._gen_vocab_size, use_bias=False, name="OutputProjection")
        
    
        
    def __call__(self, inputs, state, scope=None):
        
        if not isinstance(state, CopyNetWrapperState):
            raise TypeError("Expected state to be instance of CopyNetWrapperState. "
                      "Received type %s instead."  % type(state))
        
        last_ids = state.last_ids
        prob_c = state.prob_c
        cell_state = state.cell_state
        
        
        # find places in the input where the previous char is the same as the char previously output by copynet
        # how should this be modified to deal with the weighted summed one hot char encoding inputs???
        
        #mask = tf.cast(tf.equal(tf.expand_dims(last_ids, 1),  self._encoder_input_ids), tf.float32)
        
        mask = tf.multiply(tf.expand_dims(last_ids, 1),  self._encoder_input_ids) 
        mask = tf.reduce_sum(mask, 2)
        
        mask_sum = tf.reduce_sum(mask, axis=1)
        mask = tf.where(tf.less(mask_sum, 1e-7), mask, mask / tf.expand_dims(mask_sum, 1))
        
        
        
        """
        self._encoder_input_ids = tf.check_numerics(self._encoder_input_ids, "copynet self._encoder_input_ids")
        #last_ids = tf.check_numerics(last_ids, "copynet last_ids")
        
        last_ids_one_hot = tf.one_hot(last_ids, self._vocab_size)
        #last_ids_one_hot = tf.ones(last_ids_one_hot.shape) # to cehck if it's the one-hot function causing the NaNs
        
        last_ids_one_hot = tfp.distributions.RelaxedOneHotCategorical(temperature=0.1, logits=last_ids)
        
        
        # weighted sum input should only remain in places where the input char is the same as the previous copynet output char
        mask = tf.multiply(tf.expand_dims(last_ids_one_hot, 1),  self._encoder_input_ids) 
        mask = tf.reduce_sum(mask, 2)
        #mask = tf.check_numerics(mask, "copynet mask1")
        
        mask_sum = tf.reduce_sum(mask, axis=1)
        #mask_sum = tf.check_numerics(mask_sum, "copynet mask_sum")
        
        mask = tf.where(tf.less(mask_sum, 1e-7), mask, mask / tf.expand_dims(mask_sum, 1)) # getting a NaN here?
        #mask = tf.check_numerics(mask, "copynet mask 2")
        """
        
        
        # compute selective read
        rou = mask * prob_c
        selective_read = tf.einsum("ijk,ij->ik", self._encoder_states, rou)
        
        # setup inputs
        inputs = tf.concat([inputs, selective_read], 1)
        
        # compute outputs
        outputs, cell_state = self._cell(inputs, cell_state, scope)
        generate_score = self._projection(outputs)
        
        # compute copy score
        copy_score = tf.einsum("ijk,km->ijm", self._encoder_states, self._copy_weight)
        copy_score = tf.nn.tanh(copy_score)
        copy_score = tf.einsum("ijm,im->ij", copy_score, outputs)
        
        # ?
        encoder_input_mask = self._encoder_input_ids #tf.one_hot(self._encoder_input_ids, self._vocab_size)
        expanded_copy_score = tf.einsum("ijn,ij->ij", encoder_input_mask, copy_score)
        
        
        prob_g = generate_score
        prob_c = expanded_copy_score
        #mixed_score = tf.concat([generate_score, expanded_copy_score], 1)
        #probs = tf.nn.softmax(mixed_score)
        #prob_g = probs[:, :self._gen_vocab_size]
        #prob_c = probs[:, self._gen_vocab_size:]
        
        
        # compute final output by summing generate and copy char probabilities
        prob_c_one_hot = tf.einsum("ijn,ij->in", encoder_input_mask, prob_c)
        prob_g_total = tf.pad(prob_g, [[0, 0], [0, self._vocab_size - self._gen_vocab_size]])
        outputs = prob_g_total #+ prob_c_one_hot
        
        last_ids = tf.argmax(outputs, axis=-1, output_type=tf.int32) # char ID
        last_ids = outputs # "one-hot"
        
        #prob_c.set_shape([None, self._encoder_state_size])
        
        state = CopyNetWrapperState(cell_state=cell_state, last_ids=last_ids, prob_c=prob_c)
        
        return outputs, state, prob_c_one_hot, prob_g_total
    
    
    
    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.

            It can be represented by an Integer, a TensorShape or a tuple of Integers
            or TensorShapes.
        """
        return CopyNetWrapperState(cell_state=self._cell.state_size, last_ids=tf.TensorShape([]),
            prob_c = self._encoder_state_size)
    
    
    
    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        return self._vocab_size
    
    
    
    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            
            #last_ids = tf.zeros([batch_size], tf.int32) - 1 # char ID
            last_ids = tf.zeros([batch_size, self._vocab_size], tf.float32) # "one-hot"
            
            prob_c = tf.zeros([batch_size, tf.shape(self._encoder_states)[1]], tf.float32)
            return CopyNetWrapperState(cell_state=cell_state, last_ids=last_ids, prob_c=prob_c)



class CopyNetWrapper3(tf.nn.rnn_cell.RNNCell):
    
    def __init__(self, 
                 cell, 
                 encoder_states, 
                 encoder_input_ids, 
                 vocab_size,
                 gen_vocab_size=None, 
                 encoder_state_size=None, 
                 initial_cell_state=None, 
                 name=None):
        """
        Args:
            cell:
            encoder_states:
            encoder_input_ids:
            tgt_vocab_size:
            gen_vocab_size:
            encoder_state_size:
            initial_cell_state:
        """
        
        super(CopyNetWrapper3, self).__init__(name=name)
        
        self._cell = cell
        self._vocab_size = vocab_size
        self._gen_vocab_size = gen_vocab_size or vocab_size
        self._encoder_input_ids = encoder_input_ids
        self._encoder_states = encoder_states
        
        
        if encoder_state_size is None:
            encoder_state_size = self._encoder_states.shape[-1].value
        
            if encoder_state_size is None:
                raise ValueError("encoder_state_size must be set if we can't infer encoder_states last dimension size.")
        
        self._encoder_state_size = encoder_state_size
        
        self._initial_cell_state = initial_cell_state
        
        self._copy_weight = tf.get_variable('CopyWeight', [self._encoder_state_size , self._cell.output_size])
        
        self._projection = tf.layers.Dense(self._gen_vocab_size, use_bias=False, name="OutputProjection")
        
    
        
    def __call_old__(self, inputs, state, scope=None):
        
        if not isinstance(state, CopyNetWrapperState):
            raise TypeError("Expected state to be instance of CopyNetWrapperState. "
                      "Received type %s instead."  % type(state))
        
        last_ids = state.last_ids # why not replace this with argmax of the current input??? This would be necessary for teacher forcing
        prob_c = state.prob_c
        cell_state = state.cell_state
        
        # find places in the input where the previous char is the same as the char previously output by copynet
        last_ids_one_hot = tf.one_hot(last_ids, self._vocab_size)
        
        #last_ids_based_on_input = tf.argmax(inputs, axis=1)
        #last_ids_one_hot = tf.one_hot(last_ids_based_on_input, self._vocab_size)
        
        
        
        #mask = tf.cast(tf.equal(tf.expand_dims(last_ids, 1),  self._encoder_input_ids), tf.float32) 
        #mask = tf.cast(tf.equal(tf.expand_dims(last_ids_one_hot, 1),  self._encoder_input_ids), tf.float32)
        mask = tf.expand_dims(last_ids_one_hot, 1) * self._encoder_input_ids
        
        
        mask = tf.reduce_mean(mask, axis=2)
        
        #mask_sum = tf.reduce_sum(mask, axis=2)
        #mask = tf.where(tf.less(mask_sum, 1e-7), mask, mask / tf.expand_dims(mask_sum, 1))
        
        # compute selective read
        rou = mask * prob_c
        selective_read = tf.einsum("ijk,ij->ik", self._encoder_states, rou)
        
        
        
        # setup inputs
        #selective_read = tf.zeros(selective_read.shape) # for testing without copy score
        inputs = tf.concat([inputs, selective_read], 1)
        
        # compute outputs
        outputs, cell_state = self._cell(inputs, cell_state, scope)
        generate_score = self._projection(outputs)
        
        # compute copy score
        copy_score = tf.einsum("ijk,km->ijm", self._encoder_states, self._copy_weight)
        copy_score = tf.nn.tanh(copy_score)
        copy_score = tf.einsum("ijm,im->ij", copy_score, outputs)
        
        # ?
        #encoder_input_mask = tf.one_hot(self._encoder_input_ids, self._vocab_size)
        #expanded_copy_score = tf.einsum("ijn,ij->ij", encoder_input_mask, copy_score)
        expanded_copy_score = tf.einsum("ijn,ij->ij", self._encoder_input_ids, copy_score)
        
        
        #self._normalized_encoder_input_ids = 
        
        
        
        prob_g = generate_score
        prob_c = expanded_copy_score
        #mixed_score = tf.concat([generate_score, expanded_copy_score], 1)
        #probs = tf.nn.softmax(mixed_score)
        #prob_g = probs[:, :self._gen_vocab_size]
        #prob_c = probs[:, self._gen_vocab_size:]
        
        
        # compute final output by summing generate and copy char probabilities
        #prob_c_one_hot = tf.einsum("ijn,ij->in", encoder_input_mask, prob_c)
        prob_c_one_hot = tf.einsum("ijn,ij->in", self._encoder_input_ids, prob_c)
        
        prob_g_total = tf.pad(prob_g, [[0, 0], [0, self._vocab_size - self._gen_vocab_size]])
        outputs = prob_g_total + prob_c_one_hot
        #outputs = prob_c_one_hot
        
        
        
        last_ids = tf.argmax(outputs, axis=-1, output_type=tf.int32)
        #prob_c.set_shape([None, self._encoder_state_size])
        
        state = CopyNetWrapperState(cell_state=cell_state, last_ids=last_ids, prob_c=prob_c)
        
        return outputs, state, prob_c_one_hot, prob_g_total
    
    
    
    def __call__(self, inputs, state, scope=None):
        
        if not isinstance(state, CopyNetWrapperState):
            raise TypeError("Expected state to be instance of CopyNetWrapperState. "
                      "Received type %s instead."  % type(state))
        
        last_ids = state.last_ids # why not replace this with argmax of the current input??? This would be necessary for teacher forcing
        prob_c = state.prob_c
        cell_state = state.cell_state
        
        
        #
        # compute selective read
        #
        
        # find places in the input where the previous char is the same as the char previously output by copynet
        last_ids_one_hot = tf.one_hot(last_ids, self._vocab_size)
        
        mask1 = tf.expand_dims(last_ids_one_hot, 1) * self._encoder_input_ids
        mask2 = tf.reduce_mean(mask1, axis=2)
        
        rou = mask2 * prob_c
        selective_read = tf.einsum("ijk,ij->ik", self._encoder_states, rou)
        
        
        #
        # get the next decoder RNN state and output
        #
        inputs = tf.concat([inputs, selective_read], 1)
        outputs, cell_state = self._cell(inputs, cell_state, scope)
        
        
        #
        # compute the generate score
        #
        generate_score = self._projection(outputs)
        prob_g = generate_score
        
        
        #        
        # compute the copy score
        #
        
        # multiply copy weights by the hidden state in M (DB entry)
        hWc = tf.einsum("ijk,km->ijm", self._encoder_states, self._copy_weight)
        
        # tanh
        tanh_hWc = tf.nn.tanh(hWc)
        
        # multiply by the current state of the decoder RNN
        # this give the copy score for each step in M (DB entry)
        copy_score = tf.einsum("ijm,im->ij", tanh_hWc, outputs)
        
        copy_score_per_char_per_DB_step = tf.expand_dims(copy_score, axis=2) * self._encoder_input_ids
        
        prob_c = tf.reduce_sum(copy_score_per_char_per_DB_step, axis=1)
        
        #char_counts = tf.reduce_sum(self._encoder_input_ids, axis=1) # count how many times each char occurs in M (DB entry)
        #prob_c = prob_c / (char_counts + 1e-8) # optional step, divide the copy scores for each char by the char count so that chars don't get a higher copy score just because they occur more frequently
        
        
        
        #
        # combine the copy and generate scores to compute the final output
        #
        outputs = prob_c + prob_g
        
        last_ids = tf.argmax(outputs, axis=-1, output_type=tf.int32)
        
        state = CopyNetWrapperState(cell_state=cell_state, last_ids=last_ids, prob_c=copy_score)
        
        
        return outputs, state, prob_c, prob_g
    
    
    
    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.

            It can be represented by an Integer, a TensorShape or a tuple of Integers
            or TensorShapes.
        """
        return CopyNetWrapperState(cell_state=self._cell.state_size, last_ids=tf.TensorShape([]),
            prob_c = self._encoder_state_size)
    
    
    
    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        return self._vocab_size
    
    
    
    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            last_ids = tf.zeros([batch_size], tf.int32) - 1
            prob_c = tf.zeros([batch_size, tf.shape(self._encoder_states)[1]], tf.float32)
            return CopyNetWrapperState(cell_state=cell_state, last_ids=last_ids, prob_c=prob_c)



class CopyNetWrapper4(tf.nn.rnn_cell.RNNCell):
    
    def __init__(self, 
                 cell, 
                 db_val_seq_encodings, 
                 db_val_seq_ids,
                 interaction_state_encoding,
                 db_fact_encodings,
                 vocab_size,
                 db_fact_encoding_size,
                 gen_vocab_size=None, 
                 encoder_state_size=None, 
                 initial_cell_state=None, 
                 name=None):
        
        super(CopyNetWrapper4, self).__init__(name=name)
        
        self._cell = cell
        self._vocab_size = vocab_size
        self._db_fact_encoding_size = db_fact_encoding_size
        self._gen_vocab_size = gen_vocab_size or vocab_size
        self._db_val_seq_ids = db_val_seq_ids
        self._db_val_seq_encodings = db_val_seq_encodings
        self._interaction_state_encoding = interaction_state_encoding
        self._db_fact_encodings = db_fact_encodings
        
        if encoder_state_size is None:
            encoder_state_size = self._db_val_seq_encodings.shape[-1].value
        
            if encoder_state_size is None:
                raise ValueError("encoder_state_size must be set if we can't infer encoder_states last dimension size.")
        
        self._encoder_state_size = encoder_state_size
        self._initial_cell_state = initial_cell_state
        
        self.batchSize = tf.shape(db_val_seq_encodings)[0]
        self.numDbFacts = tf.shape(db_val_seq_encodings)[1]
        
        
        self.db_match_fact_encoding_hist = tf.zeros((self.batchSize, self._db_fact_encoding_size), tf.float32, name='teacher_forcing_prob')
        
        
        self._copy_weight = tf.get_variable('CopyWeight', [self._encoder_state_size , self._cell.output_size + self._db_fact_encoding_size], trainable=True, initializer=tf.initializers.he_normal())
        #self._copy_weight = tf.layers.Dense(self._cell.output_size, activation=tf.nn.tanh, use_bias=True, kernel_initializer=tf.initializers.he_normal(), name="CopyWeight")
        
        
        self._projection = tf.layers.Dense(self._gen_vocab_size, use_bias=False, name="OutputProjection")
        
        self._db_fact_score_layer = tf.layers.Dense(1, activation=tf.nn.tanh, use_bias=True, kernel_initializer=tf.initializers.he_normal(), name="DBFactScoreLayer")
        # TOOD: use two layers for matching...
        
        # for computing db fact scores
        self.db_fact_encodings_reshaped = tf.reshape(self._db_fact_encodings, (self.batchSize*self.numDbFacts, tf.shape(self._db_fact_encodings)[-1]))
        
        self.interaction_state_encoding_reshaped = tf.keras.backend.repeat(self._interaction_state_encoding, self.numDbFacts)
        self.interaction_state_encoding_reshaped = tf.reshape(self.interaction_state_encoding_reshaped, (self.batchSize*self.numDbFacts, tf.shape(self.interaction_state_encoding_reshaped)[-1]))    
        
            
        
    def __call__(self, inputs, state, scope=None):
        
        if not isinstance(state, CopyNetWrapperState):
            raise TypeError("Expected state to be instance of CopyNetWrapperState. "
                      "Received type %s instead."  % type(state))
        
        last_ids = state.last_ids # why not replace this with argmax of the current input??? This would be necessary for teacher forcing
        prob_c = state.prob_c
        cell_state = state.cell_state
        
        
        #
        # find the DB fact with the highest score
        #
        
        # combine DB fact encodings, interaction state encoding, and decoder state encoding
        cell_state_reshaped = tf.keras.backend.repeat(cell_state, self.numDbFacts)
        cell_state_reshaped = tf.reshape(cell_state_reshaped, (self.batchSize*self.numDbFacts, tf.shape(cell_state_reshaped)[-1]))    
        
        combined_db_fact_score_inputs = tf.concat([self.db_fact_encodings_reshaped, self.interaction_state_encoding_reshaped, cell_state_reshaped], axis=-1)
        
        # get the scores
        db_fact_match_scores = self._db_fact_score_layer(combined_db_fact_score_inputs)
        db_fact_match_scores = tf.reshape(db_fact_match_scores, (self.batchSize, self.numDbFacts, 1))
        
        # apply the scores...
        
        # the attentive reading of the DB facts - this is c_KB from the COREQA paper
        db_match_fact_encoding = tf.multiply(self._db_fact_encodings, db_fact_match_scores)
        db_match_fact_encoding = tf.reduce_sum(db_match_fact_encoding, axis=1)
        
        # update the accumulated vector which records the attentive history of each DB fact
        self.db_match_fact_encoding_hist = tf.add(self.db_match_fact_encoding_hist, db_match_fact_encoding)
        
        # for copying from
        db_fact_match_scores_reshaped = tf.expand_dims(db_fact_match_scores, axis=-1)
        
        self._db_match_val_seq_encoding = tf.multiply(self._db_val_seq_encodings, db_fact_match_scores_reshaped)
        self._db_match_val_seq_encoding = tf.reduce_sum(self._db_match_val_seq_encoding, axis=1)
        
        self._db_match_val_seq = tf.multiply(self._db_val_seq_ids, db_fact_match_scores_reshaped)
        self._db_match_val_seq = tf.reduce_sum(self._db_match_val_seq, axis=1)
        
        
        #
        # compute selective read
        #
        
        # find places in the input where the previous char is the same as the char previously output by copynet
        last_ids_one_hot = tf.one_hot(last_ids, self._vocab_size)
        
        mask1 = tf.expand_dims(last_ids_one_hot, 1) * self._db_match_val_seq
        mask2 = tf.reduce_mean(mask1, axis=2)
        
        rou = mask2 * prob_c
        selective_read = tf.einsum("ijk,ij->ik", self._db_match_val_seq_encoding, rou)
        
        
        #
        # get the next decoder RNN state and output
        #
        inputs = tf.concat([inputs, selective_read], 1)
        outputs, cell_state = self._cell(inputs, cell_state, scope)
        
        
        #
        # compute the generate score
        #
        temp = tf.concat([outputs, db_match_fact_encoding], axis=1)
        generate_score = self._projection(temp)
        prob_g = generate_score
        
        
        #        
        # compute the copy score
        #
        
        # multiply copy weights by the hidden state in M (DB entry)
        hWc = tf.einsum("ijk,km->ijm", self._db_match_val_seq_encoding, self._copy_weight)
        
        # tanh
        tanh_hWc = tf.nn.tanh(hWc)
        
        # multiply by the current state of the decoder RNN
        # this give the copy score for each step in M (DB entry)
        temp = tf.concat([outputs, self.db_match_fact_encoding_hist], axis=1)
        copy_score = tf.einsum("ijm,im->ij", tanh_hWc, temp)
        
        copy_score_per_char_per_DB_step = tf.expand_dims(copy_score, axis=2) * self._db_match_val_seq
        
        prob_c = tf.reduce_sum(copy_score_per_char_per_DB_step, axis=1)
        
        #char_counts = tf.reduce_sum(self._db_match_val_seq, axis=1) # count how many times each char occurs in M (DB entry)
        #prob_c = prob_c / (char_counts + 1e-8) # optional step, divide the copy scores for each char by the char count so that chars don't get a higher copy score just because they occur more frequently
        
        
        
        #
        # combine the copy and generate scores to compute the final output
        #
        outputs = prob_c + prob_g
        
        last_ids = tf.argmax(outputs, axis=-1, output_type=tf.int32)
        
        state = CopyNetWrapperState(cell_state=cell_state, last_ids=last_ids, prob_c=copy_score)
        
        
        return outputs, state, prob_c, prob_g, db_fact_match_scores
    
    
    
    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.

            It can be represented by an Integer, a TensorShape or a tuple of Integers
            or TensorShapes.
        """
        return CopyNetWrapperState(cell_state=self._cell.state_size, 
                                   last_ids=tf.TensorShape([]),
                                   prob_c=self._encoder_state_size)
    
    
    
    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        return self._vocab_size
    
    
    
    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            last_ids = tf.zeros([batch_size], tf.int32) - 1
            prob_c = tf.zeros([batch_size, tf.shape(self._db_val_seq_encodings)[-2]], tf.float32)
            return CopyNetWrapperState(cell_state=cell_state, last_ids=last_ids, prob_c=prob_c)




