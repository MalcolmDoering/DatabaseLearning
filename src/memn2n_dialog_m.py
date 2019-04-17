
from __future__ import absolute_import
from __future__ import division


import tensorflow as tf
import numpy as np
from six.moves import range
from datetime import datetime
from sklearn import metrics
import numpy as np




def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.op_scope([t], name, "zero_nil_slot") as name:
        
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.stack([1, s]))
        # z = tf.zeros([1, s])
        
        return tf.concat([z, tf.slice(t, [1, 0], [-1, -1])], 0, name=name)


def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        
        return tf.add(t, gn, name=name)


class MemN2NDialog(object):
    """End-To-End Memory Network."""

    def __init__(self, batch_size, vocab_size, candidates_size, sentence_size, embedding_size,
                 candidates_vec,
                 hops=3,
                 max_grad_norm=40.0,
                 nonlin=None,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 optimizer=tf.train.AdamOptimizer(learning_rate=1e-2),
                 session=tf.Session(),
                 name='MemN2N',
                 task_id=1):
        
        """Creates an End-To-End Memory Network

        Args:
            batch_size: The size of the batch.

            vocab_size: The size of the vocabulary (should include the nil word). The nil word
            one-hot encoding should be 0.

            sentence_size: The max size of a sentence in the data. All sentences should be padded
            to this length. If padding is required it should be done with nil one-hot encoding (0).

            candidates_size: The size of candidates

            memory_size: The max size of the memory. Since Tensorflow currently does not support jagged arrays
            all memories must be padded to this length. If padding is required, the extra memories should be
            empty memories; memories filled with the nil word ([0, 0, 0, ......, 0]).

            embedding_size: The size of the word embedding.

            candidates_vec: The numpy array of candidates encoding.

            hops: The number of hops. A hop consists of reading and addressing a memory slot.
            Defaults to `3`.

            max_grad_norm: Maximum L2 norm clipping value. Defaults to `40.0`.

            nonlin: Non-linearity. Defaults to `None`.

            initializer: Weight initializer. Defaults to `tf.random_normal_initializer(stddev=0.1)`.

            optimizer: Optimizer algorithm used for SGD. Defaults to `tf.train.AdamOptimizer(learning_rate=1e-2)`.

            encoding: A function returning a 2D Tensor (sentence_size, embedding_size). Defaults to `position_encoding`.

            session: Tensorflow Session the model is run with. Defaults to `tf.Session()`.

            name: Name of the End-To-End Memory Network. Defaults to `MemN2N`.
        """

        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._candidates_size = candidates_size
        self._sentence_size = sentence_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._init = initializer
        self._opt = optimizer
        self._name = name
        self._candidates = candidates_vec

        self._build_inputs()
        self._build_vars()

        # define summary directory
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.root_dir = "%s_%s_%s_%s/" % ('task', str(task_id), 'summary_output', timestamp)


        # cross entropy
        # (batch_size, candidates_size)
        logits = self._inference(self._stories, self._queries)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self._answers, name="cross_entropy")
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")

        # loss op
        loss_op = cross_entropy_sum


        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(loss_op)
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in grads_and_vars]
        # grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
        
        
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        
        train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")


        # predict ops
        predict_op = tf.argmax(logits, 1, name="predict_op")
        predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")
        predict_log_proba_op = tf.log(predict_proba_op, name="predict_log_proba_op")


        # assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.predict_proba_op = predict_proba_op
        self.predict_log_proba_op = predict_log_proba_op
        self.train_op = train_op

        self.graph_output = self.loss_op

        init_op = tf.initialize_all_variables()
        self._sess = session
        self._sess.run(init_op)



    def _build_inputs(self):
        self._stories = tf.placeholder(tf.int32, [None, None, self._sentence_size], name="stories")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None], name="answers")



    def _build_vars(self):
        with tf.variable_scope(self._name):
            
            nil_word_slot = tf.zeros([1, self._embedding_size])
            
            A = tf.concat([nil_word_slot, self._init([self._vocab_size - 1, self._embedding_size])], 0)
            self.A = tf.Variable(A, name="A")

            self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")
            
            W = tf.concat([nil_word_slot, self._init([self._vocab_size - 1, self._embedding_size])], 0)
            self.W = tf.Variable(W, name="W")
            
            # self.W = tf.Variable(self._init([self._vocab_size, self._embedding_size]), name="W")
        
        
        self._nil_vars = set([self.A.name, self.W.name])



    def _inference(self, stories, queries):
        
        with tf.variable_scope(self._name):
            q_emb = tf.nn.embedding_lookup(self.A, queries)
            u_0 = tf.reduce_sum(q_emb, 1)
            u = [u_0]
        
            for _ in range(self._hops):
            
                m_emb = tf.nn.embedding_lookup(self.A, stories)
                m = tf.reduce_sum(m_emb, 2)
                
                # hack to get around no reduce_dot
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                dotted = tf.reduce_sum(m * u_temp, 2)

                # Calculate probabilities
                probs = tf.nn.softmax(dotted)

                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
                c_temp = tf.transpose(m, [0, 2, 1])
                o_k = tf.reduce_sum(c_temp * probs_temp, 2)

                u_k = tf.matmul(u[-1], self.H) + o_k
                # u_k=u[-1]+tf.matmul(o_k,self.H)
                # nonlinearity
                
                if self._nonlin:
                    u_k = self._nonlin(u_k)

                u.append(u_k)
            
            
            candidates_emb = tf.nn.embedding_lookup(self.W, self._candidates)
            candidates_emb_sum = tf.reduce_sum(candidates_emb, 1)
            
            
            return tf.matmul(u_k, tf.transpose(candidates_emb_sum))
            
            # logits=tf.matmul(u_k, self.W)
            # return
            # tf.transpose(tf.sparse_tensor_dense_matmul(self._candidates,tf.transpose(logits)))



    def batch_fit(self, stories, queries, answers):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """

        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers}
        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
        
        return loss



    def predict(self, stories, queries):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        
        feed_dict = {self._stories: stories, self._queries: queries}
        
        return self._sess.run(self.predict_op, feed_dict=feed_dict)



def vectorize_candidates(candidates, charToIndex, maxSentLen):
    shape=(len(candidates), maxSentLen)
    C=[]
    
    for i, candidate in enumerate(candidates):
        lc=max(0, maxSentLen-len(candidate))
        C.append([charToIndex[w] if w in charToIndex else 0 for w in candidate] + [0] * lc)
        
    return tf.constant(C,shape=shape)




def vectorize_data(data, word_idx, sentence_size, batch_size, candidates_size, max_memory_size):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    S = []
    Q = []
    A = []
    
    #data.sort(key=lambda x:len(x[0]),reverse=True)
    
    for i, (story, query, answer) in enumerate(data):
    
        if i%batch_size==0:
            memory_size=max(1,min(max_memory_size,len(story)))
        
        ss = []
        
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] if w in word_idx else 0 for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] if w in word_idx else 0 for w in query] + [0] * lq


        S.append(np.array(ss))
        Q.append(np.array(q))
        A.append(np.array(answer))
        
        
    return S, Q, A



class Seq2SeqModel(object):
    """Sequence-to-sequence model with attention and for multiple buckets.
    
    This class implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder. This is the same as the model described in
    this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
    or into the seq2seq library for complete model implementation.
    This class also allows to use GRU cells in addition to LSTM cells, and
    sampled softmax to handle large output vocabulary size. A single-layer
    version of this model, but with bi-directional encoder, was presented in
        http://arxiv.org/abs/1409.0473
    and sampled softmax is described in Section 3 of the following paper.
        http://arxiv.org/abs/1412.2007
    """
    
    
    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 buckets,
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 use_lstm=False,
                 num_samples=512,
                 forward_only=False,
                 dtype=tf.float32):
        """Create the model.
        
        Args:
          source_vocab_size: size of the source vocabulary.
          target_vocab_size: size of the target vocabulary.
          buckets: a list of pairs (I, O), where I specifies maximum input length
            that will be processed in that bucket, and O specifies maximum output
            length. Training instances that have inputs longer than I or outputs
            longer than O will be pushed to the next bucket and padded accordingly.
            We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
          size: number of units in each layer of the model.
          num_layers: number of layers in the model.
          max_gradient_norm: gradients will be clipped to maximally this norm.
          batch_size: the size of the batches used during training;
            the model construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g., for decoding.
          learning_rate: learning rate to start with.
          learning_rate_decay_factor: decay learning rate by this much when needed.
          use_lstm: if true, we use LSTM cells instead of GRU cells.
          num_samples: number of samples for sampled softmax.
          forward_only: if set, we do not construct the backward pass in the model.
          dtype: the data type to use to store internal variables.
        """
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
    
        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
            output_projection = (w, b)
    
            def sampled_loss(labels, logits):
                labels = tf.reshape(labels, [-1, 1])
                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(logits, tf.float32)
                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        weights=local_w_t,
                        biases=local_b,
                        labels=labels,
                        inputs=local_inputs,
                        num_sampled=num_samples,
                        num_classes=self.target_vocab_size),
                    dtype)
            softmax_loss_function = sampled_loss
    
        # Create the internal multi-layer cell for our RNN.
        def single_cell():
          return tf.contrib.rnn.GRUCell(size)
        if use_lstm:
          def single_cell():
            return tf.contrib.rnn.BasicLSTMCell(size)
        cell = single_cell()
        if num_layers > 1:
          cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
    
        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
          return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
              encoder_inputs,
              decoder_inputs,
              cell,
              num_encoder_symbols=source_vocab_size,
              num_decoder_symbols=target_vocab_size,
              embedding_size=size,
              output_projection=output_projection,
              feed_previous=do_decode,
              dtype=dtype)
    
        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
          self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="encoder{0}".format(i)))
        for i in xrange(buckets[-1][1] + 1):
          self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="decoder{0}".format(i)))
          self.target_weights.append(tf.placeholder(dtype, shape=[None],
                                                    name="weight{0}".format(i)))
    
        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1]
                   for i in xrange(len(self.decoder_inputs) - 1)]
    
        # Training outputs and losses.
        if forward_only:
          self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
              self.encoder_inputs, self.decoder_inputs, targets,
              self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
              softmax_loss_function=softmax_loss_function)
          # If we use output projection, we need to project outputs for decoding.
          if output_projection is not None:
            for b in xrange(len(buckets)):
              self.outputs[b] = [
                  tf.matmul(output, output_projection[0]) + output_projection[1]
                  for output in self.outputs[b]
              ]
        else:
          self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
              self.encoder_inputs, self.decoder_inputs, targets,
              self.target_weights, buckets,
              lambda x, y: seq2seq_f(x, y, False),
              softmax_loss_function=softmax_loss_function)
    
        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
          self.gradient_norms = []
          self.updates = []
          opt = tf.train.GradientDescentOptimizer(self.learning_rate)
          for b in xrange(len(buckets)):
            gradients = tf.gradients(self.losses[b], params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             max_gradient_norm)
            self.gradient_norms.append(norm)
            self.updates.append(opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step))
    
        self.saver = tf.train.Saver(tf.global_variables())
    
  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
      return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

  def get_batch(self, data, bucket_id):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(self.batch_size):
      encoder_input, decoder_input = random.choice(data[bucket_id])

      # Encoder inputs are padded and then reversed.
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights







if __name__ == "__main__":
    
    #
    # simulate some interactions
    #
    databases = []
    inputs = []
    outputs = []
    
    
    database = ["D | Sony | LOCATION | DISPLAY_1",
                "D | Sony | PRICE | $550",
                "D | Nikon | LOCATION | DISPLAY_2",
                "D | Nikon | PRICE | $68"]
    
    # example 1
    databases.append(database)
    inputs.append("C | DISPLAY_1 | How much is this one?")
    outputs.append("S | DISPLAY_1 | This one is $550.")
    
    
    # example 2
    databases.append(database)
    inputs.append("C | DISPLAY_2 | How much is this one?")
    outputs.append("S | DISPLAY_2 | This one is $68.")
    
    
    
    database = ["D | Sony | LOCATION | DISPLAY_1",
                "D | Sony | PRICE | $500",
                "D | Nikon | LOCATION | DISPLAY_2",
                "D | Nikon | PRICE | $60"]
    
    # example 3
    databases.append(database)
    inputs.append("C | DISPLAY_1 | How much is this one?")
    outputs.append("S | DISPLAY_1 | This one is $500.")
    
    
    # example 4
    databases.append(database)
    inputs.append("C | DISPLAY_2 | How much is this one?")
    outputs.append("S | DISPLAY_2 | This one is $60.")
    
    
    numExamples = len(outputs)
    
    
    inputBufLens = [len(db) for db in databases]
    
    
    dbSentLens = []
    for db in databases:
        for entry in db:
            dbSentLens.append(len(entry))
    
    inputSentLens = []
    for i in inputs:
        inputSentLens.append(len(i))
    
    outputSentLens = []
    for i in outputs:
        outputSentLens.append(len(i))
    
    
    uniqueChars = []
    
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
    
    
    maxBufLen = max(inputBufLens)
    maxSentLen = max(dbSentLens + inputSentLens + outputSentLens)
    
    
    #
    #
    #
    outputIds = [0, 1, 2, 3]
    
    
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

    
    
    
    outputCandidateVectors = vectorize_candidates(outputs, charToIndex, maxSentLen)
    
    data = []
    
    for i in range(numExamples):
        data.append((databases[i], inputs[i], outputIds[i]))
    
    s, q, a = vectorize_data(data, charToIndex, maxSentLen, 4, 4, maxBufLen)
    
    
    #
    # setup the model
    #
    model = MemN2NDialog(batch_size=4,
                         vocab_size=numUniqueChars,
                         candidates_size=4,
                         sentence_size=maxSentLen,
                         embedding_size=20,
                         candidates_vec=outputCandidateVectors,
                         hops=1)
    
    
    #
    # train the model
    #
    
    numEpochs = 100
    
    for e in range(numEpochs):
        
        trainCost = model.batch_fit(s, q, a)
        
        
        trainPreds = model.predict(s, q)
        trainAcc = metrics.accuracy_score(np.array(trainPreds), a)
        
        
        print e, round(trainCost, 3), round(trainAcc, 2)
    
    






