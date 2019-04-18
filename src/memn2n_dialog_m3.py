#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 18:05:27 2018

@author: malcolm


try to modify so that a GRU is used for the output answer

"""



from __future__ import absolute_import
from __future__ import division


import tensorflow as tf

import numpy as np
from six.moves import range
from datetime import datetime
from sklearn import metrics



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

    def __init__(self, batch_size, 
                 vocab_size, 
                 sentence_size, 
                 memory_size, 
                 embedding_size):
        
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._memory_size = memory_size
        self._embedding_size = embedding_size
        self._hops = 1
        self._max_grad_norm = 40.0
        self._nonlin = None
        self._init = tf.random_normal_initializer(stddev=0.1)
        self._opt = tf.train.AdamOptimizer(learning_rate=1e-3)
        self._name = "MemN2N"
        
        
        self._build_inputs()
        self._build_vars()

        # define summary directory
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.root_dir = "%s_%s_%s_%s/" % ('task', str(0), 'summary_output', timestamp)
        
        
        sent_logits, loss_op = self._inference(self._stories, self._queries)
        

        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(loss_op)
        grads_and_vars = [(g,v) if g is None else (tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in grads_and_vars]
        # grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
        
        
        # WHAT IS THIS FOR???
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars and not g is None:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        
        train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")


        # predict ops
        predict_op = tf.argmax(sent_logits, 2, name="predict_op")
        predict_proba_op = tf.nn.softmax(sent_logits, axis=2, name="predict_proba_op")
        predict_log_proba_op = tf.log(predict_proba_op, name="predict_log_proba_op")


        # assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.predict_proba_op = predict_proba_op
        self.predict_log_proba_op = predict_log_proba_op
        self.train_op = train_op

        self.graph_output = self.loss_op

        init_op = tf.initialize_all_variables()
        self._sess = tf.Session()
        self._sess.run(init_op)


    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)


    def _build_inputs(self):
        self._stories = tf.placeholder(tf.int32, [self._batch_size, self._memory_size, self._sentence_size], name="stories")
        self._queries = tf.placeholder(tf.int32, [self._batch_size, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [self._batch_size, self._sentence_size, self._vocab_size], name="answers")



    def _build_vars(self):
        with tf.variable_scope(self._name):
            
            nil_word_slot = tf.zeros([1, self._embedding_size])
            
            A = tf.concat([nil_word_slot, self._init([self._vocab_size - 1, self._embedding_size])], 0)
            self.A = tf.Variable(A, name="A")

            self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")
            
            W = tf.concat([nil_word_slot, self._init([self._vocab_size - 1, self._embedding_size])], 0)
            self.W = tf.Variable(W, name="W")
            
            # self.W = tf.Variable(self._init([self._vocab_size, self._embedding_size]), name="W")
            
            
            #
            # for LSTM
            #
            self.lstm = tf.nn.rnn_cell.GRUCell(self._embedding_size)
            
            
            self.Wemb = tf.Variable(tf.random_uniform([self._embedding_size, self._embedding_size], -0.1, 0.1), name='Wemb')
            self.bemb = self.init_bias(self._embedding_size, name='bemb')
            
            
            self.embed_word_W = tf.Variable(tf.random_uniform([self._embedding_size, self._vocab_size], -0.1, 0.1), name='embed_word_W')
            self.embed_word_b = self.init_bias(self._vocab_size, name='embed_word_b')
            
            
        
        
        # WHAT IS THIS FOR???
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
                temp = tf.expand_dims(u[-1], -1)
                u_temp = tf.transpose(temp, [0, 2, 1])
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
            
            #
            # use u_k as input into a one-to-many char sequence model
            #
            state = tf.zeros_like(tf.placeholder(tf.float32, shape=[self._batch_size, self.lstm.state_size]))
            
            output_sent_logits = []
            
            # use the same input every timestep, this is an encoding of the question and the important DB contents
            inputToGru = tf.matmul(u_k, self.Wemb) + self.bemb
            
            
            
            
            loss = 0.0
            
            # step through each timestep of the LSTM?
            for i in range(self._sentence_size): # maxlen + 1
                
                output, state = self.lstm(inputToGru, state)
                
                # get the ground truth
                ground_truth_answers = self._answers[:, i, :] # these are one-hot char encodings at timestep i
                
                
                # get the predicted final output (char)
                # put gru output through final layer to get the correct dim
                logit_chars = tf.matmul(output, self.embed_word_W) + self.embed_word_b
                output_sent_logits.append(tf.reshape(logit_chars, shape=(tf.shape(logit_chars)[0], 1, self._vocab_size)))
                
                # compute the loss                
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_answers, logits=logit_chars)
                #cross_entropy = cross_entropy * mask[:,i] # does this just remove losses computed after the end of the ground truth sentence?
                
                current_loss = tf.reduce_sum(cross_entropy)
                loss = loss + current_loss
            
            output_sent_logits = tf.concat(output_sent_logits, 1)
            
            return output_sent_logits, loss


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


def vectorize_output_sentences(sentences, charToIndex, maxSentLen):
    
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
        
        sentVecs.append(sentVec)
        sentCharIndexLists.append(sentCharIndexList)
    
    return sentVecs, sentCharIndexLists


def unvectorize_output_sentences(sentCharIndexLists, indexToChar):
    
    sentences = []
    
    for i in range(sentCharIndexLists.shape[0]):
        
        sent = ""
        
        for j in range(sentCharIndexLists.shape[1]):
            
            sent += indexToChar[sentCharIndexLists[i,j]]
        
        sentences.append(sent)
        
    return sentences



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




if __name__ == "__main__":
    
    #
    # simulate some interactions
    #
    databases = []
    inputs = []
    outputs = []
    
    
    database = ["D | Sony | LOCATION | DISPLAY_1",
                "D | Sony | PRICE | $800",
                "D | Nikon | LOCATION | DISPLAY_2",
                "D | Nikon | PRICE | $90"]
    
    # example 1
    databases.append(database)
    inputs.append("C | DISPLAY_1 | How much is this one?")
    outputs.append("S | DISPLAY_1 | This one is $800.")
    
    
    # example 2
    databases.append(database)
    inputs.append("C | DISPLAY_2 | How much is this one?")
    outputs.append("S | DISPLAY_2 | This one is $90.")
    
    
    
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
    
    
    #
    #
    #
    database = ["D | Sony | LOCATION | DISPLAY_1",
                "D | Sony | PRICE | $600",
                "D | Nikon | LOCATION | DISPLAY_2",
                "D | Nikon | PRICE | $50"]
    
    # example 5
    databases.append(database)
    inputs.append("C | DISPLAY_1 | How much is this one?")
    outputs.append("S | DISPLAY_1 | This one is $600.")
    
    
    # example 6
    databases.append(database)
    inputs.append("C | DISPLAY_2 | How much is this one?")
    outputs.append("S | DISPLAY_2 | This one is $50.")
    
    
    database = ["D | Sony | LOCATION | DISPLAY_1",
                "D | Sony | PRICE | $700",
                "D | Nikon | LOCATION | DISPLAY_2",
                "D | Nikon | PRICE | $80"]
    
    # example 5
    databases.append(database)
    inputs.append("C | DISPLAY_1 | How much is this one?")
    outputs.append("S | DISPLAY_1 | This one is $700.")
    
    
    # example 6
    databases.append(database)
    inputs.append("C | DISPLAY_2 | How much is this one?")
    outputs.append("S | DISPLAY_2 | This one is $80.")
    
    
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
    outputIds = [0, 1, 2, 3, 4, 5, 6, 7]
    
    
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

    
    #outputCandidateVectors = vectorize_candidates(outputs, charToIndex, maxSentLen)
    vectorizedOutputSentences, sentCharIndexLists = vectorize_output_sentences(outputs, charToIndex, maxSentLen)
    
    data = []
    
    for i in range(numExamples):
        data.append((databases[i], inputs[i], outputIds[i]))
    
    s, q, _ = vectorize_data(data, charToIndex, maxSentLen, 4, 4, maxBufLen)
    
    a = np.array(vectorizedOutputSentences)
    
    #
    # setup the model
    #
    model = MemN2NDialog(batch_size=4,
                         vocab_size=numUniqueChars,
                         sentence_size=maxSentLen,
                         memory_size=maxBufLen,
                         embedding_size=40)
    
    
    #
    # train the model
    #
    
    numEpochs = 100000
    
    for e in range(numEpochs):
        
        trainCost = model.batch_fit(s[:4], q[:4], a[:4])
        
        
        trainUttPreds = model.predict(s[:4], q[:4])
        
        # flatten the arrays
        trainPredsFlat = np.array(trainUttPreds).flatten()
        sentCharIndexListsFlat = np.array(sentCharIndexLists[:4]).flatten()
        
        trainAcc = metrics.accuracy_score(trainPredsFlat, sentCharIndexListsFlat)
        
        
        #print e, round(trainCost, 3), round(trainAcc, 2)
        
        if e % 100 == 0:
            print "****************************************************************"
            print "TRAIN", e, round(trainCost, 3), round(trainAcc, 2)
            
            predSents = unvectorize_output_sentences(trainUttPreds, indexToChar)
            
            for i in range(len(predSents)):
                print "TRUE:", outputs[i]
                print "PRED:", predSents[i]
                print
            
            
            testUttPreds = model.predict(s[4:], q[4:])
            predSents = unvectorize_output_sentences(testUttPreds, indexToChar)
            
            # flatten the arrays
            testPredsFlat = np.array(testUttPreds).flatten()
            sentCharIndexListsFlat = np.array(sentCharIndexLists[4:]).flatten()
            
            testAcc = metrics.accuracy_score(testPredsFlat, sentCharIndexListsFlat)
            
            print "TEST", e, round(testAcc, 2)
            
            for i in range(len(predSents)):
                print "TRUE:", outputs[i+4]
                print "PRED:", predSents[i]
                print
            
            print "****************************************************************"
    






