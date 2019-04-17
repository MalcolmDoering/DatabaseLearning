'''
Created on Nov 15, 2018

@author: MalcolmD
'''


import tensorflow as tf

import numpy as np
from six.moves import range
from datetime import datetime
from sklearn import metrics
import editdistance


import tools
from copynet.copynetoriginal import copynet



#sessionDir = tools.create_session_dir("actionPrediction4_dbl")



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
        
        
        sentVec[-1, charToIndex["<EOS>"]] = 1
        sentCharIndexList.append(charToIndex["<EOS>"])
        
        sentVecs.append(sentVec)
        sentCharIndexLists.append(sentCharIndexList)
    
    
    return sentVecs, sentCharIndexLists



def unvectorize_sentences(sentCharIndexLists, indexToChar):
    
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




def enc_dec_model_inputs(batch_size):
    inputs = tf.placeholder(tf.int32, [batch_size, None], name='input')
    targets = tf.placeholder(tf.int32, [batch_size, None], name='targets') 
    
    target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
    max_target_len = tf.reduce_max(target_sequence_length)
    
    return inputs, targets, target_sequence_length, max_target_len



def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    # get '<GO>' id
    go_id = target_vocab_to_int['<GO>']
    
    after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    after_concat = tf.concat( [tf.fill([batch_size, 1], go_id), after_slice], 1)
    
    return after_concat



def encoding_layer(rnn_inputs, 
                   rnn_size, 
                   num_layers, 
                   keep_prob, 
                   source_vocab_size, 
                   encoding_embedding_size):
    """
    :return: tuple (RNN output, RNN state)
    """
    embed = tf.contrib.layers.embed_sequence(rnn_inputs, 
                                             vocab_size=source_vocab_size, 
                                             embed_dim=encoding_embedding_size)
    
    stacked_cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(rnn_size), keep_prob) for _ in range(num_layers)]) # , state_is_tuple=False
    #stacked_cells = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(rnn_size), keep_prob)
    
    outputs, state = tf.nn.dynamic_rnn(stacked_cells, 
                                       embed, 
                                       dtype=tf.float32)
    state = state[-1]
    
    return outputs, state



def decoding_layer_train(encoder_state,
                         dec_cell,
                         dec_embed_input,
                         target_sequence_length,
                         max_summary_length,
                         output_layer,
                         keep_prob):
    
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, 
                                             output_keep_prob=keep_prob)
    
    # for only input layer
    helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, 
                                               target_sequence_length)
    
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, 
                                              helper, 
                                              encoder_state, 
                                              output_layer)

    # unrolling the decoder layer
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                      impute_finished=True, 
                                                      maximum_iterations=max_summary_length)
    return outputs



def decoding_layer_infer(encoder_state, 
                         dec_cell, 
                         dec_embeddings,
                         start_of_sequence_id,
                         end_of_sequence_id,
                         max_target_sequence_length,
                         vocab_size,
                         output_layer,
                         batch_size,
                         keep_prob):
    
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, 
                                             output_keep_prob=keep_prob)
    
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, 
                                                      tf.fill([batch_size], start_of_sequence_id), 
                                                      end_of_sequence_id)
    
    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, 
                                              helper, 
                                              encoder_state, 
                                              output_layer)
    
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                      impute_finished=True, 
                                                      maximum_iterations=max_target_sequence_length)
    return outputs



def decoding_layer(dec_input, 
                   encoder_state,
                   target_sequence_length, 
                   max_target_sequence_length,
                   rnn_size,
                   num_layers, 
                   target_vocab_to_int, 
                   target_vocab_size,
                   batch_size, keep_prob, 
                   decoding_embedding_size):
    
    target_vocab_size = len(target_vocab_to_int)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
    cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(rnn_size) for _ in range(num_layers)]) # , state_is_tuple=False
    
    with tf.variable_scope("decode"):
        output_layer = tf.layers.Dense(target_vocab_size)
        train_output = decoding_layer_train(encoder_state, 
                                            cells, 
                                            dec_embed_input, 
                                            target_sequence_length, 
                                            max_target_sequence_length, 
                                            output_layer, 
                                            keep_prob)

    with tf.variable_scope("decode", reuse=True):
        infer_output = decoding_layer_infer(encoder_state, 
                                            cells, 
                                            dec_embeddings, 
                                            target_vocab_to_int['<GO>'], 
                                            target_vocab_to_int['<EOS>'], 
                                            max_target_sequence_length, 
                                            target_vocab_size, 
                                            output_layer,
                                            batch_size,
                                            keep_prob)

    return (train_output, infer_output)




def copy_decoding_layer(input_data,
                        dec_input,
                        encoder_outputs,
                        encoder_state,
                        target_sequence_length, 
                        max_target_sequence_length,
                        rnn_size,
                        target_vocab_to_int, 
                        target_vocab_size,
                        batch_size,
                        keep_prob, 
                        decoding_embedding_size):
    
    
    target_vocab_size = len(target_vocab_to_int)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
    #cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(rnn_size) for _ in range(num_layers)]) # , state_is_tuple=False
    cell = tf.nn.rnn_cell.GRUCell(rnn_size)
    copynet_cell = copynet.CopyNetWrapper(cell=cell,
                                          encoder_states=encoder_outputs, # GRU state is the same as its output
                                          encoder_input_ids=input_data,
                                          vocab_size=target_vocab_size)
    
    
    with tf.variable_scope("decode"):
        output_layer = tf.layers.Dense(target_vocab_size)
        
        decoder_initial_state = copynet_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state) # cell_state=?
        
        # for only input layer
        helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, target_sequence_length)
        
        decoder = tf.contrib.seq2seq.BasicDecoder(copynet_cell,
                                                  helper,
                                                  decoder_initial_state,
                                                  output_layer)
    
        # unrolling the decoder layer
        # returns final_outputs, final_state, final_sequence_lengths
        train_output, _, _= tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                               impute_finished=True, 
                                                               maximum_iterations=max_target_sequence_length)
        

    with tf.variable_scope("decode", reuse=True):
        
        decoder_initial_state = copynet_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state) # cell_state=?
        
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, 
                                                          tf.fill([batch_size], target_vocab_to_int['<GO>']), 
                                                          target_vocab_to_int['<EOS>'])
        
        decoder = tf.contrib.seq2seq.BasicDecoder(copynet_cell, 
                                                  helper, 
                                                  decoder_initial_state, 
                                                  output_layer)
        
        infer_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                               impute_finished=True, 
                                                               maximum_iterations=max_target_sequence_length)
        
        
    return (train_output, infer_output)




"""
cell = tf.nn.rnn_cell.GRUCell(embedding_size)

copynet_cell = copynetoriginal.CopyNetWrapper(cell, 
                                      encoder_outputs,
                                      encoder_input_ids,
                                      numUniqueChars, 
                                      numUniqueChars)


decoder_initial_state = copynet_cell.zero_state(batch_size, tf.float32).clone(cell_state=decoder_initial_state)

helper = tf.contrib.seq2seq.TrainingHelper(...)

decoder = tf.contrib.seq2seq.BasicDecoder(copynet_cell, 
                                          helper,
                                          decoder_initial_state, 
                                          output_layer=None)

decoder_outputs, final_state, coder_seq_length = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)

decoder_logits, decoder_ids = decoder_outputs
"""



def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence model
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    enc_outputs, enc_states = encoding_layer(input_data,
                                             rnn_size,
                                             num_layers,
                                             keep_prob,
                                             source_vocab_size,
                                             enc_embedding_size)
    
    dec_input = process_decoder_input(target_data, 
                                      target_vocab_to_int, 
                                      batch_size)
    
    train_output, infer_output = copy_decoding_layer(input_data,
                                                     dec_input,
                                                     enc_outputs,
                                                     enc_states,
                                                     target_sequence_length, 
                                                     max_target_sentence_length,
                                                     rnn_size,
                                                     target_vocab_to_int,
                                                     target_vocab_size,
                                                     batch_size,
                                                     keep_prob,
                                                     dec_embedding_size)
    
    return train_output, infer_output





if __name__ == "__main__":
    
    #
    # simulate some interactions
    #
    databases = []
    questions = []
    outputs = []
    
    
    database = ["D | Sony | LOCATION | DISPLAY_1.",
                "D | Sony | PRICE | $800.",
                "D | Nikon | LOCATION | DISPLAY_2.",
                "D | Nikon | PRICE | $90."]
    
    # example 1
    databases.append(database)
    questions.append("C | DISPLAY_1 | How much is this one?")
    outputs.append("S | DISPLAY_1 | This one is $800.")
    
    
    # example 2
    databases.append(database)
    questions.append("C | DISPLAY_2 | How much is this one?")
    outputs.append("S | DISPLAY_2 | This one is $90.")
    
    
    
    database = ["D | Sony | LOCATION | DISPLAY_1.",
                "D | Sony | PRICE | $500.",
                "D | Nikon | LOCATION | DISPLAY_2.",
                "D | Nikon | PRICE | $60."]
    
    # example 3
    databases.append(database)
    questions.append("C | DISPLAY_1 | How much is this one?")
    outputs.append("S | DISPLAY_1 | This one is $500.")
    
    
    # example 4
    databases.append(database)
    questions.append("C | DISPLAY_2 | How much is this one?")
    outputs.append("S | DISPLAY_2 | This one is $60.")
    
    
    #
    #
    #
    database = ["D | Sony | LOCATION | DISPLAY_1.",
                "D | Sony | PRICE | $600.",
                "D | Nikon | LOCATION | DISPLAY_2.",
                "D | Nikon | PRICE | $50."]
    
    # example 5
    databases.append(database)
    questions.append("C | DISPLAY_1 | How much is this one?")
    outputs.append("S | DISPLAY_1 | This one is $600.")
    
    
    # example 6
    databases.append(database)
    questions.append("C | DISPLAY_2 | How much is this one?")
    outputs.append("S | DISPLAY_2 | This one is $50.")
    
    
    database = ["D | Sony | LOCATION | DISPLAY_1.",
                "D | Sony | PRICE | $700.",
                "D | Nikon | LOCATION | DISPLAY_2.",
                "D | Nikon | PRICE | $80."]
    
    # example 5
    databases.append(database)
    questions.append("C | DISPLAY_1 | How much is this one?")
    outputs.append("S | DISPLAY_1 | This one is $700.")
    
    
    # example 6
    databases.append(database)
    questions.append("C | DISPLAY_2 | How much is this one?")
    outputs.append("S | DISPLAY_2 | This one is $80.")
    
    
    numExamples = len(outputs)
    
    
    databaseLens = [len(db) for db in databases]
    
    
    dbSentLens = []
    for db in databases:
        for entry in db:
            dbSentLens.append(len(entry))
    
    questionSentLens = []
    for i in questions:
        questionSentLens.append(len(i))
    
    outputSentLens = []
    for i in outputs:
        outputSentLens.append(len(i))
    
    
    uniqueChars = []
    
    for db in databases:
        for entry in db:
            for c in entry:
                if c not in uniqueChars:
                    uniqueChars.append(c)
    
    
    for i in questions:
        for c in i:
            if c not in uniqueChars:
                uniqueChars.append(c)
    
    for o in outputs:
        for c in o:
            if c not in uniqueChars:
                uniqueChars.append(c)
    
    
    maxDatabaseLen = max(databaseLens)
    maxSentLen = max(dbSentLens + questionSentLens + outputSentLens)
    
    
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
    
    
    charToIndex["<GO>"] = len(uniqueChars)
    indexToChar[charToIndex["<GO>"]] = "<GO>"
    
    charToIndex["<EOS>"] = len(uniqueChars)
    indexToChar[charToIndex["<EOS>"]] = "<EOS>"
    
    numUniqueChars = len(uniqueChars)
    
    
    #
    # each input is the customer question concatenated to all of the DB entries
    #
    encoderInputStrings = []
    
    for i in range(len(questions)):
        
        encoderInputString = ""
        
        encoderInputString += questions[i]
        
        for entry in databases[i]:
            encoderInputString += entry
        
        encoderInputStrings.append(encoderInputString)
    
    
    #
    # vectorize the encoder inputs as lists of one-hot char vectors
    #
    maxEncoderInputLen = max([len(eis) for eis in encoderInputStrings])
    encoderInputVectors, encoderInputOneHotIndexLists = vectorize_sentences(encoderInputStrings, charToIndex, maxEncoderInputLen)
    
    
    #
    # vectorize the outputs as lists of one-hot char vectors
    #
    maxDecoderOutputLen = max([len(o) for o in outputs])
    decoderOutputVectors, decoderOutputOneHotIndexLists = vectorize_sentences(outputs, charToIndex, maxDecoderOutputLen)
    
    
    #
    # setup the model
    #
    num_layers = 3
    embedding_size = 20
    batch_size = 4
    
    
    
    source_int_text = encoderInputOneHotIndexLists
    target_int_text = decoderOutputOneHotIndexLists
    source_vocab_to_int = charToIndex
    target_vocab_to_int =charToIndex
    
    target_sentence_lengths = [len(s) for s in target_int_text]
    
    max_target_sentence_length = max(target_sentence_lengths)
    
    
    
    train_graph = tf.Graph()
    with train_graph.as_default():
        input_data, targets, target_sequence_length, max_target_sequence_length = enc_dec_model_inputs(batch_size)
        keep_prob = 1.0
        
        train_logits, inference_logits = seq2seq_model(input_data=tf.reverse(input_data, [-1]),
                                                       target_data=targets, 
                                                       keep_prob=keep_prob, 
                                                       batch_size=batch_size,
                                                       target_sequence_length=target_sequence_length,
                                                       max_target_sentence_length=max_target_sequence_length,
                                                       source_vocab_size=len(source_vocab_to_int),   
                                                       target_vocab_size=len(target_vocab_to_int),
                                                       enc_embedding_size=embedding_size, 
                                                       dec_embedding_size=embedding_size,
                                                       rnn_size=embedding_size, 
                                                       num_layers=num_layers, 
                                                       target_vocab_to_int=target_vocab_to_int)
        
        
        training_logits = tf.identity(train_logits.rnn_output, name='logits')
        inference_logits = tf.identity(inference_logits.sample_id, name='predictions')
    
        # https://www.tensorflow.org/api_docs/python/tf/sequence_mask
        # - Returns a mask tensor representing the first N positions of each cell.
        masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
    
        with tf.name_scope("optimization"):
            # Loss function - weighted softmax cross entropy
            cost = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                targets,
                masks)
    
            # Optimizer
            optimizer = tf.train.AdamOptimizer()
    
            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
            
            
            init_op = tf.initialize_all_variables()
            
            
            #
            # train 
            #
            sess = tf.Session()
            sess.run(init_op)
            
            train_feed_dict = {input_data: source_int_text[:4], targets: target_int_text[:4], target_sequence_length: target_sentence_lengths[:4]}
            test_feed_dict = {input_data: source_int_text[4:], targets: target_int_text[4:], target_sequence_length: target_sentence_lengths[4:]}
            
            
            
            trainGroundTruth = target_int_text[:4]
            testGroundTruth = target_int_text[4:]
            trainGroundTruthFlat = []
            testGroundTruthFlat = []
            
            for i in range(numExamples):
                
                groundTruthFlat = target_int_text[i]
                
                if i < 4:
                    trainGroundTruthFlat += groundTruthFlat
                else:
                    testGroundTruthFlat += groundTruthFlat
            
                
            
            numEpochs = 10000
            
            for e in range(numEpochs):
                
                trainCost, _ = sess.run([cost, train_op], feed_dict=train_feed_dict)
                
                            
                if e % 100 == 0:
                    
                    #
                    # compute accuracy, etc.
                    #
                    
                    # TRAIN
                    trainPreds = sess.run(inference_logits, feed_dict=train_feed_dict)
                    
                    trainAcc = 0.0
                    for i in range(len(trainPreds)):
                        trainAcc = normalized_edit_distance(trainGroundTruth[i], trainPreds[i])
                    trainAcc /= len(trainGroundTruth)
                    
                    #trainPredsFlat = np.array(trainPreds).flatten()
                    #trainAcc = metrics.accuracy_score(trainPredsFlat, trainGroundTruthFlat)
                    trainPredSents = unvectorize_sentences(trainPreds, indexToChar)
                    
                    
                    # TEST
                    testPreds = sess.run(inference_logits, feed_dict=test_feed_dict)
                    
                    testAcc = 0.0
                    for i in range(len(testPreds)):
                        testAcc = normalized_edit_distance(testGroundTruth[i], testPreds[i])
                    testAcc /= len(testGroundTruth)
                    
                    #testPredsFlat = np.array(testPreds).flatten()
                    #testAcc = metrics.accuracy_score(testPredsFlat, testGroundTruthFlat)
                    testPredSents = unvectorize_sentences(testPreds, indexToChar)
                    
                    
                    
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









