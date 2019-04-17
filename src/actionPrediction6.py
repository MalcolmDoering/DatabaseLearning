'''
Created on Nov 15, 2018

@author: MalcolmD

copied from actionPrediction4, but doing manual roll out of decoder so we can get the copy scores, etc.
'''


import tensorflow as tf

import numpy as np
from six.moves import range
from datetime import datetime
from sklearn import metrics
import editdistance


import tools
from copynet import copynet



#sessionDir = tools.create_session_dir("actionPrediction4_dbl")

eosChar = "#"
goChar = "~"



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
    go_id = target_vocab_to_int[goChar]
    
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
                                                          tf.fill([batch_size], target_vocab_to_int[goChar]), 
                                                          target_vocab_to_int[eosChar])
        
        decoder = tf.contrib.seq2seq.BasicDecoder(copynet_cell, 
                                                  helper, 
                                                  decoder_initial_state, 
                                                  output_layer)
        
        infer_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                               impute_finished=True, 
                                                               maximum_iterations=max_target_sequence_length)
        
        
    return (train_output, infer_output)



def copy_decoding_layer_2(input_data,
                          target_data,
                          dec_input,
                          encoder_outputs,
                          encoder_state,
                          max_target_sequence_length,
                          rnn_size,
                          target_vocab_to_int,
                          batch_size,
                          decoding_embedding_size):
    
    
    target_vocab_size = len(target_vocab_to_int)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
    
    cell = tf.nn.rnn_cell.GRUCell(rnn_size)
    copynet_cell = copynet.CopyNetWrapper(cell=cell,
                                          encoder_states=encoder_outputs, # GRU state is the same as its output
                                          encoder_input_ids=input_data,
                                          vocab_size=target_vocab_size)
    
    
    with tf.variable_scope("decode"):
        
        loss = 0
        output_sequences = []
        copy_scores = []
        gen_scores = []
        
        # this should actually take the output of an attention layer over the encoder outputs
        decoder_initial_state = copynet_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)
        
        
        state = decoder_initial_state
        output = dec_embed_input[:, 0, :]
        
        for i in range(max_target_sequence_length):
            
            # if not using teacher forcing
            #output, state, copy_score, gen_score = copynet_cell(output, state)
            
            # if using teacher forcing
            output, state, copy_score, gen_score = copynet_cell(dec_embed_input[:, i, :], state)
            
            
            output_sequences.append(tf.reshape(output, shape=(tf.shape(output)[0], 1, target_vocab_size)))
            copy_scores.append(tf.reshape(copy_score, shape=(tf.shape(copy_score)[0], 1, target_vocab_size)))
            gen_scores.append(tf.reshape(gen_score, shape=(tf.shape(gen_score)[0], 1, target_vocab_size)))
            
            # get the ground truth output
            ground_truth_output = tf.one_hot(target_data[:, i], target_vocab_size) # these are one-hot char encodings at timestep i
            
            # compute the loss                
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_output, logits=output)
            current_loss = tf.reduce_sum(cross_entropy)
            loss = loss + current_loss
        
        
        output_sequences = tf.concat(output_sequences, 1)
        copy_scores = tf.concat(copy_scores, 1)
        gen_scores = tf.concat(gen_scores, 1)
            
    
    #
    # setup the training function
    #
    opt = tf.train.AdamOptimizer(learning_rate=1e-3)
    
    gradients = opt.compute_gradients(loss)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    
    train_op = opt.apply_gradients(capped_gradients, name="train_op")
    
    
    #
    # setup the prediction function
    #
    predict_op = tf.argmax(output_sequences, 2, name="predict_op")
    predict_proba_op = tf.nn.softmax(output_sequences, axis=2, name="predict_proba_op")
    predict_log_proba_op = tf.log(predict_proba_op, name="predict_log_proba_op")
    
    
    #
    # initialize
    #
    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)
    
        
    return loss, init_op, train_op, predict_op, copy_scores, gen_scores



def seq2seq_model(input_data, 
                  target_data, 
                  keep_prob, 
                  batch_size,
                  target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, 
                  target_vocab_size,
                  enc_embedding_size, 
                  dec_embedding_size,
                  rnn_size, num_layers, 
                  target_vocab_to_int):
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
    
    loss_op, init_op, train_op, predict_op, copy_scores, gen_scores = copy_decoding_layer_2(input_data,
                                                       target_data,
                                                       dec_input,
                                                       enc_outputs,
                                                       enc_states,
                                                       max_target_sentence_length,
                                                       rnn_size,
                                                       target_vocab_to_int,
                                                       batch_size,
                                                       dec_embedding_size)
    
    return loss_op, init_op, train_op, predict_op, copy_scores, gen_scores



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
    
    
    charToIndex[goChar] = len(uniqueChars)
    indexToChar[charToIndex[goChar]] = goChar
    
    charToIndex[eosChar] = len(uniqueChars)
    indexToChar[charToIndex[eosChar]] = eosChar
    
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
    embedding_size = 35
    batch_size = 4
    
    
    
    source_int_text = encoderInputOneHotIndexLists
    target_int_text = decoderOutputOneHotIndexLists
    vocab_to_int = charToIndex
    
    target_sentence_lengths = [len(s) for s in target_int_text]
    
    max_target_sentence_length = max(target_sentence_lengths)
    
    
    
    train_graph = tf.Graph()
    with train_graph.as_default():
        input_data, targets, target_sequence_length, max_target_sequence_length = enc_dec_model_inputs(batch_size)
        
        max_target_sequence_length = max_target_sentence_length
        
        keep_prob = 1.0
        
        loss_op, init_op, train_op, pred_op, copy_scores, gen_scores = seq2seq_model(input_data=tf.reverse(input_data, [-1]),
                                                       target_data=targets, 
                                                       keep_prob=keep_prob, 
                                                       batch_size=batch_size,
                                                       target_sequence_length=target_sequence_length,
                                                       max_target_sentence_length=max_target_sequence_length,
                                                       source_vocab_size=len(vocab_to_int),   
                                                       target_vocab_size=len(vocab_to_int),
                                                       enc_embedding_size=embedding_size, 
                                                       dec_embedding_size=len(vocab_to_int),
                                                       rnn_size=embedding_size, 
                                                       num_layers=num_layers, 
                                                       target_vocab_to_int=vocab_to_int)
            
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
            
            trainCost, _ = sess.run([loss_op, train_op], feed_dict=train_feed_dict)
            
            
            if e % 100 == 0:
                
                #
                # compute accuracy, etc.
                #
                
                # TRAIN
                trainPreds, trainCopyScores, trainGenScores = sess.run([pred_op, copy_scores, gen_scores], feed_dict=train_feed_dict)
                
                trainAcc = 0.0
                for i in range(len(trainPreds)):
                    trainAcc = normalized_edit_distance(trainGroundTruth[i], trainPreds[i])
                trainAcc /= len(trainGroundTruth)
                
                #trainPredsFlat = np.array(trainPreds).flatten()
                #trainAcc = metrics.accuracy_score(trainPredsFlat, trainGroundTruthFlat)
                trainPredSents = unvectorize_sentences(trainPreds, indexToChar)
                trainPredSents = color_results(trainPredSents, trainCopyScores, trainGenScores, charToIndex)
                
                
                # TEST
                testPreds, testCopyScores, testGenScores = sess.run([pred_op, copy_scores, gen_scores], feed_dict=test_feed_dict)
                
                testAcc = 0.0
                for i in range(len(testPreds)):
                    testAcc = normalized_edit_distance(testGroundTruth[i], testPreds[i])
                testAcc /= len(testGroundTruth)
                
                #testPredsFlat = np.array(testPreds).flatten()
                #testAcc = metrics.accuracy_score(testPredsFlat, testGroundTruthFlat)
                testPredSents = unvectorize_sentences(testPreds, indexToChar)
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









