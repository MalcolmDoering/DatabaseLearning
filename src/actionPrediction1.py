'''
Created on Aug 27, 2018

@author: MalcolmD
'''


import numpy as np
import cPickle as pkl

import theano
import theano.tensor as T
import lasagne



#
# simulate some interactions
#
inputs = []
outputs = []


database = ["D | Sony | LOCATION | DISPLAY_1",
            "D | Sony | PRICE | $550",
            "D | Nikon | LOCATION | DISPLAY_2",
            "D | Nikon | PRICE | $68"]

inputs.append(database + ["C | DISPLAY_1 | How much is this one?"])
outputs.append("S | DISPLAY_1 | This one is $550.")

inputs.append(database + ["C | DISPLAY_2 | How much is this one?"])
outputs.append("S | DISPLAY_2 | This one is $68.")



database = ["D | Sony | LOCATION | DISPLAY_1",
            "D | Sony | PRICE | $500",
            "D | Nikon | LOCATION | DISPLAY_2",
            "D | Nikon | PRICE | $60"]

inputs.append(database + ["C | DISPLAY_1 | How much is this one?"])
outputs.append("S | DISPLAY_1 | This one is $500.")

inputs.append(database + ["C | DISPLAY_2 | How much is this one?"])
outputs.append("S | DISPLAY_2 | This one is $60.")




inputBufLens = [len(i) for i in inputs]

inputSentLens = []
for i in inputs:
    for j in i:
        inputSentLens.append(len(j))

outputSentLens = []
for i in outputs:
    outputSentLens.append(len(i))


uniqueChars = []
for i in inputs:
    for j in i:
        for k in j:
            if k not in uniqueChars:
                uniqueChars.append(k)

for i in outputs:
    for j in i:
        if j not in uniqueChars:
            uniqueChars.append(j)


maxBufLen = max(inputBufLens)
maxSentLen = max(inputSentLens+outputSentLens)



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
# encode the inputs as sequences of vectors
# pad the ends of sentences with spaces
#
vectorizedInputs = []

padCharVec = np.zeros(numUniqueChars, dtype="int32")
padCharVec[charToIndex[" "]] = 1

for seq in inputs:
    seqVec = []
    
    for sent in seq:
        
        for i in range(len(sent)):
            
            char = sent[i]
            charVec = np.zeros(numUniqueChars, dtype="int32")
            charVec[charToIndex[char]] = 1
            
            seqVec.append(charVec)
        
        for i in range(maxSentLen - len(sent), maxSentLen):
            seqVec.append(padCharVec)
        
        
        """
        sentVec = np.zeros(maxSentLen*numUniqueChars)
        
        for i in range(len(sent)):
            
            char = sent[i]
            charVec = np.zeros(numUniqueChars, dtype="int32")
            charVec[charToIndex[char]] = 1
            
            sentVec[i*numUniqueChars : i*numUniqueChars+numUniqueChars] = charVec
            
        for i in range(maxSentLen - len(sent), maxSentLen):
            sentVec[i*numUniqueChars : i*numUniqueChars+numUniqueChars] = padCharVec
        
        seqVec.append(sentVec)
        """
        
        """
        sentVec = []
        
        for char in sent:
            charVec = np.zeros(numUniqueChars, dtype="int32")
            charVec[charToIndex[char]] = 1
            sentVec.append(charVec)
            
        sentVec += [padCharVec] * (maxSentLen - len(sentVec)) # padding
        seqVec.append(sentVec)
        """
    vectorizedInputs.append(seqVec)
    







#
# encode the outputs
#
vectorizedOutputs = []

for sent in outputs:
    sentVec = []
    
    for char in sent:
        charVec = np.zeros(numUniqueChars)
        charVec[charToIndex[char]] = 1
        sentVec.append(charVec)
    
    sentVec += [padCharVec] * (maxSentLen - len(sentVec)) # padding
    vectorizedOutputs.append(sentVec)



#
# setup a neural network
#
print "setting up the neural network..."
    
numHidden = 50
batchSize = 4 # None


lasagne.random.set_rng(np.random.RandomState(0))

#l_in = lasagne.layers.InputLayer(shape=(batchSize, maxBufLen, maxSentLen, numUniqueChars))


l_in = lasagne.layers.InputLayer(shape=(batchSize, maxBufLen*maxSentLen, numUniqueChars))


l_buf_encoding_gru = lasagne.layers.GRULayer(l_in,
                                             numHidden,
                                             grad_clipping=5.0,
                                             only_return_final=False)

l_buf_encoding_gru = lasagne.layers.GRULayer(l_buf_encoding_gru,
                                             numHidden,
                                             grad_clipping=5.0,
                                             only_return_final=True)

mergedSlices = []

for i in range(batchSize):
    
    # slice out the ith buf encoding
    l_slice = lasagne.layers.SliceLayer(l_buf_encoding_gru, indices=i, axis=0)
    
    # duplicate it maxSentLen times
    slices = []
    
    for j in range(maxSentLen):
        
        l_reshape_3 = lasagne.layers.ReshapeLayer(l_slice, (1, 1, numHidden))
        slices.append(l_reshape_3)
    
    l_concat_1 = lasagne.layers.ConcatLayer(slices, axis=1)
    mergedSlices.append(l_concat_1)


# merge the duplicates so they are next to each other
l_concat_2 = lasagne.layers.ConcatLayer(mergedSlices, axis=0)


# is this copying the encoding the way I want them to? Or, is it making each input to the GRU the same?
#l_concat = lasagne.layers.ConcatLayer([l_reshape_3]*maxSentLen, axis=1)


l_sent_decoding_gru = lasagne.layers.GRULayer(l_concat_2,
                                              numHidden,
                                              grad_clipping=5.0,
                                              only_return_final=False)


l_sent_decoding_gru = lasagne.layers.GRULayer(l_sent_decoding_gru,
                                              numHidden,
                                              grad_clipping=5.0,
                                              only_return_final=False)


# ?
l_reshape_4 = lasagne.layers.ReshapeLayer(l_sent_decoding_gru, (batchSize*maxSentLen, numHidden))

l_softmax = lasagne.layers.DenseLayer(l_reshape_4, numUniqueChars, W=lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)

l_reshape_5 = lasagne.layers.ReshapeLayer(l_softmax, (batchSize, maxSentLen, numUniqueChars))


network_output = lasagne.layers.get_output(l_reshape_5)

target_output = T.itensor3('target_output')


cost = T.nnet.categorical_crossentropy(network_output, target_output).mean()


all_params = lasagne.layers.get_all_params(l_softmax, trainable=True)

updates = lasagne.updates.adam(cost, all_params)

train = theano.function([l_in.input_var, target_output], cost, 
                        #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True),
                        updates=updates, allow_input_downcast=True)

probs = theano.function([l_in.input_var], network_output, allow_input_downcast=True)


#
# train the network
#
print "training the neural network..."

numEpochs = 50000

for i in range(numEpochs):
    trainError = train(vectorizedInputs, vectorizedOutputs)
    
    if i % 100 == 0 and i > 5000:
        print 
        print i, trainError
        
        probOutputs = probs(vectorizedInputs)
        
        
        #
        # conver outputs to readable sentences
        #
        predictions = []
        
        for i in range(probOutputs.shape[0]):
            prediction = ""
            
            for j in range(probOutputs.shape[1]):
                
                maxIndex = np.argmax(probOutputs[i, j, :])
                char = indexToChar[maxIndex]
                
                prediction += char
            
            predictions.append(prediction)
            
        for i in range(len(outputs)):
            print 
            print "true:", outputs[i]
            print "pred:", predictions[i]
        

np.savez("model", lasagne.layers.get_all_param_values(l_reshape_5))

optimizerState = [p.get_value() for p in updates.keys()]
pkl.dump(optimizerState, open("optimizer", "wb"))
        

#
# test the network
#
probOutputs = probs(vectorizedInputs)
print probOutputs.shape


#
# conver outputs to readable sentences
#
reproducedTrueOutputs = []
predictions = []

for i in range(probOutputs.shape[0]):
    prediction = ""
    
    for j in range(probOutputs.shape[1]):
        
        maxIndex = np.argmax(probOutputs[i, j, :])
        char = indexToChar[maxIndex]
        
        prediction += char
        
        
        
        
        
    #prediction.strip()
    predictions.append(prediction)


for i in range(len(outputs)):
    print 
    print "true:", outputs[i]
    print "pred:", predictions[i]


    
    














