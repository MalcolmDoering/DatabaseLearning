'''
Created on May 23, 2019

@author: robovie


read the csv log files from the data-driven memory training runs and create graphs for each column

'''


import os
from collections import OrderedDict
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy


#plt.style.use('seaborn-white')


#expLogDir = "E:/eclipse-log/2019-05-17_11-30-24_actionPrediction12_dbl - save" # 2 training databases, presented in 20190524 meeting
#expLogDir = "E:/eclipse-log/2019-05-21_14-54-06_actionPrediction13_dbl - save" # 10 training databases, presented in 20190524 meeting

#expLogDir = "E:/eclipse-log/2019-05-24_19-00-07_actionPrediction13_dbl - GT DB entries" # 10 databases, GT database entries given
#expLogDir = "E:/eclipse-log/2019-05-31_11-34-31_actionPrediction13_dbl" # 10 databases, GT database entries given, and the new metrics, only shopkeeper responses to question about price included in data

#expLogDir = "E:/eclipse-log/2019-06-03_18-17-08_actionPrediction13_dbl" # 10 databases, GT database entries given, DB entries padded with 0 vecs
#expLogDir = "E:/eclipse-log/2019-06-05_13-47-27_actionPrediction13_dbl" # 10 databases, GT database entries given, DB entries padded with 0 vecs

#expLogDir = "E:/eclipse-log/2019-06-06_16-33-38_actionPrediction13_dbl" # 10 databases, GT database entries given, DB entries padded with 0 vecs, reduced batch size and unrandomized training instance order
#expLogDir = "E:/eclipse-log/2019-06-06_16-55-22_actionPrediction13_dbl" # 10 databases, GT database entries given, DB entries padded with 0 vecs, reduced batch size and randomized training instance order

#expLogDir = "E:/eclipse-log/2019-06-07_15-05-37_actionPrediction13_dbl" # 10 databases all data, GT database entries given, DB entries padded with 0 vecs, reduced batch size and unrandomized training instance order



#expLogDir = "E:/eclipse-log/2019-06-10_14-30-33_actionPrediction13_dbl" # 10 databases, relu and softmax addressing, DB entries padded with 0 vecs, reduced batch size and unrandomized training instance order, softmaxed over weighted DB entry sums




#expLogDir = "E:/eclipse-log/2019-05-31_15-35-15_actionPrediction13_dbl" # 2 databases, GT database entries given, and the new metrics, only shopkeeper responses to question about price included in data



#expLogDir = "E:/eclipse-log/2019-05-31_17-54-56_actionPrediction13_dbl" # 2 databases, temp grid search, and the new metrics, only shopkeeper responses to question about price included in data
#expLogDir = "E:/eclipse-log/2019-06-03_13-03-18_actionPrediction13_dbl" # 10 databases, temp grid search, and the new metrics, only shopkeeper responses to question about price included in data

#expLogDir = "E:/eclipse-log/2019-06-07_11-47-14_actionPrediction13_dbl" # 10 databases, temp grid search, and the new metrics, only shopkeeper responses to question about price included in data, 50 batch unrandomized


#expLogDir = "E:/eclipse-log/2019-05-27_12-29-06_actionPrediction13_dbl" # 10 databases, sharpening used for addressing
#expLogDir = "E:/eclipse-log/2019-05-28_14-34-02_actionPrediction13_dbl" # 10 databases, sharpening used for addressing, and the new metrics




# 10 databases
# only S_ANSWERS_QUESTION_ABOUT_FEATURE price included
# tanh and softmax addressing
# adam .0001 learning rate
# DB entries padded with 0 vecs
# 50 reduced batch size and unrandomized training instance order
# only utt input to addressing layer and decoder initialization
# result: attr addresses learned but not camera addresses (because loc was not input to addressing layer)
#expLogDir = "E:/eclipse-log/2019-06-06_17-30-06_actionPrediction13_dbl" 

# 10 databases
# only S_ANSWERS_QUESTION_ABOUT_FEATURE price included
# tanh and softmax addressing
# adam .001 learning rate
# DB entries padded with 0 vecs
# 50 reduced batch size and unrandomized training instance order
# combined loc and utt input to addressing layer, loc layer, and decoder initialization ***
# result: camera addresses are learned but not
# Q: why is the attr loc not learned? event though it was learned in the previous experiment? 
# The things that changed are 
# 1) inputs to the addressing layer and decoder init
# 2) learning rate
# 3) removed the dedicated utterance encoder that was used for input to the addressing layer
#expLogDir = "E:/eclipse-log/2019-06-12_18-48-35_actionPrediction13_dbl" 

# 10 databases
# only S_ANSWERS_QUESTION_ABOUT_FEATURE price included
# tanh and softmax addressing
# adam .0001 *** learning rate
# DB entries padded with 0 vecs
# 50 reduced batch size and unrandomized training instance order
# combined loc and utt input to addressing layer, loc layer, and decoder initialization
# result:
#expLogDir = "E:/eclipse-log/2019-06-13_12-00-52_actionPrediction13_dbl" 


# 10 databases
# only S_ANSWERS_QUESTION_ABOUT_FEATURE price included
# tanh and softmax addressing
# adam .0001 learning rate
# DB entries padded with 0 vecs
# 50 reduced batch size and unrandomized training instance order
# combined loc and utt input to addressing layer, loc layer, and decoder initialization
# separate input encoding for the addressing layer
# result: it works!
#expLogDir = "E:/eclipse-log/2019-06-13_17-59-54_actionPrediction13_dbl" 


# 10 databases
# only S_ANSWERS_QUESTION_ABOUT_FEATURE price included
# tanh and softmax addressing
# adam .0001 learning rate
# DB entries padded with 0 vecs
# 50 reduced batch size and unrandomized training instance order
# combined loc and utt input to addressing layer, loc layer, and decoder initialization
# separate input encoding for the addressing layer
# teacher forcing not used for testing
# result: the correct addresses are being learned, but the copynet is not working
# try training without teacher forcing
# how will doing the weighted sum of the DB entry effect the decoding? 
#expLogDir = "E:/eclipse-log/2019-06-14_11-14-59_actionPrediction13_dbl" 


# 10 databases
# only S_ANSWERS_QUESTION_ABOUT_FEATURE price included
# tanh and softmax addressing
# adam .0001 learning rate
# DB entries padded with 0 vecs
# 50 reduced batch size and unrandomized training instance order
# combined loc and utt input to addressing layer, loc layer, and decoder initialization
# separate input encoding for the addressing layer
# teacher forcing not used for training or testing
# result: 
# mixed results, the addresses are learned but it takes a longer time than with teacher forcing
# generalization does not seem stable - train DB substring all correct goes to 100 but then falls back down
#expLogDir = "E:/eclipse-log/2019-06-14_16-55-45_actionPrediction13_dbl" 



# 10 databases
# only S_ANSWERS_QUESTION_ABOUT_FEATURE price included
# tanh and softmax addressing
# adam .0001 learning rate
# DB entries padded with 0 vecs
# 50 reduced batch size and unrandomized training instance order
# combined loc and utt input to addressing layer, loc layer, and decoder initialization
# separate input encoding for the addressing layer
# curriculum learning used for training, nothing for testing
# 
# result: 
#expLogDir = "E:/eclipse-log/2019-06-17_17-10-31_actionPrediction13_dbl"


# 10 databases
# only S_ANSWERS_QUESTION_ABOUT_FEATURE price included
# tanh and softmax addressing
# adam .0001 learning rate
# DB entries padded with 0 vecs
# 50 reduced batch size and unrandomized training instance order
# combined loc and utt input to addressing layer, loc layer, and decoder initialization
# separate input encoding for the addressing layer
# curriculum learning used for training, nothing for testing
# 1.0 - 1.0 / (1.0 + np.exp( - (e-500.0)/100.0))
# result: 
#expLogDir = "E:/eclipse-log/2019-06-18_12-40-37_actionPrediction13_dbl"


# 10 databases
# only S_ANSWERS_QUESTION_ABOUT_FEATURE price included
# tanh and softmax addressing
# adam .0001 learning rate
# DB entries padded with 0 vecs
# 50 reduced batch size and unrandomized training instance order
# combined loc and utt input to addressing layer, loc layer, and decoder initialization
# separate input encoding for the addressing layer
# curriculum learning used for training, nothing for testing
# 1.0 - 1.0 / (1.0 + np.exp( - (e-200.0)/10.0))
# result: 
#expLogDir = "E:/eclipse-log/2019-06-19_17-08-11_actionPrediction13_dbl"


# 10 databases
# only S_ANSWERS_QUESTION_ABOUT_FEATURE price included
# tanh and softmax addressing
# adam .0001 learning rate
# DB entries padded with 0 vecs
# 50 reduced batch size and unrandomized training instance order
# combined loc and utt input to addressing layer, loc layer, and decoder initialization
# separate input encoding for the addressing layer
# 0.3 prob of teacher forcing
# result: 
#expLogDir = "E:/eclipse-log/2019-06-20_11-22-07_actionPrediction13_dbl"


# 10 databases
# only S_ANSWERS_QUESTION_ABOUT_FEATURE price included
# tanh and softmax addressing
# adam .0001 learning rate
# DB entries padded with 0 vecs
# 50 reduced batch size and unrandomized training instance order
# combined loc and utt input to addressing layer, loc layer, and decoder initialization
# separate input encoding for the addressing layer
# 0.6 prob of teacher forcing
# result: 
#expLogDir = "E:/eclipse-log/2019-06-21_11-59-10_actionPrediction13_dbl"


# 10 databases
# only S_ANSWERS_QUESTION_ABOUT_FEATURE price included
# tanh and softmax addressing
# adam .0001 learning rate
# DB entries padded with 0 vecs
# 50 reduced batch size and unrandomized training instance order
# combined loc and utt input to addressing layer, loc layer, and decoder initialization
# separate input encoding for the addressing layer
# 0.3, 0.7, 0.8, 0.9 prob of teacher forcing
# result: 
expLogDir = "E:/eclipse-log/2019-06-24_11-46-31_actionPrediction13_dbl"


# 10 databases
# only S_ANSWERS_QUESTION_ABOUT_FEATURE price included
# tanh and softmax addressing
# adam .0001 learning rate
# DB entries padded with 0 vecs
# 50 reduced batch size and unrandomized training instance order
# combined loc and utt input to addressing layer, loc layer, and decoder initialization
# separate input encoding for the addressing layer
# 0.3, 0.7, 0.8, 0.9 prob of teacher forcing
# tried modifying the copynet so it could use teacher forcing
# result: not good
#expLogDir = "E:/eclipse-log/2019-06-21_18-34-39_actionPrediction13_dbl"


# 10 databases
# only S_ANSWERS_QUESTION_ABOUT_FEATURE price included
# tanh and softmax addressing
# adam .0001 learning rate
# DB entries padded with 0 vecs
# 50 reduced batch size and unrandomized training instance order
# combined loc and utt input to addressing layer, loc layer, and decoder initialization
# separate input encoding for the addressing layer
# 0.3, 0.7, 0.8, 0.9 prob of teacher forcing
# does not use copynet, instead tries to copy entire DB entry at once
# result: 
expLogDir = "E:/eclipse-log/2019-06-24_18-05-48_actionPrediction14_dbl"



expLogDir = "E:/eclipse-log/2019-06-25_17-58-50_actionPrediction14_dbl"


expLogDir = "E:/eclipse-log/2019-06-26_11-34-59_actionPrediction14_dbl"



# 10 databases
# only S_ANSWERS_QUESTION_ABOUT_FEATURE price included
# GT addresses
# adam .0001 learning rate
# all strings padded with 0 vecs, eos char only on output
# 50 reduced batch size and unrandomized training instance order
# 1.0 teacher forcing
# does not use copynet, instead tries to copy entire DB entry at once
# db match entry encoding and input encoding used to initialize decoder
# gen weight and db_read_weight, and copy weight used
# result: reaches 100% in the training but lots of up and down spikes. does not generalize at all to the testing set
expLogDir = "E:/eclipse-log/2019-06-26_18-06-16_actionPrediction14_dbl"


# 10 databases
# only S_ANSWERS_QUESTION_ABOUT_FEATURE price included
# GT addresses
# adam .0001 learning rate
# all strings padded with 0 vecs, eos char only on output
# 50 reduced batch size and unrandomized training instance order
# 1.0 teacher forcing
# does not use copynet, instead tries to copy entire DB entry at once
# db match entry len and input encoding used to initialize decoder
# only db_read_weight used
# result: reaches 100% in the training but lots of up and down spikes. does not generalize at all to the testing set
expLogDir = "E:/eclipse-log/2019-06-27_14-22-39_actionPrediction14_dbl"



# 10 databases
# only S_ANSWERS_QUESTION_ABOUT_FEATURE price included
# GT addresses
# adam .001* learning rate
# all strings padded with 0 vecs, eos char only on output
# 50 reduced batch size and unrandomized training instance order
# 1.0 teacher forcing
# copy net used
# result: bad... What changed since it worked?
# learning rate, 0 padding of inputs and outputs
# removal of eos char from DB entries and inputs
# something else...?
#expLogDir = "E:/eclipse-log/2019-06-27_16-16-40_actionPrediction13_dbl"


# 10 databases
# only S_ANSWERS_QUESTION_ABOUT_FEATURE price included
# GT addresses
# adam .0001* learning rate
# all strings padded with 0 vecs, eos char only on output
# 50 reduced batch size and unrandomized training instance order
# 1.0 teacher forcing
# copy net used
# result: 
#expLogDir = "E:/eclipse-log/2019-06-27_18-23-44_actionPrediction13_dbl"


# 10 databases
# only S_ANSWERS_QUESTION_ABOUT_FEATURE price included
# GT addresses
# adam .0001* learning rate
# db entries padded with 0s and no eos char, inputs and outputs padded with spaces and with eos char
# 50 reduced batch size and unrandomized training instance order
# 1.0 teacher forcing on training and inference
# result: training works, but does not generalize at all to the test set
# what has happened since last time when it worked?!
# it seems that the randomizeTrainingBatches flag was accidentally set to True... Maybe that was the problem
#expLogDir = "E:/eclipse-log/2019-06-27_18-54-45_actionPrediction13_dbl"


# try same as above but without randomization of the training batches
# 10 databases
# only S_ANSWERS_QUESTION_ABOUT_FEATURE price included
# GT addresses
# adam .0001 learning rate
# db entries padded with 0s and no eos char, inputs and outputs padded with spaces and with eos char
# 50 reduced batch size and unrandomized* training instance order
# 1.0 teacher forcing on training and inference
# result: it works. the training instance randomization was causing the problem
# 
#expLogDir = "E:/eclipse-log/2019-06-28_10-55-05_actionPrediction13_dbl"


# try the same as above but with 0 padding on all strings and eos chars on only the outputs
# 10 databases
# only S_ANSWERS_QUESTION_ABOUT_FEATURE price included
# GT addresses
# adam .0001 learning rate
# db entries padded with 0s and no eos char, inputs and outputs padded with spaces and with eos char
# 50 reduced batch size and unrandomized* training instance order
# 1.0 teacher forcing on training and inference
# result: it works. the training instance randomization was causing the problem
# 
#expLogDir = "E:/eclipse-log/2019-06-28_11-23-06_actionPrediction13_dbl"


# same as above but with TF set to .3 and no TF used on inference
#expLogDir = "E:/eclipse-log/2019-06-28_11-57-25_actionPrediction13_dbl"


# same as above but with tanh softmax used for addressing
expLogDir = "E:/eclipse-log/2019-06-28_13-58-36_actionPrediction13_dbl"



# same as above but with GS for addressing, grid search over temps
expLogDir = "E:/eclipse-log/2019-06-28_19-31-28_actionPrediction13_dbl"




# divide copy scores by num char occurrences
# 10 databases
# only S_ANSWERS_QUESTION_ABOUT_FEATURE price included
# softmax addresses
# adam .001 learning rate
# 0 padding on all strings and eos chars on only the outputs
# 32 size and unrandomized* training instance order
# 0.3 teacher forcing on training and no teacher forcing on inference
# result: 
#
expLogDir = "E:/eclipse-log/2019-07-02_17-51-46_actionPrediction13_dbl"

# same as above but with a lower learning rate
# divide copy scores by num char occurrences
# 10 databases
# only S_ANSWERS_QUESTION_ABOUT_FEATURE price included
# softmax addresses
# adam .0001 learning rate
# 0 padding on all strings and eos chars on only the outputs
# 32 size and unrandomized* training instance order
# 0.3 teacher forcing on training and no teacher forcing on inference
# result: 
#
expLogDir = "E:/eclipse-log/2019-07-02_18-21-39_actionPrediction13_dbl"


expLogDir = "E:/eclipse-log/2019-07-03_12-06-43_actionPrediction13_dbl"

# price only, uniform initialization
expLogDir = "E:/eclipse-log/2019-07-03_16-42-14_actionPrediction13_dbl"


# price only, customer utterance variety, normal initialization
expLogDir = "E:/eclipse-log/2019-07-03_17-29-23_actionPrediction13_dbl"

# price only, customer utterance variety, uniform initialization
expLogDir = "E:/eclipse-log/2019-07-04_13-10-38_actionPrediction13_dbl"



# actionPrediction14
# GT addresses
# handmade databases, 2 training
# only S_A_Q_A_F, all features
# bias unit (1.0) added to decoder initialization instead of the DB match sum
# Adam 0.001 learning rate
# teacher forcing for training and inference
# result: did not work, seemed to be learning until epoch ~250, then performance spiked
# overflow encountered in multiply around 10 minuts after sessionDir was generated
# 
expLogDir = "E:/eclipse-log/2019-07-05_17-31-56_actionPrediction14_dbl"



# copynet
# 10 train
# price only
# gumbelSoftmaxTemp = 6.0 * np.exp(-0.0003 * e) + 0.01
expLogDir = "E:/eclipse-log/"

# copynet
# 10 train
# price only
# gumbelSoftmaxTemp = max((-1.0/2000) * e + 3, 0.05)
expLogDir = "E:/eclipse-log/2019-07-08_16-34-31_actionPrediction13_dbl"


# copynet
# 10 train 
# price only
# swicth to GS 0.01 and reset optimizer at 3000
# result: performance dropped at 3000 and addressing stopped working
#expLogDir = "E:/eclipse-log/2019-07-08_18-33-50_actionPrediction13_dbl"

# copynet
# 10 train 
# price only
# switch to GS 0.01 at 3000
# result: performance dropped at 3000 and addressing stopped working
#expLogDir = "E:/eclipse-log/2019-07-09_12-26-09_actionPrediction13_dbl"



# copynet
# 10 train 
# price only
# use softmax till 3000, apply GS with temp 0.01 from 3000
# train whole network till 3000, only train decoding part (not addressing) from 3000
# result: performance dropped at 3000 and addressing stopped working
#expLogDir = "E:/eclipse-log/2019-07-09_13-23-24_actionPrediction13_dbl"



# copynet
# 10 train 
# price only
# use softmax till 3000, apply sharpening with ^10 from 3000
# train whole network till 3000, only train decoding part (not addressing) from 3000
# result: there was a bug in the code. did not swith to train_op_2 as expected
#expLogDir = "E:/eclipse-log/2019-07-09_17-51-05_actionPrediction13_dbl"


# copynet
# 10 train 
# price only
# use softmax till 3000, apply sharpening with ^10 from 3000
# train whole network till 3000, only train decoding part (not addressing) from 3000
# result: 
# successfully stopped the addressing mechanism from being trained further
# there is a decrease in performance when sharpening begins to be used
# the training seems to recover in performance
# but the testing does not seem to recover...
# in the testing (and probably the training too) the correct substrings are being copied/generated. Spaces do not have high copy scores.
# but, some of the test DB substrings are not entirely copied. Ie. they end early...
# perhaps the optimizer should be reset at that point in time too to prevent getting stuck in local minimum, or reinitialize the decoding weights too
#expLogDir = "E:/eclipse-log/2019-07-10_12-51-05_actionPrediction13_dbl"


#expLogDir = "E:/eclipse-log/2019-07-10_16-50-46_actionPrediction13_dbl"


# copynet
# 10 train 
# price only
# use softmax till 3000, apply sharpening with ^10 from 3000
# train whole network till 3000, only train decoding part (not addressing) from 3000
# reinitialize the decoding weights at 3000
# uses dataset with variety of customer utterances
# result: 
# the addressing is successfully frozen. All but 2 runs to get close to 100% accuracy. But, some of the testing runs to not get quite to 100% (i.e. around 95%)
# after reset at 3000, good runs recover testing accuracy (% db substr all correct) to 100% within 1000 epochs, but then begin to degrade aroun epoch 5000
# for the testing runs the results are not so good. Even runs with 100% addressing in testing do not get close to 100% db substr all correct.
# I thought, maybe the reason for this is that now there are a variety of customer utterances. the samples that are printed in the output files have correct substrs,
# so that means that the substrs are incorrect for some of other variations of customer inputs
# so, try running the same thing but with the old dataset that did not have variety of customer utterances...
expLogDir = "E:/eclipse-log/2019-07-11_12-29-27_actionPrediction13_dbl"


# copynet
# 10 train 
# price only
# use softmax till 3000, apply sharpening with ^10 from 3000
# train whole network till 3000, only train decoding part (not addressing) from 3000
# reinitialize the decoding weights at 3000
# uses dataset without a variety of customer utterances
# result: 
# this run is the same as the previous run but without the variety of customer utterances
# results were very bad - the correct addressing was not even learned.
# so, there is probably a bug in the code somewhere
# try doing another run but with the ground truth DB addressing and see if it works...
#expLogDir = "E:/eclipse-log/2019-07-11_17-49-44_actionPrediction13_dbl"



# copynet
# 10 train 
# price only
# does not do any reset
# uses GT addresses
# uses dataset without a variety of customer utterances
# result: 
expLogDir = "E:/eclipse-log/2019-07-12_12-07-58_actionPrediction13_dbl"


# search over teacher forcing probs for copynet
# price only, simple customer inputs
# GT database addresses
expLogDir = "E:/eclipse-log/2019-07-12_17-04-38_actionPrediction13_dbl"


# tried using TF 0.0 for copynet with GT addresses
# masked losses on output chars over the len of the GT output sentence
expLogDir = "E:/eclipse-log/2019-07-16_12-49-26_actionPrediction13_dbl"



# non-copynet architecture with the copy buffer error fixed
# with softmax addresses, price only, LR 0.0001 Adam
# two input encodings
expLogDir = "E:/eclipse-log/2019-07-17_13-46-14_actionPrediction14_dbl"

# non-copynet architecture with the copy buffer error fixed
# with softmax addresses, price only, LR 0.0001 Adam
# only one input encoding
expLogDir = "E:/eclipse-log/2019-07-17_15-16-11_actionPrediction14_dbl"


# non-copynet architecture with the copy buffer error fixed
# with softmax addresses, all attributes, LR 0.0001 Adam
# two input encodings
# simple customer utterances
expLogDir = "E:/eclipse-log/2019-07-17_16-09-29_actionPrediction14_dbl"


# non-copynet architecture with the copy buffer error fixed
# with softmax addresses, all attributes, LR 0.0001 Adam
# two input encodings
# variety of customer utterances
expLogDir = "E:/eclipse-log/2019-07-17_17-56-59_actionPrediction14_dbl"


# non-copynet architecture with the copy buffer error fixed
# with softmax addresses, all attributes, LR 0.0001 Adam
# two input encodings
# variety of customer utterances
# embedding size 100
#expLogDir = "E:/eclipse-log/2019-07-18_12-22-11_actionPrediction14_dbl"



# non-copynet architecture with the copy buffer error fixed
# with GT addresses
# all data - customer utt variety, all attributes, all actions
# 10 databases with only prices changing
# embedding size 100
#expLogDir = "E:/eclipse-log/2019-07-18_13-34-09_actionPrediction14_dbl - GT addresses all data"


# non-copynet architecture with the copy buffer error fixed
# with softmax addresses
# all data - customer utt variety, all attributes, all actions
# 10 databases with only prices changing
# embedding size 100
expLogDir = "E:/eclipse-log/2019-07-21_14-44-40_actionPrediction14_dbl - SM addresses all data"


# non-copynet architecture with the copy buffer error fixed
# with GT addresses
# all data - customer utt variety, all attributes, all actions
# 2 handmade databases with all attributes changing
# embedding size 100
#expLogDir = "E:/eclipse-log/"


# non-copynet architecture with the copy buffer error fixed
# with softmax addresses
# all data - customer utt variety, all attributes, all actions
# 2 handmade databases with all attributes changing
# embedding size 100
#expLogDir = "E:/eclipse-log/2019-07-22_16-55-13_actionPrediction14_dbl"


expLogDir = "E:/eclipse-log/2019-07-23_17-38-43_actionPrediction14_dbl"

expLogDir = "E:/eclipse-log/2019-07-24_17-48-26_actionPrediction14_dbl"

# non-copynet with RNN interaction history encoder
expLogDir = "E:/eclipse-log/2019-07-26_19-59-46_actionPrediction15_dbl"


# non-copynet with RNN interaction history encoder 1000 dim lsa utt vec
# freeze addressing, apply sharpening, and reinitialize optimizer and decoding weights at 1000 epochs
#expLogDir = "E:/eclipse-log/2019-07-29_18-53-49_actionPrediction15_dbl"

# non-copynet with RNN interaction history encoder unigram utt vec
expLogDir = "E:/eclipse-log/2019-07-31_16-34-21_actionPrediction15_dbl"


def plot_2_conditions_3_metrics(runIdToData, runDirNames, metric1Name, metric2Name, metric3Name):
    
    fig, axes = plt.subplots(3, 2, sharex='col', sharey='row')
    
    
    cmap = plt.get_cmap("tab20")
    colors = list(cmap.colors)
    runIdToColor = {}
    
    ymax = 1.05
    
    i = 0
    for runId in runIdToData:
        runIdToColor[runId] = colors[i % len(colors)]
        i += 1
    
    
        
    
        
        # training
        # graph Cost Ave
        if metric1Name == "Cost Ave":
            metric1Ymax = 100
        else:
            metric1Ymax = ymax
        
        runIdToData[runId].plot(x="Epoch", y="Train {} ({})".format(metric1Name, runId), ax=axes[0,0],
                                color=runIdToColor[runId],
                                legend=None,
                                label=runId,
                                ylim=[0, metric1Ymax])
        
        # graph Substring Correct All
        runIdToData[runId].plot(x="Epoch", y="Train {} ({})".format(metric2Name, runId), ax=axes[1,0],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, ymax])
        
        
        # graph Substring Correct Ave
        runIdToData[runId].plot(x="Epoch", y="Train {} ({})".format(metric3Name, runId), ax=axes[2,0],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, ymax])
        
            
        # testing
        
        # graph Cost Ave
        if metric1Name == "Cost Ave":
            yColName = "Test {}({})".format(metric1Name, runId) # there's a typo in these column names (missing space)
            metric1Ymax = 100
        else:
            yColName = "Test {} ({})".format(metric1Name, runId)
            metric1Ymax = ymax
        
        
        runIdToData[runId].plot(x="Epoch", y=yColName, ax=axes[0,1],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, metric1Ymax])
        
        
        # graph Substring Correct All
        runIdToData[runId].plot(x="Epoch", y="Test {} ({})".format(metric2Name, runId), ax=axes[1,1],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, ymax])
        
        
        # graph Substring Correct Ave
        runIdToData[runId].plot(x="Epoch", y="Test {} ({})".format(metric3Name, runId), ax=axes[2,1],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, ymax])
    
    
    
    plt.legend(runDirNames,
               loc="upper center",   # Position of legend
               borderaxespad=0.1,    # Small spacing around legend box
               title="Run Parameters - rs (random seed), ct (camera temp.), at (attribute temp.)",
               
               # for 120 run gridsearch
               #ncol=12,
               #bbox_to_anchor=(-0.05, -0.2)
               
               # for 8 runs
               ncol=8,
               bbox_to_anchor=(0, -.5)
               )
    
    
    
    #
    # plot the prob for teacher forcing
    #
    for runId in runDirNames:
        """
        try:
            axes2_00 = axes[0,0].twinx()  # instantiate a second axes that shares the same x-axis
            axes2_10 = axes[1,0].twinx()
            axes2_20 = axes[2,0].twinx()
            
            axes2_01 = axes[0,1].twinx()
            axes2_11 = axes[1,1].twinx()
            axes2_21 = axes[2,1].twinx()
            
            
            axes2_00.set_ylim(0, 1)
            axes2_10.set_ylim(0, 1)
            axes2_20.set_ylim(0, 1)
            
            axes2_01.set_ylim(0, 1)
            axes2_11.set_ylim(0, 1)
            axes2_21.set_ylim(0, 1)
            
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_00,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_10,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_20,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_01,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_11,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_21,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            
            axes2_11.set_ylabel("Teacher Forcing Decay Schedule", rotation=90, size='medium')
            
        except:
            pass
        """
    
    
        
    # Adjust the scaling factor to fit your legend text completely outside the plot
    # (smaller value results in more space being made for the legend)
    
    # for 120 run gridsearch
    #plt.subplots_adjust(bottom=.3)
    
    # for 8 runs
    plt.subplots_adjust(bottom=.2)
    
    
    cols = ["Training", "Testing"]
    rows = [metric1Name, metric2Name, metric3Name]
    
    
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)
    
    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, rotation=90, size='medium')
    
    
    
    plt.subplots_adjust(wspace=.1, hspace=.05)
    #fig.tight_layout()
    
    
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i,j].xaxis.set_tick_params(which='both', direction="in", length=5)
            axes[i,j].yaxis.set_tick_params(which='both', direction="in", length=5)
            
    
    
    plt.show()




#
# read the data 
#

runDirNames = os.listdir(expLogDir)
runDirNames.sort()


temp = []
"""
for rdn in runDirNames:
    
    #if "ct3_" in rdn and rdn.endswith("at2"):
    if rdn.endswith("tf1.0"):
        temp.append(rdn)

runDirNames = temp
"""

# this will contain the data from all the csv log files
runIdToData = {}

for rdn in runDirNames:
    
    runIdToData[rdn] = pd.read_csv("{}/{}/session_log_{}.csv".format(expLogDir, rdn, rdn))



#
# graph the data
#
plot_2_conditions_3_metrics(runIdToData, runDirNames, "Cost Ave", "DB Substring Correct Ave", "DB Substring Correct All")

plot_2_conditions_3_metrics(runIdToData, runDirNames, "Cam. Address Correct", "Attr. Address Correct", "Both Addresses Correct")




