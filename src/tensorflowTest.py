#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 15:37:45 2018

@author: malcolm
"""


import tensorflow as tf

with tf.device('/gpu:0'):
    
    hello = tf.constant("Hello, TensorFlow!")
    sess = tf.Session()
    print(sess.run(hello))