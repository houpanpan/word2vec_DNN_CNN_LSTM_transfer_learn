from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
from NPDLogger import NPDLogger
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import tensorflow as tf
import os,re,sys
import pandas as pd
from sklearn.externals import joblib
import logging

log = NPDLogger(log_file_name="log_dnn",log_severity_level = 'debug',console=True).log()
### For DNN log strean
logging.getLogger().setLevel(logging.INFO)
##
log.debug("Program Started")
vocabulary_size = 100000
embedding_size = 128  # Dimension of the embedding vector.
max_embed=64
hidden_units = [1000,64]
dir_st="_".join([str(i) for i in hidden_units])
model_dir="/abinitio/dev/data/ml/embed_"+ dir_st
log.debug("Model Dir: %s" %model_dir)

def line_tokenize(Linelist=None):
	data_2dlist = []
	for i in xrange(len(Linelist)):
		data_2dlist.append(tf.compat.as_str(Linelist[i]).split())
	return data_2dlist


def Word2index2d(data_2dlist=None,dictionary=None):
	NewWord2index2d_inner = []
	NewWord2index2d = []
	for i in xrange(len(data_2dlist)):
		for j in xrange(len(data_2dlist[i])):
			try: NewWord2index2d_inner.append(dictionary[data_2dlist[i][j]])
			except: NewWord2index2d_inner.append(0)
		NewWord2index2d.append(NewWord2index2d_inner)
		NewWord2index2d_inner = []
	return NewWord2index2d



def input_gen(max_embed=None,embedding_size=None,NewWord2index2d=None):
        _input = []
        for i in xrange(len(NewWord2index2d)):
                ln = len(NewWord2index2d[i])
                max_sz = max_embed * embedding_size
                if ln < max_embed:
                        _rec = embeddings[NewWord2index2d[i]]
                        zeros = np.zeros((max_embed-ln)*embedding_size)
                        _input.append(np.hstack((_rec.reshape(_rec.size),zeros)))
                else:
                        _input.append(embeddings[NewWord2index2d[i][:max_embed]].reshape(max_sz))
        return np.array(_input)


def input_sum_gen(embeddings=None,NewWord2index2d=None):
    _input = []
    for i in xrange(len(NewWord2index2d)):
        _input.append(np.array(embeddings[NewWord2index2d[i]].sum(axis=0)))
    return np.array(_input)


if not os.path.isfile('/abinitio/dev/data/ml/embed_pkl/input_x.pkl'):
        filename ='/abinitio/dev/data/ml/us_ret_final_all.dat'
        dt = pd.read_csv(filename,header=None,delimiter="|",error_bad_lines=0,names=['X','Y'],quoting=3)
        log.debug(dt.describe())

        log.debug("un-pickling the dictionary")
        dictionary = joblib.load('/abinitio/dev/data/ml/embed_pkl/dictionary.pkl')
	#data = joblib.load('/abinitio/dev/data/ml/embed_pkl/data.pkl')
	#count = joblib.load('/abinitio/dev/data/ml/embed_pkl/count.pkl')
	#reverse_dictionary = joblib.load('/abinitio/dev/data/ml/embed_pkl/reverse_dictionary.pkl')

        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        with tf.Session() as session:
                saver = tf.train.Saver()
                saver.restore(session,'/abinitio/dev/data/ml/embed/embeddings.ckpt')
                embeddings = session.run(tf.all_variables())

        embeddings = np.array(embeddings[0])
        embeddings =  np.vstack((np.zeros(128),embeddings))

	data_2dlist = line_tokenize(dt.X.tolist())
	NewWord2index2d = Word2index2d(data_2dlist,dictionary)
###Check long item desc	
	ln = [(len(i),a) for a,i in enumerate(NewWord2index2d)]
        big_ln = [(len(i), a) for a,i in enumerate(NewWord2index2d) if len(i)> max_embed]
        max_big_ln = max(big_ln)
        if len(big_ln) > 10:
                shw=10
        else:
        	shw=len(big_ln)
        print ("Following are very long > than %d" %max_embed)
        print (big_ln)

        print("The biggest one is..")

        print(dt.X.iloc[max_big_ln[1]])
	
	log.debug("input_gen() Started")
	input_x = input_gen(max_embed,embedding_size,NewWord2index2d)
	log.debug("input_gen() Ended")
	log.debug("Pickling /abinitio/dev/data/ml/embed_pkl/input_x.pkl")
	joblib.dump(input_x,'/abinitio/dev/data/ml/embed_pkl/input_x.pkl')

	log.debug("preprocessing.LabelEncoder")
	if not os.path.isfile('/abinitio/dev/data/ml/embed_pkl/le.pkl'):
		le = preprocessing.LabelEncoder()
		le.fit(dt.Y)
		input_y = le.transform(dt.Y)
		joblib.dump(le,'/abinitio/dev/data/ml/embed_pkl/le.pkl')
		joblib.dump(input_y,'/abinitio/dev/data/ml/embed_pkl/input_y.pkl')
	
	else:
		le=joblib.load('/abinitio/dev/data/ml/embed_pkl/le.pkl')
		input_y = le.transform(dt.Y)
		joblib.dump(input_y,'/abinitio/dev/data/ml/embed_pkl/input_y.pkl')
else:
	log.debug("Loading input_x & input_y from pickle")
        input_x = joblib.load('/abinitio/dev/data/ml/embed_pkl/input_x.pkl')
	input_y = joblib.load('/abinitio/dev/data/ml/embed_pkl/input_y.pkl')
	log.debug("Loading input_x & input_y pickle loading is complete")
	

X_train, X_test, y_train, y_test = train_test_split(input_x,input_y,test_size=0.33, random_state=2)

log.debug("Define Model")

feature_shape = X_train.shape[1]
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=feature_shape)]
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=hidden_units, n_classes=33,  model_dir=model_dir)

log.debug("Training Started")
classifier.fit(x=X_train, y=y_train,   steps=200)
log.debug("Traning Ended")

log.debug("Evaluation Started")
accuracy_score = classifier.evaluate(x=X_test, y=y_test)["accuracy"]
print (accuracy_score)
log.debug("Scores :" + str(accuracy_score))
log.debug("Evaluation End")
