from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile
from NPDLogger import NPDLogger
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import tensorflow as tf
import os,re,sys
import pandas as pd
from sklearn.externals import joblib

from sklearn import cross_validation
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

log_severity_level = 'debug'
log = NPDLogger(log_file_name="log",log_severity_level = log_severity_level,console=True).log()
pattern = re.compile('[\W_ ]+')

filename ='/abinitio/dev/data/ml/us_ret_final_all.dat'
dt = pd.read_csv(filename,header=None,delimiter="|",error_bad_lines=0,names=['X','Y'],quoting=3)

log.debug(dt.describe())


log.debug("un-pickling the dictionary")

dictionary = joblib.load('/abinitio/dev/data/ml/embed_pkl/dictionary.pkl')
data = joblib.load('/abinitio/dev/data/ml/embed_pkl/data.pkl')
count = joblib.load('/abinitio/dev/data/ml/embed_pkl/count.pkl')
reverse_dictionary = joblib.load('/abinitio/dev/data/ml/embed_pkl/reverse_dictionary.pkl')

def line_tokenize(Linelist):
	data_2dlist = []
	for i in xrange(len(Linelist)):
		data_2dlist.append(tf.compat.as_str(Linelist[i]).split())
	return data_2dlist


def Word2index2d(data_2dlist):
	NewWord2index2d_inner = []
	NewWord2index2d = []
	for i in xrange(len(data_2dlist)):
		for j in xrange(len(data_2dlist[i])):
			try: NewWord2index2d_inner.append(dictionary[data_2dlist[i][j]])
			except: NewWord2index2d_inner.append(0)
		NewWord2index2d.append(NewWord2index2d_inner)
		NewWord2index2d_inner = []
	return NewWord2index2d

vocabulary_size = 100000
embedding_size = 128  # Dimension of the embedding vector.
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
with tf.Session() as session:
	saver = tf.train.Saver()
	saver.restore(session,'/abinitio/dev/data/ml/embed/embeddings.ckpt')
	embeddings = session.run(tf.all_variables())

embeddings = np.array(embeddings[0])
embeddings =  np.vstack((np.zeros(128),embeddings))

data_2dlist = line_tokenize(dt.X.tolist())
NewWord2index2d = Word2index2d(data_2dlist)

# check bad data
max_embed=64
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

def input_gen(max_embed,embedding_size):
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


#input_x = input_gen(max_embed,embedding_size)

def input_sum_gen():
    _input = []
    for i in xrange(len(NewWord2index2d)):
        _input.append(np.array(embeddings[NewWord2index2d[i]].sum(axis=0)))
    return np.array(_input)

input_x = input_sum_gen()
log.debug("input_sum_gen() executed")
pipeline1 = Pipeline([('clf', LinearSVC())])

clf=pipeline1.set_params(clf__multi_class='ovr')
log.debug("Classifcation started")
scores1 = cross_validation.cross_val_score(clf,input_x,dt.Y,cv=2,n_jobs=-1, scoring='f1_macro')

print (scores1)
log.debug("Scores :" + str(scores1))

log.debug("Classification end")
