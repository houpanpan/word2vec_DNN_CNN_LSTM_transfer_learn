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
import datetime
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


log_severity_level = 'debug'
log = NPDLogger(log_file_name="log",log_severity_level = log_severity_level,console=True).log()
embedding_size_arg = '50'
window_size_arg = '1'
model_path = '/temp0/dev/models/vdwscan/word2vec/'
embedding_path = model_path + '/'+embedding_size_arg+"_"+window_size_arg+"/"

log.debug("un-pickling the dictionary")
dictionary = joblib.load(model_path +'dictionary.pkl')
reverse_dictionary = joblib.load(model_path + 'reverse_dictionary.pkl')
#w_1 = joblib.load(model_path + 'w_1.pkl')
#b_1 = joblib.load(model_path + 'b_1.pkl')

vocabulary_size = len(dictionary)
log.debug ("vocabulary_size :%s" %vocabulary_size)

final_embeddings = joblib.load(embedding_path + 'embedding_glove.pkl')
#final_embeddings = np.dot(final_embeddings,w_1) + b_1
dtm=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
filename= 'tsne_after_model_' + dtm + '.png'
def plot_with_labels(low_dim_embs, labels, filename=filename):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)
  plt.close(filename)


tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 500
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
labels = [reverse_dictionary[i] for i in xrange(plot_only)]
plot_with_labels(low_dim_embs, labels)

