from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
mport os
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

log_severity_level = 'debug'
log = NPDLogger(log_file_name="log",log_severity_level = log_severity_level,console=True).log()
model_path = '/temp0/dev/models/word2vec_match/'
pattern = re.compile('[\W_ ]+')
#filename = '/temp0/dev/models/training_set_all_trn01314_token.dat'
input_names=['poiid','outletdepartment','outletclass','outletbrand','outletdescription','industry']

#names=['poiid', 'businessid', 'itemid', 'posoutlet', 'outletdepartment', 'outletclass', 'sku', 'outletbrand', 'outletitemnumber', 'outletdescription', 'manufacturercode','ppweekfrom']
#output_names=['poiid', 'businessid', 'itemid', 'posoutlet', 'outletdepartment', 'outletclass', 'sku', 'outletbrand', 'outletitemnumber', 'outletdescription', 'manufacturercode','ppweekfrom']
output_names = ['outletdepartment','outletclass','outletbrand','outletdescription','industry']
filename_X ='/temp0/dev/models/amzn_items_train.dat'

filename_Y = '/temp0/dev/models/US_reference_data_Y.dat'
input_filename = '/abinitio/uat/data/npd_batch/serial/machine_learning/shared/ret_grp_1070230282/training_data/trn01314/training_set_all_trn01314.dat'
#input_filename = '/temp0/dev/models/turgay_WM_reference_data.dat'
#if not os.path.isfile(filename_X):
if 1!=1:
	dt = pd.read_csv(input_filename,header=None,delimiter="|",error_bad_lines=0,names=input_names,quoting=3,nrows = 5000000)
	log.info("Base record count: %d" %len(dt.index))
	dt = dt.fillna('')
	dt.dropna(how='any',inplace=1)
	#dt['X'] = dt[output_names].apply(lambda x: ' '.join(x.astype(str)), axis=1)
	dt['Y'] = dt['industry']
	log.info("Base record count after dropna: %d" %len(dt.index))
	dt['X'] = dt.outletdepartment.astype(str) + ' ' + dt.outletclass.astype(str) + ' ' + dt.outletbrand.astype(str) + ' ' + dt.outletdescription.astype(str)
	#dt['X'] = dt.poiid.astype(str) + ' ' +  dt.businessid.astype(str) + ' ' +  dt.itemid.astype(str) + ' ' +  dt.posoutlet.astype(str) + ' ' +  dt.outletdepartment.astype(str) + ' ' +  dt.outletclass.astype(str) + ' ' +  dt.outletclass.astype(str) + ' ' +  dt.sku.astype(str) + ' ' +  dt.outletbrand.astype(str) + ' ' +  dt.outletitemnumber.astype(str) + ' ' +  dt.outletdescription.astype(str) + ' ' +  dt.manufacturercode.astype(str)
	#dt['Y'] = dt.industry
	#dt = dt.ix[np.random.choice(dt.index.values, 600000)] 
	#dt = dt['X']
	#dt['X'] = dt.outletbrand.astype(str) + ' ' +dt.outletdescription.astype(str)
	#dt = dt['X']
	dt['X'] =dt['X'].apply(lambda x: pattern.sub(' ',x).strip())
	#dt['X'] = dt['X'].apply(lambda x: x.replace('|',' '))
	dt.drop_duplicates(inplace=1)
	#log.debug(dt.describe())
	dt.to_csv(filename_X,columns=['X'],index=False,header=False)
	dt.to_csv(filename_Y,columns=['Y'],index=False,header=False)

#del dt
#sys.exit(0)


# Read the data into a list of strings.
def read_data(filename):
  with open(filename) as f:
    data = tf.compat.as_str(f.read()).split()
  return data

if not os.path.isfile(model_path +'dictionary.pkl'):
	words = read_data(filename_X)
	log.debug('Data size %d' %len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
#vocabulary_size = 5000
#max_vocabulary_size = 190000
min_cnt = 1
#vocabulary_size = int(0.8 * len(words))

def build_dataset(words):
  count = [['UNK', -1]]
  #count.extend(collections.Counter(words).most_common(max_vocabulary_size - 1))
  count.extend(collections.Counter(words).most_common())
  dictionary = dict()
  for word, cnt in count:
    if cnt > min_cnt and len(word) > 1:
      try: dictionary[word] = len(dictionary)
      except: 
        print(word + ' -- '+str(cnt))
        pass
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

if not os.path.isfile(model_path +'dictionary.pkl'):
	data, count, dictionary, reverse_dictionary = build_dataset(words)
	log.debug("pickling the dictionary")
	joblib.dump(dictionary, model_path +'dictionary.pkl')
	joblib.dump(data, model_path +'data.pkl')
	joblib.dump(count,model_path +'count.pkl')
	joblib.dump(reverse_dictionary,model_path +'reverse_dictionary.pkl')

	del words  # Hint to reduce memory.
else:
	log.debug("un-pickling the dictionary")
	dictionary = joblib.load(model_path +'dictionary.pkl')
	data =  joblib.load(model_path +'data.pkl')
	count = joblib.load(model_path + 'count.pkl')
	reverse_dictionary = joblib.load(model_path + 'reverse_dictionary.pkl')

vocabulary_size = len(dictionary)
log.debug ("vocabulary_size :%s" %vocabulary_size)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0 # batch size should be in multiple of the number of skips, ex - if batch size = 128 then skip should be at least 128
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
      '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
#embedding_size = 128  # Dimension of the embedding vector.
embedding_size = 32
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels, num_sampled, vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
  similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.initialize_all_variables()

# Step 5: Begin training.
#num_steps = 100001
num_steps = 867538

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  #saver = tf.train.Saver([embeddings])
  init.run()
  log.debug("Initialized")

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      log.info("Average loss at step " + repr(step) +  ": " +repr(average_loss))
      if average_loss <= 2.5:
        break
      average_loss = 0
      

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 50000 == 0:
      sim = similarity.eval()
      valid_embeddings.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        log.debug(log_str)
  final_embeddings = normalized_embeddings.eval()
  final_embeddings = np.array(final_embeddings)
  joblib.dump(final_embeddings, model_path + 'embeddings.pkl')
  #saver.save(session,model_path + 'embeddings.ckpt')

# Step 6: Visualize the embeddings.

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
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

try:
  from sklearn.manifold import TSNE
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  log.debug("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
log.debug("Word2Vec is complete")
#with tf.Session() as session:
 #  ...:     saver = tf.train.Saver()
#     saver.restore(session,'/abinitio/dev/data/ml/embed/normalized_embeddings')

   #...:     embeddings = session.run(tf.all_variables())

