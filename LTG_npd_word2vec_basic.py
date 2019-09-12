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

log_severity_level = 'debug'
log = NPDLogger(log_file_name="log",log_severity_level = log_severity_level,console=True).log()
# Step 1: Download the data.
#url = 'http://mattmahoney.net/dc/'
pattern = re.compile('[\W_ ]+')
#filename = '/abinitio/dev/data/ml/us_ret_final_apparel.dat'
base_dir='/pub1/dev/PUBLISH/serial_00/ml/LTG/'
input_filename =base_dir+ 'navclass_training_set_LTG_20160318_v2.dat'
filename = base_dir + 'navclass_training_set_LTG_tokenized.dat'
dict_dir=base_dir + 'ltg_embed_pkl/'
model_dir=base_dir + 'ltg_dnn_model'
#if not os.path.isfile(filename):
log.info("Input file is being loaded")
if 1==1:
	dt = pd.read_csv(input_filename,header=None,delimiter="|",error_bad_lines=0,names=['poiid','posoutlet','retname','outletdepartmentname','outletclassname','outletbrand','outletdescription','industry','a','b','c','d','industry_name','e','f','g','h'],quoting=3)
	#dt = dt[dt['industry_name'] != 'Other']
	dt = dt[['retname','outletdepartmentname','outletclassname','outletbrand','outletdescription','industry_name']]
	dt = dt.fillna('')
	dt.dropna(how='any',inplace=1)
	dt['X']=dt['retname'].apply(lambda x: x.upper())+' '+dt['outletdepartmentname'].apply(lambda x: x.upper())+' '+dt['outletclassname'].apply(lambda x: x.upper()) + ' '+ dt['outletbrand'].apply(lambda x: x.upper())+' '+dt['outletdescription'].apply(lambda x: x.upper())
	dt['Y'] = dt['industry_name'].apply(lambda x: x.upper())
	#dt = dt.ix[np.random.choice(dt.index.values, 600000)] 
	dt = dt[['X','Y']]
	dt['X'] =dt['X'].apply(lambda x: pattern.sub(' ',x).strip())
	dt.drop_duplicates(inplace=1)
	log.debug(dt.describe())
	log.debug(dt.Y.unique())
	#dt=dt['retname'].apply(lambda x: pattern.sub(' ',x).strip().upper())+' '+dt['outletdepartmentname'].apply(lambda x: pattern.sub(' ',x).strip().upper())+' '+dt['outletclassname'].apply(lambda x: pattern.sub(' ',x).strip().upper()) + ' '+ dt['outletbrand'].apply(lambda x: pattern.sub(' ',x).strip().upper())+' '+dt['outletdescription'].apply(lambda x: pattern.sub(' ',x).strip().upper())+' '+dt['industry_name'].apply(lambda x: pattern.sub(' ',x).strip().upper())
	dt.to_csv(filename,sep='|',index=False)

#del dt
#sys.exit(0)
log.info("File loaded")
def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

#filename = maybe_download('text8.zip', 31344016)

# Read the data into a list of strings.
def read_data(filename):
  with open(filename) as f:
    data = tf.compat.as_str(f.read()).replace("|"," ").split()
  return data
log.info("Reading data to tokenize")
words = read_data(filename)
log.debug('Data size %d' %len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
#vocabulary_size = 5000
#max_vocabulary_size = 190000
min_cnt = 0
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
log.info("Building Dict")
data, count, dictionary, reverse_dictionary = build_dataset(words)
log.debug("pickling the dictionary")
vocabulary_size = len(dictionary)
log.debug ("vocabulary_size :%s" %vocabulary_size)
joblib.dump(dictionary,dict_dir + 'dictionary.pkl')
joblib.dump(data,dict_dir +'data.pkl')
joblib.dump(count,dict_dir +'count.pkl')
joblib.dump(reverse_dictionary,dict_dir +'reverse_dictionary.pkl')

del words  # Hint to reduce memory.
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
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
epoc = 2
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
num_steps =592950

with tf.Session(graph=graph) as session:
  init.run()
  # We must initialize all variables before we use them.
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(session, ckpt.model_checkpoint_path)
    embeddings = session.run(tf.all_variables())
  else:
    saver = tf.train.Saver([embeddings])
    #saver = tf.train.Saver(session,model_dir)
  #saver.restore(session,model_dir)
  #init.run()
  log.debug("Initialized")

  for ep in range(epoc):
    average_loss = 0
    for step in xrange(num_steps):
      batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
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
        log.info("Average loss at Epoc: "+ repr(ep) + "at step " + repr(step) +  ": " +repr(average_loss))
        average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
      if step % 10000 == 0:
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
    saver.save(session,model_dir+'/embeddings.ckpt' + '-' + repr(ep))

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

