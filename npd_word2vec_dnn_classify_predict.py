import tensorflow as tf
import pandas as pd
from sklearn.externals import joblib
import numpy as np
import sys

embedding_size = 128  # Dimension of the embedding vector.
max_embed=64
hidden_units = [64,32]
dir_st="_".join([str(i) for i in hidden_units])
base_dir='/pub1/dev/PUBLISH/serial_00/ml/LTG/'
filename = base_dir + 'navclass_training_set_LTG_tokenized.dat'
dict_dir=base_dir + 'ltg_embed_pkl/'
embed_model_dir=base_dir + 'ltg_dnn_model'

model_dir=base_dir +'dnn1_'+ dir_st
input_pickle_dir=dict_dir  ## Save all pickles in script 2
embed_dir=embed_model_dir + 'embeddings.ckpt'

split_size=300000

dictionary = joblib.load(input_pickle_dir + '/dictionary.pkl')
vocabulary_size = len(dictionary)




embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
with tf.Session() as session:
	saver = tf.train.Saver()
	saver.restore(session,embed_dir)
	embeddings = session.run(tf.all_variables())
	
embeddings = np.array(embeddings[0])
embeddings =  np.vstack((np.zeros(128),embeddings))	
	
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
        
le=joblib.load(input_pickle_dir + 'le.pkl')
print sys.argv[1]
data_2dlist = line_tokenize([sys.argv[1]])
print data_2dlist

NewWord2index2d = Word2index2d(data_2dlist,dictionary)
print NewWord2index2d
max_embed = 51
input_x = input_gen(max_embed,embedding_size,NewWord2index2d)
print input_x

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=6528)]
n_classes = len(le.classes_) -1
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, model_dir=model_dir,hidden_units=hidden_units, n_classes=n_classes)


print le.inverse_transform(classifier.predict(np.array(input_x)))
print classifier.predict_proba(np.array(input_x))

