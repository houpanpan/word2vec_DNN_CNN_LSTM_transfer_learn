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
#from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
import os,re,sys
import pandas as pd
from sklearn.externals import joblib
import logging
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

num = sys.argv[1]
log = NPDLogger(log_file_name="log_dnn2",log_severity_level = 'debug',console=True).log()
### For DNN log strean
#logging.getLogger().setLevel(logging.DEBUG)
tf.logging.set_verbosity(tf.logging.DEBUG)
##
log.debug("Program Started")
#vocabulary_size = 190000
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
log.debug("Model Dir: %s" %model_dir)
too_big=False
split_size=300000

def _start_shell(local_ns=None):
  # An interactive shell is useful for debugging/development.
	import IPython
	user_ns = {}
	if local_ns:
		user_ns.update(local_ns)
	user_ns.update(globals())
	IPython.start_ipython(argv=[], user_ns=user_ns)


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


def input_gen_steps_pickled(max_embed=None,embedding_size=None,NewWord2index2d=None):
        _input = []
        global_cnt = 0
        for i in xrange(len(NewWord2index2d)):
                ln = len(NewWord2index2d[i])
                max_sz = max_embed * embedding_size
                if ln < max_embed:
                        _rec = embeddings[NewWord2index2d[i]]
                        zeros = np.zeros((max_embed-ln)*embedding_size)
                        _input.append(np.hstack((_rec.reshape(_rec.size),zeros)))
                else:
                        _input.append(embeddings[NewWord2index2d[i][:max_embed]].reshape(max_sz))
                if i%split_size == 0: #and i > 0:
			log.debug("Pickling .. :" + 'input_x.'+repr(global_cnt)+'.pkl')
                       	global_cnt = global_cnt +1
                       	joblib.dump(np.array(_input),input_pickle_dir + 'input_x.'+repr(global_cnt)+'.pkl')
			log.debug("input_x.%d.pkl length in %d Iteration is :%d" %(global_cnt,i,len(_input)))
                       	_input=[]
	global_cnt = global_cnt +1
	log.debug("Pickling .. Final One:" + 'input_x.'+repr(global_cnt)+'.pkl')
        joblib.dump(np.array(_input),input_pickle_dir + 'input_x.'+repr(global_cnt)+'.pkl')
        return True


def input_sum_gen(embeddings=None,NewWord2index2d=None):
    _input = []
    for i in xrange(len(NewWord2index2d)):
        _input.append(np.array(embeddings[NewWord2index2d[i]].sum(axis=0)))
    return np.array(_input)

model='svc'
already_embbed=1
#if not os.path.isfile(pickle_dir + 'input_x.0.pkl'):
if model == 'dnn':
        #filename ='/abinitio/dev/data/ml/us_ret_final_all.dat'
        dt = pd.read_csv(filename,header=None,delimiter="|",error_bad_lines=0,names=['X','Y'],quoting=3, nrows=split_size*3)
	#dt = dt[(dt['Y'] == 'APPAREL') | (dt['Y'] == 'FOOTWEAR') | (dt['Y'] == 'TECHNOLOGY') | (dt['Y'] == 'ACCESSORIES')]
	dt = dt[(dt['Y'] != 'OTHER')]
        log.debug(dt.describe())
	log.debug("preprocessing.LabelEncoder")
        if not os.path.isfile(input_pickle_dir + 'le.pkl'):
                le = preprocessing.LabelEncoder()
                le.fit(dt.Y)
                input_y = le.transform(dt.Y)
                joblib.dump(le,input_pickle_dir + 'le.pkl')
                joblib.dump(input_y,input_pickle_dir + 'input_y.pkl')

        else:
                le=joblib.load(input_pickle_dir + 'le.pkl')
                input_y = le.transform(dt.Y)
                joblib.dump(input_y,input_pickle_dir + 'input_y.pkl')
	#sys.exit(0)
        log.debug("un-pickling the dictionary")
        dictionary = joblib.load(input_pickle_dir + '/dictionary.pkl')
	vocabulary_size = len(dictionary)

	#data = joblib.load('/abinitio/dev/data/ml/embed_pkl/data.pkl')
	#count = joblib.load('/abinitio/dev/data/ml/embed_pkl/count.pkl')
	#reverse_dictionary = joblib.load('/abinitio/dev/data/ml/embed_pkl/reverse_dictionary.pkl')

        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        with tf.Session() as session:
                saver = tf.train.Saver()
                saver.restore(session,embed_dir)
                embeddings = session.run(tf.all_variables())

        embeddings = np.array(embeddings[0])
        embeddings =  np.vstack((np.zeros(128),embeddings))

	data_2dlist = line_tokenize(dt.X.tolist())
	NewWord2index2d = Word2index2d(data_2dlist,dictionary)
	del data_2dlist
###Check long item desc	
	ln = [len(i) for a,i in enumerate(NewWord2index2d)]
	bg=max(ln)
	print("The biggest is %d" %bg)
	if bg < max_embed:
		max_embed = bg
        big_ln = [(len(i), a) for a,i in enumerate(NewWord2index2d) if len(i)>= max_embed]
	max_big_ln = max(big_ln)
        if len(big_ln) > 10:
               	shw=10
        else:
        	shw=len(big_ln)
	joblib.dump(max_embed,input_pickle_dir+'max_embed.pkl')
        print ("Following are very long > than %d" %max_embed)
        print (big_ln)

       	print("The biggest one is..")

       	print(dt.X.iloc[max_big_ln[1]])
	
	log.debug("input_gen() Started")
	if too_big:
		pickles = [input_pickle_dir+ i  for i in os.listdir(input_pickle_dir) if "input_x" in i.split(".")]
		if input_pickle_dir + 'input_x0.pkl' in pickles:
			input_x = joblib.load(input_pickle_dir + 'input_x0.pkl')
		else:
			input_gen_steps_pickled(max_embed,embedding_size,NewWord2index2d)
	else:
		input_x = input_gen(max_embed,embedding_size,NewWord2index2d)
		log.debug("Pickling " + input_pickle_dir + 'input_x.pkl')
        	joblib.dump(input_x,input_pickle_dir + 'input_x.pkl')

	del NewWord2index2d
	log.debug("input_gen() Ended")

	del dt
else:
	if already_embbed == 1 and model == 'dnn':
		log.debug("Loading input_x & input_y from pickle")
	#try: p_input_x = joblib.load(input_pickle_dir + 'input_x.'+repr(int(num)-1)+'.pkl')
	#except: p_input_x=np.array([0])
        #input_x = joblib.load(input_pickle_dir + 'input_x.'+num+'.pkl')
	#input_y = joblib.load(input_pickle_dir + 'input_y.pkl')
		input_x = joblib.load(input_pickle_dir + 'input_x.pkl')
        	input_y = joblib.load(input_pickle_dir + 'input_y.pkl')

	#input_y = input_y[1:split_size+1]
	#input_y = input_y[len(p_input_x)-1:(len(p_input_x) + len(input_x)-1)]
	#input_y = input_y[split_size*(int(num)-2):split_size*(int(num)-1)]	
	#del p_input_x
		log.debug("Loading input_x: %d & input_y: %d pickle loading is complete" %(len(input_x),len(input_y)))
	

log.debug("Model: %s" %model)
if model == 'svc':
	dt = pd.read_csv(filename,header=None,delimiter="|",error_bad_lines=0,names=['X','Y'],quoting=3, nrows=split_size*3)
        #dt = dt[(dt['Y'] == 'APPAREL') | (dt['Y'] == 'FOOTWEAR') | (dt['Y'] == 'TECHNOLOGY') | (dt['Y'] == 'ACCESSORIES')]
        dt = dt[(dt['Y'] != 'OTHER')]
	log.debug("svc define Model")	
	pipeline1 = Pipeline([('tfidf', TfidfVectorizer()),('clf', LinearSVC())])
	clf=pipeline1.set_params(clf__multi_class='ovr')
	log.debug("svc Classifcation/Eval started")
	scores1 = cross_validation.cross_val_score(clf,dt.X,dt.Y,cv=2,n_jobs=-1, scoring='f1_macro')

	print (scores1)
	log.debug("Scores svc:" + str(scores1))
	log.debug("svc Classifcation/Eval Ended")
if model == 'dnn':
	X_train, X_test, y_train, y_test = train_test_split((input_x - input_x.mean())  ,input_y,test_size=0.20, random_state=20)
	#del input_x

	#validation_metrics = { "accuracy": tf.contrib.learn.metric_spec.MetricSpec( metric_fn=tf.contrib.metrics.streaming_accuracy, prediction_key=tf.contrib.learn.prediction_key.PredictionKey. CLASSES), "precision": tf.contrib.learn.metric_spec.MetricSpec( metric_fn=tf.contrib.metrics.streaming_precision, prediction_key=tf.contrib.learn.prediction_key.PredictionKey. CLASSES), "recall": tf.contrib.learn.metric_spec.MetricSpec( metric_fn=tf.contrib.metrics.streaming_recall, prediction_key=tf.contrib.learn.prediction_key.PredictionKey. CLASSES) }
	val_monitor = tf.contrib.learn.monitors.ValidationMonitor(X_test, y_test,every_n_steps=300, early_stopping_rounds=200) #,metrics=validation_metrics) 
	log.debug("dnn Define Model")

	feature_shape = X_train.shape[1]
	joblib.dump(feature_shape,input_pickle_dir+'feature_shape.pkl')
	n_classes = len(set(input_y))
	print (set(input_y))
	joblib.dump(n_classes,input_pickle_dir+'n_classes.pkl')
	log.debug("X_train.Shape (%d,%d)" %(X_train.shape[0],feature_shape))
	log.debug("Number of Classes :%d" %n_classes)
	feature_columns = [tf.contrib.layers.real_valued_column("", dimension=feature_shape)]
	optimizer = tf.train.AdagradOptimizer(learning_rate=0.00001)
	classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,optimizer=optimizer,activation_fn=tf.nn.relu,hidden_units=hidden_units, n_classes=n_classes,  model_dir=model_dir, config=tf.contrib.learn.RunConfig(save_checkpoints_secs=10))

	log.debug("dnn Training Started")
	#pickles = [pickle_dir+ i  for i in os.listdir(pickle_dir) if "input_x" in i.split(".")]
	try: classifier.fit(x=X_train, y=y_train, steps=3500,batch_size=128,monitors=[val_monitor])
	except: log.debug("dnn Training Ended with Error")
	log.debug("dnn Training Ended")

	#log.debug("dnn Evaluation Started")
	#accuracy_score = classifier.evaluate(x=X_test, y=y_test)["accuracy"]
	#print (accuracy_score)
	#log.debug("Scores dnn:" + str(accuracy_score))
	#log.debug("dnn Evaluation End")
	#for i in classifier.get_variable_names():
#		print (classifier.get_variable_value(i))
	#_start_shell(locals())
