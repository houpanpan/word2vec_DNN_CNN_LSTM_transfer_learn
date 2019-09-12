import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import os,re,sys
from NPDLogger import NPDLogger
from sklearn.cluster import KMeans

tf.logging.set_verbosity(tf.logging.DEBUG)

log = NPDLogger(log_file_name="log_mydnn",log_severity_level = 'debug',console=True).log()

log.debug("***** Program Started ******")

class Options(object):
	def __init__(self):
		self.base_dir = '/temp0/dev/models/word2vec_match/'
		try: os.mkdir(self.base_dir)
                except: pass #raise #NameError("Cannot create directory: %s" %self.base_dir)
		self.embed_type = 'centriod' ## sum
		if self.embed_type=='sum':
        		self.input_x = joblib.load(self.base_dir + 'input_x_sum.pkl')
		elif self.embed_type =='centriod':
        		self.input_x = joblib.load(self.base_dir + 'input_x_centriod.pkl')
		else:
        		self.input_x = joblib.load(self.base_dir + 'input_x.pkl')
		self.input_dim = self.input_x.shape[1]
		self.dictionary = joblib.load(self.base_dir + 'dictionary.pkl')
                self.vocabulary_size = len(self.dictionary)
		self.reverse_dictionary = joblib.load(self.base_dir+ 'reverse_dictionary.pkl')
		self.filename = '/temp0/dev/models/amzn_items_train.dat'
		self.dt = pd.read_csv(self.filename,header=None,names=['itemid','X'],sep="|")
		self.filename_val = '/temp0/dev/models/amzn_items_test.dat'
                self.dt_val = pd.read_csv(self.filename_val,header=None,names=['itemid','X'],sep="|",nrows = 1)
		self.embeddings = joblib.load(self.base_dir + 'embeddings.pkl')
		self.top_k=4

def line_tokenize(Linelist):
        data_2dlist = []
        for i in xrange(len(Linelist)):
                data_2dlist.append(tf.compat.as_str(Linelist[i]).split())
        return data_2dlist


def Word2index2d(data_2dlist,dictionary):
        NewWord2index2d_inner = []
        NewWord2index2d = []
        for i in xrange(len(data_2dlist)):
                for j in xrange(len(data_2dlist[i])):
                        try: NewWord2index2d_inner.append(dictionary[data_2dlist[i][j]])
                        except: NewWord2index2d_inner.append(0)
                NewWord2index2d.append(NewWord2index2d_inner)
                NewWord2index2d_inner = []
        return NewWord2index2d


def Word2index2d_nozero(data_2dlist,dictionary):
        NewWord2index2d_inner = []
        NewWord2index2d = []
        for i in xrange(len(data_2dlist)):
                for j in xrange(len(data_2dlist[i])):
                        try:
                                indx = dictionary[data_2dlist[i][j]]
                                NewWord2index2d_inner.append(indx)
                        except: pass
                NewWord2index2d.append(NewWord2index2d_inner)
                NewWord2index2d_inner = []
        return NewWord2index2d

def input_gen(max_embed,embedding_size,NewWord2index2d,embeddings):
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

def input_gen_sum(NewWord2index2d,embedding_size,embeddings):
    _input = []
    for i in xrange(len(NewWord2index2d)):
        _input.append(np.array(embeddings[NewWord2index2d[i]].sum(axis=0)))
    return np.array(_input)


def input_gen_centriod(NewWord2index2d,embeddings):
    _input = []
    for i in xrange(len(NewWord2index2d)):
        _input.append(KMeans(max_iter=1,n_init=1,n_clusters=1, random_state=0).fit(embeddings[NewWord2index2d[i]]).cluster_centers_[0])
    return np.array(_input)


	
def evaluate_learn():
	opts = Options()
	
	while True:
		input_text = raw_input("Enter the input tem description(Type Q to quiti) : ")
		if input_text == "Q":
			break
		data_2dlist = line_tokenize([input_text])
		try:
			if opts.embed_type=='centriod':
				NewWord2index2d = Word2index2d_nozero(data_2dlist,opts.dictionary)
				input_b = input_gen_centriod(NewWord2index2d,opts.embeddings)
			elif opts.embed_type=='sum':
				NewWord2index2d = Word2index2d(data_2dlist,opts.dictionary)
				input_b = input_gen_sum(NewWord2index2d,opts.embeddings)
			else:
				NewWord2index2d = Word2index2d(data_2dlist,opts.dictionary)
				input_b = input_gen(opts.max_embed,opts.embedding_size,NewWord2index2d,opts.embeddings)
		except: 
			print " No match available for the input string. Pls try another: "
			continue
		sim1 = np.dot(input_b,opts.input_x.T)
		print sim1
		nearest1 = (-sim1.reshape(sim1.size)).argsort()[0:opts.top_k]
		print nearest1

		print "\n\n\n\n\n"
		print "-------------------------------------------------------------------------------------------------"
		print "                                    The Input Item string"
		print "-------------------------------------------------------------------------------------------------"
		print input_text
		print ""
		print "-------------------------------------------------------------------------------------------------"
		print "                             The FOUR Closest matching items are:"
		print "-------------------------------------------------------------------------------------------------"
		print ""
		cnt = 0
		for item in  opts.dt.X.iloc[nearest1].tolist():
        		print item
		print "\n\n"
evaluate_learn()

def eval_item_match_score():
        opts = Options()
        data_2dlist = line_tokenize(opts.dt_val['X'].tolist())
	print data_2dlist
       	if opts.embed_type=='centriod':
       		NewWord2index2d = Word2index2d_nozero(data_2dlist,opts.dictionary)
                input_b = input_gen_centriod(NewWord2index2d,opts.embeddings)
        elif opts.embed_type=='sum':
                NewWord2index2d = Word2index2d(data_2dlist,opts.dictionary)
                input_b = input_gen_sum(NewWord2index2d,opts.embeddings)
        else:
                NewWord2index2d = Word2index2d(data_2dlist,opts.dictionary)
                input_b = input_gen(opts.max_embed,opts.embedding_size,NewWord2index2d,opts.embeddings)
	print input_b.shape
	#print opts.dt_val.X.tolist()
	print opts.input_x.T.shape
        sim1 = np.dot(input_b,opts.input_x.T)
	print sim1.shape
        nearest1 = (-sim1.reshape(sim1.size)).argsort()[0:opts.top_k]
	match = np.argmax(sim1.T,axis=0)
	print match[:10]
	#print match.shape
        val =  opts.dt_val['itemid'].astype(np.int32)
	print opts.dt_val
	#print val.shape
	#print opts.dt.X.iloc[match].tolist()
	predict =  opts.dt.itemid.iloc[match].astype(np.int32)
 	print opts.dt.iloc[match]	
	print opts.dt.iloc[nearest1]
	#print predict.shape
	score = np.mean(np.equal(val,predict))
	log.info("Score: %s" %repr(score))
        print score
#eval_item_match_score()



