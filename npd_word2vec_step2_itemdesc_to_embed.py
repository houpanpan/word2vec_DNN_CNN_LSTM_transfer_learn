import tensorflow as tf
import pandas as pd
from sklearn.externals import joblib
from NPDLogger import NPDLogger
import numpy as np
import sys, os, re
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
#from scipy import ndimage
from sklearn.cluster import KMeans

log = NPDLogger(log_file_name=sys.argv[0],log_severity_level = 'debug',console=True).log()


embed_type='centriod' ## centriod

#embedding_size = 128  # Dimension of the embedding vector.
embedding_size = 32
max_embed=32
#filename_X = '/temp0/dev/models/US_reference_data_X.dat'  ## File to be converted to LIST & then embedding
filename_X = '/temp0/dev/models/amzn_items_train.dat'
filename_Y = '/temp0/dev/models/US_reference_data_Y.dat'
model_path = '/temp0/dev/models/word2vec_match/'
log.info("Program Started")
log.info("max_embed :%d" %max_embed)
log.info("filename: %s" %filename_X)
log.info("model_path: %s" %model_path)
log.info("Loading Dictionary from : %s" %(model_path + 'dictionary.pkl'))
dictionary = joblib.load(model_path + 'dictionary.pkl')
vocabulary_size = len(dictionary)
log.info("vocabulary_size: %d" %vocabulary_size)
log.info("Dict Loading complete")
log.info("Dataframe loading Started")
#input_filename = '/abinitio/uat/data/npd_batch/serial/machine_learning/shared/ret_grp_1070230282/training_data/trn01314/training_set_all_trn01314.dat'
pattern = re.compile('[\W_ ]+')

#if not os.path.isfile(filename):
if 1!=1:
        dt = pd.read_csv(input_filename,header=None,delimiter="|",error_bad_lines=0,names=['poiid','outletdepartment','outletclass','outletbrand','outletdescription','industry'],quoting=3,nrows = 100000)
        dt = dt[['outletdepartment','outletclass','outletbrand','outletdescription','industry']]
        log.info("Base record count: %d" %len(dt.index))
        dt = dt.fillna('')
        dt.dropna(how='any',inplace=1)
        log.info("Base record count after dropna: %d" %len(dt.index))
        dt['X'] = dt.outletdepartment.astype(str) + ' ' + dt.outletclass.astype(str) + ' ' + dt.outletbrand.astype(str) + ' ' + dt.outletdescription.astype(str)
        dt['Y'] = dt.industry
        #dt = dt.ix[np.random.choice(dt.index.values, 600000)]
        dt = dt[['X','Y']]
        dt['X'] =dt['X'].apply(lambda x: pattern.sub(' ',x).strip())
        dt.drop_duplicates(inplace=1)
	print dt.groupby(['Y']).count()
        #log.debug(dt.describe())
        dt.to_csv(filename,columns=['X','Y'],index=False,header=False)
	#dt = pd.read_csv(filename,header=None,names=['X'],nrows = 100000)

dt = pd.read_csv(filename_X,header=None,names=['itemid','X'],sep="|")
#dt_Y = pd.read_csv(filename_Y,header=None,names=['Y'])
print dt.head()
log.info("Dataframe loading complete")

#log.info("embeddings is being extracted from tensor model")
#embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
#with tf.Session() as session:
#	saver = tf.train.Saver()
#	saver.restore(session,model_path + 'embeddings.ckpt')
#	embeddings = session.run(tf.all_variables())
#log.info("Embeddeding tensor variable extracted")
log.info("Unpickling embeddings")
embeddings = joblib.load(model_path + 'embeddings.pkl')
#embeddings = np.array(embeddings[0])  ## Get the first tf variable
#embeddings =  np.vstack((np.zeros(embedding_size),embeddings))	
log.info("Unpickling embeddings -  done: Shape -> %s" %repr(embeddings.shape))	

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

def Word2index2d_nozero(data_2dlist=None,dictionary=None):
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

def input_gen_sum(NewWord2index2d=None):
    _input = []
    for i in xrange(len(NewWord2index2d)):
        _input.append(np.array(embeddings[NewWord2index2d[i]].sum(axis=0)))
    return np.array(_input)
       
def input_gen_centriod(NewWord2index2d=None):
    _input = []
    for i in xrange(len(NewWord2index2d)):
        _input.append(KMeans(n_clusters=1,max_iter=1,n_init=1, random_state=0).fit(embeddings[NewWord2index2d[i]]).cluster_centers_[0])
    return np.array(_input)

def get_input_y():
	log.debug("preprocessing.LabelEncoder")
	if not os.path.isfile(model_path + 'le.pkl'):
		le = preprocessing.LabelEncoder()
		le.fit(dt_Y.Y)
		input_y = le.transform(dt_Y.Y)
		joblib.dump(le,model_path + 'le.pkl')
		joblib.dump(input_y,model_path + 'input_y.pkl')
	else:
		le=joblib.load(model_path + 'le.pkl')
		input_y = le.transform(dt_Y.Y)
		joblib.dump(input_y,model_path + 'input_y.pkl')

#get_input_y() 
log.info("line_tokenize started")
data_2dlist = line_tokenize(dt.X.tolist())
log.info("line_tokenize ended")
if embed_type=='sum':
	log.info("Word2index2d Started")
	NewWord2index2d = Word2index2d(data_2dlist,dictionary)
	log.info("input_gen_sum Started")
	input_x_sum = input_gen_sum(NewWord2index2d)
	log.info("input_gen_sum Ended")
	joblib.dump(input_x_sum, model_path +'input_x_sum.pkl')
	log.info("input_x_sum pickled")
else:
	if embed_type=='centriod':
		log.info("Word2index2d Started")
                NewWord2index2d = Word2index2d_nozero(data_2dlist,dictionary)
		log.info("input_gen centriod  Started")
                input_x = input_gen_centriod(NewWord2index2d)
                log.info("input_gen Ended")
		#input_x = ndimage.measurements.center_of_mass(input_x.reshape(input_x.shape[0],32,32))	
                joblib.dump(input_x, model_path +'input_x_centriod.pkl')
                log.info("input_x_centriod pickled")
	else:
		log.info("Word2index2d Started")
		NewWord2index2d = Word2index2d(data_2dlist,dictionary)
		log.info("input_gen Started")
        	input_x = input_gen(max_embed,embedding_size,NewWord2index2d)
        	log.info("input_gen Ended")
		joblib.dump(input_x, model_path +'input_x.pkl')
		log.info("input_x pickled")

del NewWord2index2d
log.info("Program Ended")

