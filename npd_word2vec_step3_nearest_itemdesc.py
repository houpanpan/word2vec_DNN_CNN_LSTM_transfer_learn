import tensorflow as tf
import pandas as pd
from sklearn.externals import joblib
from NPDLogger import NPDLogger
import numpy as np
import sys
import logging

log = NPDLogger(log_file_name=sys.argv[0],log_severity_level = 'error',console=True).log()


embed_type='centriod'
embedding_size = 32  # Dimension of the embedding vector.
max_embed=32

model_path = '/temp0/dev/models/word2vec_/'
filename = '/temp0/dev/models/WM_reference_data_X.dat'
log.info("Program Started")
log.info("filename: %s" %filename)
log.info("max_embed :%d" %max_embed)
log.info("model_path: %s" %model_path)
log.info("Loading Dictionary from : %s" %(model_path + 'dictionary.pkl'))
dictionary = joblib.load(model_path + 'dictionary.pkl')
vocabulary_size = len(dictionary)
log.info("vocabulary_size: %d" %vocabulary_size)
log.info("Dict Loading complete")

log.info("Loading Reverse Dictionary from : %s" %(model_path + 'reverse_dictionary.pkl'))
reverse_dictionary = joblib.load(model_path + 'reverse_dictionary.pkl')
log.info("Reverse Dict Loading complete")


log.info("Dataframe loading Started")
dt = pd.read_csv(filename,header=None,names=['X'])##,nrows = 100000)
log.info("Dataframe loading complete")


log.info("Unpickling embeddings")
embeddings = joblib.load(model_path + 'embeddings.pkl')
log.info("Unpickling embeddings -  done")
	

	
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


def input_gen_sum(NewWord2index2d=None):
    _input = []
    for i in xrange(len(NewWord2index2d)):
        _input.append(np.array(embeddings[NewWord2index2d[i]].mean(axis=0)))
    return np.array(_input)

input_line = sys.argv[1]        
#print input_line
log.info("line_tokenize started")
data_2dlist = line_tokenize([input_line])
#print data_2dlist
log.info("line_tokenize ended")

log.info("Word2index2d Started")
NewWord2index2d = Word2index2d(data_2dlist,dictionary)
#print NewWord2index2d

log.info("Get Individual word vec 2D of the input")
input_words_vec = embeddings[NewWord2index2d[0]]
#print "input_words_vec"
#print input_words_vec

sim = np.dot(input_words_vec, embeddings.T)
#print "sim"
#print sim

#for i in range(len(sim)):
#	print "data_2dlist[0][i]"
#	print data_2dlist[0][i]
#	print "(-sim).argsort()[0]"
#	print (-sim).argsort()[i][1]
#	print reverse_dictionary[(-sim).argsort()[i][1]]
#        print sim[i].shape
#	print("Nearest to: %s -> %s, %s" %(data_2dlist[0][i],reverse_dictionary[(-sim).argsort()[i][1]],reverse_dictionary[(-sim).argsort()[i][2]]))

del data_2dlist
log.info("Word2index2d Ended")
log.info("input_gen Started")
if embed_type=='sum':
	input_b = input_gen_sum(NewWord2index2d)
	log.info("Loading input_x_sum")
	input_a = joblib.load(model_path +'input_x_sum.pkl')

else:
	input_b = input_gen(max_embed,embedding_size,NewWord2index2d)
	log.info("Loading input_x")
	input_a = joblib.load(model_path +'input_x.pkl')

print "input_b.shape"
print input_b.shape
del NewWord2index2d
print "input_a.shape"
print input_a.shape
log.info("input_x complete")
log.info("Define OPS")
sim = np.dot(input_a,input_b.T)
top_k=4
#print sim
print sim.shape
nearest = (-sim.reshape(sim.size)).argsort()[0:top_k] 

#print nearest
print input_line
print dt.shape
print dt.X.iloc[nearest ].tolist()

log.info("Program Ended")

