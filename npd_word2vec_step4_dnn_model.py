import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import os,re,sys
from NPDLogger import NPDLogger

tf.logging.set_verbosity(tf.logging.DEBUG)

log = NPDLogger(log_file_name="log_mydnn",log_severity_level = 'debug',console=True).log()

log.debug("***** Program Started ******")
embed_type='centroid'
embedding_size = 32  # Dimension of the embedding vector.
max_embed=16
hidden_units = [512,512,32]
dir_st="_".join([str(i) for i in hidden_units])

base_dir='/temp0/dev/models/word2vec/'
filename = base_dir + 'navclass_training_set_LTG_tokenized.dat'
dict_dir=base_dir 
embed_model_dir=base_dir 
save_path = base_dir + 'dnn_'+dir_st
try: os.mkdir(save_path)
except: pass
def _start_shell(local_ns=None):
  # An interactive shell is useful for debugging/development.
	import IPython
	user_ns = {}
	if local_ns:
		user_ns.update(local_ns)
	user_ns.update(globals())
	IPython.start_ipython(argv=[], user_ns=user_ns)

model_dir=base_dir +'dnn1_'+ dir_st
input_pickle_dir=dict_dir  ## Save all pickles in script 2
embed_dir=embed_model_dir + 'embeddings.ckpt'
batch_size = 128
learning_rate=0.001
# Small epsilon value for the BN transform
epsilon = 1e-3
epoc = 2
if embed_type=='sum':
	input_x = joblib.load(input_pickle_dir + 'input_x_sum.pkl')
elif embed_type =='centroid':
	input_x = joblib.load(input_pickle_dir + 'input_x_centriod.pkl')
else:
	input_x = joblib.load(input_pickle_dir + 'input_x.pkl')
input_y = joblib.load(input_pickle_dir + 'input_y.pkl')

X_train, X_test, y_train, y_test = train_test_split(input_x.astype(np.float32), input_y.astype(np.int32),test_size=0.20, random_state=40)

input_dim =X_train.shape[1]
total_samples =  X_train.shape[0]
first_hidden_unit_size = hidden_units[0]
second_hidden_unit_size = hidden_units[1]
third_hidden_unit_size = hidden_units[2]
logit_shape = len(set(input_y))

log.debug("input_dim: %d, total_samples: %d, first_hidden_unit_size: %d, second_hidden_unit_size: %d, third_hidden_unit_size: %d, batch_size: %d, learning_rate: %.5f,epoc: %d, logit_shape: %d" %(input_dim,total_samples,first_hidden_unit_size,second_hidden_unit_size,third_hidden_unit_size,batch_size,learning_rate,epoc, logit_shape))  
steps = total_samples/batch_size
#session = tf.InteractiveSession()
## Feeder
x = tf.placeholder(shape=(None ,input_dim),dtype=tf.float32, name='x')
y = tf.placeholder(shape=(None),dtype=tf.int32,name='y')
## Layer 1
w_1 = tf.Variable(tf.random_normal([input_dim,first_hidden_unit_size],0,1,dtype=tf.float32),dtype=tf.float32, name='w_1')
b_1 = tf.Variable(tf.zeros(shape=first_hidden_unit_size,dtype=tf.float32),dtype=tf.float32, name='b_1')
## Batch Norm Sacle & Beta
scale1 = tf.Variable(tf.ones(shape=first_hidden_unit_size,dtype=tf.float32),dtype=tf.float32, name='scale1')
beta1 = tf.Variable(tf.zeros(shape=first_hidden_unit_size,dtype=tf.float32),dtype=tf.float32, name='beta1')

y_1 = tf.matmul(x,w_1,name='layer1_matmul') + b_1

batch_mean1, batch_var1 = tf.nn.moments(y_1,[0])

y_1_norm = (y_1 - batch_mean1)/tf.sqrt(batch_var1 + epsilon)
y_1_bn = scale1 * y_1_norm + beta1


layer1_output = tf.nn.relu(y_1_bn,name='layer1_activation')
## Layer 2
w_2 = tf.Variable(tf.random_normal([first_hidden_unit_size,second_hidden_unit_size],0,1,dtype=tf.float32),dtype=tf.float32, name='w_2')
b_2 = tf.Variable(tf.zeros(shape=second_hidden_unit_size,dtype=tf.float32),dtype=tf.float32, name='b_2')

scale2 = tf.Variable(tf.ones(shape=second_hidden_unit_size,dtype=tf.float32),dtype=tf.float32, name='scale2')
beta2 = tf.Variable(tf.zeros(shape=second_hidden_unit_size,dtype=tf.float32),dtype=tf.float32, name='beta2')

y_2 = tf.add(tf.matmul(layer1_output,w_2,name='layer2_matmul'), b_2)
batch_mean2, batch_var2 = tf.nn.moments(y_2, [0])
y_2_norm = tf.div(tf.sub(y_2, batch_mean2),tf.sqrt(tf.add(batch_var2,epsilon)))

y_2_bn = tf.add(tf.mul(scale2, y_2_norm) , beta2)


layer2_output = tf.nn.relu(y_2_bn,name='layer2_activation')
## Layer 3
w_3 = tf.Variable(tf.random_normal([second_hidden_unit_size,third_hidden_unit_size],0,1,dtype=tf.float32),dtype=tf.float32, name='w_3')
b_3 = tf.Variable(tf.zeros(shape=third_hidden_unit_size,dtype=tf.float32),dtype=tf.float32, name='b_3')

scale3 = tf.Variable(tf.ones(shape=third_hidden_unit_size,dtype=tf.float32),dtype=tf.float32, name='scale3')
beta3 = tf.Variable(tf.zeros(shape=third_hidden_unit_size,dtype=tf.float32),dtype=tf.float32, name='beta3')

y_3 = tf.matmul(layer2_output,w_3,name='layer3_matmul') + b_3
batch_mean3, batch_var3 = tf.nn.moments(y_3, [0])
y_3_norm =  (y_3 - batch_mean3)/tf.sqrt(batch_var3 + epsilon)

y_3_bn = scale3 * y_3_norm + beta3


layer3_output = tf.nn.relu(y_3_bn,name='layer3_activation')


##logit layer -> This layer outputs un-normalized class scores
w_logit = tf.Variable(tf.random_normal([third_hidden_unit_size,logit_shape],0,1,dtype=tf.float32),dtype=tf.float32, name='w_logit')
b_logit = tf.Variable(tf.zeros(shape=logit_shape,dtype=tf.float32),dtype=tf.float32, name='b_logit')
final_layer_output = tf.matmul(layer3_output,w_logit,name='layer_logit_matmul') + b_logit


loss = tf.nn.sparse_softmax_cross_entropy_with_logits(final_layer_output,y)

opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train = opt.minimize(loss)

global_step = tf.Variable(0,name = 'global_step',trainable=False)
saver = tf.train.Saver()
init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
sm = tf.train.SessionManager()
with sm.prepare_session("", init_op=init, saver=saver, checkpoint_dir=save_path) as ss:
	for j in range(epoc):
		step = ss.run(tf.assign(global_step, global_step+1))
		for i in range(steps):
			ls, tr = ss.run([loss, train],feed_dict={x:X_train[batch_size*i:batch_size*(i+1)],y:y_train[batch_size*i:batch_size*(i+1)]})
			if i%10000 ==0:
				l1,l2,l3 = ss.run([layer1_output,layer2_output,layer3_output],feed_dict={x:X_train[batch_size*i:batch_size*(i+1)],y:y_train[batch_size*i:batch_size*(i+1)]})
				l1 = 1 - np.float(len(np.flatnonzero(l1)))/np.float(len(l1.ravel()))
				l2 = 1 - np.float(len(np.flatnonzero(l2)))/np.float(len(l2.ravel()))
				l3 = 1 - np.float(len(np.flatnonzero(l3)))/np.float(len(l3.ravel()))
				log.info("Epoc: %d - Fraction of Zeros in activation layer(Gradient Kill Ration): %.5f, %.5f, %.5f" %(j,l1,l2,l3))
		correct_prediction = tf.equal(tf.cast(tf.argmax(final_layer_output,1),tf.int32),y)
        	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        	log.info("Epoc: %d - Loss at Step -> %d: %.10f, Acurracy: %.6f" %(j,i,ls.mean(),accuracy.eval(feed_dict={x:X_test, y:y_test})))
		saver.save(ss,os.path.join(save_path, "model.ckpt"),global_step= step)
    	#summary_writer = tf.train.SummaryWriter(save_path, ss.graph)
	correct_prediction = tf.equal(tf.cast(tf.argmax(final_layer_output,1),tf.int32),y)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	log.info("Final Accuracy: %.6f" %accuracy.eval(feed_dict={x:X_test, y:y_test}))	
	joblib.dump(layer2_output.eval(feed_dict={x,input_x.astype(np.float32)}),base_dir +'layer2_output.pkl')
	joblib.dump(layer1_output.eval(feed_dict={x,input_x.astype(np.float32)}),base_dir +'layer1_output.pkl')
	#joblib.dump(w_1.eval(),base_dir + 'w_1.pkl')
	#joblib.dump(b_1.eval(),base_dir + 'b_1.pkl')
	#joblib.dump(w_2.eval(),base_dir + 'w_2.pkl')
        #joblib.dump(b_2.eval(),base_dir + 'b_2.pkl')
	#joblib.dump(w_3.eval(),base_dir + 'w_3.pkl')
        #joblib.dump(b_3.eval(),base_dir + 'b_3.pkl')
	#joblib.dump(scale1.eval(), base_dir + 'scale1.pkl')
	#joblib.dump(beta1.eval(), base_dir + 'beta1.pkl')
	#joblib.dump(scale2.eval(), base_dir + 'scale2.pkl')
        #joblib.dump(beta2.eval(), base_dir + 'beta2.pkl')
	#joblib.dump(scale3.eval(), base_dir + 'scale3.pkl')
        #joblib.dump(beta3.eval(), base_dir + 'beta3.pkl')
#_start_shell(locals())	
