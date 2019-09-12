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
	def __init__(self,opttype):
		self.batch_size = 128
		self.learning_rate = 0.01
		self.epoc = 10
		self.hidden_units = [512,512,32]
		self.epsilon = 1e-3
		self.base_dir = '/temp0/dev/models/word2vec_match/'
		try: os.mkdir(self.base_dir)
                except: pass #raise #NameError("Cannot create directory: %s" %self.base_dir)
		self.dir_tag = 'dnn1_'
		dir_str = "_".join([str(i) for i in self.hidden_units])
		self.save_path = self.base_dir + self.dir_tag +dir_str
		try: os.mkdir(self.save_path)
		except: pass #raise NameError("Cannot create directory: %s" %self.save_path)
		self.embed_type = 'centriod' ## sum
		if opttype == 'train':
			if self.embed_type=='sum':
        			self.input_x = joblib.load(self.base_dir + 'input_x_sum.pkl')
			elif self.embed_type =='centriod':
        			self.input_x = joblib.load(self.base_dir + 'input_x_centriod.pkl')
			else:
        			self.input_x = joblib.load(self.base_dir + 'input_x.pkl')
			try: self.input_y = joblib.load(self.base_dir + 'input_y.pkl')
			except: self.input_y = np.array([])	
			self.input_dim = self.input_x.shape[1]
			self.logit_shape = len(set(self.input_y))
			self.test_size = 0.20
			self.total_samples = int(self.input_x.shape[0]*self.test_size)
			self.dictionary = joblib.load(self.base_dir + 'dictionary.pkl')
                	self.vocabulary_size = len(self.dictionary)
			self.reverse_dictionary = joblib.load(self.base_dir+ 'reverse_dictionary.pkl')
			self.filename = '/temp0/dev/models/US_reference_data_X.dat'
			self.dt = pd.read_csv(self.filename,header=None,names=['X'])
			self.embeddings = joblib.load(self.base_dir + 'embeddings.pkl')
			self.top_k=4
			self.industry_lookup = pd.read_csv('/temp0/dev/models/industry_shortname_lookup.dat',header=None,delimiter="|",names=['code','industry'])
			self.industry = pd.read_csv('/temp0/dev/models/US_reference_data_Y.dat',header=None,names = ['code'])
			
			
		else:
			self.input_dim = 32
			self.total_samples = 0
		self.logit_shape = 30
		self.batch_norm = True
		self.steps = self.total_samples/self.batch_size

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


#class dnn(object): ## Incomplete
#	def __init__(self,options,session):
#	self.options = options
#	self.session = session
#	self.dnn_layer()
#	self.dnn_logits()
	
		
def dnn_layer(name_scope,input_op,input_dim, output_dim, batch_norm=True,epsilon=None):
	with tf.name_scope(name_scope):
		w = tf.Variable(tf.random_normal([input_dim,output_dim],0,1,dtype=tf.float32),dtype=tf.float32, name='w')
		b = tf.Variable(tf.zeros(shape=output_dim,dtype=tf.float32),dtype=tf.float32, name='b')
		scale = tf.Variable(tf.ones(shape=output_dim,dtype=tf.float32),dtype=tf.float32, name='scale')
        	beta = tf.Variable(tf.zeros(shape=output_dim,dtype=tf.float32),dtype=tf.float32, name='beta')
        	y = tf.matmul(input_op,w,name='matmul') + b
		if batch_norm == True:
        		batch_mean, batch_var = tf.nn.moments(y,[0])
        		y_norm = (y - batch_mean)/tf.sqrt(batch_var + epsilon)
        		y_bn = scale * y_norm + beta
        		layer_output = tf.nn.relu(y_bn,name='activation')
		else:
			layer_output = tf.nn.relu(y,name='activation')
		tf.histogram_summary(name_scope+'_histogram',layer_output)
		return layer_output
def dnn_logits(input_op,input_dim,logit_shape):
	w_logit = tf.Variable(tf.random_normal([input_dim,logit_shape],0,1,dtype=tf.float32),dtype=tf.float32, name='w_logit')
        b_logit = tf.Variable(tf.zeros(shape=logit_shape,dtype=tf.float32),dtype=tf.float32, name='b_logit')
        layer_output = tf.matmul(input_op,w_logit,name='layer_logit_matmul') + b_logit
	return layer_output

def dnn_network(input_op,input_dim,hidden_units,logit_shape, batch_norm,epsilon):
	ops = []
	if len(hidden_units) == 3:
		layer1_output =  dnn_layer(name_scope='hidden1',input_op=input_op,input_dim=input_dim, output_dim=hidden_units[0],batch_norm=batch_norm,epsilon=epsilon)
		layer2_output =  dnn_layer(name_scope='hidden2',input_op=layer1_output,input_dim=hidden_units[0], output_dim=hidden_units[1],batch_norm=batch_norm,epsilon=epsilon)
		layer3_output =  dnn_layer(name_scope='hidden3',input_op=layer2_output,input_dim=hidden_units[1], output_dim=hidden_units[2],batch_norm=batch_norm,epsilon=epsilon)
		layer_output = dnn_logits(input_op=layer3_output,input_dim=hidden_units[2],logit_shape=logit_shape)
		ops = [layer1_output,layer2_output,layer3_output,layer_output]
	elif len(hidden_units) == 2:
                layer1_output =  dnn_layer(name_scope='hidden1',input_op=input_op,input_dim=input_dim, output_dim=hidden_units[0],batch_norm=batch_norm,epsilon=epsilon)
                layer2_output =  dnn_layer(name_scope='hidden2',input_op=layer1_output,input_dim=hidden_units[0], output_dim=hidden_units[1],batch_norm=batch_norm,epsilon=epsilon)
                layer_output = dnn_logits(input_op=layer2_output,input_dim=hidden_units[1],logit_shape=logit_shape)
		ops = [layer1_output,layer2_output,layer_output]
	elif len(hidden_units) == 1:
		layer1_output =  dnn_layer(name_scope='hidden1',input_op=input_op,input_dim=input_dim, output_dim=hidden_units[0],batch_norm=batch_norm,epsilon=epsilon)
                layer_output = dnn_logits(input_op=layer1_output,input_dim=hidden_units[0],logit_shape=logit_shape)
		ops = [layer1_output,layer_output]
	return ops

def inference(input_vector,return_layer,opts):
	#opts = Options(opttype='inference')
	if return_layer == -1:
		return input_vector
	x = tf.placeholder(shape=(None ,opts.input_dim),dtype=tf.float32, name='x')
	final_layer_output = dnn_network(input_op=x,input_dim=opts.input_dim,hidden_units=opts.hidden_units,logit_shape = opts.logit_shape,batch_norm=opts.batch_norm,epsilon=opts.epsilon)		
	saver = tf.train.Saver()
	init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
	sm = tf.train.SessionManager()
	with sm.prepare_session("", init_op=init, saver=saver, checkpoint_dir=opts.save_path) as ss:
		final_vector =  final_layer_output[return_layer].eval(feed_dict={x:input_vector})
	ss.close()
	tf.reset_default_graph()
	return final_vector

def transfer_learn(return_layer,opts):
        #opts = Options(opttype='train')
	if return_layer == -1:
		return opts.input_x
        x = tf.placeholder(shape=(None ,opts.input_dim),dtype=tf.float32, name='x')
        final_layer_output = dnn_network(input_op=x,input_dim=opts.input_dim,hidden_units=opts.hidden_units,logit_shape = opts.logit_shape,batch_norm=opts.batch_norm,epsilon=opts.epsilon)
        saver = tf.train.Saver()
        init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        sm = tf.train.SessionManager()
        with sm.prepare_session("", init_op=init, saver=saver, checkpoint_dir=opts.save_path) as ss:
                final_vector =  final_layer_output[return_layer].eval(feed_dict={x:opts.input_x})
	ss.close()
	tf.reset_default_graph()
	return final_vector
        #joblib.dump(final_vector,base_dir + 'input_x_'+self.embed_type+'_'+return_layer+'.pkl')
	
def training():	
	opts = Options('train')

	log.debug("input_dim: %d, total_samples: %d, hidden_units: %s, batch_size: %d, learning_rate: %.5f,epoc: %d, logit_shape: %d" %(opts.input_dim,opts.total_samples,repr(opts.hidden_units),opts.batch_size,opts.learning_rate,opts.epoc, opts.logit_shape))

	x = tf.placeholder(shape=(None ,opts.input_dim),dtype=tf.float32, name='x')
	y = tf.placeholder(shape=(None),dtype=tf.int32,name='y')
	final_layer_output = dnn_network(input_op=x,input_dim=opts.input_dim,hidden_units=opts.hidden_units,logit_shape = opts.logit_shape,batch_norm=opts.batch_norm,epsilon=opts.epsilon)
	final_layer_output = final_layer_output[len(opts.hidden_units)]
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(final_layer_output,y)
	opt = tf.train.GradientDescentOptimizer(learning_rate=opts.learning_rate)
	train = opt.minimize(loss) 
	correct_prediction = tf.equal(tf.cast(tf.argmax(final_layer_output,1),tf.int32),y)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.scalar_summary('Accuracy',accuracy)

	global_step = tf.Variable(0,name = 'global_step',trainable=False)

	summary_op = tf.merge_all_summaries()

	saver = tf.train.Saver()
	init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
	sm = tf.train.SessionManager()
	with sm.prepare_session("", init_op=init, saver=saver, checkpoint_dir=opts.save_path) as ss:
    		train_summary_writer = tf.train.SummaryWriter(opts.save_path+'/train/', ss.graph)
		test_summary_writer = tf.train.SummaryWriter(opts.save_path+'/test/', ss.graph)
		for j in range(opts.epoc):
			X_train, X_test, y_train, y_test = train_test_split(opts.input_x.astype(np.float32), opts.input_y.astype(np.int32),test_size=opts.test_size, random_state=np.random.randint(100))
			step = ss.run(tf.assign(global_step, global_step+1))
			for i in range(opts.steps):
				ls, tr = ss.run([loss, train],feed_dict={x:X_train[opts.batch_size*i:opts.batch_size*(i+1)],y:y_train[opts.batch_size*i:opts.batch_size*(i+1)]})
				if i%1000 ==0:
					summary_step = opts.steps*(step - 1) + i
					train_summary_str = ss.run(summary_op,feed_dict={x:X_train[opts.batch_size*i:opts.batch_size*(i+1)],y:y_train[opts.batch_size*i:opts.batch_size*(i+1)]})
					train_summary_writer.add_summary(train_summary_str,summary_step)
					test_summary_str = ss.run(summary_op,feed_dict={x:X_test, y:y_test})
					test_summary_writer.add_summary(test_summary_str,summary_step)
        		log.info("Epoc: %d - Loss at Step -> %d: %.10f, Acurracy: %.6f" %(j,i,ls.mean(),accuracy.eval(feed_dict={x:X_test, y:y_test})))
			saver.save(ss,os.path.join(opts.save_path, "model.ckpt"),global_step= step)
	ss.close()
	tf.reset_default_graph()
#training()

#output = inference(input_vector=np.random.randn(1,32),return_layer='hidden3')
#print output
def evaluate_transfer_learn():
	opts = Options(opttype='train')
	level = -1
	transfer_learn_hidden1 = transfer_learn(return_layer = level,opts = opts)
	#transfer_learn_hidden2 = transfer_learn(return_layer = 'hidden2')
	#transfer_learn_hidden3 = transfer_learn(return_layer = 'hidden3')
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
			input_vector_hidden1 =  inference(input_vector=input_b,return_layer= level ,opts = opts)
		except: 
			print " No match available for the input string. Pls try another: "
			continue
		#input_vector_hidden2 =  inference(input_vector=input_b,return_layer='hidden2')
		#input_vector_hidden3 =  inference(input_vector=input_b,return_layer='hidden3')
		sim1 = np.dot(input_vector_hidden1,transfer_learn_hidden1.T)
		#sim2 = np.dot(input_vector_hidden2,transfer_learn_hidden2.T)	
		#sim3 = np.dot(input_vector_hidden3,transfer_learn_hidden3.T)

		nearest1 = (-sim1.reshape(sim1.size)).argsort()[0:opts.top_k]
		#nearest2 = (-sim2.reshape(sim2.size)).argsort()[0:opts.top_k]
		#nearest3 = (-sim3.reshape(sim3.size)).argsort()[0:opts.top_k]

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
			#item = item + ' - '+  opts.input_y[nearest1[0]]	
			industry_code = opts.industry.code.iloc[nearest1[cnt]]
			industry_name =  opts.industry_lookup[opts.industry_lookup['code']== industry_code].industry.tolist()[0]
			cnt = cnt + 1
        		print industry_name + ' -> ' + item
		#for item in  opts.dt.X.iloc[nearest2].tolist():
                #        print item
		#for item in  opts.dt.X.iloc[nearest3].tolist():
                #        print item
		print "\n\n"
evaluate_transfer_learn()
