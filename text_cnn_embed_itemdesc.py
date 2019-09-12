#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import pandas as pd
import os, sys
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
from sklearn.externals import joblib
from NPDLogger import NPDLogger

from subprocess import Popen, PIPE
#import os, sys, re,cx_Oracle, errno
# Parameters
# ==================================================
timetamp = str(int(sys.argv[1]))
log_severity_level = 'debug'
log = NPDLogger(log_file_name=sys.argv[0]+'_'+timetamp,log_severity_level = log_severity_level,console=True).log()
log.info("**** Program START *****************")
# Data Parameters
tf.flags.DEFINE_string("input_filename_train", "/temp0/dev/models/temp_file_videoscan_train_train.dat", "Data source input training items acting as reference.")
tf.flags.DEFINE_string("input_filename_test", "/temp0/dev/models/temp_file_videoscan_train_test.dat", "Data source input test items that will be matched to the reference embedded")

# Eval Parameters
#tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "/temp0/dev/models/word2vec_cnn/runs/"+timetamp+"/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_string("embed_input_dir", "/temp0/dev/models/word2vec_cnn/embed_input/"+timetamp+"/", "Save input intertms of embedding concat")
#tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

body = ""

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
	print("{}={}".format(attr.upper(), value))
	body = body + "{}={}".format(attr.upper(), value) + '\n'
	log.info("{}={}".format(attr.upper(), value))
print("")

if not os.path.exists(FLAGS.embed_input_dir):
	os.makedirs(FLAGS.embed_input_dir)
    
subject = "CNN_TEXT -" + timetamp

def send_mail(subject, body,email_recepients='anjani.sharma@npd.com'):
        read_msg = Popen(["echo", body], stdout=PIPE)
       	mail = Popen(["mail", "-s", subject, email_recepients], stdin=read_msg.stdout, stdout=PIPE)
	output = mail.communicate()[0]

def get_x_y(input_filename,data_type, nrows=0):
    #input_names=['poiid','itemid','posoutlet','subcategoryn','brand','X_train','X_test']
    if data_type == 'train':
        input_names=['itemid','Y','X_train','X_test']
    else:
        input_names=['itemid','X_test']
    if nrows ==0:
        dt = pd.read_csv(input_filename,header=None,delimiter="|",error_bad_lines=0,names=input_names,quoting=3)
    else:
        dt = pd.read_csv(input_filename,header=None,delimiter="|",error_bad_lines=0,names=input_names,quoting=3,nrows = nrows)
    dt = dt.fillna('')
    if data_type == 'train':
        x_text = [data_helpers.clean_str(sent) for sent in dt.X_train.tolist()]
    else:
        x_text = [data_helpers.clean_str(sent) for sent in dt.X_test.tolist()]
   
    vocab_path = os.path.join(FLAGS.checkpoint_dir,"..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x = np.array(list(vocab_processor.transform(x_text)))
    y_itemid = dt['itemid'].astype(np.int32)
    log.info("%s ItemId Distinct count: %d" %(data_type,len(set(y_itemid))))
    #y_subcategoryn = dt['itemid'].astype(np.int32)
    #log.info("%s Subcategory Distinct count: %d" %(data_type,len(set(y_subcategoryn))))
    return (x,y_itemid) #,y_subcategoryn)

def convert_from_embed_raw(x, pickle=False):
	#if os.path.isfile(FLAGS.embed_input_dir + 'embedding.pkl'):
	#	return joblib.load(FLAGS.embed_input_dir + 'embedding.pkl')
	embedding = joblib.load('/temp0/dev/models/word2vec_cnn/embeddings.pkl')
	vocab_size = embedding.shape[0]
	embedding_size = embedding.shape[1]
	checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
	graph = tf.Graph()
	with graph.as_default():
		session_conf = tf.ConfigProto(
		allow_soft_placement=FLAGS.allow_soft_placement,
		log_device_placement=FLAGS.log_device_placement)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
			saver.restore(sess, checkpoint_file)
			input_x = graph.get_operation_by_name("input_x").outputs[0]
			embedding_W = tf.placeholder(tf.float32, [vocab_size, embedding_size])
			embedded_chars = tf.nn.embedding_lookup(embedding_W, input_x)
			all_predictions = sess.run(embedded_chars, {input_x: x,embedding_W: embedding})
			print all_predictions.shape
			all_predictions = all_predictions.reshape(all_predictions.shape[0],all_predictions.shape[1]*all_predictions.shape[2])	
			print all_predictions.shape
			if pickle:
				joblib.dump(all_predictions, FLAGS.embed_input_dir + 'embedding_raw.pkl')
			return all_predictions


def convert_from_embed(x, pickle=False):
	#if os.path.isfile(FLAGS.embed_input_dir + 'embedding.pkl'):
	#	return joblib.load(FLAGS.embed_input_dir + 'embedding.pkl')
	checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
	graph = tf.Graph()
	with graph.as_default():
		session_conf = tf.ConfigProto(
		allow_soft_placement=FLAGS.allow_soft_placement,
		log_device_placement=FLAGS.log_device_placement)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
			saver.restore(sess, checkpoint_file)
			input_x = graph.get_operation_by_name("input_x").outputs[0]
			embedding_W = graph.get_operation_by_name("embedding/W").outputs[0]
			embedded_chars = tf.nn.embedding_lookup(embedding_W, input_x)
			all_predictions = sess.run(embedded_chars, {input_x: x})
			print all_predictions.shape
			all_predictions = all_predictions.reshape(all_predictions.shape[0],all_predictions.shape[1]*all_predictions.shape[2])	
			print all_predictions.shape
			if pickle:
				joblib.dump(all_predictions, FLAGS.embed_input_dir + 'embedding.pkl')
			return all_predictions


#  Load raw files	
log.info("Running get_x_y - test")
x, y_itemid = get_x_y(input_filename=FLAGS.input_filename_test,data_type = 'test')
log.info("Running get_x_y - ref")
x_train, y_train_itemid = get_x_y(input_filename=FLAGS.input_filename_train, data_type = 'train')


log.info("ItemId Distinct count not in train: %d" %(len(set(y_itemid) - set(y_train_itemid))))

log.info("Running convert_from_embed - test")
input_b_embed =  convert_from_embed(x =x )
log.info ("input_b_embed.shape: %s" %repr(input_b_embed.shape))

log.info("Running convert_from_embed  - ref")
input_a_embed =  convert_from_embed(x = x_train,pickle=False )
log.info ("input_a_embed.shape: %s" %repr(input_a_embed.shape))

log.info ("Start Embed similarity")
sim_embed = np.dot(input_a_embed,input_b_embed.T)
log.info ("sim_embed.shape: %s" %repr(sim_embed.shape))
log.info ("End embed similarity")

log.info("Getting embed argmax")
match_embed = np.argmax(sim_embed,axis=0)
log.info("match_embed.shape: %s" %repr(match_embed.shape))

log.info("Get Predicted by embed")
predict_embed_itemid = y_train_itemid[match_embed]


log.info("Getting scores")
s_embed_item = "Score: Embed - itemid %.5f"%np.mean(np.equal(y_itemid,predict_embed_itemid))
log.info(s_embed_item)
body = body + "\n" +s_embed_item 

send_mail( subject=subject, body=body)
log.info("**** Program END *************")
