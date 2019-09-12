#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os, sys
import time
import datetime
import data_helpers
#from text_cnn import TextCNN
from text_cnn_joint import TextCNN_joint
from tensorflow.contrib import learn
from NPDLogger import NPDLogger
# Parameters
# =================================================
print sys.argv
timetamp = str(int(sys.argv[1]))
log_severity_level = 'debug'
log = NPDLogger(log_file_name=sys.argv[0]+'_'+timetamp,log_severity_level = log_severity_level,console=True).log()
log.info("**** Program START *****************")

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("input_filename", "/temp0/dev/models/temp_file2_1train.dat", "Data source nput items.")
tf.flags.DEFINE_string("timestamp", timetamp, "Experiment Number")


# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 32, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.5, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 3, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("model_path", "/temp0/dev/models/cnn_text/", "path to the model checkpoint")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
    log.info("{}={}".format(attr.upper(), value))
print("")



# Data Preparation
# ==================================================

# Load data
print("Loading data...")
log.info("Loading data...")

x_text_test, x_text_train, y,y_brand = data_helpers.load_data_and_labels_joint(FLAGS.input_filename)
del x_text_test

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text_train]) 
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text_train)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
y_brand_shuffled = y_brand[shuffle_indices]
# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
y_brand_train, y_brand_dev = y_brand_shuffled[:dev_sample_index], y_brand_shuffled[dev_sample_index:]
log.info("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
log.info("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN_joint(
            sequence_length=x_train.shape[1],
            num_classes_subcategory=y_train.shape[1],
            num_classes_brand = y_brand_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda_sc=FLAGS.l2_reg_lambda,l2_reg_lambda_b=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                #grad_summaries.append(grad_hist_summary)
                #grad_summaries.append(sparsity_summary)
        #grad_summaries_merged = tf.merge_all_summaries([grad_hist_summary,sparsity_summary])

        # Output directory for models and summaries
        #timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(FLAGS.model_path, "runs", FLAGS.timestamp))
        print("Writing to {}\n".format(out_dir))
	log.info("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary_sc = tf.scalar_summary("loss_sc", cnn.loss_sc)
        acc_summary_sc = tf.scalar_summary("accuracy_sc", cnn.accuracy_sc)
	loss_summary_b = tf.scalar_summary("loss_b", cnn.loss_b)
        acc_summary_b = tf.scalar_summary("accuracy_b", cnn.accuracy_b)
	loss_summary = tf.scalar_summary("loss", cnn.loss)

        # Train Summaries
        #train_summary_op = tf_summary.merge([loss_summary, acc_summary, grad_summaries_merged])
	train_summary_op = tf.merge_all_summaries()
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Dev summaries
        #dev_summary_op = tf.merge_all_summaries([loss_summary, acc_summary])
	dev_summary_op = tf.merge_all_summaries()
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        #saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
	saver = tf.train.Saver()

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
	#init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables(), tf.global_variables_initializer())
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch, y_batch_brand):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y_subcategory: y_batch,
              cnn.input_y_brand: y_batch_brand,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss,loss_sc, accuracy_sc, loss_b, accuracy_b = sess.run(
                [train_op, global_step, train_summary_op,cnn.loss, cnn.loss_sc, cnn.accuracy_sc, cnn.loss_b, cnn.accuracy_b],
                feed_dict)
	    #if current_step % FLAGS.checkpoint_every == 0:
            time_str = datetime.datetime.now().isoformat()
	    if step % 100 == 0:
                print("{}: step {}, loss_sc {:g}, loss_sc {:g}, acc_sc {:g}, loss_b {:g}, acc_b {:g}".format(time_str, step,loss, loss_sc, accuracy_sc,loss_b, accuracy_b))
                train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch,y_batch_brand,current_step, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y_subcategory: y_batch,
              cnn.input_y_brand: y_batch_brand,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries,loss, loss_sc, accuracy_sc, loss_b, accuracy_b = sess.run(
                [global_step, dev_summary_op, cnn.loss,cnn.loss_sc, cnn.accuracy_sc, cnn.loss_b, cnn.accuracy_b],
                feed_dict)
            #if current_step % FLAGS.checkpoint_every == 0:
	    time_str = datetime.datetime.now().isoformat()
            print("{}: step {},loss {:g}, loss_sc {:g}, acc_sc {:g}, loss_b {:g}, acc_b {:g}".format(time_str, step,loss, loss_sc, accuracy_sc, loss_b, accuracy_b))
	    log.info("{}: step {}, loss {:g},loss_sc {:g}, acc_sc {:g}, loss_b {:g}, acc_b {:g}".format(time_str, step,loss, loss_sc, accuracy_sc, loss_b, accuracy_b))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train,y_brand_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch,y_brand_batch = zip(*batch)
            train_step(x_batch, y_batch,y_brand_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
		log.info("\nEvaluation:")
                dev_step(x_dev, y_dev,y_brand_dev, writer=dev_summary_writer,current_step = current_step)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
		log.info("Saved model checkpoint to {}\n".format(path))
