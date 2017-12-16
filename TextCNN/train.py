import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import time
import datetime
import sys
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
from tflearn.data_utils import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder

#hyperparameters
#path_to_glove = "glove.6B.100d.txt" #path to glove
embedding_size = 300 #dimension of glove
filter_sizes = "3,4,5" #Size of filter
num_filters = 128 #"Number of filters per filter size
l2_reg_lambda = 0.0 #L2 regularizaion lambda
batch_size = 64 #Batch Size
num_epochs = 6 #Number of training epochs
evaluate_every = 50 #"Evaluate model on dev set after this many steps
checkpoint_every = 50 #Save model after this many steps
showstep_every = 15
num_checkpoints = 5
dropout_keep_prob = 0.5
max_document_length = 180
isMultilabel = False
isSoftwareOnly = True


def training():
	#==============================================================================
	#Step1: Loading data
	print("NOTE: Multilabel is set to",str(isMultilabel))
	print("=="*30)
	print("Step 1: Loading Data")
	print("=="*30)
	if(isSoftwareOnly):
		all_text, y = data_helpers.load_training_data_and_labels_software(sys.argv[1])
	else:
		all_text, y = data_helpers.load_training_data_and_labels(sys.argv[1])
	#print(all_text[0:10])
	#print datasets description
	#==============================================================================
	#Step2: Building Vocabular and Apply pre-trained (i.e., create an embedding lookup table)
	print("=="*30)
	print("Step 2: Building Vocabulary")
	print("=="*30)
	#==============================================================================
	print("	Tokenizing words.....")
	max_length = max([len(data_helpers.tokenize(x)) for x in all_text])#can be a sentence if a file is a sentence
	print("		Max document length:",max_length)
	print("	Building Vocabulary")
	vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length,tokenizer_fn=data_helpers.tokenize_2)
	x = np.array(list(vocab_processor.fit_transform(all_text)))
	vocabulary = vocab_processor.vocabulary_

	vocab_dict = vocab_processor.vocabulary_._mapping
	vocab_inv = {v: k for k, v in vocab_dict.items()}

	#print(vocab_inv)

	x = pad_sequences(x,maxlen=max_document_length)

	#print(x[0:10])
	print("		Total vocabulary:",len(vocabulary))
	print("	Loading Glove, a pre-trained vectors...")
	#can also inlcude Google's word2vec here
	embedding_weights = data_helpers.load_amazon_w2v(vocab_inv)
	# print("	Glove is loaded!!")
	embedding = np.array(embedding_weights, dtype = np.float32)
	# embedding = np.array([v for v in embedding_weights.values()])
	print("Done Step 2")
	#==============================================================================
	#Step3: Randomly Shuffle Data
	print("=="*30)
	print("Step 3: Randomly Shuffle Data")
	print("=="*30)
	#==============================================================================
	# Randomly shuffle data
	from sklearn.model_selection import train_test_split
	x_train, x_dev, y_train, y_dev = train_test_split(x, y, shuffle=True,test_size=0.12,random_state=16)
	# shuffle_indices = np.random.permutation(np.arange(len(x_to_train)))
	# #shuffle xtrain and ytrain
	# x_train = x_to_train[shuffle_indices]
	# y_to_train = y_to_train[shuffle_indices]
	# train_len = int(len(x_to_train) * 0.85)
	# #split x_train into two set train and dev
	# x_train = x_to_train[:train_len]
	# y_train = y_to_train[:train_len]
	# x_dev = x_to_train[train_len:]
	# y_dev = y_to_train[train_len:]
	print('Shape of data tensor:', x_train.shape)
	print('Shape of label tensor:', y_train.shape)
	print("	We have %d data for the training set, %d for evaluating set"%(len(x_train),len(x_dev)))
	print("Done Step 3")
	#==============================================================================
	#Step4: Training
	print("=="*30)
	print("Step 4: Training")
	print("=="*30)
	#==============================================================================
	with tf.Graph().as_default():
	    session_conf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
	    sess = tf.Session(config=session_conf)
	    with sess.as_default():
	        cnn = TextCNN(
	            embedding_vector=embedding,
	            sequence_length=x_train.shape[1],
	            num_classes=y.shape[1],
	            vocab_size=len(vocabulary),
	            embedding_size=embedding_size,
	            filter_sizes=list(map(int, filter_sizes.split(","))),
	            num_filters=num_filters,
	            isMultilabel=isMultilabel,
	            l2_reg_lambda=l2_reg_lambda
	            )

	        # Define Training procedure
	        global_step = tf.Variable(0, name="global_step", trainable=False)
	        optimizer = tf.train.AdamOptimizer(1e-3)
	        grads_and_vars = optimizer.compute_gradients(cnn.loss)
	        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

	        # Keep track of gradient values and sparsity (optional)
	        grad_summaries = []
	        for g, v in grads_and_vars:
	            if g is not None:
	                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
	                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
	                grad_summaries.append(grad_hist_summary)
	                grad_summaries.append(sparsity_summary)
	        grad_summaries_merged = tf.summary.merge(grad_summaries)

	        # Output directory for models and summaries
	        # timestamp = str(int(time.time()))
	        name = sys.argv[1]
	        out_dir = os.path.abspath(os.path.join(os.path.curdir, "cnn_runs", name))
	        print("Writing to {}\n".format(out_dir))

	        # Summaries for loss and accuracy
	        loss_summary = tf.summary.scalar("loss", cnn.loss)
	        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

	        # Train Summaries
	        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
	        train_summary_dir = os.path.join(out_dir, "summaries", "train")
	        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

	        # Dev summaries
	        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
	        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
	        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

	        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
	        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
	        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
	        if not os.path.exists(checkpoint_dir):
	            os.makedirs(checkpoint_dir)
	        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

	        # Write vocabulary
	        vocab_processor.save(os.path.join(out_dir, "vocab"))

	        # Initialize all variables
	        sess.run(tf.global_variables_initializer())

	        def train_step(x_batch, y_batch):
	        	feed_dict = {
	        		cnn.input_x: x_batch,
	        		cnn.input_y: y_batch,
	        		cnn.dropout_keep_prob: dropout_keep_prob
	        		}
	        	_, step, summaries, loss, accuracy = sess.run(
	        		[train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
	        		feed_dict)
	        	time_str = datetime.datetime.now().isoformat()
	        	if step % showstep_every == 0:
	        		print("TRAIN step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
	        	train_summary_writer.add_summary(summaries, step)

	        def dev_step(x_batch, y_batch, writer=None):
	        	feed_dict = {
	        		cnn.input_x: x_batch,
	        		cnn.input_y: y_batch,
	        		cnn.dropout_keep_prob: 1.0
	        		}
	        	step, summaries, loss, accuracy = sess.run(
	        		[global_step, dev_summary_op, cnn.loss, cnn.accuracy],feed_dict)
	        	time_str = datetime.datetime.now().isoformat()
	        	print("     EVAL step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
	        	if writer:
	        		writer.add_summary(summaries, step)
	        # Generate batches
	        batches = data_helpers.batch_iter(list(zip(x_train, y_train)), batch_size, num_epochs)
	        total_steps = data_helpers.num_step(list(zip(x_train, y_train)),batch_size,num_epochs)
	        evaluate_every = total_steps/num_epochs
	        checkpoint_every = evaluate_every
	        #Training loop. For each batch...
	        start = time.time()
	        for batch in batches:
	        	x_batch, y_batch = zip(*batch)
	        	train_step(x_batch, y_batch)
	        	current_step = tf.train.global_step(sess, global_step)
	        	if current_step % evaluate_every == 0:
	        		print("  Evaluation:")
	        		dev_step(x_dev, y_dev, writer=dev_summary_writer)
	        		print("")
	        	if current_step % checkpoint_every == 0:
	        		path = saver.save(sess, checkpoint_prefix, global_step=current_step)
	        		print("Saved model checkpoint to {}\n".format(path))
	        	if current_step == total_steps:
	        		print("Last step:")
	        		print("  Evaluation:")
	        		dev_step(x_dev, y_dev, writer=dev_summary_writer)
	        		print("")
	        		path = saver.save(sess, checkpoint_prefix, global_step=current_step)
	        		print("Saved last model checkpoint to {}\n".format(path))
	end = time.time()
	elapsed = end - start
	print("Done in:",time.strftime("%H:%M:%S", time.gmtime(elapsed)))

if __name__ == '__main__':
	training()