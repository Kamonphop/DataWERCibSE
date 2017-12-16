import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn import metrics
import itertools
import matplotlib.pyplot as plt
import numpy as np
import data_helpers
import sys
from tflearn.data_utils import pad_sequences
import mlc_metrics as metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


#hyperparameters
#should be the same as in the training parameters
#path_to_glove = "glove.6B.100d.txt" #path to glove
embedding_size = 300 #dimension of glove
filter_sizes = "3,4,5" #Size of filter
num_filters = 128 #"Number of filters per filter size
l2_reg_lambda = 0.0 #L2 regularizaion lambda
batch_size = 64 #Batch Size
num_epochs = 6 #Number of training epochs
evaluate_every = 50 #"Evaluate model on dev set after this many steps
checkpoint_every = 50 #Save model after this many steps
num_checkpoints = 5
dropout_keep_prob = 0.5
max_document_length = 180
isMultilabel = False
isSoftwareOnly = True

def predict_new_input_dataset():
	#==============================================================================
	#Step1: Loading trained model
	# print("Step 1: Loading Checkpoints or trained model")
	#==============================================================================
	path_to_checkpoint_dir = sys.argv[1]
	checkpoint = tf.train.latest_checkpoint(path_to_checkpoint_dir)
	print("Successfully loaded the trained model")
	# print("Done Step 1")
	#==============================================================================
	#Step2: Loading data
	# print("Step 2: Loading a list of category saved by training procedure: train.py")
	#==============================================================================
	# print("Done Step 2")
	#==============================================================================
	#Step2: Loading vocabulary
	print("=="*30)
	print("Step 1: Loading a vocabulary")
	print("=="*30)
	#==============================================================================
	vocab_path = os.path.join(path_to_checkpoint_dir, "..", "vocab")
	vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
	# print("Done Step 3")
	#==============================================================================
	#Step3: Loading test data
	# print("Step 4: Loading datasets for testing")
	#==============================================================================
	# test_datasets= data_helpers.get_datasets_20newsgroup(subset="test",
 #                                                     categories=list_of_category,
 #                                                     shuffle=True,
 #                                                     random_state=42)
	# x_raw, y_test = data_helpers.load_data_and_labels(test_datasets)
	# x_raw = ['Articles covering the relation between religion and geography.    Wikimedia Commons has media related to Religion.   See also: Religions by country      Subcategories This category has the following 7 subcategories, out of 7 total. B ►  Biblical geography‎ (2 C, 4 P)  C ►  Religion by country‎ (218 C, 3 P)  D ►  Religious demographics‎ (2 C, 34 P)  I ►  Islamic concepts of religious geography‎ (3 P)  M ►  Religion maps‎ (empty)  P ►  Religious places‎ (26 C, 42 P)  R ►  Religious nationalism‎ (11 C, 21 P)    Pages in category "Geography of religion" The following 6 pages are in this category, out of 6 total. This list may not reflect recent changes (learn more).   Religion and geography0–9 10/40 WindowC Christianity in the Ottoman EmpireD Divisions of the world in IslamH Holy LandJ Joshua Project']
 #    #===== This has to be here =======
	# x_test = np.array(list(vocab_processor.transform(x_raw)))
	# x_test = pad_sequences(x_test,maxlen=MAX_SEQUENCE_LENGTH)
	# #===== This depends on the label data
	# # y_test = np.argmax(y_test, axis=1)
	# y_test = None
	# y_test = [5,1]
	print("=="*30)
	print("Step 2: Loading Data")
	print("=="*30)
	fold = sys.argv[2]
	if(isSoftwareOnly):
		x_test_raw, y_test = data_helpers.load_testing_data_and_labels_software(fold)
	else:
		x_test_raw, y_test = data_helpers.load_testing_data_and_labels(fold)
	print("X_test_raw:",len(x_test_raw))
	#print(all_text[0:10])
	#print datasets description
	#==============================================================================
	#Step2: Building Vocabular and Apply pre-trained (i.e., create an embedding lookup table)
	print("=="*30)
	print("Step 2: Building Vocabulary Word2idx, Idx2vector, vectorEmbedding")
	print("=="*30)
	#==============================================================================
	# print("	Tokenizing words.....")
	# max_document_length = max([len(data_helpers.tokenize(x)) for x in all_text])#can be a sentence if a file is a sentence
	# print("		Max document length:",max_document_length)
	# print("	Building Vocabulary")
	# vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
	x = np.array(list(vocab_processor.transform(x_test_raw)))
	# vocabulary = vocab_processor.vocabulary_

	# vocab_dict = vocab_processor.vocabulary_._mapping
	# vocab_inv = {v: k for k, v in vocab_dict.items()}

	#print(vocab_inv)
	x_test = pad_sequences(x,maxlen=max_document_length)
	print("X:",len(x))
	#==============================================================================
	#Step3: Randomly Shuffle Data
	print("=="*30)
	print("Step 3: Train Test spliting")
	print("=="*30)
	#==============================================================================
	# Randomly shuffle data
	# from sklearn.model_selection import train_test_split
	# x_to_train, x_test, y_to_train, y_test = train_test_split(x, y, shuffle=True,test_size=0.1,random_state=42)
	#==============================================================================
	#==============================================================================
	#Step4: Predicting
	print("=="*30)
	print("Step 4: Evaluating the testing data")
	print("=="*30)
	#==============================================================================
	graph = tf.Graph()
	with graph.as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint))
			saver.restore(sess, checkpoint)
			# Get the placeholders from the graph by name
			input_x = graph.get_operation_by_name("input_x").outputs[0]
			dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
			# Tensors we want to evaluate
			scores = graph.get_operation_by_name("output/scores").outputs[0]
			# Tensors we want to evaluate
			predictions = graph.get_operation_by_name("output/predictions").outputs[0]
			# Generate batches for one epoch
			batches = data_helpers.batch_iter(list(x_test), batch_size, 1, shuffle=False)
			all_predictions = []
			# all_probabilities = None
			# all_predictions = np.array([[0,0,0,0]])
			for x_test_batch in batches:
				# batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
				# all_predictions = np.concatenate((all_predictions, batch_predictions),axis =0)
				batch_predictions = sess.run([predictions,scores], {input_x: x_test_batch, dropout_keep_prob: 1.0})
				# if(isMultilabel):
				all_predictions.extend(batch_predictions[0])
				# else:
				# 	all_predictions = np.concatenate([all_predictions, batch_predictions[0]])
				#print(batch_predictions[0])
				#all_predictions = np.concatenate([all_predictions, batch_predictions[0]])
	
	if(isMultilabel):
		print("=="*30)
		print("Step 5: Finding Best Thredshold from the predicted data")
		print("=="*30)
		print("Total number of predictions: ",len(all_predictions))
		out = np.array(all_predictions)


		#Find the best threashold
		#https://github.com/suraj-deshmukh/Multi-Label-Image-Classification
		from sklearn.metrics import matthews_corrcoef
		threshold = np.arange(0.1,0.9,0.1)

		acc = []
		accuracies = []
		best_threshold = np.zeros(out.shape[1])
		for i in range(out.shape[1]):
		    y_prob = np.array(out[:,i])
		    for j in threshold:
		        y_pred = [1 if prob>=j else 0 for prob in y_prob]
		        acc.append( matthews_corrcoef(y_test[:,i],y_pred))
		    acc   = np.array(acc)
		    index = np.where(acc==acc.max()) 
		    accuracies.append(acc.max()) 
		    best_threshold[i] = threshold[index[0][0]]
		    acc = []
		print("Best Threshold: ",best_threshold)
		print("=="*30)
		print("Step 6: Different Accuracy Measures")
		print("=="*30)
		y_pred = np.array([[1 if out[i,j]>=best_threshold[j] else 0 for j in range(y_test.shape[1])] for i in range(len(y_test))])
		metrics.printall(y_test,y_pred)
		print("=="*30)
		print("Step 6: Default Threshold with 0.5")
		print("=="*30)
		default_threshold=[0.5,0.5,0.5,0.5]
		y_pred_default = np.array([[1 if out[i,j]>=default_threshold[j] else 0 for j in range(y_test.shape[1])] for i in range(len(y_test))])
		metrics.printall(y_test,y_pred_default)
		metrics.writeall(y_test,y_pred_default,fold)
		print("=="*30)
		print("Step 7: Done")
		print("=="*30)
	else:
		y_test = np.argmax(y_test, axis=1)
		metrics.writemulticlass(y_test,all_predictions,fold)
		# print(y_test)
		# print(all_predictions)


if __name__ == '__main__':
	if(len(sys.argv) < 3):
		print("Please specify the checkpoint path: python3 predict_cnn.py <path to checkpoint dir> <foldName> <isMultiLabel 0 = false, 1 = true>")
		sys.exit(1)
	elif(len(sys.argv) == 3):
		predict_new_input_dataset()
	else:
		print("Invalid argument! The program takes only 1 argument: python3 predict_cnn.py <path to checkpoint dir>")
		sys.exit(1)