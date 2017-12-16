import numpy as np
import re
import gensim.models.word2vec as w2v
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer
import os
import csv

def tokenize(x):
    clean = re.sub("[^a-zA-z]"," ",x)
    words = clean.strip().split()
    #print(words)
    return words


def tokenize_2(iterator):
    for value in iterator:
        clean = re.sub("[^a-zA-z]"," ",value)
        words = clean.strip().split()
        yield words

def num_step(data,batch_size,num_epochs):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    return num_batches_per_epoch*num_epochs

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    print("Total of %d batch per epoch"%(num_batches_per_epoch))
    print("Total of %d steps"%(num_batches_per_epoch*num_epochs))
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def collapse_4andabove(stringlabels):
    arraylabels = []
    for i in stringlabels.strip().split(','):
        if len(i) != 0:
            label = int(i.strip())
            if label == 1 or label == 2 or label == 3:
                arraylabels.append(label)
            else:
                label = 4
                if label not in arraylabels:
                    arraylabels.append(label)
        else:
                print("Found data not annotated")
    return arraylabels


def load_training_data_and_labels(fold):
# Load a multi-label dataset
    X = []
    Y = []
    filename='../datafinal/10fold/'+fold+'.csv'
    with open(filename, newline='', encoding='utf-8') as csvfile:
        # next(csvfile, None) # skip first header line
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        n = 1
        for row in csvreader:
            X.append(row[2].lower())
            Y.append(collapse_4andabove(row[3]))
            n += 1
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(Y)
    return X,Y

def load_testing_data_and_labels(fold):
# Load a multi-label dataset
    X = []
    Y = []
    filename='../datafinal/10fold/'+fold+'.csv'
    with open(filename, newline='', encoding='utf-8') as csvfile:
        # next(csvfile, None) # skip first header line
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        n = 1
        for row in csvreader:
            X.append(row[2].lower())
            Y.append(collapse_4andabove(row[3]))
            n += 1
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(Y)
    return X,Y


def load_training_data_and_labels_software(fold):
# Load a multi-label dataset
    X = []
    Y = []
    filename='../datafinal/10fold/'+fold+'.csv'
    with open(filename, newline='', encoding='utf-8') as csvfile:
        # next(csvfile, None) # skip first header line
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        for row in csvreader:
            if( "2" in row[3] ):
                X.append(row[2].lower())
                if(row[5] == 'Inquiry'):
                    Y.append('Problem Discovery')
                else:
                    Y.append(row[5])
    le = LabelBinarizer()
    Y = le.fit_transform(Y)
    return X,Y

def load_testing_data_and_labels_software(fold):
# Load a multi-label dataset
    X = []
    Y = []
    filename='../datafinal/10fold/'+fold+'.csv'
    with open(filename, newline='', encoding='utf-8') as csvfile:
        # next(csvfile, None) # skip first header line
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        n = 1
        for row in csvreader:
            if( "2" in row[3] ):
                X.append(row[2].lower())
                if(row[5] == 'Inquiry'):
                    Y.append('Problem Discovery')
                else:
                    Y.append(row[5])
    le = LabelBinarizer()
    Y = le.fit_transform(Y)
    return X,Y



def load_amazon_w2v(vocabulary_dict):
    #inverse vocabulary
    print("Using pretrained w2v model")
    # print("Loading Google pre-trained Word2Vec")
    print("Loading AmazonW2VtrainedLowerNew")
    Amazon_w2v = w2v.Word2Vec.load(os.path.join("../word2vec/AmazonW2VtrainedLowerNew","AmazonW2VtrainedLowerNew.w2v"))
    # print("NotIncludeDataset")
    # Amazon_w2v = w2v.Word2Vec.load(os.path.join("../word2vec/NotIncludeDataset","NotIncludeDataset.w2v"))
    print("Analyzing words....")
    vocab_size = len(vocabulary_dict)
    word_embedding_list = [[]]*vocab_size
    word_embedding_list[0] = np.zeros(300)
    bound = np.sqrt(6.0) / np.sqrt(vocab_size) 
    count_exist = 0
    count_not_exist = 0
    for i in range(1,vocab_size):
        word = vocabulary_dict[i]
        embedding = None
        if word in Amazon_w2v.wv.vocab:
            embedding = Amazon_w2v[word]
        if embedding is not None:
            word_embedding_list[i] = embedding
            count_exist +=1
        else: #no embedding for this word
            word_embedding_list[i] = np.random.uniform(-bound,bound,300)
            #print(word)
            count_not_exist +=1
    word_embedding_final = np.array(word_embedding_list)
    # word_embedding = tf.constant(word_embedding_final,dtype=tf.float32)
    # t_assign_embedding = tf.assign(textCNN.Embedding,word_embedding)
    # sess.run(t_assign_embedding)
    print("Word exists embedding:", count_exist, "; words not exist embedding:", count_not_exist)
    return word_embedding_final