import numpy as np
import scipy as sp
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

''' What you can use:
1). mlc_classification_report(y_true,y_pred)
2). mlc_hamming_loss(y_true,y_pred)
3). mlc_accuracy_score(y_true,y_pred)
4). mlc_jaccard_similarity_score(y_true,y_pred)
5). mlc_precision(y_true,y_pred)
6). mlc_recall(y_true,y_pred)
7). mlc_f1score(y_true,y_pred)
8). printall(y_true,y_pred)
'''

def mlc_classification_report(y_true,y_pred):
	return classification_report(y_true,y_pred)

'''
Hamming Loss (HL): Hamming Loss reports how many times on average, the relevance of an
example to a class label is incorrectly predicted [44]. Therefore, hamming loss takes into account the
prediction error (an incorrect label is predicted) and the missing error (a relevant label not predicted),
normalized over total number of classes and total number of examples. Lower is better
'''
def mlc_hamming_loss(y_true,y_pred):
	return hamming_loss(y_true,y_pred)

''' Accuracy Score
Accuracy (A): Accuracy for each instance is defined as the proportion of the predicted correct labels
to the total number (predicted and actual) of labels for that instance. Overall accuracy is the average
across all instances. Higher is better
'''
def calc_acc_ratio(test, pred):
    return float(sum(np.logical_and(test, pred))) / float(sum(np.logical_or(test, pred)))

def mlc_accuracy_score(Y_test, y_score):
    N = len(y_score)
    acc_ratio = [calc_acc_ratio(Y_test[i], y_score[i]) for i in range(N)]
    return sum(acc_ratio) / N

''' This should return the same number of Accuracy score. Higher is better'''
def mlc_jaccard_similarity_score(y_true,y_pred):
	return jaccard_similarity_score(y_true,y_pred)

'''
Precision (P): Precision is the proportion of predicted correct labels to the total number of actual
labels, averaged over all instances.
P recision, P = 1/n* sigma | Yi intersect Zi | / | Zi |
Source: https://en.wikipedia.org/wiki/Multi-label_classification
'''
def mlc_precision(Y_test,y_score):
    N = len(y_score)
    precision = 0.0
    for i in range(N):
        if(sum(y_score[i] > 0)):
            precision += float(sum(np.logical_and(Y_test[i],y_score[i])))/float(sum(y_score[i]))
    return precision / N

'''
Recall (R): Recall is the proportion of predicted correct labels to the total number of predicted labels,
averaged over all instances.
Recall, R = 1/n* sigma |Yi intersect Zi|/ |Yi|
'''
def mlc_recall(y_true,y_score):
    N = len(y_score)
    recall = 0.0
    for i in range(N):
        if(sum(y_true[i] > 0)):
            recall += float(sum(np.logical_and(y_true[i],y_score[i])))/float(sum(y_true[i]))
    return recall / N

'''
F1-Measure (F): Definition for precision and recall naturally leads to the following definition for
F1-measure (harmonic mean of precision and recall).
'''
def mlc_f1score(y_true,y_score):
    precision = mlc_precision(y_true,y_score)
    recall = mlc_recall(y_true,y_score)
    harmonic_mean = sp.stats.mstats.hmean([precision,recall])
    return harmonic_mean

'''
Subset Accuracy
'''
def mlc_subset_accuracy(y_true,y_score):
    return accuracy_score(y_true,y_score)

def printall(y_true,y_score):
    # print("Classification Report:")
    # print(mlc_classification_report(y_true,y_score))
    print("Hamming loss (lower is better [0,1]): ",hamming_loss(y_true,y_score))
    print("Accuracy score (higher is better [0,1]): ",mlc_accuracy_score(y_true,y_score))
    print("Jaccard similarity score (higher is better [0,1]): ",mlc_jaccard_similarity_score(y_true,y_score))
    print("F1 score: ",mlc_f1score(y_true,y_score))
    print("Subset accuracy (higher is better [0,1]): ",mlc_subset_accuracy(y_true,y_score))
    print("Average Micro Precision: ",precision_score(y_true,y_score,average='micro'))
    print("Average Micro Recall: ",recall_score(y_true,y_score,average='micro'))

def writeall(y_true,y_score,filename): 
    fp = open(str(filename) + ".txt","w")
    fp.write("Classification Report:\n")   
    fp.write(mlc_classification_report(y_true,y_score))
    fp.write("\nHamming loss (lower is better [0,1]): " + str(hamming_loss(y_true,y_score)))
    fp.write("\nAccuracy score (higher is better [0,1]): " + str(mlc_accuracy_score(y_true,y_score)))
    fp.write("\nJaccard similarity score (higher is better [0,1]): " + str(mlc_jaccard_similarity_score(y_true,y_score)))
    fp.write("\nF1 score: " + str(mlc_f1score(y_true,y_score)))
    fp.write("\nSubset accuracy (higher is better [0,1]): " + str(mlc_subset_accuracy(y_true,y_score)))
    fp.write("\nAverage precision score (higher is better [0,1]): " + str(average_precision_score(y_true,y_score)))
    fp.write("\nRanking Loss: (lower is better [0,1]) " + str(label_ranking_loss(y_true,y_score)))
    fp.write("\nAverage Micro Precision: " +str(precision_score(y_true,y_score,average='micro')))
    fp.write("\nAverage Micro Recall: "+str(recall_score(y_true,y_score,average='micro')))
    fp.close()

def writemulticlass(y_true,y_score,filename):
    fp = open(str(filename) + ".txt","w")
    fp.write("Classification Report:\n")   
    fp.write(mlc_classification_report(y_true,y_score))
    fp.write("\nAverage Micro Precision: " +str(precision_score(y_true,y_score,average='micro')))
    fp.write("\nAverage Micro Recall: "+str(recall_score(y_true,y_score,average='micro')))
    fp.write("\nSubset accuracy (higher is better [0,1]): " + str(mlc_subset_accuracy(y_true,y_score)))
    fp.close()