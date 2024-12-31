import zipfile
import string
import math
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


# This function preprocess the training files by removing stopwords, punctuation, special characters and tokenization
def process_files(zip_ref,files):
    comments=[]
    for file in files:
        with zip_ref.open(file) as f:
            content = f.read()
            content = content.decode('utf-8')
            content = content.lower()
            translator = str.maketrans('', '', string.punctuation)
            cleaned_text = content.translate(translator)
            words = cleaned_text.split()
            negative = [word for word in words if word not in stop_words]
            comments.extend(negative)

    return comments

# This function is to preprocess the test files by removing stopwords, punctuation, special characters and tokenization
def test_process_files(zip_ref,files):
    comments=[]
    for file in files:
        with zip_ref.open(file) as f:
            content = f.read()
            content = content.decode('utf-8')
            content = content.lower()
            translator = str.maketrans('', '', string.punctuation)
            cleaned_text = content.translate(translator)
            words = cleaned_text.split()
            negative = [word for word in words if word not in stop_words]
            comments.append(negative)
    return comments

# This function helps to find the frequency of each word given the class
def word_count(data):
    vocab={}
    for i in data:
        if i  not in vocab:
            vocab[i] = 1
        else:
            vocab[i]+=1
    return vocab

# To generate the vocabulary of the text
def vocabulary(pos,neg):
    vocab=[]
    for i in pos:
        if i not in vocab:
            vocab.append(i)
    for j in neg:
        if j not in vocab:
            vocab.append(j)
    return vocab


#This function calculates the likelihood of each word for both the classes, and smoothing is also done
def naive_bayes(vocab,vocab_pos,vocab_neg):
    pos={}
    neg={}
    for i in vocab:
        if i in vocab_pos:
            val = (vocab_pos[i]+1)/(sum(vocab_pos.values())+len(vocab))
            pos[i] = val
        elif i not in vocab_pos:
            val = (1)/(sum(vocab_pos.values())+len(vocab))
            pos[i] = val
        if i in vocab_neg:
            val = (vocab_neg[i]+1)/(sum(vocab_neg.values())+len(vocab))
            neg[i]=val
        elif i not in vocab_neg:
            val = (1)/(sum(vocab_neg.values())+len(vocab))
            neg[i]=val
    return pos,neg

#This function is used to predict the class for the test data set by using the likelihood calculated by the training set
def test_review(sentence,pos,neg,vocab,prior):
    for i in sentence:
        if i not in vocab:
            sentence.remove(i)
    positive=1
    negative=1
    for i in sentence:
        if i in pos:
            positive = positive * pos[i]
        if i in neg:
            negative =  negative * neg[i]
    neg_posterioir = prior * negative
    pos_posterior = prior * positive
    if pos_posterior > neg_posterioir:
        return "positive"
    else:
        return "negative"

#Generates the confusion matrix and calculates accuracy, precision, recall, and f1 score
def evaluation (TP, FP,TN, FN):
    accuracy = (TP + TN)/(TP + TN + FN + FP)
    precision = (TP) / (TP + FP)
    recall = (TP) / (TP + FN)
    f1_score = (2 * precision * recall) / (precision + recall)
    print("TP: ", TP)
    print("TN:", TN)
    print("FP:", FP)
    print("FN: ",FN)
    return accuracy, precision, recall, f1_score

zip_path='Question#2/aclImdb_v1.zip'
maxfiles=500
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get a list of all files in the zip archive for processing and training
        file_list = zip_ref.namelist()
        neg_folder = f"aclImdb_v1/aclImdb/{'train'}/neg/"
        pos_folder = f"aclImdb_v1/aclImdb/{'train'}/pos/"
        neg_files = [f for f in file_list if f.startswith(neg_folder) and not f.endswith('/')]
        pos_files = [f for f in file_list if f.startswith(pos_folder) and not f.endswith('/')]
        neg_files = neg_files[:maxfiles]
        pos_files = pos_files[:maxfiles]
        pos=process_files(zip_ref,pos_files)
        neg = process_files(zip_ref,neg_files)

        # Get a list of all files in the zip archive for processing and testing
        test_pos_folder=f"aclImdb_v1/aclImdb/{'test'}/pos/"
        test_neg_folder=f"aclImdb_v1/aclImdb/{'test'}/neg/"
        test_pos_files=[f for f in file_list if f.startswith(test_pos_folder) and not f.endswith('/')]
        test_neg_files = [f for f in file_list if f.startswith(test_neg_folder) and not f.endswith('/')]
        test_neg_files=test_neg_files[:100]
        test_pos_files=test_pos_files[:100]
        test_pos=test_process_files(zip_ref,test_pos_files)
        test_neg=test_process_files(zip_ref,test_neg_files)



# count of each word for respective class
vocab_pos= word_count(pos)
vocab_neg = word_count(neg)

#creating the overall vocabulary
vocab = vocabulary(pos,neg)

#prior probability of the training set classes
prior = 500/1000
#Training of data
likeli_pos,likeli_neg=naive_bayes(vocab,vocab_pos,vocab_neg)

#variables for confusion matrix
TP = 0
TN = 0
FP = 0
FN = 0

#Testing on test data set
test_positive=[]
test_negative=[]
for i in test_pos:
    ans = test_review(i,likeli_pos,likeli_neg,vocab,prior)
    if ans == "positive":
        TP +=1
    elif ans =="negative":
        FN+=1
    test_positive.append(ans)

for j in test_neg:
    ans = test_review(j,likeli_pos,likeli_neg,vocab,prior)
    if ans == "negative":
        TN+=1
    elif ans == "positive":
        FP+=1
    test_negative.append(ans)

#evaluation of the model
accuracy, precision, recall, f1_score = evaluation(TP,FP,TN,FN)
print("Accuracy: ", float(accuracy * 100) , "%")
print("Precsion: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1_score)


    



