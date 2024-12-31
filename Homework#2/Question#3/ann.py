import csv
import string
import numpy as np


# This function cleans text by removing punctuation, special characters and creating tokens by splitting the text
def process_text(content):
    sentence_list=[]
    content = content.lower()
    translator = str.maketrans('', '', string.punctuation)
    cleaned_text = content.translate(translator)
    cleaned_text = ''.join([char for char in cleaned_text if char.isalnum() or char.isspace()])  # Remove special characters
    words= cleaned_text.lower().split()  # Convert to lowercase and tokenize (split into words)
    sentence_list.append(words)
    return sentence_list, words

#This function process the file and extracts the sentences, the words and the ground truth for each sentence
def process_file(file):
    sentences=[]
    labels=[]
    words=[]
    with open(file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for row in csv_reader:
            sentence,tokens = process_text(row[0])
            sentences.append(sentence)
            labels.append(float(row[-1]))
            for i in tokens:
                words.append(i)


    return sentences,words,labels

#creates vocabulary of the over all text
def vocabulary(text):
    vocab={}
    count=0
    for i in text:
        if i not in vocab:
            vocab[i] = count
            count+=1
    return vocab

#creates one hot encoded vectors for each sentence
def one_hot(word,vocab):
    one_hot_vector = [0] * len(vocab)
    for i in word:
        for j in i:
            index=vocab[j]
            one_hot_vector[index] = 1
    return one_hot_vector

#activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#derivative of activation function used in back propogation
def sigmoid_derivative(x):
    return x * (1 - x)

# to initialize the weights and biases based on the size of input, hidden and output vectors
def intialize(input_size,output_size,hidden_size):
    W1 = np.random.randn(input_size, hidden_size)  # 300 x 128  -> (5000 x 300) * (300 x 128) -> (300 x 128)
    b1 = np.zeros((1, hidden_size)) 
    W2 = np.random.randn(hidden_size, output_size)  # 128 x 1  -> (300 x 128) * (128 x 1) -> (300 x 1)
    b2 = np.zeros((1, output_size))
    return W1,b1,W2,b2


#used to calculate error and update weights and biases
def back_propogation(one_hot,a1,a2,labels,weights_hidden,weights_output,bias_hidden,bias_output,learning_rate):
    # error of output layer
    output_error = (labels - a2) * sigmoid_derivative(a2)

    # Error at hidden layer
    hidden_error = np.dot(output_error, weights_output.T) * sigmoid_derivative(a1) 

    # Update output layer weights and bias
    weights_output += learning_rate * np.dot(a1.T, output_error)
    bias_output += learning_rate * np.sum(output_error, axis=0, keepdims=True)

    # Update hidden layer weights and bias
    weights_hidden += learning_rate * np.dot(one_hot.T, hidden_error)
    bias_hidden += learning_rate * np.sum(hidden_error, axis=0, keepdims=True)

    return weights_output,bias_output,weights_hidden,bias_hidden

#does forward propogation by calulating weighted sum and using activation function
def forward_propogation(one_hot,W1,W2,b1,b2):

    z1 = np.dot(one_hot, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    return a1,a2

#trains the data by intializing weights and biases, and then repeating forward propogation and backward propogation for 10 epochs
def training(one_hot,input_size,output_size,hidden_size,labels,epoch,learning_rate):
    W1,b1,W2,b2 = intialize(input_size,output_size,hidden_size)
    one_hot = np.array(one_hot)
    for i in range (epoch):
        a1,a2 = forward_propogation(one_hot,W1,W2,b1,b2)
        W2,b2,W1,b1 = back_propogation(one_hot,a1,a2,labels,W1,W2,b1,b2,learning_rate)

    return W1,b1,W2,b2,a2

#To test the test data set and predict the output by using trained model
def predict(one_hot,W1,b1,W2,b2):
    one_hot = np.array(one_hot)
    a1,a2 = forward_propogation(one_hot,W1,W2,b1,b2)
    predicted=[]
    for i in a2:
        if i >= 0.5:
            predicted.append(1)
        else:
            predicted.append(0)
    return predicted
    
#evaluates the perfomance of the model
def evaluation(output,predict):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range (len(predict)):
        if predict[i] == 1:
            if output[i] == 1:
                tp+=1
            elif output[i] == 0:
                fn+=1
        elif predict[i] == 0:
            if output[i] == 0:
                tn+=1
            elif output[i] == 1:
                fp +=1

    accuracy = (tp + tn)/(tp + tn + fn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    print("TP: ", tp)
    print("TN:", tn)
    print("FP:", fp)
    print("FN: ",fn)
    return accuracy, precision, recall, f1_score
    



#------------------------- Training part --------------------------------#
# reads the file and processes it
file_path = 'Question#3/sentiment_train_dataset.csv'
sentences,tokens,labels = process_file(file_path)
labels = np.array(labels).reshape(-1, 1)
#creates the vocabulary
vocab= vocabulary(tokens)

#geneartes one hot enocded vector for all the sentences in training data
one_hot_vectors=[]
for i in sentences:
    vector=one_hot(i,vocab)
    one_hot_vectors.append(vector)

#intializing values of input, hidden layers, output layer, learning rate
input_size= len(vocab)
hidden_size=128
output_size=1
epoch=10
learning_rate = 0.01

#training the model
W1,b1,W2,b2,predicted_output=training(one_hot_vectors,input_size,output_size,hidden_size,labels,epoch,learning_rate)


#------------------------------------- Testing Part ----------------------------------------#
#processing the test data and processing it
file_path_test = 'Question#3/sentiment_test_dataset.csv'
test_sentences,test_tokens,test_labels = process_file(file_path_test)

#generating vocabulary for the test data
test_vocab = vocabulary(test_tokens)

#generating one hot encoded vectors for test data
test_one_hot = []
for i in test_sentences:
    vector1=one_hot(i,test_vocab)
    test_one_hot.append(vector1)

#uses the weights and biases from the training model to test 
output = predict(test_one_hot,W1,b1,W2,b2)

#evaluation of the model
accuracy, precision, recall, f1_score = evaluation(output,test_labels)
print("Accuracy is: ", accuracy * 100, "%" )
print("Precsion: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1_score)







