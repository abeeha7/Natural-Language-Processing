import string
#This function iterates through the file, reads the text and processes it and returns tokens
def process_data(file_path):
    # Open and read the file
    with open(file_path, 'r') as file:
        content = file.read()
    content = content.lower()
    translator = str.maketrans('', '', string.punctuation)
    cleaned_text = content.translate(translator)
    words = cleaned_text.split()
    return words
    

# To generate vocabulary 
def vocabulary(words):
    vocab =[]
    for i in words:
        if i not in vocab:
            vocab.append(i)
    return vocab


#n-gram implementation, calculates the probability of each n-gram by applying smoothing
def n_gram_model(words,vocab,n):
    n_gram={}
    for i in range (len(words)-n+1):
        pair = words[i:i+n]
        text=' '.join(pair)
        if text not in n_gram:
            n_gram[text] = 1
        else:
            n_gram[text]+=1
    n_1gram={}
    for i in range (len(words)-(n-1)+1):
        pair = words[i:i+(n-1)]
        text=' '.join(pair)
        if text not in n_1gram:
            n_1gram[text] = 1
        else:
            n_1gram[text]+=1

    prob={}
    for i in range (len(words)-n+1):
        pair = words[i:i+n]
        text=' '.join(pair)
        history= ' '.join(pair[:-1])
        probability = (n_gram[text] + 1) / (n_1gram[history] + len(vocab) ) 
        prob[text] = probability
    return prob,n_1gram


#calculates the perplexity of the test set by using the probability of the training set
def perplexity(test,vocab,prob,n_1gram,n):
    probs={}
    N=0
    for i in range (len(test)-n+1):
        pair = test[i:i+n]
        n_1 = pair[:-1]
        his = ' '.join(n_1)
        text=' '.join(pair)
        if text in prob:
            val = prob[text]
            if text not in probs:
                probs[text] = val
            else:
                probs[text] = probs[text] * val
        else:
            if his in n_1gram:
                val = 1 / (n_1gram[his]+(len(vocab)))
                if text not in probs:
                    probs[text] = val
                else:
                    probs[text] = probs[text] * val
            else:
                val = 1/(len(vocab))
                if text not in probs:
                    probs[text] = val
                else:
                    probs[text] = probs[text] * val
        N+=1
    val=1
    for i in probs.values():
        val = val * i
    perp = (val)**(-(1/N))
    return perp




    
    

#-------------- Training of the data -----------#
file_path_train = 'Question#1/train.txt'
words_train = process_data(file_path_train)
vocab = vocabulary(words_train)

#-------------Testing of the data ----------------#
file_path_test = 'Question#1/test.txt'
words_test = process_data(file_path_test)
prob7,n_gram7=n_gram_model(words_train,vocab,7)
perp7 = perplexity(words_test,vocab,prob7,n_gram7,7)
prob6,n_gram6=n_gram_model(words_train,vocab,6)
perp6 = perplexity(words_test,vocab,prob6,n_gram6,6)
prob5,n_gram5=n_gram_model(words_train,vocab,5)
perp5 = perplexity(words_test,vocab,prob5,n_gram5,5)
prob4,n_gram4=n_gram_model(words_train,vocab,4)
perp4 = perplexity(words_test,vocab,prob4,n_gram4,4)
prob3,n_gram3=n_gram_model(words_train,vocab,3)
perp3 = perplexity(words_test,vocab,prob3,n_gram3,3)
prob2,n_gram2=n_gram_model(words_train,vocab,2)
perp2 = perplexity(words_test,vocab,prob2,n_gram2,2)
prob1,n_gram1=n_gram_model(words_train,vocab,1)
perp1 = perplexity(words_test,vocab,prob1,n_gram1,1)
print("Unigram perplexity: ", perp1)
print("Bigram perplexity: ", perp2)
print("Trigram perplexity: ",perp3)
print("Quadgram perplexity: ",perp4)
print("Pentagram perplexity: ",perp5)
print("Hexagram perplexity: ",perp6)
print("Heptagram perplexity: ",perp7)