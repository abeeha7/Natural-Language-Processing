import re
#function to calculate the pair frequency
def get_pair_freqs(words,vocab):
  pairs= dict()
  for i in range(len(words) - 1):
    if words[i] != '\n' and words[i+1]!='\n':
      pair = (words[i], words[i + 1])
      if pair not in pairs:
          pairs[pair] = 1
      else:
          pairs[pair] += 1
  return pairs

#function to calcualte score
def get_score(pair_freq,vocab,words):
  scores=dict()
  for i in range(len(words)-1):
    if words[i] != '\n' and words[i+1]!='\n':
      pair = (words[i], words[i + 1])
      # print("letter a:"+ words[i] ,":",vocab[words[i]])
      # print("letter b:"+ words[i+1] ,":",vocab[words[i+1]])
      sc = pair_freq[pair] / (vocab[words[i]] * vocab[words[i+1]])
      scores[pair] = sc
  return scores


def word_piece(content, merges):
  words = content.split()
  #Creating vocabulary
  vocab= dict()
  data=[]
  for word in words:
    for j, letter in enumerate(word):
        if j == 0:
            # First letter of the word (no ## prefix)
            if letter not in vocab:
                vocab[letter] = 1
            else:
                vocab[letter] += 1
            data.append(letter)  # Append first letter without ##
        else:
            # Subsequent letters (with ## prefix)
            string = "##" + letter
            if string not in vocab:
                vocab[string] = 1
            else:
                vocab[string] += 1
            data.append(string)
    data.append('\n')
  mergelst=[]
  for i in range(merges):
    pair_freq= get_pair_freqs(data,vocab)
    # print("Pair frequency:",pair_freq)
    score = get_score(pair_freq,vocab,data)
    max_score = max(score.items(), key=lambda x: x[1])
    # print(max_score)
    cleaned_key1 = max_score[0][1].replace('##', '')
    new_word = max_score[0][0]+cleaned_key1
    vocab[new_word] = pair_freq[max_score[0]]
    i = 0
    while i < len(data) - 1:
        # Check if consecutive elements match max_score pair
        if data[i] == max_score[0][0] and data[i+1] == max_score[0][1]:
            # Replace data[i] with the new word
            data[i] = new_word
            # Remove data[i+1] (the second part of the pair)
            data.pop(i+1)
        else:
            # Move to the next element
            i += 1
    mergelst.append(new_word)
  #   print(data)
  # print(score)
    # print(vocab)
  return mergelst



# Reading the file
file_path = 'Question2/wordpiece_input.txt'

# Open and read the file
with open(file_path, 'r') as file:
    content = file.read()
# content = "huggingface hugging face hug hugger learning learner learners learn"
content = content.lower()
merges = word_piece(content,80)
print("The merged pairs are: \n",merges)






