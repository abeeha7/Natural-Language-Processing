import nltk
from nltk.tokenize import regexp_tokenize

reg_ex = r'\w+|[^\w\s]'

#Method # 1
file_path = 'Question3/urdu_text_input.txt'

# Open and read the file with UTF-8 encoding
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()
tokens = content.split()

#Writing in the file
with open('Question3/output1.txt', 'w', encoding='utf-8') as file:
    for token in tokens:
        file.write(token + '\n')


#Mehtod # 2
token1 = regexp_tokenize(content, reg_ex)

#writing in the file
with open('Question3/output2.txt', 'w', encoding='utf-8') as file:
    for token in token1:
        file.write(token + '\n')

