import re


# Reading the file
file_path = 'Question1/date_format_dd_mm_yyyy.txt'

# Open and read the file
with open(file_path, 'r') as file:
    content = file.read()

# The \s* handles any spaces after the period
sentences = re.split(r'\.\s*', content)  
words=[i.split() for i in sentences]
#regular expression for dates
date_format = r'\d{1,2}\/\d{1,2}\/\d{1,4}'
#us words
MM_DD_YY_words= [
        "fall", "thanksgiving", "memorial day", "labor day", "fourth of july", "veterans day",
        "super bowl", "nba finals", "world series", "washington dc", "new york", "california",
        "chicago", "state", "governor", "color", "center", "honor", "semester",
        "elementary school", "high school", "college", "fiscal year", "q1", "q2", "q3", "q4"
    ] 
# european words
DD_MM_YY_words=[
        "autumn", "christmas", "boxing day", "bank holiday", "easter", "good friday",
        "new year's day", "bonfire night", "summer holidays", "half term", "london",
        "paris", "berlin", "european union", "british", "england", "united kingdom",
        "colour", "centre", "honour", "favourite", "theatre", "university", "headteacher",
        "primary school", "secondary school", "european parliament", "fifa world cup",
        "uefa", "euros", "football", "q1", "q2", "q3", "q4"
    ] 
#function to identify the words of the sentences as US or European
def context(sentence):
  for i in sentence:
    if i in DD_MM_YY_words:
      return 'DD/MM/YY'
    elif i in MM_DD_YY_words:
      return 'MM/DD/YY'
    
date_lst=[]
for i in sentences:
  dates= re.findall(date_format,i)
  context_ans = context(i)
  for j in dates:
    if context_ans == 'DD/MM/YY':
      date_lst.append((j,' The format is DD/MM/YY contextually'))
      continue
    elif context_ans == 'MM/DD/YY':
      date_lst.append((j,' The format is MM/DD/YY contextually'))
      continue
    date,month,year = j.split('/')
    if int(date) > 12:
      date_lst.append((j,' The format is DD/MM/YY logically'))
    elif int(date)< 12 and int(month)> 12:
      date_lst.append((j,' The format is MM/DD/YY logically'))
    else:
      date_lst.append((j,' The format is ambiguous '))
with open('Question1/AbeehaZehra_az07728.txt', 'w', encoding='utf-8') as file:
  for i in date_lst:
     file.write(i[0] +  ": "+  i[1] + '\n')


