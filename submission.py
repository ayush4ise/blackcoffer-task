import pandas as pd
import nltk
import re

input_data = pd.read_csv('Input.csv')

#DATA EXTRACTION
def data_extract(row):
    import requests
    from bs4 import BeautifulSoup as bs
    page = requests.get(row['URL'])
    soup = bs(page.content)
    text_list = [j.text.replace(u'\xa0', u' ') for j in soup.find_all(['h1','p','ol'])]
    text_file = open(str(row['URL_ID'])+'.txt','w',encoding = 'utf-8')
    my_string = ' '.join(text_list)
    text_file.write(my_string)
    text_file.close()
    return row

#TOKENIZE COLLECTED TEXT
def create_tokens(row,c):
    with open(str(row['URL_ID'])+'.txt', encoding='utf-8') as file:
        all_text = file.read()
    word_tokens = nltk.word_tokenize(all_text)
    punctuations = ['?','!',',','.',';','&','“','”','’']
    word_tokens = [item.lower() for item in word_tokens if item not in punctuations]
    sent_tokens = nltk.sent_tokenize(all_text)
    if (c=='word'):
        return word_tokens
    elif (c=='sentence'):
        return sent_tokens

#SYLLABLE COUNT
def syllable_count(word):
    count = len(re.findall('[aeiou]',word.lower()))
    if re.search('ed|ed$',word.lower()):
        count-=1
    return count

#STOPWORDS LIST
name_list = ['Auditor', 'Currencies', 'DatesandNumbers', 'Generic', 'GenericLong', 'Geographic', 'Names']
big_stword_list = []

for i in name_list:
    with open('./StopWords/StopWords_'+i+'.txt') as file:
        st_text = file.read()
    if (i != 'GenericLong'):
        something = nltk.word_tokenize(st_text)
        big_stword_list += [item.lower() for item in something if item.isupper()]
    else:
        big_stword_list += nltk.word_tokenize(st_text)

#POSITIVE WORDS AND NEGATIVE WORDS DICTIONARY (LIST)
with open('./MasterDictionary/positive-words.txt') as file:
    pos_dict = file.read()
pos_dict = [item for item in nltk.word_tokenize(pos_dict) if item not in big_stword_list]

with open('./MasterDictionary/negative-words.txt') as file:
    neg_dict = file.read()
neg_dict = [item for item in nltk.word_tokenize(neg_dict) if item not in big_stword_list]

def sentimental_analysis(row):
    #stopwords from StopWords Lists removed from tokenized words
    word_tokens = create_tokens(row,'word')
    if len(word_tokens) == 0:
        return row
    more_cleaned_words = [item for item in word_tokens if item not in big_stword_list]
    #scores
    pos_score = 0
    neg_score = 0
    for i in more_cleaned_words:
        if i in pos_dict:
            pos_score+=1
        elif i in neg_dict:
            neg_score-=1
    neg_score*=-1
    row['POSITIVE SCORE'] = pos_score
    row['NEGATIVE SCORE'] = neg_score

    polarity_score = (pos_score - neg_score)/((pos_score + neg_score)+0.000001)
    row['POLARITY SCORE'] = polarity_score

    subjectivity_score = (pos_score + neg_score)/(len(more_cleaned_words)+0.000001)
    row['SUBJECTIVITY SCORE'] = subjectivity_score
    return row

def readability_analysis(row):
    word_tokens = create_tokens(row,'word')
    if len(word_tokens) == 0:
        return row
    sent_tokens = create_tokens(row,'sentence')
    avg_sent_length = len(word_tokens)/len(sent_tokens)
    row['AVG SENTENCE LENGTH'] = avg_sent_length

    complex_count = 0
    for i in word_tokens:
        if syllable_count(i) > 2:
            complex_count+=1
    percent_complex = complex_count/len(word_tokens)
    row['PERCENTAGE OF COMPLEX WORDS'] = percent_complex

    fog_index = 0.4*(avg_sent_length + percent_complex)
    row['FOG INDEX'] = fog_index

    #avg number of words per sentence is same as avg sentence length
    row['AVG NUMBER OF WORDS PER SENTENCE'] = avg_sent_length

    row['COMPLEX WORD COUNT'] = complex_count

    #words cleaned with nltk stopwords package
    from nltk.corpus import stopwords
    stops = set(stopwords.words('english'))
    cleaned_words = [item for item in word_tokens if item not in stops]
    row['WORD COUNT'] = len(cleaned_words)

    #avg syllable count since each row has multiple words
    total_syllable = 0
    for i in word_tokens:
        total_syllable += syllable_count(i)
    row['SYLLABLE PER WORD'] = total_syllable/len(word_tokens)

    #Personal pronouns count
    pronouns = ['I','we','my','ours','us','We','My','Ours','Us','WE','MY','OURS']
    pro_count = len([i for i in word_tokens if i in pronouns])
    row['PERSONAL PRONOUNS'] = pro_count

    #avg word length
    len_list = sum([len(i) for i in cleaned_words])
    row['AVG WORD LENGTH'] = len_list/len(cleaned_words)
    return row

inpute_data = input_data.apply(data_extract, axis = 1)
input_data = input_data.apply(sentimental_analysis, axis = 1)
input_data = input_data.apply(readability_analysis, axis = 1)
result = input_data[['URL_ID','URL','POSITIVE SCORE','NEGATIVE SCORE','POLARITY SCORE','SUBJECTIVITY SCORE','AVG SENTENCE LENGTH','PERCENTAGE OF COMPLEX WORDS','FOG INDEX','AVG NUMBER OF WORDS PER SENTENCE','COMPLEX WORD COUNT','WORD COUNT','SYLLABLE PER WORD','PERSONAL PRONOUNS','AVG WORD LENGTH']]
result.to_excel('OUTPUT.xlsx')