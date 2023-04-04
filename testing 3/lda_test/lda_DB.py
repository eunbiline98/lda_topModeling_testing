import sqlalchemy
import pandas as pd
import numpy as np
import string
import re  # regex library
import os
import pickle
import swifter

import gensim
import pyLDAvis.gensim
import pyLDAvis
from gensim import corpora

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords


# Import dataframe into MySQL
database_username = 'root'
database_password = ''
database_ip = 'localhost'
database_name = 'lda_web'
database_connection = sqlalchemy.create_engine('mysql+mysqlconnector://{0}:{1}@{2}/{3}'.
                                               format(database_username, database_password,
                                                      database_ip, database_name))
# ------ Data Responden Read From Excel Files ---------
dataSB = pd.read_excel(
    '/Users/thaar/Documents/work/Project Ariq/LDA/testing 3/lda_test/content/dataDummy.xlsx', engine='openpyxl')  # lokasi file
data = pd.DataFrame(dataSB, columns=[
                    'Nama', 'Question1', 'Question2', 'Question3', 'Question4', 'Question5'])
tokenize = pd.DataFrame(dataSB, columns=[
    'Question1', 'Question2', 'Question3', 'Question4', 'Question5'])
# data = pd.DataFrame(dataSB)
print('Case Folding Result : \n')
print(data.head(10))

# ------ Update to DB ---------
data.to_sql(con=database_connection,
            name='data_responden', if_exists='replace')

# ------ Tokenizing ---------

nltk.download('punkt')


def remove_tweet_special(text):
    # remove tab, new line, ans back slice
    text = str(text).replace('\\t', " ").replace(
        '\\n', " ").replace('\\u', " ").replace('\\', "")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(
        re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
    # remove incomplete URL
    return text.replace("http://", " ").replace("https://", " ")


dataSB['Question1'] = dataSB['Question1'].apply(remove_tweet_special)
dataSB['Question2'] = dataSB['Question2'].apply(remove_tweet_special)
dataSB['Question3'] = dataSB['Question3'].apply(remove_tweet_special)
dataSB['Question4'] = dataSB['Question4'].apply(remove_tweet_special)
dataSB['Question5'] = dataSB['Question5'].apply(remove_tweet_special)

# NLTK word tokenize


def word_tokenize_wrapper(text):
    return word_tokenize(text)


dataSB['Question1_tokens'] = dataSB['Question1'].apply(word_tokenize_wrapper)

print('Tokenizing Result : \n')
print(dataSB['Question1_tokens'].head())

# NLTK calc frequency distribution


def freqDist_wrapper(text):
    return FreqDist(text)


dataSB['Question1_tokens_fdist'] = dataSB['Question1_tokens'].apply(
    freqDist_wrapper)

print('Frequency Tokens : \n')
print(dataSB['Question1_tokens_fdist'].head().apply(lambda x: x.most_common()))

nltk.download('stopwords')

# ----------------------- get stopword from NLTK stopword -------------------------------
# get stopword indonesia
list_stopwords = stopwords.words('indonesian')

# ---------------------------- manualy add stopword  ------------------------------------
# append additional stopword
list_stopwords.extend([])

# convert list to dictionary
list_stopwords = set(list_stopwords)

# remove stopword pada list token


def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]


dataSB['Question1_tokens_WSW'] = dataSB['Question1_tokens'].apply(
    stopwords_removal)

# print(dataSB['Question1_tokens_WSW'].head())

normalizad_word = pd.read_excel(
    '/Users/thaar/Documents/work/Project Ariq/LDA/testing 3/lda_test/content/dataBerita.xlsx', engine='openpyxl')  # lokasi file

normalizad_word_dict = {}

for index, row in normalizad_word.iterrows():
    if row[0] not in normalizad_word_dict:
        normalizad_word_dict[row[0]] = row[1]


def normalized_term(document):
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]


dataSB['Question1_normalized'] = dataSB['Question1_tokens_WSW'].apply(
    normalized_term)

dataSB['Question1_normalized'].head(25)

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stemmed


def stemmed_wrapper(term):
    return stemmer.stem(term)


term_dict = {}

for document in dataSB['Question1_normalized']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '

print(len(term_dict))

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)

    # untuk melihat hasilnya silahkan jalankan baris di bawah ini
    # print(term,":" ,term_dict[term])

    # apply stemmed term to dataframe


def get_stemmed_term(document):
    return [term_dict[term] for term in document]


dataSB['Question1_tokens_stemmed'] = dataSB['Question1_normalized'].swifter.apply(
    get_stemmed_term)

print(dataSB['Question1_tokens_stemmed'].value_counts())

for i in range(len(dataSB)):
    a = dataSB.iloc[i][0]
    document.append(a)

document[0:5]

doc_clean = dataSB['Question1_tokens_stemmed']
doc_clean

dictionary = corpora.Dictionary(doc_clean)
print(dictionary)

doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

total_topics = 3  # jumlah topik yang akan di extract
number_words = 10  # jumlah kata per topik

# Running and Trainign LDA model on the document term matrix.
lda_model = Lda(doc_term_matrix, num_topics=total_topics,
                id2word=dictionary, passes=50)

lda_model.show_topics(num_topics=total_topics, num_words=number_words)

# Word Count of Topic Keywords

topics = lda_model.show_topics(formatted=False)
data_flat = [w for w_list in doc_clean for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i, weight, counter[word]])

df_imp_wcount = pd.DataFrame(
    out, columns=['word', 'topic_id', 'importance', 'word_count'])
print(df_imp_wcount)

# Dominant topic and its percentage contribution in each topic


def format_topics_sentences(ldamodel=None, corpus=doc_term_matrix, texts=document):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series(
                    [int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic',
                              'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return (sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(
    ldamodel=lda_model, corpus=doc_term_matrix, texts=doc_clean)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = [
    'Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
print(df_dominant_topic.value_counts())

# Visualize the topics
pyLDAvis.enable_notebook()
LDAvis_data_filepath = os.path.join('ldavis_prepared_'+str(total_topics))
corpus = [dictionary.doc2bow(text) for text in doc_clean]
# proses ini mungkin agak lama
if 1 == 1:
    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)

pyLDAvis.save_html(
    LDAvis_prepared, '/Users/thaar/Documents/work/Project Ariq/LDA/testing 3/lda_test/ldavis_prepared_' + str(total_topics) + '.html')
