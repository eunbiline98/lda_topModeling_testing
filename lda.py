from collections import Counter
import pandas as pd
import numpy as np
import nltk
import gensim
from gensim import corpora
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter
from nltk.corpus import stopwords
import pyLDAvis.gensim
import pickle
import pyLDAvis
import os

import string
import re  # regex library

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

dataSB = pd.read_excel(
    'content\dataDummy.xlsx', engine='openpyxl')  # lokasi file
dataSB.head()
dataSB['textdata'] = dataSB['textdata'].str.lower()

print('Case Folding Result : \n')
print(dataSB['textdata'].head(25))


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


dataSB['textdata'] = dataSB['textdata'].apply(remove_tweet_special)

# remove number


def remove_number(text):
    return re.sub(r"\d+", "", text)


dataSB['textdata'] = dataSB['textdata'].apply(remove_number)

# remove punctuation


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


dataSB['textdata'] = dataSB['textdata'].apply(remove_punctuation)

# remove whitespace leading & trailing


def remove_whitespace_LT(text):
    return text.strip()


dataSB['textdata'] = dataSB['textdata'].apply(remove_whitespace_LT)

# remove multiple whitespace into single whitespace


def remove_whitespace_multiple(text):
    return re.sub('\s+', ' ', text)


dataSB['textdata'] = dataSB['textdata'].apply(remove_whitespace_multiple)

# remove single char


def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)


dataSB['textdata'] = dataSB['textdata'].apply(remove_singl_char)

# NLTK word tokenize


def word_tokenize_wrapper(text):
    return word_tokenize(text)


dataSB['textdata_tokens'] = dataSB['textdata'].apply(word_tokenize_wrapper)

print('Tokenizing Result : \n')
print(dataSB['textdata_tokens'].head())

# NLTK calc frequency distribution


def freqDist_wrapper(text):
    return FreqDist(text)


dataSB['textdata_tokens_fdist'] = dataSB['textdata_tokens'].apply(
    freqDist_wrapper)

print('Frequency Tokens : \n')
print(dataSB['textdata_tokens_fdist'].head().apply(lambda x: x.most_common()))

nltk.download('stopwords')

# ----------------------- get stopword from NLTK stopword -------------------------------
# get stopword indonesia
list_stopwords = stopwords.words('indonesian')


# ---------------------------- manualy add stopword  ------------------------------------
# append additional stopword
list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo',
                       'kalo', 'amp', 'biar', 'bikin', 'bilang',
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih',
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't',
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah', 'bisnis', 'pandemi', 'indonesia'])

# convert list to dictionary
list_stopwords = set(list_stopwords)

# remove stopword pada list token


def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]


dataSB['textdata_tokens_WSW'] = dataSB['textdata_tokens'].apply(
    stopwords_removal)

# print(dataSB['textdata_tokens_WSW'].head())

normalizad_word = pd.read_excel(
    'content\dataDummy.xlsx', engine='openpyxl')  # lokasi file

normalizad_word_dict = {}

for index, row in normalizad_word.iterrows():
    if row[0] not in normalizad_word_dict:
        normalizad_word_dict[row[0]] = row[1]


def normalized_term(document):
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]


dataSB['textdata_normalized'] = dataSB['textdata_tokens_WSW'].apply(
    normalized_term)

dataSB['textdata_normalized'].head(25)

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stemmed


def stemmed_wrapper(term):
    return stemmer.stem(term)


term_dict = {}

for document in dataSB['textdata_normalized']:
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


dataSB['textdata_tokens_stemmed'] = dataSB['textdata_normalized'].swifter.apply(
    get_stemmed_term)

print(dataSB['textdata_tokens_stemmed'])

# ----------------------- get stopword from NLTK stopword -------------------------------
# get stopword indonesia
list_stopwords = stopwords.words('indonesian')


# ---------------------------- manualy add stopword  ------------------------------------
# append additional stopword
list_stopwords.extend(["ada", "tan", "ton", "pt", "komentar", "juta", "unit", "menang", "artikel",
                       "smartphone", "tagar", "sedia", "kaskus", "seksi"])

# convert list to dictionary
list_stopwords = set(list_stopwords)

# remove stopword pada list token


def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]


dataSB['textdata_tokens_stemmed2'] = dataSB['textdata_tokens_stemmed'].apply(
    stopwords_removal)

print(dataSB['textdata_tokens_stemmed2'].head())


for i in range(len(dataSB)):
    a = dataSB.iloc[i][0]
    document.append(a)

document[0:5]

doc_clean = dataSB['textdata_tokens_stemmed2']
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
print(df_dominant_topic.head(25))

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
