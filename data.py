import pandas as pd
import numpy as np
import nltk

import string
import re  # regex library

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# C:\Users\thaar\Documents\work\Project Ariq\LDA\testing 3\lda_test\content\dataBerita.xlsx
dataSB = pd.read_excel('content\dataDummy.xlsx',
                       engine='openpyxl')  # lokasi file
dataSB.head()
dataSB['textdata'] = dataSB['textdata'].str.lower()

print('Case Folding Result : \n')
print(dataSB['textdata'].head(5))
