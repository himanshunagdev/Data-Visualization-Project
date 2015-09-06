# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 17:27:36 2015

@author: NAGDEV
"""
import pandas
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import cross_validation
import numpy as np
from numpy import array
from sklearn.neighbors import KNeighborsClassifier

import matplotlib
import random
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from nltk.corpus import stopwords

from collections import Counter

readFile = pd.read_csv('reviews.txt', sep='\t', header=None, names=['Review', 'Sentiment'])
reviews, sentiments = readFile['Review'], readFile['Sentiment']

len_vs_senti=[ (len(r), sentiment) for r, sentiment in zip(reviews, sentiments) ]#getting Lenth vs Sentiment List 
#converting into Data Frame
lenVsSent_df = pd.DataFrame(len_vs_senti) 
#Correlation array
x1 = (np.corrcoef(lenVsSent_df))
#plotting histograms for positve and negative sentiments
lenVsSent_df[lenVsSent_df[1]==1][0].hist(bins=100)
lenVsSent_df[lenVsSent_df[1]==0][0].hist(bins=100)
#seaborn visualization for positive and negative sentiments
sns.kdeplot(lenVsSent_df[lenVsSent_df[1]==0][0])
sns.kdeplot(lenVsSent_df[lenVsSent_df[1]==1][0])

#seaborn visualization for correlation
sns.kdeplot(lenVsSent_df)

#removing stopwords
sw = list(stopwords.words("english"))
reviewsRemStop = [ [ word for word in review.split() if word not in sw ] for review in reviews ]
stopWords_df = pd.DataFrame({'Review': reviewsRemStop, 'Sentiment': sentiments})
len_vs_senti_SW=[ (len(p), sentiment) for p, sentiment in zip(reviewsRemStop, sentiments) ]
#convert to Data Frame
lenVsSent_SW_df = pd.DataFrame(len_vs_senti_SW)

#plotting the length vs sentiment after removing stopwords
sns.pairplot(lenVsSent_SW_df)


#Most Common words in the reviews
mostCommonLenVsSent=Counter(word.lower() for r in reviews for word in r.split()).most_common(100)
lenVsSent_df_MostCom=pd.DataFrame(mostCommonLenVsSent)
sns.kdeplot(lenVsSent_df_MostCom)
key=[]
value=[]
for pairs in mostCommonLenVsSent:
    key.append(pairs[0])
    value.append(pairs[1])
sns.barplot(key,value,palette="BuGn_d")

# Check difference in length vs sentiment for positive and negative sentences by focusing on only short sentences with stop words removed
read_file = pd.read_csv('reviews.txt', sep='\t', header=None, names=['Review', 'Sentiment'])
stopWords = list(stopwords.words("english"))
reviewsSentences = [ [ word for word in review.split() if word not in sw ] for review in reviews ]
#finding length vs sentences
lenVsSentSentences = [ (len(r), sentiment) for r, sentiment in zip(reviewsSentences, sentiments) ]
#getting data frame
lenSentSentence_df = pd.DataFrame(lenVsSentSentences)
# convert negative list dataframe to list
negativeSent= lenSentSentence_df[lenSentSentence_df[1]==0][0] 
#getting short sentences for negative sentiments dataframe
shortNegative= [ val for val in negativeSent if val in range(0,20) ]
# DataFrame of Positive reviews
positiveSent=lenSentSentence_df[lenSentSentence_df[1]==1][0] 
#getting short sentences for positive sentiments dataframe
shortPositive= [ val for val in positiveSent if val in range(0,20) ]
#Check difference in length vs sentiment for short
differnceShort = abs (len(shortPositive) - len(shortNegative))
#getting long sentences for negative sentiments dataframe
longNegative= [ val for val in negativeSent if val in range(20,50) ]
#getting long sentences for positive sentiments dataframe
longPositive= [ val for val in positiveSent if val in range(20,50) ]
differnceLong = abs (len(longPositive) - len(longNegative))
