# importing libs
from typing import List, Any
import pandas as pd
import numpy as np
from scipy import optimize
import re
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

# importing raw data
data_neg = pd.read_pickle('imdb_raw_neg.pickle')
data_pos = pd.read_pickle('imdb_raw_pos.pickle')

# data preprocessing
data_neg = np.array(data_neg)
data_neg = data_neg.reshape((len(data_neg), 1))
data_pos = np.array(data_pos)
data_pos = data_pos.reshape((len(data_pos), 1))


def structure(x_var: np.array, score: string):
    # this function structure the data and adds labels for
    # each review with [0, 1]
    lenght = len(x_var)
    # x_var = x_var.rashape((lenght,1))
    if score == 'one':
        target = np.ones((lenght, 1))
    else:
        target = np.zeros((lenght, 1))
    return np.hstack((target, x_var))


negatives = structure(data_neg, score='zero')
postiives = structure(data_pos, score='one')
df = np.vstack((negatives, postiives))


# cleaning text
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + \
           ' '.join(emoticons).replace('-', '')
    return text


# putting data into a dataframe
df = pd.DataFrame(df, columns=['score', 'text'])
df['text'] = df['text'].apply(preprocessor)  # applying our preprocessor

porter = PorterStemmer()


def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


nltk.download('stopwords')
stop = stopwords.words('english')

# data split for traning
X_train = df.loc[:25000, 'text'].values
y_train = df.loc[:25000, 'score'].values
X_test = df.loc[:25000:, 'text'].values
y_test = df.loc[:25000:, 'score'].values

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None], 'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l2'], 'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)], 'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter], 'vect__use_idf': [False],
               'vect__norm': [None], 'clf__penalty': ['l2'], 'clf__C': [1.0, 10.0, 100.0]}, ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0))]
                    )

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)
