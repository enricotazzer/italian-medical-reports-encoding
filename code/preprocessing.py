import pandas as pd
import numpy as np
from utils import * 
import string
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_NB_WORDS = 3000
MAX_SEQUENCE_LENGTH = 165

def bow(X):
	tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
	tokenizer.fit_on_texts(X)
	word_index = tokenizer.word_index
	bow = tokenizer.texts_to_sequences(X)
	return bow

def tfidf(X):
	tfidf = TfidfVectorizer()
	tfidf.fit(X)
	return tfidf.transform(X)

def w2v(X, i):
	words = [sentence.split(' ') for sentence in X]
	w2v_model = Word2Vec(words, window=5, workers=4, min_count=i,vector_size=MAX_SEQUENCE_LENGTH)
	return w2v_model
