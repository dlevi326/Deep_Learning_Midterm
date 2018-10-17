# David Levi and Sam Cheng
# Midterm
# Wikipedia Article Search
# Base on the paper - StarSpace: "https://arxiv.org/abs/1709.03856"

# [1] https://github.com/minsuk-heo/python_tutorial/blob/master/data_science/nlp/word2vec_tensorflow.ipynb
# [2] https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/

import csv
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
import tqdm as tqdm
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import keras
from keras.preprocessing.text import text_to_word_sequence
# define the document

N_GRAM = 1
MAX_SENTENCE_LENGTH = 25

class CsvLoader(object):
	def __init__(self,filename):
		self.file = open(filename, mode='r')

	def tokenizeCsv(self,sentences):
		t = Tokenizer()
		t.fit_on_texts(sentences)
		#encoded_words = t.texts_to_sequences(sentences)
		return t

	def getCodedWords(self):
		csv_reader = csv.DictReader(self.file,fieldnames=['Category','Title','Summary'])
		ignore_words = set(stopwords.words('english'))
		manual_stops = ['(',')']
		ignore_words = list(ignore_words)+list(set(manual_stops))
		
		sentences = []

		for row in csv_reader:
			words = text_to_word_sequence(row['Summary'])
			sentence = ""
			for w in words:
				sentence+=w+" "
			sentences.append(sentence);
		
		return self.tokenizeCsv(sentences), sentences

	def createDataFrame(self,sentences):
		data = []
		for i,sent in enumerate(sentences):
			#print('Sentence',i,'out of',len(sentences))
			for idx, word in enumerate(sent):
				for s in sent[max(idx-N_GRAM,0) : min(idx+N_GRAM, len(sent))+1]:
					if s!=word:
						data.append([word,s])

		df = pd.DataFrame(data, columns= ['input', 'label'])
		return df





csvTrain = CsvLoader('./smallTrain.csv')

# Gets series of sentences/documents
token, sentences = csvTrain.getCodedWords() 

# Create Tokenizer
t = Tokenizer()
# Create unique indexes for words
t.fit_on_texts(sentences)
# Vocab size
vSize = len(t.word_index) + 1
# Turn each document into a bunch of indexes that map to words
encoded_docs = t.texts_to_sequences(sentences)

# Knobs
MAX_SENTENCE_LENGTH = 20
EMBED_INPUT_DIM = vSize
EMBED_OUTPUT_DIM = 2

# Pad the docs to maintain input shape
padded_docs = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=MAX_SENTENCE_LENGTH, padding='post')

# Create model
model = keras.Sequential()
# Embedding layer to learn embeddings with input size = Max sentence length
model.add(keras.layers.Embedding(EMBED_INPUT_DIM,EMBED_OUTPUT_DIM,input_length=MAX_SENTENCE_LENGTH))
# Train model
model.compile('rmsprop','mse')

#output_array = model.predict(input_array)
#output_array = np.reshape(output_array,newshape=[MAX_SENTENCE_LENGTH,EMBED_OUTPUT_DIM])

# Turn documents into a series of word embeddings using model
def encode_docs(docs):
	out = model.predict(docs)
	return out

# Turning docs into word embeddings
embedded_docs_by_words = encode_docs(padded_docs[:4])
print(embedded_docs_by_words)

# Function to turn query into similar embedding
def embedQuery(model,que,t):
	padded_queries = []
	newQueries = []
	for q in que:
		words = text_to_word_sequence(q)
		query=""
		for i,w in enumerate(words):
			query+=w+" "
		newQueries.append(query)
	codeQuery = t.texts_to_sequences(newQueries)
	padded_query = keras.preprocessing.sequence.pad_sequences(codeQuery, maxlen=MAX_SENTENCE_LENGTH, padding='post')
	#padded_queries.append(padded_queries)
	return padded_query

q = ["This is a test ok?"]
# Embed query
padQ = embedQuery(model,q,t)
#print(model.predict(codeQ))

























