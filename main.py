# David Levi and Sam Cheng
# Midterm
# Document Searching
# Based on the paper - StarSpace: "https://arxiv.org/abs/1709.03856"

'''
We based our project off of the paper:
	StarSpace: Embed All The Things!

The main goal of the paper was to design a general 
purpose neural network to compare 2 different types of 
entities.  For our implementation we followed StarSpace's
implementation of "Information Retrieval" found on page 3.
Specifically we tried to emulate the "Wikipedia Article
Search & Sentence Matching" (task 1) on page 6.

To do this we obtained a training set consisting of many 
different news articles.  We then grabbed a random sentence from
each news article and made that the label for that 
specific document.

We then trained a neural net to attempt to predict the label given
the document.  While this is obviously an extremely difficult task
it was able to produce a word embedding as a byproduct.

We were then able to average all of the word embeddings for
a given document and compare it to a specific query.  We ended up
sorting the documents by the spatial distance between the query
embedding and the document embedding and calculating the minimum
as well as the top-5 and top-10 statistics.
'''


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tqdm as tqdm
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from scipy import spatial
import dataset
import string

# [1] https://github.com/minsuk-heo/python_tutorial/blob/master/data_science/nlp/word2vec_tensorflow.ipynb
# [2] https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/
# [3] https://towardsdatascience.com/machine-learning-word-embedding-s
# [4] https://github.com/keras-team/keras/issues/3031

# Number of training samples
NUM_TRAIN = 10000

class datasetLoader(object):
	def __init__(self,dir_of_files,num_files):
		self.dir_name = dir_of_files
		self.num_files = num_files

	def get_x_and_y(self):
		news,labels = dataset.loopFiles(self.dir_name,self.num_files)
		return news,labels

	def getCodedWords(self):
		news,labels = self.get_x_and_y()
		ignore_words = set(stopwords.words('english'))
		manual_stops = ['(',')']
		ignore_words = list(ignore_words)+list(set(manual_stops))
		sentences = []
		for row in news:
			sentences.append(row);
		return sentences, labels


d = datasetLoader('./newsfiles/technology/',NUM_TRAIN+2000)

# Gets series of sentences/documents
docs,labels = d.getCodedWords()

# Create map that maps labels to its proper document
doc_label_map = {}
for i,lab in enumerate(labels):
	doc_label_map[lab] = docs[i]

# Credit for sentence cleaning is [2]
label_lines = []
for line in labels:
	tokens = word_tokenize(line)
	tokens = [w.lower() for w in tokens]
	table = str.maketrans('','',string.punctuation)
	stripped = [w.translate(table) for w in tokens]
	words = [word for word in stripped if word.isalpha()]
	stop_words = set(stopwords.words('english'))
	words = [w for w in words if not w in stop_words]
	label_lines.append(words)

doc_lines = []
for line in docs:
	tokens = word_tokenize(line)
	tokens = [w.lower() for w in tokens]
	table = str.maketrans('','',string.punctuation)
	stripped = [w.translate(table) for w in tokens]
	words = [word for word in stripped if word.isalpha()]
	stop_words = set(stopwords.words('english'))
	words = [w for w in words if not w in stop_words]
	doc_lines.append(words)


all_lines = doc_lines+label_lines
# Create Tokenizer
t = Tokenizer()
# Create unique indexes for words
t.fit_on_texts(all_lines)
# Vocab size
vSize = len(t.word_index) + 1

# Knobs
EMBED_INPUT_DIM = vSize
EMBED_OUTPUT_DIM = 1024 # 1024 best 22.5%, 256 also not bad
PAD_LEN = 30 # 20 had 22.7%
NUM_EPOCHS = 10
BATCH_SIZE = 32

# Turn each document into series of indexes that map to words
seq_docs = t.texts_to_sequences(doc_lines)
seq_labels = t.texts_to_sequences(label_lines)

# Pad the docs to maintain input shape
doc_lines_pad = keras.preprocessing.sequence.pad_sequences(seq_docs, 
			maxlen=PAD_LEN, padding='post')
label_lines_pad = keras.preprocessing.sequence.pad_sequences(seq_labels, 
			maxlen=PAD_LEN, padding='post')

# Split training and testing (We do some more splitting later)
doc_lines_pad_train = doc_lines_pad[:NUM_TRAIN]
label_lines_pad_train = label_lines_pad[:NUM_TRAIN]

doc_lines_pad_test = doc_lines_pad[NUM_TRAIN:]
label_lines_pad_test = label_lines_pad[NUM_TRAIN:]

# Define cosine similarity, credit [4]
def cos_sim(y_true, y_pred):
	def l2_normalize(x, axis):
		norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
		return K.maximum(x, K.epsilon()) / K.maximum(norm, K.epsilon())
	y_true = l2_normalize(y_true, axis=-1)
	y_pred = l2_normalize(y_pred, axis=-1)
	# We know we are getting a negative loss, but it worked well
	return -K.mean(y_true * y_pred, axis=-1)

# Create model
model = keras.Sequential()
# Embedding layer to learn embeddings with input size = Max sentence length
model.add(keras.layers.Embedding(EMBED_INPUT_DIM,EMBED_OUTPUT_DIM,input_length=PAD_LEN))
# Train model
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(PAD_LEN,activation = 'relu'))
model.compile(optimizer=keras.optimizers.Adam(lr=0.01),loss=cos_sim,metrics=['accuracy'])
model.summary()
model.fit(label_lines_pad_train, doc_lines_pad_train, epochs=NUM_EPOCHS, 
		batch_size=BATCH_SIZE, validation_split=0.2, verbose=1)
loss,acc = model.evaluate(doc_lines_pad_test, label_lines_pad_test, verbose=1)
print('Accuracy on training data is:',str(acc*100)+'%')

# Create another neural net to get actual embeddings 
# (We figured this was the best way)
model2 = keras.Sequential()
model2.add(keras.layers.Embedding(EMBED_INPUT_DIM,EMBED_OUTPUT_DIM,
			weights=model.layers[0].get_weights(),input_length=PAD_LEN))
model2.compile('rmsprop','mse')
model2.summary()


# Turn documents into a series of word embeddings using model
def encode_docs(in_docs):
	out = model2.predict(in_docs)
	return out

# Turning docs into word embeddings
embedded_docs_by_words = encode_docs(doc_lines_pad[NUM_TRAIN:])

# Create map from a document to its code
doc_codes = {}
for code,d in zip(embedded_docs_by_words,docs[NUM_TRAIN:]):
	doc_codes[d] = code

# Takes average of all word embeddings in the document
# This brings the dimension of the document embedding into 
# the same space as the query embedding
def minimize_doc_embedding(doc_dict):
	new_dict = {}
	for key in doc_dict.keys():
		newArr = []
		for dim in range(EMBED_OUTPUT_DIM):
			sumArr = 0
			for i in range(PAD_LEN):
				sumArr+=doc_dict[key][i][dim]
			newArr.append(float(sumArr)/float(PAD_LEN))
		new_dict[key] = newArr
	return new_dict

min_doc_codes = minimize_doc_embedding(doc_codes)

# Function to turn query into similar embedding
def indexQuery(model,que,t):
	padded_queries = []
	newQueries = []
	for q in que:
		words = text_to_word_sequence(q)
		query=""
		for i,w in enumerate(words):
			query+=w+" "
		newQueries.append(query)
	codeQuery = t.texts_to_sequences(newQueries)
	padded_query = keras.preprocessing.sequence.pad_sequences(codeQuery, 
				maxlen=PAD_LEN, padding='post')
	return padded_query

# Takes average of words in query embedding
def minimize_query_embedding(q_dict):
	new_dict = {}
	for q in q_dict.keys():
		newArr = []
		for dim in range(EMBED_OUTPUT_DIM):
			sumArr = 0
			for i in range(PAD_LEN):
				sumArr+=q_dict[q][i][dim]
			newArr.append(float(sumArr)/float(PAD_LEN))
		new_dict[q] = newArr
	return new_dict

# Embed query
padQ = indexQuery(model,labels,t)

# Get embeddings for queries
coded_qs = model2.predict(padQ)

# Create map from a query to its code
q_dict = {}
for q,code in zip(labels[NUM_TRAIN:],coded_qs[NUM_TRAIN:]):
	q_dict[q] = code

# Average the query embedding
min_q_dict = minimize_query_embedding(q_dict)

# This is a simple O(n^2) search operation that 
# finds the most similar document embedding to
# the query embedding and calculates the top-1,
# top-5, and top-10 statistics
def get_most_sim(q_sim_dict,d_sim_dict):
	num_right1 = 0
	num_right5 = 0
	num_right10 = 0
	total = 0

	for ind,q in enumerate(q_sim_dict.keys()):
		print("Evaluating:",ind,"out of",len(q_sim_dict.keys()))
		doc_sort = []
		doc_tracker = {}
		label_tracker = {}

		min_dist = 100000000
		min_d = "None"
		for d in d_sim_dict.keys():
			temp = abs(spatial.distance.cosine(q_sim_dict[q],d_sim_dict[d]))
			doc_tracker[temp] = d
			doc_sort.append(temp)

			if(temp<min_dist):
				min_dist = temp
				min_d = d

		if(q=="None"):
			continue

		doc_sort.sort()
		if(min_d==doc_label_map[q]):
			# IN TOP 1
			num_right1+=1
		else:
			for metric in doc_sort[:5]:
				if(doc_label_map[q] == doc_tracker[metric]):
					# IN TOP 5
					num_right5+=1
					break
			else:
				for m in doc_sort[:10]:
					if(doc_label_map[q] == doc_tracker[m]):
						# IN TOP 10
						num_right10+=1
						break

		total+=1
	# We put the probability of guessing to show the actual improvement we saw
	print("Guessing top 1 would be:",str((float(1)/float(total))*100)+"%")
	print("Top 1 Correct:",str((float(num_right1)/float(total))*100)+"%")
	print("Guessing top 5 would be:",str((float(5)/float(total))*100)+"%")
	print("Top 5 Correct:",str((float(num_right5+num_right1)/float(total))*100)+"%")
	print("Guessing top 10 would be:",str((float(10)/float(total))*100)+"%")
	print("Top 10 Correct:",str((float(num_right10+num_right5+num_right1)/float(total))*100)+"%")
	print("Total docs:",total)



get_most_sim(min_q_dict,min_doc_codes)
























