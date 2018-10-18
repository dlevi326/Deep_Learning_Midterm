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
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from copy import copy
import dataset
# define the document

N_GRAM = 1
MAX_SENTENCE_LENGTH = 25

class datasetLoader(object):
	def __init__(self,dir_of_files):
		self.dir_name = dir_of_files

	def get_x_and_y(self):
		news,labels = dataset.loopFiles(self.dir_name)
		return news,labels

	def tokenizeDoc(self,sentences):
		t = Tokenizer()
		t.fit_on_texts(sentences)
		#encoded_words = t.texts_to_sequences(sentences)
		return t

	def getCodedWords(self):
		#csv_reader = csv.DictReader(self.file,fieldnames=['Category','Title','Summary'])
		news,labels = self.get_x_and_y()
		ignore_words = set(stopwords.words('english'))
		manual_stops = ['(',')']
		ignore_words = list(ignore_words)+list(set(manual_stops))
		
		sentences = []

		for row in news:
			words = text_to_word_sequence(row)
			sentence = ""
			for w in words:
				sentence+=w+" "
			sentences.append(sentence);
		
		return self.tokenizeDoc(sentences), sentences, labels


d = datasetLoader('./newsfiles/newsfiles/')
# Gets series of sentences/documents
token,docs,labels = d.getCodedWords()

doc_label_map = {}
for i,lab in enumerate(labels):
	doc_label_map[lab] = docs[i]
#print(doc_label_map)

#csvTrain = CsvLoader('./smallTrain.csv')

# Gets series of sentences/documents
#token, docs = csvTrain.getCodedWords() 

# Create Tokenizer
t = Tokenizer()
# Create unique indexes for words
t.fit_on_texts(docs)
# Vocab size
vSize = len(t.word_index) + 1
# Turn each document into a bunch of indexes that map to words
encoded_docs = t.texts_to_sequences(docs)

# Knobs
MAX_SENTENCE_LENGTH = 30
EMBED_INPUT_DIM = vSize
EMBED_OUTPUT_DIM = 32

# Pad the docs to maintain input shape
padded_docs = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=MAX_SENTENCE_LENGTH, padding='post')

# Create model
model = keras.Sequential()
# Embedding layer to learn embeddings with input size = Max sentence length
model.add(keras.layers.Embedding(EMBED_INPUT_DIM,EMBED_OUTPUT_DIM,input_length=MAX_SENTENCE_LENGTH))
# Train model
model.compile('rmsprop','mse')

model.summary()

#output_array = model.predict(input_array)
#output_array = np.reshape(output_array,newshape=[MAX_SENTENCE_LENGTH,EMBED_OUTPUT_DIM])

# Turn documents into a series of word embeddings using model
def encode_docs(in_docs):
	out = model.predict(in_docs)
	return out




# Turning docs into word embeddings (Change to add docs)
embedded_docs_by_words = encode_docs(padded_docs)
doc_codes = {}
for code,d in zip(embedded_docs_by_words,docs):
	doc_codes[d] = code

#print(doc_codes)


def minimize_doc_embedding(doc_dict):
	new_dict = {}
	for key in doc_dict.keys():
		newArr = []
		for dim in range(EMBED_OUTPUT_DIM):
			#newArr = []
			sumArr = 0
			for i in range(MAX_SENTENCE_LENGTH):
				sumArr+=doc_dict[key][i][dim]
				#print('Document:',key,'number col:',dim,'number row:',i,'=',doc_dict[key][i][dim])
			newArr.append(float(sumArr)/float(MAX_SENTENCE_LENGTH))
			#print('Document:',key,'number col:',dim,'=',float(sumArr)/float(MAX_SENTENCE_LENGTH))
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
	padded_query = keras.preprocessing.sequence.pad_sequences(codeQuery, maxlen=MAX_SENTENCE_LENGTH, padding='post')
	#padded_queries.append(padded_queries)
	return padded_query

def minimize_query_embedding(q_dict):
	new_dict = {}
	for q in q_dict.keys():
		newArr = []
		for dim in range(EMBED_OUTPUT_DIM):
			sumArr = 0
			for i in range(MAX_SENTENCE_LENGTH):
				sumArr+=q_dict[q][i][dim]
			newArr.append(float(sumArr)/float(MAX_SENTENCE_LENGTH))
			#print('Document:',key,'number col:',dim,'=',float(sumArr)/float(MAX_SENTENCE_LENGTH))
		new_dict[q] = newArr
	return new_dict


#qList = ["usatoday retail sales bounced back bit july and new claims jobless benefits fell week the government said thursday indicating the economy is improving from a midsummer slump",
#		"America Online on Thursday said it  plans to sell a low-priced PC targeting low-income and minority service."]

qList = labels

# Embed query
padQ = indexQuery(model,qList,t)
coded_qs = model.predict(padQ)

# Create dict of queries and codes
q_dict = {}

for q,code in zip(qList,coded_qs):
	q_dict[q] = code

# Create query embedding
min_q_dict = minimize_query_embedding(q_dict)
#print(min_q_dict)

#print(min_doc_codes)


def get_most_sim(q_sim_dict,d_sim_dict):
	num_right1 = 0
	num_right5 = 0
	num_right10 = 0
	total = 0

	for q in q_sim_dict.keys():
		doc_sort = []
		doc_tracker = {}
		label_tracker = {}

		min_dist = 100000000
		min_d = "None"
		for d in d_sim_dict.keys():
			#temp = cosine_similarity((q_sim_dict[q]),(d_sim_dict[d]))
			# Should switch to cosine sim
			temp = abs(spatial.distance.cosine(q_sim_dict[q],d_sim_dict[d]))
			doc_tracker[temp] = d
			doc_sort.append(temp)

			if(temp<min_dist):
				min_dist = temp
				min_d = d
			#if(d[0:3]=="new"):
				#print('Distance metric:',temp,'with document:',d)


		#print('-'*60)
		#print('Most similar to query:',q)
		#print("Is document:",min_d)
		#print("Real document:",doc_label_map[q])
		if(q=="None"):
			continue

		doc_sort.sort()
		if(min_d==doc_label_map[q]):
			num_right1+=1
			#print("YES")
			#print(min_d)
		else:
			#print("NO")
			for metric in doc_sort[:5]:
				if(doc_label_map[q] == doc_tracker[metric]):
					#print("IN TOP 5")
					num_right5+=1
					break
			else:
				for m in doc_sort[:10]:
					if(doc_label_map[q] == doc_tracker[m]):
						#print("IN TOP 10")
						num_right10+=1
						break

			#if(doc_label_map in doc_tracker[doc_sort[:5]]):
			#	print("IN TOP 5")

		total+=1
	print("Guessing top 1 would be:",str((float(1)/float(total))*100)+"%")
	print("Top 1 Correct:",str((float(num_right1)/float(total))*100)+"%")
	print("Guessing top 5 would be:",str((float(5)/float(total))*100)+"%")
	print("Top 5 Correct:",str((float(num_right5+num_right1)/float(total))*100)+"%")
	print("Guessing top 10 would be:",str((float(10)/float(total))*100)+"%")
	print("Top 10 Correct:",str((float(num_right10+num_right5+num_right1)/float(total))*100)+"%")
	print("Total docs:",total)



get_most_sim(min_q_dict,min_doc_codes)
























