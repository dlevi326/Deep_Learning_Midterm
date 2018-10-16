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

N_GRAM = 1

class CsvLoader(object):
	def __init__(self,filename):
		self.file = open(filename, mode='r')

	def tokenizeCsv(self,sentences):
		t = Tokenizer()
		t.fit_on_texts(sentences)
		encoded_words = t.texts_to_matrix(sentences, mode='count')
		return encoded_words

	def getCodedWords(self):
		csv_reader = csv.DictReader(self.file,fieldnames=['Category','Title','Summary'])
		ignore_words = set(stopwords.words('english'))
		manual_stops = ['(',')']
		ignore_words = list(ignore_words)+list(set(manual_stops))
		
		sentences = []

		for row in csv_reader:
			sentences.append(row['Summary']);
		
		return self.tokenizeCsv(sentences), sentences

	def createDataFrame(self,sentences):
		data = []
		for sent in senteces:
			for idx, word in enumerate(sent):
				for s in sentence[max(idx-N_GRAM,0) : min(idx+N_GRAM, len(sent))+1]:
					if s!=word:
						data.append([word,s])

		df = pd.DataFram(data, columns= ['input', 'label'])
		return df



csvTrain = CsvLoader('./train.csv')
codedWords, sentences = csvTrain.getCodedWords()
#print(len(words[10]))
INPUT_DIM = len(codedWords[0])
df = csvTrain.createDataFrame(sentences)
print(df['input'][:10],df['label'][:10])


























'''
import xml.etree.ElementTree as ET


class XmlLoader(object):
	def __init__(self,filename):
		self.filename = filename
		self.file = open(filename, mode='r')

	def printLines(self,numLines):
		for i in range(numLines):
			print(self.file.readline())

	def parseXML(self,numLines):
		tree = ET.parse(self.filename)
		root = tree.getroot()

		for i,child in enumerate(root):
			print(i)
			if(i>numLines):
				break
			print(child.tag, child.attrib)


xmlFileTrain = XmlLoader('./enwiki-latest-abstract.xml')
xmlFileTrain.parseXML(100);
'''

