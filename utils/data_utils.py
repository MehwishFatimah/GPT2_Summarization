#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 14:49:41 2021
final
@author: fatimamh
"""
import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split

'''----------------------------------------------------------------
Takes text with length, truncate long text from desired length
Args:
	text: str
	len : int
Return:
	text: str
'''
def short_text(text, len):
	text = text.split()
	#print(len(text))
	#len= len-1
	s_text = text[0:len]
	s_text = ' '.join(s_text)
	return  s_text


'''----------------------------------------------------------------
Takes dataframe with text and summary length to truncate them.
Also calculates len of text
Args:
	df      : dataframe
	max_text: int
	max_sum : int
Return:
	df      : dataframe
'''
def process_dataframe(df, max_text, max_sum):

	df['text'] = df['text'].apply(lambda x: short_text(x, max_text))
	df['summary'] = df['summary'].apply(lambda x: short_text(x, max_sum))
	df['text_len'] = df['text'].apply(lambda x: len(x.split()))
	#print(df['summary'].str.split().str.len())
	#df['summary'] = df['summary'].apply(lambda x: ' <CLS> ' + x) # do this step in data loader
	#df['ts'] = df[['text', 'summary']].apply(lambda x: ''.join(x), axis=1)
	#print(df['ts'].str.split().str.len())

	return df


'''----------------------------------------------------------------
Takes dataframe with split ratio to to split data.
Args:
	df   : dataframe
	sr   : float
Return:
	(3)df: dataframe
'''
def split_data (df, sr):
	train, val_test = train_test_split(df,test_size=sr)
	val, test = train_test_split(val_test,test_size=0.5)
	return train, val, test


'''----------------------------------------------------------------
This is the main function of this file
Takes file with text and summary length to truncate them.
Process dataframe, split it into 3 sets and return them.
Args:
	file_path: str
	max_text : int
	max_sum  : int
	sr 		 : float
Return:
	(3)df    : dataframe
'''
def process_data(file, max_text, max_sum, sr):
	
	# load into a data frame
	df = pd.read_csv(file)  
	df = process_dataframe(df, max_text, max_sum)
	train, val, test = split_data(df, sr)
	print('train size: {}'.format(len(train)))
	print('val size: {}'.format(len(val)))
	print('test size: {}'.format(len(test)))
	print('test head:\n{}'.format(test.head(1)))

	return train, val, test
	