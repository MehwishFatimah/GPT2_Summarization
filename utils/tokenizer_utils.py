#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 18:04:52 2021
make it a module
@author: fatimamh

"""

#from transformers import GPT2Tokenizer
from transformers import AutoTokenizer

'''----------------------------------------------------------------
Takes tokenizer, text, summary and max sequence length.
Shape the input and tokenize it.
Returns ecodings contain ids and attention masks.

Args:
	tokenizer: tokenizer
	text	 : str
	sum      : str
	max_len	 : int
Return:
	encodings: dict
'''
def apply_tokenizer(tokenizer, text, sum, max_len):
	return tokenizer('<BOS>'+ text + '<SUMMARIZE>' + sum + '<EOS>', truncation=True, max_length=max_len, padding="max_length")

'''----------------------------------------------------------------
Takes text with length, truncate long text from desired length
Args:
	tokenizer: tokenizer
	df 		 : dataframe
	max_len	 : int
Return:
	df: dataframe
'''
def tokenize_text(tokenizer, df, max_len):
	#print('tokenize_text\n{}'.format(df.head(1)))
	df['encodings'] = df.apply(lambda x : apply_tokenizer(tokenizer, x['text'], x['summary'], max_len), axis=1)
	#encodings is dict type: contains input_ids and attention_mask
	del df['text']
	del df['summary']

	return df

'''----------------------------------------------------------------
Takes tokenizer, text, summary and max sequence length.
Shape the input and tokenize it.
Returns ecodings contain ids and attention masks.

Args:
	tokenizer: tokenizer
	text	 : str
	sum      : str
	max_len	 : int
Return:
	encodings: dict
'''
def tokenize_dataset(tokenizer, train, val, test, max_len):
	train = tokenize_text(tokenizer, train, max_len)
	val = tokenize_text(tokenizer, val, max_len)
	#test = tokenize_text(tokenizer, test, max_len)
	return train, val, test

'''----------------------------------------------------------------
Takes the path to load pretrained tokenizer along with sets for fine-tuning.

Args:
	path     : str
	train    : dataframe
	val  	 : dataframe
Return:
	tokenizer : tokenizer
	ignore_idx: int
'''
def process_tokenizer(path): #, train, val):

	#tokenizer = GPT2Tokenizer.from_pretrained(path, bos_token='<BOS>', eos_token='<EOS>', pad_token='<PAD>', cls_token ='<CLS>') 

	tokenizer = AutoTokenizer.from_pretrained(path)
	special_tokens = {'bos_token':'<BOS>', 'eos_token':'<EOS>', 'pad_token':'<PAD>', 'additional_special_tokens':['<SUMMARIZE>']}
	tokenizer.add_special_tokens(special_tokens)

	print('tokenizer len: {}'.format(len(tokenizer)))
	#tokenizer.train(val, trainer)
	#print('After trainer tokenizer len: {}'.format(len(tokenizer)))
	
	ignore_idx = tokenizer.pad_token_id
	
	return tokenizer, ignore_idx
	