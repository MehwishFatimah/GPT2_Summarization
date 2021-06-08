#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:36:41 2021

@author: fatimamh

Pretrained models of HF for summarization
Env: conda activate hugging_face 

CUDA: module load CUDA/10.1.243-GCC-8.3.0
"""

#import libraries
import os
import time
import argparse
import resource

import pandas as pd
import numpy as np
import random

import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True # for fast training

#torch.manual_seed(42)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import data_config as config

from utils.time_mem_utils import get_size

from utils.data_utils import process_data

from utils.tokenizer_utils import process_tokenizer
from utils.tokenizer_utils import tokenize_dataset
from utils.loader_utils import get_gpt2_dataset

from datasets import list_metrics, load_metric
from modules.training import Train
from modules.generate import Generate
#from modules.generation import generate_sample
#from modules.generation import generate_beam_sample


torch.cuda.empty_cache()
device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))
#import transformers
#print('use transformers version = ',transformers.__version__) # make sure it is 2.6.0


'''----------------------------------------------------------------
'''
if __name__ == "__main__":
    #args       = parser.parse_args()
	'''----------------------------------------------------------------
	2. Data processing
	'''
	
	file_name = config.file_name
	file = os.path.join(config.data_dir, file_name)
	if os.path.exists(file):
		print(file)
		train, val, test = process_data(file, config.max_text, config.max_sum, config.sr)
	else:
		print('Data file does not exist.')
	

	'''----------------------------------------------------------------
	3. load tokenizer 
	''' 
	if os.path.exists(config.model_dir):
		tokenizer, ignore_index = process_tokenizer(config.model_dir) #, train, val) 		
	else:
		print('Tokenizer path does not exist.')
	
	#save tokenizer 
	tokenizer_dir = os.path.join(config.out_dir, config.final_model)
	if not os.path.exists(tokenizer_dir):
		os.makedirs(tokenizer_dir) # Create output directory if needed
	
	tokenizer.save_pretrained(tokenizer_dir)
	tokenizer_len = len(tokenizer)
	print('ignore_index: {}'.format(ignore_index))
	print('max_len: {}'.format(config.max_seq_len))

	train, val, test = tokenize_dataset(tokenizer, train, val, test, config.max_seq_len)
	#save tokenized data
	data_dir = os.path.join(config.out_dir, config.processed_set)
	if not os.path.exists(data_dir):
		os.makedirs(data_dir) # Create output directory if needed
	file = os.path.join(data_dir,"train.csv")
	train.to_csv(file, index=False)

	file = os.path.join(data_dir,"val.csv")
	val.to_csv(file, index=False)
	
	file = os.path.join(data_dir,"test.csv")
	test.to_csv(file, index=False)
	#print(train.head(1))
	
	'''----------------------------------------------------------------
	4. Data loader 
	'''
	train_dataset, val_dataset= get_gpt2_dataset(train, val) 
	
	b = train_dataset.__getitem__(0)
	#print(b['text'])#print(len(b['text']))#print(b['mask'])#print(b['s_idx'])
	train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = config.batch_size)
	val_dataloader = DataLoader(val_dataset, sampler = SequentialSampler(val_dataset), batch_size = config.batch_size)
	
	train_loader_len = len(train_dataloader)
	
	'''----------------------------------------------------------------
	5. Training
	'''
	# fine tune pretrained model 
	if os.path.exists(config.model_dir):
		train = Train(device, config.model_dir, tokenizer_len, ignore_index, train_loader_len)
		train.train_model(train_dataloader, val_dataloader)
	else:
		print('Model path does not exist.')
	# 
	#resume finetuning
	# other path
	
	'''----------------------------------------------------------------
	6. Generation
	'''
		
	#Load a trained model and vocabulary that you have fine-tuned
	model_path = os.path.join(config.out_dir, config.final_model)
	tokenizer_path = os.path.join(config.out_dir, config.final_model)
	data_path = os.path.join(config.out_dir, config.processed_set)
	file = os.path.join(data_path, "test.csv")
	if os.path.exists(file):
		test_dataset = pd.read_csv(file)
		
		if os.path.exists(model_path) and os.path.exists(tokenizer_path):
			text_gen = Generate(device, model_path)
			#0= top_k, 1= beam_search, 2= both
			text_gen.generation(test_dataset, gen_type=config.gen_type)
			#generate_sample(test_dataset, tokenizer, model, device, num=5, length=config.max_sum, temperature=1, top_k=10, top_p=0.5)
			#generate_beam_sample(test_dataset, tokenizer, model, device, num=5, length=config.max_sum)
			
		else:
			print('Model/Vocab path does not exist.')
	
	else:
		print('Test dataset does not exist.')
	
	
	'''----------------------------------------------------------------
	7. Evaluation
	'''
	metrics_list = list_metrics()
	#print(', '.join(metric for metric in metrics_list))
	#accuracy, bertscore, bleu, bleurt, cer, comet, coval, cuad, f1, 
	#gleu, glue, indic_glue, matthews_correlation, meteor, pearsonr, 
	#precision, recall, rouge, sacrebleu, sari, seqeval, spearmanr, 
	#squad, squad_v2, super_glue, wer, xnli

	metric = load_metric('rouge') #, 'mrpc')
	summaries = os.path.join(config.out_dir, config.generated)
	file = os.path.join(summaries, config.topk_file)
	out_file = os.path.join(config.out_dir, config.topk_score_file)
	
	df = pd.read_csv(file)
	df['scores'] = pd.Series()
	for index, row in df.iterrows():
		prediction = []
		prediction.append(row['generated']) ##.split(" ") #.tolist()
		#print(prediction)
		reference = []
		reference.append(row['summary']) #.split(" ")
		#metric.add(predictions= prediction, references= reference)

		score = metric.compute(predictions= prediction, references= reference)
		#print(score)
		print(type(score))
		row['scores'] = str(score)

	
	df.to_csv(out_file, encoding='utf-8')


	'''------------------------------------------------------------------
	'''
	usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
	memory = get_size(usage)

	print ('\n-------------------Memory and time usage:  {}.--------------------\n'.format(memory))