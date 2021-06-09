#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 12:51:19 2021

@author: fatimamh
"""
import os
import pandas as pd
import numpy as np

from datasets import list_metrics, load_metric

import data_config as config

class Evaluate(object):
	'''----------------------------------------------------------------
	Initialize evaluation object 
	Args:
		model_path	 	: str
	Return: 
		object
	'''
	def __init__(self, device):

		self.device = device
		self.metric = ""
		self.folder = os.path.join(config.out_dir, config.results)
		

	'''----------------------------------------------------------------
	Rouge score
	Args:
		pred  : str 
		ref   : str
	Return:
		score : dict
	'''
	def compute_rouge_all(self, pred, ref):

		prediction = []
		prediction.append(pred) 
		reference = []
		reference.append(ref) 
		#compute takes list
		score = self.metric.compute(predictions= prediction, references= reference)

		return score

	'''----------------------------------------------------------------
	Args:
		pred  : str 
		ref   : str 
		r_type: str
	Return:
		r_mid_p, r_mid_r, r_mid_f: float
	'''
	def compute_rouge(self, pred, ref, r_type):

		prediction = []
		prediction.append(pred) 
		reference = []
		reference.append(ref) 
		#compute takes list
		score = self.metric.compute(predictions= prediction, references= reference, rouge_types=[r_type])[r_type].mid
		r_mid_p = round(score.precision, 4)
		r_mid_r = round(score.recall, 4)
		r_mid_f = round(score.fmeasure, 4)

		return r_mid_p, r_mid_r, r_mid_f

	'''----------------------------------------------------------------
	Args:
		file: file
	Return: None
	'''
	def rouge_scores(self, file):

		df = pd.read_csv(file)
		self.metric = load_metric("rouge")
		df['all_scores'] = df.apply(lambda x: self.compute_rouge_all(x['generated'], x['summary']), axis=1)
		
		df['r1_mid_precision'], df['r1_mid_recall'], df['r1_mid_fmeasure'] \
		= zip(*df.apply(lambda x: self.compute_rouge(x['generated'], x['summary'], "rouge1"), axis=1))
		
		df['r2_mid_precision'], df['r2_mid_recall'], df['r2_mid_fmeasure'] \
		= zip(*df.apply(lambda x: self.compute_rouge(x['generated'], x['summary'], "rouge2"), axis=1))

		df['rL_mid_precision'], df['rL_mid_recall'], df['rL_mid_fmeasure'] \
		= zip(*df.apply(lambda x: self.compute_rouge(x['generated'], x['summary'], "rougeL"), axis=1))

		df['rLsum_mid_precision'], df['rLsum_mid_recall'], df['rLsum_mid_fmeasure'] \
		= zip(*df.apply(lambda x: self.compute_rouge(x['generated'], x['summary'], "rougeLsum"), axis=1))
		
		file = os.path.join(self.folder, config.topk_rouge_file)
		df.to_csv(file, encoding='utf-8')

	
	'''----------------------------------------------------------------
	Bleu score
	Args:
		pred  : str 
		ref   : str
	Return:
		score : dict
	'''
	def compute_bleu_all(self, pred, ref):

		prediction = []
		prediction.append(pred.split()) #list of list
		ref_list = []
		ref_list.append(ref.split()) 
		reference = []
		reference.append(ref_list) #list of list of list is required
		#print(reference)
		#print(type(reference))
		#compute takes list
		score = self.metric.compute(predictions= prediction, references= reference)

		return score

	'''----------------------------------------------------------------
	Args:
		pred  : str 
		ref   : str 
	Return:
		bs, p, bp, ratio, r_length, p_length: float except p (list)
	'''
	def compute_bleu(self, pred, ref):

		prediction = []
		prediction.append(pred.split()) #list of list
		ref_list = []
		ref_list.append(ref.split()) 
		reference = []
		reference.append(ref_list) #list of list of list is required
		#print(reference)
		#print(type(reference)) 
		#compute takes list
		score = self.metric.compute(predictions= prediction, references= reference)
		#print(score.keys())
		bs = score.get('bleu')
		p = score.get('precisions')
		bp = score.get('brevity_penalty')
		ratio = score.get('length_ratio')
		r_length = score.get('reference_length')
		p_length = score.get('translation_length')

		return bs, p, bp, ratio, r_length, p_length
	'''----------------------------------------------------------------
	Args:
		file: file
	Return: None
	'''
	def bleu_scores(self, file):

		df = pd.read_csv(file)
		self.metric = load_metric("bleu")
		df['all_scores'] = df.apply(lambda x: self.compute_bleu_all(x['generated'], x['summary']), axis=1)
        
		df['bleu'], df['precisions'], df['brevity_penalty'], df['length_ratio'], df['reference_length'], df['prediction_length'] \
		= zip(*df.apply(lambda x: self.compute_bleu(x['generated'], x['summary']), axis=1))
		
		file = os.path.join(self.folder, config.topk_bleu_file)
		df.to_csv(file, encoding='utf-8')
	
	'''----------------------------------------------------------------
	Bert Score 
	Args:
		pred  : str 
		ref   : str
	Return:
		score : dict
	'''

	def compute_bert_all(self, pred, ref):

		prediction = []
		prediction.append(pred) 
		ref_list = []
		ref_list.append(ref) 
		reference = []
		reference.append(ref_list) #list of list
		#compute takes list
		
		score = self.metric.compute(predictions= prediction, references= reference, lang="en", device= self.device)
		#print(score)
		return score

	'''----------------------------------------------------------------
	Args:
		pred  : str 
		ref   : str 
	Return:
		p, r, f: float
	'''
	def compute_bert(self, pred, ref):

		prediction = []
		prediction.append(pred) 
		ref_list = []
		ref_list.append(ref) 
		reference = []
		reference.append(ref_list) #list of list
		#compute takes list
		score = self.metric.compute(predictions= prediction, references= reference, lang="en", device= self.device)
		#print(score.keys())
		p = score.get('precision')
		r = score.get('recall')
		f = score.get('f1')
		
		return p, r, f
	'''----------------------------------------------------------------
	Args:
		file: file
	Return: None
	'''
	def bert_scores(self, file):

		df = pd.read_csv(file)
		self.metric = load_metric("bertscore")
		df['all_scores'] = df.apply(lambda x: self.compute_bert_all(x['generated'], x['summary']), axis=1)

		df['precision'], df['recall'], df['f1'] \
		= zip(*df.apply(lambda x: self.compute_bert(x['generated'], x['summary']), axis=1))
		
		file = os.path.join(self.folder, config.topk_bert_file)
		df.to_csv(file, encoding='utf-8')
	
		'''----------------------------------------------------------------
	Bert Score 
	Args:
		pred  : str 
		ref   : str
	Return:
		score : dict
	'''

	def compute_bleurt(self, pred, ref):

		prediction = []
		prediction.append(pred) 
		reference = []
		reference.append(ref) 
		#compute takes list
		
		score = self.metric.compute(predictions= prediction, references= reference)
		#print(score)
		return score.get('scores')

	
	'''----------------------------------------------------------------
	Args:
		file: file
	Return: None
	'''
	def bleurt_scores(self, file):

		df = pd.read_csv(file)
		self.metric = load_metric("bleurt")
		df['all_scores'] = df.apply(lambda x: self.compute_bleurt(x['generated'], x['summary']), axis=1)
		
		file = os.path.join(self.folder, config.topk_bleurt_file)
		df.to_csv(file, encoding='utf-8')
		
	'''----------------------------------------------------------------
	Args:
		file: file
	Return: None
	'''
	def evaluation(self, file):
		
		self.rouge_scores(file)
		self.bleu_scores(file)
		self.bert_scores(file)		
		self.bleurt_scores(file)
		



