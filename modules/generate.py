# -*- coding: utf-8 -*-
"""
Created on Wed Jun 02 16:25:48 2021

@author: fatimamh

"""
import os
import time
import random
import torch

from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Config
import torch.nn.functional as F

from utils.print_n_plot_utils import *

import data_config as config



class Generate(object):
	'''----------------------------------------------------------------
	Initialize generation object, load configuration and model from the given path. 
	Args:
		device		 	: torch_device
		model_path	 	: str
	Return: 
		object
	'''
	def __init__(self, device, model_path):

		self.device = device
		
		self.configuration = GPT2Config.from_pretrained(model_path, 
														output_hidden_states=False) #config file
		self.model = GPT2LMHeadModel.from_pretrained(model_path, 
													 config=self.configuration) # instantiate the model
		self.model.to(self.device)
		
		self.tokenizer = AutoTokenizer.from_pretrained(model_path)

		self.summaries = os.path.join(config.out_dir, config.generated)
		if not os.path.exists(self.summaries):
			os.makedirs(self.summaries)
			
	'''----------------------------------------------------------------
	Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
	Args:
		logits: logits distribution shape (vocabulary size)
		top_k > 0: keep only top k tokens with highest probability (top-k filtering).
		top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
		Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
		From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
	Return: 
		logits
	'''
	def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
		
		assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
		top_k = min(top_k, logits.size(-1))  # Safety check
		if top_k > 0:
			# Remove all tokens with a probability less than the last token of the top-k
			indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
			logits[indices_to_remove] = filter_value

		if top_p > 0.0:
			sorted_logits, sorted_indices = torch.sort(logits, descending=True)
			cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

			# Remove tokens with cumulative probability above the threshold
			sorted_indices_to_remove = cumulative_probs > top_p
			# Shift the indices to the right to keep also the first token above the threshold
			sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
			sorted_indices_to_remove[..., 0] = 0

			indices_to_remove = sorted_indices[sorted_indices_to_remove]
			logits[indices_to_remove] = filter_value
		#print('logits in top k: {}'.format(logits.shape))

		return logits

	'''----------------------------------------------------------------
	Generates a sequence of tokens 
	Args:
		context	   : list
        length 	   : int
        temperature: int 			>0: used to control the randomness of predictions by scaling the logits before applying softmax.
		top_k 	   : float			> 0: keep only top k tokens with highest probability (top-k filtering).
		top_p  	   : float 			> 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
	Return:
		generated: tensor
	'''

	def top_k_generate(self, context, length, temperature=1, top_k=0, top_p=0.0):
		
		context = torch.tensor(context, dtype=torch.long, device=self.device)
		context = context.unsqueeze(0)
		generated = context
		#print('generated: {}'.format(generated))
		with torch.no_grad():  
			for _ in range(length):
				inputs = {'input_ids': generated}
				outputs = self.model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
				next_token_logits = outputs[0][0, -1, :] / temperature
				filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
				#print('filtered_logits.shape: {}'.format(filtered_logits.shape))
				next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
				#print('next_token.shape: {}'.format(next_token.shape))
				generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
		#print('generated: {}'.format(generated.shape))
		return generated

	'''----------------------------------------------------------------
	Generate sequence using beam search https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/
    Args:
        context    : list
        length 	   : int
        beam_size  : int 			>=1 and <= total_no_of_tokens
        temperature: int        	>0 used to control the randomness of predictions by scaling the logits before applying softmax.
    Return: 
    	scores, sequences: list, list
	'''
	def beam_search(self, context, length, beam_size, temperature=1):
		
		context = torch.tensor(context, dtype=torch.long, device=self.device)
		context = context.unsqueeze(0)
		with torch.no_grad():  
			inputs = {'input_ids': context}
			outputs = self.model(**inputs) 
			next_token_logits = outputs[0][0, -1, :] / temperature
			
			#print('next_token_logits.shape: {}'.format(next_token_logits.shape))
			next_token_probs = F.softmax(next_token_logits, dim=-1)
			#print('next_token_probs.shape: {}'.format(next_token_probs.shape))
			scores, indices = torch.topk(next_token_probs, beam_size)
			indices = indices.tolist()
			sequences = [[c] for c in indices]

			#print('in loop')
			for _ in range(length-1):
				logits = torch.zeros(beam_size*len(next_token_logits))
				for j in range(len(sequences)):
					new_generated = torch.cat((context,torch.tensor([sequences[j]], dtype=torch.long, device=self.device)),dim=1)
					inputs = {'input_ids': new_generated}
					outputs = self.model(**inputs) 
					next_token_logits = outputs[0][0, -1, :] / temperature
					#print('next_token_logits.shape: {}'.format(next_token_logits.shape))
					next_token_probs = F.softmax(next_token_logits, dim=-1)
					#print('next_token_probs.shape: {}'.format(next_token_probs.shape))

					start, stop = j*len(next_token_logits), (j+1)*len(next_token_logits)
					logits[start:stop] = scores[j]*next_token_probs
				scores, new_logits_indices = torch.topk(logits,beam_size)
				logits = (new_logits_indices%len(self.tokenizer)).tolist() 
				for j in range(len(sequences)):
					sequences[j] = sequences[j]+[logits[j]]
		return scores, sequences

	'''----------------------------------------------------------------
	Args:
		test_dataset: df

	'''
	def generation(self, test_dataset, gen_type=0):
		#print('tokenizer len: {}'.format(len(self.tokenizer)))
		time_stamp = str(int(time.time()))
		topk_file= os.path.join(self.summaries, config.topk_file)
		beam_file = "beam_summaries" + ".csv"
		beam_file= os.path.join(self.summaries, config.beam_file)

		for index, row in test_dataset.iterrows():	
			#if index == 3:
			#	break

			encodings = self.tokenizer.encode('<BOS>'+ row['text'] + '<EOS>') 
			text = self.tokenizer.decode(encodings, skip_special_tokens=True)
			summary = row['summary']
			#summay len == gen len? 
			if gen_type==0:

				generated = self.top_k_generate(encodings, length=config.max_sum, temperature=config.temperature, top_k=config.top_k, top_p=config.top_p)
				print('generated: {}'.format(generated.shape))
				generated = generated[0, len(encodings):].tolist()
				print('generated: {}'.format(len(generated)))			
				#print(row['text'])
				gen = self.tokenizer.decode(generated, skip_special_tokens=True)

				row = build_row(text, summary, gen)
				print('row: {}\n'.format(row))
				topk_writer(topk_file, row)
				
			
			elif gen_type==1:
				
				scores, sequences = self.beam_search(encodings, length=config.max_sum, beam_size=config.num_beams, temperature=config.temperature)
				print('sequences len: {}'.format(len(sequences)))
				row = build_row(text, summary)

				for i in range(len(sequences)):
					sublist = []
					generated = sequences[i]
					#print("generated_summary-{} and Score is {}.".format(i+1, scores[i]))
					score = float(scores[i])
					gen = self.tokenizer.decode(generated, skip_special_tokens=True)
					sublist.append(gen)
					sublist.append(score)
					#row = build_row(row, gen, score)
					row.extend(sublist)
					
				print('row: {}\n'.format(row))
				beam_writer(beam_file, row, config.num_beams)
			
			elif gen_type==2:
				
				generated = self.top_k_generate(encodings, length=config.max_sum, temperature=config.temperature, top_k=config.top_k, top_p=config.top_p)
				print('generated: {}'.format(generated.shape))
				generated = generated[0, len(encodings):].tolist()
				print('generated: {}'.format(len(generated)))			
				#print(row['text'])
				gen = self.tokenizer.decode(generated, skip_special_tokens=True)

				topk_row = build_row(text, summary, gen)
				print('row: {}\n'.format(topk_row))
				topk_writer(topk_file, topk_row)
				#--------------------------------

				scores, sequences = self.beam_search(encodings, length=config.max_sum, beam_size=config.num_beams, temperature=config.temperature)
				print('sequences len: {}'.format(len(sequences)))
				beam_row = build_row(text, summary)

				for i in range(len(sequences)):
					sublist = []
					generated = sequences[i]
					#print("generated_summary-{} and Score is {}.".format(i+1, scores[i]))
					score = float(scores[i])
					gen = self.tokenizer.decode(generated, skip_special_tokens=True)
					sublist.append(gen)
					sublist.append(score)
					#row = build_row(row, gen, score)
					beam_row.extend(sublist)

				print('row: {}\n'.format(beam_row))
				beam_writer(beam_file, beam_row, config.num_beams)

			
			else:
				print('Wrong choice for generation')
